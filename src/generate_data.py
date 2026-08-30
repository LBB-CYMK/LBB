# -*- coding: utf-8 -*-
"""
设备磨损时序数据生成脚本
====================================
课程设计：基于深度学习的设备磨损趋势预测与故障预警系统

本脚本基于"指数退化模型"生成设备磨损模拟数据（自建数据集）。
指数退化模型是设备剩余寿命预测(RUL)领域的经典退化模型，
描述设备磨损量随时间呈指数增长的趋势：

    w(t) = w0 + A * (exp(B * t) - 1) + ε

    其中：
        w0  —— 初始磨损量（mm），体现设备出厂时的个体差异
        A   —— 退化幅值系数，控制磨损增长的整体幅度
        B   —— 指数退化速率，控制磨损增长快慢
        ε   —— 观测噪声，服从高斯分布 N(0, σ²)

每个设备（unit）随机采样一组参数 (w0, A, B)，从而拥有不同的退化轨迹
与寿命；当磨损量 w(t) 达到失效阈值 threshold 时判定设备失效。
同时生成与磨损相关的温度信号，构成多特征时序数据，并随机注入
异常值（传感器跳变/野值）用于验证后续异常检测流程。

输出文件：
    data/raw/wear_raw.csv          —— 原始磨损数据（含异常值）
    data/raw/units_summary.csv     —— 各设备退化参数与寿命汇总表
"""

import os
import numpy as np
import pandas as pd

# 固定随机种子，保证数据可复现
np.random.seed(42)

# ============ 全局参数 ============
N_UNITS = 30             # 设备数量
FAILURE_THRESHOLD = 1.0  # 失效阈值（磨损量，mm）
NOISE_SIGMA = 0.004      # 观测噪声标准差（mm）
OUTLIER_RATIO = 0.015    # 注入异常值比例
OUTLIER_MAGNITUDE = 0.10 # 磨损量异常值偏离幅度（mm）


def sample_unit_params():
    """为单个设备随机采样指数退化参数 (w0, A, B)。"""
    w0 = np.random.uniform(0.03, 0.08)    # 初始磨损量
    A = np.random.uniform(0.003, 0.010)   # 退化幅值系数
    B = np.random.uniform(0.008, 0.020)   # 指数退化速率
    return w0, A, B


def simulate_unit(unit_id, w0, A, B):
    """模拟单个设备从投运到失效的完整退化轨迹，返回 DataFrame。"""
    records = []
    t = 0
    wear = w0
    # 温度与磨损线性相关：T(t) = T0 + k * w(t) + 噪声
    T0 = np.random.uniform(30.0, 35.0)   # 初始温度（℃）
    k = np.random.uniform(20.0, 30.0)    # 温升系数

    while wear < FAILURE_THRESHOLD:
        temp = T0 + k * wear + np.random.normal(0, 0.2)
        records.append((unit_id, t, round(wear, 5), round(temp, 3)))
        t += 1
        # 指数退化模型更新磨损量（含观测噪声）
        wear = w0 + A * (np.exp(B * t) - 1) + np.random.normal(0, NOISE_SIGMA)

    # 记录失效时刻（磨损量首次超过阈值）
    temp = T0 + k * wear + np.random.normal(0, 0.2)
    records.append((unit_id, t, round(wear, 5), round(temp, 3)))

    df = pd.DataFrame(
        records,
        columns=["unit_id", "time_cycle", "wear_value", "temperature"],
    )
    return df


def inject_outliers(df):
    """在原始数据中随机注入异常值（传感器跳变/野值），用于验证异常检测。"""
    df = df.copy()
    n = len(df)
    n_outlier = int(n * OUTLIER_RATIO)
    idx = np.random.choice(df.index, size=n_outlier, replace=False)
    signs = np.random.choice([1.0, -1.0], size=n_outlier)

    # 磨损量维度：正向尖峰或负向跌落（并裁剪到非负，避免出现无物理意义的负磨损）
    df.loc[idx, "wear_value"] = (
        df.loc[idx, "wear_value"] + signs * OUTLIER_MAGNITUDE
    ).clip(lower=0.0)

    # 温度维度：同样注入少量异常，模拟传感器漂移
    df.loc[idx, "temperature"] = (
        df.loc[idx, "temperature"]
        + signs * np.random.uniform(5.0, 15.0, size=n_outlier)
    )

    return df


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    summaries = []

    for unit_id in range(1, N_UNITS + 1):
        w0, A, B = sample_unit_params()
        unit_df = simulate_unit(unit_id, w0, A, B)
        frames.append(unit_df)

        failure_cycle = unit_df["time_cycle"].max()
        summaries.append(
            {
                "unit_id": unit_id,
                "w0": round(w0, 5),
                "A": round(A, 5),
                "B": round(B, 5),
                "failure_cycle": failure_cycle,
            }
        )

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df = inject_outliers(raw_df)

    # 保存原始数据与设备汇总表
    raw_path = os.path.join(output_dir, "wear_raw.csv")
    summary_path = os.path.join(output_dir, "units_summary.csv")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[完成] 原始数据已生成：{raw_path}")
    print(f"        设备数={N_UNITS}，总样本数={len(raw_df)}，异常值注入数≈{int(len(raw_df)*OUTLIER_RATIO)}")
    print(f"[完成] 设备汇总表已保存：{summary_path}")
    print("\n各设备寿命统计：")
    life = pd.DataFrame(summaries)["failure_cycle"]
    print(f"  寿命范围：[{life.min()}, {life.max()}]，均值={life.mean():.1f}，标准差={life.std():.1f}")


if __name__ == "__main__":
    main()
