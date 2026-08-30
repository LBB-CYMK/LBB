# -*- coding: utf-8 -*-
"""
数据预处理脚本
====================================
课程设计：基于深度学习的设备磨损趋势预测与故障预警系统

预处理流程：
    1. 异常值检测与剔除：邻域中位数去趋势 + IQR 方法
    2. 数据归一化：Min-Max 归一化
    3. 剩余寿命(RUL)标签计算：分段线性退化假设
    4. 时序样本构建：滑动窗口切分
    5. 数据集划分：按设备划分训练集/测试集（避免数据泄漏）

输出文件：
    data/processed/wear_processed.csv      —— 清洗+归一化后的完整数据
    data/processed/sequences.npz           —— 滑动窗口时序样本（供RNN训练）
    data/processed/preprocess_report.json  —— 预处理统计报告
    data/processed/preprocess_visualization.png —— 异常值剔除前后对比图
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无界面环境使用非交互后端
import matplotlib.pyplot as plt

# 配置中文字体，避免标签渲染为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============ 预处理参数 ============
DETREND_WINDOW = 7      # 去趋势用的邻域窗口大小（奇数，取当前点两侧邻域）
IQR_K = 1.5             # IQR 异常判定系数（标准取值 1.5）
SEQ_WINDOW = 20         # 时序滑动窗口长度（用前20个周期预测下一周期）
RUL_CAP = 125           # RUL 截断上限
TEST_RATIO = 0.2        # 测试集设备占比

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "wear_raw.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_raw_data():
    """加载原始数据并按 (unit_id, time_cycle) 排序。"""
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    df = df.sort_values(["unit_id", "time_cycle"]).reset_index(drop=True)
    return df


def _neighbor_median_trend(s, window=DETREND_WINDOW):
    """
    邻域中位数去趋势：用窗口内"除当前点外"的邻域中位数估计局部趋势。

    关键点：磨损量呈单调上升的指数趋势，若直接用滑动中位数（含当前点），
    对单调序列中位数就等于当前点本身，残差恒为 0，无法还原噪声；而直接
    对原始值做全局 IQR 又会把趋势尾部误判为异常。因此这里取"两侧邻域的
    中位数"作为局部趋势基准：既对单调趋势有效（两侧中点≈当前趋势值），
    又能抵抗当前点自身是野值的情况（野值被排除在中位数之外）。
    """
    def _med(arr):
        return np.median(np.delete(arr, len(arr) // 2))

    return s.rolling(window=window, center=True, min_periods=1).apply(_med, raw=True)


def iqr_outlier_removal(df, window=DETREND_WINDOW, k=IQR_K):
    """
    邻域中位数去趋势 + IQR 异常值检测与剔除。

    流程：
      1. 对每个时刻，用其两侧邻域的中位数估计局部趋势（见 _neighbor_median_trend）；
      2. 计算残差 = 原始值 - 趋势值，使残差近似同方差的高斯噪声；
      3. 对残差按标准 IQR 规则（[Q1-1.5*IQR, Q3+1.5*IQR]）判定异常；
      4. 用局部趋势值回填被标记为异常的点。

    说明：IQR 对高斯噪声存在约 0.7% 的固有误报率，属于方法本身特性；
    注入的野值幅度远大于噪声，可被稳定识别。
    """
    df = df.copy()
    df["wear_clean"] = df["wear_value"].astype(float)
    df["temp_clean"] = df["temperature"].astype(float)
    df["is_outlier"] = False

    n_outlier = {"wear_value": 0, "temperature": 0}

    for col, clean_col in [("wear_value", "wear_clean"),
                           ("temperature", "temp_clean")]:
        for uid, group in df.groupby("unit_id"):
            trend = _neighbor_median_trend(group[col], window)
            residual = group[col] - trend

            q1, q3 = residual.quantile(0.25), residual.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr

            outlier_mask = (residual < lower) | (residual > upper)
            # 回填：用局部趋势值替换异常值
            df.loc[group.index, clean_col] = group[col].where(~outlier_mask, trend)
            df.loc[group.index, "is_outlier"] = (
                df.loc[group.index, "is_outlier"] | outlier_mask.values
            )
            n_outlier[col] += int(outlier_mask.sum())

    return df, n_outlier


def minmax_normalize(df):
    """对清洗后的特征做 Min-Max 归一化，映射到 [0, 1]。"""
    df = df.copy()
    norm_params = {}
    for col, norm_col in [("wear_clean", "wear_norm"), ("temp_clean", "temp_norm")]:
        xmin, xmax = df[col].min(), df[col].max()
        df[norm_col] = (df[col] - xmin) / (xmax - xmin)
        norm_params[col] = {"min": float(xmin), "max": float(xmax)}
    return df, norm_params


def compute_rul(df, rul_cap=RUL_CAP):
    """按分段线性退化假设计算剩余寿命 RUL = min(失效周期 - 当前周期, cap)。"""
    df = df.copy()
    df["rul"] = np.nan
    for uid, group in df.groupby("unit_id"):
        failure_cycle = group["time_cycle"].max()
        rul = (failure_cycle - group["time_cycle"]).clip(upper=rul_cap)
        df.loc[group.index, "rul"] = rul
    return df


def build_sequences(df, window=SEQ_WINDOW):
    """按设备构建滑动窗口样本，输入为连续 window 个周期的特征，输出为下一周期磨损量。"""
    feature_cols = ["wear_norm", "temp_norm"]
    X, y, rul, unit_ids = [], [], [], []

    for uid, group in df.groupby("unit_id"):
        feat = group[feature_cols].values.astype(float)
        tgt = group["wear_norm"].values.astype(float)
        rul_vals = group["rul"].values.astype(float)
        for i in range(len(group) - window):
            X.append(feat[i:i + window])
            y.append(tgt[i + window])
            rul.append(rul_vals[i + window])
            unit_ids.append(uid)

    return (
        np.array(X),
        np.array(y).reshape(-1, 1),
        np.array(rul).reshape(-1, 1),
        np.array(unit_ids),
    )


def split_by_unit(X, y, rul, unit_ids, test_ratio=TEST_RATIO):
    """按设备划分训练/测试集，避免同一设备的样本泄漏到两侧。"""
    unique_units = np.sort(np.unique(unit_ids))
    n_test = max(1, int(len(unique_units) * test_ratio))
    test_units = unique_units[-n_test:]
    test_mask = np.isin(unit_ids, test_units)

    return (
        X[~test_mask], y[~test_mask], rul[~test_mask],
        X[test_mask], y[test_mask], rul[test_mask],
    )


def save_visualization(df):
    """绘制单个设备异常值剔除前后对比图，验证预处理效果。"""
    unit = df[df["unit_id"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(unit["time_cycle"], unit["wear_value"], "-o", ms=3, label="原始数据")
    axes[0].set_title("异常值剔除前（含野值）")
    axes[0].set_xlabel("运行周期")
    axes[0].set_ylabel("磨损量 (mm)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(unit["time_cycle"], unit["wear_clean"], "-o", ms=3, color="C1", label="清洗后")
    axes[1].set_title("异常值剔除后（滑动中位数残差IQR）")
    axes[1].set_xlabel("运行周期")
    axes[1].set_ylabel("磨损量 (mm)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "preprocess_visualization.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 加载原始数据
    df = load_raw_data()
    n_raw = len(df)

    # 2. IQR 异常值检测与剔除
    df, n_outlier = iqr_outlier_removal(df)

    # 3. Min-Max 归一化
    df, norm_params = minmax_normalize(df)

    # 4. RUL 标签计算
    df = compute_rul(df)

    # 5. 滑动窗口时序样本构建 + 数据集划分
    X, y, rul, unit_ids = build_sequences(df)
    X_train, y_train, rul_train, X_test, y_test, rul_test = split_by_unit(
        X, y, rul, unit_ids
    )

    # 6. 保存处理结果
    processed_path = os.path.join(OUT_DIR, "wear_processed.csv")
    df.to_csv(processed_path, index=False, encoding="utf-8-sig")

    seq_path = os.path.join(OUT_DIR, "sequences.npz")
    np.savez_compressed(
        seq_path,
        X_train=X_train, y_train=y_train, rul_train=rul_train,
        X_test=X_test, y_test=y_test, rul_test=rul_test,
    )

    viz_path = save_visualization(df)

    report = {
        "raw_samples": n_raw,
        "processed_samples": len(df),
        "outlier_removed": n_outlier,
        "outlier_total": int(df["is_outlier"].sum()),
        "normalize_params": norm_params,
        "window_size": SEQ_WINDOW,
        "rul_cap": RUL_CAP,
        "train_units": int(np.unique(np.sort(unit_ids))[:-max(1, int(len(np.unique(unit_ids)) * TEST_RATIO))].size),
        "test_units": int(max(1, int(len(np.unique(unit_ids)) * TEST_RATIO))),
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
    }
    report_path = os.path.join(OUT_DIR, "preprocess_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[完成] 预处理完成，输出如下：")
    print(f"  清洗+归一化数据：{processed_path}")
    print(f"  时序样本(训练/测试)：{seq_path}")
    print(f"  预处理报告：{report_path}")
    print(f"  可视化对比图：{viz_path}")
    print("\n预处理统计：")
    print(f"  原始样本数={n_raw}，处理后样本数={len(df)}")
    print(f"  异常值剔除：wear={n_outlier['wear_value']}，temperature={n_outlier['temperature']}，合计={int(df['is_outlier'].sum())}")
    print(f"  时序样本：训练集 X{X_train.shape} / y{y_train.shape}，测试集 X{X_test.shape} / y{y_test.shape}")


if __name__ == "__main__":
    main()
