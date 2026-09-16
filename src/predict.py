# -*- coding: utf-8 -*-
"""
预测 / 剩余寿命(RUL)估算 / 故障预警服务层
====================================
课程设计：基于深度学习的设备磨损趋势预测与故障预警系统

本模块是「算法服务层」核心，供命令行演示与 FastAPI 后端复用，提供：

    predict_next_wear   —— 单步磨损量预测（归一化）
    roll_forward_rul    —— 滚动前向模拟估算剩余寿命 RUL（状态空间搜索思想）
    check_alert         —— 故障预警判定（基于预测磨损量 + RUL）
    load_assets         —— 加载模型与归一化参数

RUL 估算思路：用训练好的 RNN 从当前窗口出发，反复「预测下一周期磨损 → 回填
窗口 → 再预测」，在状态空间中前向搜索磨损量首次越过失效阈值的周期数，即为
剩余寿命（截断上限 125，与标签一致）。这对应课程「智能搜索与状态空间搜索」专题。
"""

import os
import json

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import load_model

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============ 路径与阈值参数 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "wear_rnn.pt")
REPORT_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "preprocess_report.json")
PROCESSED_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "wear_processed.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

FAILURE_WEAR_MM = 1.0   # 失效阈值（磨损量 mm），与数据生成一致
WINDOW = 20             # 滑动窗口长度，与预处理一致
MAX_RUL = 125           # RUL 截断上限，与标签一致
WARN_RUL = 30           # 预警阈值：剩余寿命不足 30 周期
CRIT_RUL = 15           # 严重预警阈值：剩余寿命不足 15 周期

# 三级预警对应的可执行检修建议（业务闭环：预警等级 → 运维决策）
# 让系统输出不止是「状态」，而是运维人员可直接执行的结论。
MAINTENANCE_ADVICE = {
    "normal": "建议按计划周期检修",
    "warning": "建议提前准备备件，安排近期停机检查",
    "critical": "建议立即安排停机检修，避免非计划停机",
}


def load_assets(device="cpu"):
    """加载模型、归一化参数，并计算归一化空间下的失效阈值。"""
    model = load_model(MODEL_PATH).to(device)
    with open(REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)
    wear_params = report["normalize_params"]["wear_clean"]
    # 失效阈值映射到归一化空间
    threshold_norm = (FAILURE_WEAR_MM - wear_params["min"]) / (wear_params["max"] - wear_params["min"])
    return model, wear_params, threshold_norm


def predict_next_wear(model, window, device="cpu"):
    """用前 window 个周期特征预测下一周期磨损量（归一化）。

    参数
    ----
    window : np.ndarray, 形状 (>=20, 2)，取末尾 20 步作为输入

    返回
    ----
    float : 下一周期磨损量归一化值
    """
    x = torch.tensor(window[-WINDOW:], dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        return float(model(x).item())


def roll_forward_rul(model, window, threshold_norm, max_steps=MAX_RUL, device="cpu"):
    """滚动前向模拟估算剩余寿命（状态空间搜索）。

    从当前窗口出发，反复预测下一周期磨损并回填窗口，直到预测磨损越过失效
    阈值，返回所经历的周期数（剩余寿命）。温度信号在模拟中近似保持不变
    （磨损是失效主信号）。
    """
    cur = np.asarray(window[-WINDOW:], dtype=np.float32).copy()
    last_temp = cur[-1, 1]
    for step in range(1, max_steps + 1):
        x = torch.tensor(cur, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            next_wear = float(model(x).item())
        if next_wear >= threshold_norm:
            return step
        new_row = np.array([[next_wear, last_temp]], dtype=np.float32)
        cur = np.vstack([cur[1:], new_row])
    return max_steps


def get_maintenance_advice(level):
    """按预警等级返回可执行的检修建议（业务闭环输出）。"""
    return MAINTENANCE_ADVICE.get(level, MAINTENANCE_ADVICE["normal"])


def check_alert(pred_wear_norm, rul, threshold_norm,
                warn_rul=WARN_RUL, crit_rul=CRIT_RUL):
    """故障预警判定，返回 (等级, 提示信息)。

    等级取值：normal / warning / critical。
    提示信息由「判定依据」+「检修建议」拼接而成，
    使系统输出落到运维人员可直接执行的检修决策上（业务闭环）。
    """
    level = "normal"
    msgs = []
    if pred_wear_norm >= threshold_norm:
        level = "critical"
        msgs.append("预测磨损量已超过失效阈值")
    if rul <= crit_rul:
        level = "critical"
        msgs.append(f"剩余寿命仅 {rul} 周期")
    elif rul <= warn_rul:
        level = "warning"
        msgs.append(f"剩余寿命不足 {warn_rul} 周期")

    reason = "；".join(msgs) if msgs else "运行正常"
    return level, f"{reason}；{get_maintenance_advice(level)}"


def load_device(unit_id):
    """加载指定设备的预处理后数据（按运行周期排序）。"""
    df = pd.read_csv(PROCESSED_PATH, encoding="utf-8-sig")
    g = df[df["unit_id"] == unit_id].sort_values("time_cycle").reset_index(drop=True)
    return g


def forecast_device(model, unit_id, wear_params, threshold_norm, device="cpu"):
    """对指定设备做单步预测，返回 (记录表, 预测磨损 mm, 当前RUL, 预警结果)。

    预测磨损：用最后 20 周期预测下一周期磨损量。
    当前 RUL：从最后 20 周期滚动前向估算剩余寿命。
    """
    g = load_device(unit_id)
    window = g[["wear_norm", "temp_norm"]].tail(WINDOW).values.astype(np.float32)

    pred_norm = predict_next_wear(model, window, device)
    rul = roll_forward_rul(model, window, threshold_norm, device=device)
    level, msg = check_alert(pred_norm, rul, threshold_norm)

    # 归一化磨损还原到 mm 量纲
    wmin, wmax = wear_params["min"], wear_params["max"]
    pred_mm = pred_norm * (wmax - wmin) + wmin
    return g, pred_mm, rul, level, msg


def plot_device_demo(model, unit_id, wear_params, threshold_norm, device="cpu"):
    """绘制指定设备的预测演示图：磨损实际/预测曲线 + RUL 对比，返回图片路径。"""
    g, pred_mm, rul, level, msg = forecast_device(model, unit_id, wear_params, threshold_norm, device)
    wmin, wmax = wear_params["min"], wear_params["max"]

    # 单步预测曲线：对每个位置用前 20 周期预测下一周期磨损
    feat = g[["wear_norm", "temp_norm"]].values.astype(np.float32)
    preds = []
    for i in range(len(g) - WINDOW):
        preds.append(predict_next_wear(model, feat[i:i + WINDOW], device))
    preds_norm = np.array(preds)
    preds_mm = preds_norm * (wmax - wmin) + wmin
    cycles_pred = g["time_cycle"].values[WINDOW:]

    # 在若干采样点用滚动前向估算 RUL，与标签对比
    sample_idx = list(range(WINDOW, len(g), 15))
    rul_est, rul_label, rul_cycle = [], [], []
    for i in sample_idx:
        w = feat[i - WINDOW:i]
        rul_est.append(roll_forward_rul(model, w, threshold_norm, device=device))
        rul_label.append(g["rul"].values[i])
        rul_cycle.append(g["time_cycle"].values[i])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(g["time_cycle"], g["wear_clean"], "-", color="C0", label="实际磨损")
    ax.plot(cycles_pred, preds_mm, "--", color="C1", lw=1.5, label="模型单步预测")
    ax.axhline(FAILURE_WEAR_MM, color="C3", ls=":", lw=1.5, label=f"失效阈值 {FAILURE_WEAR_MM}mm")
    ax.scatter([g['time_cycle'].iloc[-1]], [pred_mm], color="C3", zorder=5, label="下一周期预测")
    ax.set_title(f"设备 {unit_id} 磨损趋势（{level}）")
    ax.set_xlabel("运行周期"); ax.set_ylabel("磨损量 (mm)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(rul_cycle, rul_label, "o-", label="RUL 标签")
    ax.plot(rul_cycle, rul_est, "s--", label="滚动前向估算")
    ax.set_title(f"剩余寿命 RUL 估算（当前预测 RUL={rul}）")
    ax.set_xlabel("运行周期"); ax.set_ylabel("剩余寿命 (周期)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(f"故障预警：{msg}", fontsize=11, color="C3" if level != "normal" else "C0")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "demo_prediction.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    model, wear_params, threshold_norm = load_assets()
    print(f"[资产] 模型已加载，失效阈值(归一化)={threshold_norm:.4f}")

    # 演示：取一台测试设备（unit 25 属按设备划分出的测试集，模型未见过）
    unit_id = 25
    g, pred_mm, rul, level, msg = forecast_device(model, unit_id, wear_params, threshold_norm)
    print(f"[预测] 设备 {unit_id}：下一周期磨损≈{pred_mm:.4f}mm，估算 RUL={rul} 周期")
    print(f"[预警] 等级={level}，信息：{msg}")

    path = plot_device_demo(model, unit_id, wear_params, threshold_norm)
    print(f"[完成] 演示图已保存：{path}")


if __name__ == "__main__":
    main()
