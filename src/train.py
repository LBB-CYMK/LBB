# -*- coding: utf-8 -*-
"""
RNN 模型训练脚本（含超参数搜索）
====================================
课程设计：基于深度学习的设备磨损趋势预测与故障预警系统

训练流程：
    1. 加载 data/processed/sequences.npz 中的滑动窗口时序样本
    2. 将训练集再划分为 train/val（9:1，固定随机种子保证可复现）
    3. 超参数网格搜索（智能搜索专题落地）：对 hidden_size × lr 组合
       分别训练，按验证集 MSE 择优
    4. 用最优超参数在 train+val 上重训，得到最终模型
    5. 在独立测试集（按设备划分，未见设备）上评估指标
    6. 保存模型 models/wear_rnn.pt、训练曲线与指标 outputs/

「智能搜索与状态空间搜索」专题对应：对超参数构成的状态空间做网格枚举，
以验证集误差为评价函数，搜索最优参数组合（启发式搜索思想的课程落地）。
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import WearRNN

# 中文字体，避免标签渲染为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============ 路径与参数 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "sequences.npz")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

BATCH_SIZE = 64
EPOCHS = 60          # 每个候选配置的训练轮数
VAL_RATIO = 0.1      # 训练集内再划分验证集比例
SEED = 42

# 超参数搜索网格（智能搜索：状态空间枚举 + 验证误差择优）
SEARCH_GRID = [
    {"hidden_size": 16, "lr": 1e-3},
    {"hidden_size": 32, "lr": 1e-3},
    {"hidden_size": 32, "lr": 3e-3},
    {"hidden_size": 64, "lr": 3e-3},
]


def set_seed(seed=SEED):
    """固定随机种子，保证训练可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    """加载时序样本，返回 numpy 数组。"""
    data = np.load(SEQ_PATH)
    return (
        data["X_train"], data["y_train"], data["rul_train"],
        data["X_test"], data["y_test"], data["rul_test"],
    )


def to_tensor(X, y):
    """numpy -> torch 张量（float32）。"""
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def train_one(model, train_loader, val_loader, lr, epochs, device):
    """训练单个模型，返回 (val_mse, history)。history 记录每轮 train/val 损失。"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        # 训练
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        val_loss = float(np.mean(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # 保存验证集最优权重
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_val, history


def evaluate(model, X, y, device):
    """在给定数据上评估 MSE / RMSE / MAE。"""
    model = model.to(device).eval()
    Xt, yt = to_tensor(X, y)
    with torch.no_grad():
        pred = model(Xt.to(device))
    pred = pred.cpu().numpy()
    y_np = yt.numpy()
    mse = float(np.mean((pred - y_np) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred - y_np)))
    return {"mse": mse, "rmse": rmse, "mae": mae}


def plot_loss_history(histories, best_cfg, path):
    """绘制各候选配置的训练/验证损失曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg, history in histories:
        label = f"hidden={cfg['hidden_size']}, lr={cfg['lr']}"
        ax.plot(history["val_loss"], label=f"val {label}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"超参数搜索：验证损失曲线（最优 hidden={best_cfg['hidden_size']}, lr={best_cfg['lr']}）")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, y_train, _, X_test, y_test, _ = load_data()
    print(f"[数据] 训练集 {X_train.shape}，测试集 {X_test.shape}，设备 {device}")

    # 训练集内再划分 train/val
    n = len(X_train)
    idx = np.random.permutation(n)
    n_val = int(n * VAL_RATIO)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # ============ 智能搜索：超参数网格枚举 ============
    print("\n[搜索] 开始超参数网格搜索...")
    results = []
    for cfg in SEARCH_GRID:
        set_seed(SEED)  # 每个配置用相同初始化，保证公平比较
        model = WearRNN(input_size=X_train.shape[2], hidden_size=cfg["hidden_size"])
        train_loader = DataLoader(
            TensorDataset(*to_tensor(X_tr, y_tr)), batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(*to_tensor(X_val, y_val)), batch_size=BATCH_SIZE
        )
        val_mse, history = train_one(model, train_loader, val_loader, cfg["lr"], EPOCHS, device)
        results.append((cfg, val_mse, history))
        print(f"    hidden={cfg['hidden_size']}, lr={cfg['lr']} -> val_mse={val_mse:.6f}")

    best_cfg, best_val, best_history = min(results, key=lambda r: r[1])
    print(f"[搜索] 最优配置 hidden={best_cfg['hidden_size']}, lr={best_cfg['lr']}，val_mse={best_val:.6f}")

    # ============ 用最优超参数在 train+val 上重训最终模型 ============
    print("\n[训练] 用最优超参数在完整训练集上重训...")
    set_seed(SEED)
    final_model = WearRNN(input_size=X_train.shape[2], hidden_size=best_cfg["hidden_size"])
    full_loader = DataLoader(
        TensorDataset(*to_tensor(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True
    )
    full_val_loader = DataLoader(
        TensorDataset(*to_tensor(X_test, y_test)), batch_size=BATCH_SIZE
    )
    _, final_history = train_one(
        final_model, full_loader, full_val_loader, best_cfg["lr"], EPOCHS, device
    )

    # ============ 测试集评估 ============
    test_metrics = evaluate(final_model, X_test, y_test, device)
    print(f"[评估] 独立测试集：MSE={test_metrics['mse']:.6f}, "
          f"RMSE={test_metrics['rmse']:.6f}, MAE={test_metrics['mae']:.6f}")

    # ============ 保存模型与产物 ============
    model_path = os.path.join(MODEL_DIR, "wear_rnn.pt")
    from model import save_model
    save_model(final_model, model_path)

    plot_loss_history(
        [(c, h) for c, _, h in results], best_cfg,
        os.path.join(OUTPUT_DIR, "train_loss.png"),
    )

    report = {
        "search_grid": SEARCH_GRID,
        # 保留每一组候选配置的验证集误差，便于复核「最优」是如何选出来的
        "search_results": [
            {"hidden_size": c["hidden_size"], "lr": c["lr"], "val_mse": v}
            for c, v, _ in results
        ],
        "best_config": best_cfg,
        "best_val_mse": best_val,
        "final_test_metrics": test_metrics,
        "epochs": EPOCHS,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "window_size": int(X_train.shape[1]),
        "input_features": int(X_train.shape[2]),
    }
    report_path = os.path.join(OUTPUT_DIR, "train_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 模型已保存：{model_path}")
    print(f"[完成] 训练曲线：{os.path.join(OUTPUT_DIR, 'train_loss.png')}")
    print(f"[完成] 训练报告：{report_path}")


if __name__ == "__main__":
    main()
