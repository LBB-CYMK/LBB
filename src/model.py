# -*- coding: utf-8 -*-
"""
RNN 模型定义与保存/加载
====================================
课程设计：基于深度学习的设备磨损趋势预测与故障预警系统

采用循环神经网络（RNN）做时序回归：输入连续 window 个周期的特征
[wear_norm, temp_norm]，输出下一周期的磨损量（归一化值）。

模型结构：
    RNN(input_size=2, hidden_size, num_layers, nonlinearity='relu')
        -> 取最后时间步隐藏状态
        -> Linear(hidden, 16) -> ReLU -> Linear(16, 1)

使用 ReLU 作为 RNN 单元激活（nn.RNN 的 nonlinearity='relu'）与前馈头
激活，对应课程「循环神经网络时序预测（RNN/ReLU）」专题。
"""

import torch
import torch.nn as nn

# 默认输入特征维度（wear_norm, temp_norm）
DEFAULT_INPUT_SIZE = 2


class WearRNN(nn.Module):
    """设备磨损量时序预测 RNN 模型。"""

    def __init__(self, input_size=DEFAULT_INPUT_SIZE, hidden_size=32,
                 num_layers=1, output_size=1):
        super().__init__()
        # 保存配置，便于保存/加载时重建模型
        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
        }
        # RNN 层：batch_first=True 使输入形状为 (batch, seq_len, features)
        # nonlinearity='relu' 对应课程所学的 ReLU 激活函数
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="relu",
        )
        # 回归头：将最后时间步的隐藏状态映射为磨损量预测值
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        """前向传播。

        参数
        ----
        x : Tensor, 形状 (batch, seq_len, input_size)

        返回
        ----
        Tensor, 形状 (batch, output_size)：下一周期磨损量预测值
        """
        out, _ = self.rnn(x)      # out: (batch, seq_len, hidden_size)
        last = out[:, -1, :]      # 取最后一个时间步的隐藏状态
        return self.head(last)


def save_model(model, path):
    """保存模型配置与权重到磁盘。"""
    torch.save(
        {"config": model.config, "state_dict": model.state_dict()},
        path,
    )


def load_model(path):
    """从磁盘加载模型（含配置），并切换到 eval 模式。"""
    ckpt = torch.load(path, map_location="cpu")
    model = WearRNN(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    # 快速自检：随机输入前向传播，确认输出形状正确
    model = WearRNN(hidden_size=32)
    x = torch.randn(4, 20, 2)  # (batch=4, seq=20, features=2)
    y = model(x)
    print(f"输入形状 {tuple(x.shape)} -> 输出形状 {tuple(y.shape)}")
    print(f"模型参数量：{sum(p.numel() for p in model.parameters())}")
