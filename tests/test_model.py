# -*- coding: utf-8 -*-
"""模型定义与保存/加载的单元测试。"""

import torch

from model import WearRNN, save_model, load_model


def test_forward_shape():
    """前向传播输出形状应为 (batch, 1)。"""
    model = WearRNN(hidden_size=32)
    x = torch.randn(4, 20, 2)  # (batch, seq, features)
    y = model(x)
    assert y.shape == (4, 1)


def test_save_load_roundtrip(tmp_path):
    """保存后再加载，权重应保持一致（前向结果一致）。"""
    model = WearRNN(hidden_size=16)
    x = torch.randn(2, 20, 2)
    before = model(x).detach()

    path = tmp_path / "m.pt"
    save_model(model, str(path))
    loaded = load_model(str(path))

    after = loaded(x).detach()
    assert torch.allclose(before, after, atol=1e-6)
    assert not loaded.training  # load_model 应切换到 eval 模式
