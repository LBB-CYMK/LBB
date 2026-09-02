# -*- coding: utf-8 -*-
"""预测 / RUL / 预警逻辑的单元测试。"""

import numpy as np

import predict


def test_load_assets():
    """模型与归一化参数应能正常加载，失效阈值落在 (0,1)。"""
    model, wear_params, threshold_norm = predict.load_assets()
    assert model is not None
    assert 0.0 < threshold_norm < 1.0
    assert "min" in wear_params and "max" in wear_params


def test_predict_next_wear_range():
    """预测磨损量应为归一化 [0,1] 区间内的浮点数。"""
    model, wear_params, threshold_norm = predict.load_assets()
    window = np.random.rand(20, 2).astype(np.float32)
    pred = predict.predict_next_wear(model, window)
    assert isinstance(pred, float)
    # 归一化磨损在合理区间（宽松边界）
    assert 0.0 <= pred <= 1.2


def test_roll_forward_rul_bounds():
    """滚动前向 RUL 应为 [1, 125] 区间内的整数。"""
    model, wear_params, threshold_norm = predict.load_assets()
    # 构造一个接近失效的窗口：磨损逼近阈值
    window = np.zeros((20, 2), dtype=np.float32)
    window[:, 0] = 0.95  # 磨损接近失效阈值
    rul = predict.roll_forward_rul(model, window, threshold_norm)
    assert isinstance(rul, int)
    assert 1 <= rul <= predict.MAX_RUL


def test_check_alert_levels():
    """预警等级判定应覆盖 normal/warning/critical 三种情况。"""
    threshold = 0.9
    # 正常
    level, _ = predict.check_alert(0.5, 50, threshold)
    assert level == "normal"
    # 预警：RUL 落入 warn 区间
    level, _ = predict.check_alert(0.5, 20, threshold)
    assert level == "warning"
    # 严重预警：RUL 极小
    level, _ = predict.check_alert(0.5, 5, threshold)
    assert level == "critical"
    # 严重预警：预测磨损超过阈值
    level, _ = predict.check_alert(0.95, 50, threshold)
    assert level == "critical"
