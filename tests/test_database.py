# -*- coding: utf-8 -*-
"""SQLite 数据层的单元测试（使用临时数据库）。"""

import database


def test_init_and_seed_devices(tmp_path):
    """初始化后 devices 表应从汇总表播种 30 台设备。"""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    devices = database.list_devices(db)
    assert len(devices) == 30
    d0 = devices[0]
    assert d0["unit_id"] == 1
    assert d0["failure_cycle"] > 0


def test_insert_and_query_prediction(tmp_path):
    """预测记录应写入 predictions 表，异常状态同步写入 alerts 表。"""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    database.insert_prediction(1, 0.85, 20, "warning", "剩余寿命不足 30 周期", db)

    preds = database.list_predictions(db_path=db)
    assert len(preds) == 1
    assert preds[0]["alert_level"] == "warning"

    alerts = database.list_alerts(db_path=db)
    assert len(alerts) == 1
    assert alerts[0]["unit_id"] == 1


def test_normal_prediction_no_alert(tmp_path):
    """正常状态预测不应写入 alerts 表。"""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    database.insert_prediction(1, 0.1, 100, "normal", "运行正常", db)
    assert len(database.list_alerts(db_path=db)) == 0
    assert len(database.list_predictions(db_path=db)) == 1
