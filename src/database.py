# -*- coding: utf-8 -*-
"""
SQLite 数据库层
====================================
课程设计：基于深度学习的设备磨损趋势预测与故障预警系统

使用标准库 sqlite3 实现数据库，承载设备档案、预测记录与预警事件，
对应任务书「完整 B/S 架构」中的数据库部分。

表结构：
    devices     —— 设备档案（设备号、失效周期、退化参数）
    predictions —— 预测记录（设备、预测磨损、RUL、预警等级、时间）
    alerts      —— 预警事件（仅记录非正常状态）
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "app.db")
SUMMARY_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "units_summary.csv")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    unit_id        INTEGER PRIMARY KEY,
    failure_cycle  INTEGER NOT NULL,
    w0             REAL,
    A              REAL,
    B              REAL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id        INTEGER NOT NULL,
    pred_wear_mm   REAL NOT NULL,
    rul            INTEGER NOT NULL,
    alert_level    TEXT NOT NULL,
    message        TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id        INTEGER NOT NULL,
    level          TEXT NOT NULL,
    message        TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
"""


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_conn(db_path=DB_PATH):
    """建立数据库连接（启用外键与行工厂）。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH):
    """建表 + 从设备汇总表播种 devices 表（幂等）。"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = get_conn(db_path)
    conn.executescript(_SCHEMA)

    # 播种设备档案（若 devices 表为空）
    cur = conn.execute("SELECT COUNT(*) FROM devices")
    if cur.fetchone()[0] == 0:
        summary = pd.read_csv(SUMMARY_PATH, encoding="utf-8-sig")
        now = _now()
        rows = [
            (int(r.unit_id), int(r.failure_cycle),
             float(r.w0), float(r.A), float(r.B), now)
            for r in summary.itertuples()
        ]
        conn.executemany(
            "INSERT INTO devices (unit_id, failure_cycle, w0, A, B, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    conn.commit()
    conn.close()


def list_devices(db_path=DB_PATH):
    """返回所有设备档案。"""
    conn = get_conn(db_path)
    rows = conn.execute("SELECT * FROM devices ORDER BY unit_id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_device(unit_id, db_path=DB_PATH):
    """返回单台设备档案，不存在返回 None。"""
    conn = get_conn(db_path)
    row = conn.execute("SELECT * FROM devices WHERE unit_id = ?", (unit_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def insert_prediction(unit_id, pred_wear_mm, rul, alert_level, message, db_path=DB_PATH):
    """写入一条预测记录，若非正常状态同时写入预警表。"""
    now = _now()
    conn = get_conn(db_path)
    conn.execute(
        "INSERT INTO predictions (unit_id, pred_wear_mm, rul, alert_level, message, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (unit_id, pred_wear_mm, rul, alert_level, message, now),
    )
    if alert_level != "normal":
        conn.execute(
            "INSERT INTO alerts (unit_id, level, message, created_at) "
            "VALUES (?, ?, ?, ?)",
            (unit_id, alert_level, message, now),
        )
    conn.commit()
    conn.close()


def list_predictions(limit=50, db_path=DB_PATH):
    """返回最近若干条预测记录。"""
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_alerts(limit=50, db_path=DB_PATH):
    """返回最近若干条预警事件。"""
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"[完成] 数据库已初始化：{DB_PATH}")
    print(f"        设备数={len(list_devices())}")
