# -*- coding: utf-8 -*-
"""
FastAPI 后端服务
====================================
课程设计：基于深度学习的设备磨损趋势预测与故障预警系统

后端承载业务逻辑：设备档案查询、磨损历史查询、磨损预测、RUL 估算、
故障预警，并将预测/预警结果写入 SQLite 数据库；同时托管前端静态页面。

对应任务书「完整 B/S 架构」的后端服务（API 层）部分。

启动方式（在仓库根目录执行）：
    uvicorn src.app:app --reload
    # 或： cd src && uvicorn app:app --reload
    浏览器访问 http://127.0.0.1:8001
"""

import os
import sys

# 保证无论从哪个目录启动，都能 import 同目录的 model/predict/database 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import predict
import database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "web", "index.html")

@asynccontextmanager
async def lifespan(_app):
    """应用启动时初始化数据库（建表 + 播种设备档案）。"""
    database.init_db()
    yield


app = FastAPI(
    title="设备磨损趋势预测与故障预警系统",
    version="1.0.0",
    lifespan=lifespan,
)

# 模型懒加载缓存（首次请求时加载，避免启动阻塞）
_assets = {}


def _get_assets():
    if "model" not in _assets:
        model, wear_params, threshold_norm = predict.load_assets()
        _assets.update(
            model=model, wear_params=wear_params, threshold_norm=threshold_norm
        )
    return _assets["model"], _assets["wear_params"], _assets["threshold_norm"]


class PredictRequest(BaseModel):
    unit_id: int


@app.get("/")
def index():
    """前端入口页面。"""
    return FileResponse(INDEX_PATH)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "wear-predict", "model_loaded": "model" in _assets}


@app.get("/api/devices")
def devices():
    """设备列表（含退化参数与失效周期）。"""
    return database.list_devices()


@app.get("/api/devices/{unit_id}/history")
def device_history(unit_id: int):
    """指定设备的磨损/温度历史曲线数据。"""
    if database.get_device(unit_id) is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    g = predict.load_device(unit_id)
    return {
        "unit_id": unit_id,
        "failure_cycle": int(g["time_cycle"].max()),
        "cycles": g["time_cycle"].tolist(),
        "wear": g["wear_clean"].tolist(),        # 清洗后磨损量（mm）
        "temperature": g["temp_clean"].tolist(),  # 清洗后温度（℃）
    }


@app.post("/api/predict")
def do_predict(req: PredictRequest):
    """对指定设备做磨损预测 + RUL 估算 + 故障预警，结果入库。"""
    if database.get_device(req.unit_id) is None:
        raise HTTPException(status_code=404, detail="设备不存在")

    model, wear_params, threshold_norm = _get_assets()
    _, pred_mm, rul, level, msg = predict.forecast_device(
        model, req.unit_id, wear_params, threshold_norm
    )
    database.insert_prediction(req.unit_id, round(pred_mm, 5), rul, level, msg)

    device = database.get_device(req.unit_id)
    return {
        "unit_id": req.unit_id,
        "pred_wear_mm": round(pred_mm, 5),
        "threshold_mm": 1.0,
        "rul": rul,
        "level": level,
        "message": msg,
        # 可执行检修建议：让前端能单独、醒目地呈现运维决策结论（业务闭环）
        "advice": predict.get_maintenance_advice(level),
        "failure_cycle": device["failure_cycle"],
    }


@app.get("/api/predictions")
def predictions():
    """最近预测记录。"""
    return database.list_predictions()


@app.get("/api/alerts")
def alerts():
    """最近预警事件。"""
    return database.list_alerts()
