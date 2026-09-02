# -*- coding: utf-8 -*-
"""FastAPI 后端接口的集成测试（TestClient）。"""

import pytest
import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """用上下文管理器包裹 TestClient，确保 lifespan 启动事件（init_db）执行。"""
    with TestClient(app.app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_devices(client):
    r = client.get("/api/devices")
    assert r.status_code == 200
    devices = r.json()
    assert len(devices) == 30
    assert devices[0]["unit_id"] == 1


def test_device_history(client):
    r = client.get("/api/devices/1/history")
    assert r.status_code == 200
    data = r.json()
    assert data["unit_id"] == 1
    assert len(data["cycles"]) == len(data["wear"]) > 0


def test_device_not_found(client):
    r = client.get("/api/devices/999/history")
    assert r.status_code == 404


def test_predict_endpoint(client):
    r = client.post("/api/predict", json={"unit_id": 1})
    assert r.status_code == 200
    data = r.json()
    assert "pred_wear_mm" in data
    assert "rul" in data
    assert data["level"] in {"normal", "warning", "critical"}
