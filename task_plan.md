# 阶段 2–4 实现计划（全部完成）

## 总目标
将「纯脚本」升级为任务书要求的「完整 B/S 可运行 demo」，补齐算法、后端、数据库、前端、自动化测试与文档。

## 技术栈
算法 PyTorch RNN · 后端 FastAPI + uvicorn · 数据库 SQLite · 前端 单页 HTML + ECharts(CDN) · 测试 pytest

## 阶段 2 —— RNN 模型搭建与训练（✅ 完成）
- [x] `src/model.py`：定义 `WearRNN`（RNN + ReLU，batch_first），含保存/加载
- [x] `src/train.py`：加载 `sequences.npz` → 超参网格搜索（智能搜索落地）→ 训练 → 保存模型与曲线
- [x] 运行训练：测试集 MAE≈0.0041

## 阶段 3 —— 预测 / RUL / 预警 + 可视化（✅ 完成）
- [x] `src/predict.py`：单步磨损预测 + 滚动前向 RUL + 三级故障预警 + matplotlib 可视化

## 阶段 4 —— 系统集成（✅ 完成）
- [x] `src/database.py`：SQLite 数据层（devices / predictions / alerts）
- [x] `src/app.py`：FastAPI 后端 API + 托管静态前端
- [x] `web/index.html`：前端 UI（选设备 → 看曲线 → 预测 → RUL/预警 → 预警日志）
- [x] `tests/`：pytest 自动化测试（14 用例全通过）
- [x] `需求规格说明书.md`：补齐提交物
- [x] 更新 README / 方案设计 / prompt 追溯记录

## 交付物 → 任务书映射（已达成）
| 任务书要求 | 落地产物 |
|---|---|
| 算法模块(≥3 方向) | 数据预处理(IQR) + RNN 预测 + 智能搜索(超参网格) + 状态空间前向模拟(RUL) |
| 后端服务 | `src/app.py` (FastAPI REST) |
| 数据库 | `src/database.py` (SQLite) |
| 前端 UI | `web/index.html` |
| 自动化测试 | `tests/` (pytest) |
| 需求规格说明书 | `需求规格说明书.md` |
| 架构完整 | 上述全部联通为可运行 demo |

## 运行方式
```bash
pip install numpy pandas matplotlib torch fastapi uvicorn pytest
python src/generate_data.py      # 阶段1
python src/preprocess.py         # 阶段1
python src/train.py              # 阶段2
python -m pytest tests/          # 阶段4 测试
uvicorn src.app:app --reload     # 阶段4 启动，浏览器打开 http://127.0.0.1:8000
```
