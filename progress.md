# 进度记录

## 阶段 1（已完成 ✅）
数据来源准备（自建指数磨损数据 30 设备/1.2 万条）、数据预处理（去趋势+IQR+归一化+RUL+滑动窗口）、提示词追溯。

## 阶段 2（已完成 ✅）
- `src/model.py`：`WearRNN` 模型（RNN+ReLU）
- `src/train.py`：超参数网格搜索（智能搜索）+ 训练 + 保存模型
- 训练结果：最优 hidden=32/lr=1e-3，独立测试集 MAE≈0.0041、RMSE≈0.0052
- 产物：`models/wear_rnn.pt`、`outputs/train_loss.png`、`outputs/train_report.json`

## 阶段 3（已完成 ✅）
- `src/predict.py`：单步磨损预测 + 滚动前向 RUL 估算（状态空间搜索）+ 三级故障预警 + 可视化
- 产物：`outputs/demo_prediction.png`

## 阶段 4（已完成 ✅）
- `src/database.py`：SQLite 数据层（devices/predictions/alerts 三表）
- `src/app.py`：FastAPI 后端（REST API + 静态前端 + lifespan 初始化）
- `web/index.html`：单页前端 UI（选设备→看曲线→预测→RUL/预警）
- `tests/`：pytest 自动化测试（模型/预测/数据库/API 共 14 用例，全部通过）
- `需求规格说明书.md`：补齐提交物
- 更新 README / 方案设计 / prompt 追溯

## 验证
- `python -m pytest` → 14 passed
- `uvicorn src.app:app` 冒烟：health / predict / 前端页面均正常返回
