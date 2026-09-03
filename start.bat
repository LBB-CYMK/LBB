@echo off
setlocal

rem ============================================================
rem  设备磨损趋势预测与故障预警系统 —— 一键启动脚本
rem  说明：仅在本脚本同目录（%~dp0）下搜索 Python 解释器，
rem        不使用系统 PATH，保证运行环境一致。
rem ============================================================

rem 切换到本脚本所在目录（同目录锚点）
cd /d "%~dp0"

rem ---- 同目录下搜索 python.exe（按优先级依次查找，不查 PATH）----
set "PYEXE="
if exist "%~dp0python.exe"                        set "PYEXE=%~dp0python.exe"
if not defined PYEXE if exist "%~dp0.venv\Scripts\python.exe" set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0venv\Scripts\python.exe"  set "PYEXE=%~dp0venv\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0python\python.exe"        set "PYEXE=%~dp0python\python.exe"
if not defined PYEXE if exist "%~dp0runtime\python.exe"       set "PYEXE=%~dp0runtime\python.exe"

if not defined PYEXE (
    echo [错误] 未在同目录找到 python.exe。
    echo        请将 Python 解释器放到本目录，或放到 .venv\、python\ 子目录后重试。
    echo        例如在项目目录执行：python -m venv --system-site-packages .venv
    pause
    exit /b 1
)

echo [信息] 使用解释器：%PYEXE%

rem ---- 若模型尚未训练，先训练（保证一键可跑）----
if not exist "%~dp0models\wear_rnn.pt" (
    echo [信息] 未检测到模型，开始训练（约需数秒）...
    "%PYEXE%" "%~dp0src\train.py"
    if errorlevel 1 (
        echo [错误] 训练失败，请检查依赖是否安装齐全。
        pause
        exit /b 1
    )
)

rem ---- 在新窗口启动后端服务，稍后自动打开浏览器 ----
echo [信息] 正在启动后端服务 http://127.0.0.1:8001 ...
start "WearPredict Server" "%PYEXE%" -m uvicorn src.app:app --host 127.0.0.1 --port 8001

ping -n 4 127.0.0.1 >nul
start "" http://127.0.0.1:8001

echo [信息] 已启动，关闭服务器窗口即可停止服务。
endlocal
