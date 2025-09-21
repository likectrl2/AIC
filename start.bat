@echo off
echo 🔌 电路分析系统
echo ==================

echo 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python未安装或未添加到PATH
    pause
    exit /b 1
)

echo 检查依赖...
python -c "import flask, cv2, numpy" 2>nul
if %errorlevel% neq 0 (
    echo ❌ 缺少依赖，正在安装...
    pip install -r backend/requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ 依赖安装失败
        pause
        exit /b 1
    )
)

echo ✅ 环境检查通过

echo 启动服务器...
cd backend
python app.py

pause
