@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo 设置控制台编码为UTF-8...
echo.
conda run -n speech_system python %*
echo.
echo 程序运行完毕，按任意键退出...
pause >nul
