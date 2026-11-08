@echo off
chcp 65001 > nul
cd /d "%~dp0"
title String Art Studio Pro v9.0

echo ========================================================
echo   String Art Studio Pro v9.0 (Island GA + Fast NumPy)
echo ========================================================
echo.
echo Запуск приложения...
echo.

python string-art-generator.py
if errorlevel 1 (
    py -3 string-art-generator.py
)
if errorlevel 1 (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" string-art-generator.py
)
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Не удалось запустить string-art-generator.py
    echo Убедитесь, что Python установлен и доступен в PATH.
    pause
)
