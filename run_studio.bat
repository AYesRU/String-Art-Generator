@echo off
cd /d "%~dp0"
chcp 65001 > nul
title String Art Studio Pro v9.0
echo Запуск String Art Studio Pro v9.0...

python string-art-generator.py
if errorlevel 1 (
    py -3 string-art-generator.py
)
if errorlevel 1 (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" string-art-generator.py
)
if errorlevel 1 (
    echo.
    echo Ошибка: Не удалось запустить программу.
    pause
)
