@echo off
chcp 65001 > nul
title String Art Studio Pro v9.0
echo Запуск String Art Studio Pro v9.0...
python string-art-generator.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Ошибка при выполнении программы. Нажмите любую клавишу для выхода...
    pause > nul
)
