@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tao_shortcut.ps1"
echo.
echo Xong. Tim bieu tuong "Residency Book" tren Desktop, nhap dup de mo app.
pause
