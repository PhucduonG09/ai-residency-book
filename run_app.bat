@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not exist "%CONDA_BAT%" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
call "%CONDA_BAT%" activate residency-book
if errorlevel 1 (
  echo Chua co moi truong residency-book. Hay chay setup.bat truoc.
  pause
  exit /b 1
)
python -m streamlit run app\app.py
pause
