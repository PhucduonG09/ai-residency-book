@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not exist "%CONDA_BAT%" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not exist "%CONDA_BAT%" (
  echo Khong tim thay conda. Mo "Anaconda Prompt", cd vao thu muc nay roi chay:
  echo     conda env create -f environment.yml
  pause
  exit /b 1
)
call "%CONDA_BAT%" env list | findstr /C:"residency-book" >nul
if %errorlevel%==0 (
  echo Cap nhat moi truong residency-book...
  call "%CONDA_BAT%" env update -n residency-book -f environment.yml --prune
) else (
  echo Tao moi truong residency-book lan dau, co the mat 5-15 phut...
  call "%CONDA_BAT%" env create -f environment.yml
)
if errorlevel 1 (
  echo.
  echo Cai dat chua xong. Xem thong bao loi o tren, roi chay lai setup.bat.
  pause
  exit /b 1
)
echo.
echo Cai giong doc cho video Manim (edge-tts)...
call "%CONDA_BAT%" run -n residency-book python -m pip install -q edge-tts
echo.
echo Kiem tra PyTorch...
call "%CONDA_BAT%" run -n residency-book python -c "import torch; print('PyTorch', torch.__version__)"
if errorlevel 1 echo Chua import duoc PyTorch. Lab chuong 6-12 can no; xem muc Cai dat trong README.
echo.
echo Xong. Nhap dup run_app.bat de mo app.
pause
