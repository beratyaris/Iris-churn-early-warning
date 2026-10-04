@echo off
chcp 65001 >nul
setlocal
title Iris - Churn Erken Uyari
rem ------------------------------------------------------------------
rem  Iris'i ilk kez kurar ve calistirir (cift tiklayin).
rem  1) .venv yoksa olusturur  2) requirements.txt paketlerini kurar
rem  3) streamlit run app.py ile uygulamayi baslatir, tarayiciyi acar.
rem  Gereken: Python 3.13 (python.org, kurulumda "Add python.exe to PATH").
rem ------------------------------------------------------------------

cd /d "%~dp0"
set "VENV=%~dp0.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "MARKER=%VENV%\.iris-kurulum-tamam"
if "%DASHBOARD_PORT%"=="" (set "PORT=8501") else (set "PORT=%DASHBOARD_PORT%")
set "URL=http://localhost:%PORT%"

if exist "%MARKER%" goto calistir

rem --- Python'u bul: once "py -3.13", sonra PATH'teki "python" ---
set "PYCMD="
py -3.13 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.13"
if not defined PYCMD (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
    echo.
    echo HATA: Python 3.13 bulunamadi.
    echo python.org/downloads adresinden Python 3.13 kurun; kurulumun ilk ekraninda
    echo "Add python.exe to PATH" kutusunu isaretleyin. Sonra bu dosyaya tekrar cift tiklayin.
    echo.
    pause
    exit /b 1
)

echo Ilk kurulum yapiliyor (internet gerekir, birkac dakika surebilir)...
if not exist "%VPY%" (
    %PYCMD% -m venv "%VENV%"
    if errorlevel 1 (
        echo HATA: Sanal ortam olusturulamadi.
        pause
        exit /b 1
    )
)
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo HATA: Paketler kurulamadi. Internet baglantisini kontrol edip tekrar deneyin.
    echo Sorun surerse .venv klasorunu silip bu dosyayi yeniden calistirin.
    pause
    exit /b 1
)
echo ok> "%MARKER%"
echo Kurulum tamamlandi.

:calistir
if not exist "data\Bank Customer Churn Prediction.csv" (
    echo UYARI: data\Bank Customer Churn Prediction.csv bulunamadi. README'deki "Veri" adimina bakin.
)

echo.
echo ============================================================
echo  Iris baslatiliyor: %URL%
echo  Tarayici hazir olunca kendiliginden acilir.
echo  Kapatmak icin bu pencereyi kapatin veya Ctrl+C
echo ============================================================
echo.

rem Sunucu hazir olunca tarayiciyi ac (en fazla ~90 sn bekler)
if not "%DASHBOARD_NO_BROWSER%"=="1" start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='%URL%'; for ($i=0; $i -lt 90; $i++) { try { if ((Invoke-WebRequest ($u + '/_stcore/health') -UseBasicParsing -TimeoutSec 2).Content -eq 'ok') { Start-Process $u; exit } } catch {}; Start-Sleep -Seconds 1 }"

"%VPY%" -m streamlit run app.py --server.headless true --server.port %PORT%

echo.
echo Iris kapandi.
pause
