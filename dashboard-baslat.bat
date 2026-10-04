@echo off
chcp 65001 >nul
setlocal
title Churn Dashboard

rem Proje klasörüne geç (bat dosyasının bulunduğu klasör)
cd /d "%~dp0"

rem Port: varsayılan 8501; test için DASHBOARD_PORT ortam değişkeniyle değiştirilebilir
if "%DASHBOARD_PORT%"=="" (set "PORT=8501") else (set "PORT=%DASHBOARD_PORT%")
set "URL=http://localhost:%PORT%"
set "STREAMLIT=%~dp0.venv\Scripts\streamlit.exe"

if not exist "%STREAMLIT%" (
    echo.
    echo HATA: Sanal ortam bulunamadı: "%STREAMLIT%"
    echo Önce README.md içindeki "Kurulum" adımlarını çalıştırın:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

rem Port zaten dinleniyorsa yeni sunucu açma, yalnız tarayıcıyı aç
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if %errorlevel%==0 (
    echo Dashboard zaten çalışıyor: %URL%
    echo Tarayıcı açılıyor...
    if not "%DASHBOARD_NO_BROWSER%"=="1" start "" "%URL%"
    timeout /t 5 >nul 2>&1
    exit /b 0
)

echo Dashboard başlatılıyor, lütfen bekleyin...
echo.
echo ============================================================
echo  Dashboard çalışıyor: %URL%
echo  Kapatmak için bu pencereyi kapatın veya Ctrl+C
echo ============================================================
echo.

rem Arka planda sağlık kontrolü yap; sunucu hazır olunca tarayıcıyı aç (en fazla ~60 sn)
if not "%DASHBOARD_NO_BROWSER%"=="1" start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='%URL%'; for ($i=0; $i -lt 60; $i++) { try { if ((Invoke-WebRequest ($u + '/_stcore/health') -UseBasicParsing -TimeoutSec 2).Content -eq 'ok') { Start-Process $u; exit } } catch {}; Start-Sleep -Seconds 1 }"

rem headless: e-posta sorusu sorulmaz ve Streamlit tarayıcıyı kendisi açmaz
"%STREAMLIT%" run app.py --server.headless true --server.port %PORT%

echo.
echo Dashboard kapandı.
pause
