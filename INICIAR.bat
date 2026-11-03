@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0caddy.exe" (
    echo.
    echo   [ERROR] No se encontro caddy.exe en esta carpeta.
    echo.
    echo   1. Descarga Caddy para Windows ^(amd64^):
    echo      https://caddyserver.com/download
    echo   2. Descomprime el ZIP.
    echo   3. Copia caddy.exe junto a este INICIAR.bat.
    echo.
    pause
    exit /b 1
)

echo.
echo   Adivina Adivinador - servidor local
echo   Abre: http://localhost:8000
echo   Para detenerlo, cierra la ventana "Servidor Adivina Adivinador".
echo.

start "Servidor Adivina Adivinador" "%~dp0caddy.exe" file-server --root "%~dp0" --listen 127.0.0.1:8000
timeout /t 2 /nobreak >nul
start "" "http://localhost:8000"
endlocal
