@echo off
setlocal
title Sistema de Identificacao de Placas - Web Fast

echo ========================================================
echo   Iniciando o Sistema de Identificacao de Placas...
echo   Interface Web Moderna (FastAPI + Tailwind CSS)
echo ========================================================
echo.

rem Garante a execucao a partir da pasta raiz do projeto
cd /d "%~dp0"

rem Define o comando do Python (python ou launcher py)
set "PY="
python --version >nul 2>&1 && set "PY=python"
if not defined PY (
    py --version >nul 2>&1 && set "PY=py"
)
if not defined PY (
    echo [ERRO] Python nao foi encontrado no PATH!
    echo Instale o Python ou adicione-o as variaveis de ambiente.
    echo.
    pause
    exit /b 1
)

rem Ativa ambiente virtual caso exista na pasta
if exist "venv\Scripts\activate.bat" (
    echo Ativando ambiente virtual venv...
    call "venv\Scripts\activate.bat"
    set "PY=python"
) else if exist ".venv\Scripts\activate.bat" (
    echo Ativando ambiente virtual .venv...
    call ".venv\Scripts\activate.bat"
    set "PY=python"
)

rem Verifica se o FastAPI e Uvicorn estao instalados; se nao, instala as dependencias
%PY% -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Dependencias web nao encontradas. Instalando requirements.txt...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERRO] Falha ao instalar as dependencias.
        pause
        exit /b 1
    )
)

set "PORTA=8000"
set "URL=http://localhost:%PORTA%"

echo.
echo Iniciando servidor Web FastAPI em %URL%
echo O navegador sera aberto automaticamente em instantes.
echo Para encerrar o servidor, feche esta janela ou pressione Ctrl+C.
echo.

rem Abre o navegador apos 3 segundos quando o servidor subir
start "" /min cmd /c "timeout /t 3 /nobreak >nul & start "" %URL%"

rem Executa o servidor FastAPI com Uvicorn (usando 127.0.0.1 para evitar bloqueios de firewall do Windows)
%PY% -m uvicorn src.server:app --host 127.0.0.1 --port %PORTA%

if errorlevel 1 (
    echo.
    echo [AVISO] Tentando porta alternativa 8080...
    set "PORTA=8080"
    set "URL=http://localhost:8080"
    start "" /min cmd /c "timeout /t 3 /nobreak >nul & start "" %URL%"
    %PY% -m uvicorn src.server:app --host 127.0.0.1 --port 8080
    if errorlevel 1 (
        echo.
        echo [AVISO] A aplicacao foi finalizada com erro.
        pause
    )
)
endlocal
