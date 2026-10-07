@echo off
setlocal
title Sistema de Identificacao de Placas

echo ========================================================
echo   Iniciando o Sistema de Identificacao de Placas...
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

rem Verifica se o Streamlit esta instalado; se nao, instala as dependencias
%PY% -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit nao encontrado. Instalando dependencias...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERRO] Falha ao instalar as dependencias.
        pause
        exit /b 1
    )
)

set "PORTA=8501"
set "URL=http://localhost:%PORTA%"

echo Iniciando servidor Streamlit em %URL%
echo O navegador sera aberto automaticamente em instantes.
echo Para encerrar, feche esta janela ou pressione Ctrl+C.
echo.

rem Abre o navegador apos alguns segundos, com o servidor ja de pe
start "" /min cmd /c "timeout /t 8 /nobreak >nul & start "" %URL%"

rem headless evita que o Streamlit abra uma segunda aba por conta propria
%PY% -m streamlit run src/app.py --server.port %PORTA% --server.headless true --browser.gatherUsageStats false

if errorlevel 1 (
    echo.
    echo [AVISO] A aplicacao foi finalizada com erro.
    pause
)
endlocal
