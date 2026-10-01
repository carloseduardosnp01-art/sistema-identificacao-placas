@echo off
chcp 65001 > nul
title Sistema de Identificação de Placas

echo ========================================================
echo   Iniciando o Sistema de Identificação de Placas...
echo ========================================================
echo.

:: Garante a execução a partir da pasta raiz do projeto
cd /d "%~dp0"

:: Verifica se o Python está acessível no PATH
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python não foi encontrado no sistema ou não está no PATH!
    echo Por favor, instale o Python ou adicione-o às variáveis de ambiente.
    echo.
    pause
    exit /b %errorlevel%
)

:: Ativa ambiente virtual caso exista na pasta
if exist "venv\Scripts\activate.bat" (
    echo Ativando ambiente virtual (venv)...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Ativando ambiente virtual (.venv)...
    call .venv\Scripts\activate.bat
)

echo Iniciando servidor Streamlit...
echo O navegador será aberto automaticamente em http://localhost:8501
echo Para encerrar a aplicação, feche esta janela ou pressione Ctrl+C.
echo.

python -m streamlit run src/app.py

if %errorlevel% neq 0 (
    echo.
    echo [AVISO] A aplicação foi finalizada com código %errorlevel%.
    pause
)
