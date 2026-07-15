@echo off
cd /d "%~dp0"

set "LOCAL_CODEX_PY=C:\Users\bison\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if exist "%LOCAL_CODEX_PY%" (
  "%LOCAL_CODEX_PY%" typing_practice.py
  exit /b %ERRORLEVEL%
)

python typing_practice.py
