@echo off
rem Emberlens SDK launcher (Windows).
rem The first run makes a private Python environment in python\.venv.
rem Nothing is installed system-wide. Start with:  emberlens help
setlocal
set "HERE=%~dp0"
set "VENV=%HERE%python\.venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
  echo First run: setting up the Emberlens SDK ^(a few seconds^)...
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv "%VENV%"
  ) else (
    python -m venv "%VENV%"
  )
)
if not exist "%PY%" (
  echo The Emberlens SDK needs Python 3.9 or newer.
  echo Get it from https://www.python.org/downloads/ and run this again.
  exit /b 1
)

set "EMBERLENS_PROG=emberlens.cmd"
set "PYTHONPATH=%HERE%python;%PYTHONPATH%"
"%PY%" -m emberlens %*
