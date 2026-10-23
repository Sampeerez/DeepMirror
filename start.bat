@echo off
rem OpenVoiceChanger launcher for Windows. Double-click it or run it from a terminal.
rem It installs uv (per user, no admin) if needed; uv then fetches Python and every
rem dependency into .venv and starts the studio. Extra arguments are passed through,
rem e.g.  start.bat --port 8080  or  start.bat --device cpu
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv, the Python package manager - one time only...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>nul
if errorlevel 1 (
  echo Could not install uv. See https://docs.astral.sh/uv/getting-started/installation/
  pause
  exit /b 1
)

rem PyTorch build: CUDA when an NVIDIA driver is present, CPU otherwise.
rem Force one with:  set OVC_TORCH=cpu   (or cuda)  before running this script.
if not defined OVC_TORCH (
  where nvidia-smi >nul 2>nul && (set "OVC_TORCH=cuda") || (set "OVC_TORCH=cpu")
)

uv run --extra %OVC_TORCH% --extra neural-f0 openvoicechanger %*
if errorlevel 1 pause
