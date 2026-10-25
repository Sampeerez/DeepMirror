@echo off
rem Voice Studio launcher for Windows. Double-click it.
rem It installs uv (per user, no admin) if needed and prepares Python and every
rem dependency in .venv (visible here, the first time it takes a few minutes).
rem The studio then starts WITHOUT a console window and opens in the browser;
rem its log is in the Logs tab. Closing the studio page stops it.
rem A Desktop and Start-menu shortcut named after the app is created, which
rem starts it directly with no window at all.
rem Extra arguments are passed through, e.g.  start.bat --port 8080
setlocal
cd /d "%~dp0"
rem The app's name (scripts\rename_project.py changes it everywhere).
set "APP_NAME=Voice Studio"

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

echo Preparing %APP_NAME% (%OVC_TORCH%)...
uv sync --extra %OVC_TORCH% --extra neural-f0
if errorlevel 1 (
  echo.
  echo Setting up the environment failed. See the messages above.
  pause
  exit /b 1
)

rem Shortcuts that start the studio with no console window (uvw = uv without a console).
for /f "delims=" %%U in ('where uvw 2^>nul') do set "UVW=%%U"
if defined UVW (
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$s = New-Object -ComObject WScript.Shell;" ^
    "foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {" ^
    "  Remove-Item -LiteralPath (Join-Path $dir 'OpenVoiceChanger.lnk') -ErrorAction SilentlyContinue;" ^
    "  $l = $s.CreateShortcut((Join-Path $dir '%APP_NAME%.lnk'));" ^
    "  $l.TargetPath = '%UVW%'; $l.Arguments = 'run --no-sync openvoicechanger';" ^
    "  $l.WorkingDirectory = '%~dp0'; $l.Description = '%APP_NAME%';" ^
    "  $l.IconLocation = 'shell32.dll,168'; $l.Save() }" >nul 2>nul
  start "" "%UVW%" run --no-sync openvoicechanger %*
) else (
  rem Older uv without uvw: run here (closing this window stops the studio).
  uv run --no-sync openvoicechanger %*
)
exit /b 0
