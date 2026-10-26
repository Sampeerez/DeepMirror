@echo off
rem VoxClone launcher for Windows. Double-click it.
rem It installs uv (per user, no admin) if needed and prepares Python and every
rem dependency in .venv (visible here, the first time it takes a few minutes).
rem The studio then starts WITHOUT a console window and opens in the browser;
rem its log is in the Logs tab. Closing the studio page stops it.
rem The first time it asks whether to add a Desktop and Start-menu shortcut
rem (the answer is kept in data\shortcuts.txt; delete it to be asked again,
rem or set VOXCLONE_SHORTCUTS=yes or no beforehand).
rem Extra arguments are passed through, e.g.  start.bat --port 8080
setlocal
cd /d "%~dp0"
rem The app's name (scripts\rename_project.py changes it everywhere).
set "APP_NAME=VoxClone"

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

rem Shortcuts that start the studio with no console window (uvw = uv without a console),
rem only if the user wants them. Shortcuts of the old name no longer work: remove them.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {" ^
  "  Remove-Item -LiteralPath (Join-Path $dir 'OpenVoiceChanger.lnk') -ErrorAction SilentlyContinue }" >nul 2>nul
set "SHORTCUTS=%VOXCLONE_SHORTCUTS%"
if not defined SHORTCUTS if exist "data\shortcuts.txt" set /p SHORTCUTS=<"data\shortcuts.txt"
if not defined SHORTCUTS call :ask_shortcuts
for /f "delims=" %%U in ('where uvw 2^>nul') do set "UVW=%%U"
if defined UVW (
  if /i "%SHORTCUTS%"=="yes" powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$s = New-Object -ComObject WScript.Shell;" ^
    "foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {" ^
    "  $l = $s.CreateShortcut((Join-Path $dir '%APP_NAME%.lnk'));" ^
    "  $l.TargetPath = '%UVW%'; $l.Arguments = 'run --no-sync voxclone';" ^
    "  $l.WorkingDirectory = '%~dp0'; $l.Description = '%APP_NAME%';" ^
    "  $l.IconLocation = 'shell32.dll,168'; $l.Save() }" >nul 2>nul
  start "" "%UVW%" run --no-sync voxclone %*
) else (
  rem Older uv without uvw: run here (closing this window stops the studio).
  uv run --no-sync voxclone %*
)
exit /b 0

:ask_shortcuts
choice /c YN /n /m "Create a Desktop and Start menu shortcut for %APP_NAME%? [Y/N] "
if errorlevel 2 (set "SHORTCUTS=no") else (set "SHORTCUTS=yes")
if not exist "data" mkdir "data"
>"data\shortcuts.txt" echo %SHORTCUTS%
exit /b 0
