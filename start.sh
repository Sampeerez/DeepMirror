#!/usr/bin/env sh
# DeepMirror launcher for Linux and macOS: ./start.sh [--port 8080] [--device cpu]
# Installs uv (per user, no root) if needed and prepares Python and every
# dependency in .venv (visible here; the first time takes a few minutes). The
# studio then runs in this terminal and opens in the browser: closing the page
# leaves it running; press Ctrl+C here to stop it (everything it started stops
# with it). On Linux it asks once whether to add an app-menu entry (answer kept
# in data/shortcuts.txt, or set DEEPMIRROR_SHORTCUTS=yes|no).
set -eu
cd "$(dirname "$0")"
HERE="$(pwd)"
APP_NAME="DeepMirror"
APP_SLUG="deepmirror"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv, the Python package manager (one time only)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
  export PATH
fi

# PyTorch build: CUDA when an NVIDIA driver is present, CPU otherwise (macOS
# gets the standard build, which uses the Apple GPU through MPS).
# Force one with OVC_TORCH=cpu or OVC_TORCH=cuda.
if [ -z "${OVC_TORCH:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then OVC_TORCH=cuda; else OVC_TORCH=cpu; fi
fi

echo "Preparing $APP_NAME ($OVC_TORCH)..."
uv sync --extra "$OVC_TORCH" --extra neural-f0

# --foreground is accepted for older shortcuts (the studio always runs in the terminal now).
ARGS=""
for arg in "$@"; do [ "$arg" != "--foreground" ] && ARGS="$ARGS $arg"; done

SHORTCUTS="${DEEPMIRROR_SHORTCUTS:-${VOXCLONE_SHORTCUTS:-}}"
if [ -z "$SHORTCUTS" ] && [ -f data/shortcuts.txt ]; then SHORTCUTS="$(cat data/shortcuts.txt)"; fi
if [ -z "$SHORTCUTS" ] && [ "$(uname)" = "Linux" ] && [ -t 0 ]; then
  printf "Add %s to the applications menu? [y/N] " "$APP_NAME"
  read -r answer
  case "$answer" in [yY]*) SHORTCUTS=yes ;; *) SHORTCUTS=no ;; esac
  mkdir -p data && echo "$SHORTCUTS" > data/shortcuts.txt
fi

if [ "$(uname)" = "Linux" ] && [ -d "$HOME/.local/share" ] && [ "$SHORTCUTS" = "yes" ]; then
  mkdir -p "$HOME/.local/share/applications"
  UV_BIN="$(command -v uv)"
  cat > "$HOME/.local/share/applications/$APP_SLUG.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=Realtime voice and face cloning
Exec=sh -c 'cd "$HERE" && exec "$UV_BIN" run --no-sync deepmirror'
Path=$HERE
Icon=audio-input-microphone
Terminal=true
Categories=AudioVideo;Audio;
EOF
fi

# shellcheck disable=SC2086
exec uv run --no-sync deepmirror $ARGS
