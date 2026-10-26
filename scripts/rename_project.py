"""Give the project a new name everywhere it is shown.

    uv run python scripts/rename_project.py "My Voice App"
    uv run python scripts/rename_project.py "My Voice App" --command myvoiceapp

The display name appears in the app header, the browser tab, the logs, the
launchers and their shortcuts, and the virtual microphone on Linux. With
``--command`` the Python package and its command (``uv run <command>``) are
renamed too; run ``uv sync`` afterwards.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def substitute(path: str, pattern: str, replacement: str, count: int = 0) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    updated, n = re.subn(pattern, lambda _: replacement, text, count=count, flags=re.MULTILINE)
    if n == 0:
        raise SystemExit(f"{path}: nothing matched {pattern!r}")
    file.write_bytes(updated.encode("utf-8"))
    print(f"  {path}: {n} change(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="the new display name, e.g. 'My Voice App'")
    parser.add_argument("--command", help="also rename the package and its command (lowercase, no spaces)")
    args = parser.parse_args()
    name = args.name.strip()
    slug = re.sub(r"[^A-Za-z0-9]", "", name) or "VoxClone"
    if any(c in name for c in "\"'`\\%$"):
        raise SystemExit("Please use a name without quotes, backslashes, % or $.")

    print(f"Renaming to {name!r} (files and devices: {slug})")
    substitute("backend/config.py", r'^APP_NAME = ".*"$', f'APP_NAME = "{name}"')
    substitute("backend/config.py", r'^APP_SLUG = "[^"]*"', f'APP_SLUG = "{slug}"')
    substitute("frontend/src/lib/brand.js", r"^export const APP_NAME = '.*';$", f"export const APP_NAME = '{name}';")
    substitute("frontend/src/lib/brand.js", r"^export const APP_SLUG = '[^']*';", f"export const APP_SLUG = '{slug}';")
    substitute("frontend/index.html", r"<title>.*</title>", f"<title>{name}</title>")
    substitute("backend/frontend_build.py", r"<title>.*?</title>", f"<title>{name}</title>")
    substitute("start.bat", r'^set "APP_NAME=.*"$', f'set "APP_NAME={name}"')
    substitute("start.sh", r'^APP_NAME=".*"$', f'APP_NAME="{name}"')
    substitute("start.sh", r'^APP_SLUG=".*"$', f'APP_SLUG="{slug.lower()}"')
    substitute("README.md", r"^# .*$", f"# {name}", count=1)

    if args.command:
        command = args.command.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", command):
            raise SystemExit("--command must be lowercase letters, digits, '-' or '_'")
        old = re.search(r'^name = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE)
        assert old is not None
        previous = old.group(1)
        substitute("pyproject.toml", rf'^name = "{previous}"', f'name = "{command}"')
        substitute("pyproject.toml", rf'^{previous} = "backend.__main__:main"', f'{command} = "backend.__main__:main"')
        substitute("backend/app.py", rf'version\("{previous}"\)', f'version("{command}")')
        substitute("backend/__main__.py", rf'prog="{previous}"', f'prog="{command}"')
        substitute("backend/__main__.py", rf"``uv run {previous}``", f"``uv run {command}``")
        for launcher in ("start.bat", "start.sh", "Makefile"):
            substitute(launcher, rf"\b{previous}\b(?!\.desktop)", command)
        substitute("frontend/package.json", rf'"name": "{previous}"', f'"name": "{command}"')
        print("Now run: uv sync  (and npm install in frontend/ to refresh package-lock.json)")
    print("Done.")


if __name__ == "__main__":
    main()
