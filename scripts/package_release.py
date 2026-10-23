"""Build the release archive: tracked sources plus the prebuilt web UI.

The archive lets people without Node.js run the studio: unzip, then run
``start.bat`` / ``start.sh``. Usage (from a clean, built checkout)::

    cd frontend && npm ci && npm run build && cd ..
    python scripts/package_release.py [--check-tag v3.0.0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXECUTABLES = {"start.sh"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def read_version() -> str:
    return (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def build_archive(output: Path) -> Path:
    version = read_version()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise SystemExit("Release packaging needs a clean checkout.")
    package = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    if package.get("version") != version:
        raise SystemExit("frontend/package.json must match VERSION.")
    dist = ROOT / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        raise SystemExit("Build the frontend first (npm ci && npm run build in frontend/).")

    files = [Path(name) for name in git("ls-files").splitlines()]
    files += [path.relative_to(ROOT) for path in sorted(dist.rglob("*")) if path.is_file()]
    prefix = f"OpenVoiceChanger-{version}"
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / f"{prefix}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            entry = zipfile.ZipInfo(f"{prefix}/{relative.as_posix()}", date_time=(1980, 1, 1, 0, 0, 0))
            mode = 0o755 if relative.name in EXECUTABLES else 0o644
            entry.external_attr = (stat.S_IFREG | mode) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, (ROOT / relative).read_bytes())
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_path.with_suffix(".zip.sha256").write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "releases")
    parser.add_argument("--check-tag", help="fail unless the tag is v<VERSION>")
    args = parser.parse_args()
    if args.check_tag and args.check_tag != f"v{read_version()}":
        parser.error("The release tag must be v followed by VERSION.")
    print(build_archive(args.output))


if __name__ == "__main__":
    main()
