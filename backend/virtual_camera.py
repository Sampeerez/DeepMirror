"""Virtual camera for calls: the cloned face as a camera in Discord, Meet, Teams...

The face process feeds OBS Virtual Camera through pyvirtualcam. OBS Studio
itself is not needed: on Windows only its virtual camera module (two DLLs of
about 200 KB, from OBS's official release zip on GitHub) is registered. The
zip is about 220 MB, so it is read over HTTP range requests and only those
two files are fetched; each is checked by SHA-256 before it is registered.
macOS needs OBS installed (its camera is a system extension), Linux the
v4l2loopback module.
"""

from __future__ import annotations

import hashlib
import io
import logging
import platform
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

OBS_VERSION = "33.0.0"
OBS_ZIP = (f"https://github.com/obsproject/obs-studio/releases/download/{OBS_VERSION}/"
           f"OBS-Studio-{OBS_VERSION}-Windows-x64.zip")
# Path inside the zip -> SHA-256 of the file.
MODULES = {
    "core/win-dshow/data/obs-virtualcam-module64.dll":
        "add42313e4143ddef98f465ad3050726ea400f8bec312ab7b34d942f317af9ae",
    "core/win-dshow/data/obs-virtualcam-module32.dll":
        "638968efd895cef57496017889941b0d62bcd18236f44be21c08fa90ccaec77c",
}
CLSID = "{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"   # OBS Virtual Camera's DirectShow filter
DEVICE_NAME = "OBS Virtual Camera"
OBS_PAGE = "https://obsproject.com/download"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined]


@dataclass(slots=True)
class VirtualCameraStatus:
    platform: str
    supported: bool
    installed: bool
    device: str = DEVICE_NAME
    state: str = "idle"            # idle | downloading | installing | done | error
    message: str | None = None
    help_url: str | None = None    # where to get it when it cannot be installed from here

    def to_dict(self) -> dict:
        return asdict(self)


class RangeFile(io.RawIOBase):
    """A remote file read on demand with HTTP Range requests (seekable, for zipfile)."""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.timeout = timeout
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            self.url = response.url          # GitHub redirects to its file host
            self.size = int(response.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, buffer) -> int:  # type: ignore[override]
        count = min(len(buffer), self.size - self.pos)
        if count <= 0:
            return 0
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={self.pos}-{self.pos + count - 1}", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = response.read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def extract_modules(archive: zipfile.ZipFile, target: Path, modules: dict[str, str] = MODULES) -> list[Path]:
    """Copy the virtual camera DLLs out of OBS's zip into ``target``, each checked by SHA-256."""
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, sha256 in modules.items():
        data = archive.read(name)
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha256:
            raise RuntimeError(f"{Path(name).name} does not match its known checksum ({digest[:12]}…)")
        path = target / Path(name).name
        path.write_bytes(data)
        paths.append(path)
    return paths


def _run(command: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None


class VirtualCamera:
    def __init__(self, drivers: Path) -> None:
        self.folder = drivers / "obs-virtualcam"
        self.system = platform.system()
        self._lock = threading.Lock()
        self._state = "idle"
        self._message: str | None = None
        self._cache: tuple[float, bool] | None = None

    def _detect(self) -> bool:
        if self._cache and time.monotonic() - self._cache[0] < 5.0:
            return self._cache[1]
        if self.system == "Windows":
            result = _run(["reg", "query", rf"HKCR\CLSID\{CLSID}\InprocServer32"])
            found = bool(result and result.returncode == 0)
        elif self.system == "Darwin":
            found = Path("/Library/SystemExtensions").exists() and any(
                Path("/Applications").glob("OBS*.app"))
        elif self.system == "Linux":
            found = any(Path("/sys/devices/virtual/video4linux").glob("video*"))
        else:
            found = False
        self._cache = (time.monotonic(), found)
        return found

    def status(self) -> dict:
        supported = self.system in {"Windows", "Darwin", "Linux"}
        help_url = OBS_PAGE if self.system == "Darwin" else None
        return VirtualCameraStatus(self.system, supported, self._detect() if supported else False,
                                   state=self._state, message=self._message, help_url=help_url).to_dict()

    def install(self, wait: bool = False) -> dict:
        """Register the virtual camera (in the background unless ``wait``)."""
        with self._lock:
            if self._state in {"downloading", "installing"}:
                return self.status()
            self._state, self._message = "downloading", None
        if wait:
            self._install()
        else:
            threading.Thread(target=self._install, name="virtual-camera", daemon=True).start()
        return self.status()

    def _install(self) -> None:
        try:
            if self.system == "Windows":
                self._install_windows()
            elif self.system == "Linux":
                self._install_linux()
            else:
                raise RuntimeError(f"Install OBS Studio from {OBS_PAGE}, then open {DEVICE_NAME} once.")
        except Exception as exc:
            logger.exception("Virtual camera installation failed")
            self._state, self._message = "error", str(exc)
            return
        self._cache = None
        self._state = "done"
        self._message = None if self._detect() else "Installed, but the system does not list it yet."
        logger.info("Virtual camera: %s", self._message or "ready")

    def _install_windows(self) -> None:
        if not all((self.folder / Path(name).name).is_file() for name in MODULES):
            logger.info("Fetching OBS Virtual Camera from %s", OBS_ZIP)
            with zipfile.ZipFile(io.BufferedReader(RangeFile(OBS_ZIP), 1 << 16)) as archive:
                extract_modules(archive, self.folder)
        self._state = "installing"
        self._message = "Allow the administrator prompt."
        # One elevated prompt registers both (32-bit call clients use the 32-bit one).
        script = self.folder / "register.cmd"
        script.write_text("".join(f'regsvr32.exe /s "{self.folder / Path(name).name}"\r\n' for name in MODULES),
                          encoding="ascii")
        quoted = str(script).replace("'", "''")
        command = (f"Start-Process -FilePath cmd.exe -ArgumentList '/c','\"{quoted}\"' "
                   "-Verb RunAs -Wait -WindowStyle Hidden")
        result = _run(["powershell", "-NoProfile", "-Command", command], timeout=300.0)
        if result is None or result.returncode != 0:
            raise RuntimeError("The administrator prompt was declined.")

    def _install_linux(self) -> None:
        self._state = "installing"
        result = _run(["pkexec", "modprobe", "v4l2loopback", "devices=1", "exclusive_caps=1",
                       f"card_label={DEVICE_NAME}"], timeout=120.0)
        if result is None or result.returncode != 0:
            raise RuntimeError("v4l2loopback could not be loaded (install the v4l2loopback-dkms package).")
