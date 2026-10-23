"""Virtual microphone: route the converted voice into Discord, Teams, Meet...

Browsers cannot create audio devices, so calls need a virtual audio cable:
the studio plays the converted voice into the cable's input and the call app
records from its output.

* Windows and macOS: VB-CABLE (VB-Audio, donationware). Its licence allows
  sharing the official package as is but not folding it into another
  program's installer, so the studio downloads the official package
  (checksum-verified) and opens VB-Audio's own installer for the user, who
  confirms it and the administrator prompt.
* Linux: PipeWire/PulseAudio create virtual devices without installing
  anything; the studio adds a null sink plus a microphone fed from it while
  call mode is on, and removes them on exit.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend.assets import Asset, download_verified

logger = logging.getLogger(__name__)

WINDOWS_PACKAGE = Asset("vbcable-windows", "VBCABLE_Driver_Pack45.zip",
                        "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip",
                        "b950e39f01af1d04ea623c8f6d8eb9b6ea5c477c637295fabf20631c85116bfb", 1318877, False)
MAC_PACKAGE = Asset("vbcable-macos", "VBCable_MACDriver_Pack108.zip",
                    "https://download.vb-audio.com/Download_MAC/VBCable_MACDriver_Pack108.zip",
                    "e46b41c6876995403cb1da37d7c0d566f59edd83aa9fcbcdc3a205eb0b6e05c7", 4544257, False)
LINUX_SINK = "ovc_voice"
LINUX_SOURCE = "ovc_mic"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined]


@dataclass(slots=True)
class VirtualMicStatus:
    platform: str
    supported: bool
    installed: bool
    # What to pick: the studio's output device, and the microphone in the call app.
    studio_output: str
    call_input: str
    state: str = "idle"            # idle | downloading | installing | done | error
    message: str | None = None
    needs_restart: bool = False
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _run(command: list[str], timeout: float = 20.0) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout or ""


class VirtualMic:
    def __init__(self, downloads: Path) -> None:
        self.downloads = downloads
        self.system = platform.system()
        self._lock = threading.Lock()
        self._state = "idle"
        self._message: str | None = None
        self._needs_restart = False
        self._linux_modules: list[str] = []
        self._cache: tuple[float, bool] | None = None

    # --- Detection ----------------------------------------------------------

    def _detect(self) -> bool:
        if self._cache and time.monotonic() - self._cache[0] < 5.0:
            return self._cache[1]
        if self.system == "Windows":
            text = _run(["powershell", "-NoProfile", "-Command",
                         "Get-PnpDevice -Class AudioEndpoint -Status OK | ForEach-Object FriendlyName"])
            found = "vb-audio virtual cable" in text.lower() or "cable input" in text.lower()
        elif self.system == "Darwin":
            found = "vb-cable" in _run(["system_profiler", "SPAudioDataType"], 30.0).lower()
        elif self.system == "Linux":
            found = LINUX_SINK in _run(["pactl", "list", "short", "sinks"])
        else:
            found = False
        self._cache = (time.monotonic(), found)
        return found

    def status(self) -> dict:
        names = {
            "Windows": ("CABLE Input (VB-Audio Virtual Cable)", "CABLE Output (VB-Audio Virtual Cable)"),
            "Darwin": ("VB-Cable", "VB-Cable"),
            "Linux": ("OpenVoiceChanger Output", "OpenVoiceChanger Microphone"),
        }
        output, call_input = names.get(self.system, ("", ""))
        supported = self.system in names and (self.system != "Linux" or shutil.which("pactl") is not None)
        return VirtualMicStatus(self.system, supported, self._detect() if supported else False, output, call_input,
                                self._state, self._message, self._needs_restart).to_dict()

    # --- Installation -------------------------------------------------------

    def install(self) -> dict:
        """Start installing (or, on Linux, creating) the virtual cable in the background."""
        with self._lock:
            if self._state in {"downloading", "installing"}:
                return self.status()
            self._state, self._message, self._needs_restart = "downloading", None, False
        threading.Thread(target=self._install, name="virtual-mic", daemon=True).start()
        return self.status()

    def _install(self) -> None:
        try:
            if self.system == "Windows":
                self._install_windows()
            elif self.system == "Darwin":
                self._install_macos()
            elif self.system == "Linux":
                self._create_linux()
            else:
                raise RuntimeError(f"Virtual microphones are not supported on {self.system}")
        except Exception as exc:
            logger.exception("Virtual microphone installation failed")
            self._state, self._message = "error", str(exc)
            return
        self._cache = None
        if self._detect():
            self._state, self._message = "done", "The virtual microphone is ready."
        else:
            self._state, self._needs_restart = "done", self.system != "Linux"
            self._message = ("Installed. VB-Audio asks for a restart before the device appears; "
                             "restart the computer, then open the studio again.") if self._needs_restart else \
                "Created, but the system did not list it yet."
        logger.info("Virtual microphone: %s", self._message)

    def _package(self, asset: Asset) -> Path:
        target = self.downloads / asset.filename
        if not target.is_file() or target.stat().st_size != asset.size:
            logger.info("Downloading %s from VB-Audio...", asset.filename)
            download_verified(asset, target)
        folder = self.downloads / target.stem
        if folder.exists():
            shutil.rmtree(folder)
        with zipfile.ZipFile(target) as archive:
            archive.extractall(folder)
        return folder

    def _install_windows(self) -> None:
        folder = self._package(WINDOWS_PACKAGE)
        setup = folder / ("VBCABLE_Setup_x64.exe" if platform.machine().endswith("64") else "VBCABLE_Setup.exe")
        self._state = "installing"
        self._message = "Click 'Install Driver' in the VB-CABLE window and allow the administrator prompt."
        logger.info("Opening the VB-CABLE installer (%s)", setup.name)
        # VB-Audio's own installer, elevated (UAC prompt), waiting until the user closes it.
        command = (f"Start-Process -FilePath '{setup}' -WorkingDirectory '{folder}' -Verb RunAs -Wait")
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True,
                                creationflags=NO_WINDOW)
        if result.returncode != 0:
            raise RuntimeError("The installer did not run (the administrator prompt may have been declined).")

    def _install_macos(self) -> None:
        folder = self._package(MAC_PACKAGE)
        dmg = folder / "VBCable_MACDriver_Pack108.dmg"
        self._state = "installing"
        self._message = "Follow the VB-CABLE installer that just opened."
        subprocess.run(["open", "-W", str(dmg)], check=False)

    def _create_linux(self) -> None:
        self._state = "installing"
        sink = _run(["pactl", "load-module", "module-null-sink", f"sink_name={LINUX_SINK}",
                     "sink_properties=device.description=OpenVoiceChanger_Output"]).strip()
        source = _run(["pactl", "load-module", "module-remap-source", f"master={LINUX_SINK}.monitor",
                       f"source_name={LINUX_SOURCE}",
                       "source_properties=device.description=OpenVoiceChanger_Microphone"]).strip()
        self._linux_modules = [m for m in (source, sink) if m.isdigit()]
        if len(self._linux_modules) < 2:
            raise RuntimeError("PulseAudio/PipeWire refused to create the virtual devices (is pactl working?)")

    def close(self) -> None:
        """Remove what only lives while the studio runs (the Linux devices)."""
        for module in self._linux_modules:
            _run(["pactl", "unload-module", module])
        self._linux_modules = []
