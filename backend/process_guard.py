"""No process outlives the studio.

The face and head processes are children of the server. A normal exit stops them, but a closed
console window, a killed process or a crash must not leave them running on the GPU. Two guards:

* ``contain_children()`` (server, Windows): the server joins a Job Object that kills every process
  in it when the job's last handle closes, i.e. when the server is gone, however it ended.
* ``exit_with_parent()`` (each child, every platform): a thread waits on the parent and exits the
  child at once when the parent is gone.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import threading

logger = logging.getLogger(__name__)

_job = None   # kept open for the server's whole life: closing it kills the children


def contain_children() -> bool:
    """Windows: put this process (and so every child it starts) in a kill-on-close job."""
    global _job
    if sys.platform != "win32" or _job is not None:
        return _job is not None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount",
                                                         "OtherOperationCount", "ReadTransferCount",
                                                         "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kill_on_close = 0x2000
    extended_information = 9
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        logger.warning("Could not create a job object for the studio's processes")
        return False
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = kill_on_close
    if not kernel32.SetInformationJobObject(job, extended_information, ctypes.byref(limits), ctypes.sizeof(limits)):
        logger.warning("Could not configure the job object (error %d)", ctypes.get_last_error())
        return False
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        logger.warning("Could not join the job object (error %d)", ctypes.get_last_error())
        return False
    _job = job
    return True


def lower_priority() -> None:
    """This process below normal CPU priority: the live voice (in the server) comes first."""
    try:
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        else:
            os.nice(5)
    except OSError:
        logger.debug("Could not lower this process's priority")


def exit_with_parent() -> None:
    """In a child process: exit at once when the parent process is gone."""
    parent = mp.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        os._exit(0)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()
