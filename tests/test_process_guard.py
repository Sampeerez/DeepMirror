"""No process outlives the studio: children exit with their parent, however the parent ends."""

from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
import textwrap
import time

import pytest

from backend.process_guard import exit_with_parent


def _child(conn) -> None:
    exit_with_parent()
    conn.send("up")
    time.sleep(60)


def _parent(conn) -> None:
    context = mp.get_context("spawn")
    child = context.Process(target=_child, args=(conn,), daemon=False)
    child.start()
    conn.send(child.pid)
    time.sleep(60)


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_a_child_exits_when_its_parent_is_killed():
    context = mp.get_context("spawn")
    ours, theirs = context.Pipe()
    parent = context.Process(target=_parent, args=(theirs,))
    parent.start()
    child_pid = ours.recv()
    assert ours.poll(60) and ours.recv() == "up"
    parent.kill()
    parent.join(10)
    deadline = time.monotonic() + 15
    while _alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(child_pid)


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows only")
def test_on_windows_the_job_takes_every_child_with_the_server():
    script = textwrap.dedent("""
        import subprocess, sys, time
        from backend.process_guard import contain_children
        assert contain_children()
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        print(child.pid, flush=True)
        time.sleep(60)
    """)
    server = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert server.stdout is not None
    child_pid = int(server.stdout.readline())
    assert _alive(child_pid)
    server.kill()   # as if the console window were closed
    server.wait(10)
    deadline = time.monotonic() + 10
    while _alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(child_pid)
