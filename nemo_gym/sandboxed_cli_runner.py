# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone sandbox subprocess runner. No Gym or harness Python imports."""

import ctypes
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep


def enable_subreaper() -> None:
    """Adopt detached grandchildren so cleanup is not limited to a process group."""
    if sys.platform != "linux":
        raise RuntimeError("Sandboxed CLI execution requires Linux child-subreaper support")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot establish the CLI cleanup boundary")


def drain_children(timeout: float) -> None:
    """Kill and reap this runner's descendants, including double-forked tools."""
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    deadline = monotonic() + timeout
    while True:
        for child in children.read_text().split():
            try:
                os.kill(int(child), signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Killing a parent reparents its children to this subreaper. Keep
        # draining until waitpid confirms no children, not just an exited CLI.
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return
        if monotonic() >= deadline:
            raise TimeoutError("CLI descendants remain alive; do not verify")
        sleep(0.01)


def run(params: dict) -> dict:
    """Run a CLI under a Linux subreaper and drain descendants before completion."""
    directory = Path(params["directory"])
    timed_out = False
    error = None
    process = None
    cleanup_confirmed = True

    def interrupt(*_):
        raise InterruptedError("runner interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        try:
            enable_subreaper()
            process = subprocess.Popen(
                params["command"],
                cwd=params["cwd"],
                env=dict(os.environ) | params["env"],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            process.wait(timeout=params["timeout"])
        except (subprocess.TimeoutExpired, InterruptedError):
            timed_out = True
        except Exception as exc:
            error = str(exc)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if process is not None:
                try:
                    # Always signal the group, even if its leader already exited:
                    # terminal tools can leave children holding files open.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=params["cleanup_timeout"])
                    drain_children(params["cleanup_timeout"])
                except Exception as exc:
                    error = f"cleanup: {exc}"
                    cleanup_confirmed = False
    return {
        "return_code": process.returncode if process else 1,
        "timed_out": timed_out,
        "cleanup_confirmed": cleanup_confirmed,
        "error": error,
    }


def main() -> None:
    params = json.loads(Path(sys.argv[1]).read_text())
    result = run(params)
    output = Path(params["directory"]) / "result.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result))
    temporary.replace(output)


if __name__ == "__main__":
    main()
