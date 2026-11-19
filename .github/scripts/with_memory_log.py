"""Run a command while logging memory every minute, so a runner lost to memory
pressure mid-suite leaves its last state in the job log.

Usage: python .github/scripts/with_memory_log.py pytest ...
"""

import signal
import subprocess
import sys
import threading
import time


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def _log_memory(stop):
    while not stop.wait(60):
        free = _run(["free", "-m"]).rstrip("\n")
        top = _run(["ps", "-eo", "rss,etime,args", "--sort=-rss"])
        # The first line labels the columns. Keep it and the eight largest processes.
        top = "\n".join(line[:160] for line in top.splitlines()[:9])
        stamp = time.strftime("%H:%M:%S", time.gmtime())
        # pytest -vv leaves its current line open while a test runs, and GitHub
        # only reads ::group:: at the start of a line: open with a newline, and
        # write the block in one call so worker output cannot land inside it.
        sys.stdout.write(
            f"\n::group::memory {stamp}\n{free}\n{top}\n::endgroup::\n"
        )
        sys.stdout.flush()


stop = threading.Event()
threading.Thread(target=_log_memory, args=(stop,), daemon=True).start()
child = subprocess.Popen(sys.argv[1:])
# Pass cancel and timeout signals on to pytest so it can print its interrupted
# summary; left alone, SIGINT kills it outright and SIGTERM orphans it.
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda signum, frame: child.send_signal(signum))
code = child.wait()
stop.set()
# A child terminated by a signal has a negative return code in subprocess.
sys.exit(128 - code if code < 0 else code)
