"""Run a command while logging memory every minute, so a runner lost to memory
pressure mid-suite leaves its last state in the job log.

Usage: python .github/scripts/with_memory_log.py pytest ...
"""

import subprocess
import sys
import threading
import time


def _log_memory(stop):
    while not stop.wait(60):
        print(f"::group::memory {time.strftime('%H:%M:%S', time.gmtime())}", flush=True)
        subprocess.run(["free", "-m"], check=False)
        top = subprocess.run(
            ["ps", "-eo", "rss,etime,args", "--sort=-rss"],
            capture_output=True,
            text=True,
            check=False,
        )
        print("\n".join(line[:160] for line in top.stdout.splitlines()[:8]))
        print("::endgroup::", flush=True)


stop = threading.Event()
threading.Thread(target=_log_memory, args=(stop,), daemon=True).start()
code = subprocess.call(sys.argv[1:])
stop.set()
# A child terminated by a signal has a negative return code in subprocess.
sys.exit(128 - code if code < 0 else code)
