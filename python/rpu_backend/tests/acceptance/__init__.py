import os
import subprocess
COLD_IMPORT_TIMEOUT_S = 600

def timeout_message(label: str, exc: subprocess.TimeoutExpired) -> str:
    try:
        load = '%.2f %.2f %.2f' % os.getloadavg()
    except OSError:
        load = 'unavailable'
    return f'{label}: child process did not finish within {exc.timeout}s. That bound is a hang backstop (~55x the ~11 s a cold `import rpu_backend` costs), so this is a DEADLOCK, not a slow machine.\n1/5/15-min load average at failure: {load}\nstdout (partial):\n{exc.stdout}\nstderr (partial):\n{exc.stderr}'
