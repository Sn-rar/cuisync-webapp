"""Step-by-step debug trace for Cuisync (put next to app.py).

Every search builds one SearchTrace. When it finishes, the whole trace is
  - printed in the terminal where `python app.py` runs,
  - appended to search_debug.log (set LOG_FILE), and
  - kept in memory (RECENT) so the browser pages /debug/log and /debug/search can show it.
"""
import datetime
import time
from collections import deque

ENABLED = True
LOG_FILE = "search_debug.log"
RECENT = deque(maxlen=50)          # the last 50 finished traces (newest at the right)


def _emit(text):
    try:
        print(text, flush=True)
    except UnicodeEncodeError:     # some Windows terminals cannot print every character
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError:
        pass


def log_line(text):
    """One-line message (used for the request/response log)."""
    if ENABLED:
        _emit(f"{datetime.datetime.now():%H:%M:%S}  {text}")


class SearchTrace:
    def __init__(self, label):
        self.enabled = ENABLED
        self.label = label
        self.start = time.perf_counter()
        self.lines = []

    def step(self, title):
        if self.enabled:
            ms = (time.perf_counter() - self.start) * 1000
            self.lines.append("")
            self.lines.append(f"[{ms:7.1f} ms] {title}")

    def say(self, text):
        if self.enabled:
            self.lines.append("              " + str(text))

    def finish(self, outcome):
        """Close the trace, print it, save it. Returns the text."""
        if not self.enabled:
            return ""
        ms = (time.perf_counter() - self.start) * 1000
        bar = "=" * 78
        text = "\n".join(
            [bar, f"{self.label}   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", bar]
            + self.lines
            + ["", f"[{ms:7.1f} ms] OUTCOME: {outcome}", bar]
        )
        RECENT.append(text)
        _emit(text)
        return text