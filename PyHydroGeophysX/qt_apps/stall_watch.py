"""Report where the UI thread stalls, with the stack that was running.

"The window freezes sometimes" is not actionable: the compute in this workbench
already runs on worker threads, so a freeze means something else is holding the
thread that paints, and only a stack taken *during* the stall says what.

A ``QTimer`` on the main thread records a heartbeat. A daemon thread watches the
heartbeat and, when it goes quiet for longer than the threshold, samples the main
thread's frames through :func:`sys._current_frames` and writes them out. Sampling
from another thread needs no cooperation from the stalled one, which is the whole
point: a stalled thread cannot report on itself.

Off unless asked for. Set the threshold in milliseconds::

    PHGX_STALL_WATCH_MS=300

Findings go to stderr immediately and to the workbench log once the UI thread is
free again. ``PHGX_STALL_WATCH_FILE`` also appends them to a file, which survives
a hard kill when the freeze never ends.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from typing import List, Optional

from PySide6.QtCore import QObject, QTimer, Signal

#: How often the main thread records a heartbeat, and how often it is checked.
BEAT_INTERVAL_MS = 50

#: Frames nearest the top of the stack to report. Deeper frames are the Qt event
#: loop and say nothing about the cause.
STACK_DEPTH = 14


def _threshold_ms() -> Optional[int]:
    raw = (os.getenv("PHGX_STALL_WATCH_MS") or "").strip()
    if not raw:
        return None
    try:
        value = int(float(raw))
    except ValueError:
        return None
    return value if value >= BEAT_INTERVAL_MS * 2 else None


class StallWatcher(QObject):
    """Sample the main thread whenever its event loop goes quiet."""

    #: Emitted (queued) once the UI thread is responsive again.
    stalled = Signal(str)

    def __init__(self, threshold_ms: int, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._threshold = threshold_ms / 1000.0
        self._beat = time.monotonic()
        self._main_thread_id = threading.get_ident()
        self._stop = threading.Event()
        self._path = (os.getenv("PHGX_STALL_WATCH_FILE") or "").strip() or None

        self._timer = QTimer(self)
        self._timer.setInterval(BEAT_INTERVAL_MS)
        self._timer.timeout.connect(self._on_beat)
        self._timer.start()

        self._thread = threading.Thread(target=self._watch, name="phgx-stall-watch",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._timer.stop()

    # -- main thread ---------------------------------------------------------
    def _on_beat(self) -> None:
        self._beat = time.monotonic()

    # -- watcher thread ------------------------------------------------------
    def _watch(self) -> None:
        reported_at = 0.0
        while not self._stop.wait(BEAT_INTERVAL_MS / 1000.0):
            beat = self._beat
            quiet = time.monotonic() - beat
            if quiet < self._threshold or beat == reported_at:
                continue
            reported_at = beat
            stack = self._sample_main_thread()
            # Wait out the stall so the report can state how long it lasted.
            while not self._stop.is_set() and self._beat == beat:
                time.sleep(BEAT_INTERVAL_MS / 1000.0)
            total = (self._beat - beat) if self._beat != beat else quiet
            self._report(total, stack)

    def _sample_main_thread(self) -> List[str]:
        frame = sys._current_frames().get(self._main_thread_id)
        if frame is None:
            return ["  (main thread frame unavailable)"]
        lines = traceback.format_stack(frame)
        return [line.rstrip() for line in lines[-STACK_DEPTH:]]

    def _report(self, seconds: float, stack: List[str]) -> None:
        head = f"UI thread stalled for {seconds * 1000:.0f} ms. Main-thread stack:"
        body = "\n".join(stack)
        text = f"{head}\n{body}"
        try:
            print(f"[stall-watch] {text}", file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001 - diagnostics must never raise
            pass
        if self._path:
            try:
                with open(self._path, "a", encoding="utf-8") as handle:
                    handle.write(f"\n=== {time.strftime('%H:%M:%S')} {text}\n")
            except Exception:  # noqa: BLE001
                pass
        # Queued across the thread boundary, so it lands after the UI recovers.
        self.stalled.emit(f"{head} see stderr for the stack "
                          f"({stack[-1].strip() if stack else 'no frames'})")


def install(parent: Optional[QObject] = None) -> Optional[StallWatcher]:
    """Start watching if ``PHGX_STALL_WATCH_MS`` asks for it, else do nothing."""
    threshold = _threshold_ms()
    if threshold is None:
        return None
    return StallWatcher(threshold, parent)
