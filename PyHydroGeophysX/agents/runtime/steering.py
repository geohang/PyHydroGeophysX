"""What the user can tell a run while it works: hold it, let it go on, and notes.

A run in the studio happens in its own process, and the person watching it may
want to hold it before the next step, or tell it something - "use lambda 20",
"leave the seismic line out", "write the report now". This is the run's side of
that. The process sets a :class:`Steering` for the run (:data:`current`), fills
it from another thread as messages arrive, and the controller reads it between
steps (:func:`~.controller.run_controller`):

- **Pause** takes effect before the next step. The step that is running runs
  to the end: an inversion stopped half-way leaves nothing anyone can use.
- **A note** reaches the controller with its next decision. It is added to the
  transcript the model reads, and that decision may change the run's
  adjustable settings (:data:`~.recovery.ADJUSTABLE`) to follow it - never a
  file, never code. A note the run cannot follow is answered with why, and a
  run with no model to read notes says so rather than pretending to.

Nothing here knows about Qt; :attr:`Steering.notify` is how the run tells
whoever is listening that it paused, went on, or read a note, and
:attr:`Steering.source` is where it reads what it is told - for the desktop, a
:class:`ControlFile` in the run's folder.
"""

import json
import threading
from collections import deque
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional


class Steering:
    """Pause, resume and notes for one run, safe to fill from another thread.

    Parameters
    ----------
    notify : callable, optional
        Receives an event dict - ``phase`` ``"paused"``, ``"resumed"`` or
        ``"steer"`` - whenever the run acts on what it was told. Errors it
        raises are ignored: a display must not stop a run.
    source : callable, optional
        Returns the messages that have arrived since it was last called -
        ``{"steer": text}``, ``{"pause": true}``, ``{"resume": true}``,
        ``{"stop": true}``. Read at every checkpoint, and while paused.

    Examples
    --------
    >>> seen = []
    >>> steer = Steering(notify=seen.append)
    >>> steer.say('use lambda 20')
    >>> steer.take_notes(), steer.take_notes()
    (['use lambda 20'], [])
    >>> steer.checkpoint()          # not paused: carries straight on
    True
    >>> steer.stop()
    >>> steer.checkpoint()
    False
    """

    #: How often a paused run looks for a resume, in seconds.
    POLL_S = 0.25

    def __init__(self, notify: Optional[Callable[[Dict[str, Any]], None]] = None,
                 source: Optional[Callable[[], List[Dict[str, Any]]]] = None) -> None:
        self.notify = notify
        self.source = source
        self._lock = threading.Lock()
        self._notes: deque = deque()
        self._running = threading.Event()
        self._running.set()
        self._stopped = False

    # -- what the listening thread calls ---------------------------------------
    def say(self, text: str) -> None:
        """Queue a note for the controller's next decision."""
        text = str(text or "").strip()
        if text:
            with self._lock:
                self._notes.append(text)

    def pause(self) -> None:
        """Hold the run before its next step."""
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def stop(self) -> None:
        """End the run at its next checkpoint - the listener has gone."""
        self._stopped = True
        self._running.set()

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def receive(self, message: Dict[str, Any]) -> None:
        """Act on one message from :attr:`source`."""
        if not isinstance(message, dict):
            return
        if "steer" in message:
            self.say(str(message.get("steer") or ""))
        elif message.get("pause"):
            self.pause()
        elif message.get("resume"):
            self.resume()
        elif message.get("stop"):
            self.stop()

    def poll(self) -> None:
        """Take in whatever :attr:`source` has received since last time."""
        if self.source is None:
            return
        try:
            messages = self.source() or []
        except Exception:  # noqa: BLE001 - an unreadable channel must not stop a run
            return
        for message in messages:
            self.receive(message)

    # -- what the run calls ------------------------------------------------------
    def take_notes(self) -> List[str]:
        """The notes not yet read, oldest first; each is returned once."""
        self.poll()
        with self._lock:
            notes = list(self._notes)
            self._notes.clear()
        return notes

    def checkpoint(self, where: str = "") -> bool:
        """Wait here while paused. False when the run should stop instead."""
        self.poll()
        if self._stopped:
            return False
        if not self._running.is_set():
            self.tell({"phase": "paused", "label": where})
            while not self._running.wait(self.POLL_S):
                self.poll()
            if self._stopped:
                return False
            self.tell({"phase": "resumed", "label": where})
        return not self._stopped

    def tell(self, event: Dict[str, Any]) -> None:
        if self.notify is None:
            return
        try:
            self.notify(dict(event))
        except Exception:  # noqa: BLE001 - a display must not stop a run
            pass


class ControlFile:
    r"""Messages appended to a file, each read once, in order.

    How the desktop talks to a run in another process. A thread reading the
    run's stdin for them froze runs on Windows: while one thread is blocked
    reading a pipe, starting a process - which a step may do - waits for that
    read to return. The desktop appends one JSON object per line instead, and
    the run reads the new lines between steps; an incomplete last line waits
    for the next read.

    Examples
    --------
    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), 'steering.jsonl')
    >>> read = ControlFile(path)
    >>> read()
    []
    >>> with open(path, 'a', encoding='utf-8') as f:
    ...     _ = f.write('{"steer": "use lambda 20"}\n{"pause": tr')
    >>> read()
    [{'steer': 'use lambda 20'}]
    >>> with open(path, 'a', encoding='utf-8') as f:
    ...     _ = f.write('ue}\n')
    >>> read(), read()
    ([{'pause': True}], [])
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._offset = 0

    def __call__(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path, "rb") as stream:
                stream.seek(self._offset)
                data = stream.read()
        except OSError:
            return []
        end = data.rfind(b"\n")
        if end < 0:
            return []
        self._offset += end + 1
        messages = []
        for line in data[:end].splitlines():
            try:
                message = json.loads(line.decode("utf-8"))
            except ValueError:
                continue
            if isinstance(message, dict):
                messages.append(message)
        return messages


#: The steering of the run in progress in this context, or None - a script, a
#: test, or a run nobody is watching, which then runs as it always did.
current: ContextVar[Optional[Steering]] = ContextVar("steering", default=None)
