"""Watch a finished run again: what the Live tab showed, in order, on a timeline.

A run is watched once, while it happens, and an agent's run is worth watching
again - to show someone how it went, to check a decision, to take a figure for
a paper. :class:`LiveRecorder` records every call the Workflow page makes on
its Live views (timeline, route, canvas, banner) with the time it was made, and
saves them beside the run's results as ``live_replay.json``. :class:`ReplayPlayer`
makes the same calls again on the same views, and :class:`ReplayBar` is its
transport: play, pause, speed, a slider to scrub, and a button that saves the
Live tab as it looks at that moment.

A replay is the recording, not a re-run: nothing is computed again and no
model is asked anything. Long waits - an inversion that took minutes - are
shortened on the replay's own timeline, while the clock shown is the run's real
one, so a replay is quick to watch and still truthful about how long things
took. The glow round the studio stays off: it means the assistant is in control
now, and during a replay it is not.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
                               QSlider)

#: The file a run's recording is saved as, in its output folder.
FILE_NAME = "live_replay.json"
#: Longest pause between two recorded calls on the replay's timeline, seconds.
GAP_CAP_S = 1.2
FORMAT_VERSION = 1


def _jsonable(value: Any) -> Any:
    """``value`` as JSON can hold it; callables, which a replay cannot use, as None."""
    if callable(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover - numpy is a dependency
        pass
    return str(value)


class LiveRecorder:
    """Record calls on named objects' methods, with the time each was made.

    Parameters
    ----------
    targets : dict
        ``{name: (object, method_names)}``. Each method is wrapped on the
        object itself, so every caller is recorded, and a recorded method that
        calls another recorded one records only the outer call - a replay of the
        outer call makes the inner one again.

    Examples
    --------
    >>> class Board:
    ...     def __init__(self): self.lines = []
    ...     def write(self, text): self.lines.append(text); self.underline()
    ...     def underline(self): self.lines.append('--')
    >>> board = Board()
    >>> recorder = LiveRecorder({'board': (board, ('write', 'underline'))})
    >>> recorder.start(); board.write('hello'); recorder.stop()
    >>> [(e['target'], e['call'], e['args']) for e in recorder.events]
    [('board', 'write', ['hello'])]
    """

    def __init__(self, targets: Dict[str, Tuple[Any, Sequence[str]]]) -> None:
        self._targets = {key: (obj, tuple(names)) for key, (obj, names) in targets.items()}
        self.events: List[Dict[str, Any]] = []
        self.recording = False
        self._t0 = time.monotonic()
        self._depth = 0
        for key, (obj, names) in self._targets.items():
            for name in names:
                setattr(obj, name, self._wrap(key, name, getattr(obj, name)))

    def objects(self) -> Dict[str, Any]:
        return {key: obj for key, (obj, _names) in self._targets.items()}

    def _wrap(self, key: str, name: str, original: Callable) -> Callable:
        def recorded(*args, **kwargs):
            if self.recording and self._depth == 0:
                self.events.append({"t": round(time.monotonic() - self._t0, 3),
                                    "target": key, "call": name,
                                    "args": _jsonable(list(args)),
                                    "kwargs": _jsonable(kwargs)})
            self._depth += 1
            try:
                return original(*args, **kwargs)
            finally:
                self._depth -= 1
        recorded.__wrapped__ = original
        return recorded

    def start(self) -> None:
        self.events = []
        self._t0 = time.monotonic()
        self.recording = True

    def stop(self) -> None:
        self.recording = False

    def save(self, path: str, meta: Optional[Dict[str, Any]] = None) -> str:
        """Write the recording to ``path``; files under its folder are kept relative.

        A run folder moved or shared keeps its replay working: figure paths
        inside the folder are stored relative to it.
        """
        folder = os.path.dirname(os.path.abspath(path))

        def relative(value):
            if isinstance(value, str) and os.path.isabs(value):
                try:
                    inside = os.path.commonpath([folder, os.path.abspath(value)]) == folder
                except ValueError:
                    inside = False
                if inside:
                    return {"__run_file__": os.path.relpath(value, folder).replace(os.sep, "/")}
            if isinstance(value, list):
                return [relative(v) for v in value]
            if isinstance(value, dict):
                return {k: relative(v) for k, v in value.items()}
            return value

        document = {"format": "pyhydrogeophysx-live-replay", "version": FORMAT_VERSION,
                    "meta": _jsonable(meta or {}),
                    "events": [dict(e, args=relative(e["args"]), kwargs=relative(e["kwargs"]))
                               for e in self.events]}
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=0)
        return path


def load(path: str) -> Dict[str, Any]:
    """A saved recording, with its run's files resolved against where it now is.

    Raises
    ------
    ValueError
        When the file is not a live replay.
    """
    with open(path, encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or document.get("format") != "pyhydrogeophysx-live-replay":
        raise ValueError(f"{os.path.basename(path)} is not a run recording.")
    folder = os.path.dirname(os.path.abspath(path))

    def absolute(value):
        if isinstance(value, dict) and set(value) == {"__run_file__"}:
            return os.path.join(folder, *str(value["__run_file__"]).split("/"))
        if isinstance(value, list):
            return [absolute(v) for v in value]
        if isinstance(value, dict):
            return {k: absolute(v) for k, v in value.items()}
        return value

    document["events"] = [dict(e, args=absolute(e.get("args") or []),
                               kwargs=absolute(e.get("kwargs") or {}))
                          for e in document.get("events") or []]
    document["path"] = path
    return document


def replay_times(times: Sequence[float], cap: float = GAP_CAP_S) -> List[float]:
    """Each recorded time on the replay's timeline: no wait longer than ``cap``.

    >>> replay_times([0.0, 0.5, 300.0, 300.2])
    [0.0, 0.5, 1.7, 1.9]
    """
    out: List[float] = []
    last_real = last_replay = 0.0
    for index, real in enumerate(times):
        gap = 0.0 if index == 0 else max(0.0, real - last_real)
        last_replay = (last_replay if index else 0.0) + min(gap, cap)
        out.append(round(last_replay, 3))
        last_real = real
    return out


class ReplayPlayer(QObject):
    """Make a recording's calls again, on its timeline, at a chosen speed.

    Parameters
    ----------
    recording : dict
        As :func:`load` returns it.
    objects : dict
        ``{target name: object}`` the calls are made on - the same views that
        were recorded.
    substitute : callable, optional
        ``(target, call, args, kwargs) -> (args, kwargs)``, to stand in for what
        a recording cannot hold, such as a button's callback.

    Signals
    -------
    moved(float, float, float)
        The replay's position, its length (both on the replay's timeline), and
        the run's own clock at that position.
    ended()
        The last call has been made.
    """

    moved = Signal(float, float, float)
    ended = Signal()

    def __init__(self, recording: Dict[str, Any], objects: Dict[str, Any],
                 substitute: Optional[Callable] = None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._events = list(recording.get("events") or [])
        self._real = [float(e.get("t") or 0.0) for e in self._events]
        self._times = replay_times(self._real)
        self._objects = objects
        self._substitute = substitute
        self._next = 0
        self._position = 0.0
        self._speed = 1.0
        self._last_tick: Optional[float] = None
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)

    def length(self) -> float:
        return self._times[-1] if self._times else 0.0

    def playing(self) -> bool:
        return self._timer.isActive()

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.1, float(speed))

    def play(self) -> None:
        if self._next >= len(self._events):
            self.seek(0.0)
        self._last_tick = time.monotonic()
        self._timer.start()

    def pause(self) -> None:
        self._timer.stop()

    def seek(self, position: float) -> None:
        """Show the run as it was at ``position``: every call up to it, made at once."""
        self._next = 0
        self._position = max(0.0, min(self.length(), float(position)))
        self._apply_until(self._position)
        self._announce()

    def finish(self) -> None:
        """Make every call that is left, and stop: the run as it ended."""
        self.pause()
        self._position = self.length()
        self._apply_until(self._position)
        self._announce()

    def real_time(self, position: float) -> float:
        """The run's own clock at a point on the replay's timeline."""
        if not self._times:
            return 0.0
        for index in range(1, len(self._times)):
            if self._times[index] >= position:
                low, high = self._times[index - 1], self._times[index]
                share = 0.0 if high <= low else (position - low) / (high - low)
                return self._real[index - 1] + share * (self._real[index] - self._real[index - 1])
        return self._real[-1]

    def _tick(self) -> None:
        now = time.monotonic()
        step = (now - (self._last_tick or now)) * self._speed
        self._last_tick = now
        self._position = min(self.length(), self._position + step)
        self._apply_until(self._position)
        self._announce()
        if self._next >= len(self._events):
            self._timer.stop()
            self.ended.emit()

    def _apply_until(self, position: float) -> None:
        while self._next < len(self._events) and self._times[self._next] <= position + 1e-9:
            self._apply(self._events[self._next])
            self._next += 1

    def _apply(self, event: Dict[str, Any]) -> None:
        target = self._objects.get(str(event.get("target")))
        method = getattr(target, str(event.get("call")), None) if target is not None else None
        if method is None:
            return
        args, kwargs = list(event.get("args") or []), dict(event.get("kwargs") or {})
        if self._substitute is not None:
            args, kwargs = self._substitute(event.get("target"), event.get("call"), args, kwargs)
        try:
            method(*args, **kwargs)
        except Exception:  # noqa: BLE001 - one call that no longer applies must not end a replay
            pass

    def _announce(self) -> None:
        self.moved.emit(self._position, self.length(), self.real_time(self._position))


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class ReplayBar(QFrame):
    """The transport of a replay: play, speed, scrub, save a frame, close.

    Signals
    -------
    playToggled(bool), speedChanged(float), scrubbed(float), frameRequested(),
    closed()
    """

    playToggled = Signal(bool)
    speedChanged = Signal(float)
    scrubbed = Signal(float)
    frameRequested = Signal()
    closed = Signal()

    SPEEDS = (("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0), ("4×", 4.0), ("8×", 8.0))
    _SCALE = 1000.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("replayBar")
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 8, 16, 8)
        row.setSpacing(10)
        tag = QLabel("REPLAY")
        tag.setObjectName("agentGoalTag")
        row.addWidget(tag)
        self.play = QPushButton("Play")
        self.play.setProperty("primary", True)
        self.play.clicked.connect(lambda: self.playToggled.emit(self.play.text() == "Play"))
        row.addWidget(self.play)
        self.speed = QComboBox()
        for label, value in self.SPEEDS:
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(
            lambda _i: self.speedChanged.emit(float(self.speed.currentData())))
        row.addWidget(self.speed)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(lambda v: self.scrubbed.emit(v / self._SCALE))
        self.slider.sliderPressed.connect(lambda: self.scrubbed.emit(
            self.slider.value() / self._SCALE))
        row.addWidget(self.slider, 1)
        self.time = QLabel("0:00 / 0:00")
        self.time.setObjectName("agentClock")
        self.time.setToolTip("The run's own clock: how long it had been running at this point.")
        row.addWidget(self.time)
        frame = QPushButton("Save frame")
        frame.setToolTip("Save the Live tab as it looks now, as a PNG in the run's folder.")
        frame.clicked.connect(self.frameRequested.emit)
        row.addWidget(frame)
        close = QPushButton("Close replay")
        close.clicked.connect(self.closed.emit)
        row.addWidget(close)
        self._run_length = 0.0

    def speed_value(self) -> float:
        return float(self.speed.currentData())

    def set_playing(self, playing: bool) -> None:
        self.play.setText("Pause" if playing else "Play")

    def set_run_length(self, seconds: float) -> None:
        self._run_length = float(seconds or 0.0)

    def show_position(self, position: float, length: float, real: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setRange(0, int(length * self._SCALE))
        if not self.slider.isSliderDown():
            self.slider.setValue(int(position * self._SCALE))
        self.slider.blockSignals(False)
        self.time.setText(f"{_clock(real)} / {_clock(self._run_length)}")
