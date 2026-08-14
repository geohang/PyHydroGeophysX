"""Colour scales that read in physical units while the data is held in log10.

Colouring by ``log10(value)`` is what makes a resistivity image readable, but
the reader wants ohm-metres on the scale beside it rather than exponents.

pyqtgraph's own answer looks like ``AxisItem.setLogMode(True)``, and it is the
wrong tool here. That switch is for an axis whose *view* is logarithmic: it
replaces the tick placement with a decade rule. Pointed at an axis whose values
are already log10, it still places ticks on whole decades, so a survey spanning
less than one decade (61 to 235 ohm-m, say) gets a colour bar carrying a single
number, and one spanning a little more gets sub-decade ticks crowded together
until they overlap the axis label. Either way the reader cannot map a colour
back to a resistivity, which is the whole purpose of the bar.

Formatting the tick strings leaves the placement alone. The ordinary linear
algorithm still picks four or five evenly spaced ticks across whatever range the
data covers, and each is printed as the value it stands for.
"""

from __future__ import annotations

import math
from typing import Sequence

#: Largest log10 tick this can raise 10 to. Past ``10**308`` a float overflows,
#: and the exponent is printed instead of the value.
_MAX_EXPONENT = 300.0


def _physical(value: float) -> str:
    """A log10 tick position as the value it represents.

    Total by construction. This runs inside ``AxisItem.tickStrings``, which Qt
    calls from the item's paint; an exception raised there unwinds through a C++
    virtual call and takes the process down with no traceback, so every input
    has to produce a string. ``10.0 ** 400`` raises ``OverflowError`` on its
    own, which is reason enough not to trust the range to be sane.
    """
    try:
        exponent = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(exponent):
        return ""
    if exponent > _MAX_EXPONENT or exponent < -_MAX_EXPONENT:
        return f"1e{exponent:+.0f}"
    magnitude = 10.0 ** exponent
    if magnitude >= 1e5 or (magnitude and magnitude < 0.01):
        return f"{magnitude:.3g}"
    if magnitude >= 1000.0:
        # Whole ohm-metres past a thousand: "1000", not "1e+03".
        return f"{magnitude:.0f}"
    # Enough digits to separate neighbouring ticks, no more: a bar reading
    # 63.1 / 79.4 / 100 is easier than one reading 63.0957 / 79.4328 / 100.
    return f"{magnitude:.4g}"


def label_axis_in_physical_units(axis, enabled: bool = True) -> None:
    """Print ``10**tick`` on ``axis``, or restore its ordinary labels.

    The override lives on the instance, so it shadows ``AxisItem.tickStrings``
    for this one axis and disappears when it is removed. Log mode is turned off
    at the same time: it and this formatter would each try to convert, and the
    labels would come out squared.
    """
    axis.setLogMode(False)
    if not enabled:
        axis.__dict__.pop("tickStrings", None)
        axis.picture = None
        axis.update()
        return

    def tick_strings(values: Sequence[float], _scale: float, _spacing: float):
        try:
            return [_physical(value) for value in values]
        except Exception:  # noqa: BLE001 - see _physical: this runs inside paint
            return ["" for _ in values]

    axis.tickStrings = tick_strings
    axis.picture = None
    axis.update()
