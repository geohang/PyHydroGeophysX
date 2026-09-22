"""Reading a chi-squared history out of an inversion result.

Two inversion paths report their misfit in different shapes, and the difference
is easy to miss because both are "a list of numbers you can index".

A single-dataset inversion reports a flat per-iteration history::

    [2461.28, 14.21, 5.52, ..., 1.63]

The time-lapse solver reports one *row per iteration*, each row holding the
three terms of the objective function (see ``TimeLapseERTInversion`` in
:mod:`PyHydroGeophysX.inversion.time_lapse`, which appends
``[chi2_ert, fmert, ftert]``)::

    [[2461.28, 0.0, 0.0], [14.21, 9638.77, 1.73], ..., [1.63, 3476.04, 1.43]]

Column 0 is the data misfit; columns 1 and 2 are the model- and
temporal-regularization terms, which are not chi-squared and are not on its
scale. Callers that read a row as if it were one dataset's history took the
wrong column - ``row[-1]`` scores the temporal regularization term as the data
fit, and ``rows[0]`` treats the first iteration's three terms as a convergence
history. Neither raises: both produce a plausible-looking number, so the error
surfaces only as a quality score that does not match the inversion.

:func:`chi2_history` accepts either shape and always returns the chi-squared
column, so callers can format it, take its range, or read its last value
without knowing which solver produced it.
"""

from typing import Any, List

import numpy as np


def chi2_history(values: Any) -> List[float]:
    """Per-iteration chi-squared from ``chi2_values`` / ``all_chi2``.

    Parameters
    ----------
    values : sequence
        Either a flat sequence of chi-squared values, or a sequence of
        ``[chi2, phi_m, phi_t]`` rows as the time-lapse solver produces. May be
        None or empty.

    Returns
    -------
    list of float
        The chi-squared value for each iteration, in order, with non-finite
        entries dropped. Empty when ``values`` holds nothing usable, so the
        result is always safe to format, to take ``min``/``max`` of, or to index
        from the end after a truth check.

    Raises
    ------
    None

    Examples
    --------
    >>> chi2_history([5.0, 2.0, 1.6])
    [5.0, 2.0, 1.6]
    >>> chi2_history([[2461.3, 0.0, 0.0], [1.63, 3476.0, 1.43]])
    [2461.3, 1.63]
    >>> chi2_history(None)
    []
    """
    if values is None:
        return []
    history: List[float] = []
    for entry in values:
        row = np.atleast_1d(np.asarray(entry, dtype=float)).ravel()
        if row.size == 0:
            continue
        # Column 0 in both shapes: a flat history's own value, or the data
        # misfit of one time-lapse iteration.
        candidate = float(row[0])
        if np.isfinite(candidate):
            history.append(candidate)
    return history


def chi2_summary(values: Any) -> str:
    """One line describing a chi-squared trajectory, or ``"N/A"``.

    Reports where the inversion started and where it ended rather than a bare
    min-max: the largest value in a converging run is its *first* iteration, so
    a range reads as disagreement between datasets when it is really the
    descent.

    Parameters
    ----------
    values : sequence
        As accepted by :func:`chi2_history`.

    Returns
    -------
    str
        For example ``"1.631 (from 2461.282 over 9 iterations)"``, or just the
        value when there is a single iteration, or ``"N/A"`` when unavailable.

    Raises
    ------
    None

    Examples
    --------
    >>> chi2_summary([[2461.3, 0.0, 0.0], [1.63, 3476.0, 1.43]])
    '1.630 (from 2461.300 over 2 iterations)'
    >>> chi2_summary([])
    'N/A'
    """
    history = chi2_history(values)
    if not history:
        return "N/A"
    if len(history) == 1:
        return f"{history[0]:.3f}"
    return (f"{history[-1]:.3f} (from {history[0]:.3f} "
            f"over {len(history)} iterations)")
