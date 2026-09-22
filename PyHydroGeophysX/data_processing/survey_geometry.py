"""Strict profile geometry for automatic raw-seismic processing."""
from dataclasses import replace
from pathlib import Path
import numpy as np


class OriginMismatch(ValueError):
    """The coordinate file and the SEG-Y headers do not share an origin.

    Its own type rather than a plain ``ValueError`` because a caller can do
    something about this one: the two frames differ by a rigid translation, so
    the survey can still be processed once somebody says which origin to
    report. Everything else that goes wrong in here is a malformed file.

    Attributes
    ----------
    shift : float
        What to add to a SEG-Y x coordinate to put it in the coordinate file's
        frame. The same number, negated, moves the file into the SEG-Y frame.
    """

    def __init__(self, message, shift=0.0):
        super().__init__(message)
        self.shift = float(shift)


def read_coordinates(path, columns):
    """Numeric geometry rows from ``path``, skipping one header line if present.

    A header is recognised by where it is and what it is - the first row, with
    the right number of non-numeric fields - rather than by its wording. The
    previous version accepted exactly two spellings, ``x z`` and
    ``receiver_id x z``, so the geometry file shipped in ``examples/data``
    (``station distance_m elevation_m``) was rejected at line 1 and the SEG-Y
    processing path could not run against the data beside it.

    Every later non-numeric row is still an error: one header is a convention,
    two is a malformed file.
    """
    rows = []
    header_skipped = False
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.replace(',', ' ').split()
        try:
            values = [float(v) for v in parts]
        except ValueError:
            if not rows and not header_skipped and len(parts) == columns:
                header_skipped = True
                continue
            raise ValueError(f'{path}:{number}: invalid geometry row; expected {columns} numeric columns')
        if len(values) != columns or not np.all(np.isfinite(values)):
            raise ValueError(f'{path}:{number}: expected {columns} finite numeric columns')
        rows.append(values)
    if not rows:
        raise ValueError(f'No coordinates in {path}')
    return np.asarray(rows)


def origin_shift(picks, profile):
    """What to add to a SEG-Y x coordinate to put it in ``profile``'s frame.

    Compares like with like: the receiver positions the SEG-Y headers carry
    against the x range the coordinate file covers. A whole-survey translation
    is the only discrepancy this can describe, which is the point - a scale or
    unit error is not something to paper over.

    Returns ``0.0`` when the headers carry no usable receiver x, in which case
    there is nothing to compare and the caller should not be aligning at all.
    """
    header_x = [float(p.receiver_x) for p in picks if np.isfinite(p.receiver_x)]
    if not header_x or profile is None or not len(profile):
        return 0.0
    return float(profile[0, 0] - min(header_x))


def _station_spacing(positions, picks):
    """Typical distance between neighbouring receivers, or ``0.0``.

    Zero when there is only one receiver and nothing to measure, which leaves no
    tolerance for off-end shots - the honest answer when the geometry gives no
    scale to judge them against.
    """
    if positions:
        xs = sorted(x for x, _ in positions.values())
    else:
        xs = sorted({float(p.receiver_x) for p in picks
                     if np.isfinite(p.receiver_x)})
    return float(np.median(np.diff(xs))) if len(xs) > 1 else 0.0


def apply_pick_geometry(picks, geophone_file=None, topography_file=None,
                        align_origin=None, warn=None):
    """Map explicit receiver IDs and interpolate elevations along an x/z profile.

    No index/CRS guessing. Receivers must lie on the profile; a shot may sit up
    to one station spacing off either end, which is ordinary off-end acquisition
    rather than a geometry error.

    Parameters
    ----------
    align_origin : {None, 'profile', 'segy'}
        What to do when the coordinate file and the SEG-Y headers disagree about
        where the line starts. ``None`` - the default, and the only behaviour
        before this existed - refuses and raises :class:`OriginMismatch`, because
        which origin is right is a fact about the survey rather than something
        to infer. The other two say whose origin to report, having been told:
        ``'profile'`` moves the shots into the coordinate file's frame, and
        ``'segy'`` moves the coordinate file into the SEG-Y frame. Both are the
        same rigid translation, so the recovered velocity model is identical
        either way; only the x labels differ, and they differ in the way that
        matters when the section is later laid beside an ERT line.
    warn : callable, optional
        Receives a sentence for anything the caller should pass on but that does
        not stop the work - at present, shots taken off the end of the spread.
    """
    positions = {}
    if geophone_file:
        values = read_coordinates(geophone_file, 3)
        for receiver, x, z in values:
            if receiver != int(receiver) or int(receiver) in positions:
                raise ValueError('Geophone receiver_id must contain unique integer SEG-Y receiver IDs.')
            positions[int(receiver)] = (x, z)
    profile = read_coordinates(topography_file, 2) if topography_file else None
    # Whether the profile *is* the receiver spread, as opposed to an independent
    # terrain line that may legitimately run past both ends of it. Only in the
    # first case can the file's x be compared directly with the headers'.
    spread_profile = profile is None and bool(positions)
    if spread_profile:
        profile = np.asarray(list(positions.values()))
    if profile is not None:
        profile = profile[np.argsort(profile[:, 0])]
        if len(profile) < 2 or np.any(np.diff(profile[:, 0]) <= 0):
            raise ValueError('Elevation profile requires at least two unique increasing x coordinates.')
    # How far off the end of the spread a shot may sit: one station spacing.
    # Taken from the receiver positions, not from the profile's own sampling -
    # an independent topography line may be described by two points 10 m apart
    # while the geophones are at 2 m, and that coarse sampling must not become
    # a licence to extrapolate five times as far.
    spacing = _station_spacing(positions, picks)
    # Applied to SEG-Y x coordinates; 'segy' expresses the same translation by
    # moving the file's frame instead, so the two choices differ only in which
    # origin the exported coordinates carry.
    shift = 0.0
    if profile is not None and spread_profile:
        offset = origin_shift(picks, profile)
        if align_origin in ('profile', 'segy'):
            shift = offset
            if align_origin == 'segy':
                profile = profile - np.array([shift, 0.0])
                positions = {r: (x - shift, z) for r, (x, z) in positions.items()}
                shift = 0.0
        elif abs(offset) > 1e-6:
            # Checked up front, against the whole survey, rather than waiting
            # for some individual pick to fall off the end. AP_411.sgy numbers
            # its first station x=0 while location.txt numbers the same station
            # x=2, so every receiver is one station spacing out - a systematic
            # fact about the two files that a per-pick bounds failure describes
            # only by accident, and that a tolerance for off-end shots would
            # hide entirely. Which origin is right is a fact about the survey,
            # so the caller is handed the offset and can put the choice to
            # somebody who knows it.
            raise OriginMismatch(
                f'The coordinate file and the SEG-Y headers do not share an '
                f'origin: the file places the line at x={profile[0, 0]:g} to '
                f'{profile[-1, 0]:g}, the headers place the same receivers '
                f'{abs(offset):g} m away (source x={min(p.source_x for p in picks):g} '
                f'to {max(p.source_x for p in picks):g}). Say which origin to '
                f'report, or correct one of the files; nothing is guessed here.',
                shift=offset)
    result = []
    off_end = 0.0
    for pick in picks:
        if positions and pick.receiver_id not in positions:
            raise ValueError(f'No coordinate for SEG-Y receiver_id {pick.receiver_id}.')
        x, z = positions.get(pick.receiver_id,
                             (pick.receiver_x + shift, pick.receiver_z))
        source_x = pick.source_x + shift
        if profile is not None:
            low, high = profile[0, 0], profile[-1, 0]
            if not np.isfinite([x, source_x]).all() or x < low or x > high:
                # A receiver with no elevation is a real geometry error: it sits
                # on the spread the profile describes, so being off it means the
                # two do not describe the same line.
                raise OriginMismatch(
                    f'Receiver x lies outside the elevation profile. '
                    f'Profile covers x={low:g} to {high:g}; '
                    f'this pick has receiver x={x:g} and source x={source_x:g}. '
                    f'Check that the geometry file and the SEG-Y headers share '
                    f'an origin and units; extrapolation is disabled.',
                    shift=origin_shift(picks, profile))
            overshoot = max(low - source_x, source_x - high, 0.0)
            if overshoot > spacing + 1e-9:
                raise OriginMismatch(
                    f'Source x={source_x:g} lies {overshoot:g} m outside the '
                    f'elevation profile (x={low:g} to {high:g}), which is more '
                    f'than the {spacing:g} m station spacing. Check that the '
                    f'geometry file and the SEG-Y headers share an origin and '
                    f'units; extrapolation is disabled.',
                    shift=origin_shift(picks, profile))
            off_end = max(off_end, overshoot)
            # np.interp holds the end value outside the range, so an off-end
            # shot takes the elevation of the station it sits beyond - which is
            # what a shot a spacing past the last geophone is standing on.
            source_z = float(np.interp(source_x, profile[:, 0], profile[:, 1]))
            if topography_file:
                z = float(np.interp(x, profile[:, 0], profile[:, 1]))
        else:
            source_z = pick.source_z
        # ``source_x`` is written back, not just used for the bounds check: with
        # no alignment asked for the shift is zero and this is the SEG-Y value
        # unchanged, and with one it is the whole point.
        result.append(replace(pick, receiver_x=float(x), receiver_z=float(z),
                              source_x=float(source_x), source_z=source_z))
    if off_end > 0 and callable(warn):
        warn(f'{off_end:g} m of the shot positions lie beyond the ends of the '
             f'receiver spread, which is ordinary off-end acquisition. Those '
             f'shots take the elevation of the station they sit beyond, so '
             f'their depth is known to about the relief over one station '
             f'spacing.')
    return result
