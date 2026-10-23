---
name: seismic-first-break-review
description: >
  Use when picking or reviewing seismic first breaks (SEG-Y, SEG-2) or before
  an SRT inversion. Gives the studio's per-shot procedure with its mandatory
  human review, the checks on each shot's picks and why, the automatic
  screening, and how the velocity model is judged.
methods: [seismic]
---

# Seismic first-break picking and review

The picks are the data. A pick on noise, a shot whose trigger fired early or
late, or a wrong shot position is spread through the velocity model, and the
fit cannot reveal it. So picking is per shot, and a person reviews every shot
before any inversion.

**Studio procedure - follow it exactly:**

1. `load_data`; then `load_geometry` if the user gives a geophone positions or
   topography file.
2. `set_geometry`: for a regular shot interval pass `first_shot_x` and
   `shot_spacing` ONCE (each record's shot x then fills itself); use `shot_x`
   only to override one irregular shot.
3. `pick_next_shot`: picks the next un-picked record and pauses
   (`awaiting_user`, `records_remaining`, `next_record`). Show the shot and its
   `suspect_traces`, then wait. The user may correct picks by hand, or ask you
   to `set_pick` or `delete_pick` a trace.
4. After the user says continue: if `records_remaining` is not empty, call
   `pick_next_shot` again; repeat until none remain.
5. `run_srt` ONLY when `records_remaining` is empty: never while shots remain,
   never in the same turn as `auto_pick`. It needs at least 8 picks.

Use `select_record` + `auto_pick` + `review_picks` only to re-pick one shot by
hand. Do not use `auto_pick_all` in place of this loop unless the user asks,
and still review every shot before `run_srt`.

**Per-shot checks.** `shot_x` matches the field notes. `suspect_traces` flags
picks more than 3 robust sigma from a straight time-offset line; a refraction
curve bends, so a flag means look, not wrong. No pick comes earlier farther
from the shot. Look at picks on noise and near `pick_max_time_ms` (150). Good:
picks rise smoothly with offset on both sides of the shot. Bad: a whole shot
shifted in time against its neighbours.

**Screening** (on by default; pass on its warnings with counts). A shot whose
median reciprocal difference exceeds 3 x the survey's typical one and at least
2 ms is dropped; shots with no reciprocal pair are kept unchecked. Picks out
of line with the neighbouring shots at the same geophone are picked again or
left out.

**Automatic run** (`pick_first_breaks`): picks every shot and screens them, with
no per-shot pause. Say the picks were not reviewed shot by shot, show the picks
and gathers figures, and offer the studio review before the model is relied
on. If the coordinate file and the SEG-Y headers disagree on the line's origin,
the choice (`profile`, `segy` or stop) is the user's.

**Evaluation** (`evaluate_seismic_inversion`, threshold 70). chi2 target 1,
0.8-1.5 accepted. More than 5 % of picks beyond three errors: check those
traces and shot positions. Rays covering under 40 %: report the model only
where covered. More than 2 % of covered cells at a velocity bound: widen the
limits or check the picks there. chi2 below 1 may only mean overstated errors
(pyGIMLi assumes 3 %). A retry halves or doubles `lam`, at most 3 inversions.

Details (actions, picker settings, screening, inversion defaults, scoring):
references/picking-and-srt.md

## Sources

Under `PyHydroGeophysX/`: `agents/assistants/aquah/__init__.py` (`PERSONA`),
`qt_apps/modules/seismic_processing.py`, `agents/seismic_agent.py`,
`data_processing/seismic.py`, `agents/runtime/catalog.py`,
`agents/_method_evaluation.py`. Docs: `docs/source/methods/seismic.rst`.
