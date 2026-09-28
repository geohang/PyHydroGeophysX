"""The manager-shaped view of an inverted model, without the solvers behind it.

Needs nothing but numpy, so a page can rebuild a result a workflow process sent
back (``workflows.objects``) without loading the inversion code that made it;
``ert_inversion`` and ``srt_inversion``, where these were written, export them.
"""

import numpy as np


class ModelResult:
    """Manager-shaped view of an inverted model, so both engines feed one viewer.

    ``MeshResultView`` and the VTK export ask for ``paraDomain``, ``model`` and
    an optional ``coverage()``; the in-house engine returns arrays rather than a
    PyGIMLi manager, so this wraps them in the same shape.

    ``velocity`` is set only by travel time, where a PyGIMLi manager exposes it
    under that name. It stays ``None`` for ERT, because a ``velocity`` attribute
    that quietly returned resistivity would be worse than a missing one.
    """

    def __init__(self, mesh, model, response=None, coverage=None, velocity=None):
        self.paraDomain = mesh
        self.model = np.asarray(model, dtype=float)
        self.response = None if response is None else np.asarray(response, dtype=float)
        self._coverage = None if coverage is None else np.asarray(coverage, dtype=float)
        self.velocity = None if velocity is None else np.asarray(velocity, dtype=float)

    def coverage(self):
        if self._coverage is None:
            raise AttributeError("no coverage available")
        return self._coverage


class RayPathModelResult(ModelResult):
    """A travel-time result that can also hand back its ray paths.

    Only built when paths were captured, so ``getRayPaths`` being present is a
    reliable signal that the overlay has something to draw; a version that
    always existed and sometimes returned nothing would show a control that
    does nothing.
    """

    def __init__(self, *args, ray_paths, **kwargs):
        super().__init__(*args, **kwargs)
        self._ray_paths = ray_paths

    def getRayPaths(self, model=None):  # noqa: N802 - matches the PyGIMLi name
        return self._ray_paths


__all__ = ["ModelResult", "RayPathModelResult"]
