"""Backend-neutral helpers for inversion result summaries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def metrics_from_manager(
    mgr: Any,
    *,
    n_data: Optional[int] = None,
    lam: Optional[float] = None,
    method: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[float]]:
    """Extract common metrics from a PyGIMLi-style manager."""
    metrics: Dict[str, Any] = {}
    if n_data is not None:
        metrics["n_data"] = int(n_data)
    if lam is not None:
        metrics["lambda"] = float(lam)
    if method:
        metrics["method"] = str(method)
    history: List[float] = []
    inv = getattr(mgr, "inv", None)
    if inv is not None:
        try:
            metrics["chi2"] = float(inv.chi2())
        except Exception:  # noqa: BLE001 - backend/version compatibility
            pass
        try:
            metrics["rrms"] = float(inv.relrms())
        except Exception:  # noqa: BLE001 - backend/version compatibility
            pass
        for attr in ("chi2History", "chi2_history"):
            values = getattr(inv, attr, None)
            if values is not None:
                try:
                    history = [float(value) for value in values]
                    break
                except Exception:  # noqa: BLE001
                    history = []
    if history:
        metrics["iterations"] = len(history)
        chi2 = metrics.get("chi2")
        if chi2 is None or chi2 != chi2:
            metrics["chi2"] = history[-1]
    return metrics, history


__all__ = ["metrics_from_manager"]
