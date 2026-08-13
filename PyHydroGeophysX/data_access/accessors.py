"""
Accessor abstraction for hydro-model data.

Three concrete implementations:
- LocalHydroAccessor  – reads directly from a local directory.
- HttpHydroAccessor   – downloads files on demand from an HTTP base URL.
- (BaseHydroAccessor) – abstract base defining the interface.

Usage
-----
>>> acc = LocalHydroAccessor("/path/to/data")
>>> ok, summary, errors = acc.validate()
>>> local_dir = acc.materialize(["Watercontent.npy", "top.txt"], "/tmp/work")
"""

from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Required files for the hydro-to-geophysics workflow
REQUIRED_FILES = ["Watercontent.npy", "Porosity.npy", "top.txt", "bot.npy"]


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the dataset manifest JSON.

    Parameters
    ----------
    manifest_path : str, optional
        Path to manifest.json.  Defaults to ``datasets/manifest.json``
        relative to the repository root (two levels up from this file).

    Returns
    -------
    dict
        Parsed manifest with a ``datasets`` key.
    """
    if manifest_path is None:
        # Resolve relative to repo root
        here = Path(__file__).resolve().parent
        repo_root = here.parent.parent  # data_access -> PyHydroGeophysX -> repo
        manifest_path = str(repo_root / "datasets" / "manifest.json")

    path = Path(manifest_path)
    if not path.exists():
        return {"version": "1.0", "datasets": []}

    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def get_manifest_entry(dataset_id: str, manifest_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return a single manifest entry by id, or None."""
    manifest = load_manifest(manifest_path)
    for entry in manifest.get("datasets", []):
        if entry.get("id") == dataset_id:
            return entry
    return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseHydroAccessor(ABC):
    """Abstract base for hydro-data access."""

    @abstractmethod
    def validate(self) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Check whether the data source is valid.

        Returns
        -------
        ok : bool
        summary : dict
            Keys: snapshot_count, water_shape, porosity_shape, bot_shape,
            grid_info, path.
        errors : list[str]
        """

    @abstractmethod
    def list_available_items(self) -> Dict[str, Any]:
        """Return available timesteps / variables / files.

        Returns
        -------
        dict with keys like ``files``, ``timesteps``, ``variables``.
        """

    @abstractmethod
    def materialize(
        self, required_files: List[str], target_dir: str
    ) -> str:
        """Ensure *required_files* are available in *target_dir*.

        For local accessors this may be a no-op (return source dir).
        For HTTP accessors this downloads missing files.

        Returns
        -------
        str – local directory path containing the files.
        """


# ---------------------------------------------------------------------------
# Local accessor
# ---------------------------------------------------------------------------

class LocalHydroAccessor(BaseHydroAccessor):
    """Read hydro data from a local filesystem directory."""

    def __init__(self, root_path: str) -> None:
        self.root = Path(root_path).resolve()

    def validate(self) -> Tuple[bool, Dict[str, Any], List[str]]:
        summary: Dict[str, Any] = {
            "path": str(self.root),
            "snapshot_count": None,
            "water_shape": None,
            "porosity_shape": None,
            "bot_shape": None,
            "grid_info": "",
        }
        errors: List[str] = []

        if not self.root.exists():
            errors.append(f"Directory does not exist: {self.root}")
            return False, summary, errors

        missing = [f for f in REQUIRED_FILES if not (self.root / f).exists()]
        if missing:
            errors.append(f"Missing required files: {', '.join(missing)}")
            return False, summary, errors

        try:
            from PyHydroGeophysX.data_processing.table_io import npy_shape

            # Only the shapes are needed, so read the headers instead of mapping
            # the arrays: a mapping would hold these files open and block whatever
            # regenerates them.
            water = npy_shape(self.root / "Watercontent.npy")
            porosity = npy_shape(self.root / "Porosity.npy")
            bot = npy_shape(self.root / "bot.npy")

            summary["water_shape"] = water
            summary["porosity_shape"] = porosity
            summary["bot_shape"] = bot
            summary["snapshot_count"] = int(water[0]) if water else 0

            if len(water) >= 3:
                summary["grid_info"] = (
                    f"{water[1]} x {water[2]} cells, {water[0]} timesteps"
                )

            valid = bool(len(water) >= 3 and len(porosity) >= 3 and len(bot) >= 2)
            if not valid:
                errors.append(
                    "Array dimensions invalid: expected water/porosity >=3D, bot >=2D."
                )
            return valid, summary, errors

        except Exception as exc:  # noqa: BLE001
            errors.append(f"Error reading data: {exc}")
            return False, summary, errors

    def list_available_items(self) -> Dict[str, Any]:
        files = [p.name for p in self.root.iterdir() if p.is_file()]
        info: Dict[str, Any] = {"files": sorted(files)}
        try:
            from PyHydroGeophysX.data_processing.table_io import npy_shape

            shape = npy_shape(self.root / "Watercontent.npy")
            info["timesteps"] = int(shape[0]) if shape else 0
        except Exception:  # noqa: BLE001
            info["timesteps"] = None
        return info

    def materialize(
        self, required_files: List[str], target_dir: str
    ) -> str:
        # Local accessor: data already on disk.  Just return the root.
        return str(self.root)


# ---------------------------------------------------------------------------
# HTTP accessor
# ---------------------------------------------------------------------------

def _stable_cache_filename(url: str) -> str:
    """Return a deterministic filename based on the URL hash + original name."""
    name = url.rsplit("/", 1)[-1] if "/" in url else url
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    return f"{url_hash}_{name}"


class HttpHydroAccessor(BaseHydroAccessor):
    """Download hydro data on demand from an HTTP base URL.

    Parameters
    ----------
    manifest_entry : dict
        A single dataset entry from manifest.json.
    cache_dir : str, optional
        Directory for caching downloaded files.  Defaults to a temp directory.
    """

    def __init__(
        self,
        manifest_entry: Dict[str, Any],
        cache_dir: Optional[str] = None,
    ) -> None:
        self.entry = manifest_entry
        self.base_url = manifest_entry["base_url"].rstrip("/")
        self.files = list(manifest_entry.get("files", REQUIRED_FILES))
        self.dataset_id = manifest_entry.get("id", "unknown")

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(tempfile.gettempdir()) / "phgx_http_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _file_url(self, filename: str) -> str:
        return f"{self.base_url}/{filename}"

    def _cached_path(self, filename: str) -> Path:
        stable = _stable_cache_filename(self._file_url(filename))
        return self.cache_dir / stable

    def _download_file(self, filename: str, dest: Path) -> None:
        """Download a single file.  Raises on failure."""
        url = self._file_url(filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, str(dest))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to download {url}: {exc}"
            ) from exc

    def _ensure_cached(self, filename: str) -> Path:
        """Return path to cached copy, downloading if missing."""
        cached = self._cached_path(filename)
        if not cached.exists():
            self._download_file(filename, cached)
        return cached

    def validate(self) -> Tuple[bool, Dict[str, Any], List[str]]:
        summary: Dict[str, Any] = {
            "path": self.base_url,
            "snapshot_count": None,
            "water_shape": None,
            "porosity_shape": None,
            "bot_shape": None,
            "grid_info": "",
        }
        errors: List[str] = []

        # Download required files to cache for validation
        for fname in REQUIRED_FILES:
            if fname not in self.files:
                errors.append(f"Manifest does not list required file: {fname}")
                return False, summary, errors
            try:
                self._ensure_cached(fname)
            except RuntimeError as exc:
                errors.append(str(exc))
                return False, summary, errors

        # Validate array shapes by reading from cache
        try:
            wpath = self._cached_path("Watercontent.npy")
            ppath = self._cached_path("Porosity.npy")
            bpath = self._cached_path("bot.npy")

            from PyHydroGeophysX.data_processing.table_io import npy_shape

            water = npy_shape(wpath)
            porosity = npy_shape(ppath)
            bot = npy_shape(bpath)

            summary["water_shape"] = water
            summary["porosity_shape"] = porosity
            summary["bot_shape"] = bot
            summary["snapshot_count"] = int(water[0]) if water else 0

            if len(water) >= 3:
                summary["grid_info"] = (
                    f"{water[1]} x {water[2]} cells, {water[0]} timesteps"
                )

            valid = bool(len(water) >= 3 and len(porosity) >= 3 and len(bot) >= 2)
            if not valid:
                errors.append(
                    "Array dimensions invalid: expected water/porosity >=3D, bot >=2D."
                )
            return valid, summary, errors

        except Exception as exc:  # noqa: BLE001
            errors.append(f"Error reading cached data: {exc}")
            return False, summary, errors

    def list_available_items(self) -> Dict[str, Any]:
        return {
            "files": sorted(self.files),
            "timesteps": None,  # Cannot know without downloading
            "dataset_id": self.dataset_id,
        }

    def materialize(
        self, required_files: List[str], target_dir: str
    ) -> str:
        """Download required files to *target_dir*, using the cache."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        for fname in required_files:
            cached = self._ensure_cached(fname)
            dest = target / fname
            if not dest.exists():
                shutil.copy2(str(cached), str(dest))

        return str(target)

    def clear_cache(self) -> int:
        """Remove all cached files.  Returns number of files removed."""
        count = 0
        if self.cache_dir.exists():
            for p in self.cache_dir.iterdir():
                if p.is_file():
                    p.unlink()
                    count += 1
        return count
