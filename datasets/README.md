# Dataset Manifest for PyHydroGeophysX

## Overview

The `manifest.json` file in this directory describes datasets available to the
**Hydro → Geophysics** workflow in the Streamlit app. It allows the app to
download data on demand from HTTP URLs — critical for Streamlit Cloud where
users cannot provide local filesystem paths.

## How it works

1. The app reads `datasets/manifest.json` at startup.
2. When a user selects a **GitHub/HTTP dataset**, the app uses
   `HttpHydroAccessor` to download only the required files (lazily, with
   caching).
3. Downloaded files are cached in a temp directory with stable filenames so
   repeat runs skip the download.

## Manifest format

```json
{
  "version": "1.0",
  "datasets": [
    {
      "id": "unique_id",
      "name": "Human-readable name",
      "description": "Short description",
      "base_url": "https://raw.githubusercontent.com/...",
      "files": ["Watercontent.npy", "Porosity.npy", "top.txt", "bot.npy"],
      "metadata": {
        "model_type": "MODFLOW",
        "grid_description": "3D structured grid",
        "notes": "Any additional notes"
      }
    }
  ]
}
```

### Fields

| Field         | Required | Description                                              |
|---------------|----------|----------------------------------------------------------|
| `id`          | yes      | Unique identifier for the dataset                        |
| `name`        | yes      | Display name shown in the UI                             |
| `description` | yes      | Short description shown in the selector                  |
| `base_url`    | yes      | HTTP base URL; each file is fetched as `base_url/file`   |
| `files`       | yes      | List of filenames available at the base URL               |
| `metadata`    | no       | Free-form dict shown in an expandable panel in the UI     |

## Adding a new dataset

1. Host your data files on a publicly accessible HTTP server. Options:
   - **GitHub raw URLs** for small files (< 100 MB each):
     `https://raw.githubusercontent.com/<owner>/<repo>/<branch>/path/to/data`
   - **GitHub Releases** for larger files:
     Upload as release assets, then use the download URL as `base_url`.
   - Any other static file host (S3, GCS, university server, etc.).

2. Add an entry to `manifest.json` with the correct `base_url` and `files` list.

3. Commit and push. The Streamlit app will pick it up automatically.

## Data size guidelines

- **Bundled example data** (`examples/data/`): keep small (< 50 MB total).
  This data is included in the Git repository and is always available locally.
- **Large datasets**: host in GitHub Releases or external storage and reference
  via the manifest. This keeps the repository lightweight while still allowing
  Cloud users to access the data.

## Accessor classes

The `PyHydroGeophysX.data_access` module provides:

- `LocalHydroAccessor(root_path)` — reads from a local directory.
- `HttpHydroAccessor(manifest_entry, cache_dir)` — downloads on demand.

Both implement the `BaseHydroAccessor` interface with:
- `validate()` → `(ok, summary_dict, errors)`
- `list_available_items()` → dict of files / timesteps
- `materialize(required_files, target_dir)` → local directory path
