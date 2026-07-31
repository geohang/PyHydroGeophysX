"""Generate the desktop-app icons from the repository logo.

Creates ``packaging/app_icon.ico`` (Windows) and ``packaging/app_icon.icns``
(macOS) from the root ``logo.png``. The PyInstaller spec picks them up when they
exist; both files are build artifacts and are not committed.

Usage (from the repository root)::

    pip install pillow
    python packaging/make_icons.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - guidance only
    raise SystemExit("Pillow is required: pip install pillow")

PACKAGING_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGING_DIR.parent
LOGO = REPO_ROOT / "logo.png"

ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
ICNS_SIZES = [16, 32, 64, 128, 256, 512]
ICNS_BASE = 1024


def _square(image: Image.Image) -> Image.Image:
    """Pad the logo to a square canvas with a transparent background."""
    image = image.convert("RGBA")
    side = max(image.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def main() -> int:
    if not LOGO.exists():
        print(f"logo not found: {LOGO}", file=sys.stderr)
        return 1
    base = _square(Image.open(LOGO))

    ico_path = PACKAGING_DIR / "app_icon.ico"
    base.save(ico_path, format="ICO", sizes=[(s, s) for s in ICO_SIZES])
    print(f"wrote {ico_path}")

    icns_path = PACKAGING_DIR / "app_icon.icns"
    big = base.resize((ICNS_BASE, ICNS_BASE), Image.LANCZOS)
    extra = [base.resize((s, s), Image.LANCZOS) for s in ICNS_SIZES]
    big.save(icns_path, format="ICNS", append_images=extra)
    print(f"wrote {icns_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
