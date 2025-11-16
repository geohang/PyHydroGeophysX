#!/usr/bin/env python
"""Test ERT loading with PyGIMLi directly"""

import os
from pathlib import Path
from pygimli.physics import ert

print("Testing ERT load methods...")

ert_file = Path('data/ERT/Bert/fielddataline2.dat')

# Method 1: Try PyGIMLi's direct ERT loader
print("\n[Method 1] Using pygimli.physics.ert.load()...")
try:
    # PyGIMLi has its own ERT data loader
    ertData = ert.load(str(ert_file))
    print(f"[OK] ERT data loaded: {ertData.sensorCount()} electrodes, {ertData.size()} measurements")
    print(f"  Apparent resistivity range: {min(ertData['rhoa']):.1f} - {max(ertData['rhoa']):.1f} Ohm-m")
except Exception as e:
    print(f"[FAIL] PyGIMLi load failed: {e}")
    import traceback
    traceback.print_exc()

# Method 2: Try with different format specification
print("\n[Method 2] Trying to manually specify format...")
try:
    import pygimli as pg
    # Load the data container
    data = pg.DataContainer(str(ert_file), sensorTokens='#')
    print(f"[OK] Data loaded as container: {data.sensorCount()} sensors, {data.size()} measurements")
except Exception as e:
    print(f"[FAIL] Manual format failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDONE")

