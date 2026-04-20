# Contributing to PyHydroGeophysX

Thank you for considering a contribution. PyHydroGeophysX is designed to be extensible,
and community contributions are welcome in four main areas.

## 1. Development setup

```bash
git clone https://github.com/geohang/PyHydroGeophysX.git
cd PyHydroGeophysX
pip install -e ".[dev,docs]"
pytest
```

Please run `pytest` and `flake8 PyHydroGeophysX/` before submitting a PR.

## 2. Extension points

### Adding a new petrophysical model
Subclass the base resistivity or velocity model in
`PyHydroGeophysX/petrophysics/resistivity_models.py` or
`PyHydroGeophysX/petrophysics/velocity_models.py`. Implement the
`forward(water_content, porosity, ...)` method and register the class name
in the module's `__all__` list.

### Adding a new hydrological-model loader
Follow the pattern in `PyHydroGeophysX/model_output/modflow_output.py` and
`parflow_output.py`. Create a new module under `model_output/` exposing
classes that load saturation, water content, and porosity into NumPy
arrays with shape `(time, z, y, x)` for 3D or `(time, z, x)` for 2D.

### Adding a new geophysical forward operator
Add a module under `PyHydroGeophysX/forward/` following the interface of
`ert_forward.py` and `srt_forward.py`: a class exposing `create_synthetic_data`
and `response` methods. Where possible, wrap an established library
(PyGIMLi, SimPEG) rather than re-implementing physics.

### Adding a new inversion scheme
Mirror the structure of `PyHydroGeophysX/inversion/ert_inversion.py`:
a class with `run()` returning a dict with `model`, `response`, and
`misfit` keys.

## 3. Pull request workflow

1. Fork the repository and create a feature branch:
   `git checkout -b feature/your-feature-name`
2. Add tests under `tests/` that cover the new functionality.
3. Run the full test suite: `pytest --cov=PyHydroGeophysX`.
4. Update the relevant example notebook in `examples/` if the feature is
   user-facing.
5. Update `README.md` and the Sphinx docs in `docs/` as appropriate.
6. Open a pull request against `main`. CI must pass on Linux, macOS, and
   Windows before review.

## 4. Reporting issues

Please use the GitHub issue tracker with a minimal reproducible example,
your operating system, and the output of `pip freeze | grep -Ei 'pygimli|simpeg|numpy|scipy'`.

## Code of conduct

We follow a standard academic-open-source code of conduct: be respectful,
assume good faith, and keep discussion technical.
