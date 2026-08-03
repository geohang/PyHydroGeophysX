from __future__ import annotations

import numpy as np

from PyHydroGeophysX.data_processing.ert_data_agent import (
    _load_ert_embedded_parsers,
    _source_error_to_relative,
)


def test_embedded_e4d_converts_absolute_v_std_to_relative_error(tmp_path):
    source = tmp_path / "survey.ohm"
    source.write_text(
        "\n".join(
            [
                "5 # Number of electrodes",
                "1 0 0 100 1",
                "2 1 0 100 1",
                "3 2 0 100 1",
                "4 3 0 100 1",
                "5 4 0 100 1",
                "3 # Number of data",
                # index A B M N v_obs v_std
                "1 1 2 3 4 2.0 0.1",
                "2 1 2 4 5 0.1 0.02",
                "3 2 3 4 5 10.0 0.01",
            ]
        ),
        encoding="utf-8",
    )

    ert = _load_ert_embedded_parsers(
        data_file=str(source),
        project_dir=str(tmp_path / "project"),
        instrument="E4D",
    )

    errors = np.asarray([obs.rel_err for obs in ert.observations], dtype=float)
    # E4D stores an absolute resistance standard deviation. The embedded path
    # clamps the resulting relative errors to its established 1%-50% range.
    np.testing.assert_allclose(errors, [0.05, 0.20, 0.01])


def test_relative_error_column_is_not_raised_to_five_percent():
    errors = _source_error_to_relative(
        [0.025, 0.05, 0.50],
        [1.0, 1.0, 1.0],
        instrument="BERT",
        min_error=0.005,
    )

    np.testing.assert_allclose(errors, [0.025, 0.05, 0.50])
