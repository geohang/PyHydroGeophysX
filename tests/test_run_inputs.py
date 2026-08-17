"""The compact format a run stores its inputs in.

A run used to record what it read by copying it, so the Project grew by the size
of the survey on every inversion: a TEMcompany folder is hundreds of megabytes,
a time-lapse sequence is one raw file per time step. These pin the two shapes
that replaced the copies, and the back-compatible path for runs recorded before
they existed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.data_processing import run_inputs


def test_a_container_round_trips_nested_arrays_and_scalars(tmp_path) -> None:
    payload = {
        "times": np.geomspace(1e-5, 1e-2, 30),
        "system": {"loop_area": 4.0, "moment": "HM", "gates": np.arange(6)},
        "flags": [True, None, 2],
    }

    stored = run_inputs.save_container(
        tmp_path / "c", payload, kind="em_soundings", meta={"method": "TDEM"}
    )
    loaded = run_inputs.load_container(stored, kind="em_soundings")

    np.testing.assert_allclose(loaded["times"], payload["times"])
    np.testing.assert_array_equal(loaded["system"]["gates"], payload["system"]["gates"])
    assert loaded["system"]["moment"] == "HM"
    assert loaded["flags"] == [True, None, 2]
    assert run_inputs.read_manifest(stored)["meta"]["method"] == "TDEM"


def test_a_container_refuses_to_be_read_as_another_kind(tmp_path) -> None:
    stored = run_inputs.save_container(
        tmp_path / "c", {"x": np.arange(3)}, kind="hydrology_arrays"
    )

    with pytest.raises(ValueError, match="hydrology_arrays"):
        run_inputs.load_container(stored, kind="em_soundings")


def test_a_sequence_stores_what_its_items_share_only_once(tmp_path) -> None:
    """This is the whole reason the sequence shape exists.

    Storing each sounding whole repeated the system geometry, the protocol and
    the gate times once per station, and made a two-sounding container larger
    than the raw files it replaced.
    """
    shared = {"loop_area": 4.0, "gates": np.arange(64, dtype=float)}
    items = [
        {"system": shared, "response": np.full(64, float(index)), "sounding": index}
        for index in range(200)
    ]

    sequence = run_inputs.save_sequence_container(
        tmp_path / "seq", items, kind="em_soundings"
    )
    whole = run_inputs.save_container(tmp_path / "whole", items, kind="em_soundings")

    assert sequence.stat().st_size < whole.stat().st_size / 10
    assert run_inputs.sequence_length(sequence) == 200
    for index in (0, 7, 199):
        item = run_inputs.load_sequence_item(sequence, index)
        assert item["sounding"] == index
        np.testing.assert_allclose(item["response"], float(index))
        np.testing.assert_allclose(item["system"]["gates"], shared["gates"])


def test_a_sequence_index_stays_inside_the_stored_range(tmp_path) -> None:
    """``invert_line`` clamps its station index; the container must agree."""
    items = [{"response": np.full(4, float(i))} for i in range(3)]
    sequence = run_inputs.save_sequence_container(
        tmp_path / "seq", items, kind="em_soundings"
    )

    np.testing.assert_allclose(
        run_inputs.load_sequence_item(sequence, 99)["response"], 2.0
    )


def test_a_sequence_survives_items_whose_arrays_differ_in_length(tmp_path) -> None:
    """Per-sounding flag filtering drops a different number of gates each time."""
    items = [{"response": np.arange(4 + index, dtype=float)} for index in range(3)]

    sequence = run_inputs.save_sequence_container(
        tmp_path / "seq", items, kind="em_soundings"
    )

    for index in range(3):
        np.testing.assert_allclose(
            run_inputs.load_sequence_item(sequence, index)["response"],
            np.arange(4 + index, dtype=float),
        )


def test_a_file_bundle_returns_every_byte_it_was_given(tmp_path) -> None:
    """A PyGIMLi mesh and a BERT data file are read back by their own loaders."""
    mesh = tmp_path / "mesh_res.bms"
    mesh.write_bytes(bytes(range(256)) * 40)
    array = tmp_path / "resmodel.npy"
    np.save(array, np.random.default_rng(0).random((500, 4)))

    bundle = run_inputs.save_file_bundle(
        tmp_path / "inputs" / "ert_model_inputs",
        {"mesh_res.bms": mesh, "resmodel.npy": array},
        kind="ert_model_bundle",
    )
    out = run_inputs.expand_file_bundle(bundle, tmp_path / "scratch")

    assert run_inputs.is_file_bundle(bundle)
    assert run_inputs.bundle_file_names(bundle) == ["mesh_res.bms", "resmodel.npy"]
    assert (out / "mesh_res.bms").read_bytes() == mesh.read_bytes()
    np.testing.assert_array_equal(np.load(out / "resmodel.npy"), np.load(array))


def test_a_bundle_keeps_the_order_it_was_given(tmp_path) -> None:
    """Step n is what step n-1 is compared to, so a time-lapse order is data."""
    names = [f"step_{index:04d}.dat" for index in range(12)]
    for name in names:
        (tmp_path / name).write_text(f"# {name}\n1 2 3\n", encoding="utf-8")

    bundle = run_inputs.save_file_bundle(
        tmp_path / "inputs" / "ert_timesteps",
        {name: tmp_path / name for name in names},
        kind="ert_timelapse_observations",
    )

    assert run_inputs.bundle_file_names(bundle) == names


def test_a_bundle_of_ascii_data_files_is_much_smaller_than_the_copies(tmp_path) -> None:
    """BERT data files are ASCII, which is where a time-lapse run grew fastest."""
    rng = np.random.default_rng(0)
    names = []
    for index in range(10):
        name = f"step_{index:04d}.dat"
        rows = "\n".join(
            " ".join(f"{value:.6f}" for value in row) for row in rng.random((400, 6))
        )
        (tmp_path / name).write_text(f"400# a b m n rhoa err\n{rows}\n", encoding="utf-8")
        names.append(name)
    loose = sum((tmp_path / name).stat().st_size for name in names)

    bundle = run_inputs.save_file_bundle(
        tmp_path / "inputs" / "ert_timesteps",
        {name: tmp_path / name for name in names},
        kind="ert_timelapse_observations",
    )

    assert bundle.stat().st_size < loose / 2


def test_a_bundle_cannot_write_outside_the_directory_it_expands_into(tmp_path) -> None:
    """A bundle is a file like any other, so its names are not to be trusted."""
    victim = tmp_path / "payload.txt"
    victim.write_text("data", encoding="utf-8")
    bundle = run_inputs.save_file_bundle(
        tmp_path / "b", {"../escaped.txt": victim}, kind="ert_model_bundle"
    )

    out = run_inputs.expand_file_bundle(bundle, tmp_path / "scratch")

    assert [item.name for item in out.iterdir()] == ["escaped.txt"]
    assert not (tmp_path / "escaped.txt").exists()


def test_a_plain_container_is_not_mistaken_for_a_bundle(tmp_path) -> None:
    stored = run_inputs.save_container(
        tmp_path / "c", {"x": np.arange(3)}, kind="em_soundings"
    )

    assert run_inputs.is_container(stored)
    assert not run_inputs.is_file_bundle(stored)


def test_an_unrelated_npz_is_not_taken_for_a_container(tmp_path) -> None:
    """``is_container`` sits in a loader's dispatch chain, so it must be sure."""
    plain = tmp_path / "plain.npz"
    np.savez(plain, x=np.arange(3))

    assert not run_inputs.is_container(plain)
    assert not run_inputs.is_container(tmp_path / "missing.npz")


# --- how a run consumes what it stored ---------------------------------------
# The readers on the other side are PyGIMLi's loaders and ``np.load``, which
# want a real path, so the bundle is expanded for the length of the run only.


def test_a_bundle_expands_for_the_run_and_leaves_nothing_behind(tmp_path) -> None:
    from PyHydroGeophysX.workflows.domain import _input_directory

    for name in ("Watercontent.npy", "Porosity.npy"):
        np.save(tmp_path / name, np.arange(10.0))
    bundle = run_inputs.save_file_bundle(
        tmp_path / "inputs" / "hydro_inputs",
        {name: tmp_path / name for name in ("Watercontent.npy", "Porosity.npy")},
        kind="hydrology_arrays",
    )

    with _input_directory(str(bundle), {}) as data_dir:
        listed = sorted(item.name for item in Path(data_dir).iterdir())
        assert listed == ["Porosity.npy", "Watercontent.npy"]
        scratch = Path(data_dir)

    assert not scratch.exists(), "the expanded copy must not outlive the run"


def test_a_run_recorded_before_the_bundle_still_resolves_its_directory(tmp_path) -> None:
    """Old runs carry copied files; those are already a directory."""
    from PyHydroGeophysX.workflows.domain import _input_directory

    legacy = tmp_path / "runs" / "old" / "inputs"
    legacy.mkdir(parents=True)
    files = {}
    for name in ("Watercontent.npy", "Porosity.npy"):
        np.save(legacy / name, np.arange(4.0))
        files[name] = str(legacy / name)

    with _input_directory(None, files) as data_dir:
        assert Path(data_dir) == legacy


def test_inputs_scattered_across_directories_are_refused(tmp_path) -> None:
    from PyHydroGeophysX.workflows.domain import _input_directory

    files = {"a.npy": str(tmp_path / "one" / "a.npy"),
             "b.npy": str(tmp_path / "two" / "b.npy")}

    with pytest.raises(ValueError, match="one directory"):
        with _input_directory(None, files):
            pass


def test_a_run_with_no_stored_inputs_asks_for_no_directory(tmp_path) -> None:
    from PyHydroGeophysX.workflows.domain import _input_directory

    with _input_directory(None, {}) as data_dir:
        assert data_dir is None
