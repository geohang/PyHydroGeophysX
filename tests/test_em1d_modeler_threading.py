"""The forward operator must not be shared between threads.

SimPEG's simulation caches the sensitivity on itself, so two threads calling
``getJ`` on one instance race and one of them can read the cache mid-write. The
symptom is a Jacobian that comes back with no dimensions at all, which surfaces
far away as a matmul complaining about operand shapes.

Blocks are assembled on the thread that reads the survey and evaluated on a pool
of workers, so an operator resolved at assembly time is exactly the shared
instance this guards against.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from PyHydroGeophysX.inversion import em1d


class _FakeModeler:
    """Stands in for the SimPEG-backed operator, which is slow to build."""

    def __init__(self, thicknesses, survey_config) -> None:
        self.thicknesses = np.asarray(thicknesses, dtype=float)
        self.config = survey_config
        self.owner = threading.get_ident()

    def forward(self, sigma):
        return np.ones(np.size(self.config.times), dtype=float)

    def sensitivity(self, sigma):
        return np.ones((np.size(self.config.times), np.size(sigma)), dtype=float)


@pytest.fixture()
def fake_modeler(monkeypatch):
    monkeypatch.setattr(
        "PyHydroGeophysX.forward.tdem_forward.TDEMForwardModeling", _FakeModeler)
    # The cache lives on a module-level threading.local, so a test that leaves
    # entries behind changes what the next one measures.
    monkeypatch.setattr(em1d, "_MODELER_CACHE", threading.local())
    return _FakeModeler


GEOMETRY = {"height": 0.0, "tx_rx_sep": 15.0, "source_radius": 0.36,
            "orientation": "z", "receiver_type": "dbdt", "waveform": "step_off"}
THICK = np.array([1.0, 2.0, 4.0])
TIMES = np.geomspace(1e-5, 1e-3, 6)


def test_one_thread_reuses_its_own_operator(fake_modeler) -> None:
    """The whole point of the cache: a line builds one, not one per station."""
    first = em1d._thread_local_modeler(THICK, GEOMETRY, TIMES)
    second = em1d._thread_local_modeler(THICK, GEOMETRY, TIMES)

    assert first is second


def test_each_thread_gets_its_own_operator(fake_modeler) -> None:
    """Two threads must never end up holding the same instance."""
    made = {}

    def build(tag: str) -> None:
        made[tag] = em1d._thread_local_modeler(THICK, GEOMETRY, TIMES)

    workers = [threading.Thread(target=build, args=(f"t{i}",)) for i in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    assert len(made) == 4
    assert len({id(m) for m in made.values()}) == 4
    # Each was built by the thread that asked for it, not handed over.
    assert len({m.owner for m in made.values()}) == 4


def test_a_different_survey_gets_a_different_operator(fake_modeler) -> None:
    a = em1d._thread_local_modeler(THICK, GEOMETRY, TIMES)
    b = em1d._thread_local_modeler(THICK, {**GEOMETRY, "tx_rx_sep": 30.0}, TIMES)
    c = em1d._thread_local_modeler(THICK * 2.0, GEOMETRY, TIMES)

    assert a is not b
    assert a is not c


def test_a_block_carries_the_configuration_not_the_instance(fake_modeler) -> None:
    """An instance stored on the block is the bug; the configuration is the fix."""
    item = {"times": TIMES, "response": np.ones(TIMES.size),
            "relative_std": np.full(TIMES.size, 0.05)}
    blocks = em1d.tdem_moment_blocks(
        {"moments": {"HM": item}}, GEOMETRY,
        {"rel_error": 0.03, "noise_floor": 0.0}, THICK)

    assert blocks, "the fixture should produce one usable block"
    for block in blocks:
        assert "modeler" not in block
        assert "geometry" in block and "thicknesses" in block


def test_blocks_built_on_one_thread_evaluate_on_a_pool(fake_modeler) -> None:
    """The failure this file exists for, end to end.

    Before the fix every block held the assembling thread's instance, so the
    workers all drove one operator at once.
    """
    item = {"times": TIMES, "response": np.ones(TIMES.size),
            "relative_std": np.full(TIMES.size, 0.05)}
    built = [
        em1d.tdem_moment_blocks({"moments": {"HM": dict(item)}}, GEOMETRY,
                                {"rel_error": 0.03, "noise_floor": 0.0}, THICK)
        for _ in range(6)
    ]
    sigma = np.full(THICK.size + 1, 0.01)

    def evaluate(blocks):
        used = em1d._block_modeler(blocks[0])
        jac = em1d._moment_jacobian(blocks)(sigma)
        return used.owner, np.ndim(jac), jac.shape

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(evaluate, built))

    # Every Jacobian is a real matrix, which is what the race destroyed.
    for _, ndim, shape in results:
        assert ndim == 2
        assert shape == (TIMES.size, THICK.size + 1)
    # And each was produced by an operator belonging to the worker thread,
    # never by the thread that assembled the blocks.
    assert threading.get_ident() not in {owner for owner, _, _ in results}
