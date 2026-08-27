"""Session-wide determinism shims (test-infra A4-3).

This module is imported by pytest *before any test module*, which makes it the
one place where we can pin the BLAS/OpenMP thread counts **before NumPy (and
thus its backing BLAS) is first imported**. Every numeric backend — OpenBLAS,
MKL, Apple Accelerate/vecLib — reads its thread-count environment variable
exactly once, at import time; setting it afterwards is a no-op. Pytest imports
``conftest.py`` ahead of the test tree and ahead of ``alphaforge`` (which pulls
in NumPy), so these assignments win the race.

Why single-threaded matters for a quant repo: a multi-threaded BLAS sums
reductions (dot products, ``A @ B``, Cholesky) in a nondeterministic order, so
the last bits of a covariance, an MVO solve, or an HMM fit can differ run to
run and machine to machine. Pinning every backend to one thread removes that
source of non-reproducibility, which is what the reference-hash regression in
``tests/integration/test_reference_hash.py`` relies on.

``PYTHONHASHSEED``: not set here. ``conftest.py`` runs inside an
already-started interpreter, and ``PYTHONHASHSEED`` is consumed at *interpreter
startup* — assigning it now would be silently ignored. It is pinned at the
process boundary instead (``.github/workflows/ci.yml`` exports
``PYTHONHASHSEED=0``); set it the same way locally if a hash-ordering bug is
ever suspected (``PYTHONHASHSEED=0 uv run pytest``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CLEAN_CHECKOUT_POLICY = (
    _ROOT / "config" / "clean_checkout_workspace_evidence_policy.json"
)
_WORKSPACE_EVIDENCE_MODULES = frozenset(
    json.loads(_CLEAN_CHECKOUT_POLICY.read_text())["test_modules"]
)

# Pin every BLAS/OpenMP backend to a single thread BEFORE NumPy is imported.
# Each variable is read once at backend import; setdefault leaves an explicit
# operator override (e.g. a CI matrix entry) untouched.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")


def pytest_collection_modifyitems(config, items) -> None:
    """Apply the clean-checkout policy and repair ``no_cover`` without coverage.

    Artifact-bound modules are centrally classified as ``workspace_evidence`` so
    GitHub's clean checkout can deselect them explicitly. They still run in the
    ordinary local suite, where the ignored research corpus is available. The
    policy is about input availability; deselection is never reported as a pass.

    pytest-cov's ``no_cover`` handling calls ``self.cov_controller.pause()``, and under
    ``--no-cov`` that controller is ``None`` — so the marker raises
    ``AttributeError: 'NoneType' object has no attribute 'pause'`` and the test FAILS for a
    reason that has nothing to do with the test. There is nothing to pause when coverage is
    already off, so the marker is simply removed.

    This is not cosmetic. ``tests/integration/test_scale_guard.py`` passes under the default
    (coverage-on) invocation and fails under ``uv run pytest --no-cov`` — the fast invocation
    anyone reaches for. It was recorded in a 2026-08-18 commit message as a genuine outstanding
    failure inherited from another lane. It was neither: it was the flag.
    """
    for item in items:
        module_path = Path(str(item.path)).resolve().relative_to(_ROOT).as_posix()
        if module_path in _WORKSPACE_EVIDENCE_MODULES:
            item.add_marker("workspace_evidence")

    if config.pluginmanager.hasplugin("_cov"):
        cov = config.pluginmanager.getplugin("_cov")
        if getattr(cov, "cov_controller", None) is not None:
            return
    for item in items:
        if item.get_closest_marker("no_cover") is not None:
            item.own_markers = [m for m in item.own_markers if m.name != "no_cover"]
