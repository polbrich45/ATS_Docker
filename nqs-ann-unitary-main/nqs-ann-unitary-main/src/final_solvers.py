r"""Optional exact and DMRG reference solves for final rotated Pauli terms."""

from __future__ import annotations

import os
import time
import warnings
from typing import Any

import numpy as np

from .config        import FinalConfig, SPIN
from .logging_utils import log
from .operators     import PauliTerm, commutes_with_total_sz


_I         = np.eye(2, dtype=np.complex128)
_X         = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y         = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z         = np.array([[1, 0], [0, -1]], dtype=np.complex128)
# Single-site Pauli indexed by the (x, z) bit pair of a Pauli string. The both-set entry is
# Y = i X Z, matching the i^(x.z) X^x Z^z convention shared with the rotation circuit.
_SITE_OP   = {(0, 0): _I, (1, 0): _X, (0, 1): _Z, (1, 1): _Y}


def _history_row(energy: float, n_terms: int, first_energy: float, seconds: float) -> dict[str, float]:
    """One trace record shared by both solvers; ``seconds`` is cumulative runtime."""
    return {
        "energy": float(energy),
        "variance": 0.0,
        "n_terms": float(n_terms),
        "dE": float(energy - first_energy),
        "seconds": float(seconds),
    }


def solve_exact(terms: list[PauliTerm], n_sites: int, n_up: int | None = None) -> dict[str, Any]:
    """Lanczos ground state in the requested spin sector."""
    import netket as nk

    from .netket_vmc import pauli_terms_to_netket

    started = time.perf_counter()
    # A fixed particle number is meaningful only inside a total-Sz sector, so require [H, Sz] = 0.
    if n_up is not None and not 0 <= n_up <= n_sites:
        raise ValueError("n_up must lie in [0, n_sites]")
    if n_up is not None and not commutes_with_total_sz(terms, n_sites):
        raise ValueError("A fixed-n_up exact solve requires [H, total Sz] = 0")
    hilbert  = nk.hilbert.Spin(s=SPIN, N=n_sites) if n_up is None else nk.hilbert.Spin(s=SPIN, N=n_sites, total_sz=SPIN * (2 * n_up - n_sites))
    operator = pauli_terms_to_netket(terms, hilbert)
    # Lanczos needs at least two states, so diagonalize those tiny sectors densely instead.
    if int(hilbert.n_states) <= 2:
        energy = float(np.linalg.eigvalsh(np.asarray(operator.to_dense()))[0])
    else:
        values = nk.exact.lanczos_ed(operator, k=1, compute_eigenvectors=False)
        energy = float(np.real(np.asarray(values).reshape(-1)[0]))
    return {
        "method"    : "exact",
        "energy"    : energy,
        "variance"  : 0.0,
        "n_up"      : n_up,
        "history"   : [_history_row(energy, len(terms), energy, time.perf_counter() - started)],
    }


# -------------------------


def _pauli_terms_to_mpo(terms: list[PauliTerm], n_sites: int, *, cutoff: float, max_bond: int) -> Any:
    """Build and compress one MPO from a Pauli expansion."""
    import quimb.tensor as qtn

    options: dict[str, Any] = {"cutoff": float(cutoff)}
    if max_bond > 0:
        options["max_bond"] = int(max_bond)
    # Accumulate product operators with a binary counter: level ``k`` holds one partial sum, and
    # a filled level is merged (and compressed) into the next to keep the intermediate bond low.
    levels: list[Any | None] = []
    for term in terms:
        # One product operator per Pauli string; the coefficient rides on the first site.
        arrays    = [_SITE_OP[((term.x >> site) & 1, (term.z >> site) & 1)] for site in range(n_sites)]
        arrays[0] = complex(term.coefficient) * arrays[0]
        piece     = qtn.MPO_product_operator(arrays)
        level = 0
        while level < len(levels) and levels[level] is not None:
            piece         = levels[level].add_MPO(piece, compress=True, **options)
            levels[level] = None
            level         += 1
        if level == len(levels):
            levels.append(piece)
        else:
            levels[level] = piece
    pieces = [piece for piece in levels if piece is not None]
    if not pieces:
        raise ValueError("Cannot build a DMRG MPO from an empty Hamiltonian")
    mpo = pieces[0]
    for piece in pieces[1:]:
        mpo = mpo.add_MPO(piece, compress=True, **options)
    mpo.compress(**options)
    return mpo


def solve_dmrg(terms: list[PauliTerm], n_sites: int, config: FinalConfig) -> dict[str, Any]:
    """Finite two-site DMRG for already-rotated Pauli terms."""
    # quimb/numba JIT artifacts need a writable cache directory in restricted runs.
    cache_dir = os.environ.get("NUMBA_CACHE_DIR") or "/tmp/numba_cache"
    os.environ["NUMBA_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    try:
        import numba

        numba.config.CACHE_DIR = cache_dir
        import quimb.tensor as qtn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("The DMRG final solver requires quimb (`pip install quimb`).") from exc

    started = time.perf_counter()
    # Warm up through small bond caps before the requested target to stabilize the sweeps.
    schedule = [bond for bond in (8, 16, 32, config.dmrg_mps_bond_dim) if bond < config.dmrg_mps_bond_dim]
    schedule.append(config.dmrg_mps_bond_dim)
    # Capture rather than raise RuntimeWarnings from the sweeps; they are counted in the result.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        mpo = _pauli_terms_to_mpo(
            terms,
            n_sites,
            cutoff=config.dmrg_mpo_cutoff,
            max_bond=config.dmrg_mpo_max_bond,
        )
        dmrg = qtn.DMRG2(mpo, bond_dims=schedule, cutoffs=config.dmrg_mpo_cutoff)
        dmrg.solve(tol=config.dmrg_tol, max_sweeps=config.dmrg_sweeps, verbosity=0)
        energy = float(np.real(dmrg.energy))
        energies = [float(np.real(value)) for value in dmrg.energies if np.isfinite(np.real(value))] or [energy]
        state = getattr(dmrg, "state", None)
        mpo_bond = int(mpo.max_bond())

    if not np.isfinite(energy):
        raise FloatingPointError("DMRG returned a non-finite final energy")
    elapsed = time.perf_counter() - started
    # Spread the sweeps evenly over the wall time; the first sweep is the dE reference.
    history = [
        _history_row(value, len(terms), energies[0], elapsed * (index + 1) / len(energies))
        for index, value in enumerate(energies)
    ]
    # DMRG reports energies but no energy variance, so mark the field explicitly undefined.
    for row in history:
        row["variance"] = float("nan")
    return {
        "method": "dmrg",
        "energy": energy,
        "variance": float("nan"),
        "history": history,
        "mps_bond_dim": int(config.dmrg_mps_bond_dim),
        "mps_bond_dim_observed": int(state.max_bond()) if state is not None else int(config.dmrg_mps_bond_dim),
        "mpo_bond_dim": mpo_bond,
        "mpo_max_bond": int(config.dmrg_mpo_max_bond),
        "mpo_terms": int(len(terms)),
        "mpo_cutoff": float(config.dmrg_mpo_cutoff),
        "dmrg_tol": float(config.dmrg_tol),
        "dmrg_runtime_warnings": sum(issubclass(item.category, RuntimeWarning) for item in caught),
    }


def solve_final(
    terms   : list[PauliTerm],
    n_sites : int,
    config  : FinalConfig,
    n_up    : int | None = None,
) -> dict[str, Any]:
    """Dispatch a final reference solve on direct, already-rotated terms."""
    solver_mode = config.solver.lower()
    if solver_mode == "none":
        return {}
    # "auto" uses exact diagonalization while the dense sector is still small.
    if solver_mode == "exact" or (solver_mode == "auto" and n_sites <= 14):
        return solve_exact(terms, n_sites, n_up=n_up)
    # The DMRG backend does not enforce a fixed particle-number sector.
    if n_up is not None:
        raise ValueError("The current DMRG backend does not enforce a fixed-n_up sector")
    try:
        return solve_dmrg(terms, n_sites, config)
    except ImportError:
        log(f"DMRG solver requested ({solver_mode}), but quimb is not installed (`pip install quimb`). Skipping final DMRG solve.", severity="warning")
        return {}


# -------------------------
#! EOF
# -------------------------
