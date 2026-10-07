r"""Capped sampled-Ritz basis construction and Epstein-Nesbet PT2."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .logging_utils import log
from .operators import PauliTerm
from .rotation_ritz import (
    _binary_codes,
    _codes_to_bits,
    _connected_codes_and_signs,
    build_rotated_subspace_hamiltonian,
    prepare_rotation_problem,
)


_FRONTIER_STATES_PER_BASIS = 64
_MIN_FRONTIER_STATES = 4096


class _FrontierLimitExceeded(RuntimeError):
    """Internal signal that frontier aggregation exceeded its state budget."""


def _frontier_amplitudes(
    terms		: list[PauliTerm],
    samples		: np.ndarray,
    vector		: np.ndarray | jax.Array,
    n_sites		: int,
    max_states	: int | None = None,
    n_up		: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return coherent frontier amplitudes and diagonal matrix elements.

    For every configuration \(y\) outside ``samples`` reached by one off-diagonal
    Pauli string, this computes

    .. math::

        A_y=\langle y|H|v\rangle,
        \qquad H_{yy}=\langle y|H|y\rangle.

    Contributions are accumulated coherently before any absolute square.
    """
    samples	= np.asarray(samples, dtype=np.int64)
    empty	= (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.complex128),
        np.zeros(0, dtype=np.float64),
    )
    if samples.shape[0] == 0 or not terms:
        return empty

    sample_codes	= _binary_codes(samples)
    in_subspace	= {int(code) for code in sample_codes}
    v0			= np.asarray(vector, dtype=np.complex128)
    amplitudes: dict[int, complex] = {}
    for term in terms:
        if term.x == 0:
            continue
        y_codes, sign = _connected_codes_and_signs(
            samples, sample_codes, int(term.x), int(term.z), n_sites,
        )
        y_phase	= (1.0j) ** int(bin(term.x & term.z).count("1"))
        contrib	= complex(term.coefficient) * y_phase * (sign * v0)
        for code, amplitude in zip(y_codes, contrib):
            code = int(code)
            if code in in_subspace or amplitude == 0.0j:
                continue
            if n_up is not None and code.bit_count() != n_sites - n_up:
                continue
            if code not in amplitudes and max_states is not None and len(amplitudes) >= max_states:
                raise _FrontierLimitExceeded
            amplitudes[code] = amplitudes.get(code, 0.0j) + amplitude
    if not amplitudes:
        return empty

    code_dtype			= np.int64 if n_sites <= 63 else object
    y_codes				= np.fromiter(amplitudes.keys(), dtype=code_dtype, count=len(amplitudes))
    a_frontier			= np.fromiter(amplitudes.values(), dtype=np.complex128, count=len(amplitudes))
    # XX and YY paths can cancel exactly, including all forbidden spin flips.
    coupled				= np.abs(a_frontier) > 0.0
    y_codes, a_frontier = y_codes[coupled], a_frontier[coupled]
    y_bits				= _codes_to_bits(y_codes, n_sites, dtype=np.int64) if n_sites <= 63 else np.empty((y_codes.shape[0], 0), dtype=np.int64)
    diagonal			= np.zeros(y_codes.shape[0], dtype=np.float64)
    for term in terms:
        if term.x != 0:
            continue
        _, sign 		= _connected_codes_and_signs(y_bits, y_codes, 0, int(term.z), n_sites)
        diagonal 	   += np.real(complex(term.coefficient)) * sign
    return y_codes, a_frontier, diagonal


def ritz_pt2_correction(
    terms	: list[PauliTerm],
    samples	: np.ndarray,
    vector	: np.ndarray | jax.Array,
    energy	: float,
    n_sites	: int,
    eps		: float = 1.0e-9,
) -> dict[str, float]:
    r"""Return the Epstein-Nesbet correction on the truncated closure frontier.

    .. math::

        \Delta E_2=\sum_{y\notin S}
        \frac{|\langle y|H|v_0\rangle|^2}
        {\epsilon_0-\langle y|H|y\rangle}.

    States with \(H_{yy}\le\epsilon_0+\varepsilon\) are excluded as crossings.
    The corrected energy is not variational; see ``docs/pt2.md``.
    """
    frontier_limit = max(_MIN_FRONTIER_STATES, _FRONTIER_STATES_PER_BASIS * int(samples.shape[0]))
    try:
        y_codes, amplitudes, diagonal = _frontier_amplitudes(
            terms, samples, vector, n_sites, max_states=frontier_limit,
        )
    except _FrontierLimitExceeded:
        log(f"!! PT2 frontier exceeded {frontier_limit} states; correction omitted", level=2)
        return {
            "pt2": float("nan"),
            "frontier_size": float(frontier_limit),
            "coupled_weight": float("nan"),
            "skipped_crossings": float("nan"),
            "frontier_truncated": 1.0,
        }
    if y_codes.shape[0] == 0:
        return {
            "pt2": 0.0,
            "frontier_size": 0.0,
            "coupled_weight": 0.0,
            "skipped_crossings": 0.0,
            "frontier_truncated": 0.0,
        }

    denominator	= float(energy) - diagonal
    valid			= denominator < -eps
    return {
        "pt2": float(np.sum(np.abs(amplitudes[valid]) ** 2 / denominator[valid])),
        "frontier_size": float(y_codes.shape[0]),
        "coupled_weight": float(np.sum(np.abs(amplitudes) ** 2)),
        "skipped_crossings": float(np.sum(~valid)),
        "frontier_truncated": 0.0,
    }


def pt2_select_subspace(
    terms		: list[PauliTerm],
    seeds		: np.ndarray,
    n_sites		: int,
    n_subspace	: int,
    eps			: float = 1.0e-9,
    *,
    n_up		: int | None = None,
) -> np.ndarray:
    r"""Append frontier configurations ranked by Epstein-Nesbet importance.

    Crossings are admitted first. The remaining frontier is ranked by

    .. math::

        \eta_y=\frac{|\langle y|H|v_0\rangle|^2}{H_{yy}-\epsilon_0}.
    """
    seeds = np.asarray(seeds, dtype=np.int64)
    if n_up is not None and (not 0 <= n_up <= n_sites or np.any(seeds.sum(axis=1) != n_sites - n_up)):
        raise ValueError("Ritz seeds violate the requested spin sector")
    if seeds.shape[0] >= n_subspace or not terms:
        return seeds[:n_subspace]

    problem = prepare_rotation_problem(terms, [], seeds, n_sites)
    matrix = np.asarray(build_rotated_subspace_hamiltonian(problem, jnp.zeros(0)), dtype=np.complex128)
    values, vectors = np.linalg.eigh(matrix)
    energy		= float(values[0])
    frontier_limit = max(_MIN_FRONTIER_STATES, _FRONTIER_STATES_PER_BASIS * n_subspace)
    try:
        y_codes, amplitudes, diagonal = _frontier_amplitudes(
            terms, seeds, vectors[:, 0], n_sites, max_states=frontier_limit, n_up=n_up,
        )
    except _FrontierLimitExceeded:
        log(f"!! PT2 selection frontier exceeded {frontier_limit} states; filling from the NQS", level=2)
        return seeds
    if y_codes.shape[0] == 0:
        return seeds

    gap			= diagonal - energy
    crossing	= gap <= eps
    importance	= np.where(crossing, np.inf, np.abs(amplitudes) ** 2 / np.where(crossing, 1.0, gap))
    order = (
        np.lexsort((y_codes, -importance))
        if y_codes.dtype != object
        else np.asarray(sorted(range(y_codes.shape[0]), key=lambda index: (-importance[index], int(y_codes[index]))))
    )
    chosen		= y_codes[order[:n_subspace - seeds.shape[0]]]
    frontier	= _codes_to_bits(chosen, n_sites, dtype=seeds.dtype)
    selected	= np.empty((seeds.shape[0] + frontier.shape[0], n_sites), dtype=seeds.dtype)
    selected[:seeds.shape[0]] = seeds
    selected[seeds.shape[0]:] = frontier
    return selected


def build_sampled_subspace(
    terms		: list[PauliTerm],
    retained	: np.ndarray,
    candidates	: np.ndarray,
    fill		: Callable[[], np.ndarray],
    n_sites		: int,
    cap			: int,
    *,
    n_up		: int | None = None,
) -> np.ndarray:
    r"""Retain old rows and append unique ANN/PT2 rows into one capped buffer."""
    if cap < 1:
        raise ValueError("sampled-subspace cap must be positive")
    retained	= np.asarray(retained, dtype=np.int8).reshape(-1, n_sites)
    candidates	= np.asarray(candidates, dtype=np.int8).reshape(-1, n_sites)
    if np.any((retained != 0) & (retained != 1)) or np.any((candidates != 0) & (candidates != 1)):
        raise ValueError("sampled subspaces must use {0, 1} bit encoding")
    if n_up is not None and (not 0 <= n_up <= n_sites or any(np.any(rows.sum(axis=1) != n_sites - n_up) for rows in (retained, candidates))):
        raise ValueError("sampled configurations violate the requested spin sector")

    output		= np.empty((cap, n_sites), dtype=np.int8)
    seen: set[int] = set()
    count		= 0
    seed_target	= max(min(retained.shape[0], cap), (cap + 1) // 2)
    for rows, limit in ((retained, cap), (candidates, seed_target)):
        for row, code in zip(rows, _binary_codes(rows)):
            if count >= limit:
                break
            code = int(code)
            if code in seen:
                continue
            output[count] = row
            seen.add(code)
            count += 1
        if count == cap:
            return output

    if count:
        selected = pt2_select_subspace(terms, output[:count], n_sites, cap, n_up=n_up)
        for row, code in zip(selected[count:], _binary_codes(selected[count:])):
            code = int(code)
            if code in seen:
                continue
            output[count] = row
            seen.add(code)
            count += 1
            if count == cap:
                return output

    fill_rows = np.asarray(fill(), dtype=np.int8).reshape(-1, n_sites)
    if np.any((fill_rows != 0) & (fill_rows != 1)) or (n_up is not None and np.any(fill_rows.sum(axis=1) != n_sites - n_up)):
        raise ValueError("fallback configurations violate the binary spin sector")
    
    # Fill the remaining slots with unique fallback rows, if any.
    for row, code in zip(fill_rows, _binary_codes(fill_rows)):
        code = int(code)
        if code in seen:
            continue
        output[count] = row
        seen.add(code)
        count += 1
        if count == cap:
            return output
    raise RuntimeError(f"sampled-subspace construction found {count} unique states, expected {cap}")


# -------------------------
#! EOF
# -------------------------
