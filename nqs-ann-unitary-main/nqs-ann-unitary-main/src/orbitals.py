r"""Short spatial-orbital circuits and compactness of selected determinants."""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np

from .rotation_ritz import sampled_renyi2_entropies, schmidt_labels


# -------------------------


class OrbitalProjection(NamedTuple):
    """Selected Slater-Condon transitions in OpenFermion tensor ordering."""

    rows        : np.ndarray    # target determinant index in the selected basis
    columns     : np.ndarray    # source determinant index in the selected basis
    indices     : np.ndarray    # (p, q, 0, 0) for one-body, (p, q, r, s) for two-body
    signs       : np.ndarray    # fermionic sign and NetKet ordering sign of each transition
    one_count   : int           # entries of ``indices`` that belong to the one-body block
    size        : int           # dimension of the selected determinant space


def _prepare_orbital_projection(basis: np.ndarray) -> OrbitalProjection:
    """Cache only one/two-body transitions inside the selected determinant set.

    OpenFermion uses ``h[p,q] a_p^dag a_q`` and
    ``v[p,q,r,s] a_p^dag a_q^dag a_r a_s`` without an extra prefactor.
    NetKet spin-block ordering is restored by the determinant permutation sign.
    """
    n_modes = basis.shape[1]
    # Reorder to OpenFermion site-interleaved modes; keep the sign of the mode permutation.
    interleaved, permutation_sign = _site_order(basis)
    # Pack each occupied-mode bit string into one integer; the map is the row lookup.
    codes = [sum(int(bit) << mode for mode, bit in enumerate(row)) for row in interleaved]
    lookup = {code: i for i, code in enumerate(codes)}
    rows, columns, indices, signs = [], [], [], []

    def act(code, mode, create):
        # Apply a_p^dag (create) or a_q (annihilate); return ``None`` if the mode is already
        # in the required occupation. The second value is the Jordan-Wigner sign: (-1) to the
        # number of occupied modes below ``mode`` (the corresponding anticommutation parity).
        occupied = (code >> mode) & 1
        if occupied == create:
            return None
        return code ^ (1 << mode), 1 - 2 * ((code & ((1 << mode) - 1)).bit_count() % 2)

    def record(code, column, modes, sign):
        # Keep only transitions that land inside the selected space, folded with the
        # NetKet-to-OpenFermion ordering sign of source and target determinants.
        row = lookup.get(code)
        if row is not None:
            rows.append(row)
            columns.append(column)
            indices.append(modes)
            signs.append(sign * permutation_sign[row] * permutation_sign[column])

    # One-body block: a_p^dag a_q, stored as (p, q, 0, 0).
    for column, code in enumerate(codes):
        for q in range(n_modes):
            removed = act(code, q, False)
            if removed is None:
                continue
            for p in range(n_modes):
                created = act(removed[0], p, True)
                if created is not None:
                    record(created[0], column, (p, q, 0, 0), removed[1] * created[1])
    one_count = len(rows)
    # Two-body block: a_p^dag a_q^dag a_r a_s, stored as (p, q, r, s) = (create, create,
    # annihilate, annihilate). Annihilation acts right-to-left, so s is removed before r.
    for column, code in enumerate(codes):
        for s in range(n_modes):
            first = act(code, s, False)
            if first is None:
                continue
            for r in range(n_modes):
                second = act(first[0], r, False)
                if second is None:
                    continue
                for q in range(n_modes):
                    third = act(second[0], q, True)
                    if third is None:
                        continue
                    for p in range(n_modes):
                        fourth = act(third[0], p, True)
                        if fourth is not None:
                            record(fourth[0], column, (p, q, r, s), first[1] * second[1] * third[1] * fourth[1])
    return OrbitalProjection(np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32), np.asarray(indices, dtype=np.int32).reshape(-1, 4), np.asarray(signs), one_count, len(basis))


@functools.lru_cache(maxsize=16)
def _cached_orbital_projection(key: bytes, shape: tuple[int, int]) -> OrbitalProjection:
    """Reuse the transition plan for an identical determinant basis across outer steps."""
    return _prepare_orbital_projection(np.frombuffer(key, dtype=np.int8).reshape(shape))


def prepare_orbital_projection(basis: np.ndarray) -> OrbitalProjection:
    """Return a cached transition plan for a {0, 1} determinant basis.

    The plan depends only on the basis, so repeated calls with the same rows
    reuse the pure-Python transition scan; see :func:`_prepare_orbital_projection`.
    """
    basis = np.ascontiguousarray(basis, dtype=np.int8)
    return _cached_orbital_projection(basis.tobytes(), basis.shape)


def projected_orbital_matrix(interaction, plan: OrbitalProjection) -> np.ndarray:
    """Contract rotated integrals into a cached selected space, without operators."""
    i = plan.one_count
    one, two = plan.indices[:i], plan.indices[i:]
    # Gather h[p, q] for one-body and v[p, q, r, s] for two-body transitions, weighted by sign.
    values = np.concatenate((interaction.one_body_tensor[one[:, 0], one[:, 1]], interaction.two_body_tensor[tuple(two.T)])) * plan.signs
    # Start from the constant (nuclear/chemical) shift; scattering adds the transitions.
    matrix = np.eye(plan.size, dtype=complex) * interaction.constant
    np.add.at(matrix, (plan.rows, plan.columns), values)
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.conj().T, atol=1e-9):
        raise ArithmeticError("rotated integral projection is non-finite or non-Hermitian")
    # Symmetrize to remove round-off asymmetry before eigen-solving.
    return (matrix + matrix.conj().T) / 2


def orbital_pairs(n_orbitals: int, n_layers: int) -> tuple[tuple[int, int], ...]:
    """One layer contains even then odd nearest-neighbor Givens gates."""
    # parity 0 covers (0,1), (2,3), ...; parity 1 covers the shifted (1,2), (3,4), ...
    return tuple((i, i + 1) for _ in range(n_layers) for parity in (0, 1) for i in range(parity, n_orbitals - 1, 2))


def orbital_matrix(n_orbitals: int, pairs: tuple, theta: np.ndarray) -> np.ndarray:
    r"""Return ``R = G_0 G_1 ...``, with ``G_ij = [[cos, sin], [-sin, cos]]``.

    OpenFermion transforms the integrals by this real orthogonal matrix. Only
    two columns are updated per gate; no Fock-space unitary is constructed.
    """
    if np.shape(theta) != (len(pairs),) or not np.all(np.isfinite(theta)):
        raise ValueError("orbital angles must be finite and match the circuit")
    rotation = np.eye(n_orbitals)
    for (i, j), angle in zip(pairs, theta):
        # Multiply on the right by G_ij: mix the current columns i and j, leaving the rest.
        c, s = np.cos(angle), np.sin(angle)
        a, b = rotation[:, i].copy(), rotation[:, j].copy()
        rotation[:, i], rotation[:, j] = c * a - s * b, s * a + c * b
    return rotation


# -------------------------
# Internal helpers
# -------------------------


def _site_order(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interleave spin blocks and return the occupied-mode permutation sign."""
    n_orbitals = basis.shape[1] // 2
    up, down = basis[:, :n_orbitals], basis[:, n_orbitals:]
    # ``N_up - cumsum(up)[i]`` counts up modes above i, so the sum over occupied down modes
    # is sum_{i<j} n_down[i] n_up[j]: the swap count that interleaves the two spin blocks.
    swaps = np.sum(down * (np.sum(up, axis=1, keepdims=True) - np.cumsum(up, axis=1)), axis=1)
    sites = np.stack((up, down), axis=-1).reshape(basis.shape)
    return sites, 1 - 2 * (swaps % 2)


def orbital_cut_maps(basis: np.ndarray, order: str = "spatial") -> tuple[tuple, np.ndarray]:
    r"""Selected-state Schmidt labels and log-rank scales in spatial-orbital or NQS spin-block order.

    Spatial cuts group both spins and include the fermionic permutation sign
    ``(-1)**sum_{i<j} n_down[i] n_up[j]``. Model cuts use the original mode
    order, including the up/down boundary, without a mode permutation.
    """
    if order == "spatial":
        bits, sign = _site_order(basis)
        step = 2
    elif order == "model":
        bits, sign = basis, np.ones(len(basis), dtype=np.int8)
        step = 1
    else:
        raise ValueError("entanglement_order must be spatial or model")
    # Balance every bipartition; spatial cuts fall on site boundaries (step 2), model cuts on modes.
    cuts    = range(step, basis.shape[1], step)
    # Max Renyi-2 entropy of a cut is log of the smaller block dimension (capped by the basis size).
    scales  = np.asarray([np.log(max(2, min(len(basis), 2 ** min(cut, basis.shape[1] - cut)))) for cut in cuts])
    return (schmidt_labels(bits, cuts), scales), sign


def cut_renyi2(vector: np.ndarray, maps: tuple, sign: np.ndarray) -> float:
    """Normalized mean cut entropy of the zero-extended selected state.

    Sparse Schmidt matrices keep the memory independent of the full sector
    dimension; see :func:`rotation_ritz.sampled_renyi2_entropies`.
    """
    labels, scales = maps
    # Fold in the ordering sign, sum the per-cut Renyi-2 entropies, then normalize by the summed maxima.
    return float(np.sum(sampled_renyi2_entropies(vector * sign, labels)) / np.sum(scales)) if labels else 0.


def orbital_ritz_objective(
    matrix                  : np.ndarray,
    energy_scale            : float,
    participation_weight    : float = 0.,
    entanglement_weight     : float = 0.,
    *,
    cut_maps                : tuple | None = None,
) -> tuple[float, dict]:
    """Energy plus signed normalized Renyi-2 participation and configured cut entropy.

    Participation entropy is ``-log(IPR)``, so positive weight favors a compact
    determinant expansion. These are selected-state proxies, not NQS entropies.
    """
    values, vectors = np.linalg.eigh(matrix)
    # Ground Ritz vector; its Born probabilities p_i = |v_i|^2 determine compactness.
    probability = np.abs(vectors[:, 0]) ** 2
    ipr = float(np.sum(probability ** 2))
    # Participation entropy -log(IPR), normalized by the maximal value log(dim).
    participation = -np.log(ipr) / np.log(len(matrix)) if len(matrix) > 1 else 0.
    # Entanglement is expensive, so evaluate it only when its weight is nonzero.
    entanglement = cut_renyi2(vectors[:, 0], *cut_maps) if entanglement_weight else 0.
    loss = values[0] / energy_scale + participation_weight * participation + entanglement_weight * entanglement
    return float(loss), {
        "ritz_energy"           : float(values[0]),
        "ipr"                   : ipr,
        "participation_entropy" : float(participation),
        "entanglement_entropy"  : entanglement,
    }
