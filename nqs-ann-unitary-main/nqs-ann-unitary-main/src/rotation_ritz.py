r"""Differentiable sampled-Ritz optimization on one prepared spin subspace.

For retained bit strings :math:`S=\{x_i\}_{i=1}^M`, this module prepares only
the angle-independent Pauli orbit, subspace connectivity, and ordered spin-cut
indices. Every optimizer step still reconstructs

.. math::

    (H_S(\theta))_{ij}=\langle x_i|U_\theta^\dagger H U_\theta|x_j\rangle

and recomputes its Ritz state. The optional participation, entanglement, and
sampled-sign terms have signed weights: positive values penalize a metric,
negative values reward it, and exact zero disables its runtime evaluation.
"""

from __future__ import annotations

import dataclasses  as dc
from functools      import lru_cache
from typing         import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from scipy.sparse   import csr_matrix
from tqdm.auto      import trange

from .config        import RotationConfig
from .logging_utils import interruptible, log, progress_disabled
from .operators     import PauliTerm, RotationGate, _anticommutes, _multiply_pauli


_RELATIVE_GAP_FLOOR = 1.0e-3
_EPS                = 1.0e-12


class SubspaceStructure(NamedTuple):
    """Angle-independent scatter pattern for the projected Hamiltonian."""

    rows		: jax.Array
    cols		: jax.Array
    term_index	: jax.Array
    geometry	: jax.Array
    size		: int


class CoefficientStep(NamedTuple):
    """One anticommuting generator step in coefficient propagation."""

    parameter	: int
    multiplier	: float
    source		: np.ndarray
    target		: np.ndarray
    phase		: np.ndarray


class CoefficientPlan(NamedTuple):
    """Reachable Pauli strings and their vectorized propagation plan."""

    keys		: tuple[tuple[int, int], ...]
    base_index	: np.ndarray
    base_coeff	: np.ndarray
    steps		: tuple[CoefficientStep, ...]


class EntanglementPairs(NamedTuple):
    """Pair-scatter indices for every ordered spin cut.

    For a cut :math:`A|B`, ``pair_index[i,j]`` is the flattened compact
    ``(a_i, a_j)`` entry of :math:`\rho_A` and ``pair_valid[i,j]`` records
    whether :math:`b_i=b_j`. These arrays depend only on the retained samples.
    """

    pair_index	: jax.Array	# (N - 1, M * M)
    pair_valid	: jax.Array	# (N - 1, M * M)
    normalization	: jax.Array	# maximum entropy of each cut
    n_cuts		: int


@jax.tree_util.register_pytree_node_class
@dc.dataclass(frozen=True, eq=False)
class RotationProblem:
    r"""One fixed sampled-subspace rotation problem.

    This record contains no angle-dependent Pauli coefficients, Ritz matrix, or
    eigenvector. It may be reused while ``samples`` are exactly unchanged; a
    different retained basis requires :func:`prepare_rotation_problem` again.
    """

    base_terms	    : tuple[PauliTerm, ...]
    gates		    : tuple[RotationGate, ...]
    samples		    : np.ndarray
    n_sites		    : int
    n_parameters	: int
    structure	    : SubspaceStructure
    plan		    : CoefficientPlan
    entanglement	: EntanglementPairs
    j_rms		    : float
    energy_scale	: float

    def tree_flatten(self):
        """Expose numerical data dynamically while keeping only topology static."""
        children = (
            self.structure.rows,
            self.structure.cols,
            self.structure.term_index,
            self.structure.geometry,
            jnp.asarray(self.plan.base_index),
            jnp.asarray(self.plan.base_coeff),
            *tuple(item for step in self.plan.steps for item in (jnp.asarray(step.source), jnp.asarray(step.target), jnp.asarray(step.phase))),
            self.entanglement.pair_index,
            self.entanglement.pair_valid,
            self.entanglement.normalization,
            jnp.asarray(self.j_rms),
            jnp.asarray(self.energy_scale),
        )
        auxiliary = (
            self.n_sites,
            self.n_parameters,
            self.samples.shape,
            self.structure.size,
            self.plan.keys,
            tuple((step.parameter, step.multiplier) for step in self.plan.steps),
            self.entanglement.n_cuts,
        )
        return children, auxiliary

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        """Reconstruct the traced problem without freezing numerical values."""
        n_sites, n_parameters, sample_shape, size, keys, step_metadata, n_cuts = auxiliary
        iterator                = iter(children)
        samples                 = np.empty(sample_shape, dtype=np.int8)
        structure               = SubspaceStructure(next(iterator), next(iterator), next(iterator), next(iterator), size)
        base_index, base_coeff  = next(iterator), next(iterator)
        steps                   = tuple(CoefficientStep(parameter, multiplier, next(iterator), next(iterator), next(iterator)) for parameter, multiplier in step_metadata)
        entanglement            = EntanglementPairs(next(iterator), next(iterator), next(iterator), n_cuts)
        j_rms, energy_scale     = next(iterator), next(iterator)
        return cls((), (), samples, n_sites, n_parameters, structure, CoefficientPlan(keys, base_index, base_coeff, steps), entanglement, j_rms, energy_scale)


class RotationDiagnostics(NamedTuple):
    """Pre-update objective values from one rotation step."""

    loss					: float
    ritz_energy			    : float
    ritz_gap				: float
    participation_entropy	: float
    entanglement			: float
    nonstoquasticity		: float
    nested_loss				: float


class RitzState(NamedTuple):
    """Lowest sampled-subspace eigenstate and first excitation gap."""

    energy	: jax.Array
    vector	: jax.Array
    gap		: jax.Array


# -------------------------


def _binary_codes(samples: np.ndarray) -> np.ndarray:
    """Encode bit rows as integers without imposing a 64-site limit."""
    samples = np.asarray(samples, dtype=np.uint8)
    if samples.ndim != 2:
        raise ValueError("binary samples must be a matrix")
    n_sites = samples.shape[1]
    if n_sites <= 63:
        return samples.astype(np.int64) @ (1 << np.arange(n_sites, dtype=np.int64))
    packed = np.packbits(samples, axis=1, bitorder="little")
    return np.asarray([int.from_bytes(row.tobytes(), byteorder="little") for row in packed], dtype=object)


def _codes_to_bits(codes: np.ndarray, n_sites: int, dtype=np.int8) -> np.ndarray:
    """Decode integer configuration keys into little-endian bit rows."""
    codes = np.asarray(codes)
    if n_sites <= 63 and codes.dtype != object:
        return ((codes[:, None] >> np.arange(n_sites, dtype=np.int64)[None, :]) & 1).astype(dtype)
    n_bytes = (n_sites + 7) // 8
    rows    = np.empty((codes.shape[0], n_sites), dtype=dtype)
    for index, code in enumerate(codes):
        packed      = np.frombuffer(int(code).to_bytes(n_bytes, byteorder="little"), dtype=np.uint8)
        rows[index] = np.unpackbits(packed, bitorder="little")[:n_sites]
    return rows


def _connected_codes_and_signs(samples: np.ndarray, codes: np.ndarray, x_mask: int, z_mask: int, n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply one Pauli bit flip and return its diagonal parity signs."""
    if n_sites <= 63:
        x_bits      = (x_mask >> np.arange(n_sites, dtype=np.int64)) & 1
        z_bits      = (z_mask >> np.arange(n_sites, dtype=np.int64)) & 1
        connected   = np.bitwise_xor(samples, x_bits[None, :]) @ (1 << np.arange(n_sites, dtype=np.int64))
        sign        = 1 - 2 * ((samples & z_bits[None, :]).sum(axis=1) & 1)
        return connected, sign
    connected   = np.asarray([int(code) ^ int(x_mask) for code in codes], dtype=object)
    sign        = np.fromiter((1 - 2 * ((int(code) & int(z_mask)).bit_count() & 1) for code in codes), dtype=np.int8, count=codes.shape[0])
    return connected, sign


@lru_cache(maxsize=32)
def _coefficient_topology(base_keys: tuple[tuple[int, int], ...], gates: tuple[RotationGate, ...], n_sites: int) -> tuple[tuple[tuple[int, int], ...], np.ndarray, tuple[CoefficientStep, ...]]:
    """Cache reachable strings and propagation indices without coefficients."""
    keys                            = list(dict.fromkeys(base_keys))
    key_to_index                    = {key: index for index, key in enumerate(keys)}
    base_index                      = np.asarray([key_to_index[key] for key in base_keys], dtype=np.int32)
    steps: list[CoefficientStep]    = []
    for gate in reversed(gates):
        for g_x, g_z, multiplier in gate.generators:
            source: list[int]       = []
            target: list[int]       = []
            phase: list[complex]    = []
            for p_x, p_z in tuple(keys):
                if not _anticommutes(g_x, g_z, p_x, p_z):
                    continue
 
                # if the generator anticommutes with the Pauli term, compute the resulting Pauli term and its phase
                local_phase, q_x, q_z   = _multiply_pauli(g_x, g_z, p_x, p_z, n_sites)
                key                     = (q_x, q_z)
                if key not in key_to_index:
                    key_to_index[key] = len(keys)
                    keys.append(key)
                source.append(key_to_index[(p_x, p_z)])
                target.append(key_to_index[key])
                phase.append(local_phase)
            steps.append(CoefficientStep(int(gate.parameter), float(multiplier), np.asarray(source, dtype=np.int32), np.asarray(target, dtype=np.int32), np.asarray(phase, dtype=np.complex128)))
    return tuple(keys), base_index, tuple(steps)


def _coefficient_plan(base_terms: tuple[PauliTerm, ...], gates: tuple[RotationGate, ...], n_sites: int) -> CoefficientPlan:
    """Combine cached topology with the current dynamic Pauli coefficients."""
    base_keys               = tuple((int(term.x), int(term.z)) for term in base_terms)
    keys, base_index, steps = _coefficient_topology(base_keys, gates, n_sites)
    base_coeff              = np.asarray([complex(term.coefficient) for term in base_terms], dtype=np.complex128)
    return CoefficientPlan(keys, base_index, base_coeff, steps)


def _subspace_structure(plan: CoefficientPlan, samples: np.ndarray, n_sites: int) -> SubspaceStructure:
    """Precompute Pauli connectivity within one retained sample set."""
    samples                         = np.asarray(samples, dtype=np.int64)
    size                            = samples.shape[0]
    codes                           = _binary_codes(samples)
    order                           = np.argsort(codes)
    sorted_codes                    = codes[order]
    all_rows                        = np.arange(size)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    term_index: list[np.ndarray] = []
    geometry: list[np.ndarray] = []
    for index, (x_mask, z_mask) in enumerate(plan.keys):
        connected, sign = _connected_codes_and_signs(samples, codes, x_mask, z_mask, n_sites)
        slot            = np.searchsorted(sorted_codes, connected)
        slot_clipped    = np.clip(slot, 0, size - 1)
        hit             = sorted_codes[slot_clipped] == connected
        rows.append(np.where(hit, order[slot_clipped], 0))
        cols.append(all_rows)
        term_index.append(np.full(size, index, dtype=np.int32))
        geometry.append((1.0j) ** int((x_mask & z_mask).bit_count()) * sign * hit)
    empty_i = np.zeros(0, dtype=np.int32)
    empty_c = np.zeros(0, dtype=np.complex128)
    return SubspaceStructure(
        jnp.asarray(np.concatenate(rows) if rows else empty_i, dtype=jnp.int32),
        jnp.asarray(np.concatenate(cols) if cols else empty_i, dtype=jnp.int32),
        jnp.asarray(np.concatenate(term_index) if term_index else empty_i, dtype=jnp.int32),
        jnp.asarray(np.concatenate(geometry) if geometry else empty_c, dtype=complex),
        size,
    )


def _block_indices(samples: np.ndarray, cut: int) -> tuple[np.ndarray, np.ndarray]:
    """Compact subsystem labels for the ordered cut ``[:cut] | [cut:]``."""
    _, a_idx = np.unique(_binary_codes(samples[:, :cut]), return_inverse=True)
    _, b_idx = np.unique(_binary_codes(samples[:, cut:]), return_inverse=True)
    return a_idx.astype(np.int32), b_idx.astype(np.int32)


# -------------------------


def entanglement_pairs_all(samples: np.ndarray, n_sites: int) -> EntanglementPairs:
    r"""Prepare pair-scatter indices for all :math:`N-1` ordered spin cuts.

    No Hilbert-space-sized tensor is created. Each cut stores two arrays of
    length :math:`M^2`; the differentiable calculation later uses one
    :math:`M^2` density-matrix buffer per ``lax.scan`` iteration.
    """
    samples                         = np.asarray(samples, dtype=np.int8)
    size                            = samples.shape[0]
    pair_index: list[np.ndarray]    = []
    pair_valid: list[np.ndarray]    = []
    normalization: list[float]      = []
    for cut in range(1, n_sites):
        a_idx, b_idx 	= _block_indices(samples, cut)
        pair_index.append((a_idx[:, None] * size + a_idx[None, :]).reshape(-1))
        pair_valid.append((b_idx[:, None] == b_idx[None, :]).reshape(-1))
        max_rank 		= min(size, 2 ** min(cut, n_sites - cut))
        normalization.append(float(np.log(max_rank)) if max_rank > 1 else 1.0)
    shape = (0, size * size)
    return EntanglementPairs(
        jnp.asarray(np.stack(pair_index) if pair_index else np.zeros(shape, dtype=np.int32), dtype=jnp.int32),
        jnp.asarray(np.stack(pair_valid) if pair_valid else np.zeros(shape, dtype=bool), dtype=bool),
        jnp.asarray(normalization, dtype=float),
        max(n_sites - 1, 0),
    )


def hamiltonian_j_rms(base_terms: list[PauliTerm] | tuple[PauliTerm, ...], n_sites: int) -> float:
    r"""Return :math:`J_{\rm rms}=\sqrt{\sum_{P\ne I}|c_P|^2/N}`.

    This is a Pauli-coefficient normalization, not a universal exchange scale.
    The identity is excluded because it changes neither states nor gradients.
    """
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    squared = sum(abs(complex(term.coefficient)) ** 2 for term in base_terms if int(term.x) != 0 or int(term.z) != 0)
    scale = float(np.sqrt(squared / n_sites))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("rotation optimization requires a nontrivial finite Hamiltonian (J_rms > 0)")
    return scale


# -------------------------
# Problem preparation and Hamiltonian assembly
# -------------------------


def prepare_rotation_problem(base_terms: list[PauliTerm], gates: list[RotationGate], samples: np.ndarray, n_sites: int, *, include_entanglement: bool = False) -> RotationProblem:
    r"""Prepare all angle-independent data for one retained Ritz basis.

    The returned problem is valid for every angle and signed objective weight,
    but only for these exact sample rows and their order. Pair-scatter arrays
    are built only when requested; entropy-weighted optimization prepares them
    on demand. Default energy-only work avoids their O(N M^2) storage.
    """
    samples = np.asarray(samples, dtype=np.int8)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != n_sites:
        raise ValueError("rotation samples must be a non-empty (M, n_sites) bit matrix")
    if np.any((samples != 0) & (samples != 1)):
        raise ValueError("rotation samples must use {0, 1} bit encoding")
    if np.unique(samples, axis=0).shape[0] != samples.shape[0]:
        raise ValueError("rotation samples must be unique")

    terms   = tuple(base_terms)
    circuit = tuple(gates)
    plan    = _coefficient_plan(terms, circuit, int(n_sites))
    j_rms   = hamiltonian_j_rms(terms, n_sites)
    return RotationProblem(terms, circuit, samples.copy(), int(n_sites),
        0 if not circuit else max(gate.parameter for gate in circuit) + 1,
        _subspace_structure(plan, samples, n_sites), plan, entanglement_pairs_all(samples, n_sites if include_entanglement else 1),
        j_rms, 1.0 / (n_sites * j_rms),
    )


def _rotated_coefficients(problem: RotationProblem, theta: jax.Array) -> jax.Array:
    """Propagate rotated Pauli coefficients in the prepared orbit order."""
    dtype = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    coefficients = jnp.zeros((len(problem.plan.keys),), dtype=dtype)
    coefficients = coefficients.at[problem.plan.base_index].add(problem.plan.base_coeff)
    for step in problem.plan.steps:
        values          = coefficients[step.source]
        angle           = 2.0 * step.multiplier * theta[step.parameter]
        coefficients    = coefficients.at[step.source].set(values * jnp.cos(angle))
        coefficients    = coefficients.at[step.target].add(values * (1.0j * step.phase) * jnp.sin(angle))
    return coefficients


def _assemble_from_coefficients(structure: SubspaceStructure, coefficients: jax.Array) -> jax.Array:
    """Assemble and explicitly Hermitize the projected Hamiltonian."""
    contributions   = coefficients[structure.term_index] * structure.geometry
    dtype           = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    matrix          = jnp.zeros((structure.size, structure.size), dtype=dtype)
    matrix          = matrix.at[structure.rows, structure.cols].add(contributions)
    return 0.5 * (matrix + jnp.conj(matrix.T))


def build_rotated_subspace_hamiltonian(problem: RotationProblem, theta: jax.Array) -> jax.Array:
    r"""Build :math:`H_S(\theta)` from a prepared fixed-subspace problem."""
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    theta = jnp.asarray(theta, dtype=dtype)
    if theta.shape != (problem.n_parameters,):
        raise ValueError(f"rotation theta has shape {theta.shape}, expected {(problem.n_parameters,)}")
    return _assemble_from_coefficients(problem.structure, _rotated_coefficients(problem, theta))


# -------------------------
# Differentiable Ritz state with regularized perturbative JVP
# -------------------------


@jax.custom_jvp
def _ground_state(matrix: jax.Array, j_rms: jax.Array) -> RitzState:
    """Hermitian lowest Ritz state with a regularized perturbative JVP."""
    values, vectors = jnp.linalg.eigh(0.5 * (matrix + jnp.conj(matrix.T)))
    gap             = values[1] - values[0] if values.shape[0] > 1 else jnp.zeros_like(values[0])
    return RitzState(jnp.real(values[0]), vectors[:, 0], jnp.real(gap))


@_ground_state.defjvp
def _ground_state_jvp(primals, tangents):
    r"""Differentiate the complex Ritz vector with a finite gap response.

    For :math:`dH`, first-order perturbation theory gives

    .. math::

        dv_0=\sum_{n>0}v_n\frac{v_n^\dagger(dH)v_0}{\epsilon_0-\epsilon_n}.

    We replace the inverse gap by
    :math:`g/(g^2+\delta^2)`, where
    :math:`g=\epsilon_0-\epsilon_n` and ``delta`` is a relative physical scale.
    This preserves the complex phase response away from crossings and keeps
    state-dependent auxiliary gradients finite near a sampled degeneracy.
    """
    matrix, j_rms   = primals
    dmatrix, _      = tangents
    hermitian       = 0.5 * (matrix + jnp.conj(matrix.T))
    dhermitian      = 0.5 * (dmatrix + jnp.conj(dmatrix.T))
    values, vectors = jnp.linalg.eigh(hermitian)
    v0              = vectors[:, 0]
    scale           = jnp.maximum(jnp.real(values[-1] - values[0]), jnp.abs(jnp.real(j_rms)))
    delta           = _RELATIVE_GAP_FLOOR * jnp.maximum(scale, jnp.finfo(values.dtype).eps)
    gaps            = values[0] - values
    inverse_gaps    = (gaps / (gaps * gaps + delta * delta)).at[0].set(0.0)
    overlaps        = jnp.conj(vectors).T @ (dhermitian @ v0)
    dvector         = vectors @ (inverse_gaps * overlaps)
    dvalues         = jnp.real(jnp.diag(jnp.conj(vectors).T @ dhermitian @ vectors))
    gap             = values[1] - values[0] if values.shape[0] > 1 else jnp.zeros_like(values[0])
    dgap            = dvalues[1] - dvalues[0] if values.shape[0] > 1 else jnp.zeros_like(values[0])
    return RitzState(jnp.real(values[0]), v0, jnp.real(gap)), RitzState(dvalues[0], dvector, dgap)


def ritz_ground_state(matrix: jax.Array, j_rms: float | jax.Array = 1.0) -> RitzState:
    """Return the differentiable lowest Ritz state of a Hermitian matrix."""
    return _ground_state(matrix, jnp.asarray(j_rms, dtype=jnp.real(matrix).dtype))


def compute_participation_entropy(vector: jax.Array, eps: float = _EPS) -> jax.Array:
    r"""Return :math:`S_{\rm part}=-\log\sum_i p_i^2` in the retained basis.

    This measures compactness only inside the selected sampled basis. It does not
    show that the wavefunction weight outside that basis is negligible.
    """
    weights 		= jnp.abs(vector) ** 2
    probabilities 	= weights / jnp.maximum(jnp.sum(weights), eps)
    return -jnp.log(jnp.maximum(jnp.sum(probabilities * probabilities), eps))


@jax.jit
def compute_renyi2_entropy(vector: jax.Array, pairs: EntanglementPairs, eps: float = _EPS) -> jax.Array:
    r"""Return the sum of zero-extended sampled-state Rényi-2 cut entropies.

    For each ordered spin cut, pair scattering constructs
    :math:`(\rho_A)_{aa'}=\sum_b\psi_{ab}\psi^*_{a'b}` using one
    :math:`M^2` buffer. This is the exact reduced density matrix of the state
    whose amplitudes are ``vector`` on the retained bit strings and zero elsewhere.
    """
    if pairs.pair_index.shape[0] == 0:
        return jnp.zeros((), dtype=jnp.real(vector).dtype)

    # reshape outer product to match the flattened pair indices, then sum over valid pairs
    outer 			= (vector[:, None] * jnp.conj(vector[None, :])).reshape(-1)
    norm_squared 	= jnp.maximum(jnp.sum(jnp.abs(vector) ** 2) ** 2, eps)

    def cut_entropy(total, data):
        indices, valid 	= data
        values 			= jnp.where(valid, outer, jnp.zeros((), dtype=outer.dtype))
        rho 			= jnp.zeros((outer.shape[0],), dtype=vector.dtype).at[indices].add(values)
        purity 			= jnp.sum(jnp.abs(rho) ** 2) / norm_squared
        return total - jnp.log(jnp.maximum(purity, eps)), None

    total, _ = jax.lax.scan(cut_entropy, jnp.zeros((), dtype=jnp.real(vector).dtype), (pairs.pair_index, pairs.pair_valid))
    return total


def schmidt_labels(samples: np.ndarray, cuts) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    r"""Compact Schmidt labels :math:`(a_i, b_i)` of every unique row for the cuts ``[:c] | [c:]``.

    For :math:`n_b` rows with right label :math:`b`, forming :math:`SS^\dagger`
    costs :math:`\sum_b n_b^2`. The zero-extended state is pure, so
    :math:`\mathrm{Tr}\rho_A^2=\mathrm{Tr}\rho_B^2`, and each pair is ordered to
    make :math:`SS^\dagger` the cheaper Gram product.
    """
    labels = []
    for cut in cuts:
        a_idx, b_idx = _block_indices(samples, cut)
        if np.sum(np.bincount(b_idx) ** 2) > np.sum(np.bincount(a_idx) ** 2):
            a_idx, b_idx = b_idx, a_idx
        labels.append((a_idx, b_idx))
    return tuple(labels)


def sampled_renyi2_entropies(vector: np.ndarray, labels: tuple[tuple[np.ndarray, np.ndarray], ...]) -> np.ndarray:
    r"""Per-cut Rényi-2 entropies :math:`-\log\mathrm{Tr}\rho_A^2` of the zero-extended selected state.

    Row :math:`i` places :math:`\psi_i/\|\psi\|` at :math:`S_{a_i b_i}`, so
    :math:`\rho_A=SS^\dagger`. Sparse Schmidt matrices need :math:`O(M)` storage
    per cut plus the Gram product, instead of the :math:`O(NM^2)` pair arrays of
    :func:`compute_renyi2_entropy`. This NumPy form is not differentiable; it
    serves reporting and SciPy objectives.
    """
    psi			= np.asarray(vector, dtype=np.complex128).reshape(-1)
    psi			= psi / np.linalg.norm(psi)
    entropies	= np.empty(len(labels))
    for index, (a_idx, b_idx) in enumerate(labels):
        schmidt				= csr_matrix((psi, (a_idx, b_idx)), shape=(int(a_idx.max()) + 1, int(b_idx.max()) + 1))
        purity				= np.sum(np.abs((schmidt @ schmidt.conj().T).data) ** 2)
        entropies[index]	= -np.log(np.clip(purity, np.finfo(float).tiny, 1.0))
    return entropies


def sampled_nonstoquasticity(matrix: jax.Array) -> jax.Array:
    r"""Return a homogeneous sign/phase obstruction on the sampled matrix.

    The ratio counts positive-real and imaginary off-diagonal weight. It depends
    on computational-basis gauge and selected subspace, so it is a proxy rather
    than a global stoquasticity certificate.
    """
    mask            = 1.0 - jnp.eye(matrix.shape[0], dtype=jnp.real(matrix).dtype)
    off_diagonal    = matrix * mask
    numerator       = jnp.sum(jax.nn.relu(jnp.real(off_diagonal)) ** 2 + jnp.imag(off_diagonal) ** 2)
    denominator     = jnp.sum(jnp.abs(off_diagonal) ** 2)
    return jnp.where(denominator > 0.0, numerator / jnp.where(denominator > 0.0, denominator, 1.0), 0.0)


# -------------------------


def _objective(problem: RotationProblem, theta: jax.Array, weights: jax.Array, state_dependent: bool, has_nested: bool = False, nested_indices: jax.Array | None = None) -> tuple[jax.Array, RotationDiagnostics]:
    """Evaluate normalized Ritz energy, active signed auxiliaries, and nested Ritz loss."""
    matrix = build_rotated_subspace_hamiltonian(problem, theta)
    if state_dependent or has_nested:
        state           = ritz_ground_state(matrix, problem.j_rms)
        energy, gap     = state.energy, state.gap
        zero            = jnp.zeros((), dtype=energy.dtype)
        participation   = jax.lax.cond(
                            weights[0] != 0.0,
                            lambda _: compute_participation_entropy(state.vector) / (jnp.log(problem.structure.size) if problem.structure.size > 1 else 1.0),
                            lambda _: zero,
                            operand=None,
                        )
        entanglement    = jax.lax.cond(
                            weights[1] != 0.0,
                            lambda _: compute_renyi2_entropy(state.vector, problem.entanglement) / jnp.maximum(jnp.sum(problem.entanglement.normalization), _EPS),
                            lambda _: zero,
                            operand=None,
                        )
    else:
        values          = jnp.linalg.eigvalsh(matrix)
        energy          = jnp.real(values[0])
        gap             = jnp.real(values[1] - values[0]) if values.shape[0] > 1 else jnp.zeros_like(energy)
        participation   = jnp.zeros_like(energy)
        entanglement    = jnp.zeros_like(energy)

    nonstoquasticity    = jax.lax.cond(weights[2] != 0.0, lambda _: sampled_nonstoquasticity(matrix), lambda _: jnp.zeros_like(energy), operand=None)

    if has_nested and nested_indices is not None and nested_indices.shape[0] > 0:
        submatrix       = matrix[nested_indices[:, None], nested_indices[None, :]]
        nested_state    = ritz_ground_state(submatrix, problem.j_rms)
        nested_loss     = problem.energy_scale * (nested_state.energy - energy)
    else:
        nested_loss     = jnp.zeros_like(energy)

    loss                = energy * jnp.asarray(problem.energy_scale, dtype=energy.dtype) + weights[0] * participation + weights[1] * entanglement + weights[2] * nonstoquasticity + (weights[3] * nested_loss if has_nested else 0.0)
    return loss, RotationDiagnostics(loss, energy, gap, participation, entanglement, nonstoquasticity, nested_loss)


def _kernel_signature(problem: RotationProblem, state_dependent: bool, has_nested: bool = False, nested_size: int = 0) -> tuple:
    """Hash only topology and compiled array dimensions, never optimizer scalars."""
    return (
        problem.n_sites,
        problem.structure.size,
        problem.n_parameters,
        len(problem.plan.keys),
        tuple((step.parameter, step.multiplier, step.source.shape[0]) for step in problem.plan.steps),
        state_dependent,
        has_nested,
        nested_size,
        "complex128",
    )


@lru_cache(maxsize=32)
def _rotation_kernel_factory(signature: tuple):
    """Return one jitted Adam step for a static topology and objective path."""
    state_dependent = bool(signature[-4])
    has_nested 		= bool(signature[-3])
    optimizer       = optax.scale_by_adam()

    @jax.jit
    def step(problem: RotationProblem, theta: jax.Array, state, weights: jax.Array, learning_rate: jax.Array, grad_clip: jax.Array, nested_indices: jax.Array | None = None):
        (_, diagnostics), gradient = jax.value_and_grad(_objective, argnums=1, has_aux=True)(
            problem, theta, weights, state_dependent, has_nested, nested_indices if has_nested else None
        )
        norm        	= jnp.linalg.norm(gradient)
        clip_scale  	= jnp.where((grad_clip > 0.0) & (norm > grad_clip), grad_clip / jnp.maximum(norm, _EPS), 1.0)
        gradient    	= gradient * clip_scale
        updates, state 	= optimizer.update(gradient, state, theta)
        return optax.apply_updates(theta, -learning_rate * updates), state, diagnostics

    return step


def _weights(cfg: RotationConfig) -> jax.Array:
    """Materialize signed objective weights as one dynamic device vector."""
    return jnp.asarray((cfg.participation_weight, cfg.entanglement_weight, cfg.nonstoquasticity_weight, cfg.nested_weight), dtype=float)


def _to_float(diagnostics: RotationDiagnostics, cfg: RotationConfig) -> RotationDiagnostics:
    """Copy diagnostics to host and mark disabled objectives as ``NaN``."""
    row = RotationDiagnostics(*(float(value) for value in diagnostics))
    return row._replace(
        participation_entropy	= row.participation_entropy if cfg.participation_weight != 0.0 else float("nan"),
        entanglement			= row.entanglement if cfg.entanglement_weight != 0.0 else float("nan"),
        nonstoquasticity		= row.nonstoquasticity if cfg.nonstoquasticity_weight != 0.0 else float("nan"),
        nested_loss				= row.nested_loss if cfg.nested_weight != 0.0 else float("nan"),
    )


def evaluate_rotation_diagnostics(problem: RotationProblem, theta: jax.Array, cfg: RotationConfig) -> RotationDiagnostics:
    """Evaluate the configured rotation objective without applying an update."""
    if cfg.entanglement_weight != 0.0 and problem.entanglement.n_cuts == 0:
        problem = dc.replace(problem, entanglement=entanglement_pairs_all(problem.samples, problem.n_sites))
        
    state_dependent = cfg.participation_weight != 0.0 or cfg.entanglement_weight != 0.0
    has_nested 		= cfg.nested_weight != 0.0
    nested_indices 	= None
    if has_nested:	
        matrix_0 		= build_rotated_subspace_hamiltonian(problem, theta)
        state_0 		= ritz_ground_state(matrix_0, problem.j_rms)
        importance 		= np.abs(np.asarray(state_0.vector)) ** 2
        m_half 			= max(1, problem.structure.size // 2)
        nested_idx 		= np.sort(np.argsort(-importance)[:m_half])
        nested_indices 	= jnp.asarray(nested_idx, dtype=jnp.int32)
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    _, diagnostics 	= _objective(problem, jnp.asarray(theta, dtype=dtype), _weights(cfg), state_dependent, has_nested, nested_indices)
    return _to_float(diagnostics, cfg)


# ------------------------------------------------------------
# Differentiable rotation optimization with dynamic signed weights
# ------------------------------------------------------------


def optimize_rotation_angles(problem: RotationProblem, cfg: RotationConfig, theta0: jax.Array | None = None, show_progress: bool = True, on_step: Callable[[int, jax.Array], None] | None = None) -> tuple[jax.Array, list[RotationDiagnostics]]:
    r"""
    Optimize angles again on the prepared basis using dynamic signed weights.
    ``on_step`` observes completed updates; the loss history describes pre-update angles.
    """
    if cfg.entanglement_weight != 0.0 and problem.entanglement.n_cuts == 0:
        problem = dc.replace(problem, entanglement=entanglement_pairs_all(problem.samples, problem.n_sites))
        
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    theta = jnp.zeros(problem.n_parameters, dtype=dtype) if theta0 is None else jnp.asarray(theta0, dtype=dtype)
    if theta.shape != (problem.n_parameters,):
        raise ValueError(f"rotation theta has shape {theta.shape}, expected {(problem.n_parameters,)}")

    if problem.n_parameters == 0 or cfg.n_rot == 0:
        row = evaluate_rotation_diagnostics(problem, theta, cfg)
        return theta, [row]

    state_dependent                     = cfg.participation_weight != 0.0 or cfg.entanglement_weight != 0.0
    nested_indices                      = None
 
    # If the nested loss weight is non-zero, compute the 
    # importance of the current ground state and select the most important half of the basis for nested loss evaluation.
    if cfg.nested_weight != 0.0:
        matrix_0 		= build_rotated_subspace_hamiltonian(problem, theta)
        state_0 		= ritz_ground_state(matrix_0, problem.j_rms)
        importance 		= np.abs(np.asarray(state_0.vector)) ** 2
        m_half 			= max(1, problem.structure.size // 2)
        nested_idx 		= np.sort(np.argsort(-importance)[:m_half])
        nested_indices 	= jnp.asarray(nested_idx, dtype=jnp.int32)

    has_nested                          = cfg.nested_weight != 0.0
    nested_size                         = 0 if nested_indices is None else int(nested_indices.shape[0])
    step                                = _rotation_kernel_factory(_kernel_signature(problem, state_dependent, has_nested, nested_size))
    state                               = optax.scale_by_adam().init(theta)
    weights                             = _weights(cfg)
    learning_rate                       = jnp.asarray(cfg.learning_rate, dtype=dtype)
    grad_clip                           = jnp.asarray(cfg.grad_clip, dtype=dtype)
    history: list[RotationDiagnostics]  = []
    bar                                 = trange(cfg.n_rot, desc="rotation (ritz)", leave=True, disable=progress_disabled() or not show_progress)
    
    for _ in interruptible(bar, on_interrupt="rotation optimization interrupted"):
        theta, state, diagnostics 	= step(problem, theta, state, weights, learning_rate, grad_clip, nested_indices)
        diag						= _to_float(diagnostics, cfg)
        bar.set_postfix(loss=f"{diag.loss:.6e}", ritz_energy=f"{diag.ritz_energy:.6e}", ritz_gap=f"{diag.ritz_gap:.3e}")
        history.append(diag)
        if on_step is not None:
            on_step(len(history), theta)

    if not np.all(np.isfinite(np.asarray(theta))):
        raise FloatingPointError("rotation update produced non-finite angles")

    min_gap = min((row.ritz_gap for row in history), default=float("inf"))
    if min_gap <= _RELATIVE_GAP_FLOOR * problem.j_rms:
        log(f"Near-degenerate sampled Ritz state: min gap={min_gap:.3e}; state-dependent gradients are regularized", level=2, severity="warning")
    return theta, history

# ------------------------------------------------------------
#! EOF
# ------------------------------------------------------------
