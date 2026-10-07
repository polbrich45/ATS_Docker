"""NQS construction, sampling, training, and ED-sized diagnostics."""

from __future__ import annotations

import functools
import time
from typing import Any

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

from .config		import SPIN, LatticeConfig, LatticeType, ModelConfig, TrainConfig, sector_n_up
from .logging_utils import log
from .lattices		import square_checkerboard_phase
from .model			import ARTransformer, bits_to_inputs, in_sector, inputs_to_bits, support_size
from .netket_vmc	import pauli_terms_to_netket, rotated_netket_hamiltonian, train_vmc
from .operators		import PauliTerm, commutes_with_total_sz, marshall_clifford_theta, rotated_hamiltonian_terms
from .rotation_ritz import sampled_renyi2_entropies, schmidt_labels

_MAX_ED 				= 12
_MAX_FULL_ENTROPY_ED 	= 10


@functools.lru_cache(maxsize=32)
def _diagnostic_apply(model: ARTransformer):
	"""Reuse one compiled, chunked evaluator across parameters and outer stages."""
	return jax.jit(nk.jax.apply_chunked(model.apply, in_axes=(None, 0), chunk_size=1024))


def normalized_amplitudes(log_psi: Any) -> np.ndarray:
	r"""Return $\psi/\|\psi\|$ from log-amplitudes on an enumerated basis.

	Non-finite entries, e.g. $\log\psi=-\infty$ outside a masked sector, are
	set to zero. The largest real part is subtracted before exponentiation.
	"""
	log_psi = np.asarray(log_psi)
	finite	= np.isfinite(log_psi.real) & np.isfinite(log_psi.imag)
	if not finite.any():
		raise FloatingPointError("NQS produced no finite amplitude on the enumerated basis")
	psi			= np.zeros(log_psi.shape, dtype=np.complex128)
	psi[finite] = np.exp(log_psi[finite] - log_psi.real[finite].max())
	return psi / np.linalg.norm(psi)


def energy_moments(matrix: Any, psi: np.ndarray) -> tuple[float, float]:
	r"""Return $\langle H\rangle$ and $\langle H^2\rangle-\langle H\rangle^2$ for a normalized $\psi$; the variance is clipped at zero."""
	h_psi	= matrix @ psi
	energy	= float(np.real(np.vdot(psi, h_psi)))
	return energy, max(float(np.real(np.vdot(h_psi, h_psi))) - energy ** 2, 0.0)


def last_energies(history: list[dict[str, float]], frac: float = 0.05) -> tuple[float, float]:
	if not history:
		return 0.0, 0.0
	n = max(1, int(len(history) * frac)) if frac <= 1.0 else min(len(history), int(frac))
	tail = history[-n:]
	return sum(r["energy"] for r in tail) / len(tail), sum(r["variance"] for r in tail) / len(tail)


def make_transformer(cfg: LatticeConfig, model_cfg: ModelConfig) -> ARTransformer:
	if model_cfg.use_marshall_sign and cfg.lattice is not LatticeType.SQUARE:
		raise ValueError("The Marshall reference phase is implemented only on square lattices")
	reference_phase = square_checkerboard_phase(cfg) if model_cfg.use_marshall_sign else ()
	return ARTransformer(
		hilbert				= nk.hilbert.Spin(s=SPIN, N=cfg.n_sites),
		n_sites				= cfg.n_sites,
		width				= model_cfg.width,
		depth				= model_cfg.depth,
		heads				= model_cfg.heads,
		mlp_ratio			= model_cfg.mlp_ratio,
		phase_head_depth	= model_cfg.phase_head_depth,
		phase_head_width	= model_cfg.phase_head_width,
		phase_scale			= model_cfg.phase_scale,
		phase_init_std		= model_cfg.phase_init_std,
		n_up				= sector_n_up(model_cfg, cfg.n_sites),
		reference_phase		= reference_phase,
	)


# ------------------------


def draw_configs(model: ARTransformer, params: Any, key: jax.Array, n: int) -> np.ndarray:
	r"""Draw ``n`` independent Born samples as ``{0, 1}`` configurations.

	This is the with-replacement sampler used by VMC, support diagnostics, energy
	estimates, and the optional empirical-frequency Ritz seed rule.
	"""
	if n < 1:
		raise ValueError("n must be positive")
	seed 	= int(jax.random.randint(key, (), 0, 2_147_483_647))
	state 	= nk.vqs.MCState(nk.sampler.ARDirectSampler(model.hilbert), model, n_samples=int(n), variables=params, sampler_seed=seed, n_discard_per_chain=0)
	spins 	= np.asarray(state.samples).reshape(-1, model.n_sites)
	if spins.shape[0] < n:
		raise RuntimeError(f"ARDirectSampler returned {spins.shape[0]} rows, expected at least {n}")
	return inputs_to_bits(model, spins[:n]).astype(np.int8)


def most_frequent_unique(configs: np.ndarray, k: int, *, statistics: dict | None = None) -> np.ndarray:
	"""Return at most ``k`` unique rows ranked by empirical frequency.

	Ties follow lexicographic ``numpy.unique`` order, making a fixed sample batch
	deterministic.	This helper performs no resampling: if a capped batch of 100
	Born draws contains only three configurations, it returns three rows.
	"""
	configs = np.asarray(configs, dtype=np.int8)
	if configs.ndim != 2:
		raise ValueError("configs must have shape (n_samples, n_sites)")
	if k < 1:
		raise ValueError("k must be positive")
	unique, counts 	= np.unique(configs, axis=0, return_counts=True)
	order 			= np.argsort(-counts, kind="stable")
	if statistics is not None:
		statistics.update(born_draws=int(len(configs)), unique_born_states=int(len(unique)), ranked_seed_states=int(min(k, len(unique))))
	return np.asarray(unique[order[: int(k)]], dtype=np.int8)



def frequency_ranked_configs(model: ARTransformer, params: Any, key: jax.Array, *, n_samples: int, k: int, statistics: dict | None = None) -> np.ndarray:
	r"""Rank unique configurations from a hard-capped Born sample.

	Exactly ``n_samples`` with-replacement draws are made; there is no adaptive
	redrawing hidden behind the cap.  For a peaked sampler this can return fewer
	than ``k`` rows.  The adaptive Ritz builder then uses unique Gumbel samples
	only to fill the missing fixed-size budget.

	Example
	-------
	If 100 Born draws visit configurations ``A, B, C`` respectively
	``80, 15, 5`` times, the physical ranking is ``A, B, C``. If only ``A`` is
	observed, this function returns one state rather than pretending that the
	unseen support was measured.

	>>> counts = {"A": 80, "B": 15, "C": 5}
	>>> sorted(counts, key=counts.get, reverse=True)
	['A', 'B', 'C']
	"""
	return most_frequent_unique(draw_configs(model, params, key, int(n_samples)), k, statistics=statistics)


# -------------------------


def _conditioned_child_gumbels(raw: jax.Array, parent_gumbel: jax.Array, valid: jax.Array) -> jax.Array:
	r"""Condition child Gumbels so their maximum equals the parent Gumbel.

	For unconditioned child scores ``raw`` with ``z=max(raw)``, the top-down
	Gumbel transform is

	``-log(exp(-parent) - exp(-z) + exp(-raw))``.

	The implementation below evaluates the non-negative difference in log space,
	avoiding cancellation for the maximum child and overflow for small
	probabilities.
	"""
	z 				= jnp.max(raw, axis=-1, keepdims=True)
	delta 			= raw - z
	finite_nonmax 	= jnp.isfinite(raw) & (delta < 0.0)
	raw_safe 		= jnp.where(finite_nonmax, raw, 0.0)
	delta_safe 		= jnp.where(finite_nonmax, delta, -1.0)
	log_difference 	= jnp.where(finite_nonmax, -raw_safe + jnp.log(-jnp.expm1(delta_safe)), -jnp.inf)
	conditioned 	= -jnp.logaddexp(-parent_gumbel[:, None], log_difference)
	return jnp.where(valid, conditioned, -jnp.inf)


@functools.partial(jax.jit, static_argnames=("model", "k"))
def _gumbel_top_k_kernel(model: ARTransformer, params: Any, key: jax.Array, *, k: int) -> jax.Array:
	"""Fixed-shape stochastic beam implementing ancestral Gumbel top-k."""
	n_sites 		= model.n_sites
	keys 			= jax.random.split(key, n_sites + 1)
	prefixes 		= jnp.zeros((k, n_sites), dtype=jnp.int32)
	dtype           = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
	log_probability = jnp.full((k,), -jnp.inf, dtype=dtype).at[0].set(0.0)
	root 			= jax.random.gumbel(keys[0], (), dtype=dtype)
	gumbel_score 	= jnp.full((k,), -jnp.inf, dtype=dtype).at[0].set(root)

	def body_fn(state: tuple[jax.Array, jax.Array, jax.Array], step_input: tuple[jax.Array, jax.Array]) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
		prefs, log_prob, gumbels	= state
		site, k_site 				= step_input
		log_psi 					= model.apply(params, bits_to_inputs(model, prefs), method=model.conditionals_log_psi)
		local_log_probability		= 2.0 * jnp.real(log_psi[:, site, :])
		child_log_probability 		= log_prob[:, None] + local_log_probability
		valid 						= jnp.isfinite(child_log_probability)
		noise 						= jax.random.gumbel(k_site, (k, 2), dtype=dtype)
		raw 						= jnp.where(valid, child_log_probability + noise, -jnp.inf)
		child_gumbel 				= _conditioned_child_gumbels(raw, gumbels, valid)

		selected_score, selected 	= jax.lax.top_k(child_gumbel.reshape(-1), k)
		parent 						= selected // 2
		bit 						= (selected % 2).astype(jnp.int32)
		prefs 						= prefs[parent].at[:, site].set(bit)
		log_prob 					= child_log_probability.reshape(-1)[selected]
		gumbels	 					= selected_score
		return (prefs, log_prob, gumbels), None

	init_state 	 				= (prefixes, log_probability, gumbel_score)
	scan_inputs 				= (jnp.arange(n_sites), keys[1:])
	(final_prefixes, final_probability, final_score), _ = jax.lax.scan(body_fn, init_state, scan_inputs)
	valid = jnp.isfinite(final_probability) & jnp.isfinite(final_score)
	return jnp.where(valid[:, None], final_prefixes, -1).astype(jnp.int8)


def gumbel_top_k_configs(model: ARTransformer, params: Any, key: jax.Array, k: int) -> np.ndarray:
	r"""Sample exactly ``k`` unique configurations without replacement.

	This uses ancestral Gumbel top-k (Kool et al., JMLR 2020), as applied
	to NQS in arXiv:2408.07625. It
	explores only ``O(k N)`` prefixes rather than enumerating the Hilbert space,
	while preserving the exact autoregressive without-replacement law.	It is
	intended for a compact Ritz basis, not for unbiased VMC estimators.

	Example
	-------
	For probabilities ``(0.98, 0.01, 0.006, 0.004)``, four ordinary Born draws
	usually repeat the dominant state. Without-replacement ``K=4`` instead
	returns the complete four-state support once, while ``K=1`` retains the
	original Born probabilities.

	>>> probabilities = (0.98, 0.01, 0.006, 0.004)
	>>> len(probabilities) == 4
	True

	References
	----------
	- Neural Quantum States and Peaked Molecular Wave Functions,
	  https://arxiv.org/abs/2408.07625
	"""
	k = int(k)
	if k < 1:
		raise ValueError("k must be positive")
	if k > support_size(model):
		raise ValueError(f"k={k} exceeds the autoregressive support size {support_size(model)}")

	samples = np.asarray(_gumbel_top_k_kernel(model, params, key, k=k), dtype=np.int8)
	valid 	= np.all((samples == 0) | (samples == 1)) and np.unique(samples, axis=0).shape[0] == k and in_sector(model, samples)
	if not valid:
		raise FloatingPointError("Gumbel beam has invalid or repeated configurations; check conditional probabilities")
	return samples


# ------------------------


def conditional_entropy(model: ARTransformer, params: Any, samples: np.ndarray) -> float:
	"""Mean per-site conditional entropy on samples drawn from the ANN itself."""
	samples = np.asarray(samples, dtype=np.int8)
	if samples.size == 0:
		return float("nan")
	log_psi 		= model.apply(params, bits_to_inputs(model, jnp.asarray(samples)), method=model.conditionals_log_psi)
	probability 	= jnp.exp(2.0 * jnp.real(log_psi))
	terms 			= jnp.where(probability > 0.0, probability * jnp.log(probability), 0.0)
	return float(-jnp.mean(jnp.sum(terms, axis=-1)))


def subspace_probability_mass(model: ARTransformer, params: Any, samples: np.ndarray) -> float:
	"""Exact ANN probability mass carried by a unique configuration set."""
	samples = np.unique(np.asarray(samples, dtype=np.int8), axis=0)
	if samples.size == 0:
		return 0.0
	log_psi = model.apply(params, bits_to_inputs(model, jnp.asarray(samples)))
	return float(jnp.sum(jnp.exp(2.0 * jnp.real(log_psi))))


def train_nqs(
	cfg			: LatticeConfig,
	model_cfg	: ModelConfig,
	train_cfg	: TrainConfig,
	base_terms	: list[PauliTerm],
	gates		: list,
	theta		: jax.Array,
	params		: Any,
	key			: jax.Array,
	*,
	model		: ARTransformer | None = None,
	fresh_init	: bool = False,
	seed		: int = 0,
	desc		: str = "NQS",
) -> dict[str, Any]:
	r"""Optimize the AR-NQS in the current basis ``H_theta = U_theta^dag H U_theta``.

	Builds the rotated/pruned Pauli operator, then runs ``train_cfg.n_nqs`` VMC/minSR
	steps (warm-started from ``params`` unless ``fresh_init``). Returns the model,
	parameters, RNG key, per-step history, and the rotated terms and operator.
	"""
	n_up 					= sector_n_up(model_cfg, cfg.n_sites)
	if model_cfg.conserve_total_sz and not commutes_with_total_sz(base_terms, cfg.n_sites):
		raise ValueError("ModelConfig.conserve_total_sz=True requires [H, total Sz] = 0")
	model					= make_transformer(cfg, model_cfg) if model is None else model
	params_in				= None if fresh_init else params
	key						= jax.random.PRNGKey(seed) if fresh_init else key

	started					= time.perf_counter()
	terms, n_raw			= rotated_hamiltonian_terms(base_terms, gates, theta, cfg.n_sites)
	basis_seconds			= time.perf_counter() - started

	# Train the NQS in the current basis
	operator_started		= time.perf_counter()
	operator				= pauli_terms_to_netket(terms, model.hilbert)
	theta_abs				= float(np.linalg.norm(np.asarray(theta)))
	log(f"[{desc}] {'fresh init' if fresh_init else 'warm start'} | |theta|={theta_abs:.6f} | {len(gates)} gates", level=2)
	log(f"Hamiltonian: {len(base_terms)} base -> {n_raw} rotated -> {len(terms)} pruned terms | max connections={operator.max_conn_size}", level=3)
	log(f"Build timings: basis={basis_seconds:.3f} s | NetKet operator={time.perf_counter() - operator_started:.3f} s", level=3)

	# Log the training details and run the VMC
	params, history, key	= train_vmc(params_in, model, operator, train_cfg, key, n_terms=len(terms), desc=desc)
	e_fin, var_fin			= last_energies(history)
	log(f"[{desc}] MC tail mean (pre-update observations): E={e_fin:+.9f} | E/N={e_fin/cfg.n_sites:+.9f} | variance={var_fin:.6g}", level=2)
	return {
		"params"	: params,
		"model"		: model,
		"key"		: key,
		"history"	: history,
		"n_up"		: n_up,
		"terms"		: terms,
		"raw_terms"	: int(n_raw),
		"operator"	: operator,
	}


# ----------------------------------------------------------------------------
# ED-sized comparisons and diagnostics
# ----------------------------------------------------------------------------


def _apply_pauli_string_to_vector(vector: np.ndarray, bits: np.ndarray, code_to_index: np.ndarray, x_mask: int, z_mask: int, n_sites: int) -> np.ndarray:
	''' Operators etc. evaluation in the computational basis, for exact-basis comparisons and diagnostics. '''
	site_weights	= (1 << np.arange(n_sites, dtype=np.int64))
	connected		= (bits @ site_weights) ^ int(x_mask)
	rows			= code_to_index[connected]
	z_bits			= (int(z_mask) >> np.arange(n_sites, dtype=np.int64)) & 1
	parity			= (bits & z_bits[None, :]).sum(axis=1) & 1
	sign			= 1 - 2 * parity
	y_phase			= (1.0j) ** bin(int(x_mask) & int(z_mask)).count("1")
	out				= np.zeros_like(vector, dtype=np.complex128)
	out[rows]		= y_phase * sign * vector
	return out

def apply_rotation_adjoint_to_vector(vector: np.ndarray, states: np.ndarray, gates: list, theta: jax.Array, n_sites: int) -> np.ndarray:
	"""Apply ``U(theta)^dag`` to a dense state vector in NetKet basis order."""
	if not gates:
		return np.asarray(vector, dtype=np.complex128)
	bits								= ((1 - states) // 2).astype(np.int64)
	site_weights						= (1 << np.arange(n_sites, dtype=np.int64))
	code_to_index						= np.empty(2 ** n_sites, dtype=np.int64)
	code_to_index[bits @ site_weights]	= np.arange(bits.shape[0], dtype=np.int64)
	out									= np.asarray(vector, dtype=np.complex128)
	theta_np							= np.asarray(theta, dtype=float)
	for gate in reversed(gates):
		for g_x, g_z, multiplier in gate.generators:
			angle		= float(multiplier) * float(theta_np[gate.parameter])
			p_out		= _apply_pauli_string_to_vector(out, bits, code_to_index, g_x, g_z, n_sites)
			out			= np.cos(angle) * out + 1.0j * np.sin(angle) * p_out
	return out


def apply_rotation_to_vector(vector: np.ndarray, states: np.ndarray, gates: list, theta: jax.Array, n_sites: int) -> np.ndarray:
	"""Apply ``U(theta)``; generators within each gate commute, gates need not."""
	return apply_rotation_adjoint_to_vector(vector, states, list(reversed(gates)), -jnp.asarray(theta), n_sites)


def marshall_transform_vector(cfg: LatticeConfig, vector: np.ndarray) -> np.ndarray:
	"""Apply the fixed Marshall Clifford (theta=pi) to a full-Hilbert-space vector.

	Reuses :func:`apply_rotation_adjoint_to_vector` in the exact-ED basis order
	(``nk.hilbert.Spin(...).all_states()``), so an ``exact_vector`` from
	:func:`netket_vmc.exact_ground_state` (computed in the ORIGINAL basis) can be
	re-expressed in the Marshall-preprocessed basis that ``base_terms`` then lives
	in -- keeping the ED overlap/phase diagnostics in :func:`ed_diagnostics`
	self-consistent when ``RotationConfig.use_marshall_clifford`` is set.
	"""
	gates, theta = marshall_clifford_theta(cfg)
	if not gates:
		return vector
	states	= np.asarray(nk.hilbert.Spin(s=SPIN, N=cfg.n_sites).all_states())
	return apply_rotation_adjoint_to_vector(vector, states, gates, theta, cfg.n_sites)


def exact_basis_entropy_comparison(cfg: LatticeConfig, gates: list, theta: jax.Array, exact_vector: np.ndarray | None, max_sites: int = _MAX_FULL_ENTROPY_ED) -> dict[str, float]:
	"""Compare summed all-cut Rényi-2 entropy before and after rotation by ED."""

	if exact_vector is None or cfg.n_sites > max_sites:
		return {}

	# Take the states in the computational basis (0/1) and check that the exact_vector is full Hilbert space
	states = np.asarray(nk.hilbert.Spin(s=SPIN, N=cfg.n_sites).all_states())
	if np.asarray(exact_vector).size != states.shape[0]:
		raise ValueError("Full-basis entropy comparison needs a full Hilbert-space ED vector")
	original	= np.asarray(exact_vector, dtype=np.complex128)
	original	= original / np.linalg.norm(original)
	rotated		= apply_rotation_adjoint_to_vector(original, states, gates, theta, cfg.n_sites)
	rotated		= rotated / np.linalg.norm(rotated)

	# The full basis is the zero-extension limit of the sampled-state formula.
	labels		= schmidt_labels(((1 - states) // 2).astype(np.int8), range(1, cfg.n_sites))
	s2_original = float(np.sum(sampled_renyi2_entropies(original, labels)))
	s2_rotated	= float(np.sum(sampled_renyi2_entropies(rotated, labels)))
	return {
		"ed_all_cuts_s2_before_trainable_rotation"	: s2_original,
		"ed_all_cuts_s2_after_trainable_rotation"	: s2_rotated,
		"ed_all_cuts_s2_delta"							: s2_rotated - s2_original,
		"ed_all_cuts_s2_n_cuts"							: float(max(cfg.n_sites - 1, 0)),
	}


# ----------------------------------------------------------------------------


def ed_diagnostics(cfg: LatticeConfig, base_terms: list[PauliTerm], gates: list, theta: jax.Array, model: ARTransformer, params: Any, exact_energy: float, exact_vector: np.ndarray | None, *, operator: Any | None = None) -> dict[str, float]:
	"""Exact full-basis energy, overlap, phase, and entropy checks for small systems.

	Full enumeration is restricted to ``_MAX_ED`` sites; all-cut entropy is
	further restricted to ``_MAX_FULL_ENTROPY_ED``. These diagnostics validate
	conventions and are not part of production-size optimization.
	"""
	out: dict[str, float] = {}
	if cfg.n_sites > _MAX_ED or exact_vector is None:
		return out
	if operator is None:
		operator = rotated_netket_hamiltonian(cfg, gates, theta, base_terms=base_terms)
	states				= np.asarray(operator.hilbert.all_states())
	if np.asarray(exact_vector).size != states.shape[0]:
		raise ValueError("ed_diagnostics requires a full-Hilbert-space exact vector")
	rotated_ground		= apply_rotation_adjoint_to_vector(exact_vector, states, gates, theta, cfg.n_sites)
	rotated_ground	   /= np.linalg.norm(rotated_ground)

	# Compute the variational energy and variance of the NQS in the rotated basis
	phi								= normalized_amplitudes(_diagnostic_apply(model)(params, jnp.asarray(states)))
	energy, variance				= energy_moments(operator.to_sparse(), phi)
	p_model							= np.abs(phi) ** 2
	p_exact							= np.abs(rotated_ground) ** 2
	amp_bhattacharyya				= float(np.sum(np.sqrt(np.maximum(p_model, 0.0) * np.maximum(p_exact, 0.0))))
	exact_weight					= p_exact / max(float(np.sum(p_exact)), 1.0e-300)
	support							= p_exact > 1.0e-14
	phase_delta						= np.angle(phi[support]) - np.angle(rotated_ground[support])
	phase_coherence					= float(abs(np.sum(exact_weight[support] * np.exp(1.0j * phase_delta)))) if support.any() else float("nan")
	out["overlap"]					= float(abs(np.vdot(phi, rotated_ground)))
	out["amplitude_bhattacharyya"]	= amp_bhattacharyya
	out["phase_coherence"]			= phase_coherence
	out["variational_energy"]		= energy
	out["variational_variance"]		= variance
	out["energy_error"]				= energy - exact_energy
	out.update(exact_basis_entropy_comparison(cfg, gates, theta, exact_vector))
	return out
