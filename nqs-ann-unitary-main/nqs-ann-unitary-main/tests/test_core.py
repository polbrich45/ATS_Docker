"""Crucial scientific guards for the explicit adaptive-basis NQS workflow.

The suite is intentionally compact.	 Every test protects either a physics
identity, the sampler/teacher support contract, or the fixed-shape execution
model.	Run with ``pytest -s`` to keep the short traces visible.
"""

from __future__ import annotations

import dataclasses as dc
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
import src.nqs as nqs_module

jax.config.update("jax_enable_x64", True)

from src.config import (
	FinalConfig,
	HamiltonianType,
	LatticeConfig,
	LatticeType,
	ModelConfig,
	RotationConfig,
	TeacherConfig,
	TrainConfig,
)
from src.final_solvers 	import solve_final
from src.logging_utils 	import display_context
from src.netket_vmc 	import exact_ground_state
from src.nqs 			import _gumbel_top_k_kernel, frequency_ranked_configs, gumbel_top_k_configs, make_transformer, most_frequent_unique, train_nqs
from src.operators 		import (
							PauliTerm,
							_multiply_pauli,
							base_hamiltonian,
							commutes_with_total_sz,
							pauli_rotation_circuit,
							pauli_term,
							rotate_terms,
							rotated_hamiltonian_terms,
						)
from src.rotation_ritz import (
							_rotation_kernel_factory,
							build_rotated_subspace_hamiltonian,
							compute_participation_entropy,
							compute_renyi2_entropy,
							entanglement_pairs_all,
							evaluate_rotation_diagnostics,
							optimize_rotation_angles,
							prepare_rotation_problem,
							ritz_ground_state,
							sampled_nonstoquasticity,
							sampled_renyi2_entropies,
							schmidt_labels,
						)
from src.ritz_subspace 	import build_sampled_subspace, pt2_select_subspace, ritz_pt2_correction
from src.teacher 		import fit_ritz_teacher, ritz_probability_target


_PAULI = {
	0: np.eye(2, dtype=np.complex128),
	1: np.array([[0, 1], [1, 0]], dtype=np.complex128),
	2: np.array([[1, 0], [0, -1]], dtype=np.complex128),
	3: np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
}


# -------------------------


def _log(message: str) -> None:
	"""Visible, concise trace for ``pytest -s``."""
	print(f"[test_core] {message}")


def _states(n_sites: int) -> np.ndarray:
	"""All computational states in the repository's little-endian bit order."""
	codes = np.arange(2**n_sites, dtype=np.int64)
	return ((codes[:, None] >> np.arange(n_sites)[None, :]) & 1).astype(np.int8)


def _dense_pauli(x_mask: int, z_mask: int, n_sites: int) -> np.ndarray:
	"""Dense physical Pauli string in the same little-endian site order."""
	out = np.ones((1, 1), dtype=np.complex128)
	for site in range(n_sites):
		code = ((x_mask >> site) & 1) + 2 * ((z_mask >> site) & 1)
		out = np.kron(_PAULI[code], out)
	return out


def _dense_terms(terms: list[PauliTerm], n_sites: int) -> np.ndarray:
	"""Dense matrix for a Pauli expansion."""
	out = np.zeros((2**n_sites, 2**n_sites), dtype=np.complex128)
	for term in terms:
		out += complex(term.coefficient) * _dense_pauli(term.x, term.z, n_sites)
	return out


def _gate_generator(gate, n_sites: int) -> np.ndarray:
	"""Dense generator stored in one ``RotationGate``."""
	return sum(multiplier * _dense_pauli(x_mask, z_mask, n_sites) for x_mask, z_mask, multiplier in gate.generators)


def _projected(terms, gates, theta, samples, n_sites):
	"""Build a projected matrix through the public prepared-problem API."""
	return build_rotated_subspace_hamiltonian(prepare_rotation_problem(terms, gates, samples, n_sites), theta)


def _tiny_problem() -> tuple[LatticeConfig, ModelConfig]:
	return (LatticeConfig(Lx=2, Ly=1, pbc=False, j2=0.0), ModelConfig(width=8, depth=1, heads=2, mlp_ratio=1))


def test_pauli_table_and_total_sz_commutator() -> None:
	"""All Pauli products and the exact U(1) commutator use physical phases."""
	_log("checking exhaustive Pauli products and [H,Sz]")
	for a in range(4):
		for b in range(4):
			phase, x_mask, z_mask = _multiply_pauli(
				a & 1, (a >> 1) & 1, b & 1, (b >> 1) & 1, 1
			)
			c = x_mask + 2 * z_mask
			assert np.allclose(_PAULI[a] @ _PAULI[b], phase * _PAULI[c])

	symmetric = LatticeConfig(Lx=2, Ly=2, pbc=False, j2=0.5)
	transverse = dc.replace(symmetric, hx=0.2)
	assert commutes_with_total_sz(base_hamiltonian(symmetric), symmetric.n_sites)
	assert not commutes_with_total_sz(base_hamiltonian(transverse), transverse.n_sites)
	tiny_transverse = [
		dc.replace(term, coefficient=1.0e-15 * term.coefficient)
		for term in base_hamiltonian(transverse)
	]
	assert not commutes_with_total_sz(tiny_transverse, transverse.n_sites)

	ising = LatticeConfig(
		hamiltonian=HamiltonianType.ISING, Lx=2, Ly=1, pbc=False, hx=0.7
	)
	kitaev = LatticeConfig(
		lattice=LatticeType.HONEYCOMB,
		hamiltonian=HamiltonianType.KITAEV_GAMMA,
		Lx=1,
		Ly=1,
		pbc=False,
		K=0.8,
		J=0.2,
		gamma=0.1,
	)
	for problem in (ising, kitaev):
		dense_energy = np.linalg.eigvalsh(
			_dense_terms(base_hamiltonian(problem), problem.n_sites)
		)[0]
		assert np.isclose(exact_ground_state(problem)[0], dense_energy, atol=1.0e-11)


@pytest.mark.parametrize("hx,hz", [(0.7, 0.0), (0.7, 0.03), (0.0, -0.2)])
def test_ising_fields_spin_normalization(hx: float, hz: float) -> None:
	"""Both Ising fields have the physical sign and S=sigma/2 normalization."""
	cfg = LatticeConfig(hamiltonian="ising", Lx=2, Ly=1, pbc=False, j1=1.2, hx=hx, hz=hz)
	x0, x1 = _dense_pauli(1, 0, 2), _dense_pauli(2, 0, 2)
	z0, z1 = _dense_pauli(0, 1, 2), _dense_pauli(0, 2, 2)
	expected = -cfg.j1 / 4.0 * (z0 @ z1) - hx / 2.0 * (x0 + x1) - hz / 2.0 * (z0 + z1)
	actual = _dense_terms(base_hamiltonian(cfg), cfg.n_sites)
	assert np.allclose(actual, expected, atol=1.0e-12)
	assert np.allclose(actual, actual.conj().T, atol=1.0e-12)
	assert np.isclose(exact_ground_state(cfg)[0], np.linalg.eigvalsh(expected)[0], atol=1.0e-11)


def test_transformer_complex_x64_and_unique_gumbel_sampler() -> None:
	"""The ANN is complex128 and Gumbel top-k is exact, unique, and sector-safe."""
	_log("checking transformer and without-replacement autoregressive sampling")
	cfg, model_cfg = _tiny_problem()
	model = make_transformer(cfg, model_cfg)
	spins = jnp.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1]])
	with warnings.catch_warnings():
		warnings.simplefilter("error", DeprecationWarning)
		params = model.init(jax.random.PRNGKey(0), spins[:1])
		log_psi = model.apply(params, spins)
	assert log_psi.dtype == jnp.complex128
	probabilities = np.exp(2.0 * np.real(np.asarray(log_psi)))
	assert np.isclose(probabilities.sum(), 1.0, atol=1.0e-12)
	assert np.allclose(probabilities, 0.25, atol=1.0e-12)

	full = gumbel_top_k_configs(model, params, jax.random.PRNGKey(1), 4)
	assert {tuple(row) for row in full} == {tuple(row) for row in _states(2)}

	# K=1 must reproduce the exact Born categorical law.
	keys = jax.random.split(jax.random.PRNGKey(4), 3000)
	draws = jax.vmap(lambda key: _gumbel_top_k_kernel(model, params, key, k=1)[0])(keys)
	codes = np.asarray(draws) @ (1 << np.arange(2))
	frequency = np.bincount(codes, minlength=4) / codes.size
	assert np.max(np.abs(frequency - probabilities)) < 0.04

	sector_model = make_transformer(
		LatticeConfig(Lx=2, Ly=2, pbc=False),
		ModelConfig(width=8, depth=1, heads=2, conserve_total_sz=True, n_up=1),
	)
	sector_params = sector_model.init(jax.random.PRNGKey(5), jnp.ones((1, 4)))
	sector_states = np.asarray(sector_model.hilbert.all_states())
	valid_sector = sector_states[np.sum(sector_states == 1, axis=1) == 1]
	sector_probability = np.exp(2.0 * np.real(np.asarray(sector_model.apply(sector_params, valid_sector))))
	assert np.allclose(sector_probability, 0.25, atol=1.0e-12)
	sector = gumbel_top_k_configs(sector_model, sector_params, jax.random.PRNGKey(6), 4)
	assert np.unique(sector, axis=0).shape[0] == 4
	assert np.all(np.sum(sector == 0, axis=1) == 1)	 # local index 0 is physical up
	with pytest.raises(ValueError, match="support size"):
		gumbel_top_k_configs(sector_model, sector_params, jax.random.PRNGKey(7), 5)


def test_frequency_seed_ranking_obeys_the_draw_cap(monkeypatch) -> None:
	"""Frequency selection ranks counts and performs exactly the configured draws."""
	configs = np.asarray([[0, 0], [1, 0], [0, 0], [0, 1], [1, 0]], dtype=np.int8)
	assert np.array_equal(
		most_frequent_unique(configs, 2),
		np.asarray([[0, 0], [1, 0]], dtype=np.int8),
	)

	calls: list[int] = []

	def capped_draws(_model, _params, _key, n):
		calls.append(int(n))
		return configs[:n]

	monkeypatch.setattr(nqs_module, "draw_configs", capped_draws)
	statistics = {}
	ranked = frequency_ranked_configs(
		object(), None, jax.random.PRNGKey(0), n_samples=5, k=2, statistics=statistics,
	)
	assert calls == [5]
	assert ranked.shape == (2, 2)
	assert statistics == {"born_draws": 5, "unique_born_states": 3, "ranked_seed_states": 2}


@dc.dataclass(frozen=True)
class _ToyAutoregressiveModel:
	"""Four-state normalized model used to test the teacher analytically."""

	n_sites: int = 2
	n_up: int | None = None
	particle_counts: tuple = ()
	local_occupations: bool = False

	def apply(self, params, spins):
		bits = ((1 - spins) // 2).astype(jnp.int32)
		codes = bits @ (1 << jnp.arange(self.n_sites, dtype=jnp.int32))
		log_probability = jax.nn.log_softmax(params["logits"])
		return 0.5 * log_probability[codes] + 1.0j * params["phase"][codes]


def test_teacher_joint_target_reinforce_entropy_and_phase() -> None:
	"""The spin teacher uses the joint Ritz target and every direct loss term."""
	_log("checking joint Ritz teacher, REINFORCE, entropy, and phase")
	vector = np.array([3.0, 1.0j], dtype=np.complex128)
	q = ritz_probability_target(vector, 2)
	assert np.allclose(q, (0.9, 0.1))
	assert np.allclose(q, ritz_probability_target(9.0j * vector, 2))
	with warnings.catch_warnings():
		warnings.simplefilter("error", RuntimeWarning)
		bad = ritz_probability_target(np.array([np.inf, np.nan]), 2)
	assert np.allclose(bad, (0.5, 0.5))

	model = _ToyAutoregressiveModel()
	old_probability = np.array([0.3, 0.3, 0.2, 0.2])
	params = {
		"logits": jnp.log(jnp.asarray(old_probability)),
		"phase": jnp.zeros(4, dtype=jnp.float64),
	}
	sampled_basis = np.array([[0, 0], [1, 0]], dtype=np.int8)
	target_vector = np.array([np.sqrt(0.9), 1.0j * np.sqrt(0.1)])
	config = TeacherConfig(
		n_teacher=250,
		learning_rate=5.0e-2,
		grad_clip=10.0,
		reinforce_weight=0.2,
		entropy_weight=0.05,
		phase_weight=0.2,
	)
	with display_context("silent"):
		updated, history, teacher_metrics = fit_ritz_teacher(
			model, params, sampled_basis, target_vector,
			np.array([-1.0, 0.5]), -1.2, 0.5, config,
		)
	probability = np.asarray(jax.nn.softmax(updated["logits"]))
	assert probability[0] > probability[1]
	assert probability[2:].sum() > 0.0
	assert history[-1]["phase_coherence"] > 0.95
	assert np.isfinite(history[-1]["reinforce_per_site"])
	assert history[-1]["entropy_per_site"] < 0.0
	assert teacher_metrics["phase_coherence"] > 0.95
	# Full-state fidelity includes the probability outside the retained basis.
	assert np.isclose(history[0]["teacher_fidelity"], .3)
	psi = np.exp(np.asarray(model.apply(updated, jnp.asarray(1 - 2 * sampled_basis))))
	assert np.isclose(teacher_metrics["teacher_fidelity"], abs(np.vdot(target_vector, psi))**2)
	# The same deterministic target admits a line-search optimizer, without clipping.
	lbfgs = TeacherConfig(n_teacher=25, optimizer="lbfgs", learning_rate=1., grad_clip=0., phase_weight=.2)
	with display_context("silent"):
		_, lbfgs_rows, lbfgs_metrics = fit_ritz_teacher(model, params, sampled_basis, target_vector, np.array([-1., .5]), -1.2, .5, lbfgs)
	assert lbfgs_metrics["teacher_fidelity"] > .9999
	assert np.all(np.diff([row["teacher_loss"] for row in lbfgs_rows]) <= 1e-10)
	assert lbfgs_metrics["function_evaluations"] == 1 + sum(row["function_evaluations"] for row in lbfgs_rows)
	assert lbfgs_metrics["function_evaluations"] > len(lbfgs_rows)
	with pytest.raises(ValueError, match="grad_clip"):
		TeacherConfig(optimizer="lbfgs")




def test_retained_basis_appends_with_cap_without_large_merge() -> None:
	"""Old states remain first while unique sampled/PT2 rows fill one hard cap."""
	_log("checking retained capped basis construction")
	retained = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int8)
	candidates = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)
	fill_calls = []
	def fill():
		fill_calls.append(True)
		return _states(3)
	basis = build_sampled_subspace([], retained, candidates, fill, 3, 5)
	assert np.array_equal(basis[:2], retained)
	assert basis.shape == (5, 3)
	assert np.unique(basis, axis=0).shape[0] == 5
	assert fill_calls == [True]

	# Configuration identity must remain exact beyond the int64 bit width.
	n_sites = 70
	zero = np.zeros(n_sites, dtype=np.int8)
	high = zero.copy()
	high[69] = 1
	middle = zero.copy()
	middle[35] = 1
	low = zero.copy()
	low[2] = 1
	wide = build_sampled_subspace(
		[], np.stack((zero, high)), np.stack((middle,)),
		lambda: np.stack((low, middle)), n_sites, 4,
	)
	assert np.array_equal(wide, np.stack((zero, high, low, middle)))
	high_terms = [pauli_term({69: "Z"}, 1.0)]
	high_matrix = _projected(high_terms, [], jnp.zeros(0), np.stack((zero, high)), n_sites)
	assert np.allclose(np.diag(high_matrix), np.array([1.0, -1.0]))
	selected_high = pt2_select_subspace(
		[pauli_term({69: "X"}, 1.0), pauli_term({69: "Z"}, 0.5)],
		zero[None, :], n_sites, 2,
	)
	assert selected_high.shape == (2, n_sites) and selected_high[1, 69] == 1


def test_rotation_gates_dense_conjugation_norm_and_symmetry() -> None:
	"""Analytic rotations match dense conjugation and canonical gates have one scale."""
	_log("checking dense rotations, generator norms, and String Givens")
	cfg, _ = _tiny_problem()
	terms = [pauli_term({0: "X"}, 0.3), pauli_term({0: "Z", 1: "Y"}, -0.2)]
	gates = pauli_rotation_circuit(
		cfg, depth=1, family=("XX", "givens", "heisenberg"), parameterization="gate"
	)
	theta = jnp.linspace(-0.2, 0.3, len(gates))
	unitary = np.eye(2**cfg.n_sites, dtype=np.complex128)
	for gate in gates:
		gate_unitary = expm(-1.0j * float(theta[gate.parameter]) * _gate_generator(gate, cfg.n_sites))
		unitary = gate_unitary @ unitary
	rotated = rotate_terms(terms, gates, theta, cfg.n_sites)
	actual = _dense_terms(
		[PauliTerm(x, z, complex(coefficient)) for (x, z), coefficient in rotated.items()],
		cfg.n_sites,
	)
	expected = unitary.conj().T @ _dense_terms(terms, cfg.n_sites) @ unitary
	assert np.allclose(actual, expected, atol=1.0e-11)
	projected = _projected(terms, gates, theta, _states(cfg.n_sites), cfg.n_sites)
	assert np.allclose(projected, expected, atol=1.0e-11)

	larger = LatticeConfig(Lx=3, Ly=2, pbc=False, j2=0.5)
	families = ("X", "Y", "Z", "XX", "givens", "xy_exchange", "heisenberg")
	representatives = [pauli_rotation_circuit(larger, 1, (family,))[0] for family in families]
	string_gates = pauli_rotation_circuit(
		larger, depth=4, family=("string_givens",), include_nnn=True
	)
	representatives.append(string_gates[0])
	identity = np.eye(2**larger.n_sites)
	total_sz = sum(
		_dense_pauli(0, 1 << site, larger.n_sites) / 2.0
		for site in range(larger.n_sites)
	)
	for gate in representatives:
		generator = _gate_generator(gate, larger.n_sites)
		assert np.allclose(generator, generator.conj().T)
		assert np.isclose(np.linalg.norm(generator, ord=2), 0.5, atol=1.0e-12)
		if gate.conserves_sz:
			left = np.einsum("ij,jk->ik", generator, total_sz)
			right = np.einsum("ij,jk->ik", total_sz, generator)
			assert np.allclose(left, right, atol=1.0e-12)
		unitary = expm(-0.37j * generator)
		if gate.real_orthogonal:
			assert np.max(np.abs(np.imag(unitary))) < 1.0e-12
		gram = np.einsum("ji,jk->ik", unitary.conj(), unitary)
		assert np.allclose(gram, identity)
	assert all(len(gate.support) == 3 for gate in string_gates)
	for layer in {gate.layer for gate in string_gates}:
		layer_gates = [gate for gate in string_gates if gate.layer == layer]
		assert all(
			set(left.support).isdisjoint(right.support)
			for index, left in enumerate(layer_gates)
			for right in layer_gates[index + 1 :]
		)
	with pytest.raises(ValueError, match="sole rotation family"):
		pauli_rotation_circuit(larger, 1, ("givens", "string_givens"), include_nnn=True)
	with pytest.raises(ValueError, match="square J1-J2"):
		pauli_rotation_circuit(
			dc.replace(larger, hamiltonian=HamiltonianType.ISING),
			1,
			("string_givens",),
			include_nnn=True,
		)


def test_scale_invariant_rotation_objective() -> None:
	"""Positive Hamiltonian rescaling leaves normalized Ritz updates unchanged."""
	_log("checking normalized Ritz-energy scale invariance")
	cfg = LatticeConfig(Lx=2, Ly=2, pbc=False, j2=0.5)
	terms = base_hamiltonian(cfg)
	scaled = [dc.replace(term, coefficient=7.0 * term.coefficient) for term in terms]
	gates = pauli_rotation_circuit(cfg, 1, ("YX",))
	samples = _states(cfg.n_sites)[:8]
	theta0 = jnp.full(max(gate.parameter for gate in gates) + 1, 0.17)
	rotation = RotationConfig(n_rot=1, n_subspace=8)
	concrete, _ = rotated_hamiltonian_terms(terms, gates, theta0, cfg.n_sites)
	tiny, _ = rotated_hamiltonian_terms(
		[dc.replace(term, coefficient=1.0e-15 * term.coefficient) for term in terms],
		gates,
		theta0,
		cfg.n_sites,
	)
	concrete_map = {(term.x, term.z): term.coefficient for term in concrete}
	tiny_map = {(term.x, term.z): term.coefficient for term in tiny}
	assert concrete_map.keys() == tiny_map.keys()
	assert all(
		np.isclose(tiny_map[key] / 1.0e-15, value, rtol=1.0e-9, atol=1.0e-12)
		for key, value in concrete_map.items()
	)
	theta_a, history_a = optimize_rotation_angles(prepare_rotation_problem(terms, gates, samples, cfg.n_sites), rotation, theta0, show_progress=False)
	theta_b, history_b = optimize_rotation_angles(prepare_rotation_problem(scaled, gates, samples, cfg.n_sites), rotation, theta0, show_progress=False)
	assert np.isclose(history_a[0].loss, history_b[0].loss, atol=1.0e-10)
	assert np.isclose(history_b[0].ritz_energy, 7.0 * history_a[0].ritz_energy, atol=1.0e-10)
	assert np.allclose(theta_a, theta_b, atol=1.0e-10)


def test_signed_auxiliary_objectives_and_kernel_reuse() -> None:
	"""Signed weights reverse metric pressure without specializing optimizer scalars."""
	_log("checking signed auxiliary gradients and cached prepared kernels")
	cfg = LatticeConfig(Lx=2, Ly=2, pbc=False, j2=0.5)
	terms = base_hamiltonian(cfg)
	gates = pauli_rotation_circuit(cfg, 1, ("YX",))
	samples = _states(cfg.n_sites)[[0, 1, 2, 3, 5, 6, 9, 10]]
	problem = prepare_rotation_problem(terms, gates, samples, cfg.n_sites, include_entanglement=True)
	theta = jnp.full(problem.n_parameters, 0.17)

	def state(theta):
		return ritz_ground_state(build_rotated_subspace_hamiltonian(problem, theta), problem.j_rms).vector

	metrics = (
		lambda theta: compute_participation_entropy(state(theta)) / np.log(problem.structure.size),
		lambda theta: compute_renyi2_entropy(state(theta), problem.entanglement) / jnp.sum(problem.entanglement.normalization),
		lambda theta: sampled_nonstoquasticity(build_rotated_subspace_hamiltonian(problem, theta)),
	)
	for metric in metrics:
		gradient = jax.grad(metric)(theta)
		assert np.linalg.norm(np.asarray(gradient)) > 1.0e-7
		assert float(metric(theta - 1.0e-4 * gradient)) < float(metric(theta))
		assert float(metric(theta + 1.0e-4 * gradient)) > float(metric(theta))

	zero = evaluate_rotation_diagnostics(problem, theta, RotationConfig(n_rot=0, n_subspace=8))
	assert all(np.isnan(getattr(zero, name)) for name in ("participation_entropy", "entanglement", "nonstoquasticity"))

	_rotation_kernel_factory.cache_clear()
	positive = RotationConfig(n_rot=1, n_subspace=8, learning_rate=0.01, grad_clip=0.5, participation_weight=0.1, entanglement_weight=0.2, nonstoquasticity_weight=0.3)
	negative = dc.replace(positive, learning_rate=0.03, grad_clip=2.0, participation_weight=-0.1, entanglement_weight=-0.2, nonstoquasticity_weight=-0.3)
	optimize_rotation_angles(problem, positive, theta, show_progress=False)
	other = prepare_rotation_problem(terms, gates, _states(cfg.n_sites)[4:12], cfg.n_sites)
	optimize_rotation_angles(other, negative, theta, show_progress=False)
	cache = _rotation_kernel_factory.cache_info()
	assert cache.misses == 1 and cache.hits >= 1


def test_participation_and_pair_scatter_entanglement() -> None:
	"""Exact states and random sparse complex states match dense entropy formulas."""
	_log("checking product, uniform, Bell, and random pair-scatter entropies")
	samples = _states(2)
	pairs = entanglement_pairs_all(samples, 2)
	product = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex128)
	uniform = jnp.ones(4, dtype=jnp.complex128) / 2.0
	bell = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.complex128) / np.sqrt(2.0)
	assert np.isclose(float(compute_participation_entropy(product)), 0.0, atol=1.0e-12)
	assert np.isclose(float(compute_participation_entropy(uniform)), np.log(4.0), atol=1.0e-12)
	assert np.isclose(float(compute_participation_entropy(bell)), np.log(2.0), atol=1.0e-12)
	assert np.isclose(float(compute_renyi2_entropy(product, pairs)), 0.0, atol=1.0e-12)
	assert np.isclose(float(compute_renyi2_entropy(uniform, pairs)), 0.0, atol=1.0e-12)
	assert np.isclose(float(compute_renyi2_entropy(bell, pairs)), np.log(2.0), atol=1.0e-12)

	rng = np.random.default_rng(4)
	sparse_samples = _states(4)[[0, 1, 3, 5, 8, 10, 14]]
	vector = rng.normal(size=7) + 1.0j * rng.normal(size=7)
	pairs = entanglement_pairs_all(sparse_samples, 4)
	dense_total = 0.0
	for cut in range(1, 4):
		psi = np.zeros((2**cut, 2**(4 - cut)), dtype=np.complex128)
		a = sparse_samples[:, :cut] @ (1 << np.arange(cut))
		b = sparse_samples[:, cut:] @ (1 << np.arange(4 - cut))
		psi[a, b] = vector
		rho = psi @ psi.conj().T / np.sum(np.abs(vector) ** 2)
		dense_total -= np.log(np.sum(np.abs(rho) ** 2))
	assert np.isclose(float(compute_renyi2_entropy(jnp.asarray(vector), pairs)), dense_total, atol=1.0e-11)
	# The sparse reporting path must agree with the differentiable pair scatter.
	assert np.isclose(np.sum(sampled_renyi2_entropies(vector, schmidt_labels(sparse_samples, range(1, 4)))), dense_total, atol=1.0e-11)


def test_regularized_complex_ritz_gradient() -> None:
	"""Complex Ritz-vector derivatives match native/finite differences and stay finite."""
	_log("checking complex regularized Ritz-vector response")
	def matrix(x, gap=1.0):
		return jnp.array([[0.0, x * (0.3 + 0.2j)], [x * (0.3 - 0.2j), gap]], dtype=jnp.complex128)

	def auxiliary(x, gap=1.0):
		return compute_participation_entropy(ritz_ground_state(matrix(x, gap), 1.0).vector)

	def native(x):
		return compute_participation_entropy(jnp.linalg.eigh(matrix(x))[1][:, 0])

	x = 0.4
	custom_gradient = float(jax.grad(auxiliary)(x))
	native_gradient = float(jax.grad(native)(x))
	h = 1.0e-5
	finite_difference = (float(auxiliary(x + h)) - float(auxiliary(x - h))) / (2.0 * h)
	assert np.isclose(custom_gradient, native_gradient, rtol=2.0e-4, atol=2.0e-6)
	assert np.isclose(custom_gradient, finite_difference, rtol=2.0e-4, atol=2.0e-6)
	assert np.isfinite(float(jax.grad(lambda value: auxiliary(value, 1.0e-12))(x)))


def test_ritz_and_pt2() -> None:
	"""Ritz and PT2 retain their variational and frontier meanings."""
	_log("checking sampled Ritz/PT2")
	cfg = LatticeConfig(Lx=3, Ly=1, pbc=False, j2=0.0)
	terms = base_hamiltonian(cfg)
	full = _states(cfg.n_sites)
	full_state = ritz_ground_state(
		_projected(terms, [], jnp.zeros(0), full, cfg.n_sites)
	)
	exact = np.linalg.eigvalsh(_dense_terms(terms, cfg.n_sites))[0]
	assert np.isclose(float(full_state.energy), exact, atol=1.0e-11)
	full_pt2 = ritz_pt2_correction(
		terms, full, full_state.vector, float(full_state.energy), cfg.n_sites
	)
	assert full_pt2["pt2"] == 0.0 and full_pt2["frontier_size"] == 0.0

	seeds = full[[0, 1, 2]]
	selected = pt2_select_subspace(terms, seeds, cfg.n_sites, 6)
	assert np.unique(selected, axis=0).shape == selected.shape
	matrix = _projected(terms, [], jnp.zeros(0), selected, cfg.n_sites)
	state = ritz_ground_state(matrix)
	pt2 = ritz_pt2_correction(terms, selected, state.vector, float(state.energy), cfg.n_sites)
	assert float(state.energy) >= exact - 1.0e-10  # variational Ritz bound
	assert np.isfinite(pt2["pt2"])




def test_fixed_sz_nqs_accepts_givens_and_rejects_broken_hamiltonian() -> None:
	"""Sector validation follows actual Hamiltonian and gate commutators."""
	_log("checking fixed-Sz training validation")
	cfg = LatticeConfig(Lx=2, Ly=2, pbc=False, j2=0.5)
	model_cfg = ModelConfig(width=8, depth=1, heads=2, conserve_total_sz=True, n_up=2)
	train = TrainConfig(n_samples=4, n_nqs=0, use_sr=False, optimizer="adam")
	terms = base_hamiltonian(cfg)
	gates = pauli_rotation_circuit(cfg, 1, ("givens",))
	result = train_nqs(
		cfg, model_cfg, train, terms, gates, jnp.zeros(max(g.parameter for g in gates) + 1),
		None, jax.random.PRNGKey(0), fresh_init=True,
	)
	assert result["n_up"] == 2
	broken = dc.replace(cfg, hx=0.1)
	with pytest.raises(ValueError, match="total Sz"):
		train_nqs(
			broken, model_cfg, train, base_hamiltonian(broken), gates,
			jnp.zeros(max(g.parameter for g in gates) + 1), None,
			jax.random.PRNGKey(1), fresh_init=True,
		)

	polarized = [pauli_term({0: "Z"}, -1.0), pauli_term({1: "Z"}, -1.0)]
	unrestricted = solve_final(polarized, 2, FinalConfig(solver="exact"))["energy"]
	fixed = solve_final(polarized, 2, FinalConfig(solver="exact"), n_up=1)
	assert np.isclose(unrestricted, -2.0)
	assert np.isclose(fixed["energy"], 0.0) and fixed["n_up"] == 1
	with pytest.raises(ValueError, match="does not enforce"):
		solve_final(polarized, 2, FinalConfig(solver="dmrg"), n_up=1)
