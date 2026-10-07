"""Physical conventions and sampling laws needed by the outer-loop experiments."""

import dataclasses as dc
import itertools

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest
from scipy.linalg import expm

from src.config import TrainConfig, LatticeConfig, ModelConfig
from src.fermionic import _hubbard_terms, _projected_matrix, _rotated_interaction, HubbardConfig
from src.nqs import _gumbel_top_k_kernel, gumbel_top_k_configs, make_transformer
from src.netket_vmc import train_vmc
from src.orbitals import orbital_matrix, orbital_pairs, orbital_ritz_objective, prepare_orbital_projection, projected_orbital_matrix, orbital_cut_maps, cut_renyi2

jax.config.update("jax_enable_x64", True)


@dc.dataclass(frozen=True)
class TableAR:
	"""Two-bit correlated categorical distribution with known ordered-pair law."""
	n_sites: int = 2
	local_occupations: bool = True
	n_up: int | None = None
	particle_counts: tuple = ()

	def conditionals_log_psi(self):
		pass

	def apply(self, probabilities, bits, method=None):
		first = jnp.stack((probabilities[0] + probabilities[1], probabilities[2] + probabilities[3]))
		second = probabilities.reshape(2, 2) / first[:, None]
		return .5 * jnp.log(jnp.stack((jnp.broadcast_to(first, (len(bits), 2)), second[bits[:, 0]]), axis=1))


def test_gumbel_ordered_pair_law():
	probability = np.asarray([.6, .1, .2, .1])
	keys = jax.random.split(jax.random.PRNGKey(42), 12000)
	samples = np.asarray(jax.vmap(lambda key: _gumbel_top_k_kernel(TableAR(), jnp.asarray(probability), key, k=2))(keys))
	codes = samples @ np.asarray([2, 1])
	assert np.all(codes[:, 0] != codes[:, 1])
	frequency = np.bincount(codes[:, 0] * 4 + codes[:, 1], minlength=16).reshape(4, 4) / len(samples)
	expected = probability[:, None] * probability[None, :] / (1 - probability[:, None])
	np.fill_diagonal(expected, 0.)
	assert np.max(np.abs(frequency - expected)) < .015
	with pytest.raises(FloatingPointError, match="Gumbel beam"):
		gumbel_top_k_configs(TableAR(), jnp.full(4, jnp.nan), jax.random.PRNGKey(43), 1)


def test_short_orbital_circuit_and_cached_projection():
	of = pytest.importorskip("openfermion")
	hubbard = HubbardConfig(Lx=3, Ly=1, n_up=1, n_down=2)
	graph = nk.graph.Chain(3, pbc=False)
	interaction = of.get_interaction_operator(_hubbard_terms(of, graph, hubbard))
	sector = nk.hilbert.SpinOrbitalFermions(3, s=.5, n_fermions_per_spin=(1, 2))
	basis = np.asarray(sector.all_states(), dtype=np.int8)
	pairs = orbital_pairs(3, 2)
	angles = np.asarray([.31, -.42, .12, .27])
	rotation = orbital_matrix(3, pairs, angles)
	assert len(pairs) == 4
	assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
	# Check signs and gate order independently through a small Fock-space oracle.
	unitary = np.eye(len(basis), dtype=complex)
	for (i, j), angle in zip(pairs, angles):
		generator = of.FermionOperator()
		for spin in (0, 1):
			p, q = 2 * i + spin, 2 * j + spin
			generator += of.FermionOperator(((p, 1), (q, 0)), 1.)
			generator += of.FermionOperator(((q, 1), (p, 0)), -1.)
		operator = nk.operator.FermionOperator2nd.from_openfermion(sector, generator, convert_spin_blocks=True, dtype=complex)
		# Generator is anti-Hermitian, so use to_dense rather than the Hermitian projector.
		unitary = unitary @ expm(angle * np.asarray(operator.to_dense()))
	base_operator = nk.operator.FermionOperator2nd.from_openfermion(sector, of.get_fermion_operator(interaction), convert_spin_blocks=True, dtype=complex)
	base_matrix = np.asarray(base_operator.to_dense())
	rotated = _rotated_interaction(interaction, pairs, angles)
	rotated_operator = nk.operator.FermionOperator2nd.from_openfermion(sector, of.get_fermion_operator(rotated), convert_spin_blocks=True, dtype=complex)
	full = _projected_matrix(rotated_operator, basis)
	assert np.allclose(full, unitary.conj().T @ base_matrix @ unitary, atol=1e-12)
	assert np.allclose(np.linalg.eigvalsh(full), np.linalg.eigvalsh(base_matrix), atol=1e-12)
	# Separate spin-block rotations preserve their full-state Schmidt spectrum.
	vector = np.random.default_rng(12).normal(size=len(basis)) + 1j * np.random.default_rng(13).normal(size=len(basis))
	_, up_index = np.unique(basis[:, :3], axis=0, return_inverse=True)
	_, down_index = np.unique(basis[:, 3:], axis=0, return_inverse=True)
	schmidt = np.zeros((3, 3), dtype=complex)
	rotated_schmidt = np.zeros_like(schmidt)
	schmidt[up_index, down_index] = vector
	rotated_schmidt[up_index, down_index] = unitary.conj().T @ vector
	assert np.allclose(np.linalg.svd(schmidt, compute_uv=False), np.linalg.svd(rotated_schmidt, compute_uv=False), atol=1e-12)
	for selected in (basis, basis[[0, 2, 5, 7]]):
		plan = prepare_orbital_projection(selected)
		assert np.allclose(projected_orbital_matrix(rotated, plan), _projected_matrix(rotated_operator, selected), atol=1e-12)
		# The transition plan depends only on the basis, so an identical basis reuses one object.
		assert prepare_orbital_projection(selected.copy()) is plan
		assert prepare_orbital_projection(selected.astype(np.int8)) is plan


def test_model_entropy_matches_independent_schmidt_decomposition():
	# Missing configurations have zero amplitude in the selected state.
	basis = np.asarray(list(itertools.product((0, 1), repeat=6)), dtype=np.int8)[::3]
	vector = np.random.default_rng(8).normal(size=len(basis)) + 1j * np.random.default_rng(9).normal(size=len(basis))
	vector /= np.linalg.norm(vector)
	maps, signs = orbital_cut_maps(basis, "model")
	assert np.all(signs == 1)
	state = np.zeros((2,) * 6, dtype=complex)
	for row, amplitude in zip(basis, vector):
		state[tuple(row)] = amplitude
	entropy, normalization = 0., 0.
	for cut in range(1, 6):
		schmidt = np.linalg.svd(state.reshape(2 ** cut, -1), compute_uv=False)
		entropy -= np.log(np.sum(schmidt ** 4))
		normalization += np.log(max(2, min(len(basis), 2 ** min(cut, 6 - cut))))
	assert np.isclose(cut_renyi2(vector, maps, signs), entropy / normalization)
	with pytest.raises(ValueError, match="entanglement_order"):
		orbital_cut_maps(basis, "invalid")


def test_local_orbital_circuit_has_bounded_support():
	# Each disjoint nearest-neighbor sublayer moves support by at most one orbital.
	for layers in (1, 2):
		pairs = orbital_pairs(12, layers)
		angles = np.random.default_rng(layers).uniform(-1., 1., len(pairs))
		rotation = orbital_matrix(12, pairs, angles)
		row, column = np.indices(rotation.shape)
		assert np.all(rotation[np.abs(row - column) > 2 * layers] == 0.)
		assert np.allclose(rotation.T @ rotation, np.eye(12), atol=1e-14)


def test_depth_zero_fermionic_nqs_has_spin_block_rank_at_most_two():
	from src.model import ARTransformer

	sector = nk.hilbert.SpinOrbitalFermions(4, s=.5, n_fermions_per_spin=(2, 2))
	basis = np.asarray(sector.all_states(), dtype=np.int8)
	model = ARTransformer(hilbert=sector, n_sites=8, width=4, depth=0, heads=1, mlp_ratio=1, phase_head_depth=0, particle_counts=(2, 2), local_occupations=True)
	_, up_index = np.unique(basis[:, :4], axis=0, return_inverse=True)
	_, down_index = np.unique(basis[:, 4:], axis=0, return_inverse=True)
	for seed in (1, 3, 7):
		params = model.init(jax.random.PRNGKey(seed), jnp.asarray(basis))
		vector = np.exp(np.asarray(model.apply(params, jnp.asarray(basis))))
		schmidt = np.zeros((6, 6), dtype=complex)
		schmidt[up_index, down_index] = vector
		values = np.linalg.svd(schmidt, compute_uv=False)
		assert np.all(values[2:] < 1e-12 * values[0])
		# All down-block conditionals depend on up occupations only through the last bit.
		for last_bit in (0, 1):
			rows = np.unique(up_index[basis[:, 3] == last_bit])
			assert np.linalg.matrix_rank(schmidt[rows], tol=1e-12) == 1
	of = pytest.importorskip("openfermion")
	terms = _hubbard_terms(of, nk.graph.Grid(extent=[2, 2], pbc=False), HubbardConfig())
	operator = nk.operator.FermionOperator2nd.from_openfermion(sector, terms, convert_spin_blocks=True, dtype=complex)
	_, vectors = np.linalg.eigh(_projected_matrix(operator, basis))
	schmidt[up_index, down_index] = vectors[:, 0]
	assert np.linalg.matrix_rank(schmidt, tol=1e-10) > 2


def test_spatial_entropy_matches_independent_schmidt_decomposition():
	basis = np.asarray(list(itertools.product((0, 1), repeat=6)), dtype=np.int8)
	vector = np.random.default_rng(4).normal(size=64) + 1j * np.random.default_rng(5).normal(size=64)
	vector /= np.linalg.norm(vector)
	maps, signs = orbital_cut_maps(basis)
	# Reorder each occupied creation-operator list independently.
	permutation = np.asarray([0, 2, 4, 1, 3, 5])
	for row, sign in zip(basis, signs):
		occupied = permutation[np.flatnonzero(row)]
		inversions = sum(a > b for i, a in enumerate(occupied) for b in occupied[i + 1:])
		assert sign == (-1) ** inversions
	state = np.zeros((2,) * 6, dtype=complex)
	for row, amplitude, sign in zip(basis, vector, signs):
		state[tuple(row[[0, 3, 1, 4, 2, 5]])] = amplitude * sign
	entropy = 0.
	for cut in (1, 2):
		schmidt = np.linalg.svd(state.reshape(4 ** cut, -1), compute_uv=False)
		entropy -= np.log(np.sum(schmidt ** 4))
	assert np.isclose(cut_renyi2(vector, maps, signs), entropy / (2 * np.log(4)))
	loss, diagnostics = orbital_ritz_objective(np.diag([0., 1.]), 2., .3)
	assert loss == 0. and diagnostics["ipr"] == 1.


def test_vmc_wall_budget_stops_after_a_completed_update():
	lattice = LatticeConfig(Lx=2, Ly=1, pbc=False)
	model = make_transformer(lattice, ModelConfig(width=4, depth=0, heads=1, mlp_ratio=1))
	operator = nk.operator.spin.sigmaz(model.hilbert, 0)
	cfg = TrainConfig(n_nqs=100, n_samples=16, use_sr=False, optimizer="adam")
	checkpoints = []
	params, history, _ = train_vmc(None, model, operator, cfg, jax.random.PRNGKey(3), time_budget_seconds=1e-9, on_step=lambda index, variables: checkpoints.append((index, variables)))
	assert checkpoints[0][0] == 1
	for checkpoint, final in zip(jax.tree_util.tree_leaves(checkpoints[0][1]), jax.tree_util.tree_leaves(params)):
		assert np.array_equal(checkpoint, final)
	assert len(history) == 1
	assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(params))
	with pytest.raises(ValueError, match="finite and positive"):
		train_vmc(params, model, operator, cfg, jax.random.PRNGKey(4), time_budget_seconds=0.)


def test_warm_polish_reaches_stationary_singlet():
	from scripts.benchmark_basis import polish_warm_state
	from src.netket_vmc import pauli_terms_to_netket
	from src.operators import base_hamiltonian

	lattice = LatticeConfig(Lx=2, Ly=1, pbc=False)
	model = make_transformer(lattice, ModelConfig(width=4, depth=0, heads=1, mlp_ratio=1, phase_head_depth=0, conserve_total_sz=True, n_up=1))
	states = np.asarray([[1, -1], [-1, 1]])
	params = model.init(jax.random.PRNGKey(9), jnp.asarray(states))
	operator = pauli_terms_to_netket(base_hamiltonian(lattice), model.hilbert)
	matrix = _projected_matrix(operator, states)
	polished, report = polish_warm_state(model, params, (states, matrix, -.75), 500)
	assert np.isclose(report["energy"], -.75, atol=1e-8)
	assert report["stationarity_checks_passed"]
	assert report["gradient_max"] < 1e-5
	assert report["minimum_curvature"] >= -1e-5
	psi = np.exp(np.asarray(model.apply(polished, jnp.asarray(states))))
	assert np.allclose(np.abs(psi) ** 2, .5, atol=1e-6)
	assert np.isclose(psi[0] / psi[1], -1., atol=1e-6)
	with pytest.raises(ValueError, match="exact sector"):
		polish_warm_state(model, params, None, 500)


def test_comparison_distinguishes_teacher_and_rotation_gains():
	from scripts.benchmark_basis import comparison_summary

	rows = []
	for seed, energy, difference, teacher in ((0, -2., -1., -2.2), (1, -2.4, -.4, -2.3)):
		rows.append({"seed": seed, "baseline": "vmc_time", "adaptive": {"energy": energy, "standard_error": .01}, "teacher_control": {"energy": teacher, "standard_error": .01}, "difference": difference, "difference_standard_error": .01, "budget_reached": True, "teacher_budget_reached": seed == 0})
	# Update-matched arms must not count as additional independent observations.
	rows.append(dict(rows[0], baseline="vmc_updates", difference=-100.))
	summary = comparison_summary(rows)
	assert summary["seeds"] == 2 and summary["resolved_wins"] == 2
	assert np.isclose(summary["mean_paired_difference"], -.7)
	assert np.isclose(summary["seed_standard_error"], .3)
	assert np.isclose(summary["teacher_comparison"]["mean_paired_difference"], .05)
	assert summary["teacher_comparison"]["resolved_wins"] == 1
	assert not summary["teacher_comparison"]["all_time_budgets_reached"]
	uncertain = dict(rows[1], teacher_control={"energy": -2.3, "standard_error": .1})
	assert comparison_summary([rows[0], uncertain])["teacher_comparison"]["resolved_wins"] == 0
	with pytest.raises(ValueError, match="independent seed"):
		comparison_summary(rows + [rows[0]])


def test_spin_loop_rejects_fermionic_basin_search():
	from src.adaptive import run_adaptive_loop
	from src.config import AdaptiveConfig, AdaptiveStep, RotationConfig

	with pytest.raises(ValueError, match="non-negative integer"):
		RotationConfig(n_basin=-1)
	with pytest.raises(ValueError, match="requires n_rot"):
		RotationConfig(n_basin=1, n_rot=0)
	config = AdaptiveConfig(steps=(AdaptiveStep(rotation=RotationConfig(n_rot=1, n_basin=1)),), display="silent")
	with pytest.raises(ValueError, match="fermionic orbital search only"):
		run_adaptive_loop(LatticeConfig(Lx=2, Ly=1, pbc=False), ModelConfig(width=4, heads=1), config)
	with pytest.raises(ValueError, match="entanglement_order"):
		RotationConfig(entanglement_order="invalid")
	config = dc.replace(config, steps=(AdaptiveStep(rotation=RotationConfig(entanglement_order="model")),))
	with pytest.raises(ValueError, match="fermionic loop only"):
		run_adaptive_loop(LatticeConfig(Lx=2, Ly=1, pbc=False), ModelConfig(width=4, heads=1), config)
