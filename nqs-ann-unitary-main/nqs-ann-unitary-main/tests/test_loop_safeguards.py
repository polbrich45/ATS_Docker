"""Physical support, bounded teaching, and accepted-state checkpoint regressions."""

import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import src.adaptive as adaptive
from src.checkpoint import load_outer_checkpoint
from src.config import AdaptiveConfig, AdaptiveStep, CircuitConfig, LatticeConfig, ModelConfig, RotationConfig, TeacherConfig, TrainConfig
from src.logging_utils import display_context
from src.netket_vmc import pauli_terms_to_netket
from src.nqs import make_transformer
from src.operators import base_hamiltonian
from src.rotation_ritz import build_rotated_subspace_hamiltonian, evaluate_rotation_diagnostics, prepare_rotation_problem, ritz_ground_state
from src.ritz_subspace import build_sampled_subspace, pt2_select_subspace
from src.teacher import fit_ritz_teacher

jax.config.update("jax_enable_x64", True)


def test_sector_completion_and_bounded_teacher():
	"""Cancelled XX+YY connections never become forbidden teacher configurations."""
	lattice = LatticeConfig(Lx=4, Ly=1, pbc=False, j2=0.0)
	terms = base_hamiltonian(lattice)
	seed = np.array([[1, 1, 0, 0]], dtype=np.int8)
	selected = pt2_select_subspace(terms, seed, 4, 6)
	assert selected.shape[0] == 2
	assert np.all(selected.sum(axis=1) == 2)
	one_up = pt2_select_subspace(terms, np.array([[0, 1, 1, 1]]), 4, 4, n_up=1)
	assert np.all(one_up.sum(axis=1) == 3)
	bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.int8)
	sector = bits[bits.sum(axis=1) == 2]
	basis = build_sampled_subspace(terms, seed[:0], seed, lambda: sector, 4, 6, n_up=2)
	assert len(np.unique(basis, axis=0)) == 6
	assert np.all(basis.sum(axis=1) == 2)
	model = make_transformer(lattice, ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, conserve_total_sz=True))
	params = model.init(jax.random.PRNGKey(0), jnp.asarray(1 - 2 * sector))
	vector = np.array([1., 0., 0., 0., 0., 0.])
	config = TeacherConfig(n_teacher=1)
	with display_context("silent"):
		_, _, metrics = fit_ritz_teacher(model, params, basis, vector, np.zeros(6), -1., 1., config)
	assert np.isfinite(metrics["teacher_loss"])
	assert np.isclose(metrics["teacher_kl_per_site"], metrics["conditional_kl_per_site"] + metrics["mass_loss_per_site"])
	assert np.isclose(metrics["subspace_mass"], np.exp(-4 * metrics["mass_loss_per_site"]))
	assert np.isclose(metrics["teacher_loss"], metrics["teacher_kl_per_site"] + metrics["phase_loss"])
	with pytest.raises(ValueError, match="unbounded"):
		fit_ritz_teacher(model, params, basis, vector, np.zeros(6), -1., 1., dc.replace(config, entropy_weight=.01))
	with pytest.raises(ValueError, match="unbounded"):
		fit_ritz_teacher(model, params, basis, np.ones(6), np.arange(6.), -1., 1., dc.replace(config, reinforce_weight=1.))
	with pytest.raises(ValueError, match="sector"):
		fit_ritz_teacher(model, params, bits[:6], vector, np.zeros(6), -1., 1., config)
	with pytest.raises(ValueError, match="sector"):
		pt2_select_subspace(terms, bits[:1], 4, 6, n_up=2)
	with display_context("silent"):
		metrics = adaptive._evaluate_nqs(lattice, model, params, TrainConfig(n_nqs=0, n_samples=32), terms, [], jnp.zeros(0), terms, len(terms), pauli_terms_to_netket(terms, model.hilbert), jax.random.PRNGKey(10), float("nan"), None)
	assert np.isfinite(metrics["energy"]) and metrics["standard_error"] >= 0.


def test_nested_gap_is_a_convex_energy_interpolation():
	"""Nested weighting uses the same per-site scale as the full Ritz energy."""
	lattice = LatticeConfig(Lx=2, Ly=2, pbc=False)
	bits = ((np.arange(8)[:, None] >> np.arange(4)) & 1).astype(np.int8)
	problem = prepare_rotation_problem(base_hamiltonian(lattice), [], bits, 4)
	assert problem.entanglement.n_cuts == 0
	assert all(array.size == 0 for array in jax.tree_util.tree_leaves(problem.entanglement) if hasattr(array, "size"))
	theta = jnp.zeros(0)
	config = RotationConfig(n_rot=0, n_subspace=8, nested_weight=.3)
	diagnostics = evaluate_rotation_diagnostics(problem, theta, config)
	matrix = build_rotated_subspace_hamiltonian(problem, theta)
	state = ritz_ground_state(matrix, problem.j_rms)
	indices = np.argsort(-np.abs(np.asarray(state.vector))**2)[:4]
	small = np.linalg.eigvalsh(np.asarray(matrix)[np.ix_(indices, indices)])[0]
	expected = problem.energy_scale * (.7 * float(state.energy) + .3 * small)
	assert np.isclose(diagnostics.loss, expected)
	with pytest.raises(ValueError, match="nested_weight"):
		RotationConfig(nested_weight=1.01)


@pytest.mark.parametrize("proposal_energy, standard_error, accept", [(-.9, 0., False), (-1.1, 0., True), (-.9, .1, True)])
def test_loop_recovery_and_checkpoint_acceptance(monkeypatch, tmp_path, proposal_energy, standard_error, accept):
	"""Only the comparison energies are controlled; all four training phases run."""
	lattice = LatticeConfig(Lx=2, Ly=2, pbc=False)
	model_cfg = ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, conserve_total_sz=True)
	step = AdaptiveStep(train=TrainConfig(n_nqs=1, n_samples=16, use_sr=False), rotation=RotationConfig(n_rot=1, n_subspace=4, sampling="gumbel"), teacher=TeacherConfig(n_teacher=1), n_recovery=1)
	config = AdaptiveConfig(circuit=CircuitConfig(n_layers=1, family=("givens",), theta_cutoff=0.), steps=(step,), display="silent")
	original_evaluate = adaptive._evaluate_nqs
	snapshots = []

	def evaluate(*args):
		metrics = original_evaluate(*args)
		snapshots.append((args[2], np.asarray(args[6]).copy()))
		metrics["energy"] = -1. if len(snapshots) == 1 else proposal_energy
		metrics["standard_error"] = standard_error
		return metrics

	monkeypatch.setattr(adaptive, "_evaluate_nqs", evaluate)
	result = adaptive.run_adaptive_loop(lattice, model_cfg, config, output_dir=tmp_path)
	assert len(snapshots) == 4
	assert result["recovery_history"]
	assert result["outer_history"][0]["accepted"] is accept
	assert result["final_energy"] == (proposal_energy if accept else -1.)
	_, saved_params, saved_theta, _, basis = load_outer_checkpoint(tmp_path)
	expected_params, expected_theta = snapshots[-1 if accept else 0]
	assert np.array_equal(saved_theta, expected_theta)
	for saved, expected in zip(jax.tree_util.tree_leaves(saved_params), jax.tree_util.tree_leaves(expected_params)):
		assert np.array_equal(saved, expected)
	problem = prepare_rotation_problem(base_hamiltonian(lattice), result["gates"], basis, 4)
	energy = ritz_ground_state(build_rotated_subspace_hamiltonian(problem, jnp.asarray(saved_theta)), problem.j_rms).energy
	assert np.isclose(result["outer_history"][0]["ritz_energy"], energy)
	proposal_energy = ritz_ground_state(build_rotated_subspace_hamiltonian(problem, jnp.asarray(snapshots[1][1])), problem.j_rms).energy
	assert np.isclose(result["outer_history"][0]["proposal_ritz_metrics"]["ritz_energy"], proposal_energy)


@pytest.mark.parametrize("recovery_energy, standard_error, restore", [(-1.1, 0., True), (-.9, 0., True), (-1.1, .1, False)])
def test_recovery_retains_resolved_better_teacher_state(monkeypatch, tmp_path, recovery_energy, standard_error, restore):
	"""Keep both measured endpoints and save the selected parameters without reevaluation."""
	lattice = LatticeConfig(Lx=2, Ly=2, pbc=False)
	model_cfg = ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, conserve_total_sz=True)
	step = AdaptiveStep(train=TrainConfig(n_nqs=1, n_samples=16, use_sr=False), rotation=RotationConfig(n_rot=1, n_subspace=4, sampling="gumbel"), teacher=TeacherConfig(n_teacher=1), n_recovery=1)
	config = AdaptiveConfig(circuit=CircuitConfig(n_layers=1, family=("givens",)), steps=(step,), display="silent")
	original_evaluate = adaptive._evaluate_nqs
	snapshots = []

	def evaluate(*args):
		metrics = original_evaluate(*args)
		snapshots.append(args[2])
		metrics["energy"] = (-1., -.8, -1.2, recovery_energy)[len(snapshots)-1]
		metrics["standard_error"] = standard_error
		return metrics

	monkeypatch.setattr(adaptive, "_evaluate_nqs", evaluate)
	result = adaptive.run_adaptive_loop(lattice, model_cfg, config, output_dir=tmp_path)
	row = result["outer_history"][0]
	assert len(snapshots) == 4
	assert row["recovery_restored_start"] is restore
	assert row["energy_after_teacher"] == -1.2
	assert row["energy_after_recovery"] == recovery_energy
	assert row["accepted"]
	assert result["final_energy"] == (-1.2 if restore else recovery_energy)
	assert row["timing"]["recovery_seconds"] > 0.
	_, saved_params, _, _, _ = load_outer_checkpoint(tmp_path)
	expected = snapshots[2 if restore else 3]
	for saved, selected in zip(jax.tree_util.tree_leaves(saved_params), jax.tree_util.tree_leaves(expected)):
		assert np.array_equal(saved, selected)


def test_exact_rotation_probe_and_steady_timing():
	"""Physical U phi probes match H_theta expectations for noncommuting gates."""
	from src.operators import pauli_rotation_circuit, rotated_hamiltonian_terms
	lattice = LatticeConfig(hamiltonian="ising", Lx=2, Ly=2, pbc=False, hx=.7, hz=.03)
	model = make_transformer(lattice, ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, conserve_total_sz=False))
	states = jnp.asarray(model.hilbert.all_states())
	params = model.init(jax.random.PRNGKey(8), states)
	terms = base_hamiltonian(lattice)
	gates = pauli_rotation_circuit(lattice, 2, family=("XY",))
	probe = adaptive._exact_rotation_probe(lattice, model, params, terms, gates)
	phi = np.exp(np.asarray(model.apply(params, states)))
	phi /= np.linalg.norm(phi)
	for magnitude in (0., .19, -.31):
		theta = jnp.linspace(-magnitude, magnitude, 1 + max(gate.parameter for gate in gates))
		rotated, _ = rotated_hamiltonian_terms(terms, gates, theta, lattice.n_sites)
		h_phi = pauli_terms_to_netket(rotated, model.hilbert).to_sparse() @ phi
		energy = float(np.vdot(phi, h_phi).real)
		metrics = probe(theta)
		assert np.isclose(metrics["energy"], energy, atol=1e-10)
		assert np.isclose(metrics["variance"], np.vdot(h_phi, h_phi).real - energy**2, atol=1e-10)
	assert np.isclose(adaptive._steady_step_seconds([{"seconds": t} for t in (10., 13., 15., 15.2, 15.4, 15.6)]), .2)
	assert np.isnan(adaptive._steady_step_seconds([{"seconds": 1.}]))
	with pytest.raises(ValueError, match="nqs_energy_every"):
		RotationConfig(nqs_energy_every=-1)


def test_exact_teacher_probe_reuses_operator_for_changed_parameters():
	"""Teacher probes evaluate the current state without rebuilding H or using ED targets."""
	from types import SimpleNamespace
	lattice = LatticeConfig(hamiltonian="ising", Lx=2, Ly=2, pbc=False, hx=.7)
	model = make_transformer(lattice, ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1))
	states = jnp.asarray(model.hilbert.all_states())
	matrix = pauli_terms_to_netket(base_hamiltonian(lattice), model.hilbert).to_sparse()
	calls = []

	def to_sparse():
		calls.append(True)
		return matrix

	probe = adaptive._exact_teacher_probe(model, SimpleNamespace(to_sparse=to_sparse))
	for seed in (4, 9):
		params = model.init(jax.random.PRNGKey(seed), states)
		phi = np.exp(np.asarray(model.apply(params, states)))
		phi /= np.linalg.norm(phi)
		h_phi = matrix @ phi
		energy = float(np.vdot(phi, h_phi).real)
		metrics = probe(params)
		assert np.isclose(metrics["energy"], energy, atol=1e-12)
		assert np.isclose(metrics["variance"], np.vdot(h_phi, h_phi).real - energy**2, atol=1e-12)
	assert len(calls) == 1


def test_rotation_energy_trace_is_frozen_and_includes_endpoint(tmp_path):
	lattice = LatticeConfig(hamiltonian="ising", Lx=2, Ly=2, pbc=False, hx=.7, hz=.03)
	model_cfg = ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, conserve_total_sz=False)
	step = AdaptiveStep(train=TrainConfig(n_nqs=4, n_samples=16), rotation=RotationConfig(n_rot=4, n_subspace=8, nqs_energy_every=1), teacher=TeacherConfig(n_teacher=2, nqs_energy_every=1), n_recovery=4)
	config = AdaptiveConfig(circuit=CircuitConfig(n_layers=1, family=("XY",)), steps=(step,), display="silent")
	result = adaptive.run_adaptive_loop(lattice, model_cfg, config, seed=3, output_dir=tmp_path)
	row = result["outer_history"][0]
	trace = result["rotation_nqs_history"]
	assert [entry["step"] for entry in trace] == [0, 1, 2, 3, 4]
	assert trace[0]["energy"] == row["energy_before_rotation"]
	assert trace[-1]["energy"] == row["energy_after_rotation"]
	assert np.isclose(row["rotation_energy_change"], trace[-1]["energy"] - trace[0]["energy"])
	assert np.isfinite(row["vmc_slowdown"]) and row["vmc_slowdown"] > 0.
	assert row["proposal_max_conn"] >= row["max_conn_before"]
	assert row["ritz_participation_entropy"] >= 0. and row["ritz_renyi2_sum"] >= -1e-10
	assert row["timing"]["rotation_probe_seconds"] > 0.
	assert row["sampling"]["born_draws"] == step.rotation.n_seed_samples
	assert row["sampling"]["basis_size"] == step.rotation.n_subspace
	assert 0.0 < row["sampling"]["nqs_mass_before"] <= 1.0 + 1e-10
	teacher_trace = result["teacher_nqs_history"]
	assert [entry["step"] for entry in teacher_trace] == [0, 1, 2]
	assert teacher_trace[0]["energy"] == row["energy_after_rotation"]
	assert teacher_trace[-1]["energy"] == row["energy_after_teacher"]
	assert np.isclose(row["proposal_ritz_metrics"]["ritz_energy"], row["proposal_ritz_energy"])
	# Exact observation must not change training or consume its sampling stream.
	unobserved = dc.replace(config, steps=(dc.replace(step, teacher=dc.replace(step.teacher, nqs_energy_every=0)),))
	control = adaptive.run_adaptive_loop(lattice, model_cfg, unobserved, seed=3, output_dir=tmp_path / "control")
	assert np.isclose(result["final_energy"], control["final_energy"], atol=1e-12)
	for observed, expected in zip(jax.tree_util.tree_leaves(result["params"]), jax.tree_util.tree_leaves(control["params"])):
		assert np.array_equal(observed, expected)
	with pytest.raises(ValueError, match="nqs_energy_every"):
		TeacherConfig(nqs_energy_every=-1)


def test_monte_carlo_probes_do_not_advance_the_training_stream(monkeypatch, tmp_path):
	"""With the exact path disabled, energy probes must still leave params unchanged."""
	lattice 	= LatticeConfig(hamiltonian="ising", Lx=3, Ly=1, pbc=False, hx=.7, hz=.03)
	model_cfg 	= ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, conserve_total_sz=False)
	# Force the Monte Carlo probe branch even though these sites admit exact diagnostics.
	monkeypatch.setattr(adaptive, "_MAX_ED", 0)

	def build(every):
		step = AdaptiveStep(
			train=TrainConfig(n_nqs=2, n_samples=64, use_sr=False, optimizer="adam"),
			rotation=RotationConfig(n_rot=2, n_subspace=4, nqs_energy_every=every),
			teacher=TeacherConfig(n_teacher=0), n_recovery=2,
		)
		return AdaptiveConfig(circuit=CircuitConfig(n_layers=1, family=("XY",)), steps=(step,), display="silent")

	observed 	= adaptive.run_adaptive_loop(lattice, model_cfg, build(1), seed=3, output_dir=tmp_path / "obs")
	control 	= adaptive.run_adaptive_loop(lattice, model_cfg, build(0), seed=3, output_dir=tmp_path / "ctl")
	assert observed["rotation_nqs_history"]
	assert observed["final_energy"] == control["final_energy"]
	for observed_leaf, control_leaf in zip(jax.tree_util.tree_leaves(observed["params"]), jax.tree_util.tree_leaves(control["params"])):
		assert np.array_equal(observed_leaf, control_leaf)


def test_small_batch_mc_energy_reports_a_finite_standard_error(monkeypatch):
	"""A non-finite NetKet error over the mean falls back to sqrt(var/N), not an abort."""
	lattice 	= LatticeConfig(hamiltonian="ising", Lx=3, Ly=1, pbc=False, hx=.7)
	model 		= make_transformer(lattice, ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, conserve_total_sz=False))
	params 		= model.init(jax.random.PRNGKey(0), jnp.asarray(model.hilbert.all_states()))
	terms		= base_hamiltonian(lattice)
	operator 	= pauli_terms_to_netket(terms, model.hilbert)
	monkeypatch.setattr(adaptive, "_MAX_ED", 0)  # force the Monte Carlo branch
	with display_context("silent"):
		metrics = adaptive._evaluate_nqs(lattice, model, params, TrainConfig(n_nqs=0, n_samples=2), terms, [], jnp.zeros(0), terms, len(terms), operator, jax.random.PRNGKey(1), float("nan"), None)
	assert np.isfinite(metrics["standard_error"]) and metrics["standard_error"] >= 0.


def test_enabling_renyi_penalty_prepares_cuts_for_cached_basis(monkeypatch, tmp_path):
	"""Changing the loss must activate cut data even when sampled states stay fixed."""
	lattice 		= LatticeConfig(hamiltonian="ising", Lx=2, Ly=1, pbc=False, hx=.7)
	model_cfg 		= ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, conserve_total_sz=False)
	step 			= AdaptiveStep(train=TrainConfig(n_nqs=0, n_samples=16), rotation=RotationConfig(n_rot=0, n_subspace=4, n_seed_samples=256), teacher=TeacherConfig(n_teacher=0), n_recovery=0)
	penalized		= dc.replace(step, rotation=dc.replace(step.rotation, n_rot=1, entanglement_weight=.1))
	config 			= AdaptiveConfig(circuit=CircuitConfig(n_layers=1, family=("XY",)), steps=(step, penalized), display="silent")
	select, prepare = adaptive._selected_subspace, adaptive.prepare_rotation_problem
	prepared 		= []

	def sorted_basis(*args, **kwargs):
		basis = select(*args, **kwargs)
		assert len(basis) == 4
		return basis[np.lexsort(basis.T[::-1])]

	def record_preparation(*args, **kwargs):
		prepared.append(kwargs["include_entanglement"])
		return prepare(*args, **kwargs)

	monkeypatch.setattr(adaptive, "_selected_subspace", sorted_basis)
	monkeypatch.setattr(adaptive, "prepare_rotation_problem", record_preparation)
	result = adaptive.run_adaptive_loop(lattice, model_cfg, config, seed=2, output_dir=tmp_path)
	assert prepared == [False, True]
	assert result["rotation_history"][-1]["entanglement"] > 0.


def test_collapsed_seed_batch_reports_bounded_completion(monkeypatch):
	"""A one-state batch never triggers extra Born draws; completion is explicit."""
	import src.nqs as nqs
	lattice = LatticeConfig(hamiltonian="ising", Lx=2, Ly=1, pbc=False)
	model = make_transformer(lattice, ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1))
	params = model.init(jax.random.PRNGKey(4), jnp.asarray(model.hilbert.all_states()))
	born_calls, gumbel_calls = [], []

	def draw(_model, _params, _key, count):
		born_calls.append(count)
		return np.zeros((count, 2), dtype=np.int8)

	original = adaptive.gumbel_top_k_configs
	def gumbel(*args):
		gumbel_calls.append(args[-1])
		return original(*args)

	monkeypatch.setattr(nqs, "draw_configs", draw)
	monkeypatch.setattr(adaptive, "gumbel_top_k_configs", gumbel)
	statistics = {}
	with display_context("silent"):
		basis = adaptive._selected_subspace(model, params, jax.random.PRNGKey(6), [], 2, RotationConfig(n_subspace=4, n_seed_samples=7), np.empty((0, 2), dtype=np.int8), statistics=statistics)
	assert born_calls == [7] and gumbel_calls == [4]
	assert statistics["born_draws"] == statistics["born_draw_budget"] == 7
	assert statistics["unique_born_states"] == statistics["ranked_seed_states"] == 1
	assert statistics["gumbel_unique_draws"] == 4
	assert len(np.unique(basis, axis=0)) == statistics["basis_size"] == 4
	assert np.isclose(statistics["nqs_mass_before"], 1.)
	for kwargs in ({"n_subspace": 2.5}, {"n_seed_samples": 0}, {"n_seed_samples": 3.5}):
		with pytest.raises(ValueError):
			RotationConfig(**kwargs)


def test_restart_matches_uninterrupted_loop(tmp_path):
	"""Restart retains the accepted Ritz weighting as well as both parameter blocks."""
	lattice = LatticeConfig(Lx=2, Ly=2, pbc=False, j2=.3)
	model = ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, conserve_total_sz=True)
	step = AdaptiveStep(train=TrainConfig(n_nqs=1, n_samples=16, use_sr=False), rotation=RotationConfig(n_rot=1, n_subspace=4, f_retain=.5, sampling="gumbel"), teacher=TeacherConfig(n_teacher=1), n_recovery=1)
	config = AdaptiveConfig(circuit=CircuitConfig(n_layers=1, family=("givens",)), steps=(step, step), display="silent")
	full = adaptive.run_adaptive_loop(lattice, model, config, seed=7)
	adaptive.run_adaptive_loop(lattice, model, dc.replace(config, steps=(step,)), seed=7, output_dir=tmp_path)
	resumed = adaptive.run_adaptive_loop(lattice, model, dc.replace(config, restart_from=str(tmp_path)), seed=7)
	with pytest.raises(ValueError, match="Restart lattice"):
		adaptive.run_adaptive_loop(dc.replace(lattice, j2=.4), model, dc.replace(config, restart_from=str(tmp_path)), seed=7)
	assert np.allclose(full["theta_star"], resumed["theta_star"], atol=1e-12)
	assert np.isclose(full["final_energy"], resumed["final_energy"], atol=1e-12)
	for first, second in zip(jax.tree_util.tree_leaves(full["params"]), jax.tree_util.tree_leaves(resumed["params"])):
		assert np.allclose(first, second, atol=1e-12)


def test_zero_update_step_reuses_state_and_preserves_sampling_key(monkeypatch, tmp_path):
	"""An existing NQS needs no duplicate MC evaluation before its phase measurement."""
	lattice = LatticeConfig(hamiltonian="ising", Lx=2, Ly=2, pbc=False, hx=.7)
	model_cfg = ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1)
	step = AdaptiveStep(train=TrainConfig(n_nqs=0, n_samples=16), rotation=RotationConfig(n_rot=0, n_subspace=4, n_seed_samples=16, sampling="gumbel"), teacher=TeacherConfig(n_teacher=0), n_recovery=0)
	config = AdaptiveConfig(circuit=CircuitConfig(n_layers=1, family=("Y",)), steps=(step,), display="silent")
	initial = adaptive.run_adaptive_loop(lattice, model_cfg, config, seed=7, output_dir=tmp_path)
	_, saved_params, theta, saved_key, _ = load_outer_checkpoint(tmp_path)
	# Obtain the old sampling schedule from the real zero-update training path.
	with display_context("silent"):
		reference = adaptive.train_nqs(lattice, model_cfg, step.train, base_hamiltonian(lattice), initial["gates"], jnp.asarray(theta), saved_params, jnp.asarray(saved_key))
	evaluation_key, _ = jax.random.split(reference["key"])
	_, expected_sample_key = jax.random.split(evaluation_key)
	selected_keys = []
	original_select = adaptive._selected_subspace

	def select(model, params, key, *args, **kwargs):
		selected_keys.append(np.asarray(key))
		return original_select(model, params, key, *args, **kwargs)

	def unexpected_training(*args, **kwargs):
		raise AssertionError("Zero-update continuation must not build or evaluate a VMC state")

	monkeypatch.setattr(adaptive, "train_nqs", unexpected_training)
	monkeypatch.setattr(adaptive, "_selected_subspace", select)
	resumed = adaptive.run_adaptive_loop(lattice, model_cfg, dc.replace(config, steps=(step, step), restart_from=str(tmp_path)), seed=7)
	assert len(selected_keys) == 1
	assert np.array_equal(selected_keys[0], expected_sample_key)
	assert resumed["nqs_history"] == []
	assert np.isclose(resumed["final_energy"], initial["final_energy"], atol=1e-12)
	assert resumed["outer_history"][0]["energy_before_rotation"] == resumed["outer_history"][0]["energy_after_rotation"]
	for first, second in zip(jax.tree_util.tree_leaves(saved_params), jax.tree_util.tree_leaves(resumed["params"])):
		assert np.array_equal(first, second)
