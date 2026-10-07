"""Cached Born sampling and the regularized minSR training defaults."""

import dataclasses as dc

import flax.core
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

from src.config import LatticeConfig, ModelConfig, RotationConfig, TrainConfig, default_n_samples, default_n_subspace, default_n_seed_samples
from src.model import ARTransformer
from src.netket_vmc import _optimizer
from src.nqs import make_transformer


@pytest.mark.parametrize("sector", ["full", "spin", "fermion"])
@pytest.mark.parametrize("phase_scale", [0., 1.])
def test_cached_conditionals_match_full_sequence(sector, phase_scale):
	"""Cached decoding preserves nonuniform probabilities and both sector masks."""
	cfg = LatticeConfig(hamiltonian="ising", Lx=3, Ly=2, pbc=False, hx=.7)
	model_cfg = ModelConfig(width=8, depth=2, heads=2, mlp_ratio=1, phase_scale=phase_scale, conserve_total_sz=sector == "spin", n_up=2)
	if sector == "fermion":
		model = ARTransformer(hilbert=nk.hilbert.SpinOrbitalFermions(3, s=.5), n_sites=6,
			width=8, depth=2, heads=2, mlp_ratio=1, phase_scale=phase_scale, local_occupations=True, particle_counts=(1, 2))
	else:
		model = make_transformer(cfg, model_cfg)
	states = np.asarray(model.hilbert.all_states())
	if sector == "spin":
		states = states[(states == 1).sum(axis=1) == 2]
	elif sector == "fermion":
		states = states[(states[:, :3].sum(axis=1) == 1) & (states[:, 3:].sum(axis=1) == 2)]
	params = flax.core.unfreeze(model.init(jax.random.PRNGKey(0), jnp.asarray(states[:1])))
	kernel = params["params"]["amplitude_head"]["kernel"]
	params["params"]["amplitude_head"]["kernel"] = .2 * jax.random.normal(jax.random.PRNGKey(1), kernel.shape)
	full = np.asarray(model.apply(params, jnp.asarray(states), method=model.conditionals))
	log_psi = np.asarray(model.apply(params, jnp.asarray(states)))
	assert np.isclose(np.exp(2 * log_psi.real).sum(), 1., atol=2e-6)
	for _ in range(2):
		cache = model.init(jax.random.PRNGKey(2), jnp.asarray(states), 0, method=model.conditional)["cache"]
		for site in range(6):
			probability, mutable = model.apply({**params, "cache": cache}, jnp.asarray(states), site,
				method=model.conditional, mutable=["cache"])
			assert np.allclose(probability, full[:, site], atol=2e-6)
			cache = mutable["cache"]


def test_positive_nqs_uses_real_sr_and_preserves_reference_signs():
	"""Disabling learned phases removes their parameters, without changing Born normalization."""
	from src.netket_vmc import pauli_terms_to_netket
	from src.operators import base_hamiltonian
	cfg = LatticeConfig(hamiltonian="ising", Lx=2, Ly=2, pbc=False, hx=.7)
	model_cfg = ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, phase_scale=0.)
	model = make_transformer(cfg, model_cfg)
	states = jnp.asarray(model.hilbert.all_states())
	params = model.init(jax.random.PRNGKey(2), states)
	assert not any("phase" in name for name in params["params"])
	logs = np.asarray(model.apply(params, states))
	assert not np.iscomplexobj(logs)
	assert np.isclose(np.exp(2*logs).sum(), 1., atol=2e-6)
	vstate = nk.vqs.MCState(nk.sampler.ARDirectSampler(model.hilbert), model, variables=params, n_samples=16, sampler_seed=3)
	op = pauli_terms_to_netket(base_hamiltonian(cfg), model.hilbert)
	driver = nk.driver.VMC_SR(op, nk.optimizer.Sgd(.02), variational_state=vstate, diag_shift=.01, use_ntk=True, on_the_fly=False)
	assert driver.mode == "real"
	# A fixed reference sign remains active even though the learned phase is disabled.
	signed = model.clone(reference_phase=(1.,0.,0.,0.))
	signed_logs = np.asarray(signed.apply(params, states))
	bits = np.asarray((1-states)//2)
	assert np.allclose(np.exp(signed_logs), np.exp(logs)*(1-2*bits[:,0]), atol=2e-6)
	with pytest.raises(ValueError, match="phase_scale"):
		ModelConfig(phase_scale=-1.)


@pytest.mark.parametrize("stop", ["time", "variance"])
@pytest.mark.parametrize("phase_scale", [0., 1.])
def test_vmc_observer_and_stopping_follow_completed_update(stop, phase_scale):
	"""A stopped first step must apply the same update as an ordinary one-step run."""
	from src.logging_utils import display_context
	from src.netket_vmc import pauli_terms_to_netket, train_vmc
	from src.operators import base_hamiltonian
	lattice = LatticeConfig(hamiltonian="ising", Lx=2, Ly=2, pbc=False, hx=.7)
	model = make_transformer(lattice, ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, phase_scale=phase_scale))
	params = model.init(jax.random.PRNGKey(7), jnp.asarray(model.hilbert.all_states()[:1]))
	operator = pauli_terms_to_netket(base_hamiltonian(lattice), model.hilbert)
	config = TrainConfig(n_samples=16, n_nqs=1)
	key = jax.random.PRNGKey(11)
	observed = []
	with display_context("silent"):
		expected, normal_history, _ = train_vmc(params, model, operator, config, key, on_step=lambda index, variables: observed.append((index, variables)))
	assert len(normal_history) == 1
	assert [index for index, _ in observed] == [1]
	assert any(not np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(expected)))
	for actual, updated in zip(jax.tree_util.tree_leaves(observed[0][1]), jax.tree_util.tree_leaves(expected)):
		assert np.array_equal(actual, updated)
	observed.clear()
	stopping_config = dc.replace(config, n_nqs=3, early_stop_var=1e9 if stop == "variance" else 0.)
	with display_context("silent"):
		actual, history, _ = train_vmc(params, model, operator, stopping_config, key, time_budget_seconds=1e-12 if stop == "time" else None, on_step=lambda index, variables: observed.append((index, variables)))
	assert len(history) == 1
	assert history[0]["energy"] == normal_history[0]["energy"]
	assert [index for index, _ in observed] == [1]
	for stopped, one_step, snapshot in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(observed[0][1])):
		assert np.array_equal(stopped, one_step)
		assert np.array_equal(snapshot, one_step)


def test_training_defaults_follow_device_and_preserve_explicit_choices(monkeypatch):
	"""Automatic batches follow the selected device; explicit batch/clip values win."""
	assert default_n_samples("cpu") == 256
	assert default_n_samples("gpu") == default_n_samples("tpu") == 1024
	assert default_n_subspace("cpu") == 128
	assert default_n_subspace("gpu") == default_n_subspace("tpu") == 256
	assert default_n_seed_samples(128, "cpu") == 4096
	assert default_n_seed_samples(256, "gpu") == 8192
	assert default_n_seed_samples(512, "gpu") == 16384
	monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
	with jax.default_device(None):
		assert TrainConfig().n_samples == 1024
		assert RotationConfig().n_subspace == 256
		assert RotationConfig().n_seed_samples == 8192
		assert RotationConfig(n_subspace=512).n_seed_samples == 16384
		assert RotationConfig(n_subspace=64, n_seed_samples=100).n_seed_samples == 100
		assert TrainConfig(n_samples=64).n_samples == 64
		assert TrainConfig(n_samples=64, grad_clip=.5).grad_clip == .5
		assert TrainConfig(n_samples=64, use_sr=False).grad_clip == 1.
	with jax.default_device(jax.devices("cpu")[0]):
		assert TrainConfig().n_samples == 256
		assert RotationConfig().n_subspace == 128
		assert RotationConfig().n_seed_samples == 4096
	cfg = TrainConfig(n_samples=64)
	assert cfg.grad_clip == 0. and cfg.sr_shift == .01 and cfg.sr_momentum == .8
	# An SR direction with norm > 1 must reach SGD without hidden rescaling.
	optimizer, _, _ = _optimizer(cfg)
	direction = {"weight": jnp.array([3., 4.])}
	updates, _ = optimizer.update(direction, optimizer.init(direction))
	assert np.allclose(updates["weight"], -.02 * direction["weight"])
	clipped, _, _ = _optimizer(dc.replace(cfg, grad_clip=1.))
	updates, _ = clipped.update(direction, clipped.init(direction))
	assert np.isclose(np.sqrt(np.sum(np.asarray(updates["weight"])**2)), .02)


def test_ising_circuit_can_represent_transverse_field_basis():
	"""The decoupled field limit distinguishes a suitable basis from pair parity rotations."""
	from src.nqs import apply_rotation_adjoint_to_vector
	from src.operators import pauli_rotation_circuit
	cfg = LatticeConfig(hamiltonian="ising", Lx=3, Ly=3, pbc=False, j1=0., hx=.7, hz=0.)
	states = np.asarray(nk.hilbert.Spin(s=.5, N=cfg.n_sites).all_states())
	phi = np.ones(len(states)) / np.sqrt(len(states))
	xy = pauli_rotation_circuit(cfg, 2, family=("XY",))
	assert all(x.bit_count() % 2 == 0 for gate in xy for x, _, _ in gate.generators)
	assert all(cfg.n_sites - 1 not in gate.support for gate in xy)
	rotated_xy = apply_rotation_adjoint_to_vector(phi, states, xy, np.linspace(-.5, .4, len(xy)), cfg.n_sites)
	odd = (((1 - states) // 2).sum(axis=1).astype(int) % 2) == 1
	assert np.isclose(np.sum(abs(rotated_xy[odd])**2), .5)
	y = pauli_rotation_circuit(cfg, 1, family=("Y",))
	assert {site for gate in y for site in gate.support} == set(range(cfg.n_sites))
	rotated_y = apply_rotation_adjoint_to_vector(phi, states, y, np.full(len(y), np.pi / 2), cfg.n_sites)
	pole = np.all(states == 1, axis=1)
	assert np.isclose(np.sum(abs(rotated_y[pole])**2), 1., atol=1e-12)
	assert np.linalg.norm(rotated_y[~pole]) < 1e-12


@pytest.mark.parametrize("longitudinal_field", [.03, -.03])
def test_ising_field_basis_has_nonpositive_off_diagonal_elements(longitudinal_field):
	"""The sign of the quarter-turn matters when both fields are present."""
	from src.netket_vmc import pauli_terms_to_netket
	from src.operators import base_hamiltonian, pauli_rotation_circuit, pauli_term, rotated_hamiltonian_terms
	cfg = LatticeConfig(hamiltonian="ising", Lx=2, Ly=2, pbc=False, j1=1., hx=.7, hz=longitudinal_field)
	gates = pauli_rotation_circuit(cfg, 1, family=("Y",))
	sign = np.sign(longitudinal_field)
	terms, _ = rotated_hamiltonian_terms(base_hamiltonian(cfg), gates, np.full(len(gates), -sign*np.pi/2), cfg.n_sites)
	# S=Pauli/2: H'=-J SxSx + sign(hz) hx Sz - |hz| Sx.
	expected = [pauli_term({a:"X", b:"X"}, -.25*cfg.j1) for a,b in ((0,1),(2,3),(0,2),(1,3))]
	expected += [pauli_term({site:"Z"}, .5*sign*cfg.hx) for site in range(cfg.n_sites)]
	expected += [pauli_term({site:"X"}, -.5*abs(cfg.hz)) for site in range(cfg.n_sites)]
	actual = {(term.x,term.z):complex(term.coefficient) for term in terms}
	assert set(actual) == {(term.x,term.z) for term in expected}
	for term in expected:
		assert np.isclose(actual[term.x,term.z], term.coefficient, atol=1e-12)
	hilbert = nk.hilbert.Spin(s=.5,N=cfg.n_sites)
	matrix = pauli_terms_to_netket(terms,hilbert).to_sparse().toarray()
	off_diagonal = matrix-np.diag(np.diag(matrix))
	assert np.max(abs(off_diagonal.imag)) < 1e-12
	assert np.max(off_diagonal.real) <= 1e-12
	wrong_terms, _ = rotated_hamiltonian_terms(base_hamiltonian(cfg), gates, np.full(len(gates), sign*np.pi/2), cfg.n_sites)
	wrong = pauli_terms_to_netket(wrong_terms,hilbert).to_sparse().toarray()
	assert np.max((wrong-np.diag(np.diag(wrong))).real) > 0.


def test_reference_precision_is_independent_of_vmc_precision():
	"""An analytic decoupled-spin reference stays accurate with float32 VMC."""
	from src.netket_vmc import exact_ground_state, netket_hilbert, pauli_terms_to_netket
	from src.operators import base_hamiltonian
	cfg = LatticeConfig(hamiltonian="ising", Lx=2, Ly=1, pbc=False, j1=0., hx=.71234567, hz=.13234567)
	previous = jax.config.jax_enable_x64
	try:
		jax.config.update("jax_enable_x64", False)
		energy, vector = exact_ground_state(cfg)
		assert not jax.config.jax_enable_x64
		with jax.enable_x64(True):
			matrix = pauli_terms_to_netket(base_hamiltonian(cfg), netket_hilbert(cfg)).to_sparse()
		assert matrix.dtype == np.complex128
		assert np.isclose(energy, -.5 * cfg.n_sites * np.hypot(cfg.hx, cfg.hz), atol=1e-12, rtol=0)
		assert np.linalg.norm(matrix @ vector - energy * vector) < 1e-12
	finally:
		jax.config.update("jax_enable_x64", previous)


def test_ising_3x3_vmc_converges_against_exact_reference():
	"""Stochastic 3x3 training must learn the state without any Ritz teacher."""
	from src.logging_utils import display_context
	from src.netket_vmc import exact_ground_state, pauli_terms_to_netket
	from src.nqs import train_nqs
	from src.operators import base_hamiltonian

	cfg = LatticeConfig(hamiltonian="ising", Lx=3, Ly=3, pbc=False, hx=.7, hz=.03)
	model_cfg = ModelConfig(width=16, depth=2, heads=4, mlp_ratio=2)
	terms = base_hamiltonian(cfg)
	with display_context("silent"):
		trained = train_nqs(cfg, model_cfg, TrainConfig(n_nqs=200, n_samples=256), terms,
			[], jnp.zeros(0), None, jax.random.PRNGKey(0), fresh_init=True)
	model = trained["model"]
	psi = np.exp(np.asarray(model.apply(trained["params"], jnp.asarray(model.hilbert.all_states()))))
	psi /= np.linalg.norm(psi)
	h_psi = pauli_terms_to_netket(terms, model.hilbert).to_sparse() @ psi
	energy = float(np.vdot(psi, h_psi).real)
	variance = float(np.vdot(h_psi, h_psi).real - energy**2)
	reference = exact_ground_state(cfg)[0]
	assert -1e-6 < (energy - reference) / abs(reference) < .005
	assert 0. <= variance < .02


def test_diagnostic_cache_uses_current_parameters_and_exact_reference(monkeypatch):
	"""Compiled diagnostics keep live parameters; final ED reuses the full-space oracle."""
	import src.adaptive as adaptive
	from src.config import AdaptiveConfig, AdaptiveStep, CircuitConfig, FinalConfig, RotationConfig, TeacherConfig
	from src.netket_vmc import exact_ground_state, pauli_terms_to_netket
	from src.nqs import _diagnostic_apply, ed_diagnostics, exact_basis_entropy_comparison
	from src.operators import base_hamiltonian

	cfg = LatticeConfig(hamiltonian="ising", Lx=2, Ly=1, pbc=False, hx=.7, hz=.03)
	model_cfg = ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1)
	model = make_transformer(cfg, model_cfg)
	params = flax.core.unfreeze(model.init(jax.random.PRNGKey(0), jnp.ones((1, 2))))
	terms = base_hamiltonian(cfg)
	reference, vector = exact_ground_state(cfg)
	op = pauli_terms_to_netket(terms, model.hilbert)
	first = ed_diagnostics(cfg, terms, [], jnp.zeros(0), model, params, reference, vector, operator=op)
	params["params"]["amplitude_head"]["bias"] = jnp.array([1., -1.])
	second = ed_diagnostics(cfg, terms, [], jnp.zeros(0), model, params, reference, vector, operator=op)
	assert _diagnostic_apply(model) is _diagnostic_apply(model)
	assert not np.isclose(first["variational_energy"], second["variational_energy"])
	psi = np.exp(np.asarray(model.apply(params, jnp.asarray(model.hilbert.all_states()))))
	expected = float((np.vdot(psi, op.to_sparse() @ psi) / np.vdot(psi, psi)).real)
	assert np.isclose(second["variational_energy"], expected)
	bell = np.array([1., 0., 0., 1.]) / np.sqrt(2.)
	entropy = exact_basis_entropy_comparison(cfg, [], jnp.zeros(0), bell)
	assert np.isclose(entropy["ed_all_cuts_s2_before_trainable_rotation"], np.log(2.))

	def duplicate_solve(*args, **kwargs):
		raise AssertionError("The exact reference was already computed")

	monkeypatch.setattr(adaptive, "solve_final", duplicate_solve)
	step = AdaptiveStep(train=TrainConfig(n_nqs=0, n_samples=16),
		rotation=RotationConfig(n_rot=0, n_subspace=4, n_seed_samples=32),
		teacher=TeacherConfig(n_teacher=0), n_recovery=0)
	protocol = AdaptiveConfig(circuit=CircuitConfig(n_layers=0), steps=(step,),
		final=FinalConfig(solver="exact"), display="silent")
	result = adaptive.run_adaptive_loop(cfg, model_cfg, protocol)
	assert result["final_reference"]["reused"]
	assert result["final_reference"]["energy"] == reference
