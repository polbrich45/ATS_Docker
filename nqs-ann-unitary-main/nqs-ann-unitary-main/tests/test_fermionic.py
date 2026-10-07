"""One fermionic-loop regression across exact and sampled Hubbard sectors."""

import dataclasses as dc

import jax
import numpy as np
import pytest

from src.checkpoint import load_outer_checkpoint
from src.config import AdaptiveConfig, AdaptiveStep, ModelConfig, RotationConfig, TeacherConfig, TrainConfig
from src.fermionic import HubbardConfig, run_fermionic_loop
from src.nqs import draw_configs, gumbel_top_k_configs

jax.config.update("jax_enable_x64", True)


def test_fermionic_loop_sector_rotation_and_restart(tmp_path):
	pytest.importorskip("openfermion")
	model_cfg = ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1)
	step = AdaptiveStep(
		train=TrainConfig(n_nqs=1, n_samples=16, use_sr=False, optimizer="adam"),
		rotation=RotationConfig(n_rot=2, n_subspace=8, sampling="gumbel"),
		teacher=TeacherConfig(n_teacher=1), n_recovery=1,
	)
	config = AdaptiveConfig(steps=(step,), display="silent")
	hubbard = HubbardConfig()
	result = run_fermionic_loop(hubbard, model_cfg, config, seed=3, output_dir=tmp_path)
	row = result["outer_history"][0]
	assert np.isfinite(result["final_energy"])
	assert result["exact_energy"] <= result["final_energy"] + 1e-8
	assert row["proposal_ritz_energy"] <= row["ritz_energy_before_rotation"] + 1e-8
	assert row["subspace_size"] == 8
	assert result["teacher_history"] and result["recovery_history"]
	assert row["energy"] == (row["energy_after_recovery"] if row["accepted"] else row["energy_before_rotation"])
	model, params = result["model"], result["params"]
	for samples in (draw_configs(model, params, jax.random.PRNGKey(1), 16), gumbel_top_k_configs(model, params, jax.random.PRNGKey(2), 8)):
		assert np.all(samples[:, :4].sum(axis=1) == 2)
		assert np.all(samples[:, 4:].sum(axis=1) == 2)
	_, saved_params, saved_theta, _, saved_basis = load_outer_checkpoint(tmp_path)
	assert np.array_equal(saved_theta, result["theta_star"])
	assert np.array_equal(saved_basis, result["basis"])
	for saved, expected in zip(jax.tree_util.tree_leaves(saved_params), jax.tree_util.tree_leaves(params)):
		assert np.array_equal(saved, expected)
	quiet = dc.replace(step, train=dc.replace(step.train, n_nqs=0), rotation=dc.replace(step.rotation, n_rot=0), teacher=dc.replace(step.teacher, n_teacher=0), n_recovery=0)
	two_steps = dc.replace(config, steps=(step, quiet))
	full = run_fermionic_loop(hubbard, model_cfg, two_steps, seed=3)
	resumed = run_fermionic_loop(hubbard, model_cfg, dc.replace(two_steps, restart_from=str(tmp_path)), seed=3)
	assert len(resumed["outer_history"]) == 1
	assert np.isclose(resumed["final_energy"], full["final_energy"], atol=1e-12)
	assert np.allclose(resumed["theta_star"], full["theta_star"], atol=1e-12)
	for restarted, uninterrupted in zip(jax.tree_util.tree_leaves(resumed["params"]), jax.tree_util.tree_leaves(full["params"])):
		assert np.allclose(restarted, uninterrupted, atol=1e-12)
	with pytest.raises(ValueError, match="differs"):
		run_fermionic_loop(dc.replace(hubbard, U=3.), model_cfg, dc.replace(two_steps, restart_from=str(tmp_path)), seed=3)
	with pytest.raises(ValueError, match="seed differs"):
		run_fermionic_loop(hubbard, model_cfg, dc.replace(two_steps, restart_from=str(tmp_path)), seed=4)
	# Six spatial sites already exceed the exact-oracle limit; run the same loop
	# with sampled energy and determinant connectivity, not a dense sector matrix.
	larger = HubbardConfig(Lx=3, Ly=2)
	assert (larger.n_up, larger.n_down, larger.sector_size) == (3, 3, 400)
	sampled_step = dc.replace(quiet, train=dc.replace(quiet.train, n_nqs=1))
	sampled = run_fermionic_loop(larger, model_cfg, dc.replace(config, steps=(sampled_step,)), seed=3)
	assert sampled["exact_energy"] is None
	assert np.isfinite(sampled["final_energy"])
	assert sampled["outer_history"][0]["standard_error"] >= 0.
	assert sampled["basis"].shape == (8, 12)
	assert np.all(sampled["basis"][:, :6].sum(axis=1) == 3)
	assert np.all(sampled["basis"][:, 6:].sum(axis=1) == 3)


@pytest.mark.parametrize("order", ("spatial", "model"))
def test_fermionic_basin_search_uses_configured_representation_objective(order):
	pytest.importorskip("openfermion")
	step = AdaptiveStep(
		train=TrainConfig(n_nqs=1, n_samples=16, use_sr=False, optimizer="adam"),
		rotation=RotationConfig(n_rot=2, n_basin=1, n_subspace=8, sampling="gumbel", participation_weight=.1, entanglement_weight=.1, entanglement_order=order),
		teacher=TeacherConfig(n_teacher=1), n_recovery=1,
	)
	result = run_fermionic_loop(HubbardConfig(), ModelConfig(width=4, depth=0, heads=1, mlp_ratio=1), AdaptiveConfig(steps=(step,), display="silent"), seed=9)
	rotation = result["rotation_history"][0]
	assert rotation["entanglement_order"] == order
	assert rotation["objective"] <= rotation["objective_before"] + 1e-6
	assert rotation["objective_evaluations"] > 0
	assert np.isfinite(rotation["entanglement_entropy"])
	assert np.isfinite(rotation["ipr"])
	outer = result["outer_history"][0]
	assert outer["energy"] <= outer["energy_before_rotation"] + 1e-8
