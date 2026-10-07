"""Paired adaptive versus ordinary NQS at equal updates and measured wall time.

Run sequentially on an otherwise idle machine. Compilation, sampling, Ritz,
teaching, recovery, acceptance and checkpoint I/O count toward adaptive cost.
The plain baseline spends that time on VMC, with the same NQS and warm state.

``--ablation`` instead runs four outer-loop arms (continued VMC, teacher only,
rotation only, full loop) from the same warm checkpoint. These arms match VMC
update budgets, not total cost.
"""

import argparse
import dataclasses as dc
import math
import platform
from importlib.metadata import version
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from src.adaptive import run_adaptive_loop
from src.checkpoint import load_outer_checkpoint, save_weights, write_json
from src.config import AdaptiveConfig, AdaptiveStep, CircuitConfig, LatticeConfig, ModelConfig, RotationConfig, TeacherConfig, TrainConfig
from src.fermionic import HubbardConfig, _evaluate, _projected_matrix, run_fermionic_loop
from src.logging_utils import display_context
from src.netket_vmc import pauli_terms_to_netket, train_vmc, rotated_netket_hamiltonian
from src.nqs import _MAX_ED
from src.operators import base_hamiltonian


def polish_warm_state(model, params, exact, maxiter):
	"""Small-sector stationarity diagnostic, shared by all benchmark branches.

	This uses the existing <=64-state oracle. Curvature is checked only for
	<=256 real parameters; a semidefinite Hessian is a local diagnostic, not
	a proof of a global minimum or of stability along all flat directions.
	"""
	if exact is None or len(exact[0]) > 64:
		raise ValueError("warm polishing requires an exact sector of at most 64 states")
	started = time.perf_counter()
	flat, unravel = ravel_pytree(params)
	if jnp.iscomplexobj(flat):
		raise ValueError("warm polishing expects real NQS parameters")
	if len(flat) > 256:
		raise ValueError("warm curvature checks are limited to 256 real parameters")
	states, matrix = jnp.asarray(exact[0]), jnp.asarray(exact[1])

	def energy(parameters):
		log_psi = model.apply(unravel(parameters), states)
		psi = jnp.exp(log_psi - jnp.max(log_psi.real))
		return jnp.real(jnp.vdot(psi, matrix @ psi)) / jnp.real(jnp.vdot(psi, psi))

	value_and_grad = jax.jit(jax.value_and_grad(energy))
	curvature = jax.jit(jax.hessian(energy))

	def objective(parameters):
		value, gradient = value_and_grad(jnp.asarray(parameters))
		if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
			raise FloatingPointError("warm-state energy or gradient is non-finite")
		return float(value), np.asarray(gradient)

	initial_energy = float(energy(flat))
	optimized = minimize(objective, np.asarray(flat), jac=True, hess=lambda point: np.asarray(curvature(jnp.asarray(point))), method="trust-exact", options={"maxiter": maxiter, "gtol": 1e-8})
	point = jnp.asarray(optimized.x) if optimized.fun <= initial_energy else flat
	final_energy, gradient = objective(point)
	hessian = np.asarray(curvature(point))
	if not np.all(np.isfinite(hessian)):
		raise FloatingPointError("warm-state curvature is non-finite")
	minimum_curvature = float(np.linalg.eigvalsh((hessian + hessian.T) / 2)[0])
	gradient_max = float(np.max(np.abs(gradient)))
	report = {"solver": "trust-exact", "gradient_tolerance": 1e-7, "curvature_tolerance": 1e-7, "initial_energy": initial_energy, "energy": final_energy, "gradient_max": gradient_max, "minimum_curvature": minimum_curvature, "stationarity_checks_passed": gradient_max < 1e-7 and minimum_curvature >= -1e-7, "parameters": len(flat), "log_probability_range": float(np.ptp(2 * np.real(np.asarray(model.apply(unravel(point), states))))), "iterations": int(optimized.nit), "evaluations": int(optimized.nfev), "optimizer_success": bool(optimized.success), "seconds": time.perf_counter() - started}
	return unravel(point), report


def vmc_control(model, params, operator, exact, cfg, key, n_terms, evaluation_key, checkpoint_every, initial_energy, budget=None):
	"""Plain VMC with retained initial, periodic and final checkpoints."""
	evaluation_samples = max(16 * cfg.n_samples, 1024)
	best = {"params": params, "energy": initial_energy, "step": 0}

	def retain_best(step_index, variables):
		if step_index % checkpoint_every:
			return
		metrics = _evaluate(model, variables, operator, jax.random.fold_in(evaluation_key, step_index), evaluation_samples, exact)
		if metrics["energy"] < best["energy"]:
			best.update(params=variables, energy=metrics["energy"], step=step_index)

	started = time.perf_counter()
	with display_context("silent"):
		final_params, history, _ = train_vmc(params, model, operator, cfg, jnp.asarray(key), n_terms=n_terms, time_budget_seconds=budget, on_step=retain_best)
	jax.block_until_ready(final_params)
	seconds = time.perf_counter() - started
	final = _evaluate(model, final_params, operator, evaluation_key, evaluation_samples, exact)
	if final["energy"] < best["energy"]:
		best.update(params=final_params, energy=final["energy"], step=len(history))
	metrics = _evaluate(model, best["params"], operator, jax.random.fold_in(evaluation_key, 2), evaluation_samples, exact)
	return best["params"], {"metrics": metrics, "final": final, "best_step": best["step"], "seconds": seconds, "updates": len(history) if cfg.n_nqs else 0, "history": history}


def comparison_summary(rows):
	"""Summarize independent seeds and distinguish teaching from rotation gains."""
	paired = [row for row in rows if row["baseline"] == "vmc_time"]
	if not paired or len({row["seed"] for row in paired}) != len(paired):
		raise ValueError("comparison requires one time-matched row per independent seed")

	def statistics(differences, errors):
		differences = np.asarray(differences)
		return {
			"mean_paired_difference": float(differences.mean()),
			"seed_standard_error": float(differences.std(ddof=1) / np.sqrt(len(paired))) if len(paired) > 1 else None,
			"resolved_wins": int(sum(difference < -max(2 * error, 1e-8) for difference, error in zip(differences, errors))),
		}

	summary = statistics([row["difference"] for row in paired], [row["difference_standard_error"] for row in paired])
	summary.update(seeds=len(paired), all_time_budgets_reached=all(row["budget_reached"] for row in paired))
	if all("teacher_control" in row for row in paired):
		summary["teacher_comparison"] = statistics(
			[row["adaptive"]["energy"] - row["teacher_control"]["energy"] for row in paired],
			[math.hypot(row["adaptive"]["standard_error"], row["teacher_control"]["standard_error"]) for row in paired],
		)
		summary["teacher_comparison"]["all_time_budgets_reached"] = all(row["teacher_budget_reached"] for row in paired)
	return summary


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--system", choices=("spin", "hubbard"), default="spin")
	parser.add_argument("--hamiltonian", choices=("j1j2", "ising"), default="j1j2")
	parser.add_argument("--hx", type=float, default=.7)
	parser.add_argument("--hz", type=float, default=.03)
	parser.add_argument("--float32", action="store_true")
	parser.add_argument("--output-dir", default="runs/budget-benchmark")
	parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
	parser.add_argument("--Lx", type=int, default=3)
	parser.add_argument("--Ly", type=int, default=2)
	parser.add_argument("--U", type=float, default=4.)
	parser.add_argument("--j2", type=float, default=.5)
	parser.add_argument("--warmup", type=int, default=60)
	parser.add_argument("--polish-warmup", type=int, default=0)
	parser.add_argument("--learning-rate", type=float, default=.02)
	parser.add_argument("--baseline-learning-rate", type=float, default=None)
	parser.add_argument("--warm-learning-rate", type=float, default=None)
	parser.add_argument("--n-outer", type=int, default=2)
	parser.add_argument("--n-nqs", type=int, default=30)
	parser.add_argument("--n-rot", type=int, default=30)
	parser.add_argument("--n-basin", type=int, default=0)
	parser.add_argument("--n-teacher", type=int, default=50)
	parser.add_argument("--teacher-learning-rate", type=float, default=.001)
	parser.add_argument("--teacher-optimizer", choices=("adam", "lbfgs"), default="adam")
	parser.add_argument("--phase-weight", type=float, default=.1)
	parser.add_argument("--n-recovery", type=int, default=20)
	parser.add_argument("--n-subspace", type=int, default=8)
	parser.add_argument("--n-samples", type=int, default=None)
	parser.add_argument("--width", type=int, default=8)
	parser.add_argument("--depth", type=int, default=1)
	parser.add_argument("--heads", type=int, default=2)
	parser.add_argument("--mlp-ratio", type=int, default=1)
	parser.add_argument("--phase-head-depth", type=int, default=1)
	parser.add_argument("--phase-init-std", type=float, default=.05)
	parser.add_argument("--phase-scale", type=float, default=1., help="0 disables learned phases; without a reference phase the NQS is positive and uses real SR")
	parser.add_argument("--sr-shift", type=float, default=1e-3)
	parser.add_argument("--n-layers", type=int, default=2)
	parser.add_argument("--rotation-family", nargs="+", default=None, help="Spin rotation generators; defaults to XY for Ising and givens otherwise")
	parser.add_argument("--rotation-learning-rate", type=float, default=.03)
	parser.add_argument("--sampling", choices=("frequency", "gumbel"), default="gumbel")
	parser.add_argument("--n-seed-samples", type=int, default=None)
	parser.add_argument("--nqs-energy-every", type=int, default=0)
	parser.add_argument("--participation-weight", type=float, default=0.)
	parser.add_argument("--entanglement-weight", type=float, default=0.)
	parser.add_argument("--entanglement-order", choices=("spatial", "model"), default="spatial")
	parser.add_argument("--checkpoint-every", type=int, default=100)
	parser.add_argument("--teacher-control", action="store_true", help="Unrotated teacher followed by VMC within the remaining adaptive wall-time budget")
	parser.add_argument("--ablation", action="store_true", help="Run the vmc/teacher/rotation/full outer-loop arms at equal VMC updates instead of the timed baselines")
	parser.add_argument("--max-baseline-updates", type=int, default=100000)
	args = parser.parse_args()
	if len(set(args.seeds)) != len(args.seeds):
		parser.error("--seeds must be distinct for independent paired comparisons")
	root = Path(args.output_dir)
	if root.exists() and any(root.iterdir()):
		parser.error("choose an empty output directory")
	if args.n_outer < 1 or args.max_baseline_updates < 1 or args.checkpoint_every < 1 or args.polish_warmup < 0:
		parser.error("outer iterations, checkpoint interval, and baseline update cap must be positive; warm polishing must be nonnegative")
	if args.system != "hubbard" and args.n_basin:
		parser.error("--n-basin applies to Hubbard orbital rotations only")
	if args.system != "hubbard" and args.entanglement_order != "spatial":
		parser.error("--entanglement-order applies to Hubbard orbital rotations only")
	if args.ablation and args.teacher_control:
		parser.error("--ablation already contains a teacher-only arm; omit --teacher-control")
	if args.system == "hubbard" and args.rotation_family is not None:
		parser.error("--rotation-family applies to spin rotations only")
	jax.config.update("jax_enable_x64", not args.float32)
	physical = HubbardConfig(Lx=args.Lx, Ly=args.Ly, U=args.U) if args.system == "hubbard" else LatticeConfig(hamiltonian=args.hamiltonian, Lx=args.Lx, Ly=args.Ly, pbc=False, j2=args.j2, hx=args.hx, hz=args.hz)
	model_cfg = ModelConfig(width=args.width, depth=args.depth, heads=args.heads, mlp_ratio=args.mlp_ratio, phase_head_depth=args.phase_head_depth, phase_init_std=args.phase_init_std, phase_scale=args.phase_scale, conserve_total_sz=args.system == "spin" and args.hamiltonian != "ising")
	step = AdaptiveStep(
		train=TrainConfig(n_nqs=args.n_nqs, n_samples=args.n_samples, learning_rate=args.learning_rate, sr_shift=args.sr_shift),
		rotation=RotationConfig(n_rot=args.n_rot, nqs_energy_every=args.nqs_energy_every, n_basin=args.n_basin, n_subspace=args.n_subspace, sampling=args.sampling, n_seed_samples=args.n_seed_samples, learning_rate=args.rotation_learning_rate, participation_weight=args.participation_weight, entanglement_weight=args.entanglement_weight, entanglement_order=args.entanglement_order),
		teacher=TeacherConfig(n_teacher=args.n_teacher, learning_rate=args.teacher_learning_rate, phase_weight=args.phase_weight, optimizer=args.teacher_optimizer, grad_clip=0. if args.teacher_optimizer == "lbfgs" else 1.), n_recovery=args.n_recovery,
	)
	args.n_samples = step.train.n_samples
	args.n_seed_samples = step.rotation.n_seed_samples
	baseline_rate = step.train.learning_rate if args.baseline_learning_rate is None else args.baseline_learning_rate
	baseline_train = dc.replace(step.train, learning_rate=baseline_rate)
	warm = dc.replace(step, train=dc.replace(step.train, n_nqs=args.warmup, learning_rate=args.warm_learning_rate if args.warm_learning_rate is not None else args.learning_rate), rotation=dc.replace(step.rotation, n_rot=0, n_basin=0), teacher=dc.replace(step.teacher, n_teacher=0), n_recovery=0)
	family = tuple(args.rotation_family) if args.rotation_family is not None else (("XY",) if args.system == "spin" and args.hamiltonian == "ising" else ("givens",))
	config = AdaptiveConfig(circuit=CircuitConfig(n_layers=args.n_layers, family=family), steps=(warm,), display="silent")
	run = run_fermionic_loop if args.system == "hubbard" else run_adaptive_loop
	rows = []
	write_json(root / "protocol.json", vars(args))
	write_json(root / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()], "versions": {package: version(package) for package in ("jax", "netket", "numpy", "scipy", "optax", "flax")}})
	for seed in args.seeds:
		warm_dir = root / f"seed-{seed}" / "warm"
		warmed = run(physical, model_cfg, config, seed=seed, output_dir=warm_dir)
		_, params, _, key, _ = load_outer_checkpoint(warm_dir)
		model = warmed["model"]
		if args.system == "hubbard":
			operator, n_terms, exact = warmed["operator"], warmed["operator_terms"], warmed["exact"]
		else:
			terms = base_hamiltonian(physical)
			operator, n_terms = pauli_terms_to_netket(terms, model.hilbert), len(terms)
			exact = None
			if not model_cfg.conserve_total_sz and physical.n_sites <= _MAX_ED:
				states = np.asarray(model.hilbert.all_states())
				exact = (states, operator.to_sparse(), warmed["exact_energy"])
			elif model_cfg.conserve_total_sz and math.comb(physical.n_sites, physical.n_sites // 2) <= 64:
				states = np.asarray(model.hilbert.all_states())
				states = states[np.sum(states == 1, axis=1) == physical.n_sites // 2]
				exact = (states, _projected_matrix(operator, states), warmed["exact_energy"])
		write_json(warm_dir / "nqs_history.json", warmed["nqs_history"])
		polish = None
		if args.polish_warmup:
			params, polish = polish_warm_state(model, params, exact, args.polish_warmup)
			save_weights(warm_dir / "checkpoint" / "weights.h5", params)
			write_json(warm_dir / "summary.json", {"final_energy": polish["energy"], "pre_polish_energy": warmed["final_energy"], "exact_energy": warmed["exact_energy"], "polish": polish, "outer_history": warmed["outer_history"]})
			warmed["final_energy"] = polish["energy"]
			print(f"seed={seed} warm E={polish['energy']:+.8f} gradient={polish['gradient_max']:.2g} curvature={polish['minimum_curvature']} stationarity_checks={polish['stationarity_checks_passed']}", flush=True)
		# Keep initialization, architecture, optimizer settings and update schedules fixed.
		branch = dc.replace(config, steps=(warm,) + (step,) * args.n_outer, restart_from=str(warm_dir))
		if args.ablation:
			for arm, rotate, teach in (("vmc", False, False), ("teacher", False, True), ("rotation", True, False), ("full", True, True)):
				stage = dc.replace(step, rotation=dc.replace(step.rotation, n_rot=step.rotation.n_rot if rotate else 0, n_basin=step.rotation.n_basin if rotate else 0), teacher=dc.replace(step.teacher, n_teacher=step.teacher.n_teacher if teach else 0))
				result = run(physical, model_cfg, dc.replace(branch, steps=(warm,) + (stage,) * args.n_outer), seed=seed, output_dir=root / f"seed-{seed}" / arm)
				history, reference = result["outer_history"], result["exact_energy"]
				row = {
					"seed": seed, "arm": arm, "warm_energy": warmed["final_energy"],
					"energy": result["final_energy"], "error": result["final_energy"] - reference if reference is not None else None,
					"variance": history[-1]["variance"], "accepted": sum(item["accepted"] for item in history),
					"seconds": sum(item["seconds"] for item in history), "operator_terms": history[-1]["operator_terms"],
				}
				rows.append(row)
				write_json(root / "ablation.json", rows)
				print(f"seed={seed} {arm:8s} E={row['energy']:+.8f} error={row['error']} accepted={row['accepted']}/{args.n_outer} seconds={row['seconds']:.2f}", flush=True)
			continue
		started = time.perf_counter()
		adaptive = run(physical, model_cfg, branch, seed=seed, output_dir=root / f"seed-{seed}" / "adaptive")
		jax.block_until_ready(adaptive["params"])
		adaptive_seconds = time.perf_counter() - started
		for stage_name in ("nqs", "rotation", "rotation_nqs", "teacher", "recovery"):
			write_json(root / f"seed-{seed}" / "adaptive" / f"{stage_name}_history.json", adaptive.get(f"{stage_name}_history", []))
		attempted_updates = sum(stage.train.n_nqs + stage.n_recovery for stage in branch.steps[1:])
		teacher_control = None
		if args.teacher_control:
			# Reserve recovery for the timed tail: a fixed recovery schedule can
			# exceed the adaptive budget when VMC is slower in the original basis.
			teacher_step = dc.replace(step, rotation=dc.replace(step.rotation, n_rot=0, n_basin=0), n_recovery=0)
			teacher_branch = dc.replace(branch, steps=(warm,) + (teacher_step,) * args.n_outer)
			started = time.perf_counter()
			teacher_control = run(physical, model_cfg, teacher_branch, seed=seed, output_dir=root / f"seed-{seed}" / "teacher")
			jax.block_until_ready(teacher_control["params"])
			teacher_control["seconds"] = time.perf_counter() - started

			# Spend the remaining adaptive budget on unrotated VMC, retaining the
			# teacher endpoint if continued optimization worsens it.
			evaluation_key = jax.random.fold_in(jax.random.PRNGKey(seed), 987653)
			schedule_metrics = _evaluate(model, teacher_control["params"], operator, evaluation_key, max(16 * args.n_samples, 1024), exact)
			teacher_control["schedule_metrics"] = schedule_metrics
			remainder = adaptive_seconds - teacher_control["seconds"]
			if remainder > 0.:
				teacher_control["params"], tail = vmc_control(model, teacher_control["params"], operator, exact, dc.replace(step.train, n_nqs=args.max_baseline_updates), jax.random.fold_in(jnp.asarray(key), 37), n_terms, evaluation_key, args.checkpoint_every, schedule_metrics["energy"], remainder)
				teacher_control["seconds"] += tail["seconds"]
				teacher_control["extra_vmc_updates"] = tail["updates"]
				write_json(root / f"seed-{seed}" / "teacher_tail.json", tail)
			save_weights(root / f"seed-{seed}" / "teacher_time" / "weights.h5", teacher_control["params"])
		# The adaptive and teacher endpoints do not depend on the baseline arm.
		evaluation_samples = max(16 * args.n_samples, 1024)
		evaluation_key = jax.random.fold_in(jax.random.PRNGKey(seed), 987654)
		if args.system == "hubbard":
			post_operator, post_exact = adaptive["operator"], adaptive["exact"]
		else:
			post_operator = rotated_netket_hamiltonian(physical, adaptive["gates"], adaptive["theta_star"], model.hilbert, prune=True)
			post_exact = (exact[0], post_operator.to_sparse() if not model_cfg.conserve_total_sz else _projected_matrix(post_operator, exact[0]), exact[2]) if exact else None
		adapted = _evaluate(model, adaptive["params"], post_operator, jax.random.fold_in(evaluation_key, 1), evaluation_samples, post_exact)
		if teacher_control is not None:
			teacher_operator, teacher_exact = (teacher_control["operator"], teacher_control["exact"]) if args.system == "hubbard" else (operator, exact)
			teacher_metrics = _evaluate(model, teacher_control["params"], teacher_operator, jax.random.fold_in(evaluation_key, 3), evaluation_samples, teacher_exact)
		for arm, cap, budget in (("vmc_time", args.max_baseline_updates, adaptive_seconds), ("vmc_updates", attempted_updates, None)):
			# No Ritz solves, teacher, basis proposals or rollback in ordinary VMC.
			baseline_cfg = dc.replace(baseline_train, n_nqs=cap)
			baseline_params, control = vmc_control(model, params, operator, exact, baseline_cfg, key, n_terms, evaluation_key, args.checkpoint_every, warmed["final_energy"], budget)
			baseline, final_baseline = control["metrics"], control["final"]
			seconds, history = control["seconds"], control["history"]
			save_weights(root / f"seed-{seed}" / arm / "weights.h5", baseline_params)
			difference = adapted["energy"] - baseline["energy"]
			error = math.hypot(adapted["standard_error"], baseline["standard_error"])
			row = {
				"seed": seed, "baseline": arm, "warm_energy": warmed["final_energy"],
				"warm_polish": polish,
				"adaptive": adapted, "vmc": baseline, "vmc_final": final_baseline, "best_vmc_step": control["best_step"], "difference": difference, "difference_standard_error": error,
				"resolved_gain": difference < -max(2 * error, 1e-8), "adaptive_seconds": adaptive_seconds, "vmc_seconds": seconds,
				"cost_ratio": seconds / adaptive_seconds, "budget_reached": budget is None or seconds >= budget,
				"vmc_updates": len(history) if cap else 0, "adaptive_attempted_vmc_updates": attempted_updates,
				"vmc_learning_rate": baseline_cfg.learning_rate,
				"adaptive_actual_vmc_updates": (len(adaptive["nqs_history"]) if args.n_nqs else 0) + len(adaptive["recovery_history"]),
				"accepted": sum(row["accepted"] for row in adaptive["outer_history"]),
				"adaptive_outer_history": adaptive["outer_history"],
				"adaptive_rotation_history": adaptive["rotation_history"],
			}
			if teacher_control is not None:
				row["teacher_control"] = teacher_metrics
				row["teacher_control_seconds"] = teacher_control["seconds"]
				row["teacher_schedule"] = teacher_control["schedule_metrics"]
				row["teacher_extra_vmc_updates"] = teacher_control.get("extra_vmc_updates", 0)
				row["teacher_cost_ratio"] = teacher_control["seconds"] / adaptive_seconds
				row["teacher_budget_reached"] = teacher_control["seconds"] >= adaptive_seconds
			rows.append(row)
			write_json(root / "comparison.json", rows)
			write_json(root / f"seed-{seed}" / f"{arm}.json", {"metrics": row, "history": history})
			print(f"seed={seed} {arm} deltaE={difference:+.6g} +/-{error:.2g} cost_ratio={row['cost_ratio']:.3f} VMC_updates={row['vmc_updates']} gain={row['resolved_gain']}", flush=True)
	if not args.ablation:
		write_json(root / "summary.json", comparison_summary(rows))


if __name__ == "__main__":
	main()
