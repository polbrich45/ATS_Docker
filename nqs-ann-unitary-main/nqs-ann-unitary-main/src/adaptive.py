r"""Explicit alternating NQS -> sampled-Ritz rotation -> teacher workflow."""

from __future__ import annotations

import dataclasses as dc
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint	import load_outer_checkpoint, save_outer_checkpoint, validate_restart_config, write_json
from .config		import AdaptiveConfig, HamiltonianType, LatticeConfig, LatticeType, ModelConfig, RotationConfig, SubspaceSampling, TrainConfig, sector_n_up
from .final_solvers import solve_final
from .logging_utils import display_context, log, section, timing_summary, title
from .model			import bits_to_inputs, support_size
from .netket_vmc	import _MAX_REFERENCE_ED, exact_ground_state, pauli_terms_to_netket, train_vmc
from .nqs			import (
                        _MAX_ED,
                        _diagnostic_apply,
                        apply_rotation_to_vector,
                        ed_diagnostics,
                        energy_moments,
                        frequency_ranked_configs,
                        gumbel_top_k_configs,
                        make_transformer,
                        marshall_transform_vector,
                        normalized_amplitudes,
                        subspace_probability_mass,
                        train_nqs,
                    )
from .operators		import (
                        apply_theta_cutoff,
                        base_hamiltonian,
                        commutes_with_total_sz,
                        impurity_metadata,
                        marshall_clifford_hamiltonian,
                        pauli_rotation_circuit,
                        rotated_hamiltonian_terms,
                    )

# The rotation is the second phase of the alternating workflow
from .rotation_ritz import (
                        RotationProblem,
                        RotationDiagnostics,
                        build_rotated_subspace_hamiltonian,
                        compute_participation_entropy,
                        hamiltonian_j_rms,
                        optimize_rotation_angles,
                        prepare_rotation_problem,
                        ritz_ground_state,
                        sampled_nonstoquasticity,
                        sampled_renyi2_entropies,
                        schmidt_labels,
                    )
from .ritz_subspace import build_sampled_subspace, ritz_pt2_correction

# The teacher distillation is implemented in a separate module to avoid circular imports.
from .teacher		import fit_ritz_teacher

# -----------------------------------------------


def _rotation_rows(history: list[RotationDiagnostics], *, outer: int) -> list[dict[str, Any]]:
    """Flatten the one pre-update diagnostic produced by each angle update."""
    rows: list[dict[str, Any]] = []
    for index, diagnostics in enumerate(history):
        row = {name: float(value) for name, value in diagnostics._asdict().items()}
        row.update({"outer": outer, "step": index})
        rows.append(row)
    return rows


def _steady_step_seconds(rows: list[dict[str, Any]]) -> float:
    """Median completed-update interval, discarding the first three warmup updates."""
    intervals = np.diff([row["seconds"] for row in rows])[2:]
    valid = intervals[np.isfinite(intervals) & (intervals > 0.0)]
    return float(np.median(valid)) if valid.size else float("nan")


def _log_nqs_metrics(label: str, metrics: dict, n_sites: int, elapsed: float, *, previous: float | None = None) -> None:
    """Physical-state diagnostics, distinct from pre-update MC training observations."""
    method = "exact enumeration" if "variational_energy" in metrics else "independent MC"
    change = f" | dE={metrics['energy'] - previous:+.6g}" if previous is not None else ""
    log(f"{label}: E={metrics['energy']:+.12f} | E/N={metrics['energy']/n_sites:+.12f}{change}", level=1)
    log(f"{method} | variance={metrics['variance']:.6g} | SE={metrics['standard_error']:.3g} | {elapsed:.3f} s", level=2)
    if np.isfinite(metrics["energy_error"]):
        log(f"E-E0={metrics['energy_error']:+.6g}" + (f" | GS fidelity={metrics['overlap']**2:.8f}" if "overlap" in metrics else ""), level=2)


def _probe_row(outer: int, step: int, metrics: dict) -> dict[str, Any]:
    """One physical NQS energy observation inside a rotation or teacher trace."""
    return {"outer": outer, "step": step, "energy": metrics["energy"], "variance": metrics["variance"], "standard_error": metrics["standard_error"]}


def _exact_rotation_probe(cfg, model, params, base_terms, gates):
    """Cache the frozen NQS once, then measure <U phi|H|U phi> for small spins."""
    states	= np.asarray(model.hilbert.all_states())
    phi		= normalized_amplitudes(_diagnostic_apply(model)(params, jnp.asarray(states)))
    matrix	= pauli_terms_to_netket(base_terms, model.hilbert).to_sparse()

    def evaluate(theta):
        energy, variance = energy_moments(matrix, apply_rotation_to_vector(phi, states, gates, theta, cfg.n_sites))
        return {"energy": energy, "variance": variance, "standard_error": 0.0}

    return evaluate


def _exact_teacher_probe(model, operator):
    """Cache H_theta and basis states while the teacher changes only NQS parameters."""
    states	= jnp.asarray(model.hilbert.all_states())
    matrix	= operator.to_sparse()
    apply	= _diagnostic_apply(model)

    def evaluate(params):
        energy, variance = energy_moments(matrix, normalized_amplitudes(apply(params, states)))
        return {"energy": energy, "variance": variance, "standard_error": 0.0}

    return evaluate


def _evaluate_nqs(
    cfg				: LatticeConfig,
    model			: Any,
    params			: Any,
    train_cfg		: TrainConfig,
    base_terms		: list,
    gates			: list,
    theta			: jax.Array,
    terms			: list,
    raw_terms		: int,
    operator		: Any,
    key				: jax.Array,
    exact_energy	: float,
    exact_vector	: np.ndarray | None,
) -> dict[str, float]:
    """Evaluate the NQS with one already materialized post-rotation operator."""
    exact				= exact_vector is not None and cfg.n_sites <= _MAX_ED
    diagnostics			= ed_diagnostics(cfg, base_terms, gates, theta, model, params, exact_energy, exact_vector, operator=operator) if exact else {}

    # Exact enumeration has no sampling error; otherwise use one independent MC estimate.
    if diagnostics:
        energy			= float(diagnostics["variational_energy"])
        variance		= float(diagnostics["variational_variance"])
        standard_error	= 0.0
    else:
        eval_cfg		= dc.replace(train_cfg, n_nqs=0)
        _, history, _	= train_vmc(params, model, operator, eval_cfg, key, n_terms=len(terms), desc="Evaluate NQS")
        energy			= float(history[-1]["energy"])
        variance		= float(history[-1]["variance"])
        standard_error	= float(history[-1]["standard_error"])
        # NetKet reports a non-finite error over the mean for very small batches; fall
        # back to the i.i.d. standard error sqrt(var/N), which reproduces NetKet exactly
        # whenever its own estimate is finite.
        if not np.isfinite(standard_error) and np.isfinite(variance) and eval_cfg.n_samples > 0:
            standard_error = float(np.sqrt(max(variance, 0.0) / eval_cfg.n_samples))
    if not (np.isfinite(energy) and np.isfinite(variance)) or not np.isfinite(standard_error) or standard_error < 0.0:
        raise FloatingPointError("NQS energy evaluation produced invalid statistics")
    result = {
        "energy"		: energy,
        "variance"		: variance,
        "standard_error": standard_error,
        "energy_error"	: float(energy - exact_energy) if np.isfinite(exact_energy) else float("nan"),
        "raw_terms"		: float(raw_terms),
        "terms"			: float(len(terms)),
    }
    result.update({name: float(value) for name, value in diagnostics.items() if np.isscalar(value)})
    return result


# -----------------------------------------------


def _selected_subspace(
    model				: Any,
    params				: Any,
    key					: jax.Array,
    terms				: list,
    n_sites				: int,
    cfg					: RotationConfig,
    retained			: np.ndarray,
    prev_ritz_vector	: np.ndarray | None = None,
    *, statistics	: dict | None = None,
) -> np.ndarray:
    r"""
    Append ANN/PT2 configurations to the retained Ritz basis up to its cap.

    - ``cfg.sampling='gumbel'`` supplies unique without-replacement ANN seeds - see :gumbel_top_k_configs.
    - ``'frequency'`` draws the hard-capped cfg.n_seed_samples Born samples and rank their unique rows. 

    Construction is delegated to :func:`ritz_subspace.build_sampled_subspace`, which retains
    the top f_retain fraction of old rows and streams new rows into one capped buffer.
    """
    n_subspace = int(cfg.n_subspace)
    if statistics is not None:
        statistics.update(method=cfg.sampling.value, basis_cap=n_subspace, born_draw_budget=cfg.n_seed_samples if cfg.sampling is SubspaceSampling.FREQUENCY else 0, born_draws=0, unique_born_states=0, gumbel_unique_draws=0)
    f_retain = float(cfg.f_retain)
    k_retain = min(retained.shape[0], int(np.floor(f_retain * retained.shape[0]))) if retained.shape[0] > 0 and f_retain > 0.0 else 0

    if retained.shape[0] > 0 and k_retain > 0:
        if prev_ritz_vector is not None and len(prev_ritz_vector) == retained.shape[0]:
            probs 			= np.abs(np.asarray(prev_ritz_vector)) ** 2
            order 			= np.argsort(-probs)
            retained_top 	= retained[order[:k_retain]]
        else:
            log_psi 		= model.apply(params, bits_to_inputs(model, jnp.asarray(retained)))
            probs 			= np.exp(2.0 * np.real(np.asarray(log_psi)))
            order 			= np.argsort(-probs)
            retained_top 	= retained[order[:k_retain]]
        log(f"Subspace retention: kept top {retained_top.shape[0]}/{retained.shape[0]} states from previous Ritz basis (f_retain={f_retain:.2f})", level=2)
    else:
        retained_top = np.zeros((0, n_sites), dtype=np.int8)

    # Split the random key for sampling and filling
    sample_key, fill_key 	= jax.random.split(key)
    gumbel_ranked			: np.ndarray | None = None

    # The sampling strategy is determined by the configuration. 
    if cfg.sampling is SubspaceSampling.FREQUENCY:
        ranked 			= frequency_ranked_configs(model, params, sample_key, n_samples=cfg.n_seed_samples, k=n_subspace, statistics=statistics)
        log(f"Seed sampling ({cfg.sampling.value}): {ranked.shape[0]} ranked seed candidates from {cfg.n_seed_samples} Born draws", level=2)
    else:
        ranked 			= gumbel_top_k_configs(model, params, sample_key, n_subspace)
        gumbel_ranked 	= ranked
        if statistics is not None:
            statistics.update(gumbel_unique_draws=int(len(ranked)), ranked_seed_states=int(len(ranked)))
        log(f"Seed sampling ({cfg.sampling.value}): {ranked.shape[0]} unique sector-valid configurations via Gumbel beam search", level=2)

    def fill() -> np.ndarray:
        nonlocal gumbel_ranked
        if gumbel_ranked is None:
            gumbel_ranked = gumbel_top_k_configs(model, params, fill_key, n_subspace)
            if statistics is not None:
                statistics["gumbel_unique_draws"] = int(len(gumbel_ranked))
        return gumbel_ranked

    subspace = build_sampled_subspace(terms, retained_top, ranked, fill, n_sites, n_subspace, n_up=model.n_up)
    if statistics is not None:
        statistics.update(retained_states=int(len(retained_top)), basis_size=int(len(subspace)), nqs_mass_before=subspace_probability_mass(model, params, subspace))
    log(f"subspace assembled: {subspace.shape[0]} states total ({retained_top.shape[0]} retained + {subspace.shape[0] - retained_top.shape[0]} seeds & connected)", level=3)
    return subspace


def _physical_angle_rms(gates: list, theta: jax.Array) -> float:
    """RMS angle over physical gates, including repetitions from shared parameters."""
    if not gates:
        return 0.0
    indices = np.fromiter((int(gate.parameter) for gate in gates), dtype=np.int64)
    angles	= np.asarray(theta, dtype=np.float64)[indices]
    return float(np.sqrt(np.mean(angles * angles)))


def _final_ritz_on_subspace(
    problem		: RotationProblem,
    theta		: jax.Array,
    terms		: list,
) -> tuple[Any, dict[str, float], list, np.ndarray]:
    """Solve the prepared projected matrix and reuse it for PT2 and teaching."""
    matrix	= build_rotated_subspace_hamiltonian(problem, theta)
    state	= ritz_ground_state(matrix, problem.j_rms)
    pt2		= ritz_pt2_correction(terms, problem.samples, np.asarray(state.vector), float(state.energy), problem.n_sites)
    metrics = {
        "ritz_energy"			: float(state.energy),
        "ritz_gap"				: float(state.gap),
        # Report these once even when their optimization weights are zero.
        "ritz_participation_entropy": float(compute_participation_entropy(state.vector)),
        # Sparse Schmidt matrices: no O(N M^2) pair arrays for a report-only metric.
        "ritz_renyi2_sum"		: float(np.sum(sampled_renyi2_entropies(np.asarray(state.vector), schmidt_labels(problem.samples, range(1, problem.n_sites))))),
        "ritz_nonstoquasticity"	: float(sampled_nonstoquasticity(matrix)),
        "pt2_correction"		: float(pt2["pt2"]),
        "ritz_energy_pt2"		: float(state.energy) + float(pt2["pt2"]),
        "pt2_frontier_size"		: float(pt2["frontier_size"]),
        "pt2_coupled_weight"	: float(pt2["coupled_weight"]),
        "pt2_skipped_crossings" : float(pt2["skipped_crossings"]),
        "pt2_frontier_truncated": float(pt2["frontier_truncated"]),
    }
    return state, metrics, terms, np.real(np.diag(np.asarray(matrix, dtype=np.complex128)))


# -----------------------------------------------


def _write_run_config(
    root			: Path | None,
    cfg				: LatticeConfig,
    model_cfg		: ModelConfig,
    adaptive_cfg	: AdaptiveConfig,
    seed			: int,
) -> None:
    if root is None:
        return
    write_json(root / "config.json", {
        "lattice"			: cfg,
        "model"				: model_cfg,
        "adaptive"			: adaptive_cfg,
        "seed"				: seed,
        "jax_enable_x64"	: bool(jax.config.jax_enable_x64),
        "jax_backend"		: jax.default_backend(),
    })
    if cfg.impurities:
        write_json(root / "impurities.json", impurity_metadata(cfg))


# -----------------------------------------------


def run_adaptive_loop(
    lattice_cfg		: LatticeConfig,
    model_cfg		: ModelConfig,
    adaptive_cfg	: AdaptiveConfig,
    *,
    seed			: int = 0,
    output_dir		: str | Path | None = None,
) -> dict[str, Any]:
    r"""Run the explicit alternating adaptive workflow.

    The circuit topology is fixed by ``adaptive_cfg.circuit``. Each entry in
    ``adaptive_cfg.steps`` is one concrete scalar schedule point:

    ``train -> evaluate -> rotate -> final Ritz -> teacher -> recover -> accept``.

    The sampled-Ritz pair at the final angle of each step is computed once and
    reused for PT2 and direct teacher distillation. A short VMC recovery follows
    teaching. Proposals that worsen the NQS energy beyond sampling uncertainty
    restore both parameter blocks; non-finite optimizer updates fail explicitly.

    Parameters
    ----------
    lattice_cfg : LatticeConfig
        The lattice and Hamiltonian configuration.
    model_cfg : ModelConfig
        The NQS model configuration.
    adaptive_cfg : AdaptiveConfig
        The adaptive workflow configuration.
    seed : int, optional
        The random seed for JAX PRNG, by default 0.
    output_dir : str | Path | None, optional
        The directory to write checkpoints and summaries, by default None (no output).

    Example
    -------
    A successful step may lower a fixed-budget Ritz error and then recover the
    post-rotation NQS energy through teaching. The exact many-body ground energy
    itself cannot change with basis; ED-sized runs verify that invariant.

    >>> exact_before, exact_after = -2.677886, -2.677886
    >>> abs(exact_after - exact_before) < 1.0e-12
    True

    References
    ----------
    - Continuous-space NQS basis transformations (conceptual motivation),
      https://arxiv.org/abs/2604.15888
    - Sample-expand-diagonalize-distill structure,
      https://arxiv.org/abs/2606.26760
    """
    workflow_started = time.perf_counter()
    cfg		= lattice_cfg
    if any(step.rotation.n_basin for step in adaptive_cfg.steps):
        raise ValueError("n_basin is supported by the fermionic orbital search only")
    if any(step.rotation.entanglement_order != "spatial" for step in adaptive_cfg.steps):
        raise ValueError("entanglement_order is configurable in the fermionic loop only")
    root	= Path(output_dir) if output_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)

    with display_context(adaptive_cfg.display):
        title("Adaptive-basis NQS")
        device = jax.config.jax_default_device or jax.devices()[0]
        log(f"Model: {cfg.hamiltonian.value} | {cfg.Lx}x{cfg.Ly} | N={cfg.n_sites} | PBC={cfg.pbc} | seed={seed}", level=1)
        log(f"Device: {device} | JAX x64={jax.config.jax_enable_x64}", level=1)
        log(f"ARTransformer: width={model_cfg.width}, depth={model_cfg.depth}, heads={model_cfg.heads}, mlp_ratio={model_cfg.mlp_ratio}, conserve_total_sz={model_cfg.conserve_total_sz}", level=1)
        if adaptive_cfg.restart_from:
            validate_restart_config(adaptive_cfg.restart_from, {"lattice": cfg, "model": model_cfg, "circuit": adaptive_cfg.circuit})

        # Compute the base Hamiltonian terms and prepare the circuit gates. If Marshall preprocessing is requested, apply it to the base Hamiltonian.
        base_terms	= base_hamiltonian(cfg)
        circuit		= adaptive_cfg.circuit
        if circuit.use_marshall_clifford:
            if cfg.lattice is not LatticeType.SQUARE or cfg.hamiltonian is not HamiltonianType.J1J2:
                raise ValueError("Marshall preprocessing is implemented only for square J1-J2 lattices")
            if model_cfg.use_marshall_sign:
                raise ValueError("Choose either CircuitConfig.use_marshall_clifford or ModelConfig.use_marshall_sign, not both")

            # Apply the fixed Marshall Clifford before building the trainable circuit.
            base_terms = marshall_clifford_hamiltonian(cfg, base_terms)

        # Circuit topology is fixed once and only its parameters are warm-started.
        gates = pauli_rotation_circuit(cfg, depth=circuit.n_layers, family=circuit.family, parameterization=circuit.parameterization.value, include_nnn=circuit.include_nnn) if circuit.n_layers > 0 else []

        # If the model is configured to conserve total Sz, check that the Hamiltonian and circuit gates are compatible with this symmetry.
        # Raise an error if there is a mismatch.
        if model_cfg.conserve_total_sz:
            sector_n_up(model_cfg, cfg.n_sites)

            if not commutes_with_total_sz(base_terms, cfg.n_sites):
                raise ValueError("ModelConfig.conserve_total_sz=True requires [H, total Sz] = 0")

            if any(not bool(getattr(gate, "conserves_sz", False)) for gate in gates):
                raise ValueError("The selected circuit does not conserve total Sz; disable ModelConfig.conserve_total_sz")

        # ED-sized systems use exact diagnostics by default. The sparse reference
        # (energy only) is cheap up to ``_MAX_REFERENCE_ED``; the explicit ground vector
        # is materialized only while the enumerated diagnostics (``_MAX_ED``) consume it.
        reference_started = time.perf_counter()
        if cfg.n_sites <= _MAX_REFERENCE_ED:
            # Call with the default flags for the vector path so the cache key matches
            # external ``exact_ground_state(cfg)`` callers and the oracle is solved once.
            exact_energy, exact_vector = exact_ground_state(cfg) if cfg.n_sites <= _MAX_ED else exact_ground_state(cfg, vectors=False)
        else:
            exact_energy, exact_vector = float("nan"), None

        # If Marshall preprocessing is used, transform the exact ground state vector accordingly.
        if circuit.use_marshall_clifford and exact_vector is not None:
            exact_vector = marshall_transform_vector(cfg, exact_vector)

        # Format the exact energy for logging. If it is finite, display it with 12 decimal places and E/N; otherwise, indicate that it was not computed.
        exact_energy_text = f"{exact_energy:+.12f} (E/N: {exact_energy / cfg.n_sites:+.12f})" if np.isfinite(exact_energy) else "not computed"
        log(f"Exact ground energy: {exact_energy_text} | {time.perf_counter() - reference_started:.3f} s", level=1)

        # Initialize the parameter vector and other necessary variables.
        n_params				= 0 if not gates else max(gate.parameter for gate in gates) + 1
        n_up					= sector_n_up(model_cfg, cfg.n_sites)
        teacher_energy_scale 	= 1.0 / (cfg.n_sites * hamiltonian_j_rms(base_terms, cfg.n_sites))

        # Other params
        theta					= jnp.zeros(n_params)
        key						= jax.random.PRNGKey(seed)
        params: Any 			= None
        model: Any				= make_transformer(cfg, model_cfg)
        sector_size				= support_size(model)
        log(f"Adaptive loop: {adaptive_cfg.n_outer} steps, {len(gates)} fixed circuit gates, {n_params} trainable parameters", level=1)

        # Start the outer loop at 0 or from a restart checkpoint
        start_outer 			= 0
        rotation_problem: RotationProblem | None = None

        # A restart restores both parameter blocks and the random stream.
        last_subspace 			= np.zeros((0, cfg.n_sites), dtype=np.int8)
        if adaptive_cfg.restart_from:
            log(f"Restarting from {adaptive_cfg.restart_from}", level=1)
            start_outer, params, theta_np, key_np, last_subspace = load_outer_checkpoint(adaptive_cfg.restart_from)
            if start_outer >= adaptive_cfg.n_outer:
                raise ValueError("Restart checkpoint already completed every configured AdaptiveStep")
            theta	= jnp.asarray(theta_np, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
            key		= jnp.asarray(key_np)
            if theta.shape != (n_params,):
                raise ValueError(f"Restart theta shape {theta.shape} is incompatible with the configured circuit ({n_params} parameters)")
        _write_run_config(root, cfg, model_cfg, adaptive_cfg, seed)

        # Store the history of each phase for later
        all_nqs			: list[dict[str, Any]] = []
        all_rotation	: list[dict[str, Any]] = []
        all_teacher		: list[dict[str, Any]] = []
        all_recovery	: list[dict[str, Any]] = []
        outer_history	: list[dict[str, Any]] = []
        last_terms		= base_terms
        last_ritz_vector: np.ndarray | None = None
        if last_subspace.shape[0]:
            rotation_problem = prepare_rotation_problem(base_terms, gates, last_subspace, cfg.n_sites)
            last_ritz_vector = np.asarray(ritz_ground_state(build_rotated_subspace_hamiltonian(rotation_problem, theta), rotation_problem.j_rms).vector)
        for outer in range(start_outer, adaptive_cfg.n_outer):
            stage			= adaptive_cfg.steps[outer]

            if stage.rotation.n_subspace > sector_size:
                raise ValueError(f"n_subspace={stage.rotation.n_subspace} exceeds the physical sector size {sector_size}")

            started 		= time.perf_counter()
            timing			= {
                                "nqs_seconds"			: 0.0,
                                "sample_seconds"		: 0.0,
                                "angle_seconds"			: 0.0,
                                "ritz_seconds"			: 0.0,
                                "teacher_seconds"		: 0.0,
                                "recovery_seconds"		: 0.0,
                                "evaluation_seconds"	: 0.0,
                                "rotation_probe_seconds": 0.0,
                                "checkpoint_seconds"	: 0.0,
                            }
            title(f"Outer step {outer + 1}/{adaptive_cfg.n_outer}")

            nqs_started = time.perf_counter()
            with section(f"Train NQS ({stage.train.n_nqs} updates)", level=1):
                if stage.train.n_nqs == 0 and params is not None:
                    current_terms, current_raw_terms = rotated_hamiltonian_terms(base_terms, gates, theta, cfg.n_sites)
                    before_operator = pauli_terms_to_netket(current_terms, model.hilbert)
                    # Preserve the training key schedule without sampling an unchanged state.
                    key = jax.random.split(key, 3)[0]
                    nqs_rows = []
                    log("NQS updates disabled; measuring the existing state below", level=2)
                else:
                    trained				= train_nqs(cfg, model_cfg, stage.train, base_terms, gates, theta, params, key, model=model, fresh_init=params is None, seed=seed + outer, desc="NQS",)
                    model, params, key	= trained["model"], trained["params"], trained["key"]
                    current_terms, current_raw_terms = trained["terms"], trained["raw_terms"]
                    before_operator = trained["operator"]
                    nqs_rows			= [dict(row, outer=outer, step=index) for index, row in enumerate(trained["history"])]
            timing["nqs_seconds"] += time.perf_counter() - nqs_started

            def measure_nqs(label, trial_params, trial_theta, terms, raw_terms, operator, evaluation_key, previous=None):
                """Time and log one physical NQS energy at fixed angles."""
                evaluation_started	= time.perf_counter()
                metrics				= _evaluate_nqs(cfg, model, trial_params, stage.train, base_terms, gates, trial_theta, terms, raw_terms, operator, evaluation_key, exact_energy, exact_vector)
                elapsed				= time.perf_counter() - evaluation_started
                timing["evaluation_seconds"] += elapsed
                _log_nqs_metrics(label, metrics, cfg.n_sites, elapsed, previous=previous)
                return metrics

            # Select once, retain the previous basis, and optimize all angles against that one well-defined Ritz problem.
            params_before, theta_before	= params, theta
            key, before_key				= jax.random.split(key)
            before						= measure_nqs("Before rotation", params, theta, current_terms, current_raw_terms, before_operator, before_key)
            key, sample_key				= jax.random.split(key)
            sample_started				= time.perf_counter()
            sampling_statistics			= {}
            last_subspace				= _selected_subspace(model, params, sample_key, current_terms, cfg.n_sites, stage.rotation, last_subspace, last_ritz_vector, statistics=sampling_statistics)
            timing["sample_seconds"]   += time.perf_counter() - sample_started

            problem_started = time.perf_counter()
            if rotation_problem is None or not np.array_equal(rotation_problem.samples, last_subspace) or (stage.rotation.entanglement_weight != 0.0 and rotation_problem.entanglement.n_cuts == 0):
                rotation_problem = prepare_rotation_problem(base_terms, gates, last_subspace, cfg.n_sites, include_entanglement=stage.rotation.entanglement_weight != 0.0)
            timing["sample_seconds"] += time.perf_counter() - problem_started
            log(f"Select Ritz basis: {last_subspace.shape[0]}/{stage.rotation.n_subspace} states | NQS mass={sampling_statistics['nqs_mass_before']:.6f} | {timing['sample_seconds']:.3f} s", level=1)
            log(f"SQD sampling: method={sampling_statistics['method']} | Born draws={sampling_statistics['born_draws']}/{sampling_statistics['born_draw_budget']} | unique Born states={sampling_statistics['unique_born_states']} | Gumbel draws={sampling_statistics['gumbel_unique_draws']}", level=2)

            rotation_nqs_rows = [_probe_row(outer, 0, before)]
            probe = None
            if stage.rotation.nqs_energy_every and stage.rotation.n_rot > 0 and cfg.n_sites <= _MAX_ED:
                probe_started = time.perf_counter()
                probe = _exact_rotation_probe(cfg, model, params, base_terms, gates)
                timing["rotation_probe_seconds"] += time.perf_counter() - probe_started

            # Physical probes never advance the training stream: exact probes already leave it
            # untouched, and Monte Carlo probes draw from a stream folded off the current key,
            # so enabling ``nqs_energy_every`` cannot change the sampling trajectory.
            probe_key_base = jax.random.fold_in(key, 0x9E37_79B9)

            def measure_rotation(step_index, trial_theta):
                if step_index % stage.rotation.nqs_energy_every or step_index == stage.rotation.n_rot:
                    return
                probe_started = time.perf_counter()
                if probe is not None:
                    metrics = probe(trial_theta)
                else:
                    probe_key = jax.random.fold_in(probe_key_base, step_index)
                    trial_terms, raw_terms = rotated_hamiltonian_terms(base_terms, gates, trial_theta, cfg.n_sites)
                    metrics = _evaluate_nqs(cfg, model, params, stage.train, base_terms, gates, trial_theta, trial_terms, raw_terms, pauli_terms_to_netket(trial_terms, model.hilbert), probe_key, exact_energy, exact_vector)
                log(f"Angle update {step_index}/{stage.rotation.n_rot}, before teacher: NQS E={metrics['energy']:+.9f} | dE={metrics['energy'] - before['energy']:+.6g} | SE={metrics['standard_error']:.3g}", level=2)
                rotation_nqs_rows.append(_probe_row(outer, step_index, metrics))
                timing["rotation_probe_seconds"] += time.perf_counter() - probe_started

            log(f"Rotate: {stage.rotation.n_rot} updates | NQS probe stride={stage.rotation.nqs_energy_every}", level=1)
            log(f"Rotation weights: participation={stage.rotation.participation_weight:g}, entanglement={stage.rotation.entanglement_weight:g}, nested={stage.rotation.nested_weight:g}, nonstoquasticity={stage.rotation.nonstoquasticity_weight:g}", level=2)
            rotation_started 			= time.perf_counter()
            probe_seconds_before        = timing["rotation_probe_seconds"]
            theta, diagnostics 			= optimize_rotation_angles(rotation_problem, stage.rotation, theta0=theta, on_step=measure_rotation if stage.rotation.nqs_energy_every else None)
            # Rotation diagnostics are pre-update; reuse their initial Ritz solve.
            ritz_energy_before          = float(diagnostics[0].ritz_energy) if diagnostics else float("nan")
            elapsed 					= time.perf_counter() - rotation_started
            timing["angle_seconds"]	   += elapsed - (timing["rotation_probe_seconds"] - probe_seconds_before)
            rotation_rows 				= _rotation_rows(diagnostics, outer=outer)
            for row in rotation_rows:
                row["normalized_ritz_energy"]   = row["ritz_energy"] * rotation_problem.energy_scale
                row["auxiliary_loss"]           = row["loss"] - row["normalized_ritz_energy"]
            log(f"Angle optimization: {len(diagnostics) if stage.rotation.n_rot and gates else 0} updates | angle RMS={_physical_angle_rms(gates, theta):.6f} | optimizer={timing['angle_seconds']:.3f} s | NQS probes={timing['rotation_probe_seconds']:.3f} s", level=2)
            if rotation_rows:
                log(f"Rotation loss (pre-update): {rotation_rows[0]['loss']:.6g} -> {rotation_rows[-1]['loss']:.6g} | loss minus normalized Ritz={rotation_rows[-1]['auxiliary_loss']:.3g} (includes roundoff)", level=2)

            # The final Ritz solve uses exactly the basis that defined the rotation loss.
            theta						= apply_theta_cutoff(theta, circuit.theta_cutoff)
            # Unchanged angles leave H_theta and the NQS state unchanged.
            angles_changed				= not np.array_equal(np.asarray(theta), np.asarray(theta_before))
            ritz_started				= time.perf_counter()
            last_terms, last_raw_terms 	= rotated_hamiltonian_terms(base_terms, gates, theta, cfg.n_sites) if angles_changed else (current_terms, current_raw_terms)
            ritz_state, ritz_metrics, last_terms, ritz_diagonal = _final_ritz_on_subspace(rotation_problem, theta, last_terms)
            last_ritz_vector			= np.asarray(ritz_state.vector)
            timing["ritz_seconds"]	   += time.perf_counter() - ritz_started

            pt2_val = ritz_metrics["pt2_correction"]
            pt2_str = f"+ PT2={pt2_val:+.6f}" if np.isfinite(pt2_val) else "+ PT2=omitted"
            log(f"Final Ritz: E={ritz_metrics['ritz_energy']:+.12f} | dE={ritz_metrics['ritz_energy'] - ritz_energy_before:+.6g} | {pt2_str} | {timing['ritz_seconds']:.3f} s", level=1)
            log(f"Ritz gap={ritz_metrics['ritz_gap']:.6g} | participation entropy={ritz_metrics['ritz_participation_entropy']:.6g} | Renyi-2 sum={ritz_metrics['ritz_renyi2_sum']:.6g} | nonstoquasticity={ritz_metrics['ritz_nonstoquasticity']:.6g}", level=2)

            # The split keeps later random streams independent of whether the angles moved.
            rotated_key, key 			= jax.random.split(key)
            if angles_changed:
                post_operator			= pauli_terms_to_netket(last_terms, model.hilbert)
                after_rotation			= measure_nqs("After rotation, before teacher (NQS parameters unchanged)", params, theta, last_terms, last_raw_terms, post_operator, rotated_key, previous=before["energy"])
            else:
                post_operator, after_rotation = before_operator, before
                log("After rotation: angles unchanged; reusing the before-rotation measurement", level=1)
            rotation_nqs_rows.append(_probe_row(outer, stage.rotation.n_rot, after_rotation))

            teacher_rows				: list[dict[str, Any]] = []
            teacher_nqs_rows			= [_probe_row(outer, 0, after_rotation)]
            teacher_summary				: dict[str, Any] = {"enabled": False}
            after_teacher 				= after_rotation
            if stage.teacher.n_teacher > 0 and last_subspace.shape[0] > 1:
                teacher_probe_seconds = 0.0
                probe_started = time.perf_counter()
                teacher_probe = _exact_teacher_probe(model, post_operator) if cfg.n_sites <= _MAX_ED and 0 < stage.teacher.nqs_energy_every < stage.teacher.n_teacher else None
                timing["evaluation_seconds"] += time.perf_counter() - probe_started
                # Monte Carlo teacher probes draw from a folded stream, so they never advance ``key``.
                teacher_probe_key_base = jax.random.fold_in(key, 0x85EB_CA6B)

                def measure_teacher(step_index, trial_params):
                    nonlocal teacher_probe_seconds
                    if step_index % stage.teacher.nqs_energy_every or step_index == stage.teacher.n_teacher:
                        return
                    probe_started = time.perf_counter()
                    if teacher_probe is not None:
                        metrics = teacher_probe(trial_params)
                    else:
                        probe_key = jax.random.fold_in(teacher_probe_key_base, step_index)
                        metrics = _evaluate_nqs(cfg, model, trial_params, stage.train, base_terms, gates, theta, last_terms, last_raw_terms, post_operator, probe_key, exact_energy, exact_vector)
                    teacher_nqs_rows.append(_probe_row(outer, step_index, metrics))
                    teacher_probe_seconds += time.perf_counter() - probe_started
                    log(f"Teacher probe {step_index}/{stage.teacher.n_teacher}: NQS E={metrics['energy']:+.9f} | SE={metrics['standard_error']:.3g}", level=2)

                log(f"Teacher: {stage.teacher.n_teacher} updates | optimizer={stage.teacher.optimizer} | phase weight={stage.teacher.phase_weight:g}", level=1)
                teacher_started 					= time.perf_counter()
                params, local_rows, teacher_metrics = fit_ritz_teacher(model, params, last_subspace, ritz_state.vector, ritz_diagonal, float(ritz_state.energy), teacher_energy_scale, stage.teacher, on_step=measure_teacher if stage.teacher.nqs_energy_every else None)
                teacher_rows 						= [dict(row, outer=outer) for row in local_rows]
                timing["teacher_seconds"] 		   += time.perf_counter() - teacher_started - teacher_probe_seconds
                timing["evaluation_seconds"] += teacher_probe_seconds
                log(f"Teacher loss={teacher_metrics['teacher_loss']:.6g} | joint KL/site={teacher_metrics['teacher_kl_per_site']:.6g} | conditional KL/site={teacher_metrics['conditional_kl_per_site']:.6g} | mass loss/site={teacher_metrics['mass_loss_per_site']:.6g}", level=2)
                log(f"Subspace mass={teacher_metrics['subspace_mass']:.6f} | phase coherence={teacher_metrics['phase_coherence']:.6f} | phase loss={teacher_metrics['phase_loss']:.6g} | {timing['teacher_seconds']:.3f} s", level=2)
                log(f"NQS fidelity with Ritz target={teacher_metrics['teacher_fidelity']:.6f} (squared full-state overlap)", level=2)
                log(f"Teacher objective evaluations={teacher_metrics['function_evaluations']:.0f} (includes line search and final evaluation)", level=2)

                teacher_key, key 					= jax.random.split(key)
                after_teacher 						= measure_nqs("After teacher", params, theta, last_terms, last_raw_terms, post_operator, teacher_key, previous=after_rotation["energy"])
                teacher_summary 					= {"enabled": True, **teacher_metrics}
                teacher_nqs_rows.append(_probe_row(outer, len(local_rows), after_teacher))
            else:
                log("Teacher: skipped", level=1)

            # Keep the alternating loop: teach the Ritz state, then recover at fixed angles.
            params_before_recovery = params
            recovery_rows = []
            after_recovery = after_teacher
            if stage.n_recovery > 0:
                log(f"Recover VMC: {stage.n_recovery} updates", level=1)
                recovery_started = time.perf_counter()
                params, recovery_history, key = train_vmc(params, model, post_operator, dc.replace(stage.train, n_nqs=stage.n_recovery), key, n_terms=len(last_terms), desc="VMC recovery")
                recovery_rows = [dict(row, outer=outer, step=index) for index, row in enumerate(recovery_history)]
                timing["recovery_seconds"] += time.perf_counter() - recovery_started
                key, recovery_key = jax.random.split(key)
                after_recovery = measure_nqs("After recovery", params, theta, last_terms, last_raw_terms, post_operator, recovery_key, previous=after_teacher["energy"])
            else:
                log("Recover VMC: skipped", level=1)

            # Recovery must not discard a resolved lower-energy state already measured.
            proposal_metrics = after_recovery
            recovery_error = math.hypot(after_teacher["standard_error"], after_recovery["standard_error"])
            recovery_tolerance = adaptive_cfg.acceptance_sigma * recovery_error + 1.0e-10 / teacher_energy_scale
            recovery_restored_start = bool(stage.n_recovery > 0 and np.isfinite(recovery_error) and after_recovery["energy"] - after_teacher["energy"] > recovery_tolerance)
            if recovery_restored_start:
                params = params_before_recovery
                proposal_metrics = after_teacher
                log(f"Recovery raised energy by {after_recovery['energy'] - after_teacher['energy']:+.6g}; restored NQS from before recovery (threshold={recovery_tolerance:.3g})", level=1)

            # This is an uncertainty-tolerant safeguard, not proof of a stochastic gain.
            energy_change           = proposal_metrics["energy"] - before["energy"]
            comparison_error        = math.hypot(before["standard_error"], proposal_metrics["standard_error"])
            tolerance               = adaptive_cfg.acceptance_sigma * comparison_error + 1.0e-10 / teacher_energy_scale
            accepted                = bool(np.isfinite(energy_change) and np.isfinite(comparison_error) and energy_change <= tolerance)
            proposal_ritz_energy    = ritz_metrics["ritz_energy"]
            proposal_ritz_metrics   = dict(ritz_metrics)
            if not accepted:
                params, theta                   = params_before, theta_before
                last_terms, last_raw_terms      = current_terms, current_raw_terms
                ritz_started                    = time.perf_counter()
                ritz_state, ritz_metrics, _, _  = _final_ritz_on_subspace(rotation_problem, theta, last_terms)
                timing["ritz_seconds"]         += time.perf_counter() - ritz_started
            last_ritz_vector    = np.asarray(ritz_state.vector)
            final_metrics       = proposal_metrics if accepted else before
            before_step_seconds = _steady_step_seconds(nqs_rows)
            after_step_seconds  = _steady_step_seconds(recovery_rows)
            slowdown            = after_step_seconds / before_step_seconds if before_step_seconds > 0.0 else float("nan")
            log(f"Proposal {'accepted' if accepted else 'rejected; restored NQS and angles'}: NQS dE={energy_change:+.6g} | combined SE={comparison_error:.3g} | acceptance threshold={tolerance:.3g}", level=1)
            log(f"Operator cost: terms {len(current_terms)} -> {int(after_rotation['terms'])} | max connections {before_operator.max_conn_size} -> {post_operator.max_conn_size}", level=2)
            if np.isfinite(slowdown):
                log(f"VMC steady update: before={before_step_seconds:.4f} s | proposed basis={after_step_seconds:.4f} s | slowdown={slowdown:.2f}x (warmup excluded)", level=2)
            else:
                log("VMC slowdown: unmeasured (need at least 4 updates in both training blocks)", level=2)
            diagnostic_started 			= time.perf_counter()
            final_subspace_mass 		= subspace_probability_mass(model, params, last_subspace)
            timing["sample_seconds"]    += time.perf_counter() - diagnostic_started

            summary = {
                "outer"							: outer,
                "teacher"						: teacher_summary,
                "energy_before_rotation"		: float(before["energy"]),
                "standard_error_before_rotation": float(before["standard_error"]),
                "energy_after_rotation"			: float(after_rotation["energy"]),
                "standard_error_after_rotation": float(after_rotation["standard_error"]),
                "rotation_energy_change"		: float(after_rotation["energy"] - before["energy"]),
                "rotation_nqs_history"			: rotation_nqs_rows,
                "teacher_nqs_history"			: teacher_nqs_rows,
                "energy_after_teacher"			: float(after_teacher["energy"]),
                "variance_after_teacher"		: float(after_teacher["variance"]),
                "standard_error_after_teacher"	: float(after_teacher["standard_error"]),
                "energy_after_recovery"			: float(after_recovery["energy"]),
                "standard_error_after_recovery": float(after_recovery["standard_error"]),
                "recovery_restored_start"		: recovery_restored_start,
                "energy"						: float(final_metrics["energy"]),
                "variance"						: float(final_metrics["variance"]),
                "standard_error"				: float(final_metrics["standard_error"]),
                "accepted"						: accepted,
                "proposal_energy_change"		: float(energy_change),
                "ritz_energy_before_rotation"	: ritz_energy_before,
                "proposal_ritz_energy"			: proposal_ritz_energy,
                "proposal_ritz_metrics"			: proposal_ritz_metrics,
                "angle_rms"						: _physical_angle_rms(gates, theta),
                "subspace_size"					: int(last_subspace.shape[0]),
                "sampling"						: sampling_statistics,
                "operator_terms"				: int(final_metrics["terms"]),
                "operator_terms_before"			: len(current_terms),
                "proposal_operator_terms"		: int(after_rotation["terms"]),
                "max_conn_before"				: int(before_operator.max_conn_size),
                "proposal_max_conn"				: int(post_operator.max_conn_size),
                "n_samples"						: stage.train.n_samples,
                "vmc_step_seconds_before"		: before_step_seconds,
                "vmc_step_seconds_proposed"		: after_step_seconds,
                "vmc_slowdown"					: slowdown,
                "subspace_mass"					: float(final_subspace_mass),
                **ritz_metrics,
                "timing"						: timing,
            }

            # Extend the history and save the checkpoint for this outer step
            checkpoint_started = time.perf_counter()
            save_outer_checkpoint(root, outer, params, theta, key, last_subspace)
            timing["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
            summary["seconds"] = time.perf_counter() - started
            all_nqs.extend(nqs_rows)
            all_rotation.extend(rotation_rows)
            all_teacher.extend(teacher_rows)
            all_recovery.extend(dict(row, accepted=accepted) for row in recovery_rows)
            outer_history.append(summary)

            e_nqs = final_metrics["energy"]
            e_ritz = ritz_metrics["ritz_energy"]
            pt2_sum = ritz_metrics["ritz_energy_pt2"]
            pt2_out_str = f"+PT2={pt2_sum:+.6f} (E/N={pt2_sum/cfg.n_sites:+.6f})" if np.isfinite(pt2_sum) else "+PT2=omitted"
            log(
                f"Outer {outer + 1}: E_NQS={e_nqs:+.6f} (E/N={e_nqs/cfg.n_sites:+.6f}) "
                f"Ritz={e_ritz:+.6f} (E/N={e_ritz/cfg.n_sites:+.6f}) "
                f"{pt2_out_str}",
                level=0,
            )
            timing_summary({
                "Train NQS"		: timing["nqs_seconds"],
                "SQD basis"		: timing["sample_seconds"],
                "Rotate angles"		: timing["angle_seconds"],
                "Ritz + PT2"		: timing["ritz_seconds"],
                "Teacher"			: timing["teacher_seconds"],
                "Recover VMC"		: timing["recovery_seconds"],
                "Evaluate NQS"		: timing["evaluation_seconds"],
                "Rotation probes"	: timing["rotation_probe_seconds"],
                "Checkpoint"		: timing["checkpoint_seconds"],
            }, summary["seconds"], label=f"Outer {outer + 1} timing")

        final_ritz: dict[str, Any] = {}
        if adaptive_cfg.final.ritz_size > 0 and model is not None:
            final_ritz_started = time.perf_counter()
            if adaptive_cfg.final.ritz_size > sector_size:
                raise ValueError(f"final Ritz size {adaptive_cfg.final.ritz_size} exceeds the physical sector size {sector_size}")

            key, final_key 			= jax.random.split(key)
            final_selection 		= dc.replace(adaptive_cfg.steps[-1].rotation, n_subspace=adaptive_cfg.final.ritz_size)
            final_subspace 			= _selected_subspace(model, params, final_key, last_terms, cfg.n_sites, final_selection, last_subspace)
            final_problem 			= rotation_problem if rotation_problem is not None and np.array_equal(rotation_problem.samples, final_subspace) else prepare_rotation_problem(base_terms, gates, final_subspace, cfg.n_sites)
            state, metrics, _, _ 	= _final_ritz_on_subspace(final_problem, theta, last_terms)
            final_ritz 				= {**metrics, "size": int(final_subspace.shape[0]), "vector": np.asarray(state.vector)}
            f_pt2 = metrics["pt2_correction"]
            f_pt2_str = f"+ PT2={f_pt2:+.6f}" if np.isfinite(f_pt2) else "+ PT2=omitted"
            log(f"Expanded final Ritz: {final_ritz['size']} states | E={final_ritz['ritz_energy']:+.12f} (E/N: {final_ritz['ritz_energy']/cfg.n_sites:+.12f}) {f_pt2_str} | {time.perf_counter() - final_ritz_started:.3f} s", level=1)

        final_terms 	= last_terms
        final_reference_started = time.perf_counter()
        # Reuse the original full-space oracle; unitary rotations preserve its energy.
        if n_up is None and np.isfinite(exact_energy) and adaptive_cfg.final.solver in {"exact", "auto"}:
            final_reference = {
                "method": "exact", "energy": float(exact_energy), "variance": 0.0,
                "n_up": None, "history": [], "basis": "original", "reused": True,
            }
        else:
            final_reference = solve_final(final_terms, cfg.n_sites, adaptive_cfg.final, n_up=n_up)
        if final_reference:
            log(f"Final reference: {final_reference['method']} | E={final_reference['energy']:+.12f} | reused={final_reference.get('reused', False)} | {time.perf_counter() - final_reference_started:.3f} s", level=1)
        else:
            log("Final reference: omitted", level=1)
        result 			= {
            "lattice"			: cfg,
            "model_cfg"			: model_cfg,
            "adaptive_cfg"		: adaptive_cfg,
            "exact_energy"		: float(exact_energy),
            "theta_star"		: np.asarray(theta, dtype=np.float64),
            "gates"				: gates,
            "params"			: params,
            "model"				: model,
            "nqs_history"		: all_nqs,
            "rotation_history"	: all_rotation,
            "rotation_nqs_history": [probe for row in outer_history for probe in row["rotation_nqs_history"]],
            "teacher_history"	: all_teacher,
            "teacher_nqs_history": [probe for row in outer_history for probe in row["teacher_nqs_history"]],
            "recovery_history"	: all_recovery,
            "outer_history"		: outer_history,
            "final_energy"		: float(outer_history[-1]["energy"]),
            "final_ritz"		: final_ritz,
            "final_reference"	: final_reference,
        }
        if root is not None:
            write_json(root / "summary.json", {
                "final_energy"		: result["final_energy"],
                "exact_energy"		: result["exact_energy"],
                "outer_history"		: outer_history,
                "final_ritz"		: {key: value for key, value in final_ritz.items() if key != "vector"},
                "final_reference"	: final_reference,
            })
            log(f"Saved summary to {root / 'summary.json'}", level=1)

        title("Adaptive workflow complete")
        error_text = f" | E-E0={result['final_energy'] - exact_energy:+.6g}" if np.isfinite(exact_energy) else ""
        log(f"NQS E={result['final_energy']:+.12f}{error_text} | accepted proposals={sum(row['accepted'] for row in outer_history)}/{len(outer_history)}", level=1)
        log(f"Total wall time: {time.perf_counter() - workflow_started:.3f} s (includes setup, reference, all outer steps, final solve, and output)", level=1)

        return result


# -------------------------
#! EOF
# -------------------------
