r"""Autoregressive Hubbard NQS with sampled-Ritz spatial-orbital rotations.

OpenFermion interleaves spin modes by site; NetKet stores two spin blocks.
All operator conversions therefore use ``convert_spin_blocks=True``.
"""

from __future__ import annotations

import dataclasses as dc
import math
import time
from pathlib import Path
from typing  import Any

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from scipy.optimize import minimize, basinhopping

from .checkpoint    import load_outer_checkpoint, save_outer_checkpoint, validate_restart_config, write_json
from .config        import AdaptiveConfig, BasisParameterization, ModelConfig, SubspaceSampling
from .logging_utils import display_context
from .model         import ARTransformer
from .netket_vmc    import train_vmc
from .nqs           import energy_moments, frequency_ranked_configs, gumbel_top_k_configs, normalized_amplitudes, subspace_probability_mass
from .orbitals      import orbital_matrix, orbital_pairs, orbital_ritz_objective, orbital_cut_maps, prepare_orbital_projection, projected_orbital_matrix
from .teacher       import fit_ritz_teacher


# -------------------------


@dc.dataclass(frozen=True)
class HubbardConfig:
    """Square-lattice Hubbard couplings and fixed up/down particle numbers."""

    Lx     : int = 2
    Ly     : int = 2
    pbc    : bool = False
    t      : float = 1.0
    U      : float = 4.0
    n_up   : int | None = None
    n_down : int | None = None

    def __post_init__(self) -> None:
        if any(type(length) is not int or length < 1 for length in (self.Lx, self.Ly)):
            raise ValueError("Hubbard lattice extents must be positive integers")
        if not all(np.isfinite(value) for value in (self.t, self.U)) or (self.t == 0 and self.U == 0):
            raise ValueError("Hubbard couplings must be finite and not both zero")
        if (self.n_up is None or self.n_down is None) and self.n_sites % 2:
            raise ValueError("Odd-site half filling needs explicit up and down particle counts")
        for name in ("n_up", "n_down"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, self.n_sites // 2)
        if any(type(count) is not int or not 0 <= count <= self.n_sites for count in (self.n_up, self.n_down)):
            raise ValueError("Hubbard particle counts must lie within the spatial-orbital count")

    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly

    @property
    def sector_size(self) -> int:
        return math.comb(self.n_sites, self.n_up) * math.comb(self.n_sites, self.n_down)


# -------------------------
# Hamiltonian and selected-space construction
# -------------------------


def _hubbard_terms(of: Any, graph: Any, cfg: HubbardConfig) -> Any:
    """Use NetKet graph edges to define the same hopping in OpenFermion order."""
    terms = of.FermionOperator()
    # OpenFermion mode 2*i+spin holds spin ``spin`` on site ``i``; both hopping directions
    # are written explicitly so the Hermitian conjugate is not left implicit.
    for i, j in graph.edges():
        for spin in (0, 1):
            p, q = 2 * i + spin, 2 * j + spin
            terms += of.FermionOperator(((p, 1), (q, 0)), -cfg.t)
            terms += of.FermionOperator(((q, 1), (p, 0)), -cfg.t)
    # On-site repulsion n_up n_down, ordered so the operators act right to left.
    for site in range(cfg.n_sites):
        up, down = 2 * site, 2 * site + 1
        terms += of.FermionOperator(((up, 1), (up, 0), (down, 1), (down, 0)), cfg.U)
    return terms


def _projected_matrix(operator: Any, basis: np.ndarray) -> np.ndarray:
    """Build only the selected determinant matrix from fermionic connectivity."""
    basis = np.asarray(basis, dtype=np.int8)
    # Rows of ``connected`` are the determinants reached from each basis state; align them
    # back to basis indices with a byte-code lookup table.
    connected, elements = operator.get_conn_padded(jnp.asarray(basis))
    lookup              = {row.tobytes(): index for index, row in enumerate(basis)}
    matrix              = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for index, (rows, values) in enumerate(zip(np.asarray(connected, dtype=np.int8), np.asarray(elements))):
        for row, value in zip(rows, values):
            if value != 0:
                other = lookup.get(row.tobytes())
                if other is not None:
                    matrix[index, other] += value
    # Symmetrize after a Hermiticity guard to strip round-off asymmetry.
    if not np.all(np.isfinite(matrix)) or np.max(np.abs(matrix - matrix.conj().T)) > 1e-8:
        raise ArithmeticError("fermionic projected Hamiltonian is non-finite or non-Hermitian")
    return (matrix + matrix.conj().T) / 2


def _evaluate(model: ARTransformer, params: Any, operator: Any, key: jax.Array, n_samples: int, exact: tuple[np.ndarray, np.ndarray, float] | None) -> dict[str, float]:
    """Exact sector energy when small; otherwise direct-sampled NetKet energy."""
    if exact is not None:
        states, matrix, reference = exact
        # Evaluate the NQS on every enumerated determinant at once: the energy is then an
        # exact expectation over the small sector, not a sampled estimate.
        log_psi = np.asarray(model.apply(params, jnp.asarray(states)))
        # Every enumerated state lies in the particle sector, so no amplitude may vanish.
        if not np.all(np.isfinite(log_psi)):
            raise FloatingPointError("fermionic NQS has non-finite sector amplitudes")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            energy, variance = energy_moments(matrix, normalized_amplitudes(log_psi))
        if not (np.isfinite(energy) and np.isfinite(variance)):
            raise FloatingPointError("fermionic NQS has a non-finite local energy")
        return {"energy": energy, "variance": variance, "standard_error": 0., "energy_error": energy - reference}
    # Large sectors: independent autoregressive samples, seeded from the JAX key.
    seed                    = int(jax.random.randint(key, (), 0, 2_147_483_647))
    state                   = nk.vqs.MCState(nk.sampler.ARDirectSampler(model.hilbert), model, variables=params, sampler_seed=seed, n_samples=n_samples, n_discard_per_chain=0, chunk_size=min(n_samples, 128))
    stats                   = state.expect(operator)
    energy, variance, error = float(np.real(stats.mean)), float(stats.variance), float(stats.error_of_mean)
    if not all(np.isfinite(value) for value in (energy, variance, error)):
        raise FloatingPointError("fermionic sampled energy or uncertainty is non-finite")
    return {"energy": energy, "variance": variance, "standard_error": error, "energy_error": float("nan")}


def _select_basis(model: ARTransformer, params: Any, key: jax.Array, operator: Any, cap: int, retain: float, previous: np.ndarray, previous_vector: np.ndarray | None, sampling: SubspaceSampling, seed_samples: int, *, statistics: dict | None = None) -> np.ndarray:
    """Retain Ritz weight, add NQS seeds, then connected determinants."""
    selected: list[np.ndarray] = []
    seen: set[bytes]           = set()
    k_retain                   = min(len(previous), int(retain * len(previous)))
    if statistics is not None:
        statistics.update(method=sampling.value, basis_cap=cap, born_draw_budget=seed_samples if sampling is SubspaceSampling.FREQUENCY else 0, born_draws=0, unique_born_states=0, gumbel_unique_draws=0)
    if k_retain:
        # Retain the previous determinants with the largest Ritz weight (or NQS probability).
        weights = np.abs(previous_vector)**2 if previous_vector is not None else np.exp(2 * np.real(np.asarray(model.apply(params, jnp.asarray(previous)))))
        for index in np.argsort(-weights)[:k_retain]:
            row = previous[index]
            selected.append(row)
            seen.add(row.tobytes())
    # Fresh NQS seeds: frequency ranking for large sectors, Gumbel top-k otherwise.
    seeds = frequency_ranked_configs(model, params, key, n_samples=seed_samples, k=cap, statistics=statistics) if sampling is SubspaceSampling.FREQUENCY else gumbel_top_k_configs(model, params, key, cap)
    if statistics is not None and sampling is SubspaceSampling.GUMBEL:
        statistics.update(gumbel_unique_draws=int(len(seeds)), ranked_seed_states=int(len(seeds)))
    seed_target = max(len(selected), (cap + 1) // 2)
    for row in seeds:
        if row.tobytes() not in seen:
            selected.append(row)
            seen.add(row.tobytes())
        if len(selected) >= seed_target:
            break
    if len(selected) < cap:
        # Grow the retained set by the determinants most strongly coupled to the current Ritz
        # state: coupling_j = sum_i conj(H_ji) v_i, i.e. the H |v> frontier outside the support.
        basis = np.asarray(selected, dtype=np.int8)
        vector = np.linalg.eigh(_projected_matrix(operator, basis))[1][:, 0]
        connected, elements = operator.get_conn_padded(jnp.asarray(basis))
        couplings: dict[bytes, complex] = {}
        candidates: dict[bytes, np.ndarray] = {}
        for index, (rows, values) in enumerate(zip(np.asarray(connected, dtype=np.int8), np.asarray(elements))):
            for row, value in zip(rows, values):
                name = row.tobytes()
                if value != 0 and name not in seen:
                    candidates[name] = row
                    # Importance weight <x|H|v_0>; the Hubbard hopping integrals are real
                    # symmetric, so this reduces to value * vector[index].
                    couplings[name] = couplings.get(name, 0.) + np.conj(value) * vector[index]
        for name in sorted(couplings, key=lambda item: abs(couplings[item])**2, reverse=True):
            if abs(couplings[name]) > 0:
                selected.append(candidates[name])
                seen.add(name)
            if len(selected) == cap:
                break
    if len(selected) < cap:
        # Last resort: top up with additional NQS samples until the cap is met.
        fill = seeds if sampling is SubspaceSampling.GUMBEL else gumbel_top_k_configs(model, params, key, cap)
        if statistics is not None:
            statistics["gumbel_unique_draws"] = int(len(fill))
        for row in fill:
            if row.tobytes() not in seen:
                selected.append(row)
                seen.add(row.tobytes())
            if len(selected) == cap:
                break
    if len(selected) != cap:
        raise RuntimeError("could not fill the fermionic Ritz basis")
    basis = np.asarray(selected, dtype=np.int8)
    if statistics is not None:
        statistics.update(retained_states=k_retain, basis_size=int(len(basis)), nqs_mass_before=subspace_probability_mass(model, params, basis))
    return basis


def _rotated_interaction(interaction: Any, pairs: tuple, theta: np.ndarray) -> Any:
    """Apply the same real spatial rotation to both spin sectors."""
    import copy

    rotated = copy.deepcopy(interaction)
    # The rotation acts on spatial orbitals, so its matrix is repeated on each spin block.
    spatial = orbital_matrix(interaction.one_body_tensor.shape[0] // 2, pairs, theta)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        rotated.rotate_basis(np.kron(spatial, np.eye(2)))
    if not np.all(np.isfinite(rotated.one_body_tensor)) or not np.all(np.isfinite(rotated.two_body_tensor)):
        raise FloatingPointError("orbital rotation produced non-finite fermionic tensors")
    return rotated


# -------------------------


def run_fermionic_loop(hubbard: HubbardConfig, model_cfg: ModelConfig, adaptive_cfg: AdaptiveConfig, *, seed: int = 0, output_dir: str | Path | None = None) -> dict[str, Any]:
    r"""Train, rotate, teach, recover, and accept on a spinful Hubbard lattice.

    Exact sector energies are used only for a small reference system. Larger
    sectors use independent NetKet autoregressive samples for acceptance.
    """
    try:
        import openfermion as of
    except ImportError as exc:
        raise ImportError("Install the optional fermion extra: pip install -e '.[fermion]'") from exc
    if model_cfg.conserve_total_sz or model_cfg.use_marshall_sign or model_cfg.n_up is not None:
        raise ValueError("HubbardConfig sets fermion counts; spin-model masks and Marshall signs do not apply")
    circuit = adaptive_cfg.circuit
    if circuit.family != ("givens",) or circuit.parameterization is not BasisParameterization.GATE or circuit.include_nnn or circuit.use_marshall_clifford:
        raise ValueError("The fermionic loop uses nearest-neighbor spatial Givens layers with gate parameters")
    if adaptive_cfg.final.ritz_size or adaptive_cfg.final.solver != "none":
        raise ValueError("The fermionic loop does not use the spin final solvers")
    if any(step.rotation.n_subspace > hubbard.sector_size for step in adaptive_cfg.steps):
        raise ValueError(f"Ritz subspace exceeds the {hubbard.sector_size}-determinant particle sector")
    for step in adaptive_cfg.steps:
        rotation = step.rotation
        if any((rotation.nonstoquasticity_weight, rotation.nested_weight)):
            raise ValueError("Fermionic rotations support energy, participation, and cut entanglement objectives")
    root = Path(output_dir) if output_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    if adaptive_cfg.restart_from:
        validate_restart_config(adaptive_cfg.restart_from, {"hubbard": hubbard, "model": model_cfg, "circuit": circuit, "seed": seed})
    if root is not None:
        write_json(root / "config.json", {"hubbard": hubbard, "model": model_cfg, "adaptive": adaptive_cfg, "seed": seed})

    with display_context(adaptive_cfg.display):
        n_orbitals = hubbard.n_sites
        n_modes    = 2 * n_orbitals
        graph      = nk.graph.Grid(extent=[hubbard.Lx] if hubbard.Ly == 1 else [hubbard.Lx, hubbard.Ly], pbc=hubbard.pbc)
        # ``sector`` fixes the up/down occupations for exact enumeration; ``hilbert`` is the
        # full spin-orbital space the autoregressive model samples over.
        sector      = nk.hilbert.SpinOrbitalFermions(n_orbitals, s=.5, n_fermions_per_spin=(hubbard.n_up, hubbard.n_down))
        hilbert     = nk.hilbert.SpinOrbitalFermions(n_orbitals, s=.5)
        model       = ARTransformer(hilbert=hilbert, n_sites=n_modes, width=model_cfg.width, depth=model_cfg.depth, heads=model_cfg.heads, mlp_ratio=model_cfg.mlp_ratio, phase_head_depth=model_cfg.phase_head_depth, phase_head_width=model_cfg.phase_head_width, phase_scale=model_cfg.phase_scale, phase_init_std=model_cfg.phase_init_std, particle_counts=(hubbard.n_up, hubbard.n_down), local_occupations=True)
        interaction = of.get_interaction_operator(_hubbard_terms(of, graph, hubbard))
        pairs       = orbital_pairs(n_orbitals, circuit.n_layers)
        # Energy is normalized by a characteristic scale so the objective is dimensionless.
        angle_scale = n_modes * max(abs(hubbard.t), abs(hubbard.U))

        def operator(theta: np.ndarray) -> tuple[Any, int]:
            # Rotate the integrals, hand the resulting fermion operator to NetKet, and report
            # the number of terms so the caller can reweight the loss by operator size.
            fermions = of.get_fermion_operator(_rotated_interaction(interaction, pairs, theta))
            return nk.operator.FermionOperator2nd.from_openfermion(hilbert, fermions, convert_spin_blocks=True, dtype=complex), len(fermions.terms)

        # Dense enumeration is an oracle, never a requirement for a larger run.
        exact_states = np.asarray(sector.all_states(), dtype=np.int8) if hubbard.sector_size <= 64 else None

        def exact_oracle(fermion_operator: Any) -> tuple[np.ndarray, np.ndarray, float] | None:
            return (exact_states, _projected_matrix(fermion_operator, exact_states), exact_energy) if exact_states is not None else None

        theta = np.zeros(len(pairs), dtype=np.float64)
        current_operator, current_terms = operator(theta)
        exact_hamiltonian = _projected_matrix(current_operator, exact_states) if exact_states is not None else None
        exact_energy = float(np.linalg.eigvalsh(exact_hamiltonian)[0]) if exact_hamiltonian is not None else None
        if exact_hamiltonian is not None:
            # Guard against a silent spin-block ordering mismatch between the two libraries.
            native = np.asarray(nk.operator.FermiHubbardJax(sector, graph=graph, t=hubbard.t, U=hubbard.U).to_dense())
            if np.max(np.abs(native - exact_hamiltonian)) > 1e-8:
                raise ArithmeticError("NetKet Hubbard and OpenFermion hopping conventions disagree")
        key         = jax.random.PRNGKey(seed)
        params: Any = None
        basis       = np.zeros((0, n_modes), dtype=np.int8)
        start_outer = 0
        if adaptive_cfg.restart_from:
            start_outer, params, theta, saved_key, basis = load_outer_checkpoint(adaptive_cfg.restart_from)
            key = jnp.asarray(saved_key)
            if theta.shape != (len(pairs),) or basis.shape[1] != n_modes:
                raise ValueError("Fermionic restart angles or determinant basis have incompatible shape")
            if start_outer >= adaptive_cfg.n_outer:
                raise ValueError("Fermionic restart already completed the requested outer steps")
            current_operator, current_terms = operator(theta)
        previous_vector = None
        if len(basis):
            previous_vector = np.linalg.eigh(_projected_matrix(current_operator, basis))[1][:, 0]
        # The accepted operator and its exact oracle are carried between outer steps.
        current_exact = exact_oracle(current_operator)
        all_nqs, all_rotation, all_teacher, all_recovery, history = [], [], [], [], []
        # One outer step follows the fixed workflow: train, evaluate, rotate, final Ritz,
        # teacher, recover, accept, summarize.
        for outer in range(start_outer, adaptive_cfg.n_outer):
            started       = time.perf_counter()
            stage         = adaptive_cfg.steps[outer]
            timing        = {}
            phase_started = time.perf_counter()
            # --- Train ---
            params, rows, key = train_vmc(params, model, current_operator, stage.train, key, n_terms=current_terms, desc="Fermion NQS")
            jax.block_until_ready(params)
            timing["nqs_seconds"] = time.perf_counter() - phase_started
            all_nqs.extend(dict(row, outer=outer) for row in rows)

            # --- Evaluate ---
            key, evaluation_key = jax.random.split(key)
            evaluation_samples  = max(4 * stage.train.n_samples, 256)
            phase_started       = time.perf_counter()
            before              = _evaluate(model, params, current_operator, evaluation_key, evaluation_samples, current_exact)
            timing["evaluation_seconds"] = time.perf_counter() - phase_started
            params_before, theta_before = params, theta.copy()
            key, sample_key             = jax.random.split(key)
            phase_started               = time.perf_counter()
            sampling_statistics         = {}
            basis                       = _select_basis(model, params, sample_key, current_operator, stage.rotation.n_subspace, stage.rotation.f_retain, basis, previous_vector, stage.rotation.sampling, stage.rotation.n_seed_samples, statistics=sampling_statistics)

            timing["sample_seconds"] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            # --- Rotate on a fixed support; cuts group both spins ---
            cut_maps              = orbital_cut_maps(basis, stage.rotation.entanglement_order) if stage.rotation.entanglement_weight else None
            projection            = prepare_orbital_projection(basis) if stage.rotation.n_rot else None
            objective_evaluations = 0

            def objective(angles: np.ndarray) -> float:
                # Rebuild the rotated integrals and re-project them onto the cached support;
                # the projection plan is angle-independent, only the values change.
                nonlocal objective_evaluations
                objective_evaluations += 1
                rotated = _rotated_interaction(interaction, pairs, angles)
                return orbital_ritz_objective(projected_orbital_matrix(rotated, projection), angle_scale, stage.rotation.participation_weight, stage.rotation.entanglement_weight, cut_maps=cut_maps)[0]

            basis_matrix = _projected_matrix(current_operator, basis)
            objective_before, diagnostics_before = orbital_ritz_objective(basis_matrix, angle_scale, stage.rotation.participation_weight, stage.rotation.entanglement_weight, cut_maps=cut_maps)
            ritz_before = diagnostics_before["ritz_energy"]
            if stage.rotation.n_rot and len(pairs):
                # A truncated symmetric Ritz space may have a stationary zero-angle basis.
                key, rotation_key = jax.random.split(key)
                start             = theta + np.asarray(jax.random.normal(rotation_key, theta.shape)) * .2
                minimizer         = {"method": "L-BFGS-B", "options": {"maxiter": stage.rotation.n_rot, "gtol": 1e-7}}
                if stage.rotation.n_basin:
                    # Global displacements explore different compact orbital representations.
                    rotation_seed = int(jax.random.randint(jax.random.fold_in(rotation_key, 1), (), 0, 2_147_483_647))
                    optimized     = basinhopping(objective, start, niter=stage.rotation.n_basin, minimizer_kwargs=minimizer, stepsize=np.pi / 2, T=.1, rng=np.random.default_rng(rotation_seed))
                else:
                    optimized = minimize(objective, start, **minimizer)
                if not np.all(np.isfinite(optimized.x)) or not np.isfinite(optimized.fun):
                    raise FloatingPointError("fermionic orbital optimization produced non-finite angles or loss")
                if optimized.fun < objective_before:
                    theta = optimized.x

            theta = np.where(np.abs(theta) < circuit.theta_cutoff, 0., theta)
            timing["rotation_seconds"] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()

            # --- Final Ritz ---
            # Unchanged angles leave the operator, the Ritz problem, and the NQS energy unchanged.
            angles_changed            = not np.array_equal(theta, theta_before)
            post_operator, post_terms = operator(theta) if angles_changed else (current_operator, current_terms)
            post_basis_matrix         = _projected_matrix(post_operator, basis) if angles_changed else basis_matrix
            # Ground state of the rotated selected space; ``diagonal`` feeds the teacher's
            # diagonal-energy bias, so it is taken from the same projected matrix.
            values, vectors                    = np.linalg.eigh(post_basis_matrix)
            ritz_energy, vector, diagonal      = float(values[0]), vectors[:, 0], np.real(np.diag(post_basis_matrix))
            objective_after, diagnostics_after = orbital_ritz_objective(post_basis_matrix, angle_scale, stage.rotation.participation_weight, stage.rotation.entanglement_weight, cut_maps=cut_maps)
            all_rotation.append(dict(diagnostics_after, outer=outer, entanglement_order=stage.rotation.entanglement_order, ritz_energy_before=ritz_before, objective_before=objective_before, objective=objective_after, objective_evaluations=objective_evaluations, angle_norm=float(np.linalg.norm(theta))))
            timing["ritz_seconds"] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            post_exact    = exact_oracle(post_operator) if angles_changed else current_exact
            # The split keeps later random streams independent of whether the angles moved.
            key, evaluation_key = jax.random.split(key)
            after_rotation      = _evaluate(model, params, post_operator, evaluation_key, evaluation_samples, post_exact) if angles_changed else before
            timing["evaluation_seconds"] += time.perf_counter() - phase_started

            # --- Teacher ---
            timing["teacher_seconds"] = 0.
            teacher_rows  = []
            after_teacher = after_rotation
            if stage.teacher.n_teacher and len(basis) > 1:
                phase_started                         = time.perf_counter()
                params, teacher_rows, teacher_metrics = fit_ritz_teacher(model, params, basis, vector, diagonal, ritz_energy, 1. / angle_scale, stage.teacher)
                jax.block_until_ready(params)
                timing["teacher_seconds"] = time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                all_teacher.extend(dict(row, outer=outer) for row in teacher_rows)
                key, evaluation_key = jax.random.split(key)
                after_teacher       = _evaluate(model, params, post_operator, evaluation_key, evaluation_samples, post_exact)
                timing["evaluation_seconds"] += time.perf_counter() - phase_started
            else:
                teacher_metrics = {"enabled": False}

            # --- Recover at fixed orbital angles ---
            timing["recovery_seconds"] = 0.
            after_recovery = after_teacher
            if stage.n_recovery:
                phase_started              = time.perf_counter()
                params, recovery_rows, key = train_vmc(params, model, post_operator, dc.replace(stage.train, n_nqs=stage.n_recovery), key, n_terms=post_terms, desc="Fermion recovery")
                jax.block_until_ready(params)
                timing["recovery_seconds"] = time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                all_recovery.extend(dict(row, outer=outer) for row in recovery_rows)
                key, evaluation_key = jax.random.split(key)
                after_recovery      = _evaluate(model, params, post_operator, evaluation_key, evaluation_samples, post_exact)
                timing["evaluation_seconds"] += time.perf_counter() - phase_started

            # --- Accept using the NQS, not the selected-state objective ---
            # Tolerance: exact reference for a small sector, otherwise the combined sampling
            # uncertainty of the before/after estimates, relaxed by ``acceptance_sigma``.
            comparison_error     = math.hypot(before["standard_error"], after_recovery["standard_error"])
            threshold            = 1e-9 * angle_scale if exact_states is not None else -adaptive_cfg.acceptance_sigma * comparison_error
            accepted             = after_recovery["energy"] - before["energy"] <= threshold
            proposal_ritz_energy = ritz_energy
            if accepted:
                current_operator, current_terms, current_exact = post_operator, post_terms, post_exact
            else:
                # Reject: restore the accepted parameters and angles, and recompute the Ritz
                # state of the accepted operator so the next step retains that support.
                params, theta       = params_before, theta_before
                values, vectors     = np.linalg.eigh(basis_matrix)
                ritz_energy, vector = float(values[0]), vectors[:, 0]
            previous_vector  = vector
            accepted_metrics = after_recovery if accepted else before

            # --- Summarize ---
            row = {
                "subspace_mass": subspace_probability_mass(model, params, basis),
                "seconds": time.perf_counter() - started, "timing": timing,
                "outer": outer, "accepted": bool(accepted),
                "energy_before_rotation": before["energy"], "energy_after_rotation": after_rotation["energy"],
                "energy_after_teacher": after_teacher["energy"], "energy_after_recovery": after_recovery["energy"],
                "energy": accepted_metrics["energy"], "variance": accepted_metrics["variance"],
                "standard_error": accepted_metrics["standard_error"], "comparison_error": comparison_error,
                "ritz_energy_before_rotation": ritz_before, "proposal_ritz_energy": proposal_ritz_energy,
                "ritz_energy": ritz_energy, "subspace_size": len(basis),
                "sampling": sampling_statistics,
                "operator_terms": post_terms if accepted else current_terms, "teacher": teacher_metrics,
            }
            history.append(row)
            save_outer_checkpoint(root, outer, params, theta, key, basis)
        key, evaluation_key = jax.random.split(key)
        final_metrics       = _evaluate(model, params, current_operator, evaluation_key, evaluation_samples, current_exact)

        if root is not None:
            write_json(root / "summary.json", {"exact_energy": exact_energy, "final_energy": final_metrics["energy"], "final_standard_error": final_metrics["standard_error"], "outer_history": history})
        return {"model": model, "params": params, "theta_star": theta, "exact_energy": exact_energy, "final_energy": final_metrics["energy"], "final_standard_error": final_metrics["standard_error"], "outer_history": history, "nqs_history": all_nqs, "rotation_history": all_rotation, "teacher_history": all_teacher, "recovery_history": all_recovery, "basis": basis, "operator": current_operator, "operator_terms": current_terms, "key": key, "exact": current_exact}


# -------------------------
#! EOF
# -------------------------
