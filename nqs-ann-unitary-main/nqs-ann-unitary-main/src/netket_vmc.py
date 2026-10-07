r"""NetKet-backed variational Monte Carlo for adaptive NQS optimization.
The Hamiltonian is supplied as a NetKet operator built below
from the same Pauli strings the rest of the code uses, 
so the NQS trains the exact operator that rotation and diagnostics analyze.

The returned ``history`` is a list of
``{"loss", "energy", "variance", "n_terms", "dE", "seconds"}`` rows.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import optax

from .config		import LatticeConfig, LearningRateSchedule, ModelConfig, SPIN, TrainConfig, TrainOptimizer, sector_n_up
from .logging_utils import log, progress_disabled
from .model			import ARTransformer
from .operators		import PauliTerm, RotationGate, base_hamiltonian, commutes_with_total_sz, rotated_hamiltonian_terms, rotated_pauli_terms

# Sparse reference solves remain small-system oracles; full NQS diagnostics stop earlier.
_MAX_REFERENCE_ED 	= 16

# ----------------------------------------------


def netket_hilbert(cfg: LatticeConfig, model_cfg: ModelConfig | None = None) -> nk.hilbert.Spin:
    r"""Spin-1/2 Hilbert space, optionally restricted to a total ``S^z`` sector.

    The sector follows ``model_cfg`` through :func:`src.config.sector_n_up`.
    Requesting a sector for a Hamiltonian that does not commute with total
    ``S^z`` is an error; this helper never silently changes the physics.
    """
    n_up = None if model_cfg is None else sector_n_up(model_cfg, cfg.n_sites)
    if n_up is None:
        return nk.hilbert.Spin(s=SPIN, N=cfg.n_sites)
    if not commutes_with_total_sz(base_hamiltonian(cfg), cfg.n_sites):
        raise ValueError("A fixed-Sz Hilbert space requires [H, total Sz] = 0")
    return nk.hilbert.Spin(s=SPIN, N=cfg.n_sites, total_sz=SPIN * (2 * n_up - cfg.n_sites))


def pauli_terms_to_netket(terms: list[PauliTerm], hilbert: nk.hilbert.AbstractHilbert) -> nk.operator.PauliStrings:
    """Convert encoded Pauli terms to one vectorized NetKet ``PauliStrings``.

    Each ``PauliTerm(x, z, c)`` becomes a length-``N`` string with ``x&z`` mapped
    to ``Y``. One bulk construction is much faster than accumulating
    ``LocalOperator`` terms and retains NetKet's JAX-native connectivity path.
    """
    n_sites = hilbert.size
    dtype   = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    if not terms:
        return nk.operator.PauliStrings(hilbert, ["I" * n_sites], [0.0], dtype=dtype)
    strings: list[str] 		= []
    weights: list[complex] 	= []
    for term in terms:
        chars = ["I"] * n_sites

        # The PauliTerm encoding uses bitmasks for X and Z operators; Y is X+Z.
        for site in range(n_sites):
            x 			= (term.x >> site) & 1
            z 			= (term.z >> site) & 1
            chars[site] = "Y" if x and z else "X" if x else "Z" if z else "I"
        strings.append("".join(chars))
        weights.append(complex(term.coefficient))
    return nk.operator.PauliStrings(hilbert, strings, weights, dtype=dtype)


@functools.lru_cache(maxsize=None)
def exact_ground_state(
    cfg			: LatticeConfig,
    model_cfg	: ModelConfig | None = None,
    k			: int = 1,
    vectors		: bool = True,
) -> tuple[float, np.ndarray | None]:
    r"""Return the cached Lanczos ground-state energy and, optionally, its vector.

    The solve is cached on ``(cfg, model_cfg, k, vectors)``. Supplying ``model_cfg``
    restricts diagonalization to its configured ``S^z`` sector; otherwise the
    full spin Hilbert space is used. Set ``vectors=False`` to skip materializing
    the state; the second return value is then ``None``.
    """
    # The CPU reference retains double precision even when VMC uses float32.
    with jax.enable_x64(True):
        hamiltonian 	= pauli_terms_to_netket(base_hamiltonian(cfg), netket_hilbert(cfg, model_cfg))
        if not vectors:
            energy 		= nk.exact.lanczos_ed(hamiltonian, k=k, compute_eigenvectors=False, scipy_args={"tol": 1e-12})
            return float(np.real(np.asarray(energy).reshape(-1)[0])), None
        energy, vector 	= nk.exact.lanczos_ed(hamiltonian, k=k, compute_eigenvectors=True, scipy_args={"tol": 1e-12})
    return float(np.real(energy[0])), np.asarray(vector[:, 0])


# ---------------------------------------------
# Model and Hamiltonian builders for NetKet VMC
# ---------------------------------------------


def rotated_netket_hamiltonian(
    cfg			: LatticeConfig,
    gates		: list[RotationGate],
    theta		: Any,
    hilbert		: nk.hilbert.AbstractHilbert | None = None,
    base_terms	: list[PauliTerm] | None = None,
    prune		: bool = False,
) -> nk.operator.PauliStrings:
    """Return ``U(theta)^dag H U(theta)`` as a NetKet spin Pauli operator."""
    hilbert = nk.hilbert.Spin(s=SPIN, N=cfg.n_sites) if hilbert is None else hilbert
    terms 	= base_hamiltonian(cfg) if base_terms is None else base_terms
    rotated = (
        rotated_hamiltonian_terms(terms, gates, theta, cfg.n_sites)[0]
        if prune
        else rotated_pauli_terms(terms, gates, theta, cfg.n_sites)
    )
    return pauli_terms_to_netket(rotated, hilbert)




# ----------------------------------------------


def _seed_from_key(key: jax.Array) -> int:
    """Deterministic 31-bit integer seed from a PRNG key (keeps runs reproducible)."""
    words	= np.asarray(jax.random.key_data(key) if hasattr(jax.random, "key_data") else key, dtype=np.uint32).reshape(-1)
    seed	= 0
    for word in words:
        seed = (seed * 1664525 + int(word) + 1013904223) & 0xFFFFFFFF
    return int(seed & 0x7FFFFFFF)


def _to_operator(operator: Any) -> Any:
    """Prefer the jax-native operator (fast, jit-friendly ``get_conn``) when available."""
    to_jax = getattr(operator, "to_jax_operator", None)
    if to_jax is not None:
        try:
            return to_jax()
        except Exception:	# pragma: no cover - fall back to the dense/numba operator
            return operator
    return operator


def _scalar_schedule(
    start		: float,
    final		: float,
    schedule	: LearningRateSchedule | str,
    steps		: int,
) -> float | optax.Schedule:
    """Build a scalar schedule accepted by Optax and NetKet SR."""
    schedule	= LearningRateSchedule(schedule)
    steps		= max(1, steps)
    if schedule is LearningRateSchedule.CONSTANT:
        return float(start)
    alpha = final / start if start > 0 else 0.0
    return optax.cosine_decay_schedule(start, steps, alpha=alpha)


def _learning_rate(cfg: TrainConfig) -> float | optax.Schedule:
    """Build an Optax learning-rate schedule from ``TrainConfig``."""
    return _scalar_schedule(cfg.learning_rate, cfg.learning_rate_final, cfg.learning_rate_schedule, cfg.n_nqs)


@functools.lru_cache(maxsize=32)
def _sr_diag_shift(cfg: TrainConfig) -> float | optax.Schedule:
    """Build the NetKet SR diagonal-shift schedule."""
    return _scalar_schedule(cfg.sr_shift, cfg.sr_shift_final, cfg.sr_schedule, cfg.n_nqs)


def _learning_rate_value(rate: float | optax.Schedule, step: int) -> float:
    """Materialize the current learning rate for histories."""
    if callable(rate):
        return float(rate(step))
    return float(rate)


@functools.lru_cache(maxsize=32)
def _optimizer(cfg: TrainConfig) -> tuple[optax.GradientTransformation, str, float | optax.Schedule]:
    """Construct the Optax optimizer used by the NetKet driver."""
    rate		= _learning_rate(cfg)
    name		= TrainOptimizer(cfg.optimizer)
    schedule	= LearningRateSchedule(cfg.learning_rate_schedule)
    chain		: list[optax.GradientTransformation] = []
    if cfg.grad_clip > 0:
        chain.append(optax.clip_by_global_norm(cfg.grad_clip))

    if name is TrainOptimizer.ADAM:
        chain.append(optax.adam(rate))
    else:
        chain.append(optax.sgd(rate))

    tag = name.value
    if schedule is not LearningRateSchedule.CONSTANT:
        tag += f"/{schedule.value}"
    return optax.chain(*chain), tag, rate


# ----------------------------------------------


def train_vmc(
    params	: Any | None,
    model	: ARTransformer,
    operator: Any,				# NetKet operator on ``model.hilbert`` (already built by the caller)
    cfg		: TrainConfig,
    key		: jax.Array,
    n_terms : int | None = None,
    desc	: str = "VMC",
    *,
    time_budget_seconds: float | None = None,
    on_step: Callable[[int, Any], None] | None = None,
) -> tuple[Any, list[dict[str, float]], jax.Array]:
    r"""Run NetKet VMC against a fixed Hamiltonian and return ``(variables, history, key)``.

    Parameters
    ----------
    params : Any | None
        Initial model variables (``{"params": ...}``).	``None`` initializes a fresh
        state from the PRNG key.
    model : ARTransformer
        The :class:`AbstractARNN` ansatz; ``model.hilbert`` defines the spin space.
    operator : netket.operator.AbstractOperator
        The (rotated) Hamiltonian as a NetKet operator on ``model.hilbert``.
    cfg : TrainConfig
        Optimizer / sampling hyperparameters.  ``use_sr`` selects ``VMC_SR``
        (``diag_shift = cfg.sr_shift``); otherwise first-order optimization.
    key : jax.Array
        PRNG key; split deterministically for parameter init and the sampler so two
        runs with the same key reproduce bit-for-bit.
    n_terms : int | None
        Number of Pauli terms, recorded in the history (diagnostic only).

    ``time_budget_seconds`` counts setup, compilation, and completed updates.
    ``on_step`` receives the completed update number and variables for optional
    checkpoint monitoring; its cost is included in that budget.
    History energies are the Monte Carlo estimates used to compute each update,
    measured before applying it; they are not post-update checkpoint energies.

    Notes
    -----
    With ``use_sr=True`` this uses minSR with SPRING momentum: the parameter-space
    metric is replaced by a sample-space neural tangent kernel. CPU batches use
    explicit Jacobians; accelerators use NetKet's on-the-fly path, which
    avoids storing the full parameter Jacobian, but still constructs a dense
    ``n_samples`` kernel (twice that linear dimension for a complex wavefunction)
    and solves it with robust Cholesky. ``chunk_size`` limits contraction memory;
    it does not make the final linear solve matrix-free.

    Example
    -------
    A complex NQS with 512 Monte Carlo samples produces a real kernel of linear
    dimension ``2 * 512``.

    >>> complex_kernel_dimension = 2 * 512
    >>> complex_kernel_dimension
    1024

    Doubling the samples therefore gives roughly eight
    times the Cholesky work, before accounting for the NTK contractions.

    References
    ----------
    - Chen and Heyl, minimum-step stochastic reconfiguration,
      https://arxiv.org/abs/2302.01941
    - Rende et al., sample-space linear-algebra identity,
      https://arxiv.org/abs/2310.05715
    """

    # Include driver construction and compilation in an end-to-end time budget.
    budget_started = time.perf_counter()
    if time_budget_seconds is not None and (not np.isfinite(time_budget_seconds) or time_budget_seconds <= 0.):
        raise ValueError("time_budget_seconds must be finite and positive")
    hilbert						= model.hilbert
    key, k_init, k_samp			= jax.random.split(key, 3)
    sampler						= nk.sampler.ARDirectSampler(hilbert)
    op							= _to_operator(operator)

    # ARDirectSampler draws uncorrelated samples, so no Metropolis warm-up is needed.
    # chunk_size splits the per-step (sample x connected-config) evaluation to cap GPU memory.
    chunk_size					= cfg.chunk_size if (cfg.chunk_size and cfg.chunk_size > 0) else None
    mc_kwargs: dict[str, Any]	= dict(
                                    n_samples			= cfg.n_samples,
                                    n_discard_per_chain = 0,
                                    sampler_seed		= _seed_from_key(k_samp),
                                    chunk_size			= chunk_size,
                                )
    if params is None:
        vstate = nk.vqs.MCState(sampler, model, seed=_seed_from_key(k_init), **mc_kwargs)
    else:
        vstate = nk.vqs.MCState(sampler, model, variables=params, **mc_kwargs)

    n_terms = int(n_terms if n_terms is not None else 0)

    # Optimization disabled (n_nqs <= 0): just one evaluation of the
    # (untrained / warm-started) state so the history and e_fin are well defined.
    if cfg.n_nqs <= 0:
        stats	= vstate.expect(op)
        energy	= float(np.real(stats.mean))
        var		= float(stats.variance)
        log(f"[{desc}] independent MC: {cfg.n_samples} samples | E={energy:+.9f} | variance={var:.6g} | SE={float(stats.error_of_mean):.3g} | {time.perf_counter() - budget_started:.3f} s", level=3)
        return vstate.variables, [{"loss": energy, "energy": energy, "variance": var, "standard_error": float(stats.error_of_mean), "n_terms": float(n_terms), "learning_rate": 0.0, "sr_shift": 0.0, "dE": 0.0, "seconds": 0.0}], key

    # Optimizer + driver: SR (kernel/minSR) or first-order Adam with a global-norm clip.
    optimizer, opt_tag, rate	= _optimizer(cfg)
    diag_shift					= _sr_diag_shift(cfg)
    if cfg.use_sr:
        # use_ntk=True -> minSR / NTK kernel (arXiv:2302.01941): solves the
        # n_samples x n_samples kernel, so cost scales with samples not parameters.
        # on_the_fly avoids materializing the parameter Jacobian while constructing
        # that kernel; it does not make the final Cholesky solve matrix-free.
        # CPU batches favor explicit Jacobians; accelerator NTKs use the lazy path.
        backend = jax.tree_util.tree_leaves(vstate.parameters)[0].device.platform
        driver		= nk.driver.VMC_SR(op, optimizer, diag_shift=diag_shift, variational_state=vstate, chunk_size_bwd=chunk_size, use_ntk=True, on_the_fly=backend != "cpu", momentum=cfg.sr_momentum or None)
        shift_tag	= f"{cfg.sr_shift:.1e}" if cfg.sr_schedule is LearningRateSchedule.CONSTANT else f"{cfg.sr_shift:.1e}->{cfg.sr_shift_final:.1e}/{cfg.sr_schedule.value}"
        opt_tag		= f"minSR:{opt_tag}(mode={driver.mode}, shift={shift_tag})" + (f" SPRING({cfg.sr_momentum})" if cfg.sr_momentum else "")
    else:
        driver		= nk.driver.VMC(op, optimizer, variational_state=vstate)

    history		: list[dict[str, float]] = []
    state		= {"e0": None, "best": float("inf"), "no_improve": 0}
    started		= time.perf_counter()

    def callback(step: int, logged: dict[str, Any], _driver: Any) -> bool:
        """Record per-step energy/variance and apply early-stopping (returns False to stop)."""
        stats	= logged["Energy"]
        energy	= float(np.real(stats.mean))
        var		= float(stats.variance)
        if state["e0"] is None:
            state["e0"] = energy
        history.append({
            "loss"			: energy,
            "energy"		: energy,
            "variance"		: var,
            "standard_error": float(stats.error_of_mean),
            "n_terms"		: float(n_terms),
            "learning_rate" : _learning_rate_value(rate, step),
            "sr_shift"		: _learning_rate_value(diag_shift, step) if cfg.use_sr else 0.0,
            "dE"			: energy - state["e0"],
            "seconds"		: time.perf_counter() - started,
        })

        if on_step is not None:
            on_step(len(history), _driver.state.variables)
        if time_budget_seconds is not None:
            # The completed-step hook has already materialized the parameter update.
            if time.perf_counter() - budget_started >= time_budget_seconds:
                return False

        if cfg.early_stop_var > 0 and var < cfg.early_stop_var:
            log(f"early stop: var(E) = {var:.2e} < {cfg.early_stop_var:.2e}", level=3)
            return False

        # Early stopping based on energy improvement: if the energy does not improve by at least ``early_stop_tol`` for ``early_stop_patience`` steps, stop the optimization.
        if cfg.early_stop_patience > 0:
            if energy < state["best"] - cfg.early_stop_tol:
                state["best"], state["no_improve"] = energy, 0
            else:
                state["no_improve"] += 1
                if state["no_improve"] >= cfg.early_stop_patience:
                    log(f"early stop: no improvement for {cfg.early_stop_patience} steps", level=3)
                    return False
        return True

    class CompletedVMCStep(nk.callbacks.AbstractCallback):
        """Legacy callbacks run before the update; observers need the updated state."""
        @property
        def callback_order(self):
            return 20	# Finish NetKet's logging and progress callbacks before stopping.

        def on_step_end(self, step, logged, completed_driver):
            jax.block_until_ready(completed_driver.state.variables)
            if not callback(step, logged, completed_driver):
                raise nk.callbacks.StopRun("VMC stopping criterion reached after a completed update")

    log(f"{desc}: {opt_tag} | lr={cfg.learning_rate:g} | {cfg.n_nqs} updates | {cfg.n_samples} MC samples | chunk={chunk_size or 'all'}", level=2)
    driver.run(
        n_iter			= cfg.n_nqs,
        out				= nk.logging.RuntimeLog(),
        callback		= CompletedVMCStep(),
        show_progress	= not progress_disabled(),
    )
    # Include the final completed update in accelerator wall times.
    jax.block_until_ready(vstate.variables)
    elapsed = time.perf_counter() - budget_started
    log(f"{desc}: completed {len(history)}/{cfg.n_nqs} updates in {elapsed:.3f} s (setup and compilation included)", level=2)

    return vstate.variables, history, key


# ----------------------------------------------
#! EOF
# ----------------------------------------------
