r"""Joint sampled-Ritz probability and relative-phase distillation.

For N spins and M retained configurations, q_i = |v_i|^2 / ||v||^2 is the
Ritz target and p_i = |psi(x_i)|^2 is the normalized NQS probability.
The default loss is KL(q || p)/N + w_phi (1-C_phi), where
C_phi = |sum_i q_i exp[i(arg psi_i - arg v_i)]| ignores global phase.

Optional diagonal and uniform log-probability biases give amplitude
coefficients a_i = q_i (1-w_E A_i) - w_H/M, with
A_i = (H_ii - sum_j q_j H_jj)/(N J_rms). J_rms is the nonidentity
Pauli-coefficient scale. The loss is bounded below only if every a_i >= 0;
centering A alone is insufficient. Both biases are disabled by default.
Zero-target states carry no phase weight. Invalid Ritz vectors and states
outside the NQS spin or fermion sector are rejected before fitting.

This is a supervised surrogate, not a physical energy minimization.
See docs/teacher.md for conventions and the relation to HI-NQS.
"""

from __future__     import annotations

from functools      import lru_cache
from typing         import Any, Callable
from tqdm.auto      import trange

import jax
import jax.numpy as jnp
import numpy as np
import optax


# Ours 
from .config        import TeacherConfig
from .logging_utils import interruptible, progress_disabled
from .model         import bits_to_inputs, in_sector

# -------------------------

def ritz_probability_target(vector: jax.Array | np.ndarray, size: int) -> np.ndarray:
    """Return stable normalized ``|v_i|^2`` weights for a Ritz vector."""
    if size < 0:
        raise ValueError("size must be non-negative")

    # Compute the normalized probabilities from the Ritz vector
    amplitudes = np.abs(np.asarray(vector, dtype=np.complex128).reshape(-1))
    if amplitudes.size != size:
        raise ValueError(f"Ritz vector has length {amplitudes.size}, expected {size}")
    if size == 0:
        return np.zeros(0, dtype=np.float64)

    # Scale the amplitudes to avoid overflow and underflow, then normalize to sum to 1.
    # A zero or all-non-finite vector has no admissible direction and falls back to uniform.
    finite  = np.isfinite(amplitudes)
    scale   = float(np.max(amplitudes[finite])) if np.any(finite) else 0.0
    if scale <= 0.0:
        return np.full(size, 1.0 / size, dtype=np.float64)
    weights = np.square(np.where(finite, amplitudes / scale, 0.0))
    total   = float(np.sum(weights))
    return np.asarray(weights / total, dtype=np.float64) if (np.isfinite(total) and total > 0.0) else np.full(size, 1.0 / size, dtype=np.float64)


# -------------------------


def _tree_l2(tree: Any) -> jax.Array:
    """L2 norm of a real or complex pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return (
        jnp.sqrt(sum(jnp.sum(jnp.real(x * jnp.conj(x))) for x in leaves))
        if leaves
        else jnp.asarray(0.0, dtype=jnp.float64)
    )


@lru_cache(maxsize=32)
def _teacher_kernel(model: Any, learning_rate: float, grad_clip: float, method: str):
    """Cache one fixed-target update; L-BFGS reuses the last line-search gradient."""
    transforms: list[optax.GradientTransformation] = []
    if grad_clip > 0.0:
        transforms.append(optax.clip_by_global_norm(grad_clip))
    if method == "lbfgs":
        optimizer = optax.lbfgs(learning_rate=learning_rate, memory_size=10, linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=20))
    else:
        transforms.append(optax.adam(learning_rate))
        optimizer = optax.chain(*transforms)

    def objective(params, spins, target, target_entropy, advantage, phase_target, phase_q, weights):
        n_sites     = jnp.asarray(spins.shape[1], dtype=float)
        log_psi     = model.apply(params, spins)                                # log-amplitude of the NQS in the rotated basis
        log_prob    = 2.0 * jnp.real(log_psi)                                   # log-probability of the NQS in the rotated basis
        # Mask zero-target states: log p is undefined there and must not enter the KL.
        target_log_prob = jnp.where(target > 0.0, log_prob, 0.0)
        # teacher_kl is the KL(q || p) over the retained support, -sum q log p - H(q); the
        # per-site division is the loss normalization used throughout this module.
        teacher_kl  = (-jnp.sum(target * target_log_prob) - target_entropy) / n_sites
        # The autoregressive state is unnormalized on the retained support, so teacher_kl omits
        # the mass outside it; adding log Z per site completes the conditional KL.
        log_mass    = jax.scipy.special.logsumexp(log_prob)
        conditional_kl = teacher_kl + log_mass / n_sites
        # Optional HI-NQS auxiliary terms; each is skipped when its weight is exactly zero.
        reinforce   = jax.lax.cond(weights[0] != 0.0, lambda: jnp.sum(target * advantage * target_log_prob) / n_sites, lambda: jnp.zeros_like(teacher_kl))
        entropy     = jax.lax.cond(weights[1] != 0.0, lambda: jnp.mean(log_prob) / n_sites, lambda: jnp.zeros_like(teacher_kl))
        
        # Global-phase-invariant coherence: the weighted mean factor exp[i(arg psi_i - arg v_i)]
        # cancels the arbitrary global phase shared by the Ritz vector and the NQS.
        phase_delta = jnp.where(phase_q > 0.0, jnp.imag(log_psi) - phase_target, 0.0)
        coherence   = jnp.abs(jnp.sum(phase_q * jnp.exp(1.0j * phase_delta)))
        # The AR state is globally normalized: this overlap includes omitted mass.
        overlap     = jnp.sum(jnp.sqrt(target) * jnp.exp(log_psi - 1.0j * phase_target))
        fidelity    = jnp.clip(jnp.abs(overlap) ** 2, 0.0, 1.0)
        # Probability KL plus the optional auxiliary objectives; only the KL remains when all
        # weights vanish.
        loss        = teacher_kl + (weights[0] * reinforce) + (weights[1] * entropy) + (weights[2] * (1.0 - coherence))
        return loss, {"teacher_kl": teacher_kl, "reinforce": reinforce, "entropy": entropy, "coherence": coherence, "log_mass": log_mass, "conditional_kl": conditional_kl, "fidelity": fidelity}

    @jax.jit
    def step(params, state, spins, target, target_entropy, advantage, phase_target, phase_q, weights):
        loss_fn                     = lambda value: objective(value, spins, target, target_entropy, advantage, phase_target, phase_q, weights)
        if method == "lbfgs":
            # The initial value/gradient share one forward pass; later line-search
            # gradients are valid because the target remains fixed throughout fitting.
            (loss, metrics), gradient = jax.lax.cond(
                jnp.isfinite(state[-1].value),
                lambda: (loss_fn(params), state[-1].grad),
                lambda: jax.value_and_grad(loss_fn, has_aux=True)(params),
            )
            updates, state = optimizer.update(gradient, state, params, value=loss, grad=gradient, value_fn=lambda value: loss_fn(value)[0])
            line_evaluations = state[-1].info.num_linesearch_steps
            decrease_error = state[-1].info.decrease_error
        else:
            (loss, metrics), gradient = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, state = optimizer.update(gradient, state, params)
            line_evaluations, decrease_error = 0, 0.0
        return optax.apply_updates(params, updates), state, loss, metrics, _tree_l2(gradient), line_evaluations + 1, decrease_error

    return optimizer, step, jax.jit(objective)


# -------------------------


def fit_ritz_teacher(
    model           : Any,
    params          : Any,
    samples         : np.ndarray,
    ritz_vector     : jax.Array | np.ndarray,
    diagonal        : np.ndarray,
    ritz_energy     : float,
    energy_scale    : float,
    config          : TeacherConfig,
    *,
    on_step         : Callable[[int, Any], None] | None = None,
) -> tuple[Any, list[dict[str, float]], dict[str, float]]:
    r"""Directly fit the joint spin Ritz target with HI-NQS auxiliary terms.

    ``diagonal`` contains $\langle x_i|H|x_i\rangle$ in the current rotated
    spin basis, and ``energy_scale = 1/(N J_rms)`` makes its REINFORCE
    advantage dimensionless. The phase target and coherence loss are defined
    in the module docstring above. ``on_step`` observes the parameters after
    each completed update; loss history entries are evaluated before it.
    """
    samples = np.asarray(samples, dtype=np.int8)
    size    = samples.shape[0] if samples.ndim == 2 else 0
 
    # Check inputs
    if samples.ndim != 2 or size == 0 or np.any((samples != 0) & (samples != 1)):
        raise ValueError("Ritz teacher needs a non-empty {0, 1} bit basis")
    if not in_sector(model, samples):
        raise ValueError("Ritz teacher configurations violate the NQS spin or particle sector")

    # Check that the Ritz vector and diagonal energies are finite and match the basis size
    diagonal = np.asarray(diagonal, dtype=np.float64).reshape(-1)
    if diagonal.shape != (size,) or not np.all(np.isfinite(diagonal)):
        raise ValueError("teacher diagonal energies must be finite and match the Ritz basis")
    if not np.isfinite(ritz_energy) or not np.isfinite(energy_scale) or energy_scale <= 0.0:
        raise ValueError("teacher Ritz energy and energy scale must be finite")

    vector          = np.asarray(ritz_vector, dtype=np.complex128).reshape(-1)
    if not np.all(np.isfinite(vector)) or not np.any(np.abs(vector) > 0.0):
        raise ValueError("Ritz teacher requires a finite, nonzero eigenvector")
    target_real     = ritz_probability_target(vector, size)
    # Centered local-energy advantage, made dimensionless by energy_scale = 1/(N J_rms).
    mean_diagonal   = float(np.sum(target_real * diagonal))
    advantage       = (diagonal - mean_diagonal) * float(energy_scale)
    # The combined amplitude loss is -sum_i a_i log p_i plus a constant.
    # Every a_i must be nonnegative; centering the advantage alone is insufficient.
    headroom          = target_real * (1.0 - config.reinforce_weight * advantage)
    amplitude_weights = headroom - config.entropy_weight / size
    if np.any(amplitude_weights < 0.0):
        # entropy_weight <= size * min_i headroom keeps every coefficient nonnegative.
        admissible = float(np.min(headroom)) * size
        raise ValueError(
            "Teacher auxiliary weights make the loss unbounded below; "
            f"entropy_weight must be below {admissible:.6g} for this target "
            "(reduce reinforce_weight or set both weights to zero)"
        )

    # Phase coherence target and weights for the global-phase-invariant term: follow the Ritz
    # weight but exclude configurations with zero target, then renormalize to a weighted mean.
    usable_phase    = np.isfinite(vector.real) & np.isfinite(vector.imag) & (np.abs(vector) > 0.0)
    phase_weights   = np.where(usable_phase, target_real, 0.0)
    phase_total     = float(np.sum(phase_weights))
    if phase_total > 0.0:
        phase_weights /= phase_total
  
    # Angle of the Ritz vector as the phase target; the placeholder 1 keeps np.angle finite where
    # the weight is already zero.
    phase_target    = np.angle(np.where(usable_phase, vector, 1.0 + 0.0j))
    # Shannon entropy of the target distribution, used for the KL constant and reporting.
    target_entropy  = float(-np.sum(target_real * np.log(np.maximum(target_real, 1.0e-300))))

    dtype           = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    # Fixed target arrays passed into the jitted kernel; dtype follows the JAX x64 policy.
    arrays          = (
                        jnp.asarray(bits_to_inputs(model, samples)),
                        jnp.asarray(target_real, dtype=dtype),
                        jnp.asarray(target_entropy, dtype=dtype),
                        jnp.asarray(advantage, dtype=dtype),
                        jnp.asarray(phase_target, dtype=dtype),
                        jnp.asarray(phase_weights, dtype=dtype),
                        jnp.asarray((config.reinforce_weight, config.entropy_weight, config.phase_weight), dtype=dtype),
                    )
 
    def summarize(loss, aux) -> dict[str, float]:
        """Per-site teacher losses; mass and phase terms derive from log Z and C_phi."""
        return {
            "teacher_loss"          : float(loss),
            "teacher_kl_per_site"   : float(aux["teacher_kl"]),
            "reinforce_per_site"    : float(aux["reinforce"]),
            "entropy_per_site"      : float(aux["entropy"]),
            "phase_coherence"       : float(aux["coherence"]),
            "teacher_fidelity"      : float(aux["fidelity"]),
            "phase_loss"            : float(config.phase_weight * (1.0 - aux["coherence"])),
            "subspace_mass"         : float(jnp.exp(aux["log_mass"])),
            "mass_loss_per_site"    : float(-aux["log_mass"] / samples.shape[1]),
            "conditional_kl_per_site": float(aux["conditional_kl"]),
        }

    # Evaluate the teacher loss and metrics before any updates
    optimizer, run_step, evaluate   = _teacher_kernel(model, float(config.learning_rate), float(config.grad_clip), config.optimizer)
    state                           = optimizer.init(params)
    history: list[dict[str, float]] = []
    bar                             = trange(config.n_teacher, desc="teacher", leave=True, disable=progress_disabled())
    for index in interruptible(bar, on_interrupt="teacher interrupted"):
        params, state, loss, aux, grad_norm, evaluations, decrease_error = run_step(params, state, *arrays)
        row = {"step": float(index), **summarize(loss, aux), "grad_norm": float(grad_norm), "function_evaluations": float(evaluations), "linesearch_decrease_error": float(decrease_error)}
        history.append(row)
        if on_step is not None:
            on_step(index + 1, params)
        bar.set_postfix(KL=f"{row['teacher_kl_per_site']:.3e}", C=f"{row['phase_coherence']:.3f}", E=f"{row['entropy_per_site']:.3f}", R=f"{row['reinforce_per_site']:.3e}", L=f"{row['teacher_loss']:.3e}", G=f"{row['grad_norm']:.3e}")

    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(params)):
        raise FloatingPointError("teacher update produced non-finite NQS parameters")

    final_loss, final_aux = evaluate(params, *arrays)
    if not np.isfinite(float(final_loss)):
        raise FloatingPointError("teacher objective is non-finite")
    metrics = { 
        "function_evaluations"              : 1 + sum(row["function_evaluations"] for row in history), **summarize(final_loss, final_aux),
        "target_entropy"                    : target_entropy,
        # The effective support fraction is the exponential of the target entropy divided by the basis size
        "target_effective_support_fraction" : float(np.exp(target_entropy) / size),
    }
    return params, history, metrics


# -------------------------
#! EOF
# -------------------------
