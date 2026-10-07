"""Small public-API benchmark for compile reuse in the adaptive loop.

The default run uses two identical :class:`AdaptiveStep` objects. With
``JAX_LOG_COMPILES=1``, a repeated hot-loop compilation at the second step is a
regression; one-off NetKet/Flax setup compilations are expected.

Examples
--------
python -m scripts.bench_adaptive --device cpu
JAX_LOG_COMPILES=1 python -m scripts.bench_adaptive --device gpu --log-compiles
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", True)

# Import our public API for the adaptive loop, which is the main entry point for this benchmark.
from src.adaptive import run_adaptive_loop
from src.config import AdaptiveConfig, AdaptiveStep, CircuitConfig, FinalConfig, LatticeConfig, ModelConfig, RotationConfig, TeacherConfig, TrainConfig


def _benchmark_config(n_outer: int, n_rot: int, n_teacher: int) -> tuple:
    """Return one tiny problem with identical static shapes in every outer step."""

    lattice = LatticeConfig(Lx=2, Ly=2, pbc=False, j2=0.5)
    model   = ModelConfig(width=8, depth=2, heads=2, mlp_ratio=1, conserve_total_sz=False)
    step    = AdaptiveStep(
        train       = TrainConfig(n_samples=32, n_nqs=5, learning_rate=1.0e-3, use_sr=False),
        rotation    = RotationConfig(n_subspace=8, n_rot=n_rot, learning_rate=1.0e-2),
        teacher     = TeacherConfig(n_teacher=n_teacher),
        n_recovery  = 2,
    )
    adaptive = AdaptiveConfig(
        circuit     = CircuitConfig(n_layers=1, family=("givens",)),
        steps       = tuple(
            dc.replace(
                step,
                rotation=dc.replace(step.rotation, learning_rate=1.0e-2 * (index + 1), grad_clip=0.5 + index,
                    participation_weight=0.05 if index % 2 == 0 else -0.05,
                    entanglement_weight=-0.02 if index % 2 == 0 else 0.02,
                    nonstoquasticity_weight=0.03 if index % 2 == 0 else -0.03,
                ),
            )
            for index in range(n_outer)
        ),
        final=FinalConfig(solver="none"),
        display="silent",
    )
    return lattice, model, adaptive


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu", "tpu", "auto"), default="cpu")
    parser.add_argument("--log-compiles", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-outer", type=int, default=2)
    parser.add_argument("--n-rot", type=int, default=2)
    parser.add_argument("--n-teacher", type=int, default=2)
    args = parser.parse_args(argv)

    if args.log_compiles:
        jax.config.update("jax_log_compiles", True)

    if args.device != "auto":
        devices = jax.devices(args.device)
        if not devices:
            raise RuntimeError(f"No JAX {args.device!r} device is available")
        jax.config.update("jax_default_device", devices[0])

    lattice, model, adaptive = _benchmark_config(args.n_outer, args.n_rot, args.n_teacher)

    started     = time.perf_counter()
    result      = run_adaptive_loop(lattice, model, adaptive, seed=args.seed)
    elapsed     = time.perf_counter() - started
    rows        = result.get("outer_history", ())

    print(f"backend: {jax.default_backend()}")
    print(f"x64: {jax.config.jax_enable_x64}")
    print(f"problem: n_outer={args.n_outer} n_rot={args.n_rot}, n_teacher={args.n_teacher} n_subspace=8")
    print(f"total time: {elapsed:.6f} s")
    print(f"outer rows: {len(rows)}")
    if rows:
        for index, row in enumerate(rows):
            timing = row.get("timing", {})
            print(
                f"outer {index}: "
                f"E={row['energy']:+.8f} accepted={row['accepted']} "
                f"nqs={timing.get('nqs_seconds', float('nan')):.4f}s "
                f"sample={timing.get('sample_seconds', float('nan')):.4f}s "
                f"angle={timing.get('angle_seconds', float('nan')):.4f}s "
                f"teacher={timing.get('teacher_seconds', float('nan')):.4f}s "
                f"recovery={timing.get('recovery_seconds', float('nan')):.4f}s "
                f"eval={timing.get('evaluation_seconds', float('nan')):.4f}s "
                f"terms={row.get('operator_terms', 0)}"
            )
    print("Inspect JAX_LOG_COMPILES output for repeated programs at identical shapes.")


if __name__ == "__main__":
    main()
    
# -----------------------------------------------
#! EOF
# -----------------------------------------------
