"""Command-line entry point for the explicit adaptive NQS workflow."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import jax

from .adaptive 		import run_adaptive_loop
from .logging_utils import display_context, log
from .config 		import (
                        AdaptiveConfig,
                        AdaptiveStep,
                        BasisParameterization,
                        CircuitConfig,
                        FinalConfig,
                        HamiltonianType,
                        LatticeConfig,
                        LatticeType,
                        LearningRateSchedule,
                        ModelConfig,
                        RotationConfig,
                        SubspaceSampling,
                        TeacherConfig,
                        TrainConfig,
                        TrainOptimizer,
                    )

# ------------------------------------------------------------


def _enum_arg(value: str, enum_type):
    """Return an enum member from a case-insensitive name or value."""

    key = value.strip().lower().replace("-", "_")
    for item in enum_type:
        if key in {item.value.lower(), item.name.lower()}:
            return item
    valid = ", ".join(item.value for item in enum_type)
    raise argparse.ArgumentTypeError(f"expected one of {{{valid}}}, got {value!r}")


def _select_device(args: argparse.Namespace) -> None:
    """Select numerical precision and, when requested, one JAX backend device."""

    jax.config.update("jax_enable_x64", args.x64)
    jax.config.update("jax_enable_compilation_cache", True)
    if args.device == "auto":
        return

    # Check that the requested device is available and set it as the default for all JAX operations.
    devices = jax.devices(args.device)
    if not devices:
        raise RuntimeError(f"No JAX {args.device!r} device is available")
    jax.config.update("jax_default_device", devices[0])


# -----------------------------------------------------------


def _apply_quick(args: argparse.Namespace) -> None:
    """Replace expensive numerical sizes with one end-to-end smoke step."""

    if not args.quick:
        return
    args.lx 				= 2
    args.ly 				= 2
    args.pbc 				= False
    args.width 				= 8
    args.depth 				= 1
    args.heads 				= 2
    args.mlp_ratio 			= 1
    args.n_outer 			= 1
    args.n_nqs 				= 1
    args.n_rot 				= 1
    args.n_teacher 			= 1
    args.n_recovery			= 1                                         # Fixed-angle VMC updates after teaching, before proposal acceptance
    args.n_samples 			= 32
    args.n_subspace 		= 4 if args.conserve_total_sz else 8        # Unique Ritz states: automatic 128 on CPU, 256 on GPU/TPU
    args.n_up				= 2 if args.conserve_total_sz else None     # Spin sector for the final exact reference; None means automatic
    args.use_sr 			= False
    args.final_solver 		= "none"
    args.final_ritz_size 	= 0


# -----------------------------------------------------------


def _configs(args: argparse.Namespace) -> tuple[LatticeConfig, ModelConfig, AdaptiveConfig]:
    """Build one scalar step and repeat it for the requested outer count."""

    hamiltonian 	= _enum_arg(args.hamiltonian, HamiltonianType)
    lattice_type 	= LatticeType.HONEYCOMB if hamiltonian is HamiltonianType.KITAEV_GAMMA else _enum_arg(args.lattice, LatticeType)
    impurities 		= tuple((int(entry[0]), *entry[1:]) for entry in (args.impurity or ()))
    lattice 		= LatticeConfig(
                        lattice=lattice_type,
                        hamiltonian=hamiltonian,
                        Lx=args.lx,
                        Ly=args.ly,
                        pbc=args.pbc,
                        j1=args.j1,
                        j2=args.j2,
                        K=args.K,
                        J=args.J,
                        delta=args.delta,
                        gamma=args.gamma,
                        hx=args.hx,
                        hy=args.hy,
                        hz=args.hz,
                        impurities=impurities,
                    )
    model 			= ModelConfig(
                        width=args.width,
                        depth=args.depth,
                        heads=args.heads,
                        mlp_ratio=args.mlp_ratio,
                        phase_head_depth=args.phase_head_depth,
                        phase_head_width=args.phase_head_width,
                        phase_scale=args.phase_scale,
                        phase_init_std=args.phase_init_std,
                        conserve_total_sz=args.conserve_total_sz,
                        n_up=args.n_up,
                        use_marshall_sign=args.marshall,
                    )
    train 			= TrainConfig(
                        n_samples=args.n_samples,
                        n_nqs=args.n_nqs,
                        learning_rate=args.learning_rate,
                        learning_rate_final=args.learning_rate_final,
                        learning_rate_schedule=args.lr_schedule,
                        optimizer=args.optimizer,
                        grad_clip=args.nqs_grad_clip,
                        chunk_size=args.chunk_size if args.chunk_size > 0 else None,
                        use_sr=args.use_sr,
                        sr_shift=args.sr_shift,
                        sr_shift_final=args.sr_shift_final,
                        sr_schedule=args.sr_schedule,
                        sr_momentum=args.sr_momentum,
                        early_stop_var=args.early_stop_var,
                        early_stop_patience=args.early_stop_patience,
                        early_stop_tol=args.early_stop_tol,
                    )
    rotation 		= RotationConfig(
                        n_rot=args.n_rot,
                        nqs_energy_every=args.nqs_energy_every,
                        n_subspace=args.n_subspace,
                        sampling=args.subspace_sampling,
                        n_seed_samples=args.n_seed_samples,
                        learning_rate=args.rotation_learning_rate,
                        grad_clip=args.rotation_grad_clip,
                        participation_weight=args.participation_weight,
                        entanglement_weight=args.entanglement_weight,
                        nonstoquasticity_weight=args.nonstoquasticity_weight,
                        nested_weight=args.nested_weight,
                        f_retain=args.f_retain,
                    )
    teacher 		= TeacherConfig(
                        n_teacher=args.n_teacher,
                        learning_rate=args.teacher_learning_rate,
                        grad_clip=args.teacher_grad_clip,
                        reinforce_weight=args.teacher_reinforce_weight,
                        entropy_weight=args.teacher_entropy_weight,
                        phase_weight=args.teacher_phase_weight,
                    )
    step 			= AdaptiveStep(train=train, rotation=rotation, teacher=teacher, n_recovery=args.n_recovery)
    circuit 		= CircuitConfig(
                        n_layers=args.n_layers,
                        family=tuple(args.family),
                        include_nnn=args.include_nnn,
                        parameterization=args.parameterization,
                        use_marshall_clifford=args.use_marshall_clifford,
                        theta_cutoff=args.theta_cutoff,
                    )
    final 			= FinalConfig(
                        ritz_size=args.final_ritz_size,
                        solver=args.final_solver,
                        dmrg_mps_bond_dim=args.dmrg_mps_bond_dim,
                        dmrg_mpo_max_bond=args.dmrg_mpo_max_bond,
                        dmrg_mpo_cutoff=args.dmrg_mpo_cutoff,
                        dmrg_tol=args.dmrg_tol,
                        dmrg_sweeps=args.dmrg_sweeps,
                    )
    adaptive 		= AdaptiveConfig(
                        circuit=circuit,
                        steps=tuple(step for _ in range(args.n_outer)),
                        final=final,
                        display=args.display,
                        restart_from=args.restart_from,
                        acceptance_sigma=args.acceptance_sigma,
                    )
    return lattice, model, adaptive


# -----------------------------------------------------------


def command_run(args: argparse.Namespace) -> None:
    """Run the configured adaptive workflow and print its output directory."""

    _select_device(args)
    _apply_quick(args)
    lattice, model, adaptive 	= _configs(args)
    root 						= Path(args.output_dir) if args.output_dir else Path("runs") / dt.datetime.now().strftime("%Y%m%d-%H%M%S-adaptive")
    result 						= run_adaptive_loop(lattice, model, adaptive, seed=args.seed, output_dir=root)
    with display_context(adaptive.display):
        log(f"Output: {root} | final NQS energy={result['final_energy']:+.12f}", level=0)


def make_parser() -> argparse.ArgumentParser:
    """Create the grouped command-line parser."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=command_run)

    # Add argument groups for each config dataclass, with defaults and help text.
    run = parser.add_argument_group("run", "execution and output")
    run.add_argument("--output-dir", default="")
    run.add_argument("--restart-from", default="")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", choices=("auto", "cpu", "gpu", "tpu"), default="auto")
    run.add_argument("--display", choices=("plain", "jupyter", "silent"), default="plain")
    run.add_argument("--quick", action="store_true", help="Run one tiny end-to-end smoke step")
    run.add_argument("--acceptance-sigma", type=float, default=2.0, help="Reject proposals whose recovered NQS energy increase exceeds this many combined standard errors")
    run.add_argument("--x64", action=argparse.BooleanOptionalAction, default=False, help="Enable 64-bit precision in JAX (use --no-x64 to run in float32 for speed)")

    # Lattice and Hamiltonian
    lattice = parser.add_argument_group("lattice and Hamiltonian", "LatticeConfig")
    lattice.add_argument("--lattice", choices=tuple(item.value for item in LatticeType), default="square")
    lattice.add_argument("--hamiltonian", choices=tuple(item.value for item in HamiltonianType), default="j1j2")
    lattice.add_argument("--lx", type=int, default=4)
    lattice.add_argument("--ly", type=int, default=4)
    lattice.add_argument("--pbc", action=argparse.BooleanOptionalAction, default=True)
    lattice.add_argument("--j1", type=float, default=1.0)
    lattice.add_argument("--j2", type=float, default=0.5)
    lattice.add_argument("--K", type=float, default=1.0, help="Kitaev coupling")
    lattice.add_argument("--J", type=float, default=0.0, help="Honeycomb Heisenberg coupling")
    lattice.add_argument("--delta", type=float, default=1.0)
    lattice.add_argument("--gamma", type=float, default=0.0)
    lattice.add_argument("--hx", type=float, default=0.0)
    lattice.add_argument("--hy", type=float, default=0.0, help="Honeycomb only")
    lattice.add_argument("--hz", type=float, default=0.0, help="Longitudinal field for Ising or honeycomb")
    lattice.add_argument("--impurity", action="append", nargs="+", type=float, metavar="SITE STRENGTH [PHI THETA]", help="Site-local classical field; angles are in units of pi; repeatable")

    # ModelConfig - ANN architecture
    model = parser.add_argument_group("ANN architecture", "ModelConfig")
    model.add_argument("--width", type=int, default=32)
    model.add_argument("--depth", type=int, default=2)
    model.add_argument("--heads", type=int, default=4)
    model.add_argument("--mlp-ratio", type=int, default=2)
    model.add_argument("--phase-head-depth", type=int, default=1)
    model.add_argument("--phase-head-width", type=int, default=0, help="0 reuses --width")
    model.add_argument("--phase-scale", type=float, default=1.0, help="0 disables learned phases; without a reference phase the NQS is positive")
    model.add_argument("--phase-init-std", type=float, default=0.05)
    model.add_argument("--conserve-total-sz", action="store_true")
    model.add_argument("--n-up", type=int, default=None)
    model.add_argument("--marshall", action="store_true", help="Fixed Marshall reference phase in the NQS")

    # VMC training
    train = parser.add_argument_group("NQS VMC", "TrainConfig for each repeated step")
    train.add_argument("--n-outer", type=int, default=1, help="Repeat the scalar AdaptiveStep this many times")
    train.add_argument("--n-nqs", type=int, default=300)
    train.add_argument("--n-recovery", type=int, default=20, help="Fixed-angle VMC updates after teaching, before proposal acceptance")
    train.add_argument("--n-samples", type=int, default=None, help="Default: 256 on CPU, 1024 on GPU/TPU")
    train.add_argument("--optimizer", choices=tuple(item.value for item in TrainOptimizer), default="sgd")
    train.add_argument("--learning-rate", type=float, default=2.0e-2)
    train.add_argument("--learning-rate-final", type=float, default=0.0)
    train.add_argument("--lr-schedule", choices=tuple(item.value for item in LearningRateSchedule), default="constant")
    train.add_argument("--nqs-grad-clip", type=float, default=None, help="Default: no SR clipping, norm 1 for first-order training")
    train.add_argument("--chunk-size", type=int, default=0, help="0 selects min(n_samples, 128)")
    train.add_argument("--no-sr", dest="use_sr", action="store_false", help="Disable the otherwise-default minSR/NTK update")
    train.set_defaults(use_sr=True)
    train.add_argument("--sr-shift", type=float, default=1.0e-2)
    train.add_argument("--sr-shift-final", type=float, default=1.0e-2)
    train.add_argument("--sr-schedule", choices=tuple(item.value for item in LearningRateSchedule), default="constant")
    train.add_argument("--sr-momentum", type=float, default=0.8)
    train.add_argument("--early-stop-var", type=float, default=0.0)
    train.add_argument("--early-stop-patience", type=int, default=0)
    train.add_argument("--early-stop-tol", type=float, default=1.0e-5)

    # CircuitConfig - basis circuit
    circuit = parser.add_argument_group("basis circuit", "CircuitConfig")
    circuit.add_argument("--n-layers", type=int, default=1)
    circuit.add_argument("--family", nargs="+", default=("givens",))
    circuit.add_argument("--include-nnn", action="store_true")
    circuit.add_argument("--parameterization", choices=tuple(item.value for item in BasisParameterization), default="gate")
    circuit.add_argument("--use-marshall-clifford", action="store_true")
    circuit.add_argument("--theta-cutoff", type=float, default=1.0e-3)

    rotation = parser.add_argument_group("sampled-Ritz rotation", "RotationConfig")
    rotation.add_argument("--n-rot", type=int, default=200)
    rotation.add_argument("--nqs-energy-every", type=int, default=0, help="Spin-loop physical energy probe interval during rotation; 0 reports endpoints only")
    rotation.add_argument("--n-subspace", type=int, default=None, help="Unique Ritz states: automatic 128 on CPU, 256 on GPU/TPU")
    rotation.add_argument("--subspace-sampling", choices=tuple(item.value for item in SubspaceSampling), default="frequency", help="ANN seed rule: capped empirical frequency (default) or exact unique Gumbel")
    rotation.add_argument("--n-seed-samples", type=int, default=None, help="Born-draw cap for frequency sampling: automatic max(32 * Ritz size, 2048 CPU / 8192 GPU/TPU)")
    rotation.add_argument("--rotation-learning-rate", type=float, default=5.0e-2)
    rotation.add_argument("--rotation-grad-clip", type=float, default=1.0)
    rotation.add_argument("--participation-weight", type=float, default=0.0, help="Signed weight on normalized sampled-basis participation entropy")
    rotation.add_argument("--entanglement-weight", type=float, default=0.0, help="Signed weight on normalized all-cut Renyi-2 entropy")
    rotation.add_argument("--nonstoquasticity-weight", type=float, default=0.0, help="Signed weight on sampled nonstoquasticity")
    rotation.add_argument("--nested-weight", type=float, default=0.0, help="Convex weight in [0, 1] on the normalized nested Ritz gap")
    rotation.add_argument("--f-retain", type=float, default=0.1, help="Fraction of Ritz subspace states retained across outer steps")

    teacher = parser.add_argument_group("Ritz teacher", "TeacherConfig")
    teacher.add_argument("--n-teacher", type=int, default=50)
    teacher.add_argument("--teacher-learning-rate", type=float, default=1.0e-3)
    teacher.add_argument("--teacher-grad-clip", type=float, default=1.0)
    teacher.add_argument("--teacher-reinforce-weight", type=float, default=0.0)
    teacher.add_argument("--teacher-entropy-weight", type=float, default=0.0)
    teacher.add_argument("--teacher-phase-weight", type=float, default=0.1)

    final = parser.add_argument_group("final reference", "FinalConfig")
    final.add_argument("--final-ritz-size", type=int, default=0, help="0 disables the extra final Ritz solve")
    final.add_argument("--final-solver", choices=("none", "exact", "dmrg"), default="none", help="Optional reference; exact follows the configured spin sector and is for small systems")
    final.add_argument("--dmrg-mps-bond-dim", type=int, default=64)
    final.add_argument("--dmrg-mpo-max-bond", type=int, default=0)
    final.add_argument("--dmrg-mpo-cutoff", type=float, default=1.0e-10)
    final.add_argument("--dmrg-tol", type=float, default=1.0e-8)
    final.add_argument("--dmrg-sweeps", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and run the selected command."""

    args = make_parser().parse_args(argv)
    if args.n_outer <= 0:
        raise SystemExit("--n-outer must be positive")
    args.func(args)

# ------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------
