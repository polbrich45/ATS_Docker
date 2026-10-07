"""Run the configurable Hubbard NQS -> orbital Ritz -> teacher -> VMC loop."""

import argparse

import jax

from src.config import AdaptiveConfig, AdaptiveStep, ModelConfig, RotationConfig, TeacherConfig, TrainConfig, CircuitConfig
from src.fermionic import HubbardConfig, run_fermionic_loop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/hubbard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--Lx", type=int, default=2)
    parser.add_argument("--Ly", type=int, default=2)
    parser.add_argument("--pbc", action="store_true")
    parser.add_argument("--t", type=float, default=1.)
    parser.add_argument("--U", type=float, default=4.)
    parser.add_argument("--n-up", type=int, default=None)
    parser.add_argument("--n-down", type=int, default=None)
    parser.add_argument("--n-outer", type=int, default=2)
    parser.add_argument("--n-nqs", type=int, default=40)
    parser.add_argument("--n-rot", type=int, default=30)
    parser.add_argument("--n-basin", type=int, default=0)
    parser.add_argument("--n-teacher", type=int, default=30)
    parser.add_argument("--n-recovery", type=int, default=20)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--participation-weight", type=float, default=0.)
    parser.add_argument("--entanglement-weight", type=float, default=0.)
    parser.add_argument("--entanglement-order", choices=("spatial", "model"), default="spatial")
    parser.add_argument("--sampling", choices=("gumbel", "frequency"), default="gumbel")
    parser.add_argument("--n-subspace", type=int, default=12)
    parser.add_argument("--n-seed-samples", type=int, default=None, help="Frequency Born-draw cap; automatic device/basis-size budget when omitted")
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--restart-from", default="")
    args = parser.parse_args()
    
    # Validate arguments
    if args.n_outer < 1:
        parser.error("--n-outer must be positive")
    jax.config.update("jax_enable_x64", True)
    step = AdaptiveStep(train=TrainConfig(n_nqs=args.n_nqs, n_samples=args.n_samples, learning_rate=args.learning_rate, use_sr=False, optimizer="adam"),
        rotation=RotationConfig(n_rot=args.n_rot, n_basin=args.n_basin, n_subspace=args.n_subspace, sampling=args.sampling, n_seed_samples=args.n_seed_samples, participation_weight=args.participation_weight, entanglement_weight=args.entanglement_weight, entanglement_order=args.entanglement_order),
        teacher=TeacherConfig(n_teacher=args.n_teacher), n_recovery=args.n_recovery,
    )
    
    # Build the configs and run the loop
    config  = AdaptiveConfig(circuit=CircuitConfig(n_layers=args.n_layers), steps=(step,) * args.n_outer, display="silent", restart_from=args.restart_from)
    model   = ModelConfig(width=args.width, depth=args.depth, heads=args.heads, mlp_ratio=1)
    hubbard = HubbardConfig(Lx=args.Lx, Ly=args.Ly, pbc=args.pbc, t=args.t, U=args.U, n_up=args.n_up, n_down=args.n_down)
    result  = run_fermionic_loop(hubbard, model, config, seed=args.seed, output_dir=args.output_dir)
    for row, rotation in zip(result["outer_history"], result["rotation_history"]):
        print(f"outer={row['outer']} accepted={row['accepted']} E={row['energy']:+.8f} Ritz={row['proposal_ritz_energy']:+.8f} angle_norm={rotation['angle_norm']:.4f}")
    reference = f"{result['exact_energy']:+.8f}" if result["exact_energy"] is not None else "not computed"
    print(f"exact={reference} final={result['final_energy']:+.8f}")


if __name__ == "__main__":
    main()

# -------------------------
#! EOF
# -------------------------
