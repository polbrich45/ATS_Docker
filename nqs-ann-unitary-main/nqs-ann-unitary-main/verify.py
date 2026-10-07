#!/usr/bin/env python3
"""Run reference Exact Diagonalization (ED) or Matrix Product State DMRG calculations

and save structured reference results to HDF5 under `data/<name>.h5`.

Example Usage
-------------
# Run exact ED for 4x3 J1-J2 lattice and save to data/j1j2_4x3.h5:
python verify.py --model j1j2 --lx 4 --ly 3 --pbc --j1 1.0 --j2 0.5 --name j1j2_4x3

# Run DMRG reference solve for 4x4 J1-J2 lattice:
python verify.py --model j1j2 --lx 4 --ly 4 --pbc --j1 1.0 --j2 0.5 --solver dmrg --dmrg-bond 128 --name j1j2_4x4_dmrg
"""

import argparse
import datetime
import pathlib
import sys
import time
from typing import Any

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FinalConfig, HamiltonianType, LatticeConfig, LatticeType
from src.final_solvers import solve_final
from src.netket_vmc import exact_ground_state
from src.operators import base_hamiltonian


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute reference ground state (ED / DMRG) and store to HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model and lattice settings
    parser.add_argument(
        "--model",
        "--hamiltonian",
        dest="hamiltonian",
        type=str,
        default="j1j2",
        choices=["j1j2", "ising", "kitaev_gamma", "heisenberg_gamma"],
        help="Hamiltonian model type",
    )
    parser.add_argument(
        "--lattice",
        type=str,
        default="square",
        choices=["square", "chain", "honeycomb"],
        help="Lattice geometry",
    )
    parser.add_argument("--lx", type=int, default=4, help="Lattice extent along X")
    parser.add_argument("--ly", type=int, default=3, help="Lattice extent along Y")
    parser.add_argument(
        "--pbc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable periodic boundary conditions",
    )

    # Physical parameters
    parser.add_argument("--j1", type=float, default=1.0, help="Nearest-neighbor exchange coupling J1")
    parser.add_argument("--j2", type=float, default=0.5, help="Next-nearest-neighbor exchange coupling J2")
    parser.add_argument("--hx", type=float, default=0.0, help="Transverse magnetic field hx")
    parser.add_argument("--hy", type=float, default=0.0, help="Magnetic field hy")
    parser.add_argument("--hz", type=float, default=0.0, help="Longitudinal magnetic field hz")
    parser.add_argument("--K", type=float, default=1.0, help="Kitaev coupling K (honeycomb)")
    parser.add_argument("--J", type=float, default=0.0, help="Heisenberg coupling J (honeycomb)")
    parser.add_argument("--gamma", type=float, default=0.0, help="Off-diagonal Gamma coupling")
    parser.add_argument("--delta", type=float, default=1.0, help="Heisenberg anisotropy delta")

    # Solver selection
    parser.add_argument(
        "--solver",
        type=str,
        default="auto",
        choices=["auto", "exact", "dmrg"],
        help="Reference solver backend ('auto' picks ED for N<=14, DMRG for N>14)",
    )
    parser.add_argument("--dmrg-bond", type=int, default=128, help="MPS bond dimension for DMRG")
    parser.add_argument("--dmrg-sweeps", type=int, default=10, help="Number of DMRG sweeps")
    parser.add_argument("--dmrg-cutoff", type=float, default=1e-10, help="MPO truncation cutoff for DMRG")

    # Storage output
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "data",
        help="Directory to save HDF5 output file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Base filename for data/<name>.h5 (defaults to auto-generated name if empty)",
    )
    return parser


def run_verification(args: argparse.Namespace) -> dict[str, Any]:
    """Build lattice, run reference solver, and save HDF5 data."""
    lattice_type = LatticeType(args.lattice)
    hamiltonian_type = HamiltonianType(args.hamiltonian)

    cfg = LatticeConfig(
        lattice=lattice_type,
        hamiltonian=hamiltonian_type,
        Lx=args.lx,
        Ly=args.ly,
        pbc=args.pbc,
        j1=args.j1,
        j2=args.j2,
        hx=args.hx,
        hy=args.hy,
        hz=args.hz,
        K=args.K,
        J=args.J,
        gamma=args.gamma,
        delta=args.delta,
    )

    print("=" * 80)
    print(f"Lattice: {cfg.lattice.value} {cfg.Lx}x{cfg.Ly} (N={cfg.n_sites} sites, dim=2^{cfg.n_sites}={cfg.dim})")
    print(f"Model  : {cfg.hamiltonian.value} | J1={cfg.j1}, J2={cfg.j2}, hx={cfg.hx}, hy={cfg.hy}, hz={cfg.hz}")
    print(f"PBC    : {cfg.pbc}")
    print("=" * 80)

    started = time.perf_counter()
    terms = base_hamiltonian(cfg)
    print(f"Hamiltonian Pauli terms: {len(terms)}")

    # Magnetization sector conservation check
    n_up = None
    if cfg.n_sites % 2 == 0 and cfg.hx == 0.0 and cfg.hy == 0.0:
        n_up = cfg.n_sites // 2
        print(f"Preserving total Sz conservation sector: n_up = {n_up} ({cfg.n_sites // 2}/{cfg.n_sites})")

    # Determine solver mode
    solver_mode = args.solver.lower()
    if solver_mode == "auto":
        solver_mode = "exact" if cfg.n_sites <= 14 else "dmrg"

    final_cfg = FinalConfig(
        solver=solver_mode,
        ritz_size=0,
        dmrg_mps_bond_dim=args.dmrg_bond,
        dmrg_sweeps=args.dmrg_sweeps,
        dmrg_mpo_cutoff=args.dmrg_cutoff,
    )

    print(f"Running solver: '{solver_mode.upper()}' ...")
    if solver_mode == "exact" and cfg.n_sites <= 14:
        exact_energy, exact_vector = exact_ground_state(cfg)
        ref_data = {"energy": float(exact_energy), "variance": 0.0, "vector": exact_vector}
    else:
        ref_data = solve_final(terms, cfg.n_sites, final_cfg, n_up=n_up)
    elapsed = time.perf_counter() - started

    energy = float(ref_data.get("energy", float("nan")))
    variance = float(ref_data.get("variance", float("nan")))
    energy_per_site = energy / cfg.n_sites if np.isfinite(energy) else float("nan")

    print("-" * 80)
    print(f"Ground State Energy          : {energy:+.12f}")
    print(f"Energy per Site (E/N)        : {energy_per_site:+.12f}")
    if np.isfinite(variance):
        print(f"Energy Variance              : {variance:.6e}")
    print(f"Elapsed Time                 : {elapsed:.2f} s")
    print("-" * 80)

    # Automatic file naming if not specified
    if not args.name:
        pbc_tag = "pbc" if cfg.pbc else "obc"
        args.name = f"{cfg.hamiltonian.value}_{cfg.lattice.value}_{cfg.Lx}x{cfg.Ly}_{pbc_tag}_{solver_mode}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = args.out_dir / f"{args.name}.h5"

    # Save to HDF5 file
    saved_path = save_to_hdf5(h5_path, cfg, terms, ref_data, energy, energy_per_site, elapsed, solver_mode)
    print(f"Saved verification dataset to: {saved_path}")

    return {
        "energy": energy,
        "energy_per_site": energy_per_site,
        "variance": variance,
        "elapsed": elapsed,
        "h5_path": str(saved_path),
        "solver": solver_mode,
        "cfg": cfg,
    }


def save_to_hdf5(
    h5_path: pathlib.Path,
    cfg: LatticeConfig,
    base_terms: list,
    ref_data: dict[str, Any],
    energy: float,
    energy_per_site: float,
    elapsed: float,
    solver_mode: str,
) -> pathlib.Path:
    """Save calculation metadata, Hamiltonian terms, and reference results into HDF5."""
    try:
        import h5py

        with h5py.File(h5_path, "w") as f:
            # Attributes / Root Metadata
            f.attrs["model"] = cfg.hamiltonian.value
            f.attrs["lattice"] = cfg.lattice.value
            f.attrs["Lx"] = int(cfg.Lx)
            f.attrs["Ly"] = int(cfg.Ly)
            f.attrs["n_sites"] = int(cfg.n_sites)
            f.attrs["dimension"] = int(cfg.dim)
            f.attrs["pbc"] = bool(cfg.pbc)
            f.attrs["j1"] = float(cfg.j1)
            f.attrs["j2"] = float(cfg.j2)
            f.attrs["hx"] = float(cfg.hx)
            f.attrs["hy"] = float(cfg.hy)
            f.attrs["hz"] = float(cfg.hz)
            f.attrs["K"] = float(cfg.K)
            f.attrs["J"] = float(cfg.J)
            f.attrs["gamma"] = float(cfg.gamma)
            f.attrs["delta"] = float(cfg.delta)
            f.attrs["solver"] = solver_mode
            f.attrs["ground_energy"] = energy
            f.attrs["ground_energy_per_site"] = energy_per_site
            f.attrs["runtime_seconds"] = elapsed
            f.attrs["timestamp"] = datetime.datetime.now().isoformat()

            # Datasets
            f.create_dataset("energy", data=np.array(energy, dtype=np.float64))
            f.create_dataset("energy_per_site", data=np.array(energy_per_site, dtype=np.float64))
            f.create_dataset("n_terms", data=np.array(len(base_terms), dtype=np.int32))

            # Store ground state vector if available (Exact ED)
            vector = ref_data.get("vector")
            if vector is not None:
                f.create_dataset("ground_state_vector", data=np.asarray(vector, dtype=np.complex128))

            # Store DMRG info if available
            if "history" in ref_data:
                dmrg_grp = f.create_group("dmrg")
                dmrg_grp.attrs["mps_bond_dim"] = int(ref_data.get("mps_bond_dim", 0))
                dmrg_grp.attrs["mpo_bond_dim"] = int(ref_data.get("mpo_bond_dim", 0))
                hist_energies = [row.get("energy", float("nan")) for row in ref_data["history"]]
                dmrg_grp.create_dataset("sweep_energies", data=np.array(hist_energies, dtype=np.float64))

        return h5_path

    except ImportError:
        # Fallback to JSON if h5py is not installed
        json_path = h5_path.with_suffix(".json")
        import json

        data = {
            "model": cfg.hamiltonian.value,
            "lattice": cfg.lattice.value,
            "Lx": cfg.Lx,
            "Ly": cfg.Ly,
            "n_sites": cfg.n_sites,
            "pbc": cfg.pbc,
            "j1": cfg.j1,
            "j2": cfg.j2,
            "solver": solver_mode,
            "ground_energy": energy,
            "ground_energy_per_site": energy_per_site,
            "runtime_seconds": elapsed,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print("Note: h5py not installed (`pip install h5py`). Saved JSON metadata instead.")
        return json_path


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    run_verification(args)


if __name__ == "__main__":
    main()
