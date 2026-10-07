"""Adaptive autoregressive NQS with shallow sampled-Ritz basis rotations."""

from .adaptive 		import run_adaptive_loop

# TODO: Fermions and orbitals are experimental and not fully supported yet - to fix!
from .fermionic		import HubbardConfig, run_fermionic_loop
from .orbitals		import orbital_matrix, orbital_pairs, orbital_ritz_objective

# -------------------------

from .config 		import (
                        AdaptiveConfig, AdaptiveStep, BasisParameterization, CircuitConfig,
                        FinalConfig, HamiltonianType, LatticeConfig, LatticeType,
                        LearningRateSchedule, ModelConfig, RotationConfig,
                        SubspaceSampling, TeacherConfig, TrainConfig, TrainOptimizer,
                    )
from .lattices 		import lattice_distance
from .operators 	import (
                        base_hamiltonian, classical_spin_vector, commutes_with_total_sz,
                        impurity_metadata, impurity_terms, 
                          ising_hamiltonian, j1j2_hamiltonian, kitaev_gamma_hamiltonian,
                    )
from .rotation_ritz import (
                        RotationProblem, build_rotated_subspace_hamiltonian, hamiltonian_j_rms,
                        optimize_rotation_angles, prepare_rotation_problem, ritz_ground_state,
                    )
from .ritz_subspace import ritz_pt2_correction
from .nqs 			import (
                        conditional_entropy, draw_configs, ed_diagnostics,
                        exact_basis_entropy_comparison, frequency_ranked_configs, gumbel_top_k_configs,
                        most_frequent_unique, subspace_probability_mass, train_nqs,
                    )

# -------------------------
from .final_solvers import solve_final
from .teacher		import fit_ritz_teacher
from .checkpoint	import load_weights, save_weights
from .netket_vmc	import exact_ground_state

__version__ = "0.1.0"
__description__ = "Adaptive autoregressive neural quantum states with sampled-Ritz basis rotations"

__all__ = [
    "__version__",
    "__description__",
    "AdaptiveConfig",
    "AdaptiveStep",
    "CircuitConfig",
    "FinalConfig",
    "LatticeConfig",
    "LatticeType",
    "HamiltonianType",
    "ModelConfig",
    "TeacherConfig",
    "RotationConfig",
    "SubspaceSampling",
    "BasisParameterization",
    "TrainConfig",
    "TrainOptimizer",
    "LearningRateSchedule",
    "base_hamiltonian",
    "ising_hamiltonian",
    "j1j2_hamiltonian",
    "kitaev_gamma_hamiltonian",
    "classical_spin_vector",
    "impurity_terms",
    "impurity_metadata",
    "lattice_distance",
    "commutes_with_total_sz",
    "build_rotated_subspace_hamiltonian",
    "prepare_rotation_problem",
    "RotationProblem",
    "hamiltonian_j_rms",
    "optimize_rotation_angles",
    "ritz_ground_state",
    "ritz_pt2_correction",
    "run_adaptive_loop",
    # TODO: Fermions and orbitals are experimental and not fully supported yet - to fix!
    "run_fermionic_loop",
    "HubbardConfig",
    "orbital_matrix",
    "orbital_pairs",
    "orbital_ritz_objective",
    "conditional_entropy",
    "draw_configs",
    "gumbel_top_k_configs",
    "frequency_ranked_configs",
    "most_frequent_unique",
    "subspace_probability_mass",
    "train_nqs",
    "fit_ritz_teacher",
    "ed_diagnostics",
    "exact_basis_entropy_comparison",
    "solve_final",
    "exact_ground_state",
    "save_weights",
    "load_weights",
]

# -------------------------
#! EOF
# -------------------------
