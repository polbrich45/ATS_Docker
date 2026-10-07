"""Lean, scalar configuration objects for the adaptive NQS workflow."""

from __future__     import annotations

import dataclasses  as dc
from typing         import Optional, Union
from enum           import Enum


SPIN = 0.5

# ------------------------------------------------------------------------------

def _backend(backend: str | None) -> str:
    """Platform of the default JAX device unless ``backend`` is given."""
    if backend is not None:
        return backend
    import jax
    device = jax.config.jax_default_device
    return device.platform if device is not None else jax.default_backend()


def default_n_samples(backend: str | None = None) -> int:
    """Conservative minSR batch sizes: 256 on CPU, 1024 on accelerators."""
    return 256 if _backend(backend) == "cpu" else 1024


def default_n_subspace(backend: str | None = None) -> int:
    """Conservative dense Ritz sizes: 128 on CPU, 256 on accelerators."""
    return 128 if _backend(backend) == "cpu" else 256


def default_n_seed_samples(n_subspace: int, backend: str | None = None) -> int:
    """Born-draw heuristic: 32 per Ritz state, with a device-dependent floor."""
    return max(32 * n_subspace, 8 * default_n_samples(backend))

def _enum(value, enum_type: type[Enum]):
    """Normalize an enum value from a case-insensitive name or value."""
    if isinstance(value, enum_type):
        return value
    
    # Check if the value is a string and try to match it with the enum members
    if isinstance(value, str):
        key = value.strip().lower()
        for item in enum_type:
            if key in {item.value.lower(), item.name.lower()}:
                return item
    valid = ", ".join(item.value for item in enum_type)
    raise ValueError(f"Expected one of {{{valid}}}, got {value!r}")


def _finite(name: str, value: float) -> float:
    """Validate a finite scalar and return it as ``float``."""
    value = float(value)
    if not -float("inf") < value < float("inf"):
        raise ValueError(f"{name} must be finite")
    return value


def _finite_nonnegative(name: str, value: float) -> float:
    """Validate a finite non-negative scalar and return it as ``float``."""
    value = _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


# ------------------------------------------------------------------------------


class LatticeType(Enum):
    """Supported lattice geometries."""
    SQUARE          = "square"
    HONEYCOMB       = "honeycomb"


class HamiltonianType(Enum):
    """Which spin Hamiltonian to build on the lattice (which couplings are read)."""
    J1J2			= "j1j2"			# J1-J2 Heisenberg (+ transverse field g); square
    ISING			= "ising"			# Ising with transverse and longitudinal fields; square
    KITAEV_GAMMA	= "kitaev_gamma"	# Kitaev-Gamma-Heisenberg (+ field h); honeycomb


# ------------------------------------------------------------------------------


def _normalize_impurities(impurities, n_sites: int) -> tuple:
    r"""Return ``((site, strength, phi, theta), ...)`` impurity records.

    The angles, phi and theta are in units of pi. 
    
    A scalar strength points along +Z. Both mappings and sequences are accepted at this physics boundary;
    the stored representation is immutable and sorted by site.
    """
    if not impurities:
        return ()
    
    # Normalize the input into a sequence of (site, strength, phi, theta) tuples.
    items               = impurities.items() if isinstance(impurities, dict) else impurities
    out: list[tuple]    = []
    
    for entry in items:
        if isinstance(impurities, dict):
            site, spec  = entry
            spec        = (spec,) if isinstance(spec, (int, float)) else tuple(spec)
        else:
            entry       = tuple(entry)
            if len(entry) < 2:
                raise ValueError("impurity entry needs at least (site, strength)")
            site, spec  = entry[0], entry[1:]
        if len(spec) == 1:
            strength, phi, theta = float(spec[0]), 0.0, 0.0
            
        elif len(spec) == 3:
            strength, phi, theta = (float(value) for value in spec)
            
        else:
            raise ValueError("impurity spec must be strength or (strength, phi, theta)")
        
        # Validate the site and angles.
        site = int(site)
        if not 0 <= site < n_sites:
            raise ValueError(f"impurity site {site} out of range [0, {n_sites})")
        
        for name, value in (("strength", strength), ("phi", phi), ("theta", theta)):
            _finite(f"impurity {name}", value)
        out.append((site, strength, phi, theta))
    return tuple(sorted(out))


# ------------------------------------------------------------------------------


@dc.dataclass(frozen=True)
class LatticeConfig:
    r"""Lattice geometry **and** spin Hamiltonian in a single config.
 
     ``n_sites`` follows the lattice; the relevant coupling block is read by the Hamiltonian builder.

    J1-J2 (square):     H = J1 sum_<ij> S.S + J2 sum_<<ij>> S.S + g sum_i S^x_i
    Kitaev-Gamma (hc):  H = -K sum_<ij>_g S^g S^g + J sum_<ij>(S^xS^x+S^yS^y+Delta S^zS^z) + Gamma sum_<ij>_g (S^a S^b + S^b S^a) - sum_i h.S_i
    Ising (square):     H = -j1 sum_<ij> S^z_i S^z_j - sum_i (hx S^x_i + hz S^z_i)
    """

    lattice		: LatticeType		                    = LatticeType.SQUARE
    hamiltonian	: HamiltonianType	                    = HamiltonianType.J1J2
    Lx			: int  = 4
    Ly			: int  = 4
    pbc			: bool = True							# periodic boundaries?

    # J1-J2 Heisenberg couplings (square)
    j1			: float = 1.0							# NN coupling
    j2			: float = 0.5							# NNN coupling

    # Kitaev-Gamma-Heisenberg couplings (honeycomb)
    K			: float = 1.0							# Kitaev coupling (g-bond S^g S^g)
    J			: float = 0.0							# Heisenberg coupling
    gamma		: float = 0.0							# symmetric off-diagonal (Gamma) coupling

    # Heisenberg-Gamma couplings (honeycomb)
    delta		: float = 1.0							# S^z anisotropy of the Heisenberg term

    # External field: +hx sum Sx for J1-J2; -hx sum Sx - hz sum Sz for Ising;
    # -h.S for honeycomb Kitaev-Gamma. hy is honeycomb-only.
    hx			: float = 0.0
    hy			: float = 0.0
    hz			: float = 0.0

    # Classical impurities: fixed classical spins S coupled as sigma_i . S at given sites.
    impurities	: tuple = ()
    
    size        : int = None                            # Number of lattice sites (N); None -> compute from Lx, Ly, and lattice
    dimension   : int = None                            # Hilbert space dimension (2^N) for the lattice; None -> compute from Lx, Ly, and lattice

    def __post_init__(self) -> None:
        object.__setattr__(self, "lattice", _enum(self.lattice, LatticeType))
        object.__setattr__(self, "hamiltonian", _enum(self.hamiltonian, HamiltonianType))
        
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("LatticeConfig.Lx and Ly must be positive")
        
        for name in ("j1", "j2", "K", "J", "gamma", "delta", "hx", "hy", "hz"):
            _finite(f"LatticeConfig.{name}", getattr(self, name))
            
        # Normalize the impurities into a sorted tuple of (site, strength, phi, theta) records.
        object.__setattr__(self, "impurities", _normalize_impurities(self.impurities, self.n_sites))
        if self.hamiltonian is HamiltonianType.KITAEV_GAMMA and self.lattice is not LatticeType.HONEYCOMB:
            raise ValueError("KITAEV_GAMMA requires lattice=HONEYCOMB")
        
        if self.hamiltonian in {HamiltonianType.J1J2, HamiltonianType.ISING} and self.lattice is not LatticeType.SQUARE:
            raise ValueError(f"{self.hamiltonian.value} requires lattice=SQUARE")

    @property
    def n_sites(self) -> int:
        if self.size is None:
            object.__setattr__(self, "size", 2 * self.Lx * self.Ly if self.lattice is LatticeType.HONEYCOMB else self.Lx * self.Ly)
        return self.size

    @property
    def dim(self) -> int:
        """Hilbert space dimension (2^N)."""
        if self.dimension is None:
            object.__setattr__(self, "dimension", 2 ** self.n_sites)
        return self.dimension

# ------------------------------------------------------------------------------


@dc.dataclass(frozen=True)
class ModelConfig:
    """
    Autoregressive Transformer dimensions.

    Parameters
    ----------
    width : int
        Transformer hidden dimension (d)
    depth : int
        Number of Transformer blocks (N_L)
    heads : int
        Number of attention heads (must divide width)
    mlp_ratio : int
        Ratio of MLP hidden dimension to width (d_mlp = mlp_ratio * d)
    conserve_total_sz : bool
        Whether to restrict the NQS to a fixed total S^z sector (U(1) symmetry).
    n_up : int | None
        If conserving total S^z, the number of up spins in that sector (None -> half filling).
        Otherwise ignored.
    use_marshall_sign : bool
        Whether to include the Marshall sign as a fixed reference phase (default False).
    phase_head_depth : int
        Number of residual MLP blocks in the phase-only head. A nonzero value
        gives the sign/phase sector more expressivity without changing the
        autoregressive sampler probabilities.
    phase_scale : float
        Nonnegative learned-phase multiplier. Zero disables the learned phase;
        without a reference phase this gives a positive NQS and real SR.
    """
    width				: int = 64					# hidden dimension of the Transformer
    depth				: int = 3					# number of Transformer blocks
    heads				: int = 4					# number of attention heads (must divide width)
    mlp_ratio			: int = 2					# ratio of MLP hidden dimension to width
    phase_head_depth	: int = 1					# residual MLP blocks only in the phase/sign head
    phase_head_width	: int = 0					# hidden width for phase head; 0 uses width
    phase_scale			: float = 1.0				# multiplier on learned phase increments
    phase_init_std		: float = 0.05				# initial std for final phase projection
    conserve_total_sz	: bool = False				# restrict to a fixed total S^z sector (U(1) symmetry); off by default
    n_up				: Union[int, None] = None	# up spins in that sector (None -> half filling); ignored if not conserving
    use_marshall_sign	: bool = False				# whether to include the Marshall sign as a fixed reference phase (default False)

    def __post_init__(self) -> None:
        
        if self.width <= 0 or self.depth < 0 or self.heads <= 0 or self.mlp_ratio <= 0:
            raise ValueError("Model width/heads/mlp_ratio must be positive and depth non-negative")
        
        if self.width % self.heads:
            raise ValueError("Model width must be divisible by heads")
        
        if self.phase_head_depth < 0 or self.phase_head_width < 0:
            raise ValueError("phase_head_depth and phase_head_width must be non-negative")
        
        _finite_nonnegative("ModelConfig.phase_scale", self.phase_scale)
        _finite_nonnegative("ModelConfig.phase_init_std", self.phase_init_std)


# ------------------------------------------------------------------------------


def sector_n_up(model: ModelConfig, n_sites: int) -> Optional[int]:
    """Resolve the conserved number of up spins, or ``None`` for full space."""
    if not model.conserve_total_sz:
        return None
    
    if model.n_up is not None:
        if not 0 <= model.n_up <= n_sites:
            raise ValueError(f"ModelConfig.n_up must lie in [0, {n_sites}]")
        return model.n_up
    if n_sites % 2:
        raise ValueError("Odd n_sites needs an explicit ModelConfig.n_up")
    return n_sites // 2


# ------------------------------------------------------------------------------
# Basis parameterization
# ------------------------------------------------------------------------------


class BasisParameterization(Enum):
    """How angles are shared across the static circuit."""
    SHARED      = "shared"
    LAYER       = "layer"
    GATE        = "gate"


class SubspaceSampling(Enum):
    """ANN seed rule used before Hamiltonian/PT2 completion."""
    GUMBEL      = "gumbel"      # Draws unique ANN seeds without replacement.
    FREQUENCY   = "frequency"


class TrainOptimizer(Enum):
    """Optimizer for NQS parameters."""
    SGD         = "sgd"         # Stochastic gradient descent with optional SR preconditioning.
    ADAM        = "adam"        # Adam with optional SR preconditioning.


class LearningRateSchedule(Enum):
    """Supported scalar schedules for NQS learning rates and SR shifts."""
    CONSTANT    = "constant"    # Keep the initial value for all steps.
    COSINE      = "cosine"      # Cosine decay from initial to final value over the configured number of steps.


# -----------------------------------------------------------------------------
# Circuit and rotation configuration
# -----------------------------------------------------------------------------


def _normalize_family(value: str) -> str:
    """Preserve Pauli labels while normalizing named circuit families."""
    token = str(value).strip()
    if token and all(char.upper() in "XYZ" for char in token):
        return token.upper()
    return token.lower()


@dc.dataclass(frozen=True)
class CircuitConfig:
    """Static topology of the shallow basis-change circuit.

    This object is deliberately separate from angle-optimization settings. The
    topology is built once, so every adaptive step sees the same parameters and
    JAX shapes.
    """

    n_layers                : int = 1
    family                  : tuple[str, ...] = ("givens",)
    include_nnn             : bool = False
    parameterization        : BasisParameterization = BasisParameterization.GATE
    # Some additional marshall helper...
    use_marshall_clifford   : bool = False
    # Treated as small angles below this cutoff are set to zero. This is a numerical help
    theta_cutoff            : float = 1.0e-6

    def __post_init__(self) -> None:
        
        family = (self.family,) if isinstance(self.family, str) else tuple(self.family)
        family = tuple(_normalize_family(name) for name in family if str(name).strip())
        
        if self.n_layers < 0:
            raise ValueError("CircuitConfig.n_layers must be non-negative")
        
        if self.n_layers > 0 and not family:
            raise ValueError("CircuitConfig.family cannot be empty when n_layers > 0")
        
        if len(set(family)) != len(family):
            raise ValueError("CircuitConfig.family cannot contain duplicates")
        
        # Validate the parameterization enum.
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "parameterization", _enum(self.parameterization, BasisParameterization))
        _finite_nonnegative("CircuitConfig.theta_cutoff", self.theta_cutoff)


@dc.dataclass(frozen=True)
class RotationConfig:
    r"""Optimization settings for one frozen-sampler sampled-Ritz stage.

    ``gumbel`` draws unique ANN seeds without replacement. ``frequency`` draws
    exactly ``n_seed_samples`` ordinary Born samples with replacement, ranks the
    unique configurations by empirical count, and never exceeds that draw cap.
    Omitted budgets resolve from the selected device and matrix size at construction.
    Explicit counts are preserved. PT2 completion is identical in both cases.
    """

    n_rot				    : int = 200                                     # number of rotation updates per adaptive step
    n_subspace			    : Optional[int] = None                          # unique Ritz states; automatic CPU/GPU budget
    learning_rate		    : float = 5.0e-2
    grad_clip			    : float = 1.0
    
    sampling			    : SubspaceSampling = SubspaceSampling.FREQUENCY # seed sampling rule for the ANN
    n_seed_samples		    : Optional[int] = None                          # hard Born-draw cap; automatic when omitted
    
    # Additional loss weights for the sampled-Ritz distillation
    participation_weight	: float = 0.0
    entanglement_weight		: float = 0.0
    nonstoquasticity_weight	: float = 0.0
    nested_weight			: float = 0.0

    # Subspace freezing retention fraction
    f_retain				: float = 0.1
    n_basin                 : int = 0                                       # fermionic basin-hopping moves; 0 uses one local solve
    entanglement_order      : str = "spatial"                               # fermionic cuts: spatial orbitals or NQS mode order
    nqs_energy_every        : int = 0                                       # physical frozen-NQS probes; 0 keeps endpoints only

    def __post_init__(self) -> None:
        
        if self.entanglement_order not in ("spatial", "model"):
            raise ValueError("RotationConfig.entanglement_order must be spatial or model")
        if self.n_rot < 0:
            raise ValueError("RotationConfig.n_rot must be non-negative")
        if type(self.nqs_energy_every) is not int or self.nqs_energy_every < 0:
            raise ValueError("RotationConfig.nqs_energy_every must be a non-negative integer")
        if type(self.n_basin) is not int or self.n_basin < 0:
            raise ValueError("RotationConfig.n_basin must be a non-negative integer")
        if self.n_basin and not self.n_rot:
            raise ValueError("RotationConfig.n_basin requires n_rot > 0")
        
        if self.n_subspace is None:
            object.__setattr__(self, "n_subspace", default_n_subspace())
        if type(self.n_subspace) is not int or self.n_subspace < 2:
            raise ValueError("RotationConfig.n_subspace must be an integer of at least 2")
        
        # Validate the sampling enum and the seed sample count
        object.__setattr__(self, "sampling", _enum(self.sampling, SubspaceSampling))
        if self.n_seed_samples is None:
            object.__setattr__(self, "n_seed_samples", default_n_seed_samples(self.n_subspace))
        if type(self.n_seed_samples) is not int or self.n_seed_samples <= 0:
            raise ValueError("RotationConfig.n_seed_samples must be a positive integer")
        
        _finite_nonnegative("RotationConfig.learning_rate", self.learning_rate)
        _finite_nonnegative("RotationConfig.grad_clip", self.grad_clip)
        
        _finite("RotationConfig.f_retain", self.f_retain)
        if not 0.0 <= self.f_retain <= 1.0:
            raise ValueError("RotationConfig.f_retain must lie in [0.0, 1.0]")
        
        for name in ("participation_weight", "entanglement_weight", "nonstoquasticity_weight", "nested_weight"):
            _finite(f"RotationConfig.{name}", getattr(self, name))
        if not 0.0 <= self.nested_weight <= 1.0:
            raise ValueError("nested_weight must lie in [0, 1] for a convex Ritz-energy mixture")


# -----------------------------------------------------------------------------


@dc.dataclass(frozen=True)
class TrainConfig:
    r"""NQS VMC settings for one adaptive step.

    Stochastic reconfiguration uses sample-space minSR with SPRING momentum.
    CPU batches use explicit Jacobians; accelerators use on-the-fly products.
    SGD applies the SR direction without default Euclidean-norm clipping.
    """

    n_samples               : Optional[int] = None # None selects a CPU/accelerator batch
    n_nqs                   : int = 300         # number of NQS updates per adaptive step
    
    # Optimizer and learning rate schedule.
    learning_rate           : float = 2.0e-2    # initial learning rate for NQS updates
    learning_rate_final     : float = 0.0       # final learning rate for NQS updates
    learning_rate_schedule  : LearningRateSchedule = LearningRateSchedule.CONSTANT
    optimizer               : TrainOptimizer = TrainOptimizer.SGD
    grad_clip               : Optional[float] = None # None disables clipping for SR; first-order uses 1
    
    # Batch size for Jacobian products in SR mode. If None, the full sample is used.
    chunk_size              : Optional[int] = None # chunk size for Jacobian products in SR mode; None uses the full sample
    use_sr                  : bool = True
    sr_shift                : float = 1.0e-2
    sr_shift_final          : float = 1.0e-2
    sr_schedule             : LearningRateSchedule = LearningRateSchedule.CONSTANT
    sr_momentum             : float = 0.8       # SPRING fills unsampled SR directions
    
    # Early stopping: if the variance of the energy estimate does not improve by at least ``early_stop_tol`` for ``early_stop_patience`` consecutive steps
    early_stop_var          : float = 0.0
    early_stop_patience     : int = 0
    early_stop_tol          : float = 1.0e-5

    def __post_init__(self) -> None:
        
        # Set appropriate enum values for the schedule and optimizer fields.
        object.__setattr__(self, "learning_rate_schedule", _enum(self.learning_rate_schedule, LearningRateSchedule))
        object.__setattr__(self, "sr_schedule", _enum(self.sr_schedule, LearningRateSchedule))
        object.__setattr__(self, "optimizer", _enum(self.optimizer, TrainOptimizer))
        if self.n_samples is None:
            object.__setattr__(self, "n_samples", default_n_samples())
        if self.grad_clip is None:
            object.__setattr__(self, "grad_clip", 0.0 if self.use_sr else 1.0)
        
        # Validate 
        if self.n_samples <= 0 or self.n_nqs < 0:
            raise ValueError("TrainConfig needs n_samples > 0 and n_nqs >= 0")
        if self.chunk_size is None or self.chunk_size == 0:
            object.__setattr__(self, "chunk_size", min(self.n_samples, 128))
        if self.chunk_size <= 0:
            raise ValueError("TrainConfig.chunk_size must be positive")
        if self.use_sr and self.optimizer is not TrainOptimizer.SGD:
            raise ValueError("TrainConfig.use_sr=True requires optimizer='sgd'")
        if not 0.0 <= float(self.sr_momentum) < 1.0:
            raise ValueError("TrainConfig.sr_momentum must lie in [0, 1)")
        if self.early_stop_patience < 0:
            raise ValueError("TrainConfig.early_stop_patience must be non-negative")
        
        # Validate the finite non-negative scalars.
        for name in (
            "learning_rate", "learning_rate_final", "grad_clip", "sr_shift",
            "sr_shift_final", "early_stop_var", "early_stop_tol",
        ):
            _finite_nonnegative(f"TrainConfig.{name}", getattr(self, name))


# -----------------------------------------------------------------------------


@dc.dataclass(frozen=True)
class TeacherConfig:
    r"""Post-rotation sampled-Ritz distillation into the spin NQS.

    The joint Ritz probabilities provide the teacher target.

    - ``reinforce_weight`` multiplies the diagonal-energy advantage term
    - ``entropy_weight`` multiplies an optional uniform log-probability bias.
    - ``phase_weight`` encourages relative-phase agreement with the Ritz state.

    Auxiliary amplitude coefficients must be nonnegative for every target state.
    The teacher validates this target-dependent condition before fitting.
    """

    n_teacher           : int = 500         # number of teacher updates per adaptive step
    learning_rate       : float = 1.0e-3    # learning rate for teacher distillation

    grad_clip           : float = 1.0       # gradient clipping for teacher distillation

    reinforce_weight    : float = 0.0       # optional diagonal bias; target-dependent boundedness is checked
    entropy_weight      : float = 0.0       # optional uniform log-probability term; not a physical entropy
    phase_weight        : float = 0.1       # weight on phase-matching term in teacher loss (0.0 -> no phase matching)
    nqs_energy_every    : int = 0           # physical energy probes; 0 keeps endpoints only
    optimizer           : str = "adam"     # fixed-target Adam or L-BFGS with a line search

    def __post_init__(self) -> None:
        if self.optimizer not in ("adam", "lbfgs"):
            raise ValueError("TeacherConfig.optimizer must be adam or lbfgs")
        if self.optimizer == "lbfgs" and self.grad_clip != 0.0:
            raise ValueError("L-BFGS teacher requires grad_clip=0 to retain the true line-search gradient")
        if self.n_teacher < 0:
            raise ValueError("TeacherConfig.n_teacher must be non-negative")
        if type(self.nqs_energy_every) is not int or self.nqs_energy_every < 0:
            raise ValueError("TeacherConfig.nqs_energy_every must be a non-negative integer")
        for name in ("learning_rate", "grad_clip", "reinforce_weight", "entropy_weight", "phase_weight"):
            _finite_nonnegative(f"TeacherConfig.{name}", getattr(self, name))
        if self.entropy_weight >= 1.0:
            raise ValueError("TeacherConfig.entropy_weight must be less than 1")

# -----------------------------------------------------------------------------
# Final adaptive workflow configuration
# -----------------------------------------------------------------------------


@dc.dataclass(frozen=True)
class AdaptiveStep:
    """One NQS -> rotation -> teacher pass, with VMC recovery before acceptance."""

    train       : TrainConfig       = dc.field(default_factory=TrainConfig)
    rotation    : RotationConfig    = dc.field(default_factory=RotationConfig)
    teacher     : TeacherConfig     = dc.field(default_factory=TeacherConfig)
    n_recovery  : int               = 20

    def __post_init__(self) -> None:
        if self.n_recovery < 0:
            raise ValueError("AdaptiveStep.n_recovery must be non-negative")


# -----------------------------------------------------------------------------


@dc.dataclass(frozen=True)
class FinalConfig:
    """Optional final sampled-Ritz and exact/DMRG reference solves."""

    ritz_size           : int = 0           # number of sampled-Ritz configurations to retain for the final NQS distillation (0 -> skip)
    solver              : str = "none"      # "none" | "exact" | "dmrg" | "auto" for the final Hamiltonian solve
    
    # DMRG solver parameters (ignored if solver != "dmrg")
    dmrg_mps_bond_dim   : int = 64          # MPS bond dimension for the DMRG solve (ignored if solver != "dmrg")
    dmrg_mpo_max_bond   : int = 0           # Maximum MPO bond dimension for the DMRG solve (0 -> no truncation, ignored if solver != "dmrg")
    dmrg_mpo_cutoff     : float = 1.0e-10   # Cutoff for the MPO truncation (ignored if solver != "dmrg")
    dmrg_tol            : float = 1.0e-8    # Tolerance for the DMRG solve (ignored if solver != "dmrg")
    dmrg_sweeps         : int = 10          # Number of DMRG sweeps (ignored if solver != "dmrg")

    def __post_init__(self) -> None:
        if self.ritz_size == 1 or self.ritz_size < 0:
            raise ValueError("ritz_size must be 0 or at least 2")


# -----------------------------------------------------------------------------


@dc.dataclass(frozen=True)
class AdaptiveConfig:
    """Complete adaptive workflow with an explicit scalar config per step."""

    circuit         : CircuitConfig             = dc.field(default_factory=CircuitConfig)
    steps           : tuple[AdaptiveStep, ...]  = dc.field(default_factory=lambda: (AdaptiveStep(),))
    final           : FinalConfig               = dc.field(default_factory=FinalConfig)
    display         : str                       = "plain"
    restart_from    : str                       = ""
    acceptance_sigma: float                     = 2.0  # spin uncertainty tolerance; fermionic required decrease in combined SE

    def __post_init__(self) -> None:
        _finite_nonnegative("AdaptiveConfig.acceptance_sigma", self.acceptance_sigma)
        if not isinstance(self.circuit, CircuitConfig):
            raise TypeError("AdaptiveConfig.circuit must be CircuitConfig")
        if not isinstance(self.final, FinalConfig):
            raise TypeError("AdaptiveConfig.final must be FinalConfig")
        steps = tuple(self.steps)
        if not steps or any(not isinstance(step, AdaptiveStep) for step in steps):
            raise ValueError("AdaptiveConfig.steps must contain at least one AdaptiveStep")
        caps = tuple(step.rotation.n_subspace for step in steps)
        if any(current < previous for previous, current in zip(caps, caps[1:])):
            raise ValueError("AdaptiveStep n_subspace caps must be non-decreasing to retain the sampled basis")
        
        # Check the display
        display = str(self.display).strip().lower()
        if display not in {"plain", "jupyter", "silent"}:
            raise ValueError("AdaptiveConfig.display must be 'plain', 'jupyter', or 'silent'")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "display", display)

    @property
    def n_outer(self) -> int:
        """Number of explicit adaptive steps (derived, never configured)."""
        return len(self.steps)


# -------------------------
#! EOF
# -------------------------
