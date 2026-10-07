"""
Spin Hamiltonians, shallow basis rotations, and local estimators.

It contains:
- Hamiltonian Pauli terms for the configured model (square J1-J2/Ising or honeycomb Kitaev-Gamma) plus classical impurities.
- Shallow Pauli-rotation circuits (brickwork NN matchings, optionally with NNN matchings) and their generators.
"""

from __future__			import annotations

from collections.abc	import Iterable
from typing				import Any

import dataclasses as dc
import math
import jax
import jax.numpy as jnp
import numpy as np


# -------------------------


from .config	import HamiltonianType, LatticeConfig, LatticeType
from .			import lattices


# ------------------------


@dc.dataclass(frozen=True)
class PauliTerm:
    """ Coefficient times a Pauli string encoded by X and Z bit masks. """
    x			: int
    z			: int
    coefficient : complex


@dc.dataclass(frozen=True)
class RotationGate:
    """One shallow-circuit gate and the index of its variational angle.

    - ``generators`` stores mutually commuting Pauli strings as ``(x_mask, z_mask, multiplier)``.

    Every public family is normalized so the complete generator has spectral norm ``1/2``; an angle therefore has the
    same physical scale across single-Pauli and composite gates.  
 
    ``support`` is the complete set of touched sites (two for local exchange, three for a
    String Givens gate).
    """

    support			: tuple[int, ...]						# sites touched by the gate
    generators		: tuple[tuple[int, int, float], ...]	# (x_mask, z_mask, multiplier) for each commuting Pauli string
    parameter		: int									# index of the variational angle in the circuit's theta vector
    layer			: int									# circuit layer index (0..depth-1)
    family			: str									# family name of the gate, e.g. "givens", "heisenberg", "xy_exchange"
    conserves_sz	: bool									# True if the gate commutes with total S^z
    real_orthogonal	: bool									# True if the gate is a real orthogonal rotation (e.g. Y, XY, YX, YZ, ZY, Givens, String Givens)


# -------------------------
# Precompute some stuff for two-site spins
# -------------------------

# The multiplication table for two-site Pauli operators in the |00>, |01>, |10>, |11> basis.
# Helps as we only need to multiply two-site Pauli strings for the exchange rotation circuit.
_MUL_TABLE = {
    (0, 0): (1.0 + 0.0j, 0), (0, 1): (1.0 + 0.0j, 1),
    (0, 2): (1.0 + 0.0j, 2), (0, 3): (1.0 + 0.0j, 3),
    (1, 0): (1.0 + 0.0j, 1), (1, 1): (1.0 + 0.0j, 0),
    (1, 2): (0.0 - 1.0j, 3), (1, 3): (0.0 + 1.0j, 2),
    (2, 0): (1.0 + 0.0j, 2), (2, 1): (0.0 + 1.0j, 3),
    (2, 2): (1.0 + 0.0j, 0), (2, 3): (0.0 - 1.0j, 1),
    (3, 0): (1.0 + 0.0j, 3), (3, 1): (0.0 - 1.0j, 2),
    (3, 2): (0.0 + 1.0j, 1), (3, 3): (1.0 + 0.0j, 0),
}


# -------------------------
# For Marshall sign: fixed gates that conjugate the Hamiltonian to stoquastic form
# -------------------------


def marshall_clifford_gates(cfg: LatticeConfig) -> list["RotationGate"]:
    """
    Single-site Z-rotation gates on sublattice B at theta=pi (Marshall sign Clifford).

    Under exp(-i pi/2 Z_j) conjugation:

    $$
        X_j -> -X_j, Y_j -> -Y_j, Z_j -> Z_j.
    $$

    For J2=0 Heisenberg this makes the rotated H stoquastic (off-diagonal <= 0).
    Clifford gates at angle=pi have zero operator growth - each Pauli string maps 1 to -1.
    """
    gates: list[RotationGate] = []
    for site in lattices.square_checkerboard_sites(cfg):
        z_term	= pauli_term({site: "Z"}, 1.0)
        gates.append(RotationGate(support=(site,), generators=((z_term.x, z_term.z, 0.5),), parameter=len(gates), layer=0, family="marshall_z", conserves_sz=True, real_orthogonal=False))

    return gates


def marshall_clifford_theta(cfg: LatticeConfig) -> tuple[list["RotationGate"], np.ndarray]:
    """Fixed Marshall Clifford gates and their ``theta=pi`` angles (empty if none apply)."""
    gates = marshall_clifford_gates(cfg)
    theta = np.full(len(gates), np.pi, dtype=np.float64) # save the angles in a numpy array for the optimizer to use
    return gates, theta


def marshall_clifford_hamiltonian(cfg: LatticeConfig, terms: list[PauliTerm]) -> list[PauliTerm]:
    """Apply the Marshall sign Clifford as preprocessing of terms...

    Conjugates ``terms`` by :func:`marshall_clifford_gates` at their fixed
    ``theta=pi``, once, before the trainable rotation circuit ever sees the
    Hamiltonian. Since each Clifford gate maps every Pauli string to exactly one
    other Pauli string (a sign flip: ``Z_j`` commutes, ``X_j``/``Y_j``
    anticommute), this has zero operator growth and adds no variational
    parameters -- it is not part of ``gates``/``theta`` at all.
    """
    gates, theta = marshall_clifford_theta(cfg)
    if not gates:
        return list(terms)
    rotated, _ = rotated_hamiltonian_terms(terms, gates, theta, cfg.n_sites)
    return rotated


# -------------------------
# Pauli terms
# -------------------------


def pauli_term(ops: dict[int, str], coefficient: complex) -> PauliTerm:
    """ Build a Pauli term from a mapping: site -> X/Y/Z. """

    x_mask = 0
    z_mask = 0
    for site, op in ops.items():
        bit = 1 << site
        if op in ("X", "Y"):
            x_mask |= bit
        if op in ("Z", "Y"):
            z_mask |= bit
        if op not in ("X", "Y", "Z"):
            raise ValueError(f"Unknown Pauli operator {op!r}")
    return PauliTerm(x_mask, z_mask, coefficient)


# -------------------------


def merge_terms(terms: Iterable[PauliTerm], minimum: float = 1.0e-12) -> list[PauliTerm]:
    """ Merge equal Pauli strings and remove relative numerical zeros. """

    if minimum < 0.0:
        raise ValueError("minimum must be non-negative")

    merged: dict[tuple[int, int], complex] = {}
    for term in terms:
        key			= (term.x, term.z)
        merged[key] = merged.get(key, 0.0j) + term.coefficient # add up coefficients for the same Pauli string
    
    scale 	= max((abs(value) for value in merged.values()), default=0.0)
    cutoff 	= minimum * scale # remove terms that are small relative to the largest coefficient
    return 	[PauliTerm(x, z, coefficient) for (x, z), coefficient in merged.items() if abs(coefficient) > cutoff]


# -------------------------
# Hamiltonian
# -------------------------


def j1j2_hamiltonian(cfg: LatticeConfig) -> list[PauliTerm]:
    """
    Construct H = J1 sum S_i.S_j + J2 sum S_i.S_j + hx sum_i S^x_i.

    Examples
    --------
    >>> from src.config		import LatticeConfig
    >>> from src.operators	import j1j2_hamiltonian
    >>> terms				= j1j2_hamiltonian(LatticeConfig(Lx=2, Ly=1, pbc=False, j1=1.0, j2=0.0))
    >>> len(terms)			# XX + YY + ZZ for the single NN bond
    3
    >>> all(abs(t.coefficient - 0.25) < 1e-12 for t in terms)
    True
    """

    nn, nnn					= lattices.square_bonds(cfg)
    terms: list[PauliTerm]	= []
    # S_i.S_j = (XX + YY + ZZ)/4 and S^x_i = sigma^x_i / 2 since S = sigma/2.
    for bonds, coupling in ((nn, cfg.j1), (nnn, cfg.j2)):
        for i, j in bonds:
            for op in ("X", "Y", "Z"):
                terms.append(pauli_term({i: op, j: op}, coupling / 4.0))
    if cfg.hx:
        for i in range(cfg.n_sites):
            terms.append(pauli_term({i: "X"}, cfg.hx / 2.0))
    return merge_terms(terms)


def ising_hamiltonian(cfg: LatticeConfig) -> list[PauliTerm]:
    """Construct Ising ``H = -j1 sum_<ij> Sz_i Sz_j - sum_i (hx Sx_i + hz Sz_i)``.

    Spin-1/2 convention: ``S = sigma/2``; ``hz=0`` gives transverse-field Ising.
    """

    nn, _					= lattices.square_bonds(cfg)
    terms: list[PauliTerm]	= []
    for i, j in nn:
        terms.append(pauli_term({i: "Z", j: "Z"}, -cfg.j1 / 4.0))

    for field, op in ((cfg.hx, "X"), (cfg.hz, "Z")):
        if field:
            for i in range(cfg.n_sites):
                terms.append(pauli_term({i: op}, -field / 2.0))
    return merge_terms(terms)


# -------------------------
# Kitaev-Gamma-Heisenberg on the honeycomb lattice
# -------------------------

_KITAEV_OP	= {"x": "X", "y": "Y", "z": "Z"}						# S^g on a g-bond
_GAMMA_PERP = {"x": ("Y", "Z"), "y": ("Z", "X"), "z": ("X", "Y")}	# (a, b) perpendicular to g


def kitaev_gamma_hamiltonian(cfg: LatticeConfig) -> list[PauliTerm]:
    r"""Kitaev-Gamma-Heisenberg Hamiltonian as Pauli strings (S = sigma / 2).

    Per nearest-neighbour bond ``(i, j)`` of Kitaev type ``g``:
      Kitaev	  -K S^g_i S^g_j							  -> -K/4 sigma^g sigma^g
      Heisenberg   J (S^x S^x + S^y S^y + Delta S^z S^z)	  ->  J/4, J/4, J*Delta/4
      Gamma		   Gamma (S^a_i S^b_j + S^b_i S^a_j)		  ->  Gamma/4 (sigma^a sigma^b + sigma^b sigma^a)
    plus a uniform field ``- sum_i h . S_i`` (``-h/2 sigma`` per site).
    """

    bonds					= lattices.honeycomb_bonds(cfg.Lx, cfg.Ly, cfg.pbc)
    terms: list[PauliTerm]	= []
    for i, j, g in bonds:
        go = _KITAEV_OP[g]
        if cfg.K:
            terms.append(pauli_term({i: go, j: go}, -cfg.K / 4.0))
        if cfg.J:
            terms.append(pauli_term({i: "X", j: "X"}, cfg.J / 4.0))
            terms.append(pauli_term({i: "Y", j: "Y"}, cfg.J / 4.0))
            terms.append(pauli_term({i: "Z", j: "Z"}, cfg.J * cfg.delta / 4.0))
        if cfg.gamma:
            a, b = _GAMMA_PERP[g]
            terms.append(pauli_term({i: a, j: b}, cfg.gamma / 4.0))
            terms.append(pauli_term({i: b, j: a}, cfg.gamma / 4.0))

    for i in range(cfg.n_sites):
        for h, op in ((cfg.hx, "X"), (cfg.hy, "Y"), (cfg.hz, "Z")):
            if h:
                terms.append(pauli_term({i: op}, -h / 2.0))
    return merge_terms(terms)


def base_hamiltonian(cfg: LatticeConfig) -> list[PauliTerm]:
    """
    Hamiltonian Pauli terms for the configured model, plus classical impurities.

    The model block (square J1-J2/Ising or honeycomb Kitaev-Gamma) is built first;
    classical-impurity terms ``sigma_i . S`` (:func:`impurity_terms`) are then merged
    in when ``cfg.impurities`` is set.
    """
    if cfg.hamiltonian is HamiltonianType.KITAEV_GAMMA:
        terms = kitaev_gamma_hamiltonian(cfg)
    elif cfg.hamiltonian is HamiltonianType.ISING:
        terms = ising_hamiltonian(cfg)
    else:
        terms = j1j2_hamiltonian(cfg)
    if cfg.impurities:
        terms = merge_terms(terms + impurity_terms(cfg))
    return terms


# -------------------------
# Classical impurities: sigma_i . S with S a fixed classical spin
# -------------------------


def classical_spin_vector(strength: float, phi: float, theta: float) -> tuple[float, float, float]:
    r"""Classical spin ``S = strength * (sin th cos ph, sin th sin ph, cos th))``.

    ``phi`` and ``theta`` are in **units of pi** (``th = pi*theta``, ``ph = pi*phi``),
    so ``(strength, 0, 0)`` points along ``+Z`` with magnitude ``strength``.

    >>> [round(c, 6) for c in classical_spin_vector(1.0, 0.0, 0.0)]
    [0.0, 0.0, 1.0]
    >>> [round(c, 6) for c in classical_spin_vector(2.0, 0.0, 0.5)]	 # +X, theta=pi/2
    [2.0, 0.0, 0.0]
    """
    th, ph	= math.pi * theta, math.pi * phi
    st		= math.sin(th)
    return (strength * st * math.cos(ph), strength * st * math.sin(ph), strength * math.cos(th))


def impurity_terms(cfg: LatticeConfig) -> list[PauliTerm]:
    r"""
    Classical-impurity Pauli terms ``sum_imp (S_x X_i + S_y Y_i + S_z Z_i)``.

    Each impurity contributes ``sigma_i . S`` at its site with the fixed classical
    spin ``S`` of :func:`classical_spin_vector` (Pauli operators, not ``S=sigma/2``).

    >>> from src.config import LatticeConfig
    >>> terms = impurity_terms(LatticeConfig(Lx=2, Ly=1, pbc=False, impurities=[(0, 1.0)]))
    >>> [(t.z, round(t.coefficient.real, 6)) for t in terms]  # pure +Z on site 0 -> Z-mask 1
    [(1, 1.0)]
    """
    terms: list[PauliTerm] = []
    for site, strength, phi, theta in cfg.impurities:
        sx, sy, sz = classical_spin_vector(strength, phi, theta)
        terms.append(pauli_term({site: "X"}, sx))
        terms.append(pauli_term({site: "Y"}, sy))
        terms.append(pauli_term({site: "Z"}, sz))
    return merge_terms(terms)


def impurity_metadata(cfg: LatticeConfig) -> dict:
    r"""Per-impurity vectors/coordinates and pairwise lattice distances + spin angles.

    Returns a JSON-friendly dict recording, for every classical impurity, its site,
    lattice coordinate, ``(strength, phi, theta)`` and classical-spin vector, and for
    every pair the lattice distance and the angle between the two classical spins.
    """
    imps	= cfg.impurities
    per		= []
    for site, strength, phi, theta in imps:
        vec					= classical_spin_vector(strength, phi, theta)
        mag					= math.sqrt(sum(c * c for c in vec))
        unit				= tuple(c / mag for c in vec) if mag > 1.0e-12 else (0.0, 0.0, 0.0)
        per.append({
            "site"			: site,
            "coordinate"	: list(lattices.site_coordinate(cfg, site)),
            "strength"		: strength,
            "phi_over_pi"	: phi,
            "theta_over_pi" : theta,
            "vector"		: list(vec),
            "unit"			: list(unit),
        })
    pairs = []
    for a in range(len(imps)):
        for b in range(a + 1, len(imps)):
            dot		= max(-1.0, min(1.0, sum(x * y for x, y in zip(per[a]["unit"], per[b]["unit"]))))
            angle	= math.acos(dot)
            pairs.append({
                "sites"				: [imps[a][0], imps[b][0]],
                "lattice_distance"	: lattices.lattice_distance(cfg, imps[a][0], imps[b][0]),
                "spin_angle_rad"	: angle,
                "spin_angle_deg"	: math.degrees(angle),
                "spin_angle_over_pi": angle / math.pi,
            })
    return {
        "n_impurities"	: len(imps),
        "lattice"		: cfg.lattice.value,
        "coupling"		: "sigma_i . S (Pauli operators, classical spin S)",
        "impurities"	: per,
        "pairs"			: pairs,
    }


# -------------------------
# Rotation circuit: generators and fixed composite families
# -------------------------


_TWO_SPIN_FAMILY: dict[str, tuple[tuple[str, str, float], ...]] = {
    # A Pauli string has norm one, hence the factor 1/2.
    "XX"			: (("X", "X", 0.5),),
    "XZ"			: (("X", "Z", 0.5),),
    "YX"			: (("Y", "X", 0.5),),
    "XY"			: (("X", "Y", 0.5),),
    "YY"			: (("Y", "Y", 0.5),),
    "YZ"			: (("Y", "Z", 0.5),),
    "ZX"			: (("Z", "X", 0.5),),
    "ZY"			: (("Z", "Y", 0.5),),
    "ZZ"			: (("Z", "Z", 0.5),),
    # The commuting composite generators are also normalized to norm 1/2.
    "givens"		: (("X", "Y", 0.25), ("Y", "X", -0.25)),
    "xy_exchange"	: (("X", "X", 0.25), ("Y", "Y", 0.25)),
    "heisenberg"	: (("X", "X", 1.0 / 6.0), ("Y", "Y", 1.0 / 6.0), ("Z", "Z", 1.0 / 6.0)),
}

_SINGLE_SPIN_FAMILY 		= frozenset({"X", "Y", "Z"})
_REAL_ORTHOGONAL_FAMILIES 	= frozenset({"Y", "YX", "XY", "YZ", "ZY", "givens", "string_givens"})
_SZ_CONSERVING_FAMILIES 	= frozenset({"Z", "ZZ", "givens", "xy_exchange", "heisenberg", "string_givens"})

# -------------------------
# Rotation circuit
# -------------------------


def pauli_rotation_circuit(cfg: LatticeConfig, depth: int, family: tuple[str, ...] = ("givens",), parameterization: str = "gate", include_nnn: bool = False) -> list[RotationGate]:
    r"""
    Build a shallow Pauli-rotation circuit

    $$
        U = prod_l prod_g exp(-i a_{l,g} G_g)
    $$
    where $l$ is the layer index, $g$ is the generator label,
    and $a_{l,g}$ is the variational angle for that layer and generator.  The
    factor convention lives in the normalized generator itself.

    The generators $G_g$ are drawn from ``family``, a subset of:
    - $X$, $Y$, or $Z$ (single-site $P_i/2$),
    - any two-site Pauli label in {"XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"},
    - the normalized composites ``givens``, ``xy_exchange``, and ``heisenberg``;
    - the opt-in three-site ``string_givens`` family on square J1-J2 diagonals.

    The local Givens generator is

    $$G_{ij}=(X_iY_j-Y_iX_j)/4,$$

    while the less-local sign-oriented variant is

    $$G_{ikj}=(X_iZ_kY_j-Y_iZ_kX_j)/4.$$

    Both conserve total $S^z$, generate real orthogonal rotations, and have
    spectral norm $1/2$.  The string family must be used on its own with
    ``include_nnn=True``; its full three-site supports are greedily colored. For
    the diagonal endpoints ``(0, 3)`` whose selected shortest path is
    ``0-1-3``, the stored support is ``(0, 1, 3)`` and the generator is
    ``(X_0 Z_1 Y_3 - Y_0 Z_1 X_3) / 4``.

    1. One-spin gates act on every site of a layer;
    2. Two-spin gates act on the layer's disjoint matching: the NN brickwork matchings
        a. When ``include_nnn`` -- the next-nearest-neighbour (diagonal / $J_2$) matchings appended to the cycle (square lattice only).

    Notes:
    * Generators with an *odd* number of $Y$ factors ($Y$, $YX$, $XY$, $YZ$, $ZY$) are
    purely imaginary antisymmetric, so their unitary is real orthogonal and
    $U^dagger H U$ stays real symmetric for a real Hamiltonian...
    * $XX$, $XZ$, $ZX$, $ZZ$, and $YY$ have an even number of $Y$ factors. They
    generate complex unitaries and yield a complex Hermitian $H_\alpha$;

    Parameters
    ----------
    cfg : LatticeConfig
        The lattice configuration.
    depth : int
        The number of layers in the rotation circuit.
    family : tuple[str, ...], optional
        The set of Pauli generators to include in the circuit. Default is
        ``("givens",)``.
    parameterization : str, optional
        The parameterization scheme for the variational angles. Must be one of "shared", "layer", or "gate". Default is "gate".
    include_nnn : bool, optional
        Whether to include next-nearest-neighbour matchings in the circuit. Default is False.

    Example
    -------
    On a frustrated square plaquette, local ``givens`` moves amplitude directly
    across one bond. ``string_givens`` moves it between diagonal endpoints while
    recording the intermediate spin through a ``Z`` string. This preserves
    total ``S^z`` but can alter a diagonal sign pattern that an endpoint-only
    nearest-neighbour gate cannot address in one layer.

    >>> diagonal_endpoints, selected_path = (0, 3), (0, 1, 3)
    >>> selected_path[0::2] == diagonal_endpoints
    True
    """

    if depth < 1: raise ValueError("Rotation depth must be at least one")
    if parameterization not in ("shared", "layer", "gate"): raise ValueError("Rotation parameterization must be shared, layer, or gate")

    # Get the normalized family labels and check for duplicates or unknowns
    family = tuple(token.upper() if token and all(char.upper() in "XYZ" for char in token) else token.lower() for token in (str(label).strip() for label in family))
    if len(set(family)) != len(family):
        raise ValueError("Rotation family cannot contain duplicates")

    # Check for unknown families and enforce string_givens rules
    unknown = set(family) - (_SINGLE_SPIN_FAMILY | {"string_givens"} | set(_TWO_SPIN_FAMILY))
    if unknown:
        raise ValueError(f"Unknown rotation generators {sorted(unknown)!r}")

    if "string_givens" in family:
        if family != ("string_givens",):
            raise ValueError("string_givens must be used as the sole rotation family")
        if not include_nnn:
            raise ValueError("string_givens requires include_nnn=True")
        if cfg.lattice is not LatticeType.SQUARE or cfg.hamiltonian is not HamiltonianType.J1J2:
            raise ValueError("string_givens is valid only for square J1-J2 lattices")

    # Combine the matchings and build the circuit layer by layer
    gates: list[RotationGate]	= []
    matchings 					= lattices.circuit_matchings(cfg, include_nnn)
    has_two_site_family 		= any(label not in _SINGLE_SPIN_FAMILY for label in family)
    if not matchings and has_two_site_family:
        return gates

    def append_gate(
        support		: tuple[int, ...],
        generators	: tuple[tuple[int, int, float], ...],
        layer		: int,
        label		: str,
    ) -> None:
        """Append one gate with the common parameter and symmetry metadata."""
        parameter = 0 if parameterization == "shared" else layer if parameterization == "layer" else len(gates)
        gates.append(RotationGate(support=support, generators=generators, parameter=parameter, layer=layer,
            family=label, conserves_sz=label in _SZ_CONSERVING_FAMILIES, real_orthogonal=label in _REAL_ORTHOGONAL_FAMILIES,
        ))

    # Build the circuit layer by layer
    for layer in range(depth):
        for label in family:
            if label not in _SINGLE_SPIN_FAMILY:
                continue

            # Single-site gates: one gate per site in this layer
            for i in range(cfg.n_sites):
                generator = pauli_term({i: label}, 1.0)
                append_gate((i,), ((generator.x, generator.z, 0.5),), layer, label)

        # Three-site gates: one gate per disjoint matching in this layer
        if family == ("string_givens",):
            sg_matchings = lattices.string_givens_matchings(cfg)
            for i, k, j in (sg_matchings[layer % len(sg_matchings)] if sg_matchings else ()):
                xzy = pauli_term({i: "X", k: "Z", j: "Y"}, 1.0)
                yzx = pauli_term({i: "Y", k: "Z", j: "X"}, 1.0)
                append_gate((i, k, j), ((xzy.x, xzy.z, 0.25), (yzx.x, yzx.z, -0.25)), layer, "string_givens")
            continue

        # Two-site gates: one gate per disjoint matching in this layer
        for i, j in (matchings[layer % len(matchings)] if matchings else ()):
            for label in family:

                # Single-site gates were already emitted above.
                if label in _SINGLE_SPIN_FAMILY:
                    continue

                # Two-site gates: emit the commuting Pauli strings for this family
                generators: list[tuple[int, int, float]] = []
                for a, b, multiplier in _TWO_SPIN_FAMILY[label]:
                    term = pauli_term({i: a, j: b}, 1.0)
                    generators.append((term.x, term.z, multiplier))
                append_gate((i, j), tuple(generators), layer, label)

    return gates


# -------------------------
# Rotate Pauli Terms for the Operator
# -------------------------


def _multiply_pauli(a_x: int, a_z: int, b_x: int, b_z: int, n_sites: int) -> tuple[complex, int, int]:
    '''
    Helper for multiplying two Pauli strings represented by their X and Z bit masks.
    Returns the phase factor and the resulting X and Z masks after multiplication.

    Examples
    --------
    >>> _multiply_pauli(0b01, 0b00, 0b10, 0b00, 2)	# X_0 * X_1: disjoint supports, phase +1
    ((1+0j), 3, 0)
    >>> _multiply_pauli(0b01, 0b00, 0b01, 0b00, 2)	# X_0 * X_0 = identity
    ((1+0j), 0, 0)
    >>> _multiply_pauli(0b01, 0b00, 0b00, 0b01, 2)	# X_0 * Z_0 = -i Y_0
    (-1j, 1, 1)
    >>> _multiply_pauli(0b01, 0b00, 0b00, 0b10, 2)	# X_0 * Z_1: disjoint, phase +1
    ((1+0j), 1, 2)
    '''

    c_x 		= a_x ^ b_x
    c_z 		= a_z ^ b_z
    m1  		= ((a_x & ~a_z & b_x & b_z) | (~a_x & a_z & b_x & ~b_z) | (a_x & a_z & ~b_x & b_z)).bit_count()
    m2  		= ((a_x & ~a_z & ~b_x & b_z) | (~a_x & a_z & b_x & b_z) | (a_x & a_z & b_x & ~b_z)).bit_count()
    diff 		= (m1 - m2) % 4
    phase_map 	= (1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j)
    return phase_map[diff], c_x, c_z


def _anticommutes(a_x: int, a_z: int, b_x: int, b_z: int) -> bool:
    ''' Check if two Pauli strings anticommute based on their X and Z bit masks. '''
    return bool(((a_x & b_z).bit_count() + (a_z & b_x).bit_count()) & 1)


def commutes_with_total_sz(terms: Iterable[PauliTerm], n_sites: int, atol: float = 1.0e-12) -> bool:
    r"""Return whether a Pauli Hamiltonian commutes with total ``S^z``.

    The check accumulates the exact Pauli expansion of
    ``[H, sum_i Z_i / 2]`` before applying ``atol`` relative to the coefficient
    one-norm. Accumulation matters: the
    ``XX`` and ``YY`` pieces of an exchange interaction do not commute
    separately, but their commutators cancel in ``XX + YY``.  This catches, for
    example, an ``X`` field or Kitaev term before a fixed-magnetization NQS is
    asked to represent a Hamiltonian that leaves its Hilbert sector.
    """

    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if atol < 0.0:
        raise ValueError("atol must be non-negative")

    commutator: dict[tuple[int, int], complex] = {}
    coefficient_scale = 0.0
    for term in terms:
        coefficient = complex(term.coefficient)
        coefficient_scale += abs(coefficient)
        for site in range(n_sites):
            # X/Y on site i anticommutes with Z_i.	Since
            # [P,Z_i]/2 = P Z_i in that case, only one Pauli product is needed.
            if not ((int(term.x) >> site) & 1):
                continue
            phase, out_x, out_z = _multiply_pauli(int(term.x), int(term.z), 0, 1 << site, n_sites)
            key 				= (out_x, out_z)
            commutator[key] 	= commutator.get(key, 0.0j) + coefficient * phase
    
    if coefficient_scale == 0.0:
        return True

    return all(abs(value) <= atol * coefficient_scale for value in commutator.values())


# -------------------------


def rotate_terms(base_terms: Iterable[PauliTerm], gates: Iterable[RotationGate], theta: jax.Array, n_sites: int) -> dict[tuple[int, int], Any]:
    r"""Compute $U(\theta)^\dagger O U(\theta)$ as a Pauli-string dictionary.

    At ``theta = 0`` the unitary is the identity *operator* -- base coefficients pass
    through unchanged and every rotation-generated key carries an exactly zero
    coefficient (``sin(0) = 0``) -- the key set is the reachable orbit, which the
    static Ritz kernels rely on.

    Nonzero angles populate those keys (operator growth).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from src.config import LatticeConfig
    >>> from src.operators import j1j2_hamiltonian, pauli_rotation_circuit, rotate_terms
    >>> cfg			= LatticeConfig(Lx=2, Ly=1, pbc=False, j2=0.0)
    >>> terms		= j1j2_hamiltonian(cfg)
    >>> gates		= pauli_rotation_circuit(cfg, depth=1, family=("YX",))
    >>> n_params	= max(g.parameter for g in gates) + 1
    >>> rotated		= rotate_terms(terms, gates, jnp.zeros(n_params), cfg.n_sites)
    >>> base		= {(t.x, t.z): complex(t.coefficient) for t in terms}
    >>> all(abs(complex(c) - base.get(k, 0.0)) < 1e-12 for k, c in rotated.items()) # theta=0 identity
    True
    >>> rotated_nz	= rotate_terms(terms, gates, jnp.ones(n_params) * 0.3, cfg.n_sites)
    >>> sum(abs(complex(c)) > 1e-12 for c in rotated_nz.values()) > len(terms)		# operator growth
    True
    """

    # Check if theta is a JAX Tracer (for JIT compilation) and set the appropriate numerical library (JAX or NumPy) accordingly.
    is_traced	= isinstance(theta, jax.core.Tracer)
    xp			= jnp if is_traced else np
    theta		= jnp.atleast_1d(theta) if is_traced else np.atleast_1d(np.asarray(theta))
    gates		= tuple(gates)
    terms		= { (term.x, term.z): (jnp.asarray(term.coefficient, dtype=complex) if is_traced else complex(term.coefficient)) for term in base_terms }

    # U^\dagger O U is conjugated in reverse list order (ket sees gates in list order -> U = G_last ... G_first)
    for gate in reversed(gates):
        gate_angle = theta[gate.parameter]

        # For each generator of the gate, compute the rotated Pauli terms by checking
        # for anticommutation and applying the appropriate phase and trigonometric factors.
        for g_x, g_z, multiplier in gate.generators:
            out		: dict[tuple[int, int], Any] = {}

            # Take the angle for the rotation as 2 * multiplier * gate_angle,
            # since the generator is scaled by multiplier.
            angle	= 2.0 * multiplier * gate_angle

            # Go through each term in the current set of terms and apply the rotation based
            # on whether it commutes or anticommutes with the generator.
            for (p_x, p_z), coefficient in terms.items():

                # Simple case, the generator commutes with the Pauli string, so we just add the term unchanged.
                if not _anticommutes(g_x, g_z, p_x, p_z):
                    out[(p_x, p_z)] = out.get((p_x, p_z), 0.0j) + coefficient
                    continue

                # Otherwise, the generator anticommutes with the Pauli string, so we need to apply the rotation formula:
                phase, q_x, q_z = _multiply_pauli(g_x, g_z, p_x, p_z, n_sites)
                out[(p_x, p_z)] = out.get((p_x, p_z), 0.0j) + coefficient * xp.cos(angle)
                out[(q_x, q_z)] = out.get((q_x, q_z), 0.0j) + coefficient * 1.0j * phase * xp.sin(angle)
            terms = out
    return terms


# Relative Hamiltonian coefficient cutoff: strings below this fraction of the
# largest rotated coefficient only inflate operator growth and are dropped.
COEFF_CUTOFF = 1.0e-10


# -------------------------


def prune_terms(terms: dict[tuple[int, int], Any], threshold: float = COEFF_CUTOFF) -> dict[tuple[int, int], Any]:
    """Drop Pauli strings below a relative coefficient threshold."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    scale = max((abs(complex(value)) for value in terms.values()), default=0.0)
    cutoff = threshold * scale
    return {key: value for key, value in terms.items() if abs(complex(value)) > cutoff}


def apply_theta_cutoff(theta: jax.Array, cutoff: float) -> jax.Array:
    """Snap rotation angles with ``|alpha| < cutoff`` to zero.

    A vanishingly small angle still spawns the full set of rotation-generated Pauli
    strings (operator growth) for no representational benefit, so snapping it to 0
    keeps the rotated Hamiltonian sparse.  ``cutoff <= 0`` disables the cut.
    """
    if cutoff <= 0.0:
        return theta
    return jnp.where(jnp.abs(theta) < cutoff, jnp.zeros_like(theta), theta)


# -------------------------
# Final concrete PauliTerm list for the rotated Hamiltonian
# -------------------------


def rotated_pauli_terms(base_terms: Iterable[PauliTerm], gates: Iterable[RotationGate], theta: jax.Array, n_sites: int) -> list[PauliTerm]:
    r"""Return $U(\theta)^\dagger O U(\theta)$ as a concrete list[PauliTerm].

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from src.config import LatticeConfig
    >>> from src.operators import j1j2_hamiltonian, pauli_rotation_circuit, rotated_pauli_terms
    >>> cfg			= LatticeConfig(Lx=2, Ly=1, pbc=False, j2=0.0)
    >>> terms		= j1j2_hamiltonian(cfg)
    >>> gates		= pauli_rotation_circuit(cfg, depth=1, family=("YX",))
    >>> n_params	= max(g.parameter for g in gates) + 1
    >>> r0			= rotated_pauli_terms(terms, gates, jnp.zeros(n_params), cfg.n_sites)
    >>> len(r0) == len(terms)	# theta=0: same terms, no operator growth
    True
    >>> rpi			= rotated_pauli_terms(terms, gates, jnp.full(n_params, 0.5), cfg.n_sites)
    >>> len(rpi) > len(terms)	# nonzero angle: new Pauli strings generated
    True
    """
    gates = tuple(gates)
    if not gates:
        return merge_terms(base_terms, minimum=COEFF_CUTOFF)
    rotated = rotate_terms(base_terms, gates, theta, n_sites)
    rotated = prune_terms(rotated, COEFF_CUTOFF)
    return [
        PauliTerm(x, z, complex(coefficient))
        for (x, z), coefficient in rotated.items()
    ]


def rotated_hamiltonian_terms(
    base_terms	: Iterable[PauliTerm],
    gates		: Iterable[RotationGate],
    theta		: jax.Array,
    n_sites		: int,
    threshold	: float = COEFF_CUTOFF,
) -> tuple[list[PauliTerm], int]:
    """Return pruned ``U(theta)^dag H U(theta)`` terms and the raw term count.

    NQS and final-solver paths use the same concrete rotated Hamiltonian.
    Keeping this as one helper avoids repeating the Heisenberg-picture rotation,
    coefficient pruning, and conversion back to concrete ``PauliTerm`` objects.
    """

    gates	= tuple(gates)
    if not gates:
        base	= tuple(base_terms)
        terms	= merge_terms(base, minimum=threshold)
        return terms, len(base)

    rotated = rotate_terms(base_terms, gates, theta, n_sites)
    raw		= len(rotated)
    pruned	= prune_terms(rotated, threshold)
    terms	= [PauliTerm(x, z, complex(coefficient)) for (x, z), coefficient in pruned.items()]
    return terms, raw


# -------------------------
#! EOF
# -------------------------
