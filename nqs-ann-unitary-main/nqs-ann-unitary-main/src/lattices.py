r"""Lattice geometry helpers.

Honeycomb uses a two-site unit cell (sublattice A = even, B = odd) on an
``Lx x Ly`` grid of cells, with site index ``i = (Y * Lx + X) * 2 + r`` and
``r in {0, 1}``.

## Honeycomb lattice

Every site has three nearest neighbours, one of each Kitaev **bond type** ``x / y / z``; from an A site ``(X, Y)`` the three B neighbours are

	y-bond (intra-cell):  B(X,	 Y)
	x-bond:				  B(X,	 Y+1)
	z-bond:				  B(X-1, Y)

"""

from __future__ import annotations

import math

from .config import LatticeConfig, LatticeType

# x / y / z Kitaev bond types
BOND_TYPES = ("x", "y", "z")


# ------------------------
# General lattice helpers
# ------------------------


def site_index(x: int, y: int, r: int, Lx: int, Ly: int) -> int:
	"""Flat index of cell ``(x, y)`` sublattice ``r in {0 (A), 1 (B)}``."""
	return ((y % Ly) * Lx + (x % Lx)) * 2 + r


def n_sites(Lx: int, Ly: int) -> int:
	return 2 * Lx * Ly


# -------------------------


def square_site(x: int, y: int, Ly: int) -> int:
	"""Flat square-lattice site index for coordinates ``(x, y)``."""
	return x * Ly + y


def square_bonds(cfg: LatticeConfig) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
	"""Return unique square-lattice nearest- and next-nearest-neighbour bonds."""

	nn	: set[tuple[int, int]] = set()
	nnn : set[tuple[int, int]] = set()

	def add(edges: set[tuple[int, int]], a: int, b: int) -> None:
		if a != b:
			edges.add(tuple(sorted((a, b))))

	for x in range(cfg.Lx):
		for y in range(cfg.Ly):
			for edges, displacements in (
				(nn,	((1, 0), (0, +1))),
				(nnn,	((1, 1), (1, -1))),
			):
				for dx, dy in displacements:
					xp, yp = x + dx, y + dy

					if cfg.pbc:
						xp %= cfg.Lx
						yp %= cfg.Ly

					if 0 <= xp < cfg.Lx and 0 <= yp < cfg.Ly:
						add(edges, square_site(x, y, cfg.Ly), square_site(xp, yp, cfg.Ly))

	return sorted(nn), sorted(nnn)


def square_checkerboard_phase(cfg: LatticeConfig) -> tuple[float, ...]:
	"""Checkerboard phase mask used by the square-lattice Marshall sign."""

	if cfg.pbc and (cfg.Lx % 2 or cfg.Ly % 2):
		raise ValueError("The Marshall checkerboard sign requires even lattice extents with periodic boundaries")

	return tuple(float((x + y) % 2) for x in range(cfg.Lx) for y in range(cfg.Ly))


def square_checkerboard_sites(cfg: LatticeConfig) -> list[int]:
	"""Sites on the odd checkerboard sublattice."""
	return [site for site, phase in enumerate(square_checkerboard_phase(cfg)) if phase]


# -------------------------
# Honeycomb helpers
# -------------------------


def honeycomb_bonds(Lx: int, Ly: int, pbc: bool) -> list[tuple[int, int, str]]:
	r"""Nearest-neighbour bonds as ``(i, j, bond_type)`` with ``bond_type in {x,y,z}``.

	One bond per (A-site, type); ``j`` is the matched B site.  Open boundaries drop
	bonds whose neighbour leaves the grid.
	"""

	bonds: list[tuple[int, int, str]] = []
	for y in range(Ly):
		for x in range(Lx):
			a = site_index(x, y, 0, Lx, Ly)
			# (target cell, bond type) for the three A->B neighbours
			for (tx, ty), bond in (((x, y), "y"), ((x, y + 1), "x"), ((x - 1, y), "z")):
				if not pbc and not (0 <= tx < Lx and 0 <= ty < Ly):
					continue
				b = site_index(tx, ty, 1, Lx, Ly)
				bonds.append((a, b, bond))
	return bonds


def honeycomb_matchings(Lx: int, Ly: int, pbc: bool) -> list[list[tuple[int, int]]]:
	"""The three bond-type matchings ``[x_pairs, y_pairs, z_pairs]`` (disjoint each)."""

	by_type: dict[str, list[tuple[int, int]]] = {t: [] for t in BOND_TYPES}
	for i, j, t in honeycomb_bonds(Lx, Ly, pbc):
		by_type[t].append((i, j))
	return [by_type[t] for t in BOND_TYPES]


# -------------------------
# Other helpers
# -------------------------


def site_coordinate(cfg: LatticeConfig, site: int) -> tuple[float, float]:
	r"""Euclidean coordinate of site (NN bond length = 1).

	Square: ``(x, y)`` with ``site = x*Ly + y``.  Honeycomb: cell ``(x, y)`` placed at
	``x*a1 + y*a2`` (``a1=(3/2, +sqrt3/2)``, ``a2=(3/2, -sqrt3/2)``) with the B
	sublattice offset by the intra-cell bond ``(1, 0)``.
	"""
	if cfg.lattice is LatticeType.HONEYCOMB:
		cell, r = divmod(site, 2)
		x, y	= cell % cfg.Lx, cell // cfg.Lx
		xx		= 1.5 * x + 1.5 * y + (1.0 if r else 0.0)
		yy		= (math.sqrt(3.0) / 2.0) * (x - y)
		return (xx, yy)
	x, y = divmod(site, cfg.Ly)
	return (float(x), float(y))


def lattice_distance(cfg: LatticeConfig, i: int, j: int) -> float:
	r"""Euclidean distance between sites i and j (minimum image under PBC)."""

	if cfg.lattice is LatticeType.HONEYCOMB:
		ci, ri		= divmod(i, 2)
		cj, rj		= divmod(j, 2)
		# Compute the cell displacements, then the Euclidean distance in the honeycomb basis
		dcx, dcy	= (cj % cfg.Lx) - (ci % cfg.Lx), (cj // cfg.Lx) - (ci // cfg.Lx)

		# Minimum image convention for periodic boundaries
		if cfg.pbc:
			dcx -= cfg.Lx * round(dcx / cfg.Lx)
			dcy -= cfg.Ly * round(dcy / cfg.Ly)

		# Compute the Euclidean distance in the honeycomb basis
		dr	= rj - ri
		dx	= 1.5 * dcx + 1.5 * dcy + 1.0 * dr
		dy	= (math.sqrt(3.0) / 2.0) * (dcx - dcy)
		return math.hypot(dx, dy)

	# Minimum image convention for square lattice
	xi, yi = divmod(i, cfg.Ly)
	xj, yj = divmod(j, cfg.Ly)
	dx, dy = xj - xi, yj - yi
	if cfg.pbc:
		dx -= cfg.Lx * round(dx / cfg.Lx)
		dy -= cfg.Ly * round(dy / cfg.Ly)
	return math.hypot(dx, dy)


# -------------------------
# Brickwork matchings for rotation circuits
# -------------------------


def disjoint_matchings(bonds: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
	"""
	Partition bonds into disjoint matchings (no site repeats per matching).

	Example
	-------
	>>> disjoint_matchings([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5)])
	[[(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 0)]]

	It simply creates a new matching whenever a bond would overlap with an existing one.

	This is for circuit layers where we want to apply all bonds in a matching simultaneously.

	"""
	matchings	: list[list[tuple[int, int]]] = []
	used		: list[set[int]] = []
	for i, j in bonds:
		for m, u in zip(matchings, used):
			if i not in u and j not in u:
				m.append((i, j)); u.update((i, j)); break
		else:
			matchings.append([(i, j)]); used.append({i, j})
	return matchings


def nn_matchings(cfg: LatticeConfig) -> list[list[tuple[int, int]]]:
	"""
	Periodic list of disjoint nearest-neighbour matchings.

	Square:
		the four brickwork matchings (x-even, y-even, x-odd, y-odd).
	Honeycomb:
		the three Kitaev bond-type (x/y/z) perfect matchings.
	"""


	def _brickwork_pairs(cfg: LatticeConfig, axis: int, offset: int) -> list[tuple[int, int]]:
		"""Return one square-lattice disjoint nearest-neighbour matching."""

		length			= cfg.Lx if axis == 0 else cfg.Ly
		other_length	= cfg.Ly if axis == 0 else cfg.Lx
		pairs			: list[tuple[int, int]] = []

		# Loop over the "other" axis (y if axis=0, x if axis=1)
		for other in range(other_length):
			start = offset

			# Loop over the "main" axis (x if axis=0, y if axis=1) in steps of 2
			while start + 1 < length:
				end = start + 1
				if axis == 0:
					i, j = start * cfg.Ly + other, end * cfg.Ly + other
				else:
					i, j = other * cfg.Ly + start, other * cfg.Ly + end
				pairs.append((i, j))
				start += 2

			# Odd-offset matching wraps around under periodic boundaries
			if cfg.pbc and length % 2 == 0 and offset == 1 and length > 2:
				if axis == 0:
					i, j = (length - 1) * cfg.Ly + other, other
				else:
					i, j = other * cfg.Ly + length - 1, other * cfg.Ly
				pairs.append((i, j))

		return pairs

	# Return the matchings based on the lattice type
	if cfg.lattice is LatticeType.HONEYCOMB:
		return honeycomb_matchings(cfg.Lx, cfg.Ly, cfg.pbc)
	return [_brickwork_pairs(cfg, axis, off) for off in (0, 1) for axis in (0, 1)]


def circuit_matchings(cfg: LatticeConfig, include_nnn: bool = False) -> list[list[tuple[int, int]]]:
	"""All disjoint matchings the rotation circuit cycles through, one per layer."""
	matchings = nn_matchings(cfg)
	if include_nnn:
		if cfg.lattice is not LatticeType.SQUARE:
			raise ValueError("NNN rotation layers are implemented for the square lattice only")
		matchings = matchings + disjoint_matchings(square_bonds(cfg)[1])
	return matchings


def string_givens_matchings(cfg: LatticeConfig) -> list[list[tuple[int, int, int]]]:
	""" 
	Partition 3-site diagonal string Givens triples (i, k, j) into disjoint matchings. 
	
	This means we group the triples such that no site appears in more than one triple in a matching.
	"""
	if cfg.lattice is not LatticeType.SQUARE:
		raise ValueError("string_givens is supported on square lattice only")
	nnn_bonds 	= square_bonds(cfg)[1]
	triples		: list[tuple[int, int, int]] = []
	for i, j in nnn_bonds:
		xi, yi 	= divmod(i, cfg.Ly)
		xj, yj 	= divmod(j, cfg.Ly)
		k	 	= xi * cfg.Ly + yj
		triples.append((i, k, j))
	
	matchings: list[list[tuple[int, int, int]]] = []
	used: list[set[int]] = []
	for triple in triples:
		nodes = set(triple)
		for m, u in zip(matchings, used):
			if nodes.isdisjoint(u):
				m.append(triple); u.update(nodes); break
		else:
			matchings.append([triple]); used.append(set(nodes))
	return matchings


# -------------------------
#! EOF
# -------------------------
