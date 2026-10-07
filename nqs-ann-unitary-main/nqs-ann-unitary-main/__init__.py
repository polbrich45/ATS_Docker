"""Adaptive autoregressive NQS with sampled-Ritz basis rotations.

The repository implements the current outer-loop workflow:

	train NQS -> jointly optimize basis rotation and ANN sampler -> optional final polish

The importable implementation lives in :mod:`src`; this root module only carries
project metadata for tools that inspect the repository root as a package.
"""

__project__     = "nqs-ann-unitary"
__version__     = "0.1.0"
__description__ = "Adaptive autoregressive neural quantum states with sampled-Ritz basis rotations"
__author__      = "Maks Kliczkowski"

__all__         = [
	"__project__",
	"__version__",
	"__description__",
]
