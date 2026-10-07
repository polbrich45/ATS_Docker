"""Primary CLI for the ANN/NQS rotation workflow."""

import os
import tempfile

# Set MPLCONFIGDIR to a temporary directory to avoid issues with matplotlib in certain environments
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "nqs-ann-unitary-mpl"))

from src.workflow import main

# ----------------------------------------------------------------------------

if __name__ == "__main__":
	main()

# ----------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------
