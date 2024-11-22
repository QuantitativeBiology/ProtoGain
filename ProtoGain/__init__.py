# Importing functionalities from each module in the package

# Core dataset utilities
from ProtoGain.dataset import (
    Data,
    generate_hint,
    generate_mask,
)

# Hyperparameter management
from ProtoGain.hypers import Params

# Core model architecture
from ProtoGain.model import Network

# Metrics and output utilities
from ProtoGain.output import Metrics

# Main ProtoGAIN logic
from ProtoGain.protogain import init_arg
    # Entry point for running the main pipeline


# General utility functions
from ProtoGain.utils import (
    create_csv,
    create_dist,
    create_missing,
    create_output,
    output,
    sample_idx,
    build_protein_matrix,
)

# Define the public API for the package
__all__ = [
    # dataset.py
    "Data",
    "generate_hint",
    "generate_mask",

    # hypers.py
    "Params",

    # model.py
    "Network",

    # output.py
    "Metrics",

    # protogain.py
    "init_arg",
    "__main__",

    # utils.py
    "create_csv",
    "create_dist",
    "create_missing",
    "create_output",
    "output",
    "sample_idx",
    "build_protein_matrix",
]