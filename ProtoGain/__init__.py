# Importing functionalities from each module in the package

# Core dataset utilities
from .dataset import (
    Data,
    generate_hint,
    generate_mask,
)

# Hyperparameter management
from .hypers import Params

# Core model architecture
from .model import Network

# Metrics and output utilities
from .output import (
    Metrics,
    evaluate_performance,  # Example metric evaluation function
)

# Main ProtoGAIN logic
from .protogain import (
    init_arg,
    main,  # Entry point for running the main pipeline
)

# General utility functions
from .utils import (
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
    "evaluate_performance",

    # protogain.py
    "init_arg",
    "main",

    # utils.py
    "create_csv",
    "create_dist",
    "create_missing",
    "create_output",
    "output",
    "sample_idx",
    "build_protein_matrix",
]
