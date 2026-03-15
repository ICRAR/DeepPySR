import os

# Fix for JuliaCall/PySR segfaults: Must be set before juliacall is imported
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
# Set Julia threads before initializing Julia
os.environ["JULIA_NUM_THREADS"] = str(max(1, (os.cpu_count() or 2) - 1))

# Try to initialize JuliaCall as early as possible to avoid conflict with torch
try:
    import juliacall
except ImportError:
    pass

from .regressor import DeepPySRRegressor
from .kan_regressor import KANPySRRegressor
from .utils import mse_and_r2

__version__ = "0.1.0"
__all__ = ["DeepPySRRegressor", "KANPySRRegressor", "mse_and_r2"]
