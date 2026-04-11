import os
import sys

# Isolate the environment to avoid conflicts with user site-packages
# This is especially important for juliapkg/juliacall which scans sys.path for juliapkg.json
os.environ["PYTHONNOUSERSITE"] = "1"
# Remove user site-packages from sys.path if they are already present
sys.path = [p for p in sys.path if ".local/lib/python" not in p]

# Fix for JuliaCall/PySR segfaults: Must be set before juliacall is imported
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
# Set Julia threads before initializing Julia
# Use a conservative number of threads to avoid memory exhaustion during parallel searches
os.environ["JULIA_NUM_THREADS"] = str(min((os.cpu_count() or 2) - 1, 8))

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
