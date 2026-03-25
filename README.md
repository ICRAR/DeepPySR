# DeepPySR

A deep symbolic regression package based on PySR, integrating Kolmogorov-Arnold Networks (KAN) for symbolic distillation.

## Installation

You can install DeepPySR directly from the source:

```bash
pip install .
```

## Usage

### DeepPySRRegressor

```python
from DeepPySR import DeepPySRRegressor

# Initialize the regressor
regressor = DeepPySRRegressor(
    max_layers=4,
    model_provider="pysr"  # or "pypysr"
)

# Fit the model
regressor.fit(X, y)

# Predict
y_pred = regressor.predict(X)
```

### KANPySRRegressor

```python
from DeepPySR import KANPySRRegressor

# Initialize the KAN-based regressor
regressor = KANPySRRegressor(
    kan_width=[X.shape[1], 5, 1],
    model_provider="pysr"
)

# Fit and predict
regressor.fit(X, y)
y_pred = regressor.predict(X)
```

## Julia Backend (pypysr)

If you are using the `pypysr` model provider, which calls a Julia backend (`mypysr.jl`), you can specify the path to the `pypysr` Python module:

1. Through the constructor:
   ```python
   regressor = DeepPySRRegressor(model_provider="pypysr", pypysr_path="/path/to/mypysr.jl/python")
   ```
2. Through an environment variable:
   ```bash
   export PYPYSR_PATH="/path/to/mypysr.jl/python"
   ```

By default, it looks for `~/Projects/mypysr.jl/python`.

6. Dynamic Parsimony Scaling
   Implement a feedback loop where adaptive_parsimony_scaling is adjusted based on the current R2:

Start with low parsimony to find a good fit.
As the R2 reaches a threshold (e.g.,
), increase the parsimony scaling to "squeeze" the expression into a simpler form.