import os
import csv
import json
import numpy as np
import sympy as sp
from pysr import PySRRegressor, jl
from .utils import is_redundant, ensure_output_dir, plot_n_layer_graph, plot_circlize

# 1. Initialize Julia and add the GPU backend package
jl.seval('import Pkg; Pkg.add("CUDA")')
jl.seval('using CUDA')

class DeepPySRRegressor(PySRRegressor):
    def __init__(
        self,
        max_layers=4,
        output_dir="outputs/deepPySR",
        binary_operators=None,
        unary_operators=None,
        decimal = 2,
        stopping_score = 2,
        **pysr_kwargs
    ):
        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]
        if unary_operators is None:
            unary_operators = ["sin", "cos", "exp", "log", "sqrt", "tanh", "square", "asin", "acos", "atanh","tan","atan"]
            
        # Initialize the base PySRRegressor with all other kwargs
        super().__init__(
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            **pysr_kwargs
        )
        self.decimal = decimal
        self.max_layers = max_layers
        self.output_dir = output_dir
        self.stopping_score = stopping_score
        self.relationships_ = []

    def _fit_single_target(self, X, y, target_name,seed):
        # Create a new PySRRegressor instance with the same parameters as self
        # but with a specific random_state and potentially other tweaks for sub-fitting
        params = self.get_params()
        # Remove deepPySR specific params before passing to PySRRegressor
        for p in ["max_layers", "output_dir", "decimal", "stopping_score"]:
            params.pop(p, None)
            
        params["random_state"] = seed
        # params["deterministic"] = True
        params["progress"] = False # Keep sub-processes quiet
        params["turbo"] = True
        # Configure PySR to use our output_dir for its files
        # We use a subfolder for each target to avoid collisions if running in parallel
        # though currently it's sequential.
        target_output_dir = os.path.join(self.output_dir, "pysr_outputs", target_name)
        os.makedirs(target_output_dir, exist_ok=True)
        
        params["output_directory"] = target_output_dir
        # Also use temp_equation_file to avoid leaving files in the root if not requested
        # but the user wants them in their specified path.
        # If we set output_directory, PySR should use it.
        
        # If procs is not set, default to a safe value
        if params.get("procs") is None:
            params["procs"] = max(1, (os.cpu_count() or 2) - 1)

        model = PySRRegressor(**params)
        model.fit(X, y)
        
        eqs = model.equations_
        if "score" in eqs.columns:
            best_idx = int(eqs["score"].idxmax())
        else:
            best_idx = len(eqs) - 1
            
        best = eqs.iloc[best_idx]
        formula = str(best["equation"]) if "equation" in best else str(best.get("sympy_format", ""))
        
        try:
            sym_expr = model.sympy(best_idx)
        except Exception:
            sym_expr = sp.sympify(formula)

        loss = float(best.get("loss", np.nan))
        score = float(best.get("score", np.nan)) if "score" in best else (1.0/loss if loss > 0 else 0)
        
        return sym_expr, score, loss, int(best.get("complexity", -1))

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        ensure_output_dir(self.output_dir)
        self.relationships_ = []
        queue = [("y", y, 1, None)]
        processed_targets = set()

        while queue:
            target_name, target_y, layer, parent_name = queue.pop(0)
            if layer > self.max_layers:
                continue

            if target_name in processed_targets and target_name != "y":
                continue
            processed_targets.add(target_name)

            print(f"--- Fitting {target_name} at layer {layer} ---")

            if target_name == "y":
                X_input = X
                cols = list(range(X.shape[1]))
            else:
                idx = int(target_name[1:])
                parent_idx = None
                if parent_name and parent_name.startswith("x"):
                    try:
                        parent_idx = int(parent_name[1:])
                    except ValueError:
                        pass

                cols = [j for j in range(X.shape[1]) if j != idx and j != parent_idx]
                X_input = X[:, cols]

            try:
                sym_expr, score, loss, complexity = self._fit_single_target(
                    X_input, target_y, target_name, seed=layer + len(self.relationships_))

                # Round coefficients
                for n in sym_expr.atoms(sp.Number):
                    sym_expr = sym_expr.xreplace({n: round(float(n), self.decimal)})

                if target_name != "y":
                    mapping = {sp.Symbol(f"x{k}"): sp.Symbol(f"x{cols[k]}") for k in range(len(cols))}
                    sym_expr = sym_expr.xreplace(mapping)

                involved = sorted({str(s) for s in sym_expr.free_symbols})

                if is_redundant(target_name, sym_expr, self.relationships_):
                    print(f"Relationship for {target_name} is redundant. Skipping.")
                    continue

                relationship = {
                    "target": target_name,
                    "target_symbol": sp.Symbol(target_name),
                    "layer": layer,
                    "sympy": sym_expr,
                    "formula": str(sym_expr),
                    "involved": involved,
                    "score": score,
                    "loss": loss,
                    "complexity": complexity
                }

                if score < self.stopping_score:
                    print(f"Goal reached for {target_name} (score={score:.2f} < {self.stopping_score}). Leaf node.")
                else:
                    self.relationships_.append(relationship)
                    if layer < self.max_layers:
                        for vname in involved:
                            try:
                                v_idx = int(vname[1:])
                                queue.append((vname, X[:, v_idx], layer + 1, target_name))
                            except (ValueError, IndexError):
                                continue
            except Exception as e:
                print(f"Error modeling {target_name}: {e}")

        self.save_relationships()

        return self

    def save_relationships(self, filename="relationships.csv"):
        path = os.path.join(self.output_dir, filename)
        columns = ["layer", "target", "formula", "involved", "score", "loss", "complexity"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in self.relationships_:
                row = {col: r.get(col) for col in columns}
                # Format involved as a string if it's a list
                if isinstance(row["involved"], list):
                    row["involved"] = ", ".join(row["involved"])
                writer.writerow(row)

    def plot(self, filename="hierarchy.png"):
        total_vars = getattr(self, "n_features_in_", 0)
        if not self.relationships_ and total_vars == 0:
            print("No relationships or variables to plot.")
            return
        path = os.path.join(self.output_dir, filename)
        plot_n_layer_graph(self.relationships_, path, total_vars=total_vars)
        print(f"Plot saved to {path}")

    def plot_circle(self, filename="circle.png"):
        total_vars = getattr(self, "n_features_in_", 0)
        if not self.relationships_ and total_vars == 0:
            print("No relationships or variables to plot.")
            return
        path = os.path.join(self.output_dir, filename)
        
        plot_circlize(self.relationships_, path, total_vars=total_vars)
        print(f"Plot saved to {path}")