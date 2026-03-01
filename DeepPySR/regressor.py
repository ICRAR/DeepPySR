import os
import csv
import json
import warnings
import pandas as pd
import numpy as np
import sympy as sp
from pysr import PySRRegressor, jl
from sklearn.exceptions import ConvergenceWarning
from .utils import is_redundant, ensure_output_dir, plot_n_layer_graph, plot_circlize

# 1. Initialize Julia and add the GPU backend package
# jl.seval('import Pkg; Pkg.add("CUDA")')
# jl.seval('using CUDA')

class DeepPySRRegressor(PySRRegressor):
    def __init__(
        self,
        max_layers=4,
        output_dir="outputs/deepPySR",
        binary_operators=None,
        unary_operators=None,
        decimal = 2,
        stopping_score = 2,
        pysr_kwargs =None
    ):
        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]
        if unary_operators is None:
            unary_operators = ["sin", "cos", "exp", "log", "sqrt", "tanh", "square", "asin", "acos", "atanh","tan","atan"]
            
        # Initialize the base PySRRegressor with all other kwargs
        super().__init__(
            binary_operators=binary_operators,
            unary_operators=unary_operators,
        )
        self.decimal = decimal
        self.max_layers = max_layers
        self.output_dir = output_dir
        self.stopping_score = stopping_score
        self.relationships_ = []
        self.equations_ = None
        self.pysr_kwargs = pysr_kwargs or {
            "model_selection": "best",
            "binary_operators": ["+", "*"],
            "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "tanh", "square"],
            "procs": 4
        }

    def predict(self, X):
        """
        Predict y from input X using the discovered symbolic hierarchy.
        
        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            Input data of shape `(n_samples, n_features)`.
            
        Returns
        -------
        y_predicted : ndarray
            Values predicted by the symbolic hierarchy.
        """
        if not self.relationships_:
            raise ValueError("Model is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_input = X.values
        else:
            X_input = X

        # Create a dictionary to store values of intermediate variables
        # Initialize with input features
        values = {f"x{i}": X_input[:, i] for i in range(X_input.shape[1])}
        
        # Sort relationships by layer in descending order to evaluate dependencies first
        sorted_rels = sorted(self.relationships_, key=lambda x: x['layer'], reverse=True)
        
        for rel in sorted_rels:
            target = rel['target']
            expr = rel['sympy']
            involved = rel['involved']
            
            # Prepare lambdified function for fast evaluation
            symbols = [sp.Symbol(s) for s in involved]
            func = sp.lambdify(symbols, expr, modules=['numpy'])
            
            # Get values for involved symbols
            args = [values[s] for s in involved]
            
            # Compute and store
            values[target] = func(*args)
            
        return values['y']

    def _fit_single_target(self, X, y, target_name):
        # Create a new PySRRegressor instance with the same parameters as self
        # but with a specific random_state and potentially other tweaks for sub-fitting
        params = self.get_params()

        # Remove deepPySR specific params before passing to PySRRegressor
        for p in ["max_layers", "output_dir", "decimal", "stopping_score", "relationships_"]:
            params.pop(p, None)

        # Configure PySR to use our output_dir for its files
        # We use a subfolder for each target to avoid collisions if running in parallel
        # though currently it's sequential.
        target_output_dir = os.path.join(self.output_dir, "pysr_outputs", target_name)
        os.makedirs(target_output_dir, exist_ok=True)
        
        params["output_directory"] = target_output_dir
        pysr_kwargs = self.pysr_kwargs.copy()

        model = PySRRegressor(**pysr_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
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
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = [str(c) for c in X.columns.tolist()]
            X_input_all = X.values
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            X_input_all = X

        self.n_features_in_ = X_input_all.shape[1]
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
                X_fit = X_input_all
                cols = list(range(X_input_all.shape[1]))
            else:
                idx = int(target_name[1:])
                parent_idx = None
                if parent_name and parent_name.startswith("x"):
                    try:
                        parent_idx = int(parent_name[1:])
                    except ValueError:
                        pass

                cols = [j for j in range(X_input_all.shape[1]) if j != idx and j != parent_idx]
                X_fit = X_input_all[:, cols]

            try:
                # We pass the internal x0, x1... names to _fit_single_target
                # PySR will use them as variable names
                sym_expr, score, loss, complexity = self._fit_single_target(
                    X_fit, target_y, target_name, )

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
                                queue.append((vname, X_input_all[:, v_idx], layer + 1, target_name))
                            except (ValueError, IndexError):
                                continue
            except Exception as e:
                print(f"Error modeling {target_name}: {e}")

        self.save_relationships()
        
        # Populate equations_ with the top-level relationship for compatibility
        if self.relationships_:
            y_rel = next((r for r in self.relationships_ if r['target'] == 'y'), None)
            if y_rel:
                # We create a minimal DataFrame that PySR expects for equations_
                self.equations_ = pd.DataFrame([
                    {
                        "equation": y_rel["formula"],
                        "sympy_format": y_rel["sympy"],
                        "loss": y_rel["loss"],
                        "score": y_rel["score"],
                        "complexity": y_rel["complexity"]
                    }
                ])
                # These attributes are often checked by PySR or scikit-learn
                self.nout_ = 1
                self.selection_mask_ = np.ones(self.n_features_in_, dtype=bool)

        return self

    def _get_mapped_relationships(self):
        """Returns a copy of relationships with original variable names."""
        if not hasattr(self, "feature_names_in_"):
            return self.relationships_
            
        mapping = {f"x{i}": name for i, name in enumerate(self.feature_names_in_)}
        mapped_rels = []
        for rel in self.relationships_:
            new_rel = rel.copy()
            # Map target if it's xi
            if new_rel["target"].startswith("x"):
                try:
                    idx_str = new_rel["target"][1:]
                    if idx_str.isdigit():
                        idx = int(idx_str)
                        new_rel["target"] = mapping.get(new_rel["target"], new_rel["target"])
                except ValueError:
                    pass
            
            # Map sympy formula
            sym_mapping = {sp.Symbol(f"x{i}"): sp.Symbol(name) for i, name in enumerate(self.feature_names_in_)}
            new_rel["sympy"] = new_rel["sympy"].xreplace(sym_mapping)
            new_rel["formula"] = str(new_rel["sympy"])
            
            # Map involved
            new_rel["involved"] = sorted({str(s) for s in new_rel["sympy"].free_symbols})
            mapped_rels.append(new_rel)
        return mapped_rels

    def save_relationships(self, filename="relationships.csv"):
        path = os.path.join(self.output_dir, filename)
        mapped_rels = self._get_mapped_relationships()
        columns = ["layer", "target", "formula", "involved", "score", "loss", "complexity"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in mapped_rels:
                row = {col: r.get(col) for col in columns}
                # Format involved as a string if it's a list
                if isinstance(row["involved"], list):
                    row["involved"] = ", ".join(row["involved"])
                writer.writerow(row)



    def plot(self, filename="hierarchy.png"):
        feature_names = getattr(self, "feature_names_in_", None)
        mapped_rels = self._get_mapped_relationships()
        if not mapped_rels and (not feature_names):
            print("No relationships or variables to plot.")
            return
        path = os.path.join(self.output_dir, filename)
        plot_n_layer_graph(mapped_rels, path, feature_names=feature_names)
        print(f"Plot saved to {path}")

    def plot_circle(self, filename="circle.png"):
        feature_names = getattr(self, "feature_names_in_", None)
        mapped_rels = self._get_mapped_relationships()
        if not mapped_rels and (not feature_names):
            print("No relationships or variables to plot.")
            return
        path = os.path.join(self.output_dir, filename)
        
        plot_circlize(mapped_rels, path, feature_names=feature_names)
        print(f"Plot saved to {path}")

    def sympy(self):
        """Returns the SymPy expression for the top-level relationship."""
        if not self.relationships_:
            return None
        y_rel = next((r for r in self.relationships_ if r['target'] == 'y'), None)
        if not y_rel:
            return None
        
        expr = y_rel['sympy']
        if not hasattr(self, "feature_names_in_"):
            return expr
            
        # Map x0, x1... to original feature names
        mapping = {sp.Symbol(f"x{i}"): sp.Symbol(name) for i, name in enumerate(self.feature_names_in_)}
        return expr.xreplace(mapping)

    def latex(self):
        """Returns the LaTeX representation of the top-level relationship."""
        s = self.sympy()
        return sp.latex(s) if s else ""