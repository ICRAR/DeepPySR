import os
import csv
import json
import warnings
import importlib
import inspect
import pandas as pd
import numpy as np
import sympy as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from .utils import is_redundant, ensure_output_dir, plot_n_layer_graph, plot_circlize

class DeepPySRRegressor:
    def __init__(
        self,
        max_layers=4,
        output_dir="outputs/deepPySR",
        stopping_score = 2,
        model_provider = "pysr",
        pypysr_path = None,
        pareto_lambda = 0.01,
        pareto_r2_weight = 1.0,
        **pysr_kwargs
    ):
        self.model_provider = model_provider
        self.pypysr_path = pypysr_path or os.environ.get("PYPYSR_PATH")
        self.pysr_kwargs = pysr_kwargs

        self.decimal = 2
        self.max_layers = max_layers
        self.output_dir = output_dir
        self.stopping_score = stopping_score
        self.pareto_lambda = pareto_lambda
        self.pareto_r2_weight = pareto_r2_weight
        self.relationships_ = []
        self.equations_ = None

    def get_params(self, deep=True):
        params = {
            "max_layers": self.max_layers,
            "output_dir": self.output_dir,
            "decimal": self.decimal,
            "stopping_score": self.stopping_score,
            "model_provider": self.model_provider,
            "pypysr_path": self.pypysr_path,
            "pareto_lambda": self.pareto_lambda,
            "pareto_r2_weight": self.pareto_r2_weight,
        }
        params.update(self.pysr_kwargs)
        return params


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
        
        # Filter to only use formulas for layer 1 (predict the target)
        sorted_rels = [rel for rel in self.relationships_ if rel.get('layer') == 1]
        
        if not sorted_rels:
            raise ValueError("No layer 1 formulas found in relationships_.")

        # If there are multiple relationships for the same target (due to grid search),
        # select the best one based on R2 (regression) or F1 (classification).
        # We group by target and pick the best.
        best_rels = {}
        for rel in sorted_rels:
            target = rel['target']
            if target not in best_rels:
                best_rels[target] = rel
            else:
                current_best = best_rels[target]
                # Determine if it's classification for this target
                # We can use the presence of F1 or some other heuristic
                is_classification = rel.get('f1', 0) > 0 or current_best.get('f1', 0) > 0
                
                if is_classification:
                    if rel.get('f1', 0) > current_best.get('f1', 0):
                        best_rels[target] = rel
                else:
                    if rel.get('r2', -np.inf) > current_best.get('r2', -np.inf):
                        best_rels[target] = rel
        
        sorted_rels = list(best_rels.values())
        
        # Prepare custom mappings for lambdify to avoid conflicts with numpy functions
        # e.g., 'cond' in PySR vs 'numpy.linalg.cond'
        extra_mappings = self.pysr_kwargs.get('extra_sympy_mappings', {})

        for rel in sorted_rels:
            target = rel['target']
            expr = rel['sympy']
            involved = rel['involved']
            
            # Prepare lambdified function for fast evaluation
            symbols = [sp.Symbol(s) for s in involved]
            
            # Use extra_mappings to ensure custom operators are correctly translated to numpy
            # Priority is given to extra_mappings to avoid conflicts with numpy functions
            modules = [extra_mappings, 'numpy']
            
            # If the expression contains custom functions that are mapped to SymPy Piecewise
            # (common in PySR), we might need to replace them with Piecewise before lambdifying
            # if the mapping provided is SymPy-based rather than NumPy-based.
            curr_expr = expr
            for name, mapping in extra_mappings.items():
                if hasattr(mapping, '__call__'):
                    # Try to see if it's a SymPy-returning lambda by testing with symbols
                    try:
                        test_args = [sp.Symbol(f"tmp{i}") for i in range(len(inspect.signature(mapping).parameters))]
                        test_res = mapping(*test_args)
                        if isinstance(test_res, sp.Basic):
                            # It's a SymPy-returning mapping. We can use it to replace the function in the expression.
                            f_sym = sp.Function(name)
                            curr_expr = curr_expr.replace(f_sym, mapping)
                    except Exception:
                        pass

            # Handle ComplexInfinity (zoo) and Infinity (oo) which can cause KeyError in lambdify for some printers
            curr_expr = curr_expr.subs({sp.zoo: sp.nan, sp.oo: sp.oo}) 
            # Note: sp.oo is usually handled, but zoo is not. Replacing zoo with nan is safe.

            func = sp.lambdify(symbols, curr_expr, modules=modules)
            
            # Get values for involved symbols
            args = [values[s] for s in involved]
            
            # Compute and store
            res = func(*args)
            if isinstance(res, (int, float, np.number)):
                # If it's a scalar, broadcast it to the input shape
                res = np.full(X_input.shape[0], res)
            values[target] = res
            
        if isinstance(self.target_name_, list):
            # Return multi-output predictions as a 2D array
            preds = [values[t] for t in self.target_name_]
            return np.column_stack(preds)
        else:
            return values[self.target_name_]

    def _fit_single_target(self, X, y, target_name):
        # Create a new PySRRegressor instance with the same parameters as self
        # but with a specific random_state and potentially other tweaks for sub-fitting
        params = self.get_params()

        # Remove deepPySR specific params before passing to PySRRegressor
        for p in ["max_layers", "output_dir", "decimal", "stopping_score", "relationships_", "model_provider", "pareto_lambda", "pareto_r2_weight"]:
            params.pop(p, None)

        # Configure PySR to use our output_dir for its files
        # We use a subfolder for each target to avoid collisions if running in parallel
        # though currently it's sequential.
        target_output_dir = os.path.join(self.output_dir, "pysr_outputs", target_name)
        os.makedirs(target_output_dir, exist_ok=True)
        
        params["output_directory"] = target_output_dir
        
        # Determine variable names for this fit
        n_features = X.shape[1]
        if self.model_provider == "pypysr":
            v_names = [f"v{i}" for i in range(n_features)]
        else:
            v_names = [f"x{i}" for i in range(n_features)]
            
        # Dynamically import PySRRegressor
        if self.model_provider == "pypysr":
            import sys
            pypysr_path = self.pypysr_path or os.path.expanduser("~/Projects/mypysr.jl/python")
            if os.path.exists(pypysr_path) and pypysr_path not in sys.path:
                sys.path.insert(0, pypysr_path)
            from pypysr import PySRRegressor
            if self.pysr_kwargs.get("verbosity", 0) > 0:
                try:
                    import pypysr
                    print(f"[DeepPySR] Using pypysr from: {os.path.abspath(pypysr.__file__)}")
                except ImportError:
                    print(f"[DeepPySR] Using pypysr (import path: {pypysr_path})")
            model = PySRRegressor(**self.pysr_kwargs)
        else:
            import sys
            pypysr_path = self.pypysr_path or os.path.expanduser("~/Projects/mypysr.jl/python")
            if pypysr_path in sys.path:
                sys.path.remove(pypysr_path)
            from pysr import PySRRegressor
            if self.pysr_kwargs.get("verbosity", 0) > 0:
                import pysr
                print(f"[DeepPySR] Using standard pysr from: {os.path.abspath(pysr.__file__)}")
            model = PySRRegressor(**self.pysr_kwargs)

        if target_name == 'y':
            if hasattr(y, 'values'):
                y_fit = y.values.ravel()
            else:
                y_fit = np.array(y).ravel()
        else:
            y_fit = np.array(y).ravel()
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            # Ensure variable_names is passed as a list of strings if X is not a DataFrame
            # PySRRegressor should handle it, but for pypysr we want to be explicit
            model.fit(X, y_fit, variable_names=v_names)
        
        eqs = model.equations_
        
        if eqs is None or len(eqs) == 0:
            raise ValueError(f"No equations found for target {target_name}")

        # Manually iterate through the equations and select the one with the highest Pareto Score
        # Score = R^2 * exp(-lambda * (complexity - 1))
        best_score = -np.inf
        best_idx = 0
        best_r2 = -np.inf
        
        for idx in range(len(eqs)):
            try:
                # Use model.predict to get predictions for this equation
                # Passing the index allows getting predictions for specific equations in Hall of Fame
                y_pred = model.predict(X, index=idx)
                # Handle NaNs and Infs in predictions
                y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
                r2 = r2_score(y_fit, y_pred)
                complexity = eqs.iloc[idx].get("complexity", 1)
                
                # Calculate Pareto Score
                # Using the weight as a power for r2
                # Ensure r2 is positive to avoid issues with non-integer powers
                safe_r2 = max(0.0001, r2)
                score = (safe_r2 ** self.pareto_r2_weight) * np.exp(-self.pareto_lambda * (complexity - 1))
                
                if score > best_score:
                    best_score = score
                    best_r2 = r2
                    best_idx = idx
            except Exception as e:
                print(f"Error calculating Score for equation {idx}: {e}")
                continue
            
        best = eqs.iloc[best_idx]
        formula = str(best["equation"]) if "equation" in best else str(best.get("sympy_format", ""))
        
        try:
            sym_expr = model.sympy(best_idx)
        except Exception:
            sym_expr = sp.sympify(formula)

        loss = float(best.get("loss", np.nan))
        # Use the calculated Pareto score as the score
        score = best_score
        
        return sym_expr, score, loss, int(best.get("complexity", -1)), best_r2

    def fit(self, X, y):
        if isinstance(y, pd.Series):
            self.target_name_ = str(y.name) if y.name else "y"
            y_input = y.values
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                self.target_name_ = str(y.columns[0])
                y_input = y.values.flatten()
            else:
                self.target_name_ = [str(c) for c in y.columns]
                y_input = y.values
        else:
            y_input = np.array(y)
            if y_input.ndim == 1:
                self.target_name_ = "y"
            else:
                self.target_name_ = [f"y{i}" for i in range(y_input.shape[1])]

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = [str(c) for c in X.columns.tolist()]
            X_input_all = X.values
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            X_input_all = X

        self.n_features_in_ = X_input_all.shape[1]
        ensure_output_dir(self.output_dir)
        self.relationships_ = []
        
        # Initialize queue with all targets
        queue = []
        if isinstance(self.target_name_, list):
            for i, name in enumerate(self.target_name_):
                queue.append((name, y_input[:, i], 1, None))
        else:
            queue.append((self.target_name_, y_input, 1, None))
            
        processed_targets = set()

        while queue:
            target_name, target_y, layer, parent_name = queue.pop(0)
            if layer > self.max_layers:
                continue

            # Skip if target already processed, but allow roots to be re-processed if they appear as features
            # (though in multi-output, roots shouldn't typically be features of each other initially,
            # but DeepPySR's logic might allow it later).
            is_root = (isinstance(self.target_name_, list) and target_name in self.target_name_) or \
                      (not isinstance(self.target_name_, list) and target_name == self.target_name_)
            
            if target_name in processed_targets and not is_root:
                continue
            processed_targets.add(target_name)

            print(f"--- Fitting {target_name} at layer {layer} ---")

            if is_root:
                X_fit = X_input_all
                cols = list(range(X_input_all.shape[1]))
            else:
                try:
                    idx = int(target_name[1:])
                except (ValueError, IndexError):
                    # For non-indexed targets that aren't root, we might have an issue
                    # but typically intermediate targets are x0, x1...
                    idx = -1
                
                parent_idx = None
                if parent_name and parent_name.startswith("x") and not ((isinstance(self.target_name_, list) and parent_name in self.target_name_) or (not isinstance(self.target_name_, list) and parent_name == self.target_name_)):
                    try:
                        parent_idx = int(parent_name[1:])
                    except (ValueError, IndexError):
                        pass

                cols = [j for j in range(X_input_all.shape[1]) if j != idx and j != parent_idx]
                X_fit = X_input_all[:, cols]

            try:
                # We pass the internal x0, x1... names to _fit_single_target
                # PySR will use them as variable names
                sym_expr, score, loss, complexity, r2 = self._fit_single_target(
                    X_fit, target_y, target_name, )

                # Ensure sym_expr is a sympy expression
                if not hasattr(sym_expr, "xreplace"):
                    sym_expr = sp.sympify(sym_expr)

                # Round coefficients
                for n in sym_expr.atoms(sp.Number):
                    sym_expr = sym_expr.xreplace({n: round(float(n), self.decimal)})

                # pypysr uses 'v' prefix to avoid its internal x1->x0 translation
                # pysr uses 'x' prefix.
                prefix = "v" if self.model_provider == "pypysr" else "x"
                
                mapping = {}
                for k in range(len(cols)):
                    mapping[sp.Symbol(f"{prefix}{k}")] = sp.Symbol(f"x{cols[k]}")
                
                sym_expr = sym_expr.xreplace(mapping)
                
                # Update involved variables after mapping to global x indices
                involved = sorted({str(s) for s in sym_expr.free_symbols}) if hasattr(sym_expr, "free_symbols") else []

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
                    "complexity": complexity,
                    "r2": r2
                }

                if score < self.stopping_score and not is_root:
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
            root_name = self.target_name_[0] if isinstance(self.target_name_, list) else self.target_name_
            y_rel = next((r for r in self.relationships_ if r['target'] == root_name), None)
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
                self.nout_ = len(self.target_name_) if isinstance(self.target_name_, list) else 1
                self.selection_mask_ = np.ones(self.n_features_in_, dtype=bool)

        return self

    def _get_mapped_relationships(self):
        """Returns a copy of relationships with original variable names."""
        if not hasattr(self, "feature_names_in_"):
            return self.relationships_
            
        # Create a mapping from internal names to feature names
        # Internal names are always x0, x1, ... x{n-1}
        mapping = {f"x{i}": name for i, name in enumerate(self.feature_names_in_)}
        
        # Sort internal names by length descending to avoid partial replacements (e.g., x10 replacing x1)
        internal_names = sorted(mapping.keys(), key=len, reverse=True)
        
        mapped_rels = []
        for rel in self.relationships_:
            new_rel = rel.copy()
            
            # Map target if it's an internal name like xi
            if new_rel["target"] in mapping:
                new_rel["target"] = mapping[new_rel["target"]]
            
            # Ensure sym_expr is a sympy expression
            if not hasattr(new_rel["sympy"], "xreplace"):
                new_rel["sympy"] = sp.sympify(new_rel["sympy"])

            # Map sympy formula using SymPy's xreplace with symbols
            # xreplace with symbols is safe against partial name matches.
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
        columns = ["layer", "target", "formula", "involved", "score", "r2", "loss", "complexity"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in mapped_rels:
                row = {col: r.get(col) for col in columns}
                # Format involved as a string if it's a list
                if isinstance(row["involved"], list):
                    row["involved"] = ", ".join(row["involved"])
                writer.writerow(row)



    def plot(self, filename="hierarchy.png", target_variable=None):
        if target_variable is None:
            target_variable = getattr(self, "target_name_", "y")
        feature_names = getattr(self, "feature_names_in_", None)
        mapped_rels = self._get_mapped_relationships()
        if not mapped_rels and (not feature_names):
            print("No relationships or variables to plot.")
            return
        path = os.path.join(self.output_dir, filename)
        plot_n_layer_graph(mapped_rels, path, feature_names=feature_names, target_variable=target_variable)
        print(f"Plot saved to {path}")

    def plot_circle(self, filename="circle.png", target_variable=None):
        if target_variable is None:
            target_variable = getattr(self, "target_name_", "y")
        feature_names = getattr(self, "feature_names_in_", None)
        mapped_rels = self._get_mapped_relationships()
        if not mapped_rels and (not feature_names):
            print("No relationships or variables to plot.")
            return
        path = os.path.join(self.output_dir, filename)
        
        plot_circlize(mapped_rels, path, feature_names=feature_names, target_variable=target_variable)
        print(f"Plot saved to {path}")

    def sympy(self):
        """Returns the SymPy expression(s) for the top-level relationship(s)."""
        if not self.relationships_:
            return None
            
        if isinstance(self.target_name_, list):
            exprs = {}
            for name in self.target_name_:
                rel = next((r for r in self.relationships_ if r['target'] == name), None)
                if rel:
                    expr = rel['sympy']
                    if hasattr(self, "feature_names_in_"):
                        sym_mapping = {sp.Symbol(f"x{i}"): sp.Symbol(name) for i, name in enumerate(self.feature_names_in_)}
                        expr = expr.xreplace(sym_mapping)
                    exprs[name] = expr
            return exprs
        else:
            y_rel = next((r for r in self.relationships_ if r['target'] == self.target_name_), None)
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