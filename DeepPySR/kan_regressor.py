import importlib
import torch
import numpy as np
import pandas as pd
import sympy as sp
import os
import csv
from kan import KAN
from sklearn.metrics import r2_score
from .utils import ensure_output_dir, plot_n_layer_graph, plot_circlize

class KANPySRRegressor:
    """
    KANPySRRegressor combines Kolmogorov-Arnold Networks (KAN) with PySR for symbolic distillation.
    
    1. KAN Training: Shallow KAN architecture using B-splines.
    2. Univariate Extraction: Extract learned splines as 1D datasets.
    3. Symbolic Distillation via PySR: Fit extracted splines with exact mathematical expressions.
    """
    def __init__(
        self,
        kan_width=[2, 5, 1],
        kan_grid=5,
        kan_k=3,
        kan_steps=500,
        kan_lamb=0.1,
        kan_lamb_entropy=2.0,
        output_dir="outputs/kan_pysr",
        model_provider="pysr",
        pypysr_path=None,
        pareto_lambda=0.01,
        pareto_r2_weight=1.0,
        **pysr_kwargs
    ):
        self.kan_width = kan_width
        self.kan_grid = kan_grid
        self.kan_k = kan_k
        self.kan_steps = kan_steps
        self.kan_lamb = kan_lamb
        self.kan_lamb_entropy = kan_lamb_entropy
        self.output_dir = output_dir
        self.model_provider = model_provider
        self.pypysr_path = pypysr_path or os.environ.get("PYPYSR_PATH")
        self.pareto_lambda = pareto_lambda
        self.pareto_r2_weight = pareto_r2_weight
        self.pysr_kwargs = pysr_kwargs
        
        # Dynamically import PySRRegressor
        if self.model_provider == "pypysr":
            import sys
            pypysr_path = self.pypysr_path or os.path.expanduser("~/Projects/mypysr.jl/python")
            if os.path.exists(pypysr_path) and pypysr_path not in sys.path:
                sys.path.insert(0, pypysr_path)
            module = importlib.import_module("pypysr")
            if self.pysr_kwargs.get("verbosity", 0) > 0:
                print(f"[DeepPySR] KAN using pypysr from: {os.path.abspath(module.__file__)}")
        else:
            import sys
            pypysr_path = self.pypysr_path or os.path.expanduser("~/Projects/mypysr.jl/python")
            if pypysr_path in sys.path:
                sys.path.remove(pypysr_path)
            module = importlib.import_module("pysr")
            if self.pysr_kwargs.get("verbosity", 0) > 0:
                print(f"[DeepPySR] KAN using standard pysr from: {os.path.abspath(module.__file__)}")
        
        self._PySRRegressor = module.PySRRegressor
        
        self.model = None
        self.symbolic_expressions = {}
        self.relationships_ = []

    def get_params(self, deep=True):
        return {
            "kan_width": self.kan_width,
            "kan_grid": self.kan_grid,
            "kan_k": self.kan_k,
            "kan_steps": self.kan_steps,
            "kan_lamb": self.kan_lamb,
            "kan_lamb_entropy": self.kan_lamb_entropy,
            "output_dir": self.output_dir,
            "model_provider": self.model_provider,
            "pypysr_path": self.pypysr_path,
            "pareto_lambda": self.pareto_lambda,
            "pareto_r2_weight": self.pareto_r2_weight,
            **self.pysr_kwargs
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in ["kan_width", "kan_grid", "kan_k", "kan_steps", "kan_lamb", "kan_lamb_entropy", "output_dir", "model_provider", "pypysr_path"]:
                setattr(self, key, value)
            else:
                self.pysr_kwargs[key] = value
        
        # Re-initialize the internal model if provider or path changed
        if "model_provider" in params or "pypysr_path" in params:
            if self.model_provider == "pypysr":
                import sys
                pypysr_path = self.pypysr_path or os.path.expanduser("~/Projects/mypysr.jl/python")
                if os.path.exists(pypysr_path) and pypysr_path not in sys.path:
                    sys.path.insert(0, pypysr_path)
                module = importlib.import_module("pypysr")
                if self.pysr_kwargs.get("verbosity", 0) > 0:
                    print(f"[DeepPySR] KAN (re)using pypysr from: {os.path.abspath(module.__file__)}")
            else:
                import sys
                pypysr_path = self.pypysr_path or os.path.expanduser("~/Projects/mypysr.jl/python")
                if pypysr_path in sys.path:
                    sys.path.remove(pypysr_path)
                module = importlib.import_module("pysr")
                if self.pysr_kwargs.get("verbosity", 0) > 0:
                    print(f"[DeepPySR] KAN (re)using standard pysr from: {os.path.abspath(module.__file__)}")
            self._PySRRegressor = module.PySRRegressor
            
        return self

    def fit(self, X, y):
        ensure_output_dir(self.output_dir)
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = [str(c) for c in X.columns.tolist()]
            X_data = torch.from_numpy(X.values).float()
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            X_data = torch.from_numpy(X).float()

        if isinstance(y, pd.DataFrame):
            self.names_out = [str(c) for c in y.columns.tolist()]
            y_data = torch.from_numpy(y.values).float()
        elif isinstance(y, pd.Series):
            self.names_out = [str(y.name) if y.name else "y0"]
            y_data = torch.from_numpy(y.values).float().reshape(-1, 1)
        else:
            y_data = torch.from_numpy(y).float()
            if len(y_data.shape) == 1:
                y_data = y_data.reshape(-1, 1)
            self.names_out = [f"y{i}" for i in range(y_data.shape[1])]
        
        # 1. KAN Training
        self.model = KAN(width=self.kan_width, grid=self.kan_grid, k=self.kan_k)
        # Update self.kan_width because KAN might have normalized it
        self.kan_width = self.model.width
        
        dataset = {
            'train_input': X_data,
            'train_label': y_data,
            'test_input': X_data,
            'test_label': y_data
        }
        
        print("--- Training KAN ---")
        self.model.fit(dataset, opt="LBFGS", steps=self.kan_steps, lamb=self.kan_lamb, lamb_entropy=self.kan_lamb_entropy)
        
        print("--- Pruning KAN ---")
        try:
            self.model.prune()
        except Exception as e:
            print(f"Warning: KAN pruning failed: {e}")
        
        # 2. Univariate Extraction and 3. Symbolic Distillation
        print("--- Extracting Univariate Splines and Distilling with PySR ---")
        self._distill_splines(X_data, X, y)
        
        return self

    def _distill_splines(self, X_data, X_orig=None, y_orig=None):
        self.model(X_data)  # forward pass to populate model.acts
        self.symbolic_expressions = {} # (layer, node_index) -> sympy_expression
        self.relationships_ = []

        for l in range(len(self.kan_width) - 1):
            width_l = self.kan_width[l]
            width_next = self.kan_width[l+1]
            if isinstance(width_l, list): width_l = width_l[0]
            if isinstance(width_next, list): width_next = width_next[0]

            for j in range(width_next):
                node_target_name = f"L{l+1}_N{j}" if l < len(self.kan_width) - 2 else "y"
                
                # Identify involved inputs for this node j in layer l
                # self.model.act_fun[l].mask is [in_dim, out_dim]
                mask = self.model.act_fun[l].mask[:, j].detach().cpu().numpy()
                involved_indices = np.where(mask > 0)[0]
                
                if len(involved_indices) == 0:
                    print(f"Node L{l+1}_N{j} has no involved inputs. Skipping.")
                    self.symbolic_expressions[(l, j)] = sp.Float(0.0)
                    continue

                if self.model_provider == "pypysr":
                    involved_names = [f"L{l}_N{i}" if l > 0 else f"v{i}" for i in involved_indices]
                else:
                    involved_names = [f"L{l}_N{i}" if l > 0 else f"x{i}" for i in involved_indices]
                
                # Extract inputs to this node
                X_node = self.model.acts[l][:, involved_indices].detach().cpu().numpy()
                
                # Extract target for this node
                with torch.no_grad():
                    postacts = self.model.act_fun[l](self.model.acts[l])[0]
                    y_node = postacts[:, j].detach().cpu().numpy()

                print(f"Distilling node {node_target_name} with {len(involved_indices)} inputs: {involved_names}")

                pysr_model = self._PySRRegressor(**self.pysr_kwargs)
                
                # pypysr may require variable_names to be a specific type or not a PyList{Any}
                # when passed from Python via PythonCall.jl
                v_names = involved_names
                if self.model_provider == "pypysr":
                    # For pypysr (MyPySR.jl), it expects AbstractVector{String}.
                    # We have added explicit conversion to Julia Vector{String} in pypysr/sr.py,
                    # so passing a standard Python list of strings here is now safe and correct.
                    v_names = [str(n) for n in involved_names]
                
                pysr_model.fit(X_node, y_node, variable_names=v_names)

                # Manually select best based on Pareto Score
                # Score = R^2 * exp(-lambda * (complexity - 1))
                eqs = pysr_model.equations_
                best_score = -np.inf
                best_r2 = -np.inf
                best_idx = 0
                
                if eqs is not None and len(eqs) > 0:
                    for idx in range(len(eqs)):
                        try:
                            y_pred = pysr_model.predict(X_node, index=idx)
                            # Handle NaNs and Infs
                            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
                            r2 = r2_score(y_node, y_pred)
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
                        except Exception:
                            continue
                
                if best_score == -np.inf:
                    best_score = 0.0
                    best_r2 = 0.0
                    best_idx = 0

                # Get the best formula using the selected index
                best_expr = pysr_model.sympy(best_idx)

                # Ensure best_expr is a sympy expression
                if not hasattr(best_expr, "xreplace"):
                    best_expr = sp.sympify(best_expr)

                best_eq_row = eqs.iloc[best_idx] if eqs is not None else {}
                
                # Check if PySR pruned some variables that KAN thought were important
                pysr_involved = [str(s) for s in best_expr.free_symbols]
                final_involved = involved_names

                # But we can make sure 'involved' matches what's in the formula.
                if any(isinstance(best_expr, (sp.Symbol, sp.Expr)) for _ in [0]): # Just to have a block
                    final_involved = [name for name in involved_names if sp.Symbol(name) in best_expr.free_symbols]
                    if not final_involved and not best_expr.is_constant:
                        # Fallback if somehow free_symbols doesn't match involved_names
                        final_involved = involved_names

                self.symbolic_expressions[(l, j)] = best_expr

                self.relationships_.append({
                    "target": node_target_name,
                    "target_symbol": sp.Symbol(node_target_name),
                    "layer": l + 1,
                    "sympy": best_expr,
                    "formula": str(best_expr),
                    "involved": final_involved,
                    "score": best_score,
                    "r2": best_r2,
                    "loss": float(best_eq_row.get("loss", 0.0)),
                    "complexity": int(best_eq_row.get("complexity", 1))
                })

        self.save_relationships(X_orig=X_orig, y_orig=y_orig)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_input = X.values
        else:
            X_input = X
            
        current_values = X_input.T # shape (n_features, n_samples)
        
        for l in range(len(self.kan_width) - 1):
            width_l = self.kan_width[l]
            width_next = self.kan_width[l+1]
            if isinstance(width_l, list): width_l = width_l[0]
            if isinstance(width_next, list): width_next = width_next[0]
            
            next_values = np.zeros((width_next, X_input.shape[0]))
            for j in range(width_next):
                # Identify involved inputs for this node
                expr = self.symbolic_expressions[(l, j)]
                involved_names = [str(s) for s in expr.free_symbols]
                
                # Filter involved_names to only include those from this layer
                # internal names are like L{l}_N{i} or x{i}
                valid_prefix = f"L{l}_N" if l > 0 else "x"
                involved_names = [n for n in involved_names if n.startswith(valid_prefix)]
                
                if not involved_names:
                    if expr.is_constant:
                        next_values[j] = float(expr)
                    else:
                        next_values[j] = 0
                    continue
                
                # Get indices for involved names
                involved_indices = []
                for name in involved_names:
                    try:
                        idx = int(name[len(valid_prefix):])
                        involved_indices.append(idx)
                    except ValueError:
                        pass

                # Wrap the expression for evaluation
                # Prepare custom mappings for lambdify to avoid conflicts with numpy functions
                # e.g., 'cond' in PySR vs 'numpy.linalg.cond'
                extra_mappings = self.pysr_kwargs.get('extra_sympy_mappings', {})
                # Priority is given to extra_mappings to avoid conflicts with numpy functions
                modules = [extra_mappings, 'numpy']
                
                # If the expression contains custom functions that are mapped to SymPy Piecewise
                # (common in PySR), we might need to replace them with Piecewise before lambdifying
                # if the mapping provided is SymPy-based rather than NumPy-based.
                import inspect
                curr_expr = expr
                for name, mapping in extra_mappings.items():
                    if hasattr(mapping, '__call__'):
                        # Try to see if it's a SymPy-returning lambda by testing with symbols
                        try:
                            # Use inspect.signature to handle different number of arguments
                            test_args = [sp.Symbol(f"tmp{i}") for i in range(len(inspect.signature(mapping).parameters))]
                            test_res = mapping(*test_args)
                            if isinstance(test_res, sp.Basic):
                                # It's a SymPy-returning mapping. We can use it to replace the function in the expression.
                                f_sym = sp.Function(name)
                                curr_expr = curr_expr.replace(f_sym, mapping)
                        except Exception:
                            pass

                func = sp.lambdify([sp.Symbol(name) for name in involved_names], curr_expr, modules=modules)
                
                # Prepare arguments for the lambdified function
                args = [current_values[idx] for idx in involved_indices]
                next_values[j] = func(*args)
                
            current_values = next_values
            
        return current_values[0] # Assuming single output for now

    def sympy(self):
        """Returns the full symbolic expression by composing distilled splines."""
        
        # We start from input symbols
        current_exprs = [sp.Symbol(name) for name in self.feature_names_in_]
        
        for l in range(len(self.kan_width) - 1):
            width_l = self.kan_width[l]
            width_next = self.kan_width[l+1]
            if isinstance(width_l, list): width_l = width_l[0]
            if isinstance(width_next, list): width_next = width_next[0]
            
            next_exprs = []
            for j in range(width_next):
                expr = self.symbolic_expressions[(l, j)]
                
                # Identify involved inputs for this node from the expression
                involved_names = [str(s) for s in expr.free_symbols]
                valid_prefix = f"L{l}_N" if l > 0 else "x"
                involved_names = [n for n in involved_names if n.startswith(valid_prefix)]
                
                if not involved_names:
                    if expr.is_constant:
                        next_exprs.append(sp.Float(float(expr)))
                    else:
                        next_exprs.append(sp.Float(0.0))
                    continue
                
                # Substitute each involved input symbol with its expression from previous layer
                sub_map = {}
                for name in involved_names:
                    try:
                        idx_in_layer = int(name[len(valid_prefix):])
                        sub_map[sp.Symbol(name)] = current_exprs[idx_in_layer]
                    except ValueError:
                        pass
                
                node_expr = expr.subs(sub_map)
                next_exprs.append(node_expr)
                
            current_exprs = next_exprs
            
        return current_exprs[0] # Assuming single output

    def _get_mapped_relationships(self):
        """Returns a copy of relationships with original variable names."""
        if not hasattr(self, "feature_names_in_"):
            return self.relationships_
            
        # Create a mapping from internal names to feature names
        # Internal names are always x0, x1, ... x{n-1} or v0, v1, ... for pypysr
        prefix = "v" if self.model_provider == "pypysr" else "x"
        mapping = {f"{prefix}{i}": name for i, name in enumerate(self.feature_names_in_)}
        
        mapped_rels = []
        for rel in self.relationships_:
            new_rel = rel.copy()
            
            # Map target if it's an internal name like xi
            if new_rel["target"] in mapping:
                new_rel["target"] = mapping[new_rel["target"]]
            
            # Ensure sym_expr is a sympy expression
            if not hasattr(new_rel["sympy"], "subs"):
                new_rel["sympy"] = sp.sympify(new_rel["sympy"])

            # Map sympy formula
            # Note: For KAN, involved are x{i} at layer 0, and L{l}_N{i} at other layers.
            # We only want to map x{i} to original feature names.
            sym_mapping = {sp.Symbol(name): sp.Symbol(mapping[name]) for name in mapping}
            new_rel["sympy"] = new_rel["sympy"].subs(sym_mapping)
            new_rel["formula"] = str(new_rel["sympy"])
            
            # Map involved names in the list too
            new_involved = []
            for inv in new_rel["involved"]:
                if inv in mapping:
                    new_involved.append(mapping[inv])
                else:
                    new_involved.append(inv)
            new_rel["involved"] = new_involved
            
            mapped_rels.append(new_rel)
        return mapped_rels

    def save_relationships(self, filename="relationships.csv", X_orig=None, y_orig=None):
        path = os.path.join(self.output_dir, filename)
        mapped_rels = self._get_mapped_relationships()
        
        # Calculate final y in terms of raw variables
        if mapped_rels and hasattr(self, "feature_names_in_"):
            # Build expressions map
            expressions = {name: sp.Symbol(name) for name in self.feature_names_in_}
            
            # Sort by layer to ensure dependencies are resolved
            sorted_rels = sorted(mapped_rels, key=lambda x: x['layer'])
            
            for rel in sorted_rels:
                target = rel['target']
                expr = rel['sympy']
                # The rel['sympy'] in mapped_rels already has x_i replaced by feature names
                # But it might still have L1_N... etc.
                involved = [str(s) for s in expr.free_symbols]
                sub_map = {sp.Symbol(name): expressions[name] for name in involved if name in expressions}
                final_expr = expr.subs(sub_map)
                expressions[target] = final_expr
            
            if 'y' in expressions:
                final_y_expr = expressions['y']
                involved_raw = sorted([str(s) for s in final_y_expr.free_symbols if str(s) in self.feature_names_in_])
                
                final_r2 = 0.0
                if X_orig is not None and y_orig is not None:
                    try:
                        # Prepare mapping for lambdify
                        extra_mappings = self.pysr_kwargs.get('extra_sympy_mappings', {})
                        # Need to handle potential piecewise or other things
                        modules = [{'cond': lambda x, y: np.where(x > 0, y, 0)}, 'numpy']
                        
                        func = sp.lambdify([sp.Symbol(name) for name in involved_raw], final_y_expr, modules=modules)
                        
                        # Prepare data
                        if isinstance(X_orig, pd.DataFrame):
                            X_eval = [X_orig[name].values for name in involved_raw]
                        else:
                            # If it's numpy, we need to map names to indices
                            mapping = {name: i for i, name in enumerate(self.feature_names_in_)}
                            X_eval = [X_orig[:, mapping[name]] for name in involved_raw]
                            
                        y_pred = func(*X_eval)
                        final_r2 = r2_score(y_orig, y_pred)
                    except Exception as e:
                        print(f"Error calculating final R2: {e}")
                
                # Add the final y row
                mapped_rels.append({
                    "layer": sorted_rels[-1]['layer'] + 1,
                    "target": "y_final",
                    "formula": str(final_y_expr),
                    "involved": involved_raw,
                    "score": sorted_rels[-1]['score'], # Just copy last layer's score for now
                    "r2": final_r2,
                    "loss": 0,
                    "complexity": len(final_y_expr.free_symbols)
                })

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

    def plot(self, filename="kan_pysr_hierarchy.png", target_variable=None):
        if target_variable is None:
            target_variable = self.names_out[0] if hasattr(self, "names_out") and self.names_out else "y"
        feature_names = getattr(self, "feature_names_in_", None)
        mapped_rels = self._get_mapped_relationships()
        if not mapped_rels and (not feature_names):
            print("No relationships or variables to plot.")
            return
        path = os.path.join(self.output_dir, filename)
        plot_n_layer_graph(mapped_rels, path, feature_names=feature_names, target_variable=target_variable)
        print(f"Plot saved to {path}")

    def plot_circle(self, filename="kan_pysr_circle.png", target_variable=None):
        if target_variable is None:
            target_variable = self.names_out[0] if hasattr(self, "names_out") and self.names_out else "y"
        feature_names = getattr(self, "feature_names_in_", None)
        mapped_rels = self._get_mapped_relationships()
        if not mapped_rels and (not feature_names):
            print("No relationships or variables to plot.")
            return
        path = os.path.join(self.output_dir, filename)
        plot_circlize(mapped_rels, path, feature_names=feature_names, target_variable=target_variable)
        print(f"Plot saved to {path}")
