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
        pareto_lambda = 0.001,
        pareto_r2_weight = 1.0,
        batch_size = None,
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
        self.batch_size = batch_size
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
            "batch_size": self.batch_size,
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
            X_input = X.values.astype(np.float64)
        else:
            X_input = np.asarray(X).astype(np.float64)

        # Create a dictionary to store values of intermediate variables
        # Initialize with input features
        values = {f"x{i}": X_input[:, i] for i in range(X_input.shape[1])}
        
        # Sort relationships by layer in descending order to evaluate dependencies first
        sorted_rels = sorted(self.relationships_, key=lambda x: x['layer'], reverse=True)
        
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
            # We map custom functions to SymPy Piecewise first, before lambdifying.
            curr_expr = expr
            
            # Identify and replace custom functions in extra_mappings
            # We map custom functions to SymPy expressions first, before lambdifying.
            curr_expr = expr
            
            # Map of name to function/lambda for lambdify
            custom_funcs = {}
            for name, mapping in extra_mappings.items():
                if hasattr(mapping, '__call__'):
                    f_sym = sp.Function(name)
                    if curr_expr.has(f_sym):
                        # Try to replace calls to this function with the result of calling the mapping
                        # We use SymPy's replace method which can handle function replacements
                        def make_replacer(m, fs):
                            def replacement(*args):
                                try:
                                    res = m(*args)
                                    if isinstance(res, sp.Basic):
                                        return res
                                except Exception:
                                    pass
                                return fs(*args) # Fallback to original
                            return replacement
                        
                        # We call it twice to handle nested custom functions if any
                        curr_expr = curr_expr.replace(f_sym, make_replacer(mapping, f_sym))
                        
                        # If after replacement it still exists, it means mapping didn't handle it
                        # as a SymPy expression, so we pass it to lambdify directly.
                        if curr_expr.has(f_sym):
                            custom_funcs[name] = mapping
                else:
                    # Not a callable, likely a direct mapping string/dict for lambdify
                    custom_funcs[name] = mapping

            # A special case: SymPy's ITE (If-Then-Else) or Piecewise 
            # can't handle Piecewise in the condition. We use piecewise_fold
            # to lift nested Piecewise and merge them into a single top-level Piecewise.
            try:
                curr_expr = sp.piecewise_fold(curr_expr)
            except Exception:
                # Fallback if piecewise_fold fails for some reason
                pass

            # Actually, even after piecewise_fold, we might have ITE(piecewise, ...)
            # if piecewise_fold didn't resolve everything.
            def merge_ite_piecewise(e):
                if isinstance(e, sp.ITE) and isinstance(e.args[0], sp.Piecewise):
                    cond_piecewise = e.args[0]
                    true_val = e.args[1]
                    false_val = e.args[2]
                    
                    new_args = []
                    for val, cond in cond_piecewise.args:
                        # If cond is true, result is ITE(val, true_val, false_val)
                        new_val = sp.ITE(val, true_val, false_val)
                        new_args.append((new_val, cond))
                    return sp.Piecewise(*new_args)
                
                if hasattr(e, 'args') and e.args:
                    new_args = [merge_ite_piecewise(arg) for arg in e.args]
                    if new_args != list(e.args):
                        return e.func(*new_args)
                return e

            curr_expr = merge_ite_piecewise(curr_expr)

            # Ensure we use a safe select that handles non-boolean conditions in Piecewise
            def safe_select(condlist, choicelist, default=np.nan):
                safe_condlist = []
                for c in condlist:
                    if isinstance(c, np.ndarray) and c.dtype != bool:
                        try:
                            # Try to convert to boolean
                            safe_condlist.append(c.astype(bool))
                        except Exception:
                            # If conversion fails, keep original and let np.select handle it
                            safe_condlist.append(c)
                    else:
                        safe_condlist.append(c)
                return np.select(safe_condlist, choicelist, default=default)

            # We pass both extra_mappings (for direct function names) 
            # and a dictionary of custom_funcs (mappings) to modules.
            # We also ensure 'select' is mapped to our safe_select.
            modules = [custom_funcs, extra_mappings, {'select': safe_select}, 'numpy']
            
            # Handle ComplexInfinity (zoo) and Infinity (oo) which can cause KeyError in lambdify for some printers
            curr_expr = curr_expr.subs({sp.zoo: sp.nan, sp.oo: sp.oo}) 
            # Note: sp.oo is usually handled, but zoo is not. Replacing zoo with nan is safe.

            func = sp.lambdify(symbols, curr_expr, modules=modules)
            
            # Get values for involved symbols
            args = [values[s] for s in involved]
            
            # Compute and store
            with np.errstate(all='ignore'):
                res = func(*args)
            if isinstance(res, (int, float, np.number)):
                # If it's a scalar, broadcast it to the input shape
                res = np.full(X_input.shape[0], res)
            values[target] = res
            
        if isinstance(self.target_name_, list):
            # Return multi-output predictions as a 2D array
            preds = [values[t] for t in self.target_name_]
            res = np.column_stack(preds)
        else:
            res = values[self.target_name_]
            
        # Robustly handle NaNs and Infs
        return np.nan_to_num(res, nan=0.0, posinf=1e10, neginf=-1e10)

    def _fit_single_target(self, X, y, target_name):
        # Create a new PySRRegressor instance with the same parameters as self
        # but with a specific random_state and potentially other tweaks for sub-fitting
        params = self.get_params()

        # Remove deepPySR specific params before passing to PySRRegressor
        for p in [
            "max_layers", "output_dir", "decimal", "stopping_score", "relationships_", 
            "model_provider", "pareto_lambda", "pareto_r2_weight", "pypysr_path", "batch_size",
            "variable_prune_max", "variable_prune_start", "variable_prune_ramp"
        ]:
            if self.model_provider != "pypysr" and p in ["variable_prune_max", "variable_prune_start", "variable_prune_ramp"]:
                params.pop(p, None)
            elif p in ["max_layers", "output_dir", "decimal", "stopping_score", "relationships_", "model_provider", "pareto_lambda", "pareto_r2_weight", "pypysr_path", "batch_size"]:
                params.pop(p, None)

        # Apply conservative defaults for stability if not provided
        if "procs" not in params:
            # Use at most 4 workers to avoid Distributed.ProcessExitedException (memory exhaustion)
            # Standard PySR 1.x default is often too aggressive for many-layered DeepPySR
            params["procs"] = min((os.cpu_count() or 2) - 1, 4)


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
        import sys

        # 2. Clear sys.modules to force re-import when switching providers
        # This is necessary because both providers might share sub-module names
        # or we want to ensure we're getting the one from the current sys.path.
        if "pysr" in sys.modules or "pypysr" in sys.modules:
            for mod in list(sys.modules.keys()):
                if mod == 'pysr' or mod.startswith('pysr.') or mod == 'pypysr' or mod.startswith('pypysr.'):
                    del sys.modules[mod]

        # 2.1 Handle Julia environment switching
        # pypysr activates its own internal Julia environment, which can break standard pysr.
        # If we are switching back to pysr, we might need to reactivate the default environment.
        if "juliacall" in sys.modules:
            try:
                from juliacall import Main as jl
                jl.seval("using Pkg")
                if self.model_provider == "pypysr":
                    # pypysr's fit() will handle its own Pkg.activate()
                    pass
                else:
                    # If we are in pysr mode, ensure we are NOT in pypysr's environment.
                    # Standard pysr expects the environment in .venv/julia_env or similar.
                    # We try to activate the default project.
                    curr_proj = jl.Pkg.project().path
                    if "pypysr" in curr_proj:
                        # Find the .venv path if possible, or just use the default env.
                        # Usually, activating "" or "@v1.x" works, but let's be more specific.
                        # Pkg.activate() with no args activates the default project.
                        jl.Pkg.activate()
                        if params.get("verbosity", 0) > 0:
                            print(f"[DeepPySR] Reactivated default Julia environment (was {curr_proj})")
            except Exception as e:
                if params.get("verbosity", 0) > 0:
                    print(f"[DeepPySR] Warning: Failed to manage Julia environment: {e}")

        # 3. Import the requested provider
        if self.model_provider == "pypysr":
            from pypysr import PySRRegressor
            if params.get("verbosity", 0) > 0:
                try:
                    import pypysr
                    print(f"[DeepPySR] Using pypysr from: {os.path.abspath(pypysr.__file__)}")
                except ImportError:
                    print(f"[DeepPySR] Using pypysr")
        else:
            from pysr import PySRRegressor
            if params.get("verbosity", 0) > 0:
                try:
                    import pysr
                    print(f"[DeepPySR] Using standard pysr from: {os.path.abspath(pysr.__file__)}")
                except ImportError:
                    print(f"[DeepPySR] Using standard pysr")
        
        model = PySRRegressor(**params)

        if X.shape[0] > 10000:
            if params.get("verbosity", 0) > 0:
                print(f"[DeepPySR] Data samples > 10000 ({X.shape[0]}). Enabling batching.")
            model.set_params(batching=True)
            if self.batch_size is not None:
                model.set_params(batch_size=self.batch_size)
            elif "batch_size" not in params:
                model.set_params(batch_size=500) # Default PySR is 50, but maybe 500 is better for >10000
        elif self.batch_size is not None:
            model.set_params(batching=True)
            model.set_params(batch_size=self.batch_size)

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

        # Prepare parameters for grid search
        r2w_list = self.pareto_r2_weight if isinstance(self.pareto_r2_weight, (list, np.ndarray)) else [self.pareto_r2_weight]
        lambda_list = self.pareto_lambda if isinstance(self.pareto_lambda, (list, np.ndarray)) else [self.pareto_lambda]

        all_results = []
        
        # Dictionary to cache predictions to avoid redundant model.predict calls
        prediction_cache = {}

        for r2w in r2w_list:
            for lam in lambda_list:
                # Manually iterate through the equations and select the one with the highest Pareto Score
                # Score = R^2 * exp(-lambda * (complexity - 1))
                best_score = -np.inf
                best_idx = 0
                best_r2 = -np.inf
                
                for idx in range(len(eqs)):
                    try:
                        # Use model.predict to get predictions for this equation
                        if idx not in prediction_cache:
                            y_pred = model.predict(X, index=idx)
                            # Handle NaNs and Infs in predictions
                            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
                            
                            # If targets are integer-like, we might be in classification-via-regression.
                            # Rounding here could significantly improve R2/Score for classification.
                            # But we only do it if the target values are actually integers and non-binary.
                            # For binary targets, we don't round as standard R2 or loss should work fine,
                            # but for multiclass encoded as 0, 1, 2, rounding is often better.
                            unique_fit = np.unique(y_fit)
                            if len(unique_fit) > 2 and np.all(np.equal(np.mod(y_fit, 1), 0)):
                                y_pred_eval = np.round(y_pred)
                            else:
                                y_pred_eval = y_pred
                                
                            prediction_cache[idx] = (y_pred, y_pred_eval)
                        else:
                            y_pred, y_pred_eval = prediction_cache[idx]
                        
                        r2 = r2_score(y_fit, y_pred_eval)
                        complexity = eqs.iloc[idx].get("complexity", 1)
                        
                        # Calculate Pareto Score
                        # Using the weight as a power for r2
                        # Ensure r2 is positive to avoid issues with non-integer powers
                        safe_r2 = max(0.0001, r2)
                        score = (safe_r2 ** r2w) * np.exp(-lam * (complexity - 1))
                        
                        if score > best_score:
                            best_score = score
                            best_r2 = r2
                            best_idx = idx
                    except Exception as e:
                        # Catching Distributed.ProcessExitedException or other Julia/PySR errors
                        print(f"[DeepPySR] Warning: Error calculating Score for equation {idx} (possibly dead worker): {e}")
                        if "ProcessExitedException" in str(e):
                            print("[DeepPySR] Worker process died. Aborting Pareto score optimization for this target.")
                            break
                        continue
                
                # If all equation evaluations failed for this combination, best_idx might be 0 but best_score still -inf
                if best_score == -np.inf:
                    print(f"[DeepPySR] Warning: All equation evaluations failed for r2w={r2w}, lambda={lam}. Using default first equation.")
                    best_idx = 0
                    best_r2 = 0.0
                    best_score = 0.0
                    
                best = eqs.iloc[best_idx]
                formula = str(best["equation"]) if "equation" in best else str(best.get("sympy_format", ""))
                
                try:
                    sym_expr = model.sympy(best_idx)
                except Exception:
                    sym_expr = sp.sympify(formula)

                loss = float(best.get("loss", np.nan))
                
                all_results.append({
                    "sym_expr": sym_expr,
                    "score": best_score,
                    "loss": loss,
                    "complexity": int(best.get("complexity", -1)),
                    "r2": best_r2,
                    "pareto_r2_weight": r2w,
                    "pareto_lambda": lam
                })
        return all_results

    def fit(self, X, y):
        if isinstance(y, pd.Series):
            self.target_name_ = str(y.name) if y.name else "y"
            y_input = y.values.astype(np.float64)
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                self.target_name_ = str(y.columns[0])
                y_input = y.values.flatten().astype(np.float64)
            else:
                self.target_name_ = [str(c) for c in y.columns]
                y_input = y.values.astype(np.float64)
        else:
            y_input = np.array(y).astype(np.float64)
            if y_input.ndim == 1:
                self.target_name_ = "y"
            else:
                self.target_name_ = [f"y{i}" for i in range(y_input.shape[1])]

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = [str(c) for c in X.columns.tolist()]
            X_input_all = X.values.astype(np.float64)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            X_input_all = X.astype(np.float64)

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
                results = self._fit_single_target(X_fit, target_y, target_name)

                # Loop through all results (grid search might return multiple)
                for i, res in enumerate(results):
                    sym_expr = res["sym_expr"]
                    score = res["score"]
                    loss = res["loss"]
                    complexity = res["complexity"]
                    r2 = res["r2"]
                    r2w = res.get("pareto_r2_weight")
                    lam = res.get("pareto_lambda")

                    # Ensure sym_expr is a sympy expression
                    if not hasattr(sym_expr, "xreplace"):
                        sym_expr = sp.sympify(sym_expr)

                    # Round coefficients
                    if hasattr(sym_expr, "atoms"):
                        # We need to make sure we don't modify the original sym_expr in the results list
                        # though here it's fine as we are processing it.
                        for n in sym_expr.atoms(sp.Number):
                            sym_expr = sym_expr.xreplace({n: sp.Float(round(float(n), self.decimal))})

                    # pypysr uses 'v' prefix to avoid its internal x1->x0 translation
                    # pysr uses 'x' prefix.
                    prefix = "v" if self.model_provider == "pypysr" else "x"
                    
                    mapping = {}
                    for k in range(len(cols)):
                        mapping[sp.Symbol(f"{prefix}{k}")] = sp.Symbol(f"x{cols[k]}")
                    
                    if hasattr(sym_expr, "xreplace"):
                        sym_expr = sym_expr.xreplace(mapping)
                    
                    # Update involved variables after mapping to global x indices
                    involved = sorted({str(s) for s in sym_expr.free_symbols}) if hasattr(sym_expr, "free_symbols") else []

                    # For grid search, we only want to follow the 'best' one for layering.
                    # We pick the first result as the primary one for layering.
                    is_primary = (i == 0)

                    if is_primary and is_redundant(target_name, sym_expr, self.relationships_):
                        # If the primary one is redundant, we might still want to record the others?
                        # But DeepPySR's logic usually stops here.
                        # For now, if primary is redundant, we skip this target's results.
                        print(f"Relationship for {target_name} is redundant. Skipping.")
                        break

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
                        "r2": r2,
                        "pareto_r2_weight": r2w,
                        "pareto_lambda": lam
                    }

                    if is_primary and not is_root and score < self.stopping_score:
                        print(f"Goal reached for {target_name} (score={score:.2f} < {self.stopping_score}). Leaf node.")
                        # We still add it to relationships so it's visible, but we won't expand it.
                        self.relationships_.append(relationship)
                        # We don't expand primary, but we might still add other grid search results for this target
                        # However, typically if primary reached goal, we stop this target.
                        # For consistency, let's add all results for this target but don't expand primary.
                        continue 

                    self.relationships_.append(relationship)
                    
                    # Only the primary relationship triggers new entries in the queue
                    if is_primary and layer < self.max_layers:
                        for vname in involved:
                            try:
                                v_idx = int(vname[1:])
                                queue.append((vname, X_input_all[:, v_idx], layer + 1, target_name))
                            except (ValueError, IndexError):
                                continue

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error modeling {target_name}: {e}")

        self.save_relationships()
        print(self.relationships_)
        # Populate equations_ with the best relationship (highest R2 and highest score) for the root target
        if self.relationships_:
            root_name = self.target_name_[0] if isinstance(self.target_name_, list) else self.target_name_
            root_rels = [r for r in self.relationships_ if r['target'] == root_name]
            if root_rels:
                # Select the one with highest R2 and the one with highest score
                best_r2_rel = max(root_rels, key=lambda x: x.get('r2', -np.inf))
                best_score_rel = max(root_rels, key=lambda x: x.get('score', -np.inf))
                
                # Use a unique list of candidate relationships
                candidates = []
                # First one is highest R2
                candidates.append(best_r2_rel)
                # Second one is highest score, if different
                if best_score_rel["formula"] != best_r2_rel["formula"]:
                    candidates.append(best_score_rel)
                
                # We create a DataFrame that PySR expects for equations_
                self.equations_ = pd.DataFrame([
                    {
                        "equation": rel["formula"],
                        "sympy_format": rel["sympy"],
                        "loss": rel["loss"],
                        "score": rel["score"],
                        "complexity": rel["complexity"],
                        "r2": rel.get("r2", 0.0)
                    }
                    for rel in candidates
                ])
                # For user convenience, also set a single best equation string attribute (the one with highest R2)
                self._equation = best_r2_rel["formula"]
                
                # These attributes are often checked by PySR or scikit-learn
                self.nout_ = len(self.target_name_) if isinstance(self.target_name_, list) else 1
                self.selection_mask_ = np.ones(self.n_features_in_, dtype=bool)
        # print(self._equation)
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
            if hasattr(new_rel["sympy"], "xreplace"):
                new_rel["sympy"] = new_rel["sympy"].xreplace(sym_mapping)
            new_rel["formula"] = str(new_rel["sympy"])
            
            # Map involved
            new_rel["involved"] = sorted({str(s) for s in new_rel["sympy"].free_symbols})
            mapped_rels.append(new_rel)
        return mapped_rels

    def save_relationships(self, filename="relationships.csv"):
        path = os.path.join(self.output_dir, filename)
        mapped_rels = self._get_mapped_relationships()
        columns = ["layer", "target", "formula", "involved", "score", "r2", "loss", "complexity", "pareto_r2_weight", "pareto_lambda"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in mapped_rels:
                row = {col: r.get(col) for col in columns}
                # Format involved as a string if it's a list
                if isinstance(row["involved"], list):
                    row["involved"] = ", ".join(row["involved"])
                writer.writerow(row)



    def plot(self, filename="hierarchy.png", target_variable=None, pareto_r2_weight=None, pareto_lambda=None):
        if target_variable is None:
            target_variable = getattr(self, "target_name_", "y")
        feature_names = getattr(self, "feature_names_in_", None)
        
        # Default values for grid search selection if not specified
        r2w = pareto_r2_weight if pareto_r2_weight is not None else 1.0
        lam = pareto_lambda if pareto_lambda is not None else 0.001

        mapped_rels = self._get_mapped_relationships()
        
        # Filter relationships based on the specified r2w and lambda if multiple exist
        # We only filter if the relationship has these keys (old models might not)
        if mapped_rels and "pareto_r2_weight" in mapped_rels[0]:
            filtered_rels = []
            # If there are multiple relationships for the same target, we pick the one matching r2w/lam
            for rel in mapped_rels:
                if rel.get("pareto_r2_weight") == r2w and rel.get("pareto_lambda") == lam:
                    filtered_rels.append(rel)
            
            mapped_rels = filtered_rels

        if not mapped_rels and (not feature_names):
            print(f"No relationships found for r2w={r2w}, lambda={lam}.")
            return
        path = os.path.join(self.output_dir, filename)
        plot_n_layer_graph(mapped_rels, path, feature_names=feature_names, target_variable=target_variable)
        print(f"Plot saved to {path}")

    def plot_circle(self, filename="circle.png", target_variable=None, pareto_r2_weight=None, pareto_lambda=None):
        if target_variable is None:
            target_variable = getattr(self, "target_name_", "y")
        feature_names = getattr(self, "feature_names_in_", None)
        
        # Default values for grid search selection if not specified
        r2w = pareto_r2_weight if pareto_r2_weight is not None else 1.0
        lam = pareto_lambda if pareto_lambda is not None else 0.001

        mapped_rels = self._get_mapped_relationships()
        
        # Filter relationships based on the specified r2w and lambda
        if mapped_rels and "pareto_r2_weight" in mapped_rels[0]:
            filtered_rels = []
            for rel in mapped_rels:
                if rel.get("pareto_r2_weight") == r2w and rel.get("pareto_lambda") == lam:
                    filtered_rels.append(rel)
            mapped_rels = filtered_rels

        if not mapped_rels and (not feature_names):
            print(f"No relationships found for r2w={r2w}, lambda={lam}.")
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