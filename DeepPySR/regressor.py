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
        max_layers=1,
        output_dir="outputs/deepPySR",
        stopping_score = 2,
        model_provider = "pypysr",
        pypysr_path = None,
        pareto_lambda = 0.001,
        pareto_r2_weight = 1.0,
        batching = False,
        batch_size = 50,
        use_mdl = True,
        use_nsga2 = True,
        use_lexicase = True,
        use_hotspot_protection = True,
        loss_history = False,
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
        self.batching = batching
        self.batch_size = batch_size
        self.use_mdl = use_mdl
        self.use_nsga2 = use_nsga2
        self.use_lexicase = use_lexicase
        self.use_hotspot_protection = use_hotspot_protection
        self.relationships_ = []
        self.equations_ = None
        self.loss_history = loss_history
        self.loss_history_ = []

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
            "batching": self.batching,
            "batch_size": self.batch_size,
            "use_mdl": self.use_mdl,
            "use_nsga2": self.use_nsga2,
            "use_lexicase": self.use_lexicase,
            "use_hotspot_protection": self.use_hotspot_protection,
        }
        params.update(self.pysr_kwargs)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.pysr_kwargs[key] = value
        return self


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

    def predict_proba(self, X):
        """
        Return the raw predictions clipped to [0, 1] to serve as probabilities.
        """
        preds = self.predict(X)
        return np.clip(preds, 0, 1)

    def _fit_provider(self, X, y, target_names):
        """Fits one or more targets using the current model provider (pysr or pypysr)."""
        # y can be 1D or 2D (multi-target)
        if not isinstance(target_names, list):
            target_names = [target_names]
            y = y.reshape(-1, 1)
            
        # Create a new PySRRegressor instance with the same parameters as self
        # but with a specific random_state and potentially other tweaks for sub-fitting
        params = self.get_params()

        # Remove deepPySR specific params before passing to PySRRegressor
        for p in [
            "max_layers", "output_dir", "decimal", "stopping_score", "relationships_", 
            "model_provider", "pareto_lambda", "pareto_r2_weight", "pypysr_path",
            "use_mdl", "use_nsga2", "use_lexicase", "use_hotspot_protection",
            "variable_prune_max", "variable_prune_start", "variable_prune_ramp"
        ]:
            if self.model_provider not in ["pypysr", "pypysrdev1"] and p in ["variable_prune_max", "variable_prune_start", "variable_prune_ramp"]:
                params.pop(p, None)
            elif p in ["use_mdl", "use_nsga2", "use_lexicase", "use_hotspot_protection"]:
                params.pop(p, None)
            elif p in ["max_layers", "output_dir", "decimal", "stopping_score", "relationships_", "model_provider", "pareto_lambda", "pareto_r2_weight", "pypysr_path"]:
                params.pop(p, None)

        # Batching logic: if X.shape[0] > 10000 and user didn't specify batching, enable it.
        # But if user DID specify batching or batch_size, keep those.
        if X.shape[0] > 10000:
            if "batching" not in self.pysr_kwargs:
                params["batching"] = True
            if "batch_size" not in self.pysr_kwargs and self.batch_size == 50:
                 # If user didn't override batch_size, use 500 for large datasets
                params["batch_size"] = 500

        # Apply conservative defaults for stability if not provided
        if "procs" not in params:
            # Use at most 4 workers to avoid Distributed.ProcessExitedException (memory exhaustion)
            # Standard PySR 1.x default is often too aggressive for many-layered DeepPySR
            params["procs"] = min((os.cpu_count() or 2) - 1, 4)

        if self.model_provider in ["pypysr", "pypysrdev1"] and len(target_names) > 1:
            # pypysrdev1 (and potentially pypysr) fails with multi-output if progress is True
            params["progress"] = False


        # Configure PySR to use our output_dir for its files
        # We use a subfolder for each target set to avoid collisions if running in parallel
        # though currently it's sequential.
        target_output_dir = os.path.join(self.output_dir, "pysr_outputs", "_".join(target_names[:3]) + (f"_{len(target_names)}" if len(target_names) > 3 else ""))
        os.makedirs(target_output_dir, exist_ok=True)
        
        params["output_directory"] = target_output_dir
        
        # Determine variable names for this fit
        n_features = X.shape[1]
        v_names = [f"x{i}" for i in range(n_features)]
            
        # Dynamically import PySRRegressor
        import sys

        # 2. Clear sys.modules to force re-import when switching providers
        # This is necessary because both providers might share sub-module names
        # or we want to ensure we're getting the one from the current sys.path.
        if "pysr" in sys.modules or "pypysr" in sys.modules:
            for mod in list(sys.modules.keys()):
                if mod in ['pysr', 'pypysr'] or mod.startswith(('pysr.', 'pypysr.')):
                    del sys.modules[mod]

        # 2.1 Handle Julia environment switching
        # pypysr activates its own internal Julia environment, which can break standard pysr.
        # If we are switching back to pysr, we might need to reactivate the default environment.
        if "juliacall" in sys.modules:
            try:
                from juliacall import Main as jl
                jl.seval("using Pkg")
                if self.model_provider in ["pypysr"]:
                    # pypysr's fit() will handle its own Pkg.activate()
                    pass
                else:
                    # If we are in pysr mode, ensure we are NOT in pypysr's environment.
                    # Standard pysr expects the environment in .venv/julia_env or similar.
                    # We try to activate the environment managed by juliapkg.
                    try:
                        import juliapkg
                        pysr_env = juliapkg.project()
                        jl.Pkg.activate(pysr_env)
                        if params.get("verbosity", 0) > 0:
                            print(f"[DeepPySR] Reactivated PySR Julia environment at {pysr_env}")
                    except (ImportError, Exception):
                        # Fallback to default activation if juliapkg is not available or fails
                        jl.Pkg.activate()
                        if params.get("verbosity", 0) > 0:
                            print(f"[DeepPySR] Reactivated default Julia environment")
            except Exception as e:
                if params.get("verbosity", 0) > 0:
                    print(f"[DeepPySR] Warning: Failed to manage Julia environment: {e}")

        # 3. Import the requested provider
        try:
            if self.model_provider == "pypysr":
                # If we are in pypysr mode, and we were previously in pysr mode,
                # we should check if MyPySR can be loaded.
                # pypysr's _initialize_julia() will activate its own environment.
                try:
                    # Add MyPySR python path if not already in sys.path
                    pypysr_path = self.pypysr_path or os.path.expanduser("~/Projects/mypysr.jl/python")
                    if pypysr_path not in sys.path:
                        sys.path.insert(0, pypysr_path)
                    from pypysr import PySRRegressor
                    from juliacall import Main as jl
                    # Ensure MyPySR is loaded and available
                    try:
                        import pypysr
                        pypysr.sr._initialize_julia()
                    except:
                        pass
                except Exception as e:
                    if params.get("verbosity", 0) > 0:
                        print(f"[DeepPySR] Error importing pypysr: {e}")
                    # Try to force re-import if it failed due to already loaded julia packages
                    if "pypysr" in sys.modules:
                        del sys.modules["pypysr"]
                    from pypysr import PySRRegressor

                if params.get("verbosity", 0) > 0:
                    try:
                        import pypysr
                        print(f"[DeepPySR] Using pypysr from: {os.path.abspath(pypysr.__file__)}")
                    except ImportError:
                        print(f"[DeepPySR] Using pypysr")
            else:
                # We must import pysr first
                import pysr
                
                # 4. Handle pysr-specific Julia package binding
                # If pypysr was previously used, it might have changed the Julia environment
                # or even loaded a different version of SymbolicRegression.
                # We force a reload of pysr.julia_import to ensure 'SymbolicRegression' 
                # is correctly bound to the one in the current (reactivated) environment.
                try:
                    import importlib
                    importlib.reload(pysr.julia_import)
                except Exception as e:
                    if params.get("verbosity", 0) > 0:
                        print(f"[DeepPySR] Warning: Failed to reload pysr.julia_import: {e}")

                from pysr import PySRRegressor
                if params.get("verbosity", 0) > 0:
                    try:
                        print(f"[DeepPySR] Using standard pysr from: {os.path.abspath(pysr.__file__)}")
                    except Exception:
                        print(f"[DeepPySR] Using standard pysr")
        except ImportError as e:
            if params.get("verbosity", 0) > 0:
                print(f"[DeepPySR] Critical Error: Failed to import {self.model_provider}: {e}")
            raise e
        
        # Remove record_loss_history from params as it's not a PySR keyword arg
        params.pop("record_loss_history", None)
        params.pop("record_loss", None) # Also remove record_loss just in case
        # Also remove logger if we are using it
        logger_to_use = params.pop("logger", None)

        # Set up loss logger for pypysr or pysr
        if self.model_provider == "pypysr":
            try:
                from juliacall import Main as jl
                # Define a custom logger in Julia to record loss per iteration
                # We use 'Main.MyPySR' to be explicit since we're in Main
                jl.seval("""
                if !isdefined(Main, :PythonLossLogger)
                    try
                        import MyPySR: AbstractSRLogger, get_logger
                        import .MyPySR.LoggingModule: logging_callback!, should_log, increment_log_step!
                    catch
                        import .MyPySR: AbstractSRLogger, get_logger
                        import .MyPySR.LoggingModule: logging_callback!, should_log, increment_log_step!
                    end
                    using Logging: ConsoleLogger
                    
                    mutable struct PythonLossLogger <: AbstractSRLogger
                        losses::Vector{Float64}
                        iterations::Vector{Int}
                        cur_iteration::Int
                        log_interval::Int
                    end
                    
                    get_logger(logger::PythonLossLogger) = ConsoleLogger()
                    should_log(logger::PythonLossLogger) = true
                    increment_log_step!(logger::PythonLossLogger) = nothing
                    
                    function logging_callback!(logger::PythonLossLogger; state, datasets, ropt, options)
                        logger.cur_iteration += 1
                        if logger.cur_iteration % logger.log_interval == 0
                            # Get min loss from first dataset's Hall of Fame
                            hof = state.halls_of_fame[1]
                            if hof.exists[end] # Check if anything exists
                                # find min loss
                                min_loss = Inf
                                for member in hof.members[hof.exists]
                                    if member.loss < min_loss
                                        min_loss = member.loss
                                    end
                                end
                                push!(logger.losses, Float64(min_loss))
                                push!(logger.iterations, Int(logger.cur_iteration))
                            end
                        end
                        return nothing
                    end
                end
                """)
                # Create the logger instance
                log_interval = 1 # Record every iteration
                self._jl_logger = jl.PythonLossLogger([], [], 0, log_interval)
                # We must use 'full_options' or similar if the provider doesn't support 'logger' in constructor
                # but pypysr should support it if it's based on recent SymbolicRegression.jl
                # For now, let's try to pass it via pysr_kwargs if it fails in constructor
                logger_to_use = self._jl_logger
            except Exception as e:
                print(f"[DeepPySR] Warning: Failed to setup Julia loss logger: {e}")

        elif self.model_provider == "pysr":
            try:
                from pysr.logger_specs import AbstractLoggerSpec
                from juliacall import Main as jl

                jl.seval("""
                if !isdefined(Main, :PythonLossLogger)
                    import SymbolicRegression: AbstractSRLogger, get_logger
                    import SymbolicRegression.LoggingModule
                    using Logging
                    
                    mutable struct PythonLossLogger <: AbstractLogger
                        losses::Vector{Float64}
                        iterations::Vector{Int}
                        cur_iteration::Int
                        log_interval::Int
                    end
                    
                    Logging.min_enabled_level(logger::PythonLossLogger) = Logging.Info
                    Logging.shouldlog(logger::PythonLossLogger, level, _module, group, id) = true
                    Logging.log(logger::PythonLossLogger, level, message, _module, group, id, file, line; kwargs...) = nothing
                    
                    # Define the logging_callback! method in the SymbolicRegression.LoggingModule
                    SymbolicRegression.LoggingModule.eval(quote
                        function logging_callback!(logger::SRLogger{Main.PythonLossLogger}; state, datasets, ropt, options)
                            if should_log(logger)
                                my_logger = logger.logger
                                hof = state.halls_of_fame[1]
                                if any(hof.exists)
                                    min_loss = minimum(member.loss for member in hof.members[hof.exists])
                                    my_logger.cur_iteration += 1
                                    push!(my_logger.losses, Float64(min_loss))
                                    push!(my_logger.iterations, Int(my_logger.cur_iteration))
                                end
                                increment_log_step!(logger)
                            end
                        end
                    end)
                end

                """)

                log_interval = 1
                my_logger = jl.PythonLossLogger([], [], 0, log_interval)
                self._jl_logger = jl.SRLogger(logger=my_logger, log_interval=log_interval)
                class PySRLossLoggerSpec(AbstractLoggerSpec):
                    def __init__(self, logger):
                        self._logger = logger

                    def create_logger(self):
                        return self._logger

                    def write_hparams(self, logger, hparams):
                        return None

                    def close(self, logger):
                        return None

                params["logger_spec"] = PySRLossLoggerSpec(self._jl_logger)
            except Exception as e:
                if params.get("verbosity", 0) > 0:
                    print(f"[DeepPySR] Warning: Failed to setup pysr loss logger: {e}")

        model = PySRRegressor(**params)
        
        if logger_to_use is not None:
             # PySRRegressor (pypysr) uses logger_spec in constructor
             model.logger_spec = logger_to_use

        # Robustly handle y shape
        y_fit = np.array(y).astype(np.float64)
        if y_fit.ndim == 2:
            if y_fit.shape[1] == 1:
                y_fit = y_fit.ravel()
            else:
                # Multiple targets: check if we are using pypysr
                # They expect [targets, rows] in Julia, but the Python wrapper 
                # for pypysr handles transposing internally if we pass [rows, targets].
                # Wait, if pypysr.sr.PySRRegressor.fit does y_jl = y_np.T, 
                # then it expects [rows, targets] from us.
                pass
        
        # Ensure X is [rows, features] for model.fit() as it is standard Scikit-Learn.
        X_fit = X
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            try:
                # Both standard pysr and our custom providers now support multi-target in a single fit
                model.fit(X_fit, y_fit, variable_names=v_names)
            except Exception as e:
                if self.model_provider in ["pypysr"] and "juliacall" in sys.modules:
                    print(f"\n[DeepPySR] Error during {self.model_provider} fit: {e}")
                    print("[DeepPySR] This is often caused by Julia environment conflicts after using standard pysr in the same session.")
                    print("[DeepPySR] Please try to restart the Python session if you need to switch between providers.")
                raise e
        
        # Record loss history if requested - one entry per target
        if self.loss_history and self.model_provider in ["pypysr", "pysr"] and hasattr(self, "_jl_logger"):
            try:
                # For multi-target, create a separate loss history entry for each target
                # They all optimize together so iterations/losses are the same, but we record per-target
                for target_name in target_names:
                    if self.model_provider == "pypysr":
                        iterations = list(self._jl_logger.iterations)
                        losses = list(self._jl_logger.losses)
                    else:
                        iterations = list(self._jl_logger.logger.iterations)
                        losses = list(self._jl_logger.logger.losses)
                    self.loss_history_.append({
                        "target": target_name,
                        "iterations": iterations,
                        "losses": losses
                    })
            except Exception:
                pass
        
        # PySR returns equations_ as a list of dataframes for multi-target, or a single dataframe for single-target
        all_eqs = model.equations_
        if all_eqs is None or (isinstance(all_eqs, list) and len(all_eqs) == 0):
             raise ValueError(f"No equations found for targets {target_names}")

        # pypysr might return a dictionary of dataframes for multi-target
        # but let's check for standard PySR list first
        if isinstance(all_eqs, dict):
            all_eqs = [all_eqs[i] for i in sorted(all_eqs.keys())]
        elif not isinstance(all_eqs, list):
            all_eqs = [all_eqs]

        # Prepare parameters for grid search
        r2w_list = self.pareto_r2_weight if isinstance(self.pareto_r2_weight, (list, np.ndarray)) else [self.pareto_r2_weight]
        lambda_list = self.pareto_lambda if isinstance(self.pareto_lambda, (list, np.ndarray)) else [self.pareto_lambda]

        multi_target_results = {}
        
        for t_idx, target_name in enumerate(target_names):
            eqs = all_eqs[t_idx]
            
            # Dictionary to cache predictions to avoid redundant model.predict calls
            prediction_cache = {}
            target_results = []
            
            # Extract current target y
            if y_fit.ndim == 1:
                cur_y_fit = y_fit
            else:
                cur_y_fit = y_fit[:, t_idx]

            for r2w in r2w_list:
                for lam in lambda_list:
                    best_score = -np.inf
                    best_idx = 0
                    best_r2 = -np.inf
                    
                    for idx in range(len(eqs)):
                        try:
                            # Use model.predict to get predictions for this equation
                            # model.predict handles multi-target correctly if we pass index and target
                            cache_key = (t_idx, idx)
                            if cache_key not in prediction_cache:
                                if len(all_eqs) > 1:
                                    # Multi-target predict
                                    # Check if the model is from pypysr or standard pysr
                                    # Standard pysr 0.x might not support 'target' in predict, 
                                    # but it supports multiple outputs in one predict call if index is a list?
                                    # Actually, for standard pysr, we can just call predict(X, index=idx) 
                                    # and it returns a 2D array if it's a multi-target model.
                                    # 1. Try passing index directly (works for some versions, returns 2D if multi-target)
                                    try:
                                        y_pred_all = model.predict(X, index=idx)
                                        if isinstance(y_pred_all, (list, tuple)) and len(y_pred_all) > t_idx:
                                             y_pred = y_pred_all[t_idx]
                                        elif hasattr(y_pred_all, "ndim") and y_pred_all.ndim == 2:
                                            y_pred = y_pred_all[:, t_idx]
                                        else:
                                            y_pred = y_pred_all
                                    except Exception:
                                        # 2. Try passing indices as a list (common for newer multi-output PySR)
                                        try:
                                            indices = [None] * len(all_eqs)
                                            indices[t_idx] = idx
                                            y_pred_all = model.predict(X, index=indices)
                                            if isinstance(y_pred_all, (list, tuple)) and len(y_pred_all) > t_idx:
                                                 y_pred = y_pred_all[t_idx]
                                            elif hasattr(y_pred_all, "ndim") and y_pred_all.ndim == 2:
                                                y_pred = y_pred_all[:, t_idx]
                                            else:
                                                y_pred = y_pred_all
                                        except Exception:
                                            # 3. Last resort fallback for other providers (pypysr)
                                            try:
                                                # Check if the provider is actually pypysr
                                                if self.model_provider in ["pypysr"]:
                                                    try:
                                                        y_pred = model.predict(X, index=idx, target=t_idx)
                                                    except TypeError:
                                                        # Fallback if 'target' is not a keyword arg
                                                        y_pred_all = model.predict(X, index=idx)
                                                        if hasattr(y_pred_all, "ndim") and y_pred_all.ndim == 2:
                                                            y_pred = y_pred_all[:, t_idx]
                                                        elif isinstance(y_pred_all, (list, tuple)):
                                                            y_pred = y_pred_all[t_idx]
                                                        else:
                                                            y_pred = y_pred_all
                                                else:
                                                    # Standard PySR multi-output: try to predict everything and slice
                                                    # best_idx is usually 0 for the first equation
                                                    y_pred_all = model.predict(X) 
                                                    if isinstance(y_pred_all, (list, tuple)) and len(y_pred_all) > t_idx:
                                                         y_pred = y_pred_all[t_idx]
                                                    elif hasattr(y_pred_all, "ndim") and y_pred_all.ndim == 2:
                                                        y_pred = y_pred_all[:, t_idx]
                                                    else:
                                                        y_pred = y_pred_all
                                                    
                                                    if idx != 0:
                                                        # We can't easily get other indices for standard PySR multi-output
                                                        # if the above list-of-indices method failed.
                                                        # We'll just use these but R2/Score will only be accurate for idx=0.
                                                        pass
                                            except Exception as e:
                                                # 4. If everything fails, maybe we can't get individual predictions
                                                # for specific equations in multi-output mode easily
                                                print(f"[DeepPySR] Warning: Could not get individual prediction for target {t_idx} index {idx}: {e}")
                                                y_pred = np.zeros(X.shape[0])
                                else:
                                    y_pred = model.predict(X, index=idx)
                                    
                                # Handle NaNs and Infs in predictions
                                y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
                                
                                unique_fit = np.unique(cur_y_fit)
                                if len(unique_fit) > 2 and np.all(np.equal(np.mod(cur_y_fit, 1), 0)):
                                    y_pred_eval = np.round(y_pred)
                                else:
                                    y_pred_eval = y_pred
                                    
                                prediction_cache[cache_key] = (y_pred, y_pred_eval)
                            else:
                                y_pred, y_pred_eval = prediction_cache[cache_key]
                            
                            r2 = r2_score(cur_y_fit, y_pred_eval)
                            complexity = eqs.iloc[idx].get("complexity", 1)
                            
                            safe_r2 = max(0.0001, r2)
                            score = (safe_r2 ** r2w) * np.exp(-lam * (complexity - 1))
                            
                            if score > best_score:
                                best_score = score
                                best_r2 = r2
                                best_idx = idx
                        except Exception as e:
                            print(f"[DeepPySR] Warning: Error calculating Score for target {target_name} equation {idx}: {e}")
                            continue
                    
                    if best_score == -np.inf:
                        best_idx = 0
                        best_r2 = 0.0
                        best_score = 0.0
                        
                    best = eqs.iloc[best_idx]
                    formula = str(best["equation"]) if "equation" in best else str(best.get("sympy_format", ""))
                    
                    try:
                        # model.sympy(best_idx) might need target for multi-output
                        if len(all_eqs) > 1:
                            try:
                                sym_expr = model.sympy(best_idx, target=t_idx)
                            except Exception:
                                # Fallback if target is not supported
                                try:
                                    sym_expr_list = model.sympy(best_idx)
                                    if isinstance(sym_expr_list, list):
                                        sym_expr = sym_expr_list[t_idx]
                                    else:
                                        sym_expr = sym_expr_list
                                except Exception:
                                    sym_expr = sp.sympify(formula)
                        else:
                             sym_expr = model.sympy(best_idx)
                    except Exception:
                        sym_expr = sp.sympify(formula)

                    target_results.append({
                        "sym_expr": sym_expr,
                        "score": best_score,
                        "loss": float(best.get("loss", 1e10)),
                        "complexity": int(best.get("complexity", -1)),
                        "r2": best_r2,
                        "pareto_r2_weight": r2w,
                        "pareto_lambda": lam
                    })
            multi_target_results[target_name] = target_results
            
        return multi_target_results

    def _process_fit_results(self, target_name, results, layer, parent_name, X_input_all, cols, queue):
        """Processes results from _fit_provider and updates relationships and queue."""
        is_root = (isinstance(self.target_name_, list) and target_name in self.target_name_) or \
                  (not isinstance(self.target_name_, list) and target_name == self.target_name_)

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
                for n in list(sym_expr.atoms(sp.Number)):
                    sym_expr = sym_expr.xreplace({n: sp.Float(round(float(n), self.decimal))})

            prefix = "x"
            
            mapping = {}
            for k in range(len(cols)):
                mapping[sp.Symbol(f"{prefix}{k}")] = sp.Symbol(f"x{cols[k]}")
            
            if hasattr(sym_expr, "xreplace"):
                sym_expr = sym_expr.xreplace(mapping)
            
            involved = sorted({str(s) for s in sym_expr.free_symbols}) if hasattr(sym_expr, "free_symbols") else []
            is_primary = (i == 0)

            if is_primary and is_redundant(target_name, sym_expr, self.relationships_, layer=layer):
                print(f"Relationship for {target_name} is redundant. Skipping.")
                # Important: don't 'break' as it would skip OTHER targets if processed in batch
                # But here we are in a loop over grid search results for a SINGLE target.
                # So break is actually correct for THIS target.
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
                self.relationships_.append(relationship)
                continue 

            self.relationships_.append(relationship)
            
            if is_primary and layer < self.max_layers:
                for vname in involved:
                    try:
                        v_idx = int(vname[1:])
                        queue.append((vname, X_input_all[:, v_idx], layer + 1, target_name))
                    except (ValueError, IndexError):
                        continue

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
        self.loss_history_ = []
        
        # Initialize queue with all targets
        queue = []
        if isinstance(self.target_name_, list):
            # Fit all root targets together
            print(f"--- Fitting root targets {self.target_name_} at layer 1 ---")
            try:
                multi_results = self._fit_provider(X_input_all, y_input, self.target_name_)
                for target_name in self.target_name_:
                    results = multi_results[target_name]
                    self._process_fit_results(target_name, results, 1, None, X_input_all, list(range(X_input_all.shape[1])), queue)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error modeling root targets: {e}")
                raise e
        else:
            queue.append((self.target_name_, y_input, 1, None))
            
        processed_targets = set()
        if isinstance(self.target_name_, list):
            processed_targets.update(self.target_name_)
        else:
            # We don't add it yet as it's in the queue to be processed
            pass

        while queue:
            target_name, target_y, layer, parent_name = queue.pop(0)
            if layer > self.max_layers:
                continue

            # Skip if target already processed, but allow roots to be re-processed if they appear as features
            is_root = (isinstance(self.target_name_, list) and target_name in self.target_name_) or \
                      (not isinstance(self.target_name_, list) and target_name == self.target_name_)
            
            if target_name in processed_targets and not is_root:
                continue
            processed_targets.add(target_name)

            print(f"--- Fitting {target_name} at layer {layer} ---")

            # For non-root targets, we still fit one by one for now as they might have different feature masks
            # though in current DeepPySR they mostly use all features (minus themselves).
            try:
                idx = -1
                if not is_root:
                    try:
                        idx = int(target_name[1:])
                    except (ValueError, IndexError):
                        idx = -1
                
                parent_idx = None
                if parent_name and parent_name.startswith("x") and not ((isinstance(self.target_name_, list) and parent_name in self.target_name_) or (not isinstance(self.target_name_, list) and parent_name == self.target_name_)):
                    try:
                        parent_idx = int(parent_name[1:])
                    except (ValueError, IndexError):
                        pass

                cols = [j for j in range(X_input_all.shape[1]) if j != idx and j != parent_idx]
                X_fit = X_input_all[:, cols]

                multi_results = self._fit_provider(X_fit, target_y, target_name)
                results = multi_results[target_name]
                self._process_fit_results(target_name, results, layer, parent_name, X_input_all, cols, queue)

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error modeling {target_name}: {e}")
                raise e

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
                prefix = "x"
                sym_mapping = {sp.Symbol(f"{prefix}{i}"): sp.Symbol(name) for i, name in enumerate(self.feature_names_in_)}
                if hasattr(new_rel["sympy"], "xreplace"):
                    new_rel["sympy"] = new_rel["sympy"].xreplace(sym_mapping)
                new_rel["formula"] = str(new_rel["sympy"])
                
                # If target_name is a root target, also map it to its original name
                if isinstance(self.target_name_, list):
                    # Find which original target this is
                    try:
                        t_idx = int(new_rel["target"][1:])
                        new_rel["target"] = self.target_name_[t_idx]
                    except (ValueError, IndexError):
                        pass
                
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
                        prefix = "x"
                        sym_mapping = {sp.Symbol(f"{prefix}{i}"): sp.Symbol(name) for i, name in enumerate(self.feature_names_in_)}
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
                
            # Map x0, x1... (or v0, v1... for pypysr) to original feature names
            prefix = "x"
            mapping = {sp.Symbol(f"{prefix}{i}"): sp.Symbol(name) for i, name in enumerate(self.feature_names_in_)}
            return expr.xreplace(mapping)

    def latex(self):
        """Returns the LaTeX representation of the top-level relationship."""
        s = self.sympy()
        return sp.latex(s) if s else ""