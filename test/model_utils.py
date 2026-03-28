import os
import numpy as np
import torch
import sympy
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from kan import KAN
from DeepPySR.regressor import DeepPySRRegressor

# --- DeepPySR Configs ---
def get_deeppysr_configs():
    configs = {}
    vps_list = [25, 50]
    vpr_list = [50, 100]
    aps_list = [0.1, 1.0, 10.0, 50.0]
    vpm = 0.7  # Fixed tuned value for variable_prune_max

    # 1. stdsr: All parameters set to 0
    configs["stdsr_vps0_vpr0_aps0"] = {
        "adaptive_parsimony_scaling": 0.0,
        "variable_prune_start": 0,
        "variable_prune_ramp": 0,
        "variable_prune_max": 0.0,
    }

    # 2. srpsm: Only tune adaptive_parsimony_scaling, others set to 0
    for aps in aps_list:
        configs[f"srpsm_vps0_vpr0_aps{aps}"] = {
            "adaptive_parsimony_scaling": aps,
            "variable_prune_start": 0,
            "variable_prune_ramp": 0,
            "variable_prune_max": 0.0,
        }

    # 3. srprn: Only tune variable_prune_start/ramp/max, set adaptive_parsimony_scaling to 0
    for vps in vps_list:
        for vpr in vpr_list:
            configs[f"srprn_vps{vps}_vpr{vpr}_aps0"] = {
                "adaptive_parsimony_scaling": 0.0,
                "variable_prune_start": vps,
                "variable_prune_ramp": vpr,
                "variable_prune_max": vpm,
            }

    # 4. fullsr: Tune all 4 parameters
    for vps in vps_list:
        for vpr in vpr_list:
            for aps in aps_list:
                configs[f"fullsr_vps{vps}_vpr{vpr}_aps{aps}"] = {
                    "adaptive_parsimony_scaling": aps,
                    "variable_prune_start": vps,
                    "variable_prune_ramp": vpr,
                    "variable_prune_max": vpm,
                }
    return configs

# --- PySR Configs ---
def get_pysr_configs():
    configs = {}
    aps_list = [0.1, 1.0, 10.0, 50.0]
    for aps in aps_list:
        configs[f"pysr_aps{aps}"] = {
            "adaptive_parsimony_scaling": aps,
        }
    return configs

# --- KAN Wrapper ---
class KANWrapper:
    def __init__(self, input_dim, output_dim=1, hidden_dim=5, steps=200, update_grid=False, task='regression', lamb=0.01, lamb_l1=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure input, hidden, and output dims are integers
        input_dim = int(input_dim) if not isinstance(input_dim, list) else int(input_dim[0])
        hidden_dim = int(hidden_dim) if not isinstance(hidden_dim, list) else int(hidden_dim[0])
        output_dim = int(output_dim) if not isinstance(output_dim, list) else int(output_dim[0])
        
        self.width = [input_dim, hidden_dim, output_dim]
        # Pass a copy to avoid in-place modification of self.width by pykan
        self.model = KAN(width=self.width.copy(), device=self.device)
        self.lamb = lamb
        self.lamb_l1 = lamb_l1
        self.steps = steps
        self.task = task
        self.update_grid = update_grid
        self.formula = None
        self.vars = None

    def fit(self, X, y):
        # Reset symbolic info for a new training
        self.formula = None
        self.vars = None
        
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        # Ensure we use an integer for reshape
        out_dim = self.width[-1]
        if isinstance(out_dim, list):
            out_dim = out_dim[0]
            
        if self.task == 'classification' and out_dim == 1:
            y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        else:
            y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, int(out_dim)).to(self.device)
            
        dataset = {'train_input': X_t, 'train_label': y_t, 'test_input': X_t, 'test_label': y_t}
        self.model.fit(dataset, steps=self.steps, update_grid=self.update_grid, opt="LBFGS", lamb=self.lamb, lamb_l1=self.lamb_l1, lamb_entropy=2.,)
        return self

    def symbolize(self):
        if self.formula is not None:
            return self
        try:
            self.model.prune()
            self.model.auto_symbolic()
            res = self.model.symbolic_formula()
            if isinstance(res, tuple) and len(res) == 2:
                formulas, variables = res
                if formulas:
                    self.formula = formulas[0]
                    self.vars = variables
            elif isinstance(res, list):
                self.formula = res[0]
        except Exception as e:
            print(f"Warning: KAN symbolization failed: {e}")
        
        # If still None, default to 0.0
        if self.formula is None:
            self.formula = 0.0
            
        return self

    def predict_symbolic(self, X):
        if self.formula is None:
            return self.predict(X)
        
        # Handle default/constant formula
        if isinstance(self.formula, (int, float)) and self.formula == 0.0:
            y_pred = np.zeros(X.shape[0])
            if self.task == 'classification':
                return y_pred.astype(int)
            return y_pred
            
        try:
            # vars are x_1, x_2, ...
            if self.vars is None:
                return self.predict(X)
                
            f = sympy.lambdify(self.vars, self.formula, 'numpy')
            inputs = [X[:, i] for i in range(min(X.shape[1], len(self.vars)))]
            y_pred = f(*inputs)
            
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()
            
            if np.isscalar(y_pred):
                y_pred = np.full(X.shape[0], y_pred)
            
            y_pred = y_pred.ravel()
            # Clean NaNs and Infs for symbolic evaluation
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
            
            if self.task == 'classification':
                return (y_pred > 0.5).astype(int)
            return y_pred
        except Exception as e:
            print(f"Warning: KAN symbolic prediction failed: {e}")
            return self.predict(X)

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_pred = self.model(X_t).detach().cpu().numpy().ravel()
        # Clean NaNs and Infs
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if self.task == 'classification':
            out_dim = self.width[-1]
            if isinstance(out_dim, list):
                out_dim = out_dim[0]
                
            if out_dim == 1:
                return (y_pred > 0.5).astype(int)
            else:
                y_prob_multi = self.model(X_t).detach().cpu().numpy()
                y_prob_multi = np.nan_to_num(y_prob_multi, nan=0.0, posinf=1e10, neginf=-1e10)
                return np.argmax(y_prob_multi, axis=1)
        return y_pred

    def predict_proba(self, X):
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_prob = self.model(X_t).detach().cpu().numpy()
        # Clean NaNs and Infs
        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
        
        out_dim = self.width[-1]
        if isinstance(out_dim, list):
            out_dim = out_dim[0]

        if out_dim == 1:
            y_prob = np.clip(y_prob.ravel(), 0, 1)
            return np.column_stack([1 - y_prob, y_prob])
        return y_prob

# --- Model Factories ---
def get_baseline_models(task='regression', input_dim=None, output_dim=1, random_state=42):
    if task == 'classification':
        return {
            'LogisticRegression': LogisticRegression(C=0.1, max_iter=1000, random_state=random_state),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=random_state),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=random_state),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, reg_lambda=1, reg_alpha=0.1, subsample=0.8, random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
            'MLP': MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.1, max_iter=2000, random_state=random_state),
            'KAN': KANWrapper(input_dim=input_dim, output_dim=output_dim, hidden_dim=5, steps=200, update_grid=False, task='classification')
        }
    else: # regression
        return {
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=random_state),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=random_state),
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, reg_lambda=1, reg_alpha=0.1, subsample=0.8, random_state=random_state),
            'MLP': MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.1, max_iter=2000, random_state=random_state),
            'KAN': KANWrapper(input_dim=input_dim, output_dim=output_dim, hidden_dim=5, steps=200, update_grid=False, task='regression')
        }

def get_pysr_base_kwargs(os_cpu_count=None):
    if os_cpu_count is None:
        try:
            os_cpu_count = os.cpu_count() or 2
        except:
            os_cpu_count = 2
    
    sympy_cond = lambda x, y: sympy.Piecewise((y, x > 0), (0, True))
    
    # In PySR 1.x, parallelism='multiprocessing' with procs > 0 is standard for multi-core.
    # parallelism='multithreading' is for Julia-level threads and procs should be 0.
    # We'll use multithreading for better compatibility with PythonCall/Python objects on workers.
    parallelism = "multithreading" if os_cpu_count > 1 else "serial"
    
    # Use a more conservative default for procs to avoid memory exhaustion (Distributed.ProcessExitedException)
    # 4 workers is usually plenty for most symbolic regression tasks without excessive RAM overhead.
    # For multithreading, procs should be 0 as it uses Julia threads instead.
    default_procs = 0 if parallelism == "multithreading" else 0
    
    return {
        "parallelism": parallelism,
        "maxsize": 40,
        "binary_operators": ["+", "*", "/", "-", "cond(x,y) = x > 0 ? y : y*0"],
        "extra_sympy_mappings": {'cond': sympy_cond},
        "unary_operators": ["exp", "log"],
        "parsimony": 0.001,
        "populations": 30,
        "population_size": 200,
        "ncycles_per_iteration": 200,
        "verbosity": 1,
        "denoise": False, # Denoising can be very slow or get stuck, disabled for stability
        "turbo": False, # Disabled to avoid LoopVectorization warnings that clutter output
        "procs": default_procs,
        "niterations": 100,
        "timeout_in_seconds": 600, # 10 minute timeout per fit to prevent hanging
    }
