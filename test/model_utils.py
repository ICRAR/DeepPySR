import os
import sys
import inspect
import numpy as np
import torch
import sympy
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from kan import KAN
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

# Optional AI Feynman 2.0 support.
AIFeynman = None
for _mod_name in ("aifeynman", "ai_feynman"):
    try:
        _mod = __import__(_mod_name, fromlist=["AIFeynman"])
        AIFeynman = getattr(_mod, "AIFeynman", None)
        if AIFeynman is not None:
            break
    except ImportError:
        continue

# --- Torch MLP with Dropout ---
class TorchMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(128, 64, 32), output_dim=1, dropout=0.2, task='regression', activation='leaky_relu'):
        super().__init__()
        layers = []
        last_dim = input_dim
        
        if activation == 'leaky_relu':
            act_fn = torch.nn.LeakyReLU(0.01)
        elif activation == 'gelu':
            act_fn = torch.nn.GELU()
        else:
            act_fn = torch.nn.ReLU()

        for h in hidden_layer_sizes:
            layers.append(torch.nn.Linear(last_dim, h))
            layers.append(torch.nn.BatchNorm1d(h))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            last_dim = h
        layers.append(torch.nn.Linear(last_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)
        self.task = task

    def forward(self, x):
        return self.network(x)

class MLPWrapper(BaseEstimator):
    def __init__(self, input_dim=None, hidden_layer_sizes=(256, 128, 64), dropout=0.1, 
                 lr=0.001, epochs=300, batch_size=32, weight_decay=1e-3, 
                 activation='leaky_relu', task='regression', random_state=42):
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.activation = activation
        self.task = task
        self.random_state = random_state
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        output_dim = 1
        if self.task == 'classification':
            unique_y = np.unique(y)
            if len(unique_y) > 2:
                output_dim = len(unique_y)
        
        self.model = TorchMLP(self.input_dim, self.hidden_layer_sizes, output_dim, self.dropout, self.task, self.activation).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        if self.task == 'regression':
            criterion = torch.nn.MSELoss()
        else:
            if output_dim == 1:
                criterion = torch.nn.BCEWithLogitsLoss()
            else:
                criterion = torch.nn.CrossEntropyLoss()

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        if self.task == 'regression':
            y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        else:
            if output_dim == 1:
                y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
            else:
                y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step(epoch_loss / len(loader))
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_t)
            if self.task == 'regression':
                return outputs.cpu().numpy().ravel()
            else:
                if outputs.shape[1] == 1:
                    return (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int).ravel()
                else:
                    return torch.argmax(outputs, dim=1).cpu().numpy()

    def predict_proba(self, X):
        if self.task != 'classification':
            raise ValueError("predict_proba only for classification")
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_t)
            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs).cpu().numpy().ravel()
                return np.column_stack([1 - probs, probs])
            else:
                return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

class MLPRegressorWrapper(MLPWrapper, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(task='regression', **kwargs)

class MLPClassifierWrapper(MLPWrapper, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(task='classification', **kwargs)

# --- DeepPySR Configs ---
def get_deeppysr_configs():
    configs = {}
    vps_list = [25, 50, 75]
    vpr_list = [50, 100, 150]
    aps_list = [1.0, 10.0, 50.0]
    # vps_list = [25]
    # vpr_list = [100]
    # aps_list = [50.0,100]
    vpm = 0.7  # Fixed tuned value for variable_prune_max

    # # 1. stdsr: All parameters set to 0
    # configs["stdsr_vps0_vpr0_aps0"] = {
    #     "adaptive_parsimony_scaling": 0.0,
    #     "variable_prune_start": 0,
    #     "variable_prune_ramp": 0,
    #     "variable_prune_max": 0.0,
    # }

    # # 2. srpsm: Only tune adaptive_parsimony_scaling, others set to 0
    # for aps in aps_list:
    #     configs[f"srpsm_vps0_vpr0_aps{aps}"] = {
    #         "adaptive_parsimony_scaling": aps,
    #         "variable_prune_start": 0,
    #         "variable_prune_ramp": 0,
    #         "variable_prune_max": 0.0,
    #     }
    #
    # # 3. srprn: Only tune variable_prune_start/ramp/max, set adaptive_parsimony_scaling to 0
    # for vps in vps_list:
    #     for vpr in vpr_list:
    #         configs[f"srprn_vps{vps}_vpr{vpr}_aps0"] = {
    #             "adaptive_parsimony_scaling": 0.0,
    #             "variable_prune_start": vps,
    #             "variable_prune_ramp": vpr,
    #             "variable_prune_max": vpm,
    #         }

    # 4. fullsr: Tune all 4 parameters
    for vps in vps_list:
        for vpr in vpr_list:
            for aps in aps_list:
                configs[f"fullsr_vps{vps}_vpr{vpr}_aps{aps}"] = {
                    "model_provider": "deeppysr",
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
            # "model_provider": "pysr",
            "adaptive_parsimony_scaling": aps,
        }
    return configs

# --- KAN Wrapper ---
class KANWrapper:
    def __init__(self, input_dim, output_dim=1, hidden_dim=5, steps=200, update_grid=False, task='regression', lamb=0.05, lamb_l1=1):
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

            # Ensure self.vars are converted to sympy symbols for lambdify
            symbols = [sympy.Symbol(str(v)) for v in self.vars]
            f = sympy.lambdify(symbols, self.formula, 'numpy')
            inputs = [X[:, i] for i in range(min(X.shape[1], len(symbols)))]
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


class AIFeynmanWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, max_runtime=300, max_complexity=20, transformations=None,
                 use_transformations=True, random_state=42):
        self.max_runtime = max_runtime
        self.max_complexity = max_complexity
        self.transformations = transformations or ["sin", "cos", "exp", "log", "sqrt"]
        self.use_transformations = use_transformations
        self.random_state = random_state
        self.model = None

    def _filter_kwargs(self, kwargs):
        if AIFeynman is None:
            return {}
        sig = inspect.signature(AIFeynman.__init__)
        return {k: v for k, v in kwargs.items() if k in sig.parameters and k != "self"}

    def fit(self, X, y):
        if AIFeynman is None:
            raise ImportError(
                "AI Feynman 2.0 is not installed. Install the 'aifeynman' package to use AI Feynman 2.0."
            )
        init_kwargs = self._filter_kwargs({
            "max_runtime": self.max_runtime,
            "max_complexity": self.max_complexity,
            "transformations": self.transformations,
            "use_transformations": self.use_transformations,
            "random_state": self.random_state,
        })
        self.model = AIFeynman(**init_kwargs)
        if hasattr(self.model, "fit"):
            self.model.fit(X, y)
        elif hasattr(self.model, "fit_model"):
            self.model.fit_model(X, y)
        else:
            raise AttributeError("AIFeynman class does not expose a fit method")
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("AI Feynman model is not fitted")
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(X)).ravel()
        if hasattr(self.model, "predict_from_features"):
            return np.asarray(self.model.predict_from_features(X)).ravel()
        raise AttributeError("AIFeynman class does not expose a predict method")

    def __repr__(self):
        return (
            f"AIFeynmanWrapper(max_runtime={self.max_runtime}, "
            f"max_complexity={self.max_complexity}, "
            f"transformations={self.transformations})"
        )

from deeppysr import DeepPySR
from pysr import PySRRegressor

# --- Model Factories ---
def get_baseline_models(task='regression', input_dim=None, output_dim=1, random_state=42):
    if task == 'classification':
        return {
            'LogisticRegression': LogisticRegression(C=0.1, max_iter=1000, random_state=random_state),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=random_state),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=random_state),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, reg_lambda=1, reg_alpha=0.1, subsample=0.8, random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
            'MLP': MLPClassifierWrapper(hidden_layer_sizes=(256, 128, 64), dropout=0.1, activation='leaky_relu', random_state=random_state),
            'KAN': KANWrapper(input_dim=input_dim, output_dim=output_dim, hidden_dim=5, steps=200, update_grid=False, task='classification')
        }
    else: # regression
        return {
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=random_state),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=random_state),
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, reg_lambda=1, reg_alpha=0.1, subsample=0.8, random_state=random_state),
            'MLP': MLPRegressorWrapper(hidden_layer_sizes=(256, 128, 64), dropout=0.1, activation='leaky_relu', random_state=random_state),
            # 'AI Feynman 2.0': AIFeynmanWrapper(
            #     max_runtime=300,
            #     max_complexity=20,
            #     transformations=['sin', 'cos', 'exp', 'log', 'sqrt'],
            #     random_state=random_state
            # ),
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
    
    # niterations 100 might be too slow for many runs, use 20 for tests
    return {
        "parallelism": parallelism,
        "maxsize": 40,
        "binary_operators": ["+", "*", "/", "-", "cond"],
        "extra_sympy_mappings": {
            'cond': sympy_cond,
        },
        "warm_start": True,
        "unary_operators": ["exp", "log", "sin", "sqrt"],
        "parsimony": 0.001,
        "populations": 100,
        "population_size": 200,
        "ncycles_per_iteration": 200,
        "verbosity": 0,
        "denoise": False, # Denoising can be very slow or get stuck, disabled for stability
        "turbo": False, # Disabled to avoid LoopVectorization warnings that clutter output
        "procs": default_procs,
        "niterations": 500,#100,
        # "timeout_in_seconds": 3000, # 10 minute timeout per fit to prevent hanging
    }
