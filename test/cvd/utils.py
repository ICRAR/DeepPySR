import torch
import numpy as np
import sympy, os
from kan import KAN
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class KANClassifierWrapper:
    def __init__(self, input_dim, hidden_dim=5, lamb=0.1, lamb_l1=1.0, lamb_entropy=2.0, steps=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Width: [input, hidden, output]. For binary classification, output=1 (sigmoid) or 2 (softmax)
        # Using 1 output for binary classification with threshold
        self.model = KAN(width=[input_dim, hidden_dim, 1], device=self.device)
        self.formula = None
        self.lamb = lamb
        self.lamb_l1 = lamb_l1
        self.lamb_entropy = lamb_entropy
        self.steps = steps
    
    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        # Minimal training
        dataset = {'train_input': X_t, 'train_label': y_t, 'test_input': X_t, 'test_label': y_t}
        self.model.fit(dataset, 
                      steps=self.steps, 
                      lamb=self.lamb,
                      lamb_l1=self.lamb_l1,
                      lamb_entropy=self.lamb_entropy,
                      update_grid=False)
        return self
    
    def symbolize(self):
        try:
            self.model.prune()
            # Try to force some symbolic conversion if possible, or handle failures gracefully
            self.model.auto_symbolic()
            formulas = self.model.symbolic_formula()
            # For 1 output, formula is at index 0
            if formulas and len(formulas) > 0 and len(formulas[0]) > 0:
                self.formula = formulas[0][0]
                print(f"KAN Formula: {self.formula}")
            else:
                print("  Warning: No formula returned by KAN")
                self.formula = None
        except Exception as e:
            print(f"  Warning: KAN symbolization failed: {e}")
            self.formula = None
        return self

    def feature_importance(self):
        # KAN feature importance is available via the feature_score attribute
        # It's a tensor representing the importance of each input feature
        try:
            # MultiKAN has feature_score attribute (not a method)
            fs = self.model.feature_score
            if hasattr(fs, 'detach'):
                scores = fs.detach().cpu().numpy()
            else:
                scores = np.array(fs)
            
            # Ensure it's 1D
            if scores.ndim > 1:
                scores = scores.flatten()
            return scores
        except Exception as e:
            # Fallback: if not available, return zeros
            print(f"  Warning: KAN feature_importance extraction failed: {e}")
            return np.zeros(self.model.width[0])

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_pred = self.model(X_t).detach().cpu().numpy().ravel()
        return (y_pred > 0.5).astype(int)
    
    def predict_sym(self, X):
        if self.formula is None:
            return self.predict(X)
        
        # Use sympy to evaluate the formula
        # pykan typically uses x_1, x_2, ... for 1-based indexing in symbols
        # But let's be robust and check what symbols are actually in the formula
        all_symbols = sorted(list(self.formula.free_symbols), key=lambda s: s.name)
        
        # Create a mapping from symbol name to its index (assuming x_i where i is 1-based)
        # If it uses x_0, x_1, ... we handle that too.
        # However, to be safest, we should match the symbols by their names.
        
        # Determine if it's 1-based or 0-based
        # We'll create a list of symbols that matches the features in X
        # If KAN used x_1 for the first feature, we want to map x_1 to X[:, 0]
        
        symbol_names = [s.name for s in all_symbols]
        is_1_based = any(name == f"x_{X.shape[1]}" for name in symbol_names)
        
        if is_1_based:
            input_symbols = [sympy.Symbol(f"x_{i+1}") for i in range(X.shape[1])]
        else:
            input_symbols = [sympy.Symbol(f"x_{i}") for i in range(X.shape[1])]
            
        f = sympy.lambdify(input_symbols, self.formula, "numpy")
        
        y_pred_sym = f(*[X[:, i] for i in range(X.shape[1])])
        # Sometimes lambdify returns a scalar if the formula is constant
        if np.isscalar(y_pred_sym):
            y_pred_sym = np.full(X.shape[0], y_pred_sym)
            
        return (y_pred_sym > 0.5).astype(int)

    def predict_proba(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_prob = self.model(X_t).detach().cpu().numpy().ravel()
        # Clip to [0, 1] as KAN output might exceed it
        y_prob = np.clip(y_prob, 0, 1)
        return np.column_stack([1 - y_prob, y_prob])

    def predict_proba_sym(self, X):
        if self.formula is None:
            return self.predict_proba(X)
            
        all_symbols = sorted(list(self.formula.free_symbols), key=lambda s: s.name)
        symbol_names = [s.name for s in all_symbols]
        is_1_based = any(name == f"x_{X.shape[1]}" for name in symbol_names)
        
        if is_1_based:
            input_symbols = [sympy.Symbol(f"x_{i+1}") for i in range(X.shape[1])]
        else:
            input_symbols = [sympy.Symbol(f"x_{i}") for i in range(X.shape[1])]

        f = sympy.lambdify(input_symbols, self.formula, "numpy")
        y_prob = f(*[X[:, i] for i in range(X.shape[1])])
        if np.isscalar(y_prob):
            y_prob = np.full(X.shape[0], y_prob)
            
        y_prob = np.clip(y_prob, 0, 1)
        return np.column_stack([1 - y_prob, y_prob])


# --- Result Saving ---
def save_metrics(model_name, outdir, y_true, y_pred, y_prob, patient_ids, feature_importances=None, feature_names=None, filename_suffix=''):
    os.makedirs(outdir, exist_ok=True)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5 # fallback

    # Also save metrics to a CSV file
    metrics_dict = {
        'model': [model_name],
        'accuracy': [acc],
        'precision': [prec],
        'recall': [rec],
        'f1': [f1],
        'auc': [auc]
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(os.path.join(outdir, f'metrics{filename_suffix}.csv'), index=False)

    # Save predictions with patient id
    predictions_df = pd.DataFrame({
        'patientid': patient_ids,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    })
    predictions_df.to_csv(os.path.join(outdir, f'predictions{filename_suffix}.csv'), index=False)

    # Save feature importance if available
    if feature_importances is not None:
        if feature_names is None:
            feature_names = [f"x_{i}" for i in range(len(feature_importances))]
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        fi_df.to_csv(os.path.join(outdir, 'feature_importance.csv'), index=False)


    print(f"Results for {model_name}: Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    return {"model": model_name, "accuracy": acc, "f1": f1, "auc": auc}