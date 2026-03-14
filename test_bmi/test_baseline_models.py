import os
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from kan import KAN
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from utils import load_agg_data

# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def save_baseline_results(model_name, setting, age, outdir, results_df, feature_importances=None):
    os.makedirs(outdir, exist_ok=True)
    
    # Save predictions
    results_df.to_csv(os.path.join(outdir, 'predictions.csv'), index=False)
    
    # Save feature importances if provided
    if feature_importances is not None:
        feature_importances.to_csv(os.path.join(outdir, 'feature_importances.csv'), index=False)
    
    # Calculate metrics
    y_true = results_df['target_bmi']
    y_pred = results_df['pred_bmi']
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate symbolic metrics if available
    r2_sym = mae_sym = rmse_sym = None
    if 'pred_bmi_sym' in results_df.columns:
        y_pred_sym = results_df['pred_bmi_sym']
        r2_sym = r2_score(y_true, y_pred_sym)
        mae_sym = mean_absolute_error(y_true, y_pred_sym)
        rmse_sym = np.sqrt(mean_squared_error(y_true, y_pred_sym))
    
    # Save metrics to a file
    metrics_path = os.path.join(outdir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"R2: {r2:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        if r2_sym is not None:
            f.write(f"\nSymbolic Metrics:\n")
            f.write(f"R2_sym: {r2_sym:.4f}\n")
            f.write(f"MAE_sym: {mae_sym:.4f}\n")
            f.write(f"RMSE_sym: {rmse_sym:.4f}\n")
    
    print(f"Results for {model_name} ({setting} {age if age else ''}) saved to {outdir}")
    print(f"  R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    if r2_sym is not None:
        print(f"  Symbolic R2: {r2_sym:.4f}, MAE: {mae_sym:.4f}, RMSE: {rmse_sym:.4f}")

def run_baselines():
    # The ages from the project
    ages = [8, 10, 14, 17, 20, 23, 27]
    settings = ['longitudinal', 'age']
    
    # The models requested: KAN, ERF, Elastic Net, XGBoost, MLP
    # For KAN, I will use KANPySRRegressor if possible, else skip or use a simple KAN.
    # Given the project title is DeepPySR, KANPySRRegressor is likely the "KAN" they mean.
    
    for setting in settings:
        if setting == 'longitudinal':
            print("\nRunning Longitudinal Baselines...")
            # load_agg_data returns dataid, datain, dataout
            child_id, X, y = load_agg_data()
            runs = [(None, child_id, X, y)]
        else:
            print("\nRunning Age-specific Baselines...")
            runs = []
            for age in ages:
                child_id_age, X_age, y_age = load_agg_data(age=age)
                runs.append((age, child_id_age, X_age, y_age))
            
        for age, current_id, X, y in runs:
            X_cols = X.columns.tolist()
            X_values = X.values
            y_values = y.values.ravel()
            
            # Base models
            models_dict = {
                'ElasticNet': lambda: ElasticNet(random_state=42),
                'ERF': lambda: ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_leaf=5, random_state=42),
                'MLP': lambda: MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.1, max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=42),
                'XGBoost': lambda: XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
            }

            # KAN (standard implementation)
            try:
                class SimpleKANWrapper:
                    def __init__(self, width):
                        # width: [in, hidden1, ..., out]
                        # pykan expects torch tensors
                        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        self.model = KAN(width=width, device=self.device)
                        self.width = width
                        self.formula = None
                    
                    def fit(self, X, y):
                        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
                        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
                        # Minimal training for baseline speed
                        dataset = {
                            'train_input': X_t,
                            'train_label': y_t,
                            'test_input': X_t,
                            'test_label': y_t
                        }
                        self.model.fit(dataset, steps=200, update_grid=False, opt="LBFGS")
                        return self
                    def prune(self):
                        try:
                            self.model.prune()
                            return self
                        except Exception as e:
                            print(f"  Warning: KAN pruning failed: {e}")
                    def symbolize(self):
                        # Symbolize and extract formula
                        try:
                            self.model.auto_symbolic()
                            formulas = self.model.symbolic_formula()
                            if formulas and len(formulas[0]) > 0:
                                self.formula = str(formulas[0][0])
                            return self
                        except Exception as e:
                            print(f"  Warning: KAN symbolic extraction failed: {e}")
                            self.formula = "Failed to extract formula"

                        
                    def predict(self, X):
                        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
                        return self.model(X_t).detach().cpu().numpy().ravel()

                models_dict['KAN'] = lambda: SimpleKANWrapper(width=[X_values.shape[1], 5, 1])
            except Exception as e:
                print(f"Standard KAN failed to import or initialize: {e}")

            for name, model_factory in models_dict.items():
                print(f"  Running 5-fold CV for {name}...")
                all_test_results = []
                all_feature_importances = []
                
                # Setup CV
                if setting == 'longitudinal':
                    # Stratified on age
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    splits = skf.split(X_values, X['age'])
                else:
                    # Simple KFold for age-specific
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    splits = kf.split(X_values)

                try:
                    for fold, (train_index, test_index) in enumerate(splits):
                        X_train, X_test = X_values[train_index], X_values[test_index]
                        y_train, y_test = y_values[train_index], y_values[test_index]
                        id_test = current_id[test_index]
                        age_test = X['age'].values[test_index]
                        
                        # Standardization
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        model = model_factory()
                        model.fit(X_train_scaled, y_train)
                        
                        # Handle regular predictions
                        y_pred = model.predict(X_test_scaled)
                        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
                        
                        # Handle symbolic part for KAN
                        y_pred_sym = None
                        if name == 'KAN':
                            model = model.prune()
                            model = model.symbolize()
                            y_pred_sym = model.predict(X_test_scaled)
                            y_pred_sym = np.nan_to_num(y_pred_sym, nan=0.0, posinf=1e10, neginf=-1e10)
                            
                            if hasattr(model, 'formula') and model.formula:
                                if setting == 'longitudinal':
                                    kan_outdir = f"./results_bmi/baseline_longitudinal/{name.lower()}"
                                else:
                                    kan_outdir = f"./results_bmi/baseline_age/yr{age}_{name.lower()}"
                                os.makedirs(kan_outdir, exist_ok=True)
                                with open(os.path.join(kan_outdir, f'formula_fold{fold}.txt'), 'w') as f:
                                    f.write(model.formula)
                        
                        fold_results_dict = {
                            'id': id_test,
                            'age': age_test,
                            'target_bmi': y_test,
                            'pred_bmi': y_pred
                        }
                        if y_pred_sym is not None:
                            fold_results_dict['pred_bmi_sym'] = y_pred_sym
                            
                        fold_results = pd.DataFrame(fold_results_dict)
                        all_test_results.append(fold_results)
                        
                        # Extract feature importance
                        try:
                            if name == 'ElasticNet':
                                importances = np.abs(model.coef_)
                            elif name == 'ERF':
                                importances = np.abs(model.feature_importances_)
                            elif name == 'XGBoost':
                                importances = np.abs(model.feature_importances_)
                            elif name == 'MLP':
                                # Use permutation importance for MLP
                                r = permutation_importance(model, X_test_scaled, y_test, n_repeats=5, random_state=42)
                                importances = np.abs(r.importances_mean)
                            elif name == 'KAN':
                                # KAN feature_score attribute
                                if hasattr(model.model, 'feature_score'):
                                    importances = np.abs(model.model.feature_score.detach().cpu().numpy())
                                else:
                                    importances = np.zeros(len(X_cols))
                            else:
                                importances = np.zeros(len(X_cols))
                            
                            all_feature_importances.append(importances)
                        except Exception as e:
                            print(f"    Warning: Could not extract importance for {name} fold {fold}: {e}")
                            all_feature_importances.append(np.zeros(len(X_cols)))
                    
                    combined_results = pd.concat(all_test_results, ignore_index=True)
                    
                    # Calculate average feature importance
                    avg_importances = np.mean(all_feature_importances, axis=0)
                    feat_imp_df = pd.DataFrame({
                        'feature': X_cols,
                        'importance': avg_importances
                    }).sort_values(by='importance', ascending=False)
                    
                    if setting == 'longitudinal':
                        outdir = f"./results_bmi/baseline_longitudinal/{name.lower()}"
                    else:
                        outdir = f"./results_bmi/baseline_age/yr{age}_{name.lower()}"
                    
                    save_baseline_results(name, setting, age, outdir, combined_results, feat_imp_df)
                except Exception as e:
                    print(f"  Error running {name}: {e}")

if __name__ == "__main__":
    run_baselines()
