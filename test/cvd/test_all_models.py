import os
import numpy as np
import pandas as pd
import sympy
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

from DeepPySR.regressor import DeepPySRRegressor
from utils import save_metrics,KANClassifierWrapper

# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# --- DeepPySR Config ---
sympy_cond = lambda x, y: sympy.Piecewise((y, x > 0), (0, True))

base_pysr_kwargs = {
    "parallelism": "multithreading",
    "maxsize": 30,
    "binary_operators": ["+", "*", "/", "-", "cond(x,y) = x > 0 ? y : y*0"],
    "extra_sympy_mappings": {'cond': sympy_cond},
    "unary_operators": ["exp", "log"],
    "parsimony": 0.001,
    "populations": 20,
    "population_size": 100,
    "ncycles_per_iteration": 200,
    "adaptive_parsimony_scaling": 50.0,
    "verbosity": 1,
    "denoise": True,
    "turbo": True,
    "procs": max(1, (os.cpu_count() or 2) - 1),
}

pypysr_kwargs = base_pysr_kwargs.copy()
pypysr_kwargs.update({
    "variable_prune_start": 50,
    "variable_prune_ramp": 80,
    "variable_prune_max": 0.7,
})

# --- Data Loading ---
def load_cvd_data():
    file_path = os.path.join(os.path.dirname(__file__), '../../test_data/Health/Cardiovascular_Disease_Dataset.csv')
    df = pd.read_csv(file_path)
    # Return patientid, X, y
    patient_ids = df['patientid']
    X = df.drop(columns=['patientid', 'target'])
    y = df['target'].astype(np.float64)
    return patient_ids, X, y



# --- Main Execution ---
def run_all_cvd_models():
    patient_ids, X, y = load_cvd_data()
    X_values = X.values
    y_values = y.values
    patient_ids_values = patient_ids.values
    
    out_root = "./results_cvd/"
    os.makedirs(out_root, exist_ok=True)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define models
    models_dict = {
        'LogisticRegression': lambda _: LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        'RandomForest': lambda _: RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42),
        'ExtraTrees': lambda _: ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42),
        'XGBoost': lambda _: XGBClassifier(n_estimators=100, max_depth=3, reg_lambda=1, reg_alpha=0.1, subsample=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'MLP': lambda _: MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.1, max_iter=500, random_state=42),
        'KAN': None,
        'DeepPySR': None,
        'PySR': None,
    }

    summary_results = []
    
    # Define four argument configurations for DeepPySR
    arg_configs = {
        # Standard SR (no pruning, no adaptive parsimony scaling)
        "stdsr": {
            "adaptive_parsimony_scaling": 0.0,
            "variable_prune_max": 0.0,
            "variable_prune_start": 0,
            "variable_prune_ramp": 0,
        },
        # SR with pruning only
        "srprn": {
            "adaptive_parsimony_scaling": 0.0,
            "variable_prune_start": 50,
            "variable_prune_ramp": 150,
            "variable_prune_max": 0.7,
        },
        # SR with parsimony scaling only
        "srpsm": {
            "adaptive_parsimony_scaling": 1040.0,
            "variable_prune_max": 0.0,
            "variable_prune_start": 0,
            "variable_prune_ramp": 0,
        },
        # Full SR (use current/default arguments)
        "fullsr": {
            "adaptive_parsimony_scaling": 1040.0,
            "variable_prune_start": 50,
            "variable_prune_ramp": 150,
            "variable_prune_max": 0.7,
        },
    }

    r2w_list = [1,1.5,2]
    l_list = [0.001,0.005,0.01]
    
    kan_lamb_list = [0.0, 0.01, 0.1, 0.5]
    kan_lamb_l1_list = [0.0, 0.1, 1.0]
    kan_lamb_entropy_list = [0.0, 0.5, 2.0]
    kan_hidden_list = [2, 5]
    kan_steps = 200

    # Models that do NOT need scaling
    no_scaler_models = ['DeepPySR', 'PySR', 'KAN']

    for model_name, model_fn in models_dict.items():
        if model_name in ['DeepPySR', 'PySR', 'KAN']:
            if model_name == 'KAN':
                for lamb in kan_lamb_list:
                    for l1 in kan_lamb_l1_list:
                        for entropy in kan_lamb_entropy_list:
                            for hidden in kan_hidden_list:
                                full_model_name = f"KAN_lamb{lamb}_l1{l1}_ent{entropy}_hidden{hidden}"
                                print(f"\n--- Evaluating {full_model_name} ---")
                                all_y_true, all_y_pred, all_y_prob = [], [], []
                                all_patient_ids = []
                                
                                model_out_root = os.path.join(out_root, model_name.lower(), f"lamb{lamb}_l1{l1}_ent{entropy}_hidden{hidden}")
                                os.makedirs(model_out_root, exist_ok=True)
                                
                                # Check if already done
                                if os.path.exists(os.path.join(model_out_root, 'predictions.csv')) and \
                                   os.path.exists(os.path.join(model_out_root, 'nocv', 'nocv_formula.csv')) and \
                                   os.path.exists(os.path.join(model_out_root, 'nocv', 'feature_importance.csv')):
                                    print(f"Skipping {full_model_name} as all results exist.")
                                    continue

                                if not os.path.exists(os.path.join(model_out_root, 'predictions.csv')):
                                    for fold, (train_idx, test_idx) in enumerate(skf.split(X_values, y_values)):
                                        print(f"  Fold {fold+1}/5...")
                                        X_train, X_test = X_values[train_idx], X_values[test_idx]
                                        y_train, y_test = y_values[train_idx], y_values[test_idx]
                                        pids_test = patient_ids_values[test_idx]
                                        
                                        # KAN doesn't use scaling in no_scaler_models
                                        model = KANClassifierWrapper(X_train.shape[1], hidden_dim=hidden, lamb=lamb, lamb_l1=l1, lamb_entropy=entropy, steps=kan_steps)
                                        model.fit(X_train, y_train)
                                
                                        y_pred = model.predict(X_test)
                                        y_prob = model.predict_proba(X_test)[:, 1]
                                        
                                        all_y_true.extend(y_test)
                                        all_y_pred.extend(y_pred)
                                        all_y_prob.extend(y_prob)
                                        all_patient_ids.extend(pids_test)
                            
                                    # Save initial predictions.csv
                                    save_metrics(full_model_name, model_out_root, 
                                                 np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob),
                                                 np.array(all_patient_ids))

                                # --- Train on full dataset (No-CV) for KAN/KANSYM ---
                                print(f"  Training (No-CV) for {full_model_name}...")
                                nocv_out_root = os.path.join(model_out_root, "nocv")
                                
                                if not os.path.exists(os.path.join(nocv_out_root, 'nocv_formula.csv')):
                                    os.makedirs(nocv_out_root, exist_ok=True)
                                    
                                    model_nocv = KANClassifierWrapper(X_values.shape[1], hidden_dim=hidden, lamb=lamb, lamb_l1=l1, lamb_entropy=entropy, steps=kan_steps)
                                    model_nocv.fit(X_values, y_values)
                                    
                                    # Feature importance from No-CV model
                                    fi = model_nocv.feature_importance()
                                    fi_df = pd.DataFrame({
                                        'feature': X.columns.tolist(),
                                        'importance': fi
                                    }).sort_values('importance', ascending=False)
                                    fi_df.to_csv(os.path.join(nocv_out_root, 'feature_importance.csv'), index=False)

                                    # Symbolic KAN results
                                    model_nocv.symbolize()
                                    if model_nocv.formula:
                                        # Save nocv_formula.csv
                                        formula_df = pd.DataFrame([{'formula': str(model_nocv.formula)}])
                                        formula_df.to_csv(os.path.join(nocv_out_root, 'nocv_formula.csv'), index=False)
                                        
                                        # Update predictions.csv in the main folder to include y_pred_kansym
                                        preds_df = pd.read_csv(os.path.join(model_out_root, 'predictions.csv'))
                                        
                                        # Apply No-CV formula to all data to get y_pred_kansym
                                        y_pred_kansym_all = model_nocv.predict_sym(X_values)
                                        y_prob_kansym_all = model_nocv.predict_proba_sym(X_values)[:, 1]
                                        
                                        # Create a mapping from patientid to kansym prediction
                                        kansym_map = dict(zip(patient_ids_values, y_pred_kansym_all))
                                        kansym_prob_map = dict(zip(patient_ids_values, y_prob_kansym_all))
                                        preds_df['y_pred_kansym'] = preds_df['patientid'].map(kansym_map)
                                        preds_df['y_prob_kansym'] = preds_df['patientid'].map(kansym_prob_map)
                                        
                                        # Rename columns to match requirement: y_pred_kan, y_prob_kan
                                        preds_df = preds_df.rename(columns={'y_pred': 'y_pred_kan', 'y_prob': 'y_prob_kan'})
                                        
                                        # Reorder columns as requested: patientid, y_true, y_pred_kan, y_prob_kan, y_pred_kansym, y_prob_kansym
                                        # We include y_prob_kansym as well to compute AUC in analysis
                                        preds_df = preds_df[['patientid', 'y_true', 'y_pred_kan', 'y_prob_kan', 'y_pred_kansym', 'y_prob_kansym']]
                                        preds_df.to_csv(os.path.join(model_out_root, 'predictions.csv'), index=False)
                continue

            for r2w in r2w_list:
                for l in l_list:
                    if model_name == 'DeepPySR':
                        for cfg_name, cfg_overrides in arg_configs.items():
                            parsimony = pypysr_kwargs["parsimony"]
                            population = pypysr_kwargs["populations"]
                            pop_size = pypysr_kwargs["population_size"]
                            parsimony_scaling = cfg_overrides.get("adaptive_parsimony_scaling", pypysr_kwargs.get("adaptive_parsimony_scaling"))
                            prune_start = cfg_overrides.get("variable_prune_start", pypysr_kwargs.get("variable_prune_start"))
                            prune_ramp = cfg_overrides.get("variable_prune_ramp", pypysr_kwargs.get("variable_prune_ramp"))
                            prune_max = cfg_overrides.get("variable_prune_max", pypysr_kwargs.get("variable_prune_max"))

                            full_model_name = (f"cfg{cfg_name}_par{parsimony}_pop{population}_popsz{pop_size}_"
                                              f"scl{parsimony_scaling}_prnst{prune_start}_ramp{prune_ramp}_max{prune_max}_"
                                              f"r2w{r2w}_lambda{l}")
                            print(f"\n--- Evaluating {full_model_name} ---")
                            
                            model_out_root = os.path.join(out_root, model_name.lower(), full_model_name)
                            os.makedirs(model_out_root, exist_ok=True)
                            
                            # Check if already done
                            if os.path.exists(os.path.join(model_out_root, 'predictions.csv')) and \
                               os.path.exists(os.path.join(model_out_root, 'nocv', 'predictions.csv')):
                                print(f"Skipping {full_model_name} as predictions and nocv predictions exist.")
                                continue

                            if not os.path.exists(os.path.join(model_out_root, 'predictions.csv')):
                                all_y_true, all_y_pred, all_y_prob = [], [], []
                                all_patient_ids = []
                                equations = []
                                for fold, (train_idx, test_idx) in enumerate(skf.split(X_values, y_values)):
                                    print(f"  Fold {fold+1}/5...")
                                    X_train, X_test = X_values[train_idx], X_values[test_idx]
                                    y_train, y_test = y_values[train_idx], y_values[test_idx]
                                    pids_test = patient_ids_values[test_idx]
                                    
                                    fold_outdir = os.path.join(model_out_root, f"fold_{fold}")
                                    os.makedirs(fold_outdir, exist_ok=True)
                                    
                                    provider = 'pypysr'
                                    current_kwargs = pypysr_kwargs.copy()
                                    current_kwargs.update(cfg_overrides)
                                    
                                    model = DeepPySRRegressor(
                                        max_layers=1,
                                        output_dir=fold_outdir,
                                        stopping_score=0.001,
                                        model_provider=provider,
                                        pareto_lambda=l,
                                        pareto_r2_weight=r2w,
                                        **current_kwargs,
                                    )
                                    
                                    model.fit(X_train, y_train)
                                    y_pred_raw = model.predict(X_test)
                                    y_pred = (y_pred_raw > 0.5).astype(int)
                                    
                                    all_y_true.extend(y_test)
                                    all_y_pred.extend(y_pred)
                                    all_y_prob.extend(np.clip(y_pred_raw, 0, 1))
                                    all_patient_ids.extend(pids_test)
                                    
                                    if model.relationships_:
                                        equations.append(model.relationships_[0]['formula'])
                                    model.save_relationships()
                                    # model.plot()

                                save_metrics(full_model_name, model_out_root, 
                                                  np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob),
                                                  np.array(all_patient_ids))

                            # --- Train on full dataset (No-CV) for symbolic models (DeepPySR) ---
                            print(f"  Training (No-CV) for {full_model_name}...")
                            nocv_out_root = os.path.join(model_out_root, "nocv")
                            if os.path.exists(os.path.join(nocv_out_root, 'predictions.csv')):
                                print(f"  Skipping (No-CV) as predictions.csv exists.")
                            else:
                                os.makedirs(nocv_out_root, exist_ok=True)
                                provider = 'pypysr'
                                current_kwargs = pypysr_kwargs.copy()
                                current_kwargs.update(cfg_overrides)
                                model_nocv = DeepPySRRegressor(
                                    max_layers=1,
                                    output_dir=nocv_out_root,
                                    stopping_score=0.001,
                                    model_provider=provider,
                                    pareto_lambda=l,
                                    pareto_r2_weight=r2w,
                                    **current_kwargs,
                                )
                                model_nocv.fit(X_values, y_values)
                                y_pred_raw_nocv = model_nocv.predict(X_values)
                                y_pred_nocv = (y_pred_raw_nocv > 0.5).astype(int)
                                save_metrics(f"{full_model_name}_nocv", nocv_out_root, 
                                             y_values, y_pred_nocv, np.clip(y_pred_raw_nocv, 0, 1),
                                             patient_ids_values)
                                model_nocv.save_relationships()
                                # model_nocv.plot()
                    else: # PySR
                        full_model_name = f"{model_name}_r2w{r2w}_l{l}"
                        print(f"\n--- Evaluating {full_model_name} ---")
                        all_y_true, all_y_pred, all_y_prob = [], [], []
                        all_patient_ids = []
                        equations = []
                        
                        model_out_root = os.path.join(out_root, 'pysr', f"r2w{r2w}_l{l}")
                        os.makedirs(model_out_root, exist_ok=True)
                        
                        # Check if already done
                        if os.path.exists(os.path.join(model_out_root, 'predictions.csv')) and \
                           os.path.exists(os.path.join(model_out_root, 'nocv', 'predictions.csv')):
                            print(f"Skipping {full_model_name} as predictions and nocv predictions exist.")
                            continue

                        if not os.path.exists(os.path.join(model_out_root, 'predictions.csv')):
                            for fold, (train_idx, test_idx) in enumerate(skf.split(X_values, y_values)):
                                print(f"  Fold {fold+1}/5...")
                                X_train, X_test = X_values[train_idx], X_values[test_idx]
                                y_train, y_test = y_values[train_idx], y_values[test_idx]
                                pids_test = patient_ids_values[test_idx]
                                
                                fold_outdir = os.path.join(model_out_root, f"fold_{fold}")
                                os.makedirs(fold_outdir, exist_ok=True)

                                model = DeepPySRRegressor(
                                    max_layers=1,
                                    output_dir=fold_outdir,
                                    stopping_score=0.001,
                                    model_provider='pysr',
                                    pareto_lambda=l,
                                    pareto_r2_weight=r2w,
                                    **base_pysr_kwargs,
                                )
                                
                                model.fit(X_train, y_train)
                                y_pred_raw = model.predict(X_test)
                                y_pred = (y_pred_raw > 0.5).astype(int)
                                
                                all_y_true.extend(y_test)
                                all_y_pred.extend(y_pred)
                                all_y_prob.extend(np.clip(y_pred_raw, 0, 1))
                                all_patient_ids.extend(pids_test)
                                
                                equations.append(str(model.sympy()))
                                model.save_relationships()

                            save_metrics(full_model_name, model_out_root, 
                                              np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob),
                                              np.array(all_patient_ids))

                        # --- Train on full dataset (No-CV) for symbolic models (PySR) ---
                        print(f"  Training (No-CV) for {full_model_name}...")
                        nocv_out_root = os.path.join(model_out_root, "nocv")
                        if os.path.exists(os.path.join(nocv_out_root, 'predictions.csv')):
                            print(f"  Skipping (No-CV) as predictions.csv exists.")
                        else:
                            os.makedirs(nocv_out_root, exist_ok=True)
                            model_nocv = model = DeepPySRRegressor(
                                max_layers=1,
                                output_dir=nocv_out_root,
                                stopping_score=0.001,
                                model_provider='pysr',
                                pareto_lambda=l,
                                pareto_r2_weight=r2w,
                                **base_pysr_kwargs,
                            )
                            model_nocv.fit(X_values, y_values)
                            y_pred_raw_nocv = model_nocv.predict(X_values)
                            y_pred_nocv = (y_pred_raw_nocv > 0.5).astype(int)
                            save_metrics(f"{full_model_name}_nocv", nocv_out_root, 
                                         y_values, y_pred_nocv, np.clip(y_pred_raw_nocv, 0, 1),
                                         patient_ids_values)
                            model_nocv.save_relationships()
            continue

        print(f"\n--- Evaluating {model_name} ---")
        all_y_true, all_y_pred, all_y_prob = [], [], []
        all_y_pred_sym, all_y_prob_sym = [], []
        all_patient_ids = []
        equations = []
        
        model_out_path = os.path.join(out_root, model_name.lower())
        if os.path.exists(os.path.join(model_out_path, 'predictions.csv')):
            print(f"Skipping {model_name} as predictions.csv exists.")
            if model_name == 'KAN':
                # Also skip KANSYM if it exists
                kansym_path = os.path.join(out_root, "kansym")
                if os.path.exists(os.path.join(kansym_path, 'predictions.csv')):
                    pass # We just skip
            continue

        fold_feature_importances = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_values, y_values)):
            X_train, X_test = X_values[train_idx], X_values[test_idx]
            y_train, y_test = y_values[train_idx], y_values[test_idx]
            pids_test = patient_ids_values[test_idx]
            
            if model_name not in no_scaler_models:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled, X_test_scaled = X_train, X_test
            
            model = model_fn(X_train.shape[1])
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = y_pred
                
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)
            all_patient_ids.extend(pids_test)

            # Feature Importance for this fold
            if hasattr(model, 'feature_importances_'):
                fold_feature_importances.append(model.feature_importances_)
            elif hasattr(model, 'coef_'):
                fold_feature_importances.append(np.abs(model.coef_[0]))

            if model_name == 'KAN':
                model.symbolize()
                if model.formula:
                    equations.append(model.formula)
                    # Save relationships for KAN fold
                    fold_outdir = os.path.join(out_root, model_name.lower(), f"fold_{fold}")
                    os.makedirs(fold_outdir, exist_ok=True)
                    # We can use save_metrics to just save predictions or save relationships manually
                    # But the requirement says "save the relationships.csv as well for each fold"
                    # KAN formula saving
                    rel_df = pd.DataFrame([{'fold': fold, 'formula': str(model.formula)}])
                    rel_df.to_csv(os.path.join(fold_outdir, 'relationships.csv'), index=False)

                    y_pred_sym = model.predict_sym(X_test_scaled)
                    y_prob_sym = model.predict_proba_sym(X_test_scaled)[:, 1]
                    all_y_pred_sym.extend(y_pred_sym)
                    all_y_prob_sym.extend(y_prob_sym)
                    
                    # For KANSYM as well
                    kansym_fold_outdir = os.path.join(out_root, "kansym", f"fold_{fold}")
                    os.makedirs(kansym_fold_outdir, exist_ok=True)
                    rel_df.to_csv(os.path.join(kansym_fold_outdir, 'relationships.csv'), index=False)
            
        # Feature Importance for traditional models - average across folds
        avg_fi = None
        if fold_feature_importances:
            avg_fi = np.mean(fold_feature_importances, axis=0)
        
        save_metrics(model_name, os.path.join(out_root, model_name.lower()), 
                          np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob),
                          np.array(all_patient_ids),
                          feature_importances=avg_fi,
                          feature_names=X.columns.tolist())

        if model_name == 'KAN' and all_y_pred_sym:
                # This part is probably no longer reached if KAN is handled in the grid search block above
                # but we leave it for consistency if we ever add KAN back to models_dict with a lambda
                pass
            
    print(f"\nUnified evaluation completed. Results saved to {out_root}")

if __name__ == "__main__":
    run_all_cvd_models()
