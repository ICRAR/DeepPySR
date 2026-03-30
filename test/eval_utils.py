import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

def save_predictions(outdir, y_true, y_pred, y_prob=None, ids=None, fold=None, y_pred_sym=None, extra_data=None):
    os.makedirs(outdir, exist_ok=True)
    suffix = f"_fold{fold}" if fold is not None else ""
    data = {'y_true': y_true, 'y_pred': y_pred}
    if y_pred_sym is not None:
        data['y_pred_kansym'] = y_pred_sym
    if y_prob is not None:
        if y_prob.ndim > 1:
            for i in range(y_prob.shape[1]):
                data[f'y_prob_{i}'] = y_prob[:, i]
        else:
            data['y_prob'] = y_prob
    if ids is not None:
        data['id'] = ids
    
    if extra_data is not None:
        for k, v in extra_data.items():
            data[k] = v
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(outdir, f"predictions{suffix}.csv"), index=False)

def save_formulas(outdir, formulas, fold=None):
    os.makedirs(outdir, exist_ok=True)
    suffix = f"_fold{fold}" if fold is not None else ""
    df = pd.DataFrame(formulas)
    df.to_csv(os.path.join(outdir, f"formulas{suffix}.csv"), index=False)

def calculate_metrics(y_true, y_pred, y_prob=None, task='regression'):
    # Clean NaNs and Infs for metrics calculation to avoid ValueError in sklearn
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
    if y_prob is not None:
        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
        
    if task == 'regression':
        return {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    else:
        # Check if multiclass
        unique_y = np.unique(y_true)
        is_multiclass = len(unique_y) > 2
        avg = 'macro' if is_multiclass else 'binary'
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=avg, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=avg, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=avg, zero_division=0)
        }
        if y_prob is not None:
            try:
                if is_multiclass:
                    # For multiclass, y_prob should be (n_samples, n_classes)
                    metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = 0.5
        return metrics

def run_cv(model_factory, X, y, ids=None, groups=None, stratify_by=None, task='regression', n_splits=5, random_state=42, outdir=None, scaler=True, extra_data=None):
    if groups is not None:
        if stratify_by is not None:
            skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = skf.split(X, stratify_by, groups=groups)
        else:
            skf = GroupKFold(n_splits=n_splits)
            splits = skf.split(X, y, groups=groups)
    else:
        if task == 'classification':
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = skf.split(X, y)
        else:
            skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = skf.split(X, y)
    
    all_y_true, all_y_pred, all_y_prob = [], [], []
    all_y_pred_sym = []
    all_ids = []
    all_extra_data = {k: [] for k in extra_data.keys()} if extra_data is not None else {}
    fold_metrics = []
    fold_importances = []
    
    X_values = X.values if hasattr(X, 'values') else X
    y_values = y.values if hasattr(y, 'values') else y
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {fold+1}/{n_splits}")
        X_train, X_test = X_values[train_idx], X_values[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        
        if scaler:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        
        model = model_factory()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = None
        if task == 'classification' and hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if y_prob.ndim > 1:
                if y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                # If > 2 classes, keep all columns for roc_auc_score(multi_class='ovr')
            elif y_prob.ndim == 1:
                # If already 1D, assume it is probabilities of the positive class
                pass
    
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
        if hasattr(model, 'symbolize'):
            model.symbolize()
            y_pred_sym = model.predict_symbolic(X_test)
            all_y_pred_sym.extend(y_pred_sym)
    
        if y_prob is not None:
            all_y_prob.extend(y_prob)
        if ids is not None:
            all_ids.extend(ids.iloc[test_idx].values if hasattr(ids, 'iloc') else ids[test_idx])
        
        if extra_data is not None:
            for k, v in extra_data.items():
                all_extra_data[k].extend(v.iloc[test_idx].values if hasattr(v, 'iloc') else v[test_idx])
            
        metrics = calculate_metrics(y_test, y_pred, y_prob if y_prob is not None else None, task=task)
        fold_metrics.append(metrics)
        
        # Save feature importance if available
        if hasattr(model, 'feature_importances_'):
            fold_importances.append(model.feature_importances_)
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute values of coefficients as importance
            # if y is not flattened, coef_ might be 2D
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
            fold_importances.append(coef)
        
        if outdir:
            # Save formula if available
            if hasattr(model, 'symbolize'):
                model.symbolize()
            
            if hasattr(model, 'formula') and model.formula is not None:
                save_formulas(outdir, [{'fold': fold, 'formula': str(model.formula)}], fold=fold)
            elif hasattr(model, 'relationships_'): # DeepPySR
                model.save_relationships(filename=f"relationships_fold{fold}.csv")

    overall_metrics = calculate_metrics(np.array(all_y_true), np.array(all_y_pred), 
                                        np.array(all_y_prob) if all_y_prob else None, task=task)
    
    if outdir:
        save_predictions(outdir, np.array(all_y_true), np.array(all_y_pred), 
                         np.array(all_y_prob) if all_y_prob else None, np.array(all_ids) if all_ids else None,
                         y_pred_sym=np.array(all_y_pred_sym) if all_y_pred_sym else None,
                         extra_data={k: np.array(v) for k, v in all_extra_data.items()})
        pd.DataFrame([overall_metrics]).to_csv(os.path.join(outdir, "overall_metrics.csv"), index=False)
        
        if fold_importances:
            avg_importance = np.mean(fold_importances, axis=0)
            importance_df = pd.DataFrame({
                'feature': X.columns if hasattr(X, 'columns') else [f'x{i}' for i in range(len(avg_importance))],
                'importance': avg_importance
            })
            importance_df.to_csv(os.path.join(outdir, "feature_importance.csv"), index=False)
        
        if all_y_pred_sym:
            overall_metrics_sym = calculate_metrics(np.array(all_y_true), np.array(all_y_pred_sym), task=task)
            pd.DataFrame([overall_metrics_sym]).to_csv(os.path.join(outdir, "overall_metrics_sym.csv"), index=False)
        
    return overall_metrics

def run_nocv(model_factory, X, y, ids=None, task='regression', outdir=None, scaler=True, extra_data=None):
    print(f"  Training on full dataset (no-CV)")
    X_values = X.values if hasattr(X, 'values') else X
    y_values = y.values if hasattr(y, 'values') else y
    
    if scaler:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_values)
    else:
        X_train = X_values
    
    model = model_factory()
    model.fit(X_train, y_values)
    
    y_pred = model.predict(X_train)
    y_prob = None
    if task == 'classification' and hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_train)
        if y_prob.ndim > 1:
            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]
            # If > 2 classes, keep all columns for roc_auc_score(multi_class='ovr')
        elif y_prob.ndim == 1:
            # If already 1D, assume it is probabilities of the positive class
            pass

    y_pred_sym = None
    if hasattr(model, 'symbolize'):
        model.symbolize()
        y_pred_sym = model.predict_symbolic(X_train)

    metrics = calculate_metrics(y_values, y_pred, y_prob if y_prob is not None else None, task=task)
    
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        save_predictions(outdir, y_values, y_pred, 
                         y_prob=y_prob, ids=ids,
                         fold='nocv',
                         y_pred_sym=y_pred_sym,
                         extra_data=extra_data)
        
        pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "overall_metrics.csv"), index=False)
        
        # Save formula if available
        if hasattr(model, 'formula') and model.formula is not None:
            save_formulas(outdir, [{'fold': 'nocv', 'formula': str(model.formula)}], fold='nocv')
        # elif hasattr(model, 'relationships_'): # DeepPySR
        #     model.save_relationships(filename=f"relationships_nocv.csv")

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': X.columns if hasattr(X, 'columns') else [f'x{i}' for i in range(len(importance))],
                'importance': importance
            })
            importance_df.to_csv(os.path.join(outdir, "feature_importance.csv"), index=False)
            
        if y_pred_sym is not None:
            metrics_sym = calculate_metrics(y_values, y_pred_sym, task=task)
            pd.DataFrame([metrics_sym]).to_csv(os.path.join(outdir, "overall_metrics_sym.csv"), index=False)

    return metrics

def aggregate_results(base_dir, task='regression'):
    results = []
    for root, dirs, files in os.walk(base_dir):
        if 'overall_metrics.csv' in files:
            metrics_df = pd.read_csv(os.path.join(root, 'overall_metrics.csv'))
            metrics = metrics_df.iloc[0].to_dict()
            metrics['path'] = root
            # Use relative path as model name for better identification
            metrics['model'] = os.path.relpath(root, base_dir)
            results.append(metrics)
        
        if 'overall_metrics_sym.csv' in files:
            metrics_df = pd.read_csv(os.path.join(root, 'overall_metrics_sym.csv'))
            metrics = metrics_df.iloc[0].to_dict()
            metrics['path'] = root
            metrics['model'] = os.path.relpath(root, base_dir) + "_sym"
            results.append(metrics)
    
    if not results:
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    # df.to_csv(os.path.join(base_dir, "aggregated_results.csv"), index=False)
    
    # # Find best models
    # if task == 'regression':
    #     # Sort by r2 descending
    #     best_df = df.sort_values('r2', ascending=False)
    #     best_df.to_csv(os.path.join(base_dir, "best_models_ranked.csv"), index=False)
    # else:
    #     # Sort by f1 descending
    #     best_df = df.sort_values('f1', ascending=False)
    #     best_df.to_csv(os.path.join(base_dir, "best_models_ranked.csv"), index=False)
        
    return df
