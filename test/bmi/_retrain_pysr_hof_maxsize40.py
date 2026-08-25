"""Genuine hall_of_fame capture for the winning longitudinal PySR config
(aps=1.0), all 5 folds, with maxsize EXPLICITLY pinned to 40 to match the
budget the original May 2026 production DeepPySR/PySR runs used (before
model_utils.get_pysr_base_kwargs's maxsize default drifted 40->50 in commit
c5c961a, 2026-07-04) and what the Methods section states. Uses
parallelism="multiprocessing" (the production default, "multithreading",
deadlocked in a Julia GC safepoint wait after fold 0 on the first attempt
at this capture). Writes into an isolated directory; merge into production
only after confirming all 5 folds complete cleanly.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from sklearn.model_selection import StratifiedGroupKFold
from pysr import PySRRegressor
from model_utils import get_pysr_base_kwargs
from bmi_utils import load_bmi_agg_data

def main():
    ids, X, y = load_bmi_agg_data()
    print(f"Longitudinal data shape: {X.shape}")

    pysr_base_kwargs = get_pysr_base_kwargs()
    pysr_base_kwargs["parallelism"] = "multiprocessing"
    pysr_base_kwargs["procs"] = 4
    pysr_base_kwargs["maxsize"] = 40
    nit = pysr_base_kwargs.get('niterations', 100)
    pop = pysr_base_kwargs.get('populations', 30)
    sz = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f"nit{nit}_pop{pop}_sz{sz}"
    full_name = f"pysr_{param_suffix}_aps1.0_grid"

    outdir = os.path.join(current_dir, "results_bmi_all", "longitudinal", "pysr_hof_capture_ms40", full_name)
    os.makedirs(outdir, exist_ok=True)

    X_values = X.values
    y_values = y.values if hasattr(y, 'values') else y
    stratify_by = X['age']

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, stratify_by, groups=ids))

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"Fold {fold_idx+1}/5 ...")
        X_train, y_train = X_values[train_idx], y_values[train_idx]

        model = PySRRegressor(**pysr_base_kwargs, adaptive_parsimony_scaling=1.0)
        model.output_directory = os.path.join(outdir, 'pysr_outputs', 'y')
        model.fit(X_train, y_train)
        print(f"Fold {fold_idx+1}/5 done.")

    print("Done.")

if __name__ == "__main__":
    main()
