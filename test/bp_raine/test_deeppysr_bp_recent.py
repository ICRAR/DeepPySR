import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from deeppysr import DeepPySR
from model_utils import get_deeppysr_configs, get_pysr_base_kwargs
from eval_utils import run_cv, aggregate_results
from data_utils import load_data_recent, _BP_AGES, _BP_TARGETS

import argparse

_N_TOP = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='sys_bp',
                        choices=_BP_TARGETS)
    parser.add_argument('--age', type=int, default=17,
                        choices=_BP_AGES)
    parser.add_argument('--vps', type=int, default=25)
    parser.add_argument('--feateng', dest='feateng', action='store_true', default=True,
                        help='Add first-difference/second-derivative longitudinal features')
    parser.add_argument('--no_feateng', dest='feateng', action='store_false',
                        help='Disable longitudinal feature engineering')
    args = parser.parse_args()

    results_dir = "results_bp/results_bp_df_recent" if args.feateng else "results_bp/results_bp_recent"
    out_root = os.path.join(current_dir, results_dir)
    os.makedirs(out_root, exist_ok=True)

    deeppysr_configs = get_deeppysr_configs()
    if args.vps is not None:
        deeppysr_configs = {k: v for k, v in deeppysr_configs.items()
                            if f"vps{args.vps}_" in k}

    pysr_base_kwargs = get_pysr_base_kwargs()
    nit = pysr_base_kwargs.get('niterations', 100)
    pop = pysr_base_kwargs.get('populations', 30)
    sz  = pysr_base_kwargs.get('population_size', 200)
    param_suffix = f"nit{nit}_pop{pop}_sz{sz}"

    r2w_list = [1, 1.5, 2]
    lambda_list = [0.001, 0.005, 0.01]

    print(f"\nLoading recent data for target={args.target}, age={args.age}, feateng={args.feateng}...")
    ids, X, y = load_data_recent(args.target, args.age, feateng=args.feateng)
    y = y.rename(args.target)

    run_name = f"age_{args.age}_{args.target}"
    print(f"\n--- Run: {run_name} ---")
    print(f"  n={len(X)}, features={X.shape[1]}, target={y.name}")

    run_out = os.path.join(out_root, run_name)
    os.makedirs(run_out, exist_ok=True)

    cv_kwargs = {
        'ids': ids,
        'task': 'regression',
        'n_splits': 5,
        'random_state': 42,
        'extra_data': X[['age']] if 'age' in X.columns else None,
    }

    # DeepPySR models — two runs per config: all features and top-100
    print("\nEvaluating DeepPySR Models...")
    for cfg_name, cfg_overrides in deeppysr_configs.items():
        parts = cfg_name.split('_', 1)
        setting_prefix = parts[0]
        params_part = parts[1] if len(parts) > 1 else ""
        full_name = f"{setting_prefix}_{param_suffix}_{params_part}_grid"

        for subfolder, fs_kwargs in [
            ("all_features", {}),
            (f"top{_N_TOP}", {"feature_selection": True, "n_features_to_select": _N_TOP}),
        ]:
            deeppysr_out = os.path.join(run_out, "deeppysr", full_name) if subfolder == "all_features" \
                else os.path.join(run_out, "deeppysr", full_name, subfolder)
            if os.path.exists(os.path.join(deeppysr_out, "overall_metrics.csv")):
                print(f"  Skipping {full_name}/{subfolder} (results exist)")
                continue
            print(f"  {full_name}/{subfolder}...")

            def deeppysr_factory(co=cfg_overrides, out=deeppysr_out):
                kwargs = pysr_base_kwargs.copy()
                kwargs.update(co)
                return DeepPySR(
                    max_layers=1,
                    output_dir=out,
                    pareto_r2_weight=r2w_list,
                    pareto_lambda=lambda_list,
                    **kwargs,
                )

            run_cv(deeppysr_factory, X, y, outdir=deeppysr_out, scaler=False, **cv_kwargs, **fs_kwargs)

    print(f"\nAggregating results for {run_name}...")
    aggregate_results(run_out, task='regression')


if __name__ == "__main__":
    main()