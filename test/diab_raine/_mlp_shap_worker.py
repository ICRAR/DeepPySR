"""Standalone worker: computes MLP SHAP feature importance for one
(variant, age) combo and writes the result dict to a JSON file.

Run in its own subprocess by analysis_insulin.py's
_insulin_mlp_shap_importance_subprocess -- importing model_utils (and
therefore pysr/deeppysr/juliacall) inside a long-running analysis process
that has already done heavy matplotlib/numpy work reliably segfaults (a
Julia-embedding fragility -- see the identical fix and its docstring in
test/bp_raine/analysis_bp.py's _bp_mlp_shap_importance_subprocess, where
this was root-caused via isolated repro). Every invocation of this script
is a fresh process, matching the conditions under which the same import +
SHAP computation reliably succeeds.
"""
import argparse
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--age', type=int, required=True)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_explain', type=int, default=60)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    import analysis_insulin as ai

    load_fn, _ = ai.VARIANTS[args.variant]
    _, X, y = ai._load_age(load_fn, args.age)
    imp = ai._insulin_mlp_shap_importance(X, y, task='regression',
                                           random_state=args.random_state,
                                           n_explain=args.n_explain)

    with open(args.out, 'w') as f:
        json.dump(imp, f)


if __name__ == "__main__":
    main()
