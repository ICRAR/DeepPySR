"""Standalone worker: computes MLP SHAP feature importance for one
(input_type, target, age) combo and writes the result dict to a JSON file.

Run in its own subprocess by analysis_lipids.py's
_lipids_mlp_shap_importance_subprocess -- importing model_utils (and
therefore pysr/deeppysr/juliacall) inside the long-running analysis_lipids.py
process segfaults once enough prior matplotlib/numpy work has accumulated
there (a Julia-embedding fragility -- see bp_raine/_mlp_shap_worker.py and
analysis_lipids.py's _lipids_mlp_shap_importance_subprocess docstring for the
full explanation; the same fragility hit lipids' in-process SHAP call and
killed the whole analysis run before this worker existed). Every invocation
of this script is a fresh process, matching the conditions under which the
same import + SHAP computation reliably succeeds.
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
    parser.add_argument('--input_type', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--age', type=int, required=True)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_explain', type=int, default=60)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    import analysis_lipids as al

    _, X, y = al.INPUT_TYPE_LOADERS[args.input_type](args.target, args.age)
    imp = al._lipids_mlp_shap_importance(X, y, task='regression',
                                          random_state=args.random_state,
                                          n_explain=args.n_explain)

    with open(args.out, 'w') as f:
        json.dump(imp, f)


if __name__ == "__main__":
    main()
