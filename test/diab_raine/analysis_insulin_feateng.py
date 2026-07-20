"""Regression analysis for insulin-prediction feature-set variants
(PGS, to8, PGSto8, recent) using the *feature-engineered* data
(longitudinal first/second-derivative features added, see
data_utils._add_longitudinal_features) and its results under
results_insulin_feateng/results_insulin_df_*.

This mirrors analysis_insulin_variant.py exactly, except:
  - load functions are called with feateng=True so the correct
    (feature-engineered) cached data/columns line up with the formulas.
  - VARIANTS point at the "results_insulin_df_*" subfolders instead of
    "results_insulin_*".
"""
import os
import sys
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))
sys.path.append(current_dir)

import analysis_insulin_variant as base
from data_utils import (
    load_data_PGS_only, load_data_keepto8, load_data_PGSto8, load_data_recent,
)

base.VARIANTS = {
    'PGS':    (partial(load_data_PGS_only, feateng=True),  'results_insulin_df_PGS'),
    'to8':    (partial(load_data_keepto8, feateng=True),   'results_insulin_df_to8'),
    'PGSto8': (partial(load_data_PGSto8, feateng=True),    'results_insulin_df_PGSto8'),
    'recent': (partial(load_data_recent, feateng=True),    'results_insulin_df_recent'),
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_base', type=str, default='results_insulin_feateng',
                        help="Base results folder to analyse, relative to this script's directory.")
    parser.add_argument('--variant', type=str, default=None, choices=list(base.VARIANTS.keys()),
                        help="Which variant to analyse. Default: all four.")
    parser.add_argument('--skip_combined', action='store_true',
                        help="Skip the cross-variant combined comparison plot.")
    args = parser.parse_args()

    base.RESULTS_BASE_DIR = os.path.join(current_dir, args.results_base)

    names = [args.variant] if args.variant else list(base.VARIANTS.keys())
    for name in names:
        print("\n" + "=" * 60)
        print(f"ANALYSIS: {name}")
        print("=" * 60)
        base.run_variant(name)

    if not args.variant and not args.skip_combined:
        print("\n" + "=" * 60)
        print("COMBINED ANALYSIS")
        print("=" * 60)
        base.run_combined()