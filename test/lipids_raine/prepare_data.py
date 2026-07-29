"""Pre-build and cache X/y data for every (target, age, test-type) combo used
by the lipids_raine test scripts.

Caching is keyed on (test-type, target, age) (see data_utils.py). Combos
that are already cached on disk are skipped automatically inside each
load_fn.
"""
import sys
import traceback

from data_utils import (
    load_data_PGS_only,
    load_data_keepto8,
    load_data_PGSto8,
    load_data_recent,
    load_data_nblood,
    _LIPIDS_AGES,
    _LIPID_TARGETS,
)

_LOAD_FN = {
    'PGS':    load_data_PGS_only,
    'to8':    load_data_keepto8,
    'PGSto8': load_data_PGSto8,
    'recent': load_data_recent,
    'nblood': load_data_nblood,
}


def main():
    combos = [
        (test_name, target, age)
        for target in _LIPID_TARGETS
        for age in _LIPIDS_AGES
        for test_name in _LOAD_FN
    ]
    total = len(combos)
    failures = []

    for i, (test_name, target, age) in enumerate(combos, 1):
        load_fn = _LOAD_FN[test_name]
        label = f"[{i}/{total}] test={test_name} target={target} age={age}"
        try:
            ids, X, y = load_fn(target, age)
            print(f"{label} -> OK n={len(X)} features={X.shape[1]}")
        except Exception as e:
            print(f"{label} -> FAILED: {e}")
            traceback.print_exc()
            failures.append((test_name, target, age, str(e)))

    print(f"\nDone. {total - len(failures)}/{total} combos succeeded.")
    if failures:
        print(f"\n{len(failures)} failures:")
        for test_name, target, age, err in failures:
            print(f"  test={test_name} target={target} age={age}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()