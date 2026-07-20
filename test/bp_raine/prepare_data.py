"""Pre-build and cache X/y data for every (target, age, test-type) combo used
by the bp_raine test scripts, for both feateng=False and feateng=True.

Caching is keyed on (test-type, target, age) only (see data_utils.py) --
feateng is applied on top of the cached raw data, so this mainly warms the
on-disk cache once per combo and then sanity-checks the feateng path.
Combos that are already cached on disk are skipped automatically inside
each load_fn.
"""
import sys
import traceback

from data_utils import (
    load_data_PGS_only,
    load_data_keepto5,
    load_data_PGSto5,
    load_data_recent,
    _BP_AGES,
    _BP_TARGETS,
)

_LOAD_FN = {
    'PGS':    load_data_PGS_only,
    'to5':    load_data_keepto5,
    'PGSto5': load_data_PGSto5,
    'recent': load_data_recent,
}


def main():
    combos = [
        (test_name, target, age)
        for target in _BP_TARGETS
        for age in _BP_AGES
        for test_name in _LOAD_FN
    ]
    total = len(combos)
    failures = []

    for i, (test_name, target, age) in enumerate(combos, 1):
        load_fn = _LOAD_FN[test_name]
        for feateng in (False, True):
            label = f"[{i}/{total}] test={test_name} target={target} age={age} feateng={feateng}"
            try:
                ids, X, y = load_fn(target, age, feateng=feateng)
                print(f"{label} -> OK n={len(X)} features={X.shape[1]}")
            except Exception as e:
                print(f"{label} -> FAILED: {e}")
                traceback.print_exc()
                failures.append((test_name, target, age, feateng, str(e)))

    print(f"\nDone. {total * 2 - len(failures)}/{total * 2} combos succeeded.")
    if failures:
        print(f"\n{len(failures)} failures:")
        for test_name, target, age, feateng, err in failures:
            print(f"  test={test_name} target={target} age={age} feateng={feateng}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()