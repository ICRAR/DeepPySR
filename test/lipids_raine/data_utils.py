import re
import sys
from functools import lru_cache
from itertools import combinations
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import RFE
from sklearn.impute import IterativeImputer, SimpleImputer

# import categorical definitions from bmiforecast_utils
_bmiforecast_dir = str(Path(__file__).parents[1] / "bmiforecast")
if _bmiforecast_dir not in sys.path:
    sys.path.insert(0, _bmiforecast_dir)
from bmiforecast_utils import _is_categorical_col

_BASE = Path(__file__).parents[2] / "test_data" / "Health"
_RAINE_PATH = _BASE / "raine" / "merged.csv"
_BMI_PATH = _BASE / "bmi"
_PGS_ROOT = _BASE / "raine" / "PGSlipids"

# Maps G1XX / G2XX timepoint codes to follow-up year labels
_TIMEPOINT_YEAR = {
    "G108": "yr8", "G114": "yr14", "G117": "yr17",
    "G201": "yr1", "G202": "yr2", "G203": "yr3", "G205": "yr5",
    "G208": "yr8", "G210": "yr10", "G214": "yr14", "G217": "yr17",
    "G220": "yr20", "G222": "yr22", "G227": "yr27", "G228": "yr28",
}

# lipid targets available at these ages (G2xx_B3-B6)
_LIPIDS_AGES = [14, 17, 20, 22, 27, 28]
_LIPID_TARGETS = ["cholesterol", "triglyceride", "hdl", "ldl"]
_FEATURE_CUTOFF = 8

_CACHE_PGS     = _BASE / "raine" / "lipids_PGS"
_CACHE_KEEPTO8 = _BASE / "raine" / "lipids_keepto8"
_CACHE_PGSTO8  = _BASE / "raine" / "lipids_PGSto8"
_CACHE_RECENT  = _BASE / "raine" / "lipids_recent"


def _extract_timepoint(var_name: str) -> str:
    """Return year suffix like 'yr8' from variable name, or '' if none."""
    m = re.match(r"(G\d+)", var_name)
    if m:
        code = m.group(1)
        return _TIMEPOINT_YEAR.get(code, code.lower())
    return ""


def _extract_reporter(var_name: str) -> str:
    """Return reporter suffix: '_teacher' for TQ, '_parent' for PQ, else ''."""
    if "_TQ_" in var_name:
        return "_teacher"
    if "_PQ_" in var_name:
        return "_parent"
    return ""


def _is_original(var_name: str) -> bool:
    """True for _0-suffix variables (e.g. G214_B12_0 = original diab_raine)."""
    return var_name.endswith("_0")


def _shorten_label(label: str, var_name: str) -> str:
    """Convert a long variable label to a short, meaningful snake_case name."""
    # ---- demographics / identifiers ----
    if label == "SEX of Gen2":
        return "sex"
    if label == "STUDYNO":
        return "study_no"
    if label == "FAMID":
        return "family_id"
    if re.match(r"Mean age across points of contact", label):
        return "age"
    if re.match(r"Gen1 Mothers Age", label):
        return "mother_age"
    if re.match(r"Gen1 Fathers Age", label):
        return "father_age"

    # ---- anthropometrics ----
    if label == "Weight (kg)":
        return "weight"
    if label == "Standing height (cm)":
        return "height"
    if "Waist girth" in label:
        return "waist"

    # ---- blood pressure & heart rate ----
    if label == "Systolic blood pressure" or label.startswith("Systolic blood pressure"):
        return "sys_bp"
    if label == "Diastolic blood pressure" or label.startswith("Diastolic blood pressure"):
        return "dia_bp"
    if label == "Heart rate" or label.startswith("Heart rate"):
        return "hr"
    if "blood pressure code" in label.lower():
        return "bp_code"
    if "blood pressure state" in label.lower():
        return "bp_state"

    # ---- blood chemistry ----
    if "Glucose" in label:
        return "glucose"
    if "Total Cholesterol" in label:
        return "cholesterol"
    if "Triglyceride" in label:
        return "triglyceride" + ("_orig" if _is_original(var_name) else "")
    if "HDL cholesterol" in label:
        return "hdl"
    if "LDL cholesterol" in label:
        return "ldl"
    if "Insulin" in label:
        return "diab_raine" + ("_orig" if _is_original(var_name) else "")
    if "C Reactive Protein" in label:
        return "crp" + ("_orig" if _is_original(var_name) else "")

    # ---- physical activity (IPAQ) ----
    if "IPAQ" in label:
        if "Walking" in label:
            return "ipaq_walk"
        if "Moderate" in label:
            return "ipaq_mod"
        if "Vigorous" in label:
            return "ipaq_vig"
        if "Total" in label:
            return "ipaq_total"

    # ---- sleep (PSQI) ----
    if "PSQI" in label:
        if "hours of actual sleep" in label.lower():
            return "psqi_sleep"
        if "Categorical" in label:
            return "psqi_cat"
        return "psqi_score"

    # ---- diet (VCC nutrients) ----
    if "Nutrients computed from food" in label:
        if "Energy" in label and "excluding fibre" in label.lower():
            return "diet_energy_no_fibre"
        if "Energy" in label and "including fibre" in label.lower():
            return "diet_energy_fibre"
        if "Energy" in label:
            return "diet_energy"
        if "SatFat" in label or ("Sat" in label and "Fat" in label):
            return "diet_sat_fat"
        if "PolyFat" in label or ("Poly" in label and "Fat" in label):
            return "diet_poly_fat"
        if "MonoFat" in label or ("Mono" in label and "Fat" in label):
            return "diet_mono_fat"
        if "Fat" in label:
            return "diet_fat"
        if "Protein" in label:
            return "diet_protein"
        if "Sugars" in label:
            return "diet_sugars"
        if "Starch" in label:
            return "diet_starch"
        if "Fibre" in label:
            return "diet_fibre"
        if "Carbohydrate" in label:
            return "diet_carb"

    # ---- CBCL behavioural scales ----
    if "CBCL" in label:
        # measurement type
        if "Indicator for T score" in label:
            mtype = "_indicator"
        elif "Categorisation of T score" in label:
            mtype = "_category"
        elif "Raw score" in label:
            mtype = "_raw_score"
        elif "Count of number" in label:
            mtype = "_syndrome_count"
        else:
            mtype = ""

        subscale_map = [
            ("Social Withdrawal",  "cbcl_social_withdrawal"),
            ("Withdrawn",          "cbcl_withdrawn"),
            ("Somatic",            "cbcl_somatic"),
            ("Anxious/Depressed",  "cbcl_anxious_depressed"),
            ("Depressed",          "cbcl_depressed"),
            ("Social Problems",    "cbcl_social"),
            ("Thought Problems",   "cbcl_thought"),
            ("Attention Deficit",  "cbcl_dsm_adhd"),
            ("Attention Problems", "cbcl_attention"),
            ("Delinquent",         "cbcl_delinquent"),
            ("Aggressive",         "cbcl_aggressive"),
            ("Internalising",      "cbcl_internalising"),
            ("Externalising",      "cbcl_externalising"),
            ("Affective",          "cbcl_dsm_affective"),
            ("Anxiety Problems",   "cbcl_dsm_anxiety"),
            ("Oppositional",       "cbcl_dsm_oppositional"),
            ("Conduct",            "cbcl_dsm_conduct"),
            ("Total problems",     "cbcl_total"),
            ("Count of number",    "cbcl_syndrome_count"),
        ]
        for keyword, name in subscale_map:
            if keyword in label:
                # avoid double suffix when name already ends with mtype concept
                if mtype and name.endswith(mtype.lstrip("_")):
                    return name
                return name + mtype

    # ---- exam date ----
    if re.search(r"\bdate\b", label, re.I) or "XDAT" in var_name:
        return "exam_date"

    # ---- fallback: sanitized label, capped at 50 chars ----
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug[:30]


def _build_rename_map(dict_csv: Path, prefix: str) -> dict:
    """Build column rename map: original_name → meaningful_name."""
    df = pd.read_csv(dict_csv)
    rename: dict[str, str] = {}
    used: dict[str, int] = {}

    for _, row in df.iterrows():
        var = str(row["variable_name"])
        label = str(row["variable_label"])
        if var == "ID":
            continue

        reporter = _extract_reporter(var)
        timepoint = _extract_timepoint(var)
        short = _shorten_label(label, var)

        # build full name: prefix + short concept + reporter + timepoint
        parts = [prefix, short]
        if reporter:
            parts.append(reporter.lstrip("_"))
        if timepoint:
            parts.append(timepoint)
        candidate = "_".join(p for p in parts if p)

        # deduplicate
        if candidate in used:
            used[candidate] += 1
            name = f"{candidate}_{used[candidate]}"
        else:
            used[candidate] = 0
            name = candidate

        rename[var] = name

    return rename


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # replace sentinel -99 with NaN
    df = df.replace([-99, -999, -9999, 999, 9999], float("nan"))

    # drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")

    for col in df.select_dtypes(include="object").columns:
        print(f"\n[preprocessing] {col}")
        series = df[col]

        # date columns: parse to datetime and leave as datetime dtype
        as_dt = pd.to_datetime(series, errors="coerce", format="ISO8601")
        if as_dt.notna().sum() > series.notna().sum() * 0.5:
            now = pd.Timestamp.now()
            future_mask = as_dt > now
            if future_mask.any():
                as_dt = as_dt.copy()
                as_dt[future_mask] = as_dt[future_mask] - pd.DateOffset(years=100)
            df[col] = as_dt
            print(f"  [date column] {col}: {as_dt.dropna().iloc[0] if as_dt.notna().any() else 'all NaN'}")
            continue

        # detection-limit columns: values like "<0.16" or "<2.00"
        # multiple thresholds: smallest → 0, each next → previous threshold value
        has_limit = series.str.match(r"^[<>]=?\s*[\d.]", na=False).any()
        if has_limit:
            # collect all unique below-limit tokens and their numeric thresholds
            limit_vals = series[series.str.match(r"^<\s*[\d.]", na=False)].unique()
            thresholds = sorted(
                {tok: pd.to_numeric(re.sub(r"^[<>]=?\s*", "", tok), errors="coerce")
                 for tok in limit_vals}.items(),
                key=lambda x: x[1]
            )
            # assign: smallest → 0, each subsequent → previous threshold
            limit_map: dict[str, float] = {}
            for rank, (tok, thresh) in enumerate(thresholds):
                limit_map[tok] = 0.0 if rank == 0 else float(thresholds[rank - 1][1])

            coding: dict[str, float] = {}
            result = pd.Series(float("nan"), index=series.index, dtype=float)
            for i, val in series.items():
                if pd.isna(val):
                    result[i] = float("nan")
                elif str(val) in limit_map:
                    result[i] = limit_map[str(val)]
                    coding[val] = limit_map[str(val)]
                elif re.match(r"^[<>]=?\s*[\d.]", str(val)):
                    # > or >= tokens: just use the numeric value
                    num = pd.to_numeric(re.sub(r"^[<>]=?\s*", "", str(val)), errors="coerce")
                    result[i] = num
                    coding[val] = num
                else:
                    num = pd.to_numeric(val, errors="coerce")
                    result[i] = num
                    if pd.notna(num):
                        coding[val] = num
            df[col] = pd.to_numeric(result, errors="coerce")
            censored = {k: v for k, v in coding.items() if re.match(r"^[<>]", k)}
            print(f"\n[detection-limit coding] {col}")
            for orig, code in sorted(censored.items()):
                print(f"  {orig!r} → {code}")
            continue

        # remaining strings → label-encode (NaN stays NaN)
        codes = series.astype("category").cat.codes
        df[col] = codes.where(codes != -1, other=float("nan"))

    # drop any columns that became all-NaN after coercion
    df = df.dropna(axis=1, how="all")

    return df


# raw RAINE anthropometric exam codes ("y1_a1", "y1_a2", ...) -> concept name
_ANTHRO_A_CODE = {
    1: "weight", 2: "height", 3: "sitting_height", 4: "head_circum",
    5: "chest_circ", 6: "mid_arm_circum", 7: "triceps_skinfold",
    8: "subscapular_skinfold", 9: "suprailiac_skinfold",
    10: "abdominal_skinfold", 12: "waist_girth_avg", 13: "hip_avg",
}


def _normalize_timepoint_name(col: str) -> str | None:
    """Map a raw RAINE column with an inconsistent timepoint encoding to the
    canonical 'birth_<concept>' / '<concept>_yr<N>' form used elsewhere in
    this codebase. Returns None if `col` doesn't match a known scheme."""

    # literal birth_* fields: unify "length" into the "height" concept
    if col == "birth_length":
        return "birth_height"
    if col in ("birth_weight", "birth_head_circum"):
        return None  # already canonical

    # fam_splitup<N> (no underscore before the digit): N=0 is birth
    m = re.match(r"^fam_splitup(\d)$", col)
    if m:
        n = int(m.group(1))
        return "birth_fam_splitup" if n == 0 else f"fam_splitup_yr{n}"

    # y<N>_a<M> anthropometric exam codes
    m = re.match(r"^y(\d+)_a(\d{1,2})$", col)
    if m:
        year, code = int(m.group(1)), int(m.group(2))
        concept = _ANTHRO_A_CODE.get(code)
        if concept:
            return f"birth_{concept}" if year == 0 else f"{concept}_yr{year}"

    # generic y<N>_<name> / yr<N>_<name> prefix (underscore between year & name)
    m = re.match(r"^yr?(\d+)_([a-zA-Z].*)$", col)
    if m:
        year, rest = int(m.group(1)), m.group(2)
        return f"birth_{rest}" if year == 0 else f"{rest}_yr{year}"

    # generic y<N><name> prefix, no underscore (e.g. "y8obese", "yr8sbpmn", "y8bmi_x")
    m = re.match(r"^yr?(\d+)([a-zA-Z].*)$", col)
    if m:
        year, rest = int(m.group(1)), m.group(2)
        return f"birth_{rest}" if year == 0 else f"{rest}_yr{year}"

    # generic <name>_y<N> / <name>_yr<N> suffix (normalize "_y" to "_yr")
    m = re.match(r"^(.+)_yr?(\d{1,2})$", col)
    if m and 0 <= int(m.group(2)) <= 30:
        base, year = m.group(1), int(m.group(2))
        return f"birth_{base}" if year == 0 else f"{base}_yr{year}"

    # generic <name>_<N> bare-digit suffix (no "y"/"yr" letter): N=0 is birth,
    # else measured at year N (e.g. "cohab_0", "weight_12", "hhincome_1")
    m = re.match(r"^(.+)_(\d{1,2})$", col)
    if m and 0 <= int(m.group(2)) <= 30:
        base, year = m.group(1), int(m.group(2))
        return f"birth_{base}" if year == 0 else f"{base}_yr{year}"

    return None


def _normalize_raine_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw RAINE columns to a consistent timepoint naming scheme so
    longitudinal concepts (e.g. weight/height across ages) can be grouped.

    Two distinct raw columns can map to the same canonical name (e.g.
    "birth_length" and "height_0" both represent birth height). To avoid
    producing duplicate-labelled columns, only the first column encountered
    is renamed; later collisions are left under their original raw name."""
    rename = {}
    taken = set(df.columns)
    for col in df.columns:
        new = _normalize_timepoint_name(col)
        if not new or new == col:
            continue
        if new in taken:
            print(f"[normalize timepoints] skipping {col!r} -> {new!r} (name already in use)")
            continue
        rename[col] = new
        taken.add(new)
    if rename:
        print(f"\n[normalize timepoints] renaming {len(rename)} columns")
        df = df.rename(columns=rename)
    return df


def _score_file(pgs_dir: Path) -> Path | None:
    """Return the aggregated_scores file for a PGS directory (.txt or .txt.gz), or None."""
    txt = pgs_dir / "raine" / "score" / "aggregated_scores.txt"
    if txt.exists():
        return txt
    gz = pgs_dir / "raine" / "score" / "aggregated_scores.txt.gz"
    if gz.exists():
        return gz
    return None


def _get_pgs_cols() -> list[str]:
    return [d.name for d in sorted(_PGS_ROOT.iterdir()) if _score_file(d) is not None]


@lru_cache(maxsize=1)
def _build_merged() -> pd.DataFrame:
    """Load, merge, and preprocess the raw data once (no age-specific filtering).

    Memoized per-process: this is called once per (target, age) combo that
    isn't already cached to disk, and the underlying merge/preprocess is
    identical every time within a run."""
    raine = pd.read_csv(_RAINE_PATH, low_memory=False)
    raine = _normalize_raine_columns(raine)

    g1_rename = _build_rename_map(_BMI_PATH / "G1_data_dictionary.csv", "g1")
    g1 = pd.read_csv(_BMI_PATH / "G1_data.csv", low_memory=False).rename(
        columns={**g1_rename, "ID": "mother_id"}
    )

    g2_rename = _build_rename_map(_BMI_PATH / "G2_data_dictionary.csv", "g2")
    g2 = pd.read_csv(_BMI_PATH / "G2_data.csv", low_memory=False).rename(
        columns={**g2_rename, "ID": "child_id"}
    )

    merged = raine.merge(g1, on="mother_id", how="left")
    merged = merged.merge(g2, on="child_id", how="left")
    merged = merged.dropna(subset=["child_id"])

    for pgs_dir in sorted(_PGS_ROOT.iterdir()):
        score_file = _score_file(pgs_dir)
        if score_file is None:
            continue
        pgs_id = pgs_dir.name
        pgs = pd.read_csv(score_file, sep="\t")[["FID", "SUM"]].rename(
            columns={"FID": "child_id", "SUM": pgs_id}
        )
        pgs["child_id"] = pgs["child_id"].astype(merged["child_id"].dtype)
        merged = merged.merge(pgs, on="child_id", how="left")

    pgs_cols = _get_pgs_cols()
    merged = merged.dropna(subset=pgs_cols)
    merged = _preprocess(merged)
    return merged


_YR_COL_RE = re.compile(r"^(.+)_yr(\d+)$")
_BIRTH_COL_RE = re.compile(r"^birth_(.+)$")


def _longitudinal_groups(X: pd.DataFrame) -> dict[str, list[tuple[int, str, str]]]:
    """Group numeric columns that track the same concept across timepoints.

    Returns concept -> sorted list of (year, label, column_name), keeping only
    concepts observed at 2+ distinct timepoints.
    """
    groups: dict[str, list[tuple[int, str, str]]] = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            continue
        m = _YR_COL_RE.match(col)
        if m:
            base, year = m.group(1), int(m.group(2))
            concept = re.sub(r"^g[12]_", "", base)
            label = f"y{year}"
        else:
            m2 = _BIRTH_COL_RE.match(col)
            if not m2:
                continue
            concept, year, label = m2.group(1), 0, "birth"
        groups.setdefault(concept, []).append((year, label, col))

    result = {}
    for concept, entries in groups.items():
        dedup: dict[int, tuple[str, str]] = {}
        for year, label, col in entries:
            dedup.setdefault(year, (label, col))
        if len(dedup) >= 2:
            result[concept] = sorted((y, l, c) for y, (l, c) in dedup.items())
    return result


def _add_longitudinal_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add first-difference (df1_) and second-derivative (df2_) features
    between every pair/triple of timepoints for each longitudinal concept."""
    groups = _longitudinal_groups(X)
    print(f"\n[feature engineering] found {len(groups)} longitudinal concept groups")

    new_cols: dict[str, pd.Series] = {}
    for concept, entries in groups.items():
        print(f"  {concept}: {[label for _, label, _ in entries]}")

        diffs: dict[tuple[str, str], pd.Series] = {}
        for (yi, li, ci), (yj, lj, cj) in combinations(entries, 2):
            d = (X[cj] - X[ci]) / (yj - yi)
            diffs[(li, lj)] = d
            new_cols[f"df1_{concept}_{li}_{lj}"] = d

        for (yi, li, _), (yj, lj, _), (yk, lk, _) in combinations(entries, 3):
            d_ij = diffs[(li, lj)]
            d_jk = diffs[(lj, lk)]
            d2 = (d_jk - d_ij) / ((yk - yi) / 2)
            new_cols[f"df2_{concept}_{li}{lj}_{lj}{lk}"] = d2

    if new_cols:
        X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)
    print(f"[feature engineering] added {len(new_cols)} new features, total columns: {X.shape[1]}")
    return X


def _clean_and_impute(X: pd.DataFrame) -> pd.DataFrame:
    """Drop high-NaN, duplicate, perfectly-correlated, and date columns; then impute."""
    nan_props = X.isna().mean().sort_values(ascending=False)
    print("\n[NaN proportions per column]")
    for col, prop in nan_props.items():
        print(f"  {col}: {prop:.3f}")

    X = X.loc[:, X.isna().mean() <= 0.3]

    # drop duplicate columns
    dup_mask = X.T.duplicated()
    dup_cols = X.columns[dup_mask].tolist()
    if dup_cols:
        print(f"\n[duplicate columns dropped]")
        for col in dup_cols:
            col_vals = X[col].values
            orig = X.columns[~dup_mask][X.loc[:, ~dup_mask].apply(lambda r: (r.values == col_vals).all())].tolist()
            print(f"  {col} (duplicate of {orig[0] if orig else '?'})")
    X = X.loc[:, ~dup_mask]

    # drop perfectly correlated columns (|r| == 1), keeping the first
    numeric = X.select_dtypes(include="number")
    corr = numeric.corr().abs()
    upper = corr.where(pd.DataFrame(
        [[i < j for j in range(len(corr.columns))] for i in range(len(corr.columns))],
        index=corr.index, columns=corr.columns
    ))
    perfect_corr_cols = [c for c in upper.columns if (upper[c] >= 1.0).any()]
    if perfect_corr_cols:
        print(f"\n[perfectly correlated columns dropped]")
        for col in perfect_corr_cols:
            partners = upper.index[(upper[col] >= 1.0)].tolist()
            print(f"  {col} (corr=1 with {partners})")
    X = X.drop(columns=perfect_corr_cols)

    # drop date columns
    date_cols = X.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if date_cols:
        print(f"\n[dropping date columns] {date_cols}")
    X = X.drop(columns=date_cols)

    # impute: categorical cols → mode, continuous → IterativeImputer
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in numeric_cols
                if _is_categorical_col(c) or X[c].nunique() < 13]
    cont_cols = [c for c in numeric_cols if c not in cat_cols]

    print(f"\n[imputation] {len(cat_cols)} categorical, {len(cont_cols)} continuous")
    if cat_cols:
        X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
    if cont_cols:
        X[cont_cols] = IterativeImputer(max_iter=50, random_state=42).fit_transform(X[cont_cols])

    return X


def _high_age_suffixes(cutoff: int) -> set:
    """Return yr/y suffix strings for timepoints with age > cutoff.

    Covers every possible year (not just the known G-code timepoints), since
    normalized RAINE columns (e.g. "_yr12", "_yr24") can carry ages that
    don't appear in _TIMEPOINT_YEAR."""
    suffixes = set()
    for yr in range(cutoff + 1, 41):
        suffixes.add(f"yr{yr}")
        suffixes.add(f"y{yr}")
    return suffixes


def _get_target_col(merged: pd.DataFrame, target: str, age: int) -> str:
    y_suffix = f"yr{age}"
    candidates = [c for c in merged.columns if target in c and y_suffix in c]
    if not candidates:
        raise ValueError(f"No '{target}' column found for {y_suffix}")
    return candidates[0]


def _dedup_by_child_id(df: pd.DataFrame) -> pd.DataFrame:
    if "child_id" not in df.columns:
        return df
    before = len(df)
    df = (df.assign(_n_valid=df.notna().sum(axis=1))
          .sort_values("_n_valid", ascending=False)
          .drop_duplicates(subset="child_id", keep="first")
          .drop(columns="_n_valid"))
    dropped = before - len(df)
    if dropped:
        print(f"[dedup by child_id] dropped {dropped} duplicate rows")
    return df


def _cache_path(cache_dir: Path, target: str, age: int, feateng: bool) -> Path:
    suffix = "_feateng" if feateng else ""
    return cache_dir / f"{target}_{age}{suffix}.csv"


def _save_cache(cache_dir: Path, target: str, age: int, feateng: bool,
                 id_col: pd.Series, X: pd.DataFrame, y: pd.Series) -> Path:
    cache_path = _cache_path(cache_dir, target, age, feateng)
    y = y.rename(target)
    cache_df = pd.concat([id_col.reset_index(drop=True),
                          X.reset_index(drop=True),
                          y.reset_index(drop=True)], axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(cache_path, index=False)
    print(f"[cache] saved to {cache_path}")
    return cache_path


def _load_cache(cache_path: Path, target: str):
    print(f"[cache] loading from {cache_path}")
    cached = pd.read_csv(cache_path, low_memory=False)
    id_col = cached["child_id"]
    y = cached[target]
    X = cached.drop(columns=["child_id", target])
    return id_col, X, y


def load_data_PGS_only(target: str, age: int, feateng: bool = False) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Load PGS-only features to predict `target` lipid at the given age.

    feateng is irrelevant here (PGS columns carry no per-timepoint suffix,
    so longitudinal feature engineering is always a no-op), so the cache is
    always saved/loaded as "{target}_{age}.csv" regardless of the flag."""
    cache_path = _cache_path(_CACHE_PGS, target, age, False)
    if cache_path.exists():
        return _load_cache(cache_path, target)

    pgs_cols = _get_pgs_cols()
    merged = _build_merged()
    y_col = _get_target_col(merged, target, age)

    merged = merged.dropna(subset=[y_col])
    merged = _dedup_by_child_id(merged)

    id_col = merged["child_id"]
    y = merged[y_col].rename(target)
    X = merged[pgs_cols].copy()

    _save_cache(_CACHE_PGS, target, age, False, id_col, X, y)
    return id_col, X, y


def load_data_keepto8(target: str, age: int, feateng: bool = False) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Load features with timepoints <= 8, no PGS, to predict `target` lipid at the given age."""
    cache_path = _cache_path(_CACHE_KEEPTO8, target, age, feateng)
    if cache_path.exists():
        return _load_cache(cache_path, target)

    pgs_cols = _get_pgs_cols()
    merged = _build_merged()
    y_col = _get_target_col(merged, target, age)

    high_suffixes = _high_age_suffixes(_FEATURE_CUTOFF)
    pgs_set = set(pgs_cols)
    drop_cols = {c for c in merged.columns
                 if any(suf in c for suf in high_suffixes)
                 or c in pgs_set or c.startswith("PGS")}

    merged = merged.dropna(subset=[y_col])
    merged = _dedup_by_child_id(merged)

    id_col = merged["child_id"]
    y = merged[y_col].rename(target)
    X = merged.drop(columns=["child_id"] + list(drop_cols))
    X = _clean_and_impute(X)
    if feateng:
        X = _add_longitudinal_features(X)

    _save_cache(_CACHE_KEEPTO8, target, age, feateng, id_col, X, y)
    return id_col, X, y


def load_data_PGSto8(target: str, age: int, feateng: bool = False) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Load PGS + features with timepoints <= 8 to predict `target` lipid at the given age."""
    cache_path = _cache_path(_CACHE_PGSTO8, target, age, feateng)
    if cache_path.exists():
        return _load_cache(cache_path, target)

    merged = _build_merged()
    y_col = _get_target_col(merged, target, age)

    high_suffixes = _high_age_suffixes(_FEATURE_CUTOFF)
    drop_cols = {c for c in merged.columns
                 if any(suf in c for suf in high_suffixes)}

    merged = merged.dropna(subset=[y_col])
    merged = _dedup_by_child_id(merged)

    id_col = merged["child_id"]
    y = merged[y_col].rename(target)
    X = merged.drop(columns=["child_id"] + list(drop_cols))
    X = _clean_and_impute(X)
    if feateng:
        X = _add_longitudinal_features(X)

    _save_cache(_CACHE_PGSTO8, target, age, feateng, id_col, X, y)
    return id_col, X, y


def load_data_recent(target: str, age: int, feateng: bool = False) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Load PGS + all features collected strictly before target age to predict `target` lipid."""
    cache_path = _cache_path(_CACHE_RECENT, target, age, feateng)
    if cache_path.exists():
        return _load_cache(cache_path, target)

    merged = _build_merged()
    y_col = _get_target_col(merged, target, age)

    # Drop all timepoints >= age (keep timepoints < age)
    high_suffixes = _high_age_suffixes(age - 1)
    drop_cols = {c for c in merged.columns
                 if any(suf in c for suf in high_suffixes)}

    merged = merged.dropna(subset=[y_col])
    merged = _dedup_by_child_id(merged)

    id_col = merged["child_id"]
    y = merged[y_col].rename(target)
    X = merged.drop(columns=["child_id"] + list(drop_cols))
    X = _clean_and_impute(X)
    if feateng:
        X = _add_longitudinal_features(X)

    _save_cache(_CACHE_RECENT, target, age, feateng, id_col, X, y)
    return id_col, X, y


if __name__ == '__main__':
    for target in _LIPID_TARGETS:
        for age in _LIPIDS_AGES:
            ids, X, y = load_data_PGS_only(target, age)
            ids, X, y = load_data_keepto8(target, age)
            ids, X, y = load_data_PGSto8(target, age)
            ids, X, y = load_data_recent(target, age)