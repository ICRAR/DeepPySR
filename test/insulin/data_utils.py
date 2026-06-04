import re
import sys
import pandas as pd
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer

# import categorical definitions from bmiforecast_utils
_bmiforecast_dir = str(Path(__file__).parents[1] / "bmiforecast")
if _bmiforecast_dir not in sys.path:
    sys.path.insert(0, _bmiforecast_dir)
from bmiforecast_utils import _is_categorical_col

_BASE = Path(__file__).parents[2] / "test_data" / "Health"
_RAINE_PATH = _BASE / "raine" / "merged.csv"
_BMI_PATH = _BASE / "bmi"
_PGS_ROOT = _BASE / "raine" / "PGSt2d" / "pgs_score"

# Maps G1XX / G2XX timepoint codes to follow-up year labels
_TIMEPOINT_YEAR = {
    "G108": "yr8", "G114": "yr14", "G117": "yr17",
    "G201": "yr1", "G202": "yr2", "G203": "yr3", "G205": "yr5",
    "G208": "yr8", "G210": "yr10", "G214": "yr14", "G217": "yr17",
    "G220": "yr20", "G222": "yr22", "G227": "yr27", "G228": "yr28",
}


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
    """True for _0-suffix variables (e.g. G214_B12_0 = original insulin)."""
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
        return "insulin" + ("_orig" if _is_original(var_name) else "")
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
    df = df.replace(-99, float("nan"))

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
            result = series.copy().astype(float, errors="ignore")
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


def load_data(targets: str | list[str], age: int) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    target_list = [targets] if isinstance(targets, str) else list(targets)
    cache_name = "_".join(target_list) + f"_{age}.csv"
    cache_path = _BASE / "raine" / cache_name

    if cache_path.exists():
        print(f"[load_data] loading cached data from {cache_path}")
        cached = pd.read_csv(cache_path, low_memory=False)
        y_suffix = f"yr{age}"
        y_cols = [c for c in cached.columns
                  if any(t in c and y_suffix in c for t in target_list)]
        id_col = cached["child_id"]
        y = cached[y_cols]
        X = cached.drop(columns=["child_id"] + y_cols)
        return id_col, X, y

    raine = pd.read_csv(_RAINE_PATH, low_memory=False)

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
        score_file = pgs_dir / "raine" / "score" / "aggregated_scores.txt"
        if not score_file.exists():
            continue
        pgs_id = pgs_dir.name  # e.g. "PGS000729"
        pgs = pd.read_csv(score_file, sep="\t")[["FID", "SUM"]].rename(
            columns={"FID": "child_id", "SUM": pgs_id}
        )
        pgs["child_id"] = pgs["child_id"].astype(merged["child_id"].dtype)
        merged = merged.merge(pgs, on="child_id", how="left")

    pgs_cols = [d.name for d in sorted(_PGS_ROOT.iterdir()) if (d / "raine" / "score" / "aggregated_scores.txt").exists()]
    merged = merged.dropna(subset=pgs_cols)

    merged = _preprocess(merged)

    target_list = [targets] if isinstance(targets, str) else list(targets)
    y_suffix = f"yr{age}"

    # find y columns: one per target, must contain target name and yr{age}
    y_cols = []
    for target in target_list:
        candidates = [c for c in merged.columns if target in c and y_suffix in c]
        if not candidates:
            raise ValueError(f"No column found containing '{target}' and '{y_suffix}'")
        y_cols.append(candidates[0])

    # drop all columns with yr{age} or y{age} where age > 8
    high_age_suffixes = set()
    for yr in _TIMEPOINT_YEAR.values():
        m = re.match(r"yr(\d+)$", yr)
        if m and int(m.group(1)) > 8:
            high_age_suffixes.add(yr)
            high_age_suffixes.add(f"y{m.group(1)}")
    drop_cols = {
        c for c in merged.columns
        if any(suf in c for suf in high_age_suffixes)
    }

    merged = merged.dropna(subset=y_cols)

    id_col = merged["child_id"]
    y = merged[y_cols]

    X = merged.drop(columns=["child_id"] + list(drop_cols))

    nan_props = X.isna().mean().sort_values(ascending=False)
    print("\n[NaN proportions per column]")
    for col, prop in nan_props.items():
        print(f"  {col}: {prop:.3f}")

    X = X.loc[:, X.isna().mean() <= 0.3]

    # drop duplicate columns
    dup_mask = X.T.duplicated()
    dup_cols = X.columns[dup_mask].tolist()
    X = X.drop(columns=dup_cols)

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
    # categorical = manually defined list OR fewer than 13 unique values
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in numeric_cols
                if _is_categorical_col(c) or X[c].nunique() < 13]
    cont_cols = [c for c in numeric_cols if c not in cat_cols]

    print(f"\n[imputation] {len(cat_cols)} categorical, {len(cont_cols)} continuous")
    if cat_cols:
        X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
    if cont_cols:
        X[cont_cols] = IterativeImputer(max_iter=50, random_state=42).fit_transform(X[cont_cols])

    # save cache: child_id + X + y columns
    cache_df = pd.concat([id_col.reset_index(drop=True),
                          X.reset_index(drop=True),
                          y.reset_index(drop=True)], axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(cache_path, index=False)
    print(f"[load_data] saved cache to {cache_path}")

    return id_col, X, y

if __name__ == '__main__':
    ids, X, y = load_data(["insulin", "glucose"], 14)