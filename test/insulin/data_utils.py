import re
import sys
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
_CACHE_KEEPTO14 = _BASE / "raine" / "insulin_glucose_keepto14"
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


_LONGITUDINAL_AGES_KEEPTO14 = [17, 20, 22, 27, 28]

def _build_merged() -> pd.DataFrame:
    """Load, merge, and preprocess the raw data once (no age-specific filtering)."""
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
        pgs_id = pgs_dir.name
        pgs = pd.read_csv(score_file, sep="\t")[["FID", "SUM"]].rename(
            columns={"FID": "child_id", "SUM": pgs_id}
        )
        pgs["child_id"] = pgs["child_id"].astype(merged["child_id"].dtype)
        merged = merged.merge(pgs, on="child_id", how="left")

    pgs_cols = [d.name for d in sorted(_PGS_ROOT.iterdir())
                if (d / "raine" / "score" / "aggregated_scores.txt").exists()]
    merged = merged.dropna(subset=pgs_cols)
    merged = _preprocess(merged)
    return merged


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
    """Return yr/y suffix strings for timepoints with age > cutoff."""
    suffixes = set()
    for yr in _TIMEPOINT_YEAR.values():
        m = re.match(r"yr(\d+)$", yr)
        if m and int(m.group(1)) > cutoff:
            suffixes.add(yr)
            suffixes.add(f"y{m.group(1)}")
    return suffixes


def _select_features(X: pd.DataFrame, y: pd.DataFrame, n_features: int) -> pd.DataFrame:
    """Use RFE with a RandomForest to select the top n_features columns."""
    print(f"\n[feature selection] RFE selecting {n_features} from {X.shape[1]} features")
    estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # RFE requires a single target; use the first column when y is multi-target
    y_rfe = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    selector = RFE(estimator, n_features_to_select=n_features, step=0.1)
    selector.fit(X, y_rfe)
    selected = X.columns[selector.support_].tolist()
    print(f"[feature selection] selected: {selected}")
    return X[selected]


def load_data_keepto14(targets: str | list[str], age: int, n_features: int | None = None) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Like load_data but drops variables with timepoints after age 14."""
    target_list = [targets] if isinstance(targets, str) else list(targets)
    base_name = "_".join(target_list) + f"_{age}"
    cache_path = _CACHE_KEEPTO14 / f"{base_name}.csv"
    rfe_cache_path = _CACHE_KEEPTO14 / f"{base_name}_top{n_features}.csv" if n_features is not None else None

    if rfe_cache_path is not None and rfe_cache_path.exists():
        print(f"[load_data_keepto14] loading RFE cache from {rfe_cache_path}")
        cached = pd.read_csv(rfe_cache_path, low_memory=False)
        y_suffix = f"yr{age}"
        y_cols = [c for c in cached.columns
                  if any(t in c and y_suffix in c for t in target_list)]
        id_col = cached["child_id"]
        y = cached[y_cols]
        X = cached.drop(columns=["child_id"] + y_cols)
        return id_col, X, y

    if cache_path.exists():
        print(f"[load_data_keepto14] loading cached data from {cache_path}")
        cached = pd.read_csv(cache_path, low_memory=False)
        y_suffix = f"yr{age}"
        y_cols = [c for c in cached.columns
                  if any(t in c and y_suffix in c for t in target_list)]
        id_col = cached["child_id"]
        y = cached[y_cols]
        X = cached.drop(columns=["child_id"] + y_cols)
        if n_features is not None:
            X = _select_features(X, y, n_features)
            rfe_df = pd.concat([id_col.reset_index(drop=True),
                                X.reset_index(drop=True),
                                y.reset_index(drop=True)], axis=1)
            rfe_df.to_csv(rfe_cache_path, index=False)
            print(f"[load_data_keepto14] saved RFE cache to {rfe_cache_path}")
        return id_col, X, y

    merged = _build_merged()
    y_suffix = f"yr{age}"

    y_cols = []
    for target in target_list:
        candidates = [c for c in merged.columns if target in c and y_suffix in c]
        if not candidates:
            raise ValueError(f"No column found containing '{target}' and '{y_suffix}'")
        y_cols.append(candidates[0])

    drop_cols = {c for c in merged.columns if any(suf in c for suf in _high_age_suffixes(14))}

    merged = merged.dropna(subset=y_cols)
    # deduplicate rows by child_id: keep the row with the most non-null values
    if "child_id" in merged.columns:
        before = len(merged)
        merged = (merged
                  .assign(_n_valid=merged.notna().sum(axis=1))
                  .sort_values("_n_valid", ascending=False)
                  .drop_duplicates(subset="child_id", keep="first")
                  .drop(columns="_n_valid"))
        dropped = before - len(merged)
        if dropped:
            print(f"\n[dedup by child_id] dropped {dropped} duplicate rows, kept row with most values")

    id_col = merged["child_id"]
    y = merged[y_cols]
    X = merged.drop(columns=["child_id"] + list(drop_cols))
    X = _clean_and_impute(X)

    cache_df = pd.concat([id_col.reset_index(drop=True),
                          X.reset_index(drop=True),
                          y.reset_index(drop=True)], axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(cache_path, index=False)
    print(f"[load_data_keepto14] saved cache to {cache_path}")

    if n_features is not None:
        X = _select_features(X, y, n_features)
        rfe_df = pd.concat([id_col.reset_index(drop=True),
                            X.reset_index(drop=True),
                            y.reset_index(drop=True)], axis=1)
        rfe_df.to_csv(rfe_cache_path, index=False)
        print(f"[load_data_keepto14] saved RFE cache to {rfe_cache_path}")

    return id_col, X, y


def load_data_longitudinal_keepto14(targets: str | list[str],
                                    ages: list[int] | None = None,
                                    n_features: int | None = None,
                                    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Like load_data_longitudinal but drops variables with timepoints after age 14."""
    target_list = [targets] if isinstance(targets, str) else list(targets)
    if ages is None:
        ages = _LONGITUDINAL_AGES_KEEPTO14

    base_name = "_".join(target_list) + "_longitudinal"
    cache_path = _CACHE_KEEPTO14 / f"{base_name}.csv"
    rfe_cache_path = _CACHE_KEEPTO14 / f"{base_name}_top{n_features}.csv" if n_features is not None else None

    if rfe_cache_path is not None and rfe_cache_path.exists():
        print(f"[load_data_longitudinal_keepto14] loading RFE cache from {rfe_cache_path}")
        cached = pd.read_csv(rfe_cache_path, low_memory=False)
        id_col = cached["child_id"]
        y = cached[target_list]
        X = cached.drop(columns=["child_id"] + target_list)
        return id_col, X, y

    if cache_path.exists():
        print(f"[load_data_longitudinal_keepto14] loading cached data from {cache_path}")
        cached = pd.read_csv(cache_path, low_memory=False)
        id_col = cached["child_id"]
        y = cached[target_list]
        X = cached.drop(columns=["child_id"] + target_list)
        if n_features is not None:
            X = _select_features(X, y, n_features)
            rfe_df = pd.concat([id_col.reset_index(drop=True),
                                X.reset_index(drop=True),
                                y.reset_index(drop=True)], axis=1)
            rfe_df.to_csv(rfe_cache_path, index=False)
            print(f"[load_data_longitudinal_keepto14] saved RFE cache to {rfe_cache_path}")
        return id_col, X, y

    merged = _build_merged()

    # baseline (non-timepoint) columns: keep only up to age 14
    high_age_suffixes = _high_age_suffixes(14)

    baseline_cols = [c for c in merged.columns
                     if not any(suf in c for suf in high_age_suffixes)
                     and c != "child_id"]

    slices = []
    for age in ages:
        y_suffix = f"yr{age}"
        y_col_map = {}
        missing = False
        for target in target_list:
            candidates = [c for c in merged.columns if target in c and y_suffix in c]
            if not candidates:
                print(f"  [longitudinal_keepto14] no column for target='{target}' age={age}, skipping age")
                missing = True
                break
            y_col_map[target] = candidates[0]

        if missing:
            continue

        actual_y_cols = list(y_col_map.values())
        sub = merged.dropna(subset=actual_y_cols).copy()
        sub["age"] = age
        sub = sub.rename(columns={v: k for k, v in y_col_map.items()})
        keep_cols = ["child_id", "age"] + baseline_cols + target_list
        slices.append(sub[[c for c in keep_cols if c in sub.columns]].copy())
        print(f"  [longitudinal_keepto14] age={age}: {len(slices[-1])} rows")

    if not slices:
        raise RuntimeError("No age slices could be built for longitudinal_keepto14 data.")

    stacked: pd.DataFrame = pd.concat(slices, axis=0, ignore_index=True)

    id_col = stacked["child_id"]
    y = stacked[target_list]
    X = stacked.drop(columns=["child_id"] + target_list)

    X = _clean_and_impute(X)

    cache_df = pd.concat([id_col.reset_index(drop=True),
                          X.reset_index(drop=True),
                          y.reset_index(drop=True)], axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(cache_path, index=False)
    print(f"[load_data_longitudinal_keepto14] saved cache to {cache_path}")

    if n_features is not None:
        X = _select_features(X, y, n_features)
        rfe_df = pd.concat([id_col.reset_index(drop=True),
                            X.reset_index(drop=True),
                            y.reset_index(drop=True)], axis=1)
        rfe_df.to_csv(rfe_cache_path, index=False)
        print(f"[load_data_longitudinal_keepto14] saved RFE cache to {rfe_cache_path}")

    return id_col, X, y


if __name__ == '__main__':
    n_feature = 200
    ids, X, y = load_data_keepto14(["insulin", "glucose"], 17, n_features=n_feature)
    ids, X, y = load_data_keepto14(["insulin", "glucose"], 20, n_features=n_feature)
    ids, X, y = load_data_keepto14(["insulin", "glucose"], 22, n_features=n_feature)
    ids, X, y = load_data_keepto14(["insulin", "glucose"], 27, n_features=n_feature)
    ids, X, y = load_data_keepto14(["insulin", "glucose"], 28, n_features=n_feature)
    ids, X, y = load_data_longitudinal_keepto14(["insulin", "glucose"], ages=_LONGITUDINAL_AGES_KEEPTO14, n_features=n_feature)
