import re
import sys
from functools import lru_cache
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

# lipid targets available at these ages (G2xx_B3-B6). Age 27 is no longer a
# standalone target age -- its values are merged into age 28 by
# _consolidate_yr27_into_yr28 (called from _build_merged, fixing a severe class
# imbalance in participant counts between the two adjacent waves) along with
# every other yr27 variable, not just the 4 lipid targets.
_LIPIDS_AGES = [14, 17, 20, 22, 28]
_LIPID_TARGETS = ["cholesterol", "triglyceride", "hdl", "ldl"]
_LIPID_TARGET_KEYWORDS = ("cholesterol", "triglyceride", "hdl", "ldl")
_FEATURE_CUTOFF = 8

# Known data-entry errors for child_id 10840 at yr27: cholesterol=76.0 mmol/L
# (~15x any other value in the cohort; next-highest is 8.1) and
# triglyceride=78.0 mmol/L (also the max in the cohort by a huge margin),
# while their yr28 values (5.2 and 1.4 respectively) are unremarkable --
# almost certainly transcription errors (e.g. a misplaced decimal), not real
# measurements. Nulled out in _build_merged, before _consolidate_yr27_into_yr28
# runs, so they can neither contaminate the age-27/28 merge nor leak in as raw
# yr27 feature values in any input_type that keeps that timepoint.
_BAD_LIPID_YR27_CHILD_ID = 10840
_BAD_LIPID_YR27_TARGETS = ("cholesterol", "triglyceride")

_CACHE_PGS     = _BASE / "raine" / "lipids_PGS"
_CACHE_KEEPTO8 = _BASE / "raine" / "lipids_keepto8"
_CACHE_PGSTO8  = _BASE / "raine" / "lipids_PGSto8"
_CACHE_RECENT  = _BASE / "raine" / "lipids_recent"
_CACHE_NBLOOD  = _BASE / "raine" / "lipids_nblood"


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

    merged = _null_bad_lipid_yr27_values(merged)
    merged = _consolidate_yr27_into_yr28(merged)
    merged = _consolidate_sex_columns(merged)
    merged = _consolidate_bmi_yr8(merged)
    merged = _add_missing_bmi(merged)
    return merged


def _consolidate_yr27_into_yr28(merged: pd.DataFrame) -> pd.DataFrame:
    """Fold every yr27 column into its yr28 sibling: row-wise mean where both exist,
    whichever one exists when only one does. Ages 27 and 28 are the same RAINE
    follow-up wave, just split across participants by scheduling (see _LIPIDS_AGES),
    so this generalizes what used to be a lipid-target-only special case (previously
    done ad hoc in _lipid_target_series) to every variable collected at that wave --
    afterwards no 'yr27' column remains anywhere in the dataset, only 'yr28'.

    Must run after _null_bad_lipid_yr27_values so the known-bad yr27 values are
    nulled before they can pollute an averaged yr28 value."""
    yr27_cols = [c for c in merged.columns if "yr27" in c]
    n_merged = n_renamed = 0
    for col27 in yr27_cols:
        col28 = col27.replace("yr27", "yr28")
        if col28 in merged.columns:
            merged[col28] = pd.concat([merged[col27], merged[col28]], axis=1).mean(axis=1, skipna=True)
            merged = merged.drop(columns=[col27])
            n_merged += 1
        else:
            merged = merged.rename(columns={col27: col28})
            n_renamed += 1
    print(f"\n[yr27/yr28 consolidation] merged {n_merged} column pair(s) into their "
          f"yr28 sibling, renamed {n_renamed} yr27-only column(s) to yr28 "
          f"({len(yr27_cols)} yr27 columns processed, 0 remain)")
    return merged


def _null_bad_lipid_yr27_values(merged: pd.DataFrame) -> pd.DataFrame:
    """Null out the known-bad yr27 cholesterol/triglyceride values for
    _BAD_LIPID_YR27_CHILD_ID (see constant docstring). Done once here, on the
    raw columns, so the fix applies uniformly whether that timepoint is read
    as a target (via _lipid_target_series) or leaks in as a raw feature in
    any input_type that keeps yr27 columns (e.g. 'recent' at age 28)."""
    bad = merged["child_id"] == _BAD_LIPID_YR27_CHILD_ID
    for target in _BAD_LIPID_YR27_TARGETS:
        col27 = _get_target_col(merged, target, 27)
        n_bad = int((bad & merged[col27].notna()).sum())
        if n_bad:
            print(f"[data cleaning] nulling {n_bad} known-bad {col27} value(s) "
                  f"for child_id {_BAD_LIPID_YR27_CHILD_ID}")
            merged.loc[bad, col27] = float("nan")
    return merged


_SEX_COL_RE = re.compile(r"^(g[12]_)?sex(01|_x)?$", re.IGNORECASE)


def _consolidate_sex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse every sex-encoding column (raw + dictionary-derived, whatever
    coding they use) into a single canonical 'sex' feature so downstream
    duplicate/correlation-based dropping doesn't have to be relied on."""
    candidates = [c for c in df.columns if _SEX_COL_RE.match(c)]
    if len(candidates) <= 1:
        return df

    canonical = "g2_sex" if "g2_sex" in candidates else sorted(candidates)[0]
    combined = df[canonical].copy()
    for c in candidates:
        if c == canonical:
            continue
        other = df[c]
        both = pd.concat([combined.rename("canon"), other.rename("other")], axis=1).dropna()
        if len(both) and (both["canon"] == both["other"]).mean() < 0.5:
            # differently-coded (e.g. 1/2 vs 0/1): remap to canonical's scale
            mapping = both.groupby("other")["canon"].agg(lambda s: s.mode().iloc[0])
            other = other.map(mapping)
        combined = combined.fillna(other)

    print(f"\n[sex columns] consolidated {candidates} -> '{canonical}' "
          f"({int(combined.notna().sum())} non-null)")
    df = df.drop(columns=[c for c in candidates if c != canonical])
    df[canonical] = combined
    return df


def _consolidate_bmi_yr8(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the two precomputed yr8 BMI columns (bmi_x_yr8, bmi_y_yr8 --
    two independent sources in the raw RAINE data) into a single
    consistently-named 'bmi_yr8', row-wise mean where both are present."""
    x_col, y_col = "bmi_x_yr8", "bmi_y_yr8"
    has_x, has_y = x_col in df.columns, y_col in df.columns
    if not (has_x or has_y):
        return df

    if has_x and has_y:
        bmi_yr8 = pd.concat([df[x_col], df[y_col]], axis=1).mean(axis=1, skipna=True)
        drop_cols = [x_col, y_col]
    else:
        bmi_yr8 = df[x_col] if has_x else df[y_col]
        drop_cols = [x_col if has_x else y_col]

    print(f"\n[bmi] consolidated {drop_cols} -> 'bmi_yr8'")
    df = df.drop(columns=drop_cols)
    df["bmi_yr8"] = bmi_yr8
    return df


# (weight_column, weight->kg scale), (height_column, height->m scale) for
# every BMI checkpoint missing from the raw data (birth, yr1). Later
# checkpoints (yr5/8/10/14/17/20/22/27) are already precomputed upstream;
# yr28 can't be added since no height measurement exists at that timepoint.
_MISSING_BMI_SOURCES = {
    "birth_bmi": (("birth_weight", 1 / 1000), ("birth_height", 1 / 100)),
    "bmi_yr1":   (("weight_yr1", 1), ("height_yr1", 1 / 100)),
}


def _add_missing_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Add BMI columns not present in the raw data (birth, yr1) from their
    underlying weight/height measurements, named consistently with the
    existing 'birth_<concept>' / '<concept>_yr<N>' convention."""
    for bmi_col, ((w_col, w_scale), (h_col, h_scale)) in _MISSING_BMI_SOURCES.items():
        if bmi_col in df.columns or w_col not in df.columns or h_col not in df.columns:
            continue
        weight_kg = df[w_col] * w_scale
        height_m = df[h_col] * h_scale
        df[bmi_col] = weight_kg / (height_m ** 2)
        print(f"\n[bmi] computed '{bmi_col}' from {w_col}/{h_col} "
              f"({int(df[bmi_col].notna().sum())} non-null)")
    return df


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


def _recent_feature_cutoff(age: int) -> int:
    """Highest wave (inclusive) still usable as a 'recent'/'nblood' feature for the
    given target age. Age 28 is the merged yr27/yr28 wave: yr27 is effectively the
    same wave as the target, not a genuine prior observation, so it must be excluded
    too -- the cutoff drops to 26 instead of the usual age - 1.

    _consolidate_yr27_into_yr28 already folds every yr27 column into yr28 at the
    source (_build_merged), so no 'yr27'-named column should exist by the time this
    runs -- this special case is a defensive backstop, not the primary mechanism."""
    return 26 if age == 28 else age - 1


def _get_target_col(merged: pd.DataFrame, target: str, age: int) -> str:
    y_suffix = f"yr{age}"
    candidates = [c for c in merged.columns if target in c and y_suffix in c]
    if not candidates:
        raise ValueError(f"No '{target}' column found for {y_suffix}")
    return candidates[0]


def _lipid_target_series(merged: pd.DataFrame, target: str, age: int) -> pd.Series:
    """y values for `target` at `age`, aligned to merged's index.

    age=28 already reflects the merged yr27/yr28 wave -- _build_merged folds every
    yr27 column (including the 4 lipid targets) into its yr28 sibling via
    _consolidate_yr27_into_yr28 (row-wise mean where both waves are present,
    whichever one exists when only one is), with known-bad yr27 values already
    nulled out beforehand (see _null_bad_lipid_yr27_values). So this is now a
    plain lookup for every age, including 28."""
    return merged[_get_target_col(merged, target, age)]


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


def _cache_path(cache_dir: Path, target: str, age: int) -> Path:
    return cache_dir / f"{target}_{age}.csv"


def _save_cache(cache_dir: Path, target: str, age: int,
                 id_col: pd.Series, X: pd.DataFrame, y: pd.Series) -> Path:
    cache_path = _cache_path(cache_dir, target, age)
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


def _blood_lipid_cols(columns) -> set:
    """Every column tracking one of the 4 blood-lipid concepts at any
    timepoint (used by load_data_nblood to exclude lipid history as a
    feature, not just the current target)."""
    return {c for c in columns if any(kw in c for kw in _LIPID_TARGET_KEYWORDS)}


def load_data_PGS_only(target: str, age: int) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Load PGS-only features to predict `target` lipid at the given age."""
    cache_path = _cache_path(_CACHE_PGS, target, age)
    if cache_path.exists():
        return _load_cache(cache_path, target)

    pgs_cols = _get_pgs_cols()
    merged = _build_merged()
    y_all = _lipid_target_series(merged, target, age)

    merged = merged.loc[y_all.notna()]
    merged = _dedup_by_child_id(merged)

    id_col = merged["child_id"]
    y = y_all.loc[merged.index].rename(target)
    X = merged[pgs_cols].copy()

    _save_cache(_CACHE_PGS, target, age, id_col, X, y)
    return id_col, X, y


def load_data_keepto8(target: str, age: int) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Load features with timepoints <= 8, no PGS, to predict `target` lipid at the given age."""
    cache_path = _cache_path(_CACHE_KEEPTO8, target, age)
    if cache_path.exists():
        return _load_cache(cache_path, target)

    pgs_cols = _get_pgs_cols()
    merged = _build_merged()
    y_all = _lipid_target_series(merged, target, age)

    high_suffixes = _high_age_suffixes(_FEATURE_CUTOFF)
    pgs_set = set(pgs_cols)
    drop_cols = {c for c in merged.columns
                 if any(suf in c for suf in high_suffixes)
                 or c in pgs_set or c.startswith("PGS") or c.startswith("SUM_PGS")}

    merged = merged.loc[y_all.notna()]
    merged = _dedup_by_child_id(merged)

    id_col = merged["child_id"]
    y = y_all.loc[merged.index].rename(target)
    X = merged.drop(columns=["child_id"] + list(drop_cols))
    X = _clean_and_impute(X)

    _save_cache(_CACHE_KEEPTO8, target, age, id_col, X, y)
    return id_col, X, y


def load_data_PGSto8(target: str, age: int) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Load PGS + features with timepoints <= 8 to predict `target` lipid at the given age."""
    cache_path = _cache_path(_CACHE_PGSTO8, target, age)
    if cache_path.exists():
        return _load_cache(cache_path, target)

    merged = _build_merged()
    y_all = _lipid_target_series(merged, target, age)

    high_suffixes = _high_age_suffixes(_FEATURE_CUTOFF)
    drop_cols = {c for c in merged.columns
                 if any(suf in c for suf in high_suffixes) or c.startswith("SUM_PGS")}

    merged = merged.loc[y_all.notna()]
    merged = _dedup_by_child_id(merged)

    id_col = merged["child_id"]
    y = y_all.loc[merged.index].rename(target)
    X = merged.drop(columns=["child_id"] + list(drop_cols))
    X = _clean_and_impute(X)

    _save_cache(_CACHE_PGSTO8, target, age, id_col, X, y)
    return id_col, X, y


def load_data_recent(target: str, age: int) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Load PGS + all features collected strictly before target age to predict `target` lipid."""
    cache_path = _cache_path(_CACHE_RECENT, target, age)
    if cache_path.exists():
        return _load_cache(cache_path, target)

    merged = _build_merged()
    y_all = _lipid_target_series(merged, target, age)

    # Drop all timepoints >= the cutoff (keep timepoints strictly before it)
    high_suffixes = _high_age_suffixes(_recent_feature_cutoff(age))
    drop_cols = {c for c in merged.columns
                 if any(suf in c for suf in high_suffixes) or c.startswith("SUM_PGS")}

    merged = merged.loc[y_all.notna()]
    merged = _dedup_by_child_id(merged)

    id_col = merged["child_id"]
    y = y_all.loc[merged.index].rename(target)
    X = merged.drop(columns=["child_id"] + list(drop_cols))
    X = _clean_and_impute(X)

    _save_cache(_CACHE_RECENT, target, age, id_col, X, y)
    return id_col, X, y


def load_data_nblood(target: str, age: int) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Like load_data_recent, but also excludes every blood-lipid column
    (cholesterol/triglyceride/hdl/ldl at any timepoint) from the features,
    not just the current target -- so no lipid history can leak in."""
    cache_path = _cache_path(_CACHE_NBLOOD, target, age)
    if cache_path.exists():
        return _load_cache(cache_path, target)

    merged = _build_merged()
    y_all = _lipid_target_series(merged, target, age)

    high_suffixes = _high_age_suffixes(_recent_feature_cutoff(age))
    drop_cols = {c for c in merged.columns
                 if any(suf in c for suf in high_suffixes) or c.startswith("SUM_PGS")}
    drop_cols |= _blood_lipid_cols(merged.columns)

    merged = merged.loc[y_all.notna()]
    merged = _dedup_by_child_id(merged)

    id_col = merged["child_id"]
    y = y_all.loc[merged.index].rename(target)
    X = merged.drop(columns=["child_id"] + list(drop_cols))
    X = _clean_and_impute(X)

    _save_cache(_CACHE_NBLOOD, target, age, id_col, X, y)
    return id_col, X, y


if __name__ == '__main__':
    for target in _LIPID_TARGETS:
        for age in _LIPIDS_AGES:
            ids, X, y = load_data_PGS_only(target, age)
            ids, X, y = load_data_keepto8(target, age)
            ids, X, y = load_data_PGSto8(target, age)
            ids, X, y = load_data_recent(target, age)
            ids, X, y = load_data_nblood(target, age)