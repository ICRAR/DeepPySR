import numpy as np
import pandas as pd
from scipy.stats import norm
import os
import sys
sys.path.append('~/Projects/pgs')
def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None
paths = [os.path.abspath(os.path.expanduser(path)) for path in sys.path]
for path in paths:
    if os.path.isfile(os.path.join(path,'./bmizscore/CDCref_d.csv')):
        break
ref = pd.read_csv(os.path.join(path,'./bmizscore/CDCref_d.csv'))

# for 2 to 19 year old
def _zscore(agemos, var, l, m, s):

    if var > 0 and agemos >= 24:
        if abs(l) >= 0.01:
            z = ((var / m) ** l - 1) / (l * s)
        else:
            z = np.log(var / m) / s
        p = norm.cdf(z) * 100

        sdl = (m - m * (1 - 2 * l * s) ** (1 / l)) / 2
        sdh = (m * (1 + 2 * l * s) ** (1 / l) - m) / 2
        if var < m:
            f = (var - m) / sdl
        else:
            f = (var - m) / sdh
    else:
        z, p, f = None, None, None

    return z, p, f


def _cuts(var, l, u):

    if l <= var <= u:
        out = 0
    elif var > u:
        out = 1
    elif var < l:
        out = -1
    return out


def get_bmiz(mydata_row, ref=ref):
    # Set lengths for agemos and sex columns
    agemos = mydata_row["agemos"].astype(float)
    sex = mydata_row["sex"].astype(int)
    height = mydata_row["height"].astype(float)
    weight = mydata_row["weight"].astype(float)
    bmi = mydata_row["bmi"].astype(float) if mydata_row["bmi"] else None
    headcir = mydata_row["headcir"].astype(float) if mydata_row["headcir"] else None

    _agecat = 0 if 0 <= agemos <= 0.5 else round(agemos + 0.5) - 0.5
    if bmi is None or bmi < 0:
        if weight > 0 and height > 0 and agemos >= 24:  # for 2-19 year
            bmi = weight / (height / 100) ** 2

    # begin the for-age calcs - note that this calls up the refdir
    ref = ref.sort_values(by=["SEX", "_AGECAT"])
    refp = ref[
        (ref["denom"] == "age") & (ref["SEX"] == sex) & (ref["_AGECAT"] == _agecat)
    ]
    ageint = refp["_AGEMOS2"].values[0] - refp["_AGEMOS1"].values[0]
    dage = agemos - refp["_AGEMOS1"].values[0]

    _llg = (
        refp["_LLG1"].values[0]
        + (dage * (refp["_LLG2"].values[0] - refp["_LLG1"].values[0])) / ageint
    )
    _mlg = (
        refp["_MLG1"].values[0]
        + (dage * (refp["_MLG2"].values[0] - refp["_MLG1"].values[0])) / ageint
    )
    _slg = (
        refp["_SLG1"].values[0]
        + (dage * (refp["_SLG2"].values[0] - refp["_SLG1"].values[0])) / ageint
    )
    _lht = (
        refp["_LHT1"].values[0]
        + (dage * (refp["_LHT2"].values[0] - refp["_LHT1"].values[0])) / ageint
    )
    _mht = (
        refp["_MHT1"].values[0]
        + (dage * (refp["_MHT2"].values[0] - refp["_MHT1"].values[0])) / ageint
    )
    _sht = (
        refp["_SHT1"].values[0]
        + (dage * (refp["_SHT2"].values[0] - refp["_SHT1"].values[0])) / ageint
    )
    _lwt = (
        refp["_LWT1"].values[0]
        + (dage * (refp["_LWT2"].values[0] - refp["_LWT1"].values[0])) / ageint
    )
    _mwt = (
        refp["_MWT1"].values[0]
        + (dage * (refp["_MWT2"].values[0] - refp["_MWT1"].values[0])) / ageint
    )
    _swt = (
        refp["_SWT1"].values[0]
        + (dage * (refp["_SWT2"].values[0] - refp["_SWT1"].values[0])) / ageint
    )
    _lhc = (
        refp["_LHC1"].values[0]
        + (dage * (refp["_LHC2"].values[0] - refp["_LHC1"].values[0])) / ageint
    )
    _mhc = (
        refp["_MHC1"].values[0]
        + (dage * (refp["_MHC2"].values[0] - refp["_MHC1"].values[0])) / ageint
    )
    _shc = (
        refp["_SHC1"].values[0]
        + (dage * (refp["_SHC2"].values[0] - refp["_SHC1"].values[0])) / ageint
    )
    _lbmi = (
        refp["_LBMI1"].values[0]
        + (dage * (refp["_LBMI2"].values[0] - refp["_LBMI1"].values[0])) / ageint
    )
    _mbmi = (
        refp["_MBMI1"].values[0]
        + (dage * (refp["_MBMI2"].values[0] - refp["_MBMI1"].values[0])) / ageint
    )
    _sbmi = (
        refp["_SBMI1"].values[0]
        + (dage * (refp["_SBMI2"].values[0] - refp["_SBMI1"].values[0])) / ageint
    )

    if agemos < 24:  # theres a valid value for 23.5 months!
        _mbmi = np.nan

    # lgz, lgpct, mod_lenz = _zscore(agemos, height, _llg, _mlg, _slg)
    # _bivlg = _cuts(mod_lenz, -5, 4)
    # stz, stpct, mod_statz = _zscore(agemos, height, _lht, _mht, _sht)
    # _bivst = _cuts(mod_statz, -5, 4)
    # waz, wapct, mod_waz = _zscore(agemos, weight, _lwt, _mwt, _swt)
    # _bivwt = _cuts(mod_waz, -5, 8)
    # headcz, headcpct, mod_headcz = _zscore(agemos, headcir, _lhc, _mhc, _shc)
    # _bivhc = _cuts(mod_headcz, -5, 5)
    bmiz, bmipct, mod_bmiz = _zscore(agemos, bmi, _lbmi, _mbmi, _sbmi)
    _bivbmi = _cuts(mod_bmiz, -4, 8)

    bmi50 = _mbmi * ((1 + _lbmi * _sbmi * norm.ppf(0.50)) ** (1 / _lbmi))
    bmip50 = 100 * (bmi / bmi50)
    bmi95 = _mbmi * ((1 + _lbmi * _sbmi * norm.ppf(0.95)) ** (1 / _lbmi))
    bmip95 = 100 * (bmi / bmi95)
    bmi120 = 1.2 * bmi95

    if sex == 1:
        mref = 23.02029
        sref = 0.13454
    elif sex == 2:
        mref = 21.71700
        sref = 0.15297

    z1 = ((bmi / _mbmi) - 1) / _sbmi
    adj_perc_median = z1 * 100 * sref

    # calculations for extended BMIz and other metrics;
    original_bmiz = bmiz
    original_bmipct = bmipct
    agey = agemos / 12

    if sex == 1:
        sigma = 0.3728 + 0.5196 * agey - 0.0091 * agey**2
    elif sex == 2:
        sigma = 0.8334 + 0.3712 * agey - 0.0011 * agey**2

    if bmipct > 95:
        bmipct = 90 + 10 * (norm.cdf((bmi - bmi95) / sigma))
    if bmipct <= 99.999999999999992:
        bmiz = norm.ppf(bmipct / 100)
    if bmipct > 99.9999999 and pd.isnull(bmiz):
        bmiz = 8.21

    return bmiz, bmipct, z1, adj_perc_median

def get_bmiz_singlevalue(agemos,sex_m1f2,height_cm, weight_kg,ref=ref):
    # Set lengths for agemos and sex columns
    agemos = float(agemos)
    sex = int(sex_m1f2)
    height = float(height_cm)
    weight = float(weight_kg)
    bmi = weight / (height / 100) ** 2
    # headcir = mydata_row["headcir"].astype(float) if mydata_row["headcir"] else None

    _agecat = 0 if 0 <= agemos <= 0.5 else round(agemos + 0.5) - 0.5
    if bmi is None or bmi < 0:
        if weight > 0 and height > 0 and agemos >= 24:  # for 2-19 year
            bmi = weight / (height / 100) ** 2

    # begin the for-age calcs - note that this calls up the refdir
    ref = ref.sort_values(by=["SEX", "_AGECAT"])
    refp = ref[
        (ref["denom"] == "age") & (ref["SEX"] == sex) & (ref["_AGECAT"] == _agecat)
        ]
    ageint = refp["_AGEMOS2"].values[0] - refp["_AGEMOS1"].values[0]
    dage = agemos - refp["_AGEMOS1"].values[0]

    _llg = (
            refp["_LLG1"].values[0]
            + (dage * (refp["_LLG2"].values[0] - refp["_LLG1"].values[0])) / ageint
    )
    _mlg = (
            refp["_MLG1"].values[0]
            + (dage * (refp["_MLG2"].values[0] - refp["_MLG1"].values[0])) / ageint
    )
    _slg = (
            refp["_SLG1"].values[0]
            + (dage * (refp["_SLG2"].values[0] - refp["_SLG1"].values[0])) / ageint
    )
    _lht = (
            refp["_LHT1"].values[0]
            + (dage * (refp["_LHT2"].values[0] - refp["_LHT1"].values[0])) / ageint
    )
    _mht = (
            refp["_MHT1"].values[0]
            + (dage * (refp["_MHT2"].values[0] - refp["_MHT1"].values[0])) / ageint
    )
    _sht = (
            refp["_SHT1"].values[0]
            + (dage * (refp["_SHT2"].values[0] - refp["_SHT1"].values[0])) / ageint
    )
    _lwt = (
            refp["_LWT1"].values[0]
            + (dage * (refp["_LWT2"].values[0] - refp["_LWT1"].values[0])) / ageint
    )
    _mwt = (
            refp["_MWT1"].values[0]
            + (dage * (refp["_MWT2"].values[0] - refp["_MWT1"].values[0])) / ageint
    )
    _swt = (
            refp["_SWT1"].values[0]
            + (dage * (refp["_SWT2"].values[0] - refp["_SWT1"].values[0])) / ageint
    )
    _lhc = (
            refp["_LHC1"].values[0]
            + (dage * (refp["_LHC2"].values[0] - refp["_LHC1"].values[0])) / ageint
    )
    _mhc = (
            refp["_MHC1"].values[0]
            + (dage * (refp["_MHC2"].values[0] - refp["_MHC1"].values[0])) / ageint
    )
    _shc = (
            refp["_SHC1"].values[0]
            + (dage * (refp["_SHC2"].values[0] - refp["_SHC1"].values[0])) / ageint
    )
    _lbmi = (
            refp["_LBMI1"].values[0]
            + (dage * (refp["_LBMI2"].values[0] - refp["_LBMI1"].values[0])) / ageint
    )
    _mbmi = (
            refp["_MBMI1"].values[0]
            + (dage * (refp["_MBMI2"].values[0] - refp["_MBMI1"].values[0])) / ageint
    )
    _sbmi = (
            refp["_SBMI1"].values[0]
            + (dage * (refp["_SBMI2"].values[0] - refp["_SBMI1"].values[0])) / ageint
    )

    if agemos < 24:  # theres a valid value for 23.5 months!
        _mbmi = np.nan

    # lgz, lgpct, mod_lenz = _zscore(agemos, height, _llg, _mlg, _slg)
    # _bivlg = _cuts(mod_lenz, -5, 4)
    # stz, stpct, mod_statz = _zscore(agemos, height, _lht, _mht, _sht)
    # _bivst = _cuts(mod_statz, -5, 4)
    # waz, wapct, mod_waz = _zscore(agemos, weight, _lwt, _mwt, _swt)
    # _bivwt = _cuts(mod_waz, -5, 8)
    # headcz, headcpct, mod_headcz = _zscore(agemos, headcir, _lhc, _mhc, _shc)
    # _bivhc = _cuts(mod_headcz, -5, 5)
    bmiz, bmipct, mod_bmiz = _zscore(agemos, bmi, _lbmi, _mbmi, _sbmi)
    _bivbmi = _cuts(mod_bmiz, -4, 8)

    bmi50 = _mbmi * ((1 + _lbmi * _sbmi * norm.ppf(0.50)) ** (1 / _lbmi))
    bmip50 = 100 * (bmi / bmi50)
    bmi95 = _mbmi * ((1 + _lbmi * _sbmi * norm.ppf(0.95)) ** (1 / _lbmi))
    bmip95 = 100 * (bmi / bmi95)
    bmi120 = 1.2 * bmi95

    if sex == 1:
        mref = 23.02029
        sref = 0.13454
    elif sex == 2:
        mref = 21.71700
        sref = 0.15297

    z1 = ((bmi / _mbmi) - 1) / _sbmi
    adj_perc_median = z1 * 100 * sref

    # calculations for extended BMIz and other metrics;
    original_bmiz = bmiz
    original_bmipct = bmipct
    agey = agemos / 12

    if sex == 1:
        sigma = 0.3728 + 0.5196 * agey - 0.0091 * agey**2
    elif sex == 2:
        sigma = 0.8334 + 0.3712 * agey - 0.0011 * agey**2

    if bmipct > 95:
        bmipct = 90 + 10 * (norm.cdf((bmi - bmi95) / sigma))
    if bmipct <= 99.999999999999992:
        bmiz = norm.ppf(bmipct / 100)
    if bmipct > 99.9999999 and pd.isnull(bmiz):
        bmiz = 8.21

    return bmiz

def get_weightz(mydata_row, ref=ref):
    # begin for-length calcs, birth to 36 mos
    # Set lengths for agemos and sex columns
    agemos = mydata_row["agemos"].astype(float)
    sex = mydata_row["sex"].astype(int)
    height = mydata_row["height"].astype(float)
    weight = mydata_row["weight"].astype(float)
    bmi = mydata_row["bmi"].astype(float) if mydata_row["bmi"] else None
    headcir = mydata_row["headcir"].astype(float) if mydata_row["headcir"] else None

    if agemos < 24:
        length = height
        if length >= 45:
            _htcat = int(length + 0.5) - 0.5
        if 45 <= length < 45.5:
            _htcat = 45

        # begin the for-age calcs - note that this calls up the refdir
        refp = ref[
            (ref["denom"] == "length") & (ref["SEX"] == sex) & (ref["_htcat"] == _htcat)
        ]

        if 43 < length < 104:
            lenint = refp["_LG2"].values[0] - refp["_LG1"].values[0]
            dlen = length - refp["_LG1"].values[0]

            _lwl = (
                refp["_LWLG1"].values[0]
                + (dlen * (refp["_LWLG2"].values[0] - refp["_LWLG1"].values[0]))
                / lenint
            )
            _mwl = (
                refp["_MWLG1"].values[0]
                + (dlen * (refp["_MWLG2"].values[0] - refp["_MWLG1"].values[0]))
                / lenint
            )
            _swl = (
                refp["_SWLG1"].values[0]
                + (dlen * (refp["_SWLG2"].values[0] - refp["_SWLG1"].values[0]))
                / lenint
            )

            whz, whpct, mod_whz = _zscore(agemos, weight, _lwl, _mwl, _swl)
            _bivwht = _cuts(mod_whz, -4, 8)

    elif agemos >= 24:
        stand_ht = height
        if stand_ht >= 77.5:
            _htcat = int(stand_ht + 0.5) - 0.5
        elif 77 <= stand_ht < 77.5:
            _htcat = 77

        # begin the for-age calcs - note that this calls up the refdir
        refp = ref[
            (ref["denom"] == "height") & (ref["SEX"] == sex) & (ref["_htcat"] == _htcat)
        ]

        if 77 < height < 122:
            htint = refp["_HT2"].values[0] - refp["_HT1"].values[0]
            dht = height - refp["_HT1"].values[0]

            _lwh = (
                refp["_LWHT1"].values[0]
                + (dht * (refp["_LWHT2"].values[0] - refp["_LWHT1"].values[0])) / htint
            )
            _mwh = (
                refp["_MWHT1"].values[0]
                + (dht * (refp["_MWHT2"].values[0] - refp["_MWHT1"].values[0])) / htint
            )
            _swh = (
                refp["_SWHT1"].values[0]
                + (dht * (refp["_SWHT2"].values[0] - refp["_SWHT1"].values[0])) / htint
            )

            whz, whpct, mod_whz = _zscore(agemos, weight, _lwh, _mwh, _swh)
            _bivwht = _cuts(mod_whz, -4, 8)

    return whz, whpct, mod_whz, _bivwht
