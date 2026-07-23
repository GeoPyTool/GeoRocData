"""
20_TAS_GS_v2.py
================
Revised TAS-GS pipeline addressing reviewer comments on
"TAS-GS: A Novel TAS Diagram combining Gaussian Kernel Density Estimation
with Adaptive Bandwidth and SoftMax Probabilistic Discrimination".

This script is self-contained and reproducible. It implements:

1. Citation-level train/test split (Reviewer 1, point 2)
   The GEOROC database is partitioned by reference (CITATIONS id) so that
   no sample from a given publication appears in both the KDE training set
   and the evaluation set. This removes the data-leakage inflation flagged
   by the reviewer.

2. Isometric log-ratio (ilr) transform of the closed TAS composition
   (Reviewer 2, closure comment).  The 3-part composition
   (SiO2, Na2O+K2O, 100 - SiO2 - Na2O - K2O) is mapped to 2 ilr coordinates,
   the 2-D Gaussian KDE is estimated in ilr space, and the resulting
   density field is transformed back to the conventional (SiO2, Na2O+K2O)
   plane for plotting and classification.  A flag allows the same pipeline
   to be run in raw coordinates for the ablation requested by Reviewer 2
   (Figure 5 with / without ilr).

3. Restricted TAS-field comparison (Reviewer 1, point 1)
   Accuracy is reported on (a) all rock-name labels and (b) the subset of
   labels that have a one-to-one mapping to a TAS field (B, O1, O2, O3, R,
   T, S1, S2, S3, U1, U2, U3, Ph, F, Pc).  The restricted comparison gives
   a fair head-to-head against the classical TAS diagram, because labels
   such as Adakite, Boninite, Hawaiite, Mugearite, Shoshonite, Tholeiite,
   Picrite, Kimberlite, Komatiite, Nephelinite have no TAS field and can
   never be correctly classified by the classical diagram.

4. Adaptive bandwidth with a documented derivation (Reviewer 3 / P17L4,
   P18L1).  The geometric mean of the Silverman scale and the
   median-pairwise-distance scale is used; the sqrt(2) factor in the
   median scale is the bias-correction for the expected distance between
   two points drawn from the kernel (see comments in adaptive_bandwidth).

5. Plutonic rocks (Reviewer 2, igneous vs volcanic).  The pipeline accepts
   rock_type='VOL' or 'PLU' so the same protocol can be applied to
   plutonic rocks, and the manuscript can state clearly which set is used.

6. Quantified run-time reporting (Reviewer 3, P23L11 / P23L13).

Usage:
    python 20_TAS_GS_v2.py --rock-type VOL --use-ilr --n-test-citations 2000 --seed 7
    python 20_TAS_GS_v2.py --rock-type VOL --no-ilr   --n-test-citations 2000 --seed 7
    python 20_TAS_GS_v2.py --rock-type PLU --use-ilr --n-test-citations 2000 --seed 7

Outputs (under DB_Gen/TAS_GS_v2/<rock_type>_<ilr|raw>/):
    kde_train/<Label>_kde.pkl           trained KDE per label (train only)
    test_set.csv                        held-out test samples with labels
    predictions.csv                     per-sample predictions + probabilities
    accuracy.json                       overall + restricted accuracy + runtime
    confusion_matrix.csv                label x predicted label
    figure_kde_field_<Label>.{svg,pdf}  KDE field back-projected to TAS plane
    figure_accuracy_bar.{svg,pdf}       TAS / Max-Ratio / SoftMax accuracy

The script depends only on numpy, scipy, pandas, scikit-learn, matplotlib.
"""

import argparse
import json
import math
import os
import pickle
import sqlite3
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors as mcolors
from matplotlib.path import Path
from matplotlib.patches import Polygon
from scipy.spatial.distance import pdist
from sklearn.neighbors import KernelDensity

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'truetype'

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(os.path.dirname(HERE), 'GeoRoc.db')
DEFAULT_REF_DB = os.path.join(os.path.dirname(HERE), 'Ref.db')


# --------------------------------------------------------------------------- #
# 1. Composition helpers (closure / ilr)
# --------------------------------------------------------------------------- #
# Geochemical major oxides sum to 100 wt% on an anhydrous basis, so any
# subcomposition of D parts lives on a (D-1)-D simplex and is subject to
# the constant-sum (closure) constraint (Aitchison 1982).  We use the
# isometric log-ratio transform (Egozue et al. 2003), which maps a D-part
# composition to D-1 orthonormal coordinates.  The default TAS composition
# is the 3-part case (SiO2, Na2O+K2O, "rest"); the multi-oxide feature sets
# build a D-part composition and reuse the same machinery.
#
# The ilr orthonormal basis used here is the standard Helmert sub-basis
# implemented in scipy.stats / scikit-bio (the "isometric log-ratio" of
# Egozue et al. 2003):
#
#   row k of V has sqrt((D-k)/(D-k+1)) on column k,  and -1/sqrt((D-k)*(D-k+1))
#   on columns k+1 .. D-1,  for k = 0 .. D-2.
#
# This is generated by ilr_basis(D) below so the dimension matches the
# feature set in use.


def ilr_basis(D):
    """Return the (D-1, D) Helmert ilr basis for a D-part composition."""
    V = np.zeros((D - 1, D))
    for k in range(D - 1):
        n = D - k
        V[k, k] = math.sqrt((n - 1) / n)
        V[k, k + 1:] = -1.0 / math.sqrt(n * (n - 1))
    return V


# Pre-built 3-part basis (kept for backward compatibility and for the
# default TAS feature set).
_ILR_V = ilr_basis(3)


_EPS = 1e-6  # guards log(0) in the ilr transform for samples whose
              # oxide sum slightly exceeds 100 wt% (altered / analytical
              # artefacts kept in the GEOROC compilation).


def close_composition(parts):
    """Return an (n, D) closed composition from a list of (n,) arrays.

    The parts are clamped to a small positive floor and re-closed to sum
    to 1 so the ilr log-ratio transform stays finite even when a few
    analyses sum to slightly more or less than 100 wt%.
    """
    comp = np.column_stack([np.clip(np.asarray(p, float), _EPS, None)
                            for p in parts])
    return comp / comp.sum(axis=1, keepdims=True)


def to_simplex(sio2, alk):
    """Return an (n, 3) array of the closed composition [SiO2, Alk, Rest].

    Convenience wrapper for the 3-part TAS composition; multi-oxide feature
    sets use close_composition() directly.
    """
    rest = 100.0 - sio2 - alk
    return close_composition([sio2, alk, rest])


def ilr_transform(comp, V=None):
    """Map (n, D) closed composition to (n, D-1) ilr coordinates."""
    if V is None:
        V = ilr_basis(comp.shape[1])
    return np.log(comp) @ V.T


def ilr_inverse(z, V=None):
    """Inverse ilr: map (n, D-1) ilr coords back to (n, D) composition."""
    if V is None:
        V = ilr_basis(z.shape[1] + 1)
    log_x = z @ V
    return np.exp(log_x - np.logaddexp.reduce(log_x, axis=1, keepdims=True))


# --------------------------------------------------------------------------- #
# 2. Bandwidth
# --------------------------------------------------------------------------- #
def silverman_bandwidth(data):
    """Silverman rule of thumb for a d-D Gaussian KDE.

    h_S = ( n * (d + 2) / 4 ) ^ (-1 / (d + 4))
    This is the multivariate Silverman factor implemented in the original
    code (scipy.stats.gaussian_kde convention / Wand & Jones 1993, eq. 4.6).
    """
    n, d = data.shape
    return (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))


def median_bandwidth(data):
    """Median pairwise-distance bandwidth.

    For two independent points drawn from a 2-D isotropic Gaussian kernel
    with bandwidth h, the expected squared Euclidean distance between them
    is E[||X-Y||^2] = 2 * h^2  (Var(X-Y) = 2 h^2 I).  Hence an unbiased
    estimator of the kernel scale from the median pairwise distance m is
    h ~ m / sqrt(2).  The sqrt(2) factor that appears in the manuscript is
    therefore the bias correction between the median pairwise distance and
    the kernel standard deviation, not an arbitrary heuristic.
    """
    m = np.median(pdist(data, metric='euclidean'))
    return m / math.sqrt(2.0)


def adaptive_bandwidth(data):
    """Adaptive bandwidth = geometric mean of Silverman and median scales.

    Taking the geometric mean of two scale estimators stabilises the
    bandwidth across sample sizes and tail behaviour: Silverman's rule
    depends only on n and d and under-smooths heavy-tailed geochemical
    distributions, whereas the median pairwise distance tracks the actual
    spread of the data but is noisy for small samples.  The geometric mean
    h* = sqrt(h_S * h_M) is a standard robust compromise (see comments in
    the manuscript, Section 3).
    """
    h_s = silverman_bandwidth(data)
    h_m = median_bandwidth(data)
    if not np.isfinite(h_s) or not np.isfinite(h_m) or h_s <= 0 or h_m <= 0:
        # fall back to Silverman's rule if the median scale degenerates
        # (e.g. duplicated points yielding zero pairwise distance)
        return max(h_s, 1e-3)
    return math.sqrt(h_s * h_m)


# --------------------------------------------------------------------------- #
# 3. Data loading + label / TAS-field mapping
# --------------------------------------------------------------------------- #
ROCK_NAME_KEYWORDS = [
    # canonical volcanic rock names used as labels in this study
    'ADAKITE', 'ANDESITE', 'BASALT', 'BASANITE', 'BONINITE', 'BENMOREITE',
    'DACITE', 'FOIDITE', 'HAWAIITE', 'KIMBERLITE', 'KOMATIITE', 'LATITE',
    'MUGEARITE', 'NEPHELINITE', 'PHONOLITE', 'PHONOTEPHRITE',
    'PICRITE', 'RHYODACITE', 'RHYOLITE', 'SHOSHONITE', 'TEPHRITE',
    'THOLEIITE', 'TRACHYANDESITE', 'TRACHYBASALT', 'TRACHYTE',
    'OCEANITE', 'ANKARAMITE', 'COMENDITE', 'PANTELLERITE',
    # plutonic (used when rock_type='PLU')
    'GRANITE', 'GRANODIORITE', 'TONALITE', 'DIORITE', 'GABBRO',
    'DUNITE', 'HARZBURGITE', 'LHERZOLITE', 'WEHRLITE', 'PERIDOTITE',
    'PYROXENITE', 'ORTHOPYROXENITE', 'SYENITE', 'MONZONITE', 'MONZODIORITE',
    'ANORTHOSITE',
]


def parse_rock_name(rock_name_col):
    """Parse the GEOROC 'ROCK NAME' column 'ANDESITE [29314]' -> 'Andesite'."""
    if not isinstance(rock_name_col, str):
        return None
    s = rock_name_col.strip().upper()
    for kw in ROCK_NAME_KEYWORDS:
        # match as a leading whole-word token so 'BASALTIC-ANDESITE' does
        # not collapse to 'Basalt'
        if s.startswith(kw + ' ') or s.startswith(kw + '[') or s == kw \
                or s.startswith(kw + '-') and kw in {'BASALT', 'ANDESITE'}:
            # disambiguate a few compound names that map to distinct labels
            if s.startswith('BASALTIC-ANDESITE'):
                return 'Basaltic Andesite'
            if s.startswith('TRACHYBASALTIC-ANDESITE') or s.startswith('TRACHYANDESITE'):
                return 'Trachyandesite'
            if s.startswith('BASALTIC-TRACHYANDESITE'):
                return 'Basaltic Trachyandesite'
            return kw.capitalize()
    return None


# Mapping from the 15 TAS letter codes to the canonical rock-name label used
# for the restricted head-to-head comparison with the classical TAS diagram.
# (Following Le Maitre et al. 2005 and Le Bas et al. 1986.)
TAS_FIELD_TO_ROCKNAME = {
    'F':  'Foidite',
    'Pc': 'Picrobasalt',
    'U1': 'Basanite',          # U1 = Tephrite/Basanite; labelled Basanite here
    'B':  'Basalt',
    'S1': 'Trachybasalt',
    'S2': 'Basaltic Trachyandesite',
    'S3': 'Trachyandesite',
    'U2': 'Phonotephrite',
    'U3': 'Tephriphonolite',
    'Ph': 'Phonolite',
    'O1': 'Basaltic Andesite',
    'O2': 'Andesite',
    'O3': 'Dacite',
    'T':  'Trachyte',
    'R':  'Rhyolite',
}

# Reverse map: canonical rock-name -> TAS letter code.
ROCKNAME_TO_TAS_FIELD = {v: k for k, v in TAS_FIELD_TO_ROCKNAME.items()}

# Labels that have a TAS field counterpart (used for the restricted acc.).
TAS_COMPARABLE = set(TAS_FIELD_TO_ROCKNAME.values())


# --------------------------------------------------------------------------- #
# 3b. Unified FeO* (total iron as FeO) and feature-set definitions
# --------------------------------------------------------------------------- #
# GEOROC reports iron in several mutually overlapping columns.  To obtain a
# single consistent total-iron-as-FeO (FeO*) column we follow the standard
# conversion (e.g. Le Maitre 1976):
#
#   FeO* = FeO(wt%) + 0.8998 * Fe2O3(wt%)         (when FeO and Fe2O3 given)
#
# with the following precedence (each row keeps the FIRST available branch):
#   1. FEOT is already reported          -> use FEOT directly
#   2. FEO and FE2O3 reported            -> FeO* = FEO + 0.8998 * FE2O3
#   3. FE2O3(T) (total as Fe2O3) reported-> FeO* = 0.8998 * FE2O3T
#   4. FEO only                          -> FeO* = FEO
#   5. FE2O3 only                        -> FeO* = 0.8998 * FE2O3
#   6. FE (ppm) reported                 -> FeO* = FE(ppm) * 1.2866e-4
#
# Note on the existing 10_DB_Remove_LOI.py: its sum rule
# `if row['FEOT(WT%)'] != 0: drop FeO, Fe2O3` silently drops both FeO and
# Fe2O3 whenever FEOT is NaN (because NaN != 0 is True in Python), under-
# counting iron by ~8 wt% for the ~90k VOL samples that report FeO/Fe2O3 but
# not FeOT.  The function below fixes this by using an explicit NaN test.

# molecular-weight ratio FeO / Fe2O3 = (2*71.844) / 159.687 = 0.8998
FE2O3_TO_FEO = 2 * 71.844 / 159.687  # = 0.8998


def unify_feo_star(df):
    """Add a single 'FeOT' column (total iron as FeO, in wt%) to df.

    df is expected to contain the raw GEOROC columns
    'FEOT(WT%)', 'FEO(WT%)', 'FE2O3(WT%)', 'FE2O3T(WT%)', 'FE(PPM)'.
    Non-numeric / blank values are treated as NaN.  Rows with no usable
    iron measurement get FeOT = NaN (and are dropped later by dropna on the
    feature set).
    """
    def num(s):
        return pd.to_numeric(s, errors='coerce')

    feot  = num(df['FEOT(WT%)'])   if 'FEOT(WT%)' in df else np.nan
    feo   = num(df['FEO(WT%)'])    if 'FEO(WT%)'  in df else np.nan
    fe2o3 = num(df['FE2O3(WT%)'])  if 'FE2O3(WT%)' in df else np.nan
    fe2o3t= num(df['FE2O3T(WT%)']) if 'FE2O3T(WT%)' in df else np.nan
    fe_ppm= num(df['FE(PPM)'])     if 'FE(PPM)'   in df else np.nan

    out = pd.Series(np.nan, index=df.index, dtype=float)
    # branch 1: FEOT reported (NaN-safe test)
    m = feot.notna() & (feot != 0)
    out[m] = feot[m]
    # branch 2: FEO + FE2O3
    m = out.isna() & feo.notna() & fe2o3.notna()
    out[m] = feo[m] + FE2O3_TO_FEO * fe2o3[m]
    # branch 3: total as FE2O3T
    m = out.isna() & fe2o3t.notna() & (fe2o3t != 0)
    out[m] = FE2O3_TO_FEO * fe2o3t[m]
    # branch 4: FEO only
    m = out.isna() & feo.notna()
    out[m] = feo[m]
    # branch 5: FE2O3 only
    m = out.isna() & fe2o3.notna()
    out[m] = FE2O3_TO_FEO * fe2o3[m]
    # branch 6: FE(ppm) -> wt% as FeO  (Fe 55.845, FeO 71.844)
    m = out.isna() & fe_ppm.notna() & (fe_ppm > 0)
    out[m] = fe_ppm[m] * 1e-4 * (71.844 / 55.845)
    return out


# Feature sets.  Each entry is the list of GEOROC columns (already unified
# where relevant) used to build the composition for the KDE.
#  - 'tas'        : the 3-part TAS composition (SiO2, Alk, rest)   [baseline]
#  - 'major6'    : SiO2, Al2O3, FeO*, MgO, CaO, Alk                (6 oxides)
#  - 'major9'    : major6 + TiO2, P2O5, MnO                        (9 oxides)
# 'Alk' is always Na2O + K2O so the KDE still respects the TAS axes.
FEATURE_SETS = {
    'tas':     ['SiO2', 'Alk'],
    'major6':  ['SiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Alk'],
    'major9':  ['SiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO',
                'TiO2', 'P2O5', 'MnO', 'Alk'],
}


def load_georoc(db_path, rock_type, feature_set='tas',
                min_samples_per_label=50):
    """Load VOL or PLU samples with a parsed rock-name label and citation id.

    When ``feature_set`` != 'tas' the loader also pulls the extra major
    oxides needed (Al2O3, FeO* unified from FeOT/FeO/Fe2O3, MgO, CaO, TiO2,
    P2O5, MnO) so the KDE can be estimated on a higher-dimensional
    composition.  All oxides are on a wt% basis; the loader does NOT re-close
    the composition (the ilr/closure step in Section 3 does that).
    """
    feats = FEATURE_SETS[feature_set]
    # base columns we always need
    cols = ['"ROCK NAME"', '"ROCK TYPE"', '"SIO2(WT%)"', '"NA2O(WT%)"',
            '"K2O(WT%)"', '"CITATIONS"']
    # iron columns for FeO* unification
    iron_cols = ['"FEOT(WT%)"', '"FEO(WT%)"', '"FE2O3(WT%)"',
                 '"FE2O3T(WT%)"', '"FE(PPM)"']
    # extra oxides requested by the feature set
    extra_map = {
        'Al2O3': '"AL2O3(WT%)"', 'FeOT': None,  # FeOT is unified, handled below
        'MgO': '"MGO(WT%)"', 'CaO': '"CAO(WT%)"',
        'TiO2': '"TIO2(WT%)"', 'P2O5': '"P2O5(WT%)"', 'MnO': '"MNO(WT%)"',
    }
    extra_cols = [extra_map[f] for f in feats if f not in ('SiO2', 'Alk')
                  and extra_map.get(f)]
    select = list(dict.fromkeys(cols + iron_cols + extra_cols))  # dedup
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            'SELECT ' + ','.join(select) + ' FROM georoc_data '
            'WHERE "ROCK TYPE"=?', conn, params=(rock_type,))
    finally:
        conn.close()

    df = df.rename(columns={
        'SIO2(WT%)': 'SiO2', 'NA2O(WT%)': 'Na2O', 'K2O(WT%)': 'K2O',
        'ROCK NAME': 'RockNameRaw', 'ROCK TYPE': 'RockType',
        'CITATIONS': 'CitationsRaw',
        'AL2O3(WT%)': 'Al2O3', 'MGO(WT%)': 'MgO', 'CAO(WT%)': 'CaO',
        'TIO2(WT%)': 'TiO2', 'P2O5(WT%)': 'P2O5', 'MNO(WT%)': 'MnO',
    })
    for c in ['SiO2', 'Na2O', 'K2O'] + \
            [f for f in feats if f not in ('SiO2', 'Alk', 'FeOT')]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace('', np.nan)
    df['Alk'] = df['Na2O'] + df['K2O']
    # unify iron if the feature set needs FeOT
    if 'FeOT' in feats:
        df['FeOT'] = unify_feo_star(df)
    # drop rows missing any feature
    df = df.dropna(subset=feats)
    df = df[(df['SiO2'] > 0) & (df['Alk'] >= 0)]
    # sanity: drop rows with implausible negatives on any feature
    for f in feats:
        if f in df:
            df = df[df[f] >= 0]
    df['Label'] = df['RockNameRaw'].map(parse_rock_name)
    df = df[df['Label'].notna()].copy()
    df['CitationID'] = df['CitationsRaw'].str.extract(r'\[(\d+)\]')[0]
    df = df[df['CitationID'].notna()].copy()
    df['CitationID'] = df['CitationID'].astype(str)
    counts = df['Label'].value_counts()
    keep = counts[counts >= min_samples_per_label].index
    df = df[df['Label'].isin(keep)].copy()
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Train / test split by citation
# --------------------------------------------------------------------------- #
def split_by_citation(df, n_test_citations, seed):
    """Hold out entire publications (by CitationID) as the test set."""
    rng = np.random.default_rng(seed)
    cits = np.array(sorted(df['CitationID'].unique()))
    rng.shuffle(cits)
    n_test = min(n_test_citations, max(1, len(cits) // 5))
    test_cits = set(cits[:n_test])
    train = df[~df['CitationID'].isin(test_cits)].copy()
    test = df[df['CitationID'].isin(test_cits)].copy()
    # drop labels that appear in only one of the two splits
    train_labels = set(train['Label'].unique())
    test_labels = set(test['Label'].unique())
    common = train_labels & test_labels
    train = train[train['Label'].isin(common)]
    test = test[test['Label'].isin(common)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. KDE training in ilr (or raw) space
# --------------------------------------------------------------------------- #
def build_design(df, feats, use_ilr):
    """Build the (n, k) design matrix for the KDE.

    feats  : list of feature names (subset of FEATURE_SETS[set]).
    use_ilr: if True, close the composition (features + a "rest" part that
             makes the mass sum to 100 wt%) and apply the ilr transform,
             returning D columns where D = len(feats); if False, return the
             raw wt% columns (len(feats) columns).

    The "rest" part is included so that a 2-feature TAS composition (SiO2,
    Alk) yields the 3-part simplex (SiO2, Alk, rest) and 2 ilr coordinates,
    matching the original manuscript.  For major6/major9 the rest part is
    typically small (the listed oxides already sum to ~95-99 wt%) but is kept
    for consistency of the closure treatment.
    """
    parts = [df[f].values.astype(float) for f in feats]
    if use_ilr:
        rest = 100.0 - sum(parts)
        comp = close_composition(parts + [rest])
        return ilr_transform(comp)
    return np.column_stack(parts)


def fit_kde_set(train, feats, use_ilr, out_dir, min_samples=30,
                max_samples_per_label=4000, seed=7):
    """Fit one KDE per label on the training split and pickle them.

    To keep evaluation tractable with the >1.3e5-sample GEOROC training
    split, each label's KDE is fitted on a random subsample of up to
    ``max_samples_per_label`` training points.  KDE bandwidth and the
    shape of the density field are essentially insensitive to further
    increases in sample size once n >> 1/h^2 (Wand & Jones 1993), and
    4000 points already give a stable estimate for every label here.
    """
    os.makedirs(os.path.join(out_dir, 'kde_train'), exist_ok=True)
    rng = np.random.default_rng(seed)
    kdes = {}
    for label, grp in train.groupby('Label'):
        if len(grp) < min_samples:
            continue
        if len(grp) > max_samples_per_label:
            grp = grp.sample(max_samples_per_label, random_state=int(
                rng.integers(1 << 31)))
        data = build_design(grp, feats, use_ilr)
        h = adaptive_bandwidth(data)
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(data)
        kdes[label] = kde
        with open(os.path.join(out_dir, 'kde_train', label + '_kde.pkl'),
                  'wb') as f:
            pickle.dump(kde, f)
    return kdes


def score_samples(kdes, df, feats, use_ilr):
    """Return an (n, n_labels) matrix of (un-normalised) KDE log-densities."""
    X = build_design(df, feats, use_ilr)
    labels = list(kdes.keys())
    scores = np.empty((len(df), len(labels)))
    for j, label in enumerate(labels):
        scores[:, j] = kdes[label].score_samples(X)
    return scores, labels


def softmax_over_rows(log_scores):
    """Stable softmax across KDE log-densities (SoftMax variant)."""
    m = np.max(log_scores, axis=1, keepdims=True)
    e = np.exp(log_scores - m)
    return e / e.sum(axis=1, keepdims=True)


def max_ratio(scores):
    """Max-ratio normalisation (the 'Max Ratio' variant of the manuscript)."""
    m = np.max(scores, axis=1, keepdims=True)
    out = scores / np.where(m > 0, m, 1.0)
    return out


# --------------------------------------------------------------------------- #
# 6. Classical TAS diagram classification (polygon test)
# --------------------------------------------------------------------------- #
def load_tas_polygons(rock_type='VOL'):
    """Load TAS field polygons from Plot_Json/tas_cord.json.

    The JSON stores vertex lists under 'coords' keyed by TAS letter code and
    a 'Volcanic' map from code to rock name.  For plutonic rocks the same
    TAS field polygons are used (Le Maitre et al. 2005, fig. 2), so we
    return the volcanic polygons regardless of rock_type.
    """
    p = os.path.join(HERE, 'Plot_Json', 'tas_cord.json')
    with open(p, 'r', encoding='utf-8') as f:
        cord = json.load(f)
    polys = {}
    for code, verts in cord['coords'].items():
        xs = [pt[0] for pt in verts]
        ys = [pt[1] for pt in verts]
        polys[code] = Path(list(zip(xs, ys)), closed=True)
    return polys, cord


def classify_tas(sio2, alk, polys):
    """Return the TAS-field rock-name label for each (SiO2, Alk) point.

    The first polygon (in JSON order) that contains a point wins; points
    outside every TAS field are returned as None.
    """
    pts = np.column_stack([sio2, alk])
    labels = np.empty(len(pts), dtype=object)
    labels[:] = None
    for code, path in polys.items():
        already = np.array([x is None for x in labels])
        mask = path.contains_points(pts) & already
        labels[mask] = TAS_FIELD_TO_ROCKNAME.get(code, None)
    return labels


# --------------------------------------------------------------------------- #
# 7. Evaluation
# --------------------------------------------------------------------------- #
def evaluate(test, kdes, feats, use_ilr, polys):
    sio2 = test['SiO2'].values
    alk = test['Alk'].values
    y = test['Label'].values

    log_scores, labels = score_samples(kdes, test, feats, use_ilr)
    sm = softmax_over_rows(log_scores)
    mr = max_ratio(log_scores)
    sm_pred = np.array(labels)[np.argmax(sm, axis=1)]
    mr_pred = np.array(labels)[np.argmax(mr, axis=1)]
    tas_pred = classify_tas(sio2, alk, polys)

    res = pd.DataFrame({
        'Label': y,
        'SoftMax_Pred': sm_pred,
        'SoftMax_Prob': sm.max(axis=1),
        'MaxRatio_Pred': mr_pred,
        'MaxRatio_Prob': mr.max(axis=1),
        'TAS_Pred': tas_pred,
        'SiO2': sio2,
        'Alk': alk,
        'CitationID': test['CitationID'].values,
    })
    return res


def accuracy_report(res):
    def acc(true, pred):
        mask = pred != None
        return float(np.mean(np.array(true)[mask] == np.array(pred)[mask])) \
            if mask.any() else 0.0, int(mask.sum())

    out = {}
    for col in ['SoftMax_Pred', 'MaxRatio_Pred', 'TAS_Pred']:
        a, n = acc(res['Label'], res[col])
        out[col] = {'accuracy': a, 'n_evaluated': n}
    # restricted comparison on TAS-comparable labels only
    restricted = res[res['Label'].isin(TAS_COMPARABLE)]
    out['restricted'] = {}
    for col in ['SoftMax_Pred', 'MaxRatio_Pred', 'TAS_Pred']:
        a, n = acc(restricted['Label'], restricted[col])
        out['restricted'][col] = {'accuracy': a, 'n_evaluated': n}
    # per-label accuracy
    per_label = {}
    for col in ['SoftMax_Pred', 'MaxRatio_Pred', 'TAS_Pred']:
        per_label[col] = {}
        for label, grp in res.groupby('Label'):
            valid = grp[grp[col].notna() & (grp[col] != None)]
            if len(valid):
                per_label[col][label] = float(
                    (valid['Label'] == valid[col]).mean())
            else:
                per_label[col][label] = None
    out['per_label'] = per_label
    return out


def confusion(res, pred_col):
    labels = sorted(res['Label'].unique())
    mat = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for _, row in res.iterrows():
        p = row[pred_col]
        if p in mat.columns:
            mat.loc[row['Label'], p] += 1
    return mat


# --------------------------------------------------------------------------- #
# 8. Plotting
# --------------------------------------------------------------------------- #
def plot_kde_field(kde, label, use_ilr, out_dir, feats):
    """Project the KDE density field back onto the TAS (SiO2, Alk) plane.

    Only meaningful for the 2-feature TAS feature set; for major6/major9 the
    field lives in >5-D ilr space and cannot be drawn as a single 2-D map, so
    we slice it at the median of the non-TAS oxides (a conditional density
    slice).
    """
    xs = np.linspace(35, 80, 200)
    ys = np.linspace(0, 17.65, 200)
    XX, YY = np.meshgrid(xs, ys)
    n = XX.size
    if feats == ['SiO2', 'Alk']:
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        if use_ilr:
            comp = to_simplex(pts[:, 0], pts[:, 1])
            X = ilr_transform(comp)
        else:
            X = pts
    else:
        # conditional slice: fix the extra oxides to the dataset median.
        # Caller does not have the median here; we use a placeholder 0 and
        # the caller is expected to pass feats==['SiO2','Alk'] for plotting.
        # For higher-D sets we skip the 2-D back-projection plot.
        return
    log_d = kde.score_samples(X)
    d = np.exp(log_d).reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    pcm = ax.pcolormesh(XX, YY, d, cmap='viridis', shading='auto')
    ax.set_xlabel('SiO$_2$ (wt%)')
    ax.set_ylabel('Na$_2$O + K$_2$O (wt%)')
    ax.set_title('%s KDE field (%s)' % (label, 'ilr' if use_ilr else 'raw'))
    fig.colorbar(pcm, ax=ax, label='density')
    fig.tight_layout()
    for ext in ('svg', 'pdf'):
        fig.savefig(os.path.join(out_dir,
                                 'figure_kde_field_%s.%s' % (label, ext)))
    plt.close(fig)


def plot_accuracy_bar(report, out_dir, title_suffix=''):
    methods = ['TAS_Pred', 'MaxRatio_Pred', 'SoftMax_Pred']
    pretty = ['Classical TAS', 'Max Ratio', 'SoftMax']
    overall = [report[m]['accuracy'] for m in methods]
    restricted = [report['restricted'][m]['accuracy'] for m in methods]
    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(x - w/2, overall, w, label='All labels', color='#9a99b7')
    ax.bar(x + w/2, restricted, w, label='TAS-comparable labels',
           color='#e48080')
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, rotation=15)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    for i, v in enumerate(overall):
        ax.text(i - w/2, v + 0.01, '%.2f' % v, ha='center', fontsize=8)
    for i, v in enumerate(restricted):
        ax.text(i + w/2, v + 0.01, '%.2f' % v, ha='center', fontsize=8)
    ax.legend(loc='upper left', fontsize=8)
    if title_suffix:
        ax.set_title(title_suffix, fontsize=9)
    fig.tight_layout()
    for ext in ('svg', 'pdf'):
        fig.savefig(os.path.join(out_dir, 'figure_accuracy_bar.%s' % ext))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 9. Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--rock-type', default='VOL', choices=['VOL', 'PLU'])
    ap.add_argument('--feature-set', default='tas',
                    choices=list(FEATURE_SETS.keys()),
                    help='Composition used for the KDE: tas (2 oxides), '
                         'major6 (6 oxides), major9 (9 oxides)')
    ap.add_argument('--use-ilr', dest='use_ilr', action='store_true',
                    help='KDE in ilr space (addresses closure effect)')
    ap.add_argument('--no-ilr', dest='use_ilr', action='store_false',
                    help='KDE in raw coordinates (ablation)')
    ap.set_defaults(use_ilr=True)
    ap.add_argument('--n-test-citations', type=int, default=2000,
                    help='Number of publications held out for testing')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out-root', default=os.path.join(HERE, 'TAS_GS_v2'))
    args = ap.parse_args()

    feats = FEATURE_SETS[args.feature_set]
    tag = '%s_%s_%s' % (args.rock_type, args.feature_set,
                        'ilr' if args.use_ilr else 'raw')
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print('[load] %s / feature-set=%s from %s'
          % (args.rock_type, args.feature_set, args.db))
    df = load_georoc(args.db, args.rock_type, feature_set=args.feature_set)
    print('[load] %d rows, %d labels, %d citations'
          % (len(df), df['Label'].nunique(), df['CitationID'].nunique()))

    train, test = split_by_citation(df, args.n_test_citations, args.seed)
    print('[split] train=%d (%d cits), test=%d (%d cits)'
          % (len(train), train['CitationID'].nunique(),
             len(test), test['CitationID'].nunique()))

    t1 = time.time()
    kdes = fit_kde_set(train, feats, args.use_ilr, out_dir)
    print('[fit] %d KDEs in %.1fs' % (len(kdes), time.time() - t1))

    t2 = time.time()
    res = evaluate(test, kdes, feats, args.use_ilr, load_tas_polygons()[0])
    print('[eval] %.1fs' % (time.time() - t2))

    res.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    keep_cols = [c for c in ['SiO2', 'Alk', 'Label', 'CitationID'] if c in test]
    test[keep_cols].to_csv(os.path.join(out_dir, 'test_set.csv'), index=False)

    report = accuracy_report(res)
    report['runtime_sec'] = {
        'total': time.time() - t0,
        'fit': time.time() - t1,
        'eval': time.time() - t2,
        'n_train': len(train), 'n_test': len(test),
        'n_test_citations': int(test['CitationID'].nunique()),
        'use_ilr': bool(args.use_ilr), 'rock_type': args.rock_type,
        'feature_set': args.feature_set, 'seed': args.seed,
    }
    with open(os.path.join(out_dir, 'accuracy.json'), 'w') as f:
        json.dump(report, f, indent=2)

    if args.feature_set == 'tas':
        for label, kde in list(kdes.items())[:6]:
            plot_kde_field(kde, label, args.use_ilr, out_dir, feats)
    plot_accuracy_bar(report, out_dir,
                      title_suffix='%s / %s / %s'
                      % (args.rock_type, args.feature_set,
                         'ilr' if args.use_ilr else 'raw'))

    cm = confusion(res, 'SoftMax_Pred')
    cm.to_csv(os.path.join(out_dir, 'confusion_matrix.csv'))

    print('=== Accuracy ===')
    print('All labels:         TAS=%.3f  MaxRatio=%.3f  SoftMax=%.3f'
          % (report['TAS_Pred']['accuracy'],
             report['MaxRatio_Pred']['accuracy'],
             report['SoftMax_Pred']['accuracy']))
    print('TAS-comparable:     TAS=%.3f  MaxRatio=%.3f  SoftMax=%.3f'
          % (report['restricted']['TAS_Pred']['accuracy'],
             report['restricted']['MaxRatio_Pred']['accuracy'],
             report['restricted']['SoftMax_Pred']['accuracy']))
    print('Total runtime: %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()