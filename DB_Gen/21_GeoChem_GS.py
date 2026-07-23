"""
21_GeoChem_GS.py
================
Stage 3 of the evolution roadmap (see Evolution_Roadmap.md):
GeoChem-GS — whole-rock Gaussian-SoftMax.

Extends the closure-aware TAS-GS of `20_TAS_GS_v2.py` to the full
major-oxide + trace-element composition, using two ilr-transformed
compositional blocks (major oxides on the wt% simplex, trace elements on
the ppm simplex), a per-dimension adaptive bandwidth, and a
label-confidence filter that drops systematically mis-labelled training
analyses (Mahalanobis outlier rule of Rousseeuw & Van Zomeren 1990)
without touching the test set.

Usage:
    python 21_GeoChem_GS.py --rock-type VOL --n-test-citations 1500 --seed 7
    python 21_GeoChem_GS.py --rock-type VOL --trace-variant ilr \
        --n-test-citations 1500 --seed 7
    python 21_GeoChem_GS.py --rock-type VOL --no-trace     # major oxides only

Outputs (under DB_Gen/TAS_GS_v2/<rock_type>_geochem_<variant>/):
    accuracy.json, predictions.csv, test_set.csv, confusion_matrix.csv,
    figure_accuracy_bar.{svg,pdf}, per_label_cleanup.csv
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
from matplotlib.path import Path
from sklearn.neighbors import KernelDensity

# Reuse everything we built in stage 2.
HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'tasgs', os.path.join(HERE, '20_TAS_GS_v2.py'))
tasgs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tasgs)
import sys
sys.modules['tasgs'] = tasgs

from tasgs import (  # noqa: E402
    ilr_basis, ilr_transform, close_composition, unify_feo_star,
    parse_rock_name, split_by_citation, classify_tas, load_tas_polygons,
    softmax_over_rows, max_ratio, accuracy_report, confusion,
    plot_accuracy_bar, DEFAULT_DB, FE2O3_TO_FEO,
)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'truetype'


# --------------------------------------------------------------------------- #
# Feature definitions
# --------------------------------------------------------------------------- #
# Major oxides (wt%, anhydrous basis).  'Alk' is always Na2O + K2O so the
# TAS axes are recoverable.  FeOT is the unified total-iron-as-FeO column
# produced by unify_feo_star().
MAJOR_OXIDES = ['SiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O',
                'TiO2', 'P2O5', 'MnO']

# GEOROC column -> canonical short name.
OXIDE_COL_MAP = {
    'SiO2': 'SIO2(WT%)', 'Al2O3': 'AL2O3(WT%)', 'MgO': 'MGO(WT%)',
    'CaO': 'CAO(WT%)', 'Na2O': 'NA2O(WT%)', 'K2O': 'K2O(WT%)',
    'TiO2': 'TIO2(WT%)', 'P2O5': 'P2O5(WT%)', 'MnO': 'MNO(WT%)',
}

# Trace-element subsets.  'all' uses 26 elements (>40% coverage each);
# 'core8' uses 8 high-coverage petrogenetically informative elements
# (Sr, Rb, Ba, Y, Zr, Nb, V, Cr) — each >50% covered and selected to
# avoid the curse of dimensionality that strikes at ~35-D.
TRACE_SETS = {
    'all':   ['Sr', 'Rb', 'Ba', 'Y', 'Zr', 'Nb', 'La', 'Ce', 'Nd',
              'Sm', 'Eu', 'V', 'Cr', 'Ni', 'Co', 'Sc', 'Th', 'U',
              'Pb', 'Hf', 'Ta', 'Zn', 'Ga', 'Cs', 'Dy', 'Gd'],
    'core8': ['Sr', 'Rb', 'Ba', 'Y', 'Zr', 'Nb', 'V', 'Cr'],
    'lree':  ['La', 'Ce', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy'],  # LREE+HREE subset
    'hfse':  ['Zr', 'Nb', 'Hf', 'Ta', 'Y', 'Th', 'U'],    # HFSE+actinides
    'lile':  ['Sr', 'Rb', 'Ba', 'Cs', 'Pb'],              # LILE subset
    'trans': ['V', 'Cr', 'Ni', 'Co', 'Sc', 'Zn', 'Ga'],   # transition metals
}
TRACE_COL_MAP = {e: e.upper() + '(PPM)' for e in
                 set(sum(TRACE_SETS.values(), []))}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_georoc_geochem(db_path, rock_type, use_trace=True,
                        trace_set='all', min_samples_per_label=50,
                        min_trace_count=None):
    """Load VOL/PLU samples with major oxides (+ optional trace elements).

    Rows are kept if they have all major oxides (after FeO* unification)
    and, when trace elements are requested, at least ``min_trace_count``
    of the requested trace subset non-missing (default = 80% of subset
    size).  Missing trace values are floored to a small value before the
    ilr transform (see build_design_geochem) so partial-trace analyses
    are still usable; this trades a small bias for a much larger sample
    size, which the diagonal-bandwidth KDE needs in higher dimensions.
    """
    trace_elements = TRACE_SETS[trace_set]
    if min_trace_count is None:
        min_trace_count = max(1, int(0.8 * len(trace_elements)))
    oxide_cols = [OXIDE_COL_MAP[o] for o in MAJOR_OXIDES
                  if o != 'FeOT']
    iron_cols = ['FEOT(WT%)', 'FEO(WT%)', 'FE2O3(WT%)', 'FE2O3T(WT%)',
                 'FE(PPM)']
    trace_cols = [TRACE_COL_MAP[e] for e in trace_elements] if use_trace else []
    select = ['"ROCK NAME"', '"ROCK TYPE"', '"CITATIONS"'] + \
        ['"' + c + '"' for c in oxide_cols + iron_cols + trace_cols]

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            'SELECT ' + ','.join(select) + ' FROM georoc_data '
            'WHERE "ROCK TYPE"=?', conn, params=(rock_type,))
    finally:
        conn.close()

    # rename to short names
    rename = {'ROCK NAME': 'RockNameRaw', 'ROCK TYPE': 'RockType',
              'CITATIONS': 'CitationsRaw'}
    for short, raw in OXIDE_COL_MAP.items():
        rename[raw] = short
    df = df.rename(columns=rename)

    # numeric
    for c in MAJOR_OXIDES:
        if c != 'FeOT' and c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace('', np.nan)

    # unify iron
    df['FeOT'] = unify_feo_star(df)
    df['Alk'] = df['Na2O'] + df['K2O']

    # require all major oxides present and positive
    major_present = [o for o in MAJOR_OXIDES if o != 'FeOT'] + ['FeOT']
    df = df.dropna(subset=major_present)
    for o in major_present:
        df = df[df[o] >= 0]
    df = df[(df['SiO2'] > 0)]

    # trace elements: numeric + coverage filter
    if use_trace:
        for e in trace_elements:
            col = TRACE_COL_MAP[e]
            if col in df:
                df[e] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[e] = np.nan
        present = df[trace_elements].notna() & (df[trace_elements] > 0)
        n_trace = present.sum(axis=1)
        df = df[n_trace >= min_trace_count].copy()

    # label + citation
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
# Design matrix: two ilr blocks
# --------------------------------------------------------------------------- #
def build_design_geochem(df, use_trace=True, trace_variant='ilr',
                         trace_set='all', trace_floor=1e-3):
    """Build the joint (major ilr, trace ilr/raw) design matrix.

    Major oxides are always ilr-transformed on the wt% simplex (with a
    'rest' part closing to 100 wt%).  Trace elements are, depending on
    ``trace_variant``:
      - 'ilr': ilr-transformed on the ppm simplex (with a 'trace-rest'
        part closing to 1e6 ppm), so the constant-sum effect on trace
        elements is removed (this is the closure-aware default);
      - 'log': log10(ppm) raw, no closure treatment (ablation);
      - 'raw': raw ppm values (ablation).
    Missing trace values are floored to ``trace_floor`` ppm before any
    transform so log stays finite.
    """
    trace_elements = TRACE_SETS[trace_set]
    major_parts = [df[o].values.astype(float) for o in MAJOR_OXIDES]
    major_rest = 100.0 - sum(major_parts)
    major_comp = close_composition(major_parts + [major_rest])
    Z_major = ilr_transform(major_comp)  # (n, D_M-1)

    if not use_trace:
        return Z_major

    T = df[trace_elements].values.astype(float)
    T = np.where(np.isfinite(T) & (T > 0), T, trace_floor)
    if trace_variant == 'ilr':
        # close the trace budget to 1e6 ppm; trace-rest = 1e6 - sum(T)
        trace_rest = 1e6 - T.sum(axis=1)
        trace_comp = close_composition(list(T.T) + [trace_rest])
        Z_trace = ilr_transform(trace_comp)  # (n, D_T-1)
    elif trace_variant == 'log':
        Z_trace = np.log10(T)
    elif trace_variant == 'raw':
        Z_trace = T
    else:
        raise ValueError('unknown trace_variant %r' % trace_variant)
    return np.hstack([Z_major, Z_trace])


# --------------------------------------------------------------------------- #
# Per-dimension adaptive bandwidth
# --------------------------------------------------------------------------- #
def adaptive_bandwidth_per_dim(data):
    """Return a per-dimension bandwidth vector for a diagonal-kernel KDE.

    For each coordinate k we compute the Stage-2 adaptive rule
    h_k = sqrt(h_S(z_k) * h_M(z_k)) on that coordinate alone.  This is a
    standard robust compromise (Wand & Jones 1993, §4.5) and avoids the
    exponential sample-size requirement of an isotropic kernel in high
    dimensions.
    """
    n, d = data.shape
    h = np.empty(d)
    for k in range(d):
        col = data[:, k][:, None]
        h_s = tasgs.silverman_bandwidth(col)
        try:
            h_m = tasgs.median_bandwidth(col)
        except ValueError:
            h_m = h_s
        if not np.isfinite(h_m) or h_m <= 0:
            h_k = max(h_s, 1e-3)
        else:
            h_k = math.sqrt(h_s * h_m)
        h[k] = h_k
    return h


def fit_kde_diag(data, bandwidth_vec):
    """Fit a Gaussian KDE with a diagonal bandwidth matrix.

    We scale each coordinate by 1/h_k, fit a unit-bandwidth isotropic
    KernelDensity on the scaled data, and undo the scaling at scoring
    time.  This is the standard trick to get a diagonal bandwidth out of
    scikit-learn's KernelDensity (which only takes a scalar bandwidth).
    """
    scaled = data / bandwidth_vec[None, :]
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(scaled)
    return kde, bandwidth_vec


def score_kde_diag(kde, bandwidth_vec, X):
    """Score samples under the diagonal-bandwidth KDE (undo the scaling).

    Because the KDE was fit on X/h, evaluating at a point x requires
    querying at x/h.  The returned log-density must be shifted by
    -sum(log h_k) to account for the Jacobian of the scaling.
    """
    scaled = X / bandwidth_vec[None, :]
    log_d = kde.score_samples(scaled)
    log_d = log_d - np.sum(np.log(bandwidth_vec))
    return log_d


# --------------------------------------------------------------------------- #
# Label-confidence filter (training-only, leakage-free)
# --------------------------------------------------------------------------- #
def filter_mislabelled(train, use_trace, trace_variant,
                       trace_set='all', chi2_thresh=None, verbose=True):
    """Drop systematically mis-labelled training analyses.

    For each (Label, CitationID) pair we compute the median Mahalanobis
    distance of that citation's samples from the label's robust centroid
    (computed without that citation, via a leave-one-citation-out
    covariance).  Pairs whose median distance exceeds the 95th percentile
    of the chi distribution with D degrees of freedom are dropped from
    training only — the test set is never touched, so the evaluation
    remains leakage-free and the classification target is still the
    original GEOROC label.

    This addresses the reviewer/user concern that the original TAS labels
    in GEOROC may themselves be wrong: we remove obvious label noise
    before fitting the KDE without changing the test labels.
    """
    if chi2_thresh is None:
        # 95th percentile of chi^2_D — Rousseeuw & Van Zomeren (1990)
        from scipy.stats import chi2
        D = build_design_geochem(train, use_trace, trace_variant,
                                  trace_set=trace_set).shape[1]
        chi2_thresh = chi2.ppf(0.95, D)

    keep_idx = []
    cleanup = []
    for label, grp in train.groupby('Label'):
        Z = build_design_geochem(grp, use_trace, trace_variant,
                                  trace_set=trace_set)
        for cit, sub in grp.groupby('CitationID'):
            mask = grp.index.isin(sub.index)
            # leave-one-citation-out robust centroid + covariance
            other_Z = Z[~mask]
            if len(other_Z) < 20:
                keep_idx.extend(sub.index.tolist())
                continue
            med = np.median(other_Z, axis=0)
            # robust covariance: shrinkage estimator for stability
            cov = np.cov(other_Z, rowvar=False)
            if Z.shape[1] > 1:
                # Ledoit-Wolf-style shrink toward diagonal
                mean_var = np.mean(np.diag(cov))
                cov = 0.9 * cov + 0.1 * mean_var * np.eye(cov.shape[0])
            try:
                inv = np.linalg.pinv(cov)
            except np.linalg.LinAlgError:
                keep_idx.extend(sub.index.tolist())
                continue
            sub_Z = Z[mask]
            d2 = np.sum(((sub_Z - med) @ inv) * (sub_Z - med), axis=1)
            med_d2 = np.median(d2)
            keep = med_d2 <= chi2_thresh
            if keep:
                keep_idx.extend(sub.index.tolist())
            else:
                cleanup.append({'Label': label, 'CitationID': cit,
                                'n_dropped': len(sub),
                                'median_mahalanobis': float(med_d2)})
    out = train.loc[train.index.isin(keep_idx)].copy()
    if verbose and cleanup:
        print('[cleanup] dropped %d citation-label pairs (%d samples)'
              % (len(cleanup), sum(c['n_dropped'] for c in cleanup)))
    return out, pd.DataFrame(cleanup)


# --------------------------------------------------------------------------- #
# Fit + score per label
# --------------------------------------------------------------------------- #
def fit_kde_set_geochem(train, use_trace, trace_variant, out_dir,
                        trace_set='all', min_samples=30, seed=7):
    os.makedirs(os.path.join(out_dir, 'kde_train'), exist_ok=True)
    rng = np.random.default_rng(seed)
    D = build_design_geochem(train, use_trace, trace_variant,
                              trace_set=trace_set).shape[1]
    # dimension-aware subsampling cap
    cap = max(4000, int(50 * D * math.log(D)))
    kdes = {}
    bws = {}
    for label, grp in train.groupby('Label'):
        if len(grp) < min_samples:
            continue
        if len(grp) > cap:
            grp = grp.sample(cap, random_state=int(rng.integers(1 << 31)))
        Z = build_design_geochem(grp, use_trace, trace_variant,
                                  trace_set=trace_set)
        bw = adaptive_bandwidth_per_dim(Z)
        kde, bw = fit_kde_diag(Z, bw)
        kdes[label] = kde
        bws[label] = bw
        with open(os.path.join(out_dir, 'kde_train', label + '_kde.pkl'),
                  'wb') as f:
            pickle.dump((kde, bw), f)
    return kdes, bws


def score_samples_geochem(kdes, bws, df, use_trace, trace_variant,
                          trace_set='all'):
    X = build_design_geochem(df, use_trace, trace_variant, trace_set=trace_set)
    labels = list(kdes.keys())
    scores = np.empty((len(df), len(labels)))
    for j, label in enumerate(labels):
        scores[:, j] = score_kde_diag(kdes[label], bws[label], X)
    return scores, labels


# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #
def evaluate(test, kdes, bws, use_trace, trace_variant, polys,
              trace_set='all'):
    sio2 = test['SiO2'].values
    alk = test['Alk'].values
    y = test['Label'].values
    log_scores, labels = score_samples_geochem(
        kdes, bws, test, use_trace, trace_variant, trace_set=trace_set)
    sm = softmax_over_rows(log_scores)
    mr = max_ratio(log_scores)
    sm_pred = np.array(labels)[np.argmax(sm, axis=1)]
    mr_pred = np.array(labels)[np.argmax(mr, axis=1)]
    tas_pred = classify_tas(sio2, alk, polys)
    return pd.DataFrame({
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--rock-type', default='VOL', choices=['VOL', 'PLU'])
    ap.add_argument('--no-trace', dest='use_trace', action='store_false',
                    help='Use major oxides only (no trace elements)')
    ap.set_defaults(use_trace=True)
    ap.add_argument('--trace-variant', default='ilr',
                    choices=['ilr', 'log', 'raw'],
                    help='How trace elements are represented in the KDE')
    ap.add_argument('--trace-set', default='all',
                    choices=list(TRACE_SETS.keys()),
                    help='Which trace subset to use: all (26), core8, '
                         'lree, hfse, lile, trans')
    ap.add_argument('--no-cleanup', dest='cleanup', action='store_false',
                    help='Disable the training-only mislabel filter')
    ap.set_defaults(cleanup=True)
    ap.add_argument('--n-test-citations', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out-root',
                    default=os.path.join(HERE, 'TAS_GS_v2'))
    args = ap.parse_args()

    tag = '%s_geochem_%s_%s%s' % (
        args.rock_type,
        'major' if not args.use_trace else args.trace_set,
        args.trace_variant,
        '' if args.use_trace else '_notrace')
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print('[load] %s / trace=%s (%s / %s)' %
          (args.rock_type, args.use_trace, args.trace_set, args.trace_variant))
    df = load_georoc_geochem(args.db, args.rock_type,
                             use_trace=args.use_trace,
                             trace_set=args.trace_set)
    print('[load] %d rows, %d labels, %d citations'
          % (len(df), df['Label'].nunique(), df['CitationID'].nunique()))

    train, test = split_by_citation(df, args.n_test_citations, args.seed)
    print('[split] train=%d (%d cits), test=%d (%d cits)'
          % (len(train), train['CitationID'].nunique(),
             len(test), test['CitationID'].nunique()))

    if args.cleanup:
        t_c = time.time()
        train, cleanup_df = filter_mislabelled(
            train, args.use_trace, args.trace_variant,
            trace_set=args.trace_set)
        print('[cleanup] %.1fs, train now %d' % (time.time() - t_c,
                                                 len(train)))
        if len(cleanup_df):
            cleanup_df.to_csv(os.path.join(out_dir, 'per_label_cleanup.csv'),
                              index=False)

    t1 = time.time()
    kdes, bws = fit_kde_set_geochem(train, args.use_trace, args.trace_variant,
                                    out_dir, trace_set=args.trace_set,
                                    seed=args.seed)
    print('[fit] %d KDEs in %.1fs' % (len(kdes), time.time() - t1))

    t2 = time.time()
    polys, _ = load_tas_polygons()
    res = evaluate(test, kdes, bws, args.use_trace, args.trace_variant,
                   polys, trace_set=args.trace_set)
    print('[eval] %.1fs' % (time.time() - t2))

    res.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    test[['SiO2', 'Alk', 'Label', 'CitationID']].to_csv(
        os.path.join(out_dir, 'test_set.csv'), index=False)

    report = accuracy_report(res)
    report['runtime_sec'] = {
        'total': time.time() - t0,
        'fit': time.time() - t1,
        'eval': time.time() - t2,
        'n_train': len(train), 'n_test': len(test),
        'n_test_citations': int(test['CitationID'].nunique()),
        'use_trace': bool(args.use_trace),
        'trace_variant': args.trace_variant,
        'trace_set': args.trace_set,
        'cleanup': bool(args.cleanup),
        'rock_type': args.rock_type, 'seed': args.seed,
    }
    with open(os.path.join(out_dir, 'accuracy.json'), 'w') as f:
        json.dump(report, f, indent=2)

    plot_accuracy_bar(report, out_dir,
                      title_suffix='GeoChem-GS %s/%s%s'
                      % (args.rock_type,
                         args.trace_variant if args.use_trace else 'major',
                         '' if args.use_trace else ' (no trace)'))
    confusion(res, 'SoftMax_Pred').to_csv(
        os.path.join(out_dir, 'confusion_matrix.csv'))

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