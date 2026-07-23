"""
23_Classifiers_Compare.py
=========================
Systematic comparison of chemistry-only classifiers on the GEOROC
volcanic subset, on the same publication-disjoint train/test split.

Methods compared:
  - TAS (classical polygon baseline)
  - KDE+SoftMax (Stage 2/3, ilr on the chosen feature set)
  - KDE+SoftMax (refined: per-label bandwidth selected by leave-one-
                citation-out log-likelihood CV)
  - LDA (Linear Discriminant Analysis on the ilr design matrix)
  - QDA (Quadratic Discriminant Analysis on the ilr design matrix)
  - PCA(n)+KDE+SoftMax (reduce dimension then KDE)
  - PCA(n)+LDA
  - Trace-only KDE+SoftMax (ilr on the trace budget alone)
  - Trace-only LDA

All methods share:
  - the same by-citation train/test split (no leakage)
  - the same ilr-transformed design matrix (closure-aware)
  - the same label set (labels present in both splits, >=50 samples)
  - the same test citations

The script writes:
  master_accuracy_table.csv    one row per method, with accuracy and
                               bootstrap 95% CI on all labels and on
                               the TAS-comparable subset
  per_method_predictions/      per-method CSV with predictions
  figure_method_comparison.{svg,pdf}  grouped bar chart
  accuracy.json                all numbers

Usage:
  python 23_Classifiers_Compare.py --rock-type VOL \
      --feature-set core8 --n-test-citations 1500 --seed 7
  python 23_Classifiers_Compare.py --rock-type VOL \
      --feature-set core8 --n-test-citations 500 --seed 42
"""

import argparse
import json
import math
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'geo', os.path.join(HERE, '21_GeoChem_GS.py'))
geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geo)
import sys
sys.modules['geo'] = geo

from tasgs import (  # via geo -> tasgs already loaded
    split_by_citation, classify_tas, load_tas_polygons,
    softmax_over_rows, accuracy_report, DEFAULT_DB,
)
from geo import (  # noqa: E402
    TRACE_SETS, MAJOR_OXIDES, build_design_geochem,
    adaptive_bandwidth_per_dim, fit_kde_diag, score_kde_diag,
)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'truetype'


# --------------------------------------------------------------------------- #
# Refined probability field: per-label CV-selected bandwidth
# --------------------------------------------------------------------------- #
# Instead of a single geometric-mean bandwidth per label, we try a small
# grid of bandwidth scales {0.5, 1, 2} x {Silverman, median, geometric}
# and pick, per label, the bandwidth that maximises the
# leave-one-citation-out average log-likelihood on the training set.
# This is the "refined probability field" of the manuscript.

_BW_GRID = [0.5, 1.0, 2.0]  # multiplicative scales on the adaptive bw


def _adaptive_bw_scaled(Z, scale):
    """Adaptive per-dimension bandwidth, scaled by ``scale``."""
    bw = adaptive_bandwidth_per_dim(Z)
    return bw * scale


def fit_kde_cv(train, use_trace, trace_variant, trace_set,
               out_dir=None, n_cv_citations=50, seed=7):
    """Per-label KDE with bandwidth chosen by leave-one-citation-out CV.

    For each label we hold out ``n_cv_citations`` of its training
    citations, fit the KDE on the rest with each bandwidth scale, and
    keep the scale that maximises the average log-likelihood on the
    held-out citations.  We then refit on all training samples with the
    winning scale.
    """
    rng = np.random.default_rng(seed)
    kdes, bws, scales = {}, {}, {}
    for label, grp in train.groupby('Label'):
        if len(grp) < 30:
            continue
        cits = np.array(sorted(grp['CitationID'].unique()))
        if len(cits) < 5:
            # too few citations for CV; use default adaptive bw
            Z = build_design_geochem(grp, use_trace, trace_variant,
                                     trace_set=trace_set)
            bw = adaptive_bandwidth_per_dim(Z)
            kde, bw = fit_kde_diag(Z, bw)
            kdes[label] = kde
            bws[label] = bw
            scales[label] = 1.0
            continue
        rng.shuffle(cits)
        n_hold = min(n_cv_citations, max(1, len(cits) // 5))
        hold_cits = set(cits[:n_hold])
        train_sub = grp[~grp['CitationID'].isin(hold_cits)]
        hold_sub = grp[grp['CitationID'].isin(hold_cits)]
        if len(train_sub) < 20 or len(hold_sub) < 5:
            Z = build_design_geochem(grp, use_trace, trace_variant,
                                     trace_set=trace_set)
            bw = adaptive_bandwidth_per_dim(Z)
            kde, bw = fit_kde_diag(Z, bw)
            kdes[label] = kde
            bws[label] = bw
            scales[label] = 1.0
            continue
        Z_tr = build_design_geochem(train_sub, use_trace, trace_variant,
                                    trace_set=trace_set)
        Z_hold = build_design_geochem(hold_sub, use_trace, trace_variant,
                                      trace_set=trace_set)
        best_ll = -np.inf
        best_scale = 1.0
        for scale in _BW_GRID:
            bw = _adaptive_bw_scaled(Z_tr, scale)
            kde, bw = fit_kde_diag(Z_tr, bw)
            ll = kde.score(Z_hold / bw[None, :]) - np.sum(np.log(bw)) * len(Z_hold)
            ll /= max(1, len(Z_hold))
            if ll > best_ll:
                best_ll = ll
                best_scale = scale
        # refit on all training samples with the winning scale
        Z_all = build_design_geochem(grp, use_trace, trace_variant,
                                     trace_set=trace_set)
        bw = _adaptive_bw_scaled(Z_all, best_scale)
        kde, bw = fit_kde_diag(Z_all, bw)
        kdes[label] = kde
        bws[label] = bw
        scales[label] = best_scale
    return kdes, bws, scales


# --------------------------------------------------------------------------- #
# Unified evaluation helpers
# --------------------------------------------------------------------------- #
def kde_predict(kdes, bws, X):
    """Top-1 prediction + SoftMax probability from per-label KDEs."""
    labels = list(kdes.keys())
    log_scores = np.empty((len(X), len(labels)))
    for j, lab in enumerate(labels):
        log_scores[:, j] = score_kde_diag(kdes[lab], bws[lab], X)
    sm = softmax_over_rows(log_scores)
    pred = np.array(labels)[np.argmax(sm, axis=1)]
    prob = sm.max(axis=1)
    return pred, prob


def accuracy_with_ci(y_true, y_pred, n_boot=200, seed=7):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.array([p is not None and str(p) != 'nan' for p in y_pred])
    yt = y_true[mask]
    yp = y_pred[mask].astype(object)
    acc = float((yt == yp).mean()) if len(yt) else 0.0
    rng = np.random.default_rng(seed)
    n = len(yt)
    if n == 0:
        return acc, 0.0, 0.0, 0
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = (yt[idx] == yp[idx]).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return acc, float(lo), float(hi), int(n)


def evaluate_method(y_true, y_pred, restricted_mask, name, n_boot=200, seed=7):
    """Return a dict row for the master table."""
    acc, lo, hi, n = accuracy_with_ci(y_true, y_pred, n_boot, seed)
    r_yt = np.asarray(y_true)[restricted_mask]
    r_yp = np.asarray(y_pred)[restricted_mask]
    r_acc, r_lo, r_hi, r_n = accuracy_with_ci(r_yt, r_yp, n_boot, seed)
    return {
        'Method': name,
        'Accuracy_all': acc, 'CI_all': '[%.3f, %.3f]' % (lo, hi),
        'n_all': n,
        'Accuracy_TAScomp': r_acc,
        'CI_TAScomp': '[%.3f, %.3f]' % (r_lo, r_hi),
        'n_TAScomp': r_n,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--rock-type', default='VOL', choices=['VOL', 'PLU'])
    ap.add_argument('--no-trace', dest='use_trace', action='store_false')
    ap.set_defaults(use_trace=True)
    ap.add_argument('--trace-variant', default='ilr',
                    choices=['ilr', 'log', 'raw'])
    ap.add_argument('--trace-set', default='core8',
                    choices=list(TRACE_SETS.keys()))
    ap.add_argument('--no-cleanup', dest='cleanup', action='store_false')
    ap.set_defaults(cleanup=True)
    ap.add_argument('--no-cv-bw', dest='cv_bw', action='store_false',
                    help='Skip the refined CV-bandwidth KDE (slow)')
    ap.set_defaults(cv_bw=True)
    ap.add_argument('--n-test-citations', type=int, default=1500)
    ap.add_argument('--pca-n', type=int, default=8,
                    help='Number of PCA components for the PCA+KDE / '
                         'PCA+LDA methods')
    ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out-root',
                    default=os.path.join(HERE, 'TAS_GS_v2'))
    args = ap.parse_args()

    tag = '%s_compare_%s_%s%s' % (
        args.rock_type,
        'major' if not args.use_trace else args.trace_set,
        args.trace_variant,
        '' if args.use_trace else '_notrace')
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = os.path.join(out_dir, 'per_method_predictions')
    os.makedirs(pred_dir, exist_ok=True)

    t0 = time.time()
    print('[load] %s / trace=%s (%s / %s)' %
          (args.rock_type, args.use_trace, args.trace_set, args.trace_variant))
    df = geo.load_georoc_geochem(args.db, args.rock_type,
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
        train, cleanup_df = geo.filter_mislabelled(
            train, args.use_trace, args.trace_variant,
            trace_set=args.trace_set)
        print('[cleanup] %.1fs, train now %d' % (time.time() - t_c, len(train)))
        if len(cleanup_df):
            cleanup_df.to_csv(os.path.join(out_dir, 'per_label_cleanup.csv'),
                              index=False)

    # ---- design matrices ----
    Z_train = build_design_geochem(train, args.use_trace, args.trace_variant,
                                   trace_set=args.trace_set)
    Z_test = build_design_geochem(test, args.use_trace, args.trace_variant,
                                  trace_set=args.trace_set)
    y_train = train['Label'].values
    y_test = test['Label'].values
    D = Z_train.shape[1]
    print('[design] D=%d, train=%d, test=%d' % (D, len(Z_train), len(Z_test)))

    # TAS-comparable mask for the restricted accuracy
    from tasgs import TAS_COMPARABLE
    restricted_mask = np.array(
        [lab in TAS_COMPARABLE for lab in y_test])

    # label encoders for LDA/QDA
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y_train)
    y_train_enc = le.transform(y_train)

    polys, _ = load_tas_polygons()
    rows = []

    # ---- Method 0: classical TAS ----
    tas_pred = classify_tas(test['SiO2'].values, test['Alk'].values, polys)
    rows.append(evaluate_method(y_test, tas_pred, restricted_mask,
                                'Classical TAS', args.n_boot, args.seed))
    pd.DataFrame({'Label': y_test, 'TAS_Pred': tas_pred,
                  'SiO2': test['SiO2'].values,
                  'Alk': test['Alk'].values,
                  'CitationID': test['CitationID'].values}
                 ).to_csv(os.path.join(pred_dir, 'TAS.csv'), index=False)
    print('[done] Classical TAS')

    # ---- Method 1: KDE+SoftMax (adaptive bw) ----
    t1 = time.time()
    kdes, bws = {}, {}
    cap = max(4000, int(50 * D * math.log(D)))
    rng = np.random.default_rng(args.seed)
    for label, grp in train.groupby('Label'):
        if len(grp) < 30:
            continue
        g = grp.sample(min(len(grp), cap),
                       random_state=int(rng.integers(1 << 31)))
        Z = build_design_geochem(g, args.use_trace, args.trace_variant,
                                 trace_set=args.trace_set)
        bw = adaptive_bandwidth_per_dim(Z)
        kde, bw = fit_kde_diag(Z, bw)
        kdes[label] = kde
        bws[label] = bw
    pred, prob = kde_predict(kdes, bws, Z_test)
    rows.append(evaluate_method(y_test, pred, restricted_mask,
                                'KDE+SoftMax (adaptive bw)',
                                args.n_boot, args.seed))
    pd.DataFrame({'Label': y_test, 'Pred': pred, 'Prob': prob,
                  'CitationID': test['CitationID'].values}
                 ).to_csv(os.path.join(pred_dir, 'KDE_SoftMax.csv'),
                          index=False)
    print('[done] KDE+SoftMax (%.1fs)' % (time.time() - t1))

    # ---- Method 2: KDE+SoftMax (refined CV bw) ----
    if args.cv_bw:
        t1 = time.time()
        kdes_cv, bws_cv, scales = fit_kde_cv(
            train, args.use_trace, args.trace_variant, args.trace_set,
            seed=args.seed)
        pred_cv, prob_cv = kde_predict(kdes_cv, bws_cv, Z_test)
        rows.append(evaluate_method(y_test, pred_cv, restricted_mask,
                                    'KDE+SoftMax (refined CV bw)',
                                    args.n_boot, args.seed))
        pd.DataFrame({'Label': y_test, 'Pred': pred_cv, 'Prob': prob_cv,
                      'CitationID': test['CitationID'].values}
                     ).to_csv(os.path.join(pred_dir, 'KDE_SoftMax_CV.csv'),
                              index=False)
        print('[done] KDE+SoftMax refined CV bw (%.1fs, scales=%s)'
              % (time.time() - t1,
                 {k: v for k, v in scales.items() if v != 1.0}))

    # ---- Method 3: LDA ----
    t1 = time.time()
    lda = LinearDiscriminantAnalysis()
    lda.fit(Z_train, y_train_enc)
    lda_pred_enc = lda.predict(Z_test)
    lda_pred = le.inverse_transform(lda_pred_enc)
    rows.append(evaluate_method(y_test, lda_pred, restricted_mask,
                                'LDA (on ilr)',
                                args.n_boot, args.seed))
    pd.DataFrame({'Label': y_test, 'Pred': lda_pred,
                  'CitationID': test['CitationID'].values}
                 ).to_csv(os.path.join(pred_dir, 'LDA.csv'), index=False)
    print('[done] LDA (%.1fs)' % (time.time() - t1))

    # ---- Method 4: QDA ----
    t1 = time.time()
    try:
        qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
        qda.fit(Z_train, y_train_enc)
        qda_pred_enc = qda.predict(Z_test)
        qda_pred = le.inverse_transform(qda_pred_enc)
        rows.append(evaluate_method(y_test, qda_pred, restricted_mask,
                                    'QDA (on ilr)',
                                    args.n_boot, args.seed))
        pd.DataFrame({'Label': y_test, 'Pred': qda_pred,
                      'CitationID': test['CitationID'].values}
                     ).to_csv(os.path.join(pred_dir, 'QDA.csv'),
                              index=False)
        print('[done] QDA (%.1fs)' % (time.time() - t1))
    except Exception as e:
        print('[skip] QDA: %s' % e)

    # ---- Method 5: PCA + KDE + SoftMax ----
    t1 = time.time()
    n_pca = min(args.pca_n, D - 1)
    pca = PCA(n_components=n_pca, random_state=args.seed)
    Zp_train = pca.fit_transform(Z_train)
    Zp_test = pca.transform(Z_test)
    kdes_p, bws_p = {}, {}
    cap_p = max(4000, int(50 * n_pca * math.log(n_pca)))
    for label, grp in train.groupby('Label'):
        if len(grp) < 30:
            continue
        idx = grp.index
        Zp = Zp_train[train.index.get_indexer(idx)]
        Zp = Zp[train['Label'].iloc[
            train.index.get_indexer(idx)].values == label]
        g_idx = rng.choice(len(Zp), min(len(Zp), cap_p),
                           replace=len(Zp) < cap_p)
        Zp_s = Zp[g_idx]
        bw = adaptive_bandwidth_per_dim(Zp_s)
        kde, bw = fit_kde_diag(Zp_s, bw)
        kdes_p[label] = kde
        bws_p[label] = bw
    pred_p, prob_p = kde_predict(kdes_p, bws_p, Zp_test)
    rows.append(evaluate_method(y_test, pred_p, restricted_mask,
                                'PCA(%d)+KDE+SoftMax' % n_pca,
                                args.n_boot, args.seed))
    pd.DataFrame({'Label': y_test, 'Pred': pred_p, 'Prob': prob_p,
                  'CitationID': test['CitationID'].values}
                 ).to_csv(os.path.join(pred_dir, 'PCA_KDE.csv'),
                          index=False)
    print('[done] PCA(%d)+KDE (%.1fs)' % (n_pca, time.time() - t1))

    # ---- Method 6: PCA + LDA ----
    t1 = time.time()
    lda_p = LinearDiscriminantAnalysis()
    lda_p.fit(Zp_train, y_train_enc)
    pred_p_lda = le.inverse_transform(lda_p.predict(Zp_test))
    rows.append(evaluate_method(y_test, pred_p_lda, restricted_mask,
                                'PCA(%d)+LDA' % n_pca,
                                args.n_boot, args.seed))
    pd.DataFrame({'Label': y_test, 'Pred': pred_p_lda,
                  'CitationID': test['CitationID'].values}
                 ).to_csv(os.path.join(pred_dir, 'PCA_LDA.csv'),
                          index=False)
    print('[done] PCA(%d)+LDA (%.1fs)' % (n_pca, time.time() - t1))

    # ---- Method 7: Trace-only KDE+SoftMax (ilr on trace budget alone) ----
    if args.use_trace:
        t1 = time.time()
        # build a trace-only design: ilr on the trace budget only
        trace_elements = TRACE_SETS[args.trace_set]
        T_train = train[trace_elements].values.astype(float)
        T_train = np.where(np.isfinite(T_train) & (T_train > 0),
                           T_train, 1e-3)
        from tasgs import close_composition, ilr_transform
        tr_rest = 1e6 - T_train.sum(axis=1)
        tr_comp = close_composition(list(T_train.T) + [tr_rest])
        Zt_train = ilr_transform(tr_comp)
        T_test = test[trace_elements].values.astype(float)
        T_test = np.where(np.isfinite(T_test) & (T_test > 0), T_test, 1e-3)
        tr_rest_t = 1e6 - T_test.sum(axis=1)
        tr_comp_t = close_composition(list(T_test.T) + [tr_rest_t])
        Zt_test = ilr_transform(tr_comp_t)
        Dt = Zt_train.shape[1]
        kdes_t, bws_t = {}, {}
        cap_t = max(4000, int(50 * Dt * math.log(Dt)))
        for label, grp in train.groupby('Label'):
            if len(grp) < 30:
                continue
            idx = grp.index
            Zt = Zt_train[train.index.get_indexer(idx)]
            Zt = Zt[train['Label'].iloc[
                train.index.get_indexer(idx)].values == label]
            g_idx = rng.choice(len(Zt), min(len(Zt), cap_t),
                               replace=len(Zt) < cap_t)
            bw = adaptive_bandwidth_per_dim(Zt[g_idx])
            kde, bw = fit_kde_diag(Zt[g_idx], bw)
            kdes_t[label] = kde
            bws_t[label] = bw
        pred_t, prob_t = kde_predict(kdes_t, bws_t, Zt_test)
        rows.append(evaluate_method(y_test, pred_t, restricted_mask,
                                    'Trace-only KDE+SoftMax (ilr)',
                                    args.n_boot, args.seed))
        pd.DataFrame({'Label': y_test, 'Pred': pred_t, 'Prob': prob_t,
                      'CitationID': test['CitationID'].values}
                     ).to_csv(os.path.join(pred_dir, 'Trace_KDE.csv'),
                              index=False)
        print('[done] Trace-only KDE (D_t=%d) (%.1fs)'
              % (Dt, time.time() - t1))

        # ---- Method 8: Trace-only LDA ----
        t1 = time.time()
        lda_t = LinearDiscriminantAnalysis()
        lda_t.fit(Zt_train, y_train_enc)
        pred_t_lda = le.inverse_transform(lda_t.predict(Zt_test))
        rows.append(evaluate_method(y_test, pred_t_lda, restricted_mask,
                                    'Trace-only LDA (on ilr)',
                                    args.n_boot, args.seed))
        pd.DataFrame({'Label': y_test, 'Pred': pred_t_lda,
                      'CitationID': test['CitationID'].values}
                     ).to_csv(os.path.join(pred_dir, 'Trace_LDA.csv'),
                              index=False)
        print('[done] Trace-only LDA (%.1fs)' % (time.time() - t1))

    # ---- Method 9: Shrinkage LDA ----
    t1 = time.time()
    lda_shrink = LinearDiscriminantAnalysis(
        solver='lsqr', shrinkage='auto')
    lda_shrink.fit(Z_train, y_train_enc)
    pred_shrink = le.inverse_transform(lda_shrink.predict(Z_test))
    rows.append(evaluate_method(y_test, pred_shrink, restricted_mask,
                                'LDA (shrinkage, on ilr)',
                                args.n_boot, args.seed))
    pd.DataFrame({'Label': y_test, 'Pred': pred_shrink,
                  'CitationID': test['CitationID'].values}
                 ).to_csv(os.path.join(pred_dir, 'LDA_shrinkage.csv'),
                          index=False)
    print('[done] Shrinkage LDA (%.1fs)' % (time.time() - t1))

    # ---- Method 10: Random Forest ----
    t1 = time.time()
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20,
        random_state=args.seed, n_jobs=-1)
    rf.fit(Z_train, y_train_enc)
    pred_rf = le.inverse_transform(rf.predict(Z_test))
    prob_rf = rf.predict_proba(Z_test).max(axis=1)
    rows.append(evaluate_method(y_test, pred_rf, restricted_mask,
                                'Random Forest (on ilr)',
                                args.n_boot, args.seed))
    pd.DataFrame({'Label': y_test, 'Pred': pred_rf, 'Prob': prob_rf,
                  'CitationID': test['CitationID'].values}
                 ).to_csv(os.path.join(pred_dir, 'RandomForest.csv'),
                          index=False)
    print('[done] Random Forest (%.1fs)' % (time.time() - t1))

    # ---- Method 11: RF on major-only (if trace used) for comparison ----
    if args.use_trace:
        t1 = time.time()
        Z_tr_major = build_design_geochem(
            train, False, 'ilr', trace_set='all')
        Z_te_major = build_design_geochem(
            test, False, 'ilr', trace_set='all')
        rf_major = RandomForestClassifier(
            n_estimators=300, max_depth=20,
            random_state=args.seed, n_jobs=-1)
        rf_major.fit(Z_tr_major, y_train_enc)
        pred_rf_major = le.inverse_transform(
            rf_major.predict(Z_te_major))
        rows.append(evaluate_method(
            y_test, pred_rf_major, restricted_mask,
            'Random Forest (major-only, on ilr)',
            args.n_boot, args.seed))
        pd.DataFrame({'Label': y_test, 'Pred': pred_rf_major,
                      'CitationID': test['CitationID'].values}
                     ).to_csv(os.path.join(pred_dir, 'RF_major.csv'),
                              index=False)
        print('[done] RF major-only D=%d (%.1fs)'
              % (Z_tr_major.shape[1], time.time() - t1))

    # ---- Save master table + figure ----
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, 'master_accuracy_table.csv'),
                 index=False)
    with open(os.path.join(out_dir, 'accuracy.json'), 'w') as f:
        json.dump({'rows': rows, 'runtime_sec': time.time() - t0,
                   'config': vars(args)}, f, indent=2, default=str)

    # grouped bar chart
    methods = table['Method'].tolist()
    all_acc = table['Accuracy_all'].tolist()
    tas_acc = table['Accuracy_TAScomp'].tolist()
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, all_acc, w, label='All labels', color='#9a99b7')
    ax.bar(x + w/2, tas_acc, w, label='TAS-comparable',
           color='#e48080')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Top-1 accuracy')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title('%s / %s — method comparison' % (args.rock_type,
                'major' if not args.use_trace else args.trace_set),
                fontsize=10)
    for i, v in enumerate(all_acc):
        ax.text(i - w/2, v + 0.01, '%.2f' % v, ha='center', fontsize=7)
    for i, v in enumerate(tas_acc):
        ax.text(i + w/2, v + 0.01, '%.2f' % v, ha='center', fontsize=7)
    fig.tight_layout()
    for ext in ('svg', 'pdf'):
        fig.savefig(os.path.join(out_dir,
                                 'figure_method_comparison.%s' % ext))
    plt.close(fig)

    print('\n=== Master accuracy table ===')
    print(table[['Method', 'Accuracy_all', 'CI_all',
                 'Accuracy_TAScomp', 'CI_TAScomp']].to_string(index=False))
    print('\nTotal runtime: %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()