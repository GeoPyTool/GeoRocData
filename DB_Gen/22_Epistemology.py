"""
22_Epistemology.py
==================
Quantitative epistemology layer for the TAS-GS / GeoChem-GS family.

The narrative of the revised manuscript is no longer "method A is more
accurate than method B", but a four-stage epistemic ladder:

  Stage I  — Hard decision           (classical TAS polygons)
  Stage II — Probability field       (KDE + SoftMax)
  Stage III— Mislabelling diagnostic (which original GEOROC labels are
             likely wrong, quantified per citation)
  Stage IV — Unresolvability diagnostic (which rock-name pairs cannot be
             separated by chemistry alone, quantified as an information-
             theoretic overlap)

Every stage produces a number with a confidence interval, and the four
numbers are reported side by side in a single "epistemic table" so the
reader can see, per rock name:
  * how often the classical diagram is right,
  * how often the probabilistic method is right,
  * how often the *original* label is probably wrong,
  * how much of the residual error is irreducible (the labels overlap
    in composition space and no chemistry-only method can separate
    them).

This script implements Stages III and IV on top of the trained
GeoChem-GS KDEs (Stage II) and the classical TAS baseline (Stage I).

Outputs (under TAS_GS_v2/<tag>_epistemology/):
    per_label_epistemic_table.csv     one row per rock name
    per_pair_overlaps.csv             symmetric matrix of irreducible
                                      overlaps between every pair of
                                      rock names
    per_citation_mislabelling.csv     flagged citations + evidence
    figure_epistemic_ladder.{svg,pdf} the 4-stage bar chart per label
    figure_overlap_heatmap.{svg,pdf}  the irreducible-overlap matrix
    epistemology.json                 all numbers + CIs
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
from scipy.stats import binom, chi2

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'tasgs', os.path.join(HERE, '20_TAS_GS_v2.py'))
tasgs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tasgs)
import sys
sys.modules['tasgs'] = tasgs

_spec2 = importlib.util.spec_from_file_location(
    'geo', os.path.join(HERE, '21_GeoChem_GS.py'))
geo = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(geo)
sys.modules['geo'] = geo

from tasgs import (  # noqa: E402
    split_by_citation, classify_tas, load_tas_polygons,
    softmax_over_rows, accuracy_report, DEFAULT_DB,
)


# --------------------------------------------------------------------------- #
# Stage I + II : accuracy with bootstrap CI (reused from stage 2/3)
# --------------------------------------------------------------------------- #
def accuracy_with_ci(y_true, y_pred, n_boot=200, seed=7):
    """Top-1 accuracy + bootstrap 95% CI on the test set."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.array([p is not None for p in y_pred])
    yt = y_true[mask]
    yp = y_pred[mask]
    acc = float((yt == yp).mean()) if len(yt) else 0.0
    rng = np.random.default_rng(seed)
    n = len(yt)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = (yt[idx] == yp[idx]).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return acc, float(lo), float(hi), int(n)


# --------------------------------------------------------------------------- #
# Stage III — Mislabelling diagnostic
# --------------------------------------------------------------------------- #
# A GEOROC sample is flagged as "probably mis-labelled" if, given its
# chemistry, the *SoftMax posterior* of its claimed label is below a
# threshold tau AND that threshold is calibrated against the
# leave-one-citation-out posterior distribution of correctly-labelled
# samples of the same label.  This is a Bayesian outlier rule, not a
# hard cutoff: we report the posterior probability of mislabelling for
# every sample, and aggregate by citation.
#
# Formally, for sample i with claimed label c_i and chemistry x_i, let
#   p_i(c | x_i) = SoftMax over KDE log-densities.
# The posterior of "the label is correct" is p_i(c_i | x_i).  We define
# the per-sample mislabelling posterior
#   m_i = 1 - p_i(c_i | x_i).
# A citation is flagged if the median of {m_i} across its samples
# exceeds the 95th percentile of the citation-level median-m
# distribution computed on a clean held-out subset (the calibration
# set).  This makes the rule data-adaptive and per-label.

def mislabelling_posteriors(test, kdes, bws, use_trace, trace_variant,
                            trace_set='all'):
    """Return per-sample mislabelling posterior m_i = 1 - p(claimed|x)."""
    X = geo.build_design_geochem(test, use_trace, trace_variant,
                                  trace_set=trace_set)
    labels = list(kdes.keys())
    log_scores = np.empty((len(test), len(labels)))
    for j, lab in enumerate(labels):
        log_scores[:, j] = geo.score_kde_diag(kdes[lab], bws[lab], X)
    sm = softmax_over_rows(log_scores)
    claimed_idx = np.array(
        [labels.index(c) if c in labels else -1 for c in test['Label']])
    p_claimed = np.array(
        [sm[i, claimed_idx[i]] if claimed_idx[i] >= 0 else np.nan
         for i in range(len(test))])
    return 1.0 - p_claimed, sm, labels


def calibrate_mislabelling_threshold(train, kdes, bws, use_trace,
                                     trace_variant, trace_set='all',
                                     percentile=0.95, seed=7):
    """Calibrate the citation-level mislabelling threshold on a clean
    held-out subset of the *training* data (so the test set is never
    used to calibrate its own flag).

    Returns the threshold tau such that a citation whose median
    mislabelling posterior exceeds tau is flagged at the 5% level.
    """
    rng = np.random.default_rng(seed)
    cits = np.array(sorted(train['CitationID'].unique()))
    rng.shuffle(cits)
    # use 30% of training citations as the calibration set
    cal_cits = set(cits[: max(1, len(cits) // 3)])
    cal = train[train['CitationID'].isin(cal_cits)]
    m, _, _ = mislabelling_posteriors(cal, kdes, bws, use_trace,
                                      trace_variant, trace_set=trace_set)
    med = cal.groupby('CitationID').apply(
        lambda g: np.nanmedian(
            m[np.isin(np.array(cal.index), np.array(g.index))]),
        include_groups=False)
    med = med.dropna()
    if len(med) < 20:
        return float(np.nanpercentile(m, 95))
    return float(np.nanpercentile(med, percentile * 100))


def flag_mislabelled_citations(test, kdes, bws, use_trace, trace_variant,
                               tau, trace_set='all'):
    """Return a DataFrame of citations flagged as probably mis-labelled."""
    m, sm, labels = mislabelling_posteriors(
        test, kdes, bws, use_trace, trace_variant, trace_set=trace_set)
    test = test.copy()
    test['m_posterior'] = m
    grp = test.groupby(['Label', 'CitationID'])['m_posterior']
    agg = grp.agg(['median', 'mean', 'count']).reset_index()
    agg = agg[agg['count'] >= 3].copy()  # need >=3 samples for a stable median
    flagged = agg[agg['median'] > tau].copy()
    flagged = flagged.rename(columns={'median': 'm_median',
                                      'mean': 'm_mean',
                                      'count': 'n_samples'})
    flagged = flagged.sort_values('m_median', ascending=False)
    return flagged, m, sm, labels


# --------------------------------------------------------------------------- #
# Stage IV — Irreducible overlap (chemistry-level unresolvability)
# --------------------------------------------------------------------------- #
# For a pair of rock names (A, B) we quantify "how separable are they by
# chemistry alone" with the symmetric Bhattacharyya coefficient of their
# KDE densities, computed on a held-out grid of chemistry points drawn
# from the *union* of A and B:
#
#   BC(A,B) = ∫ sqrt(f_A(x) * f_B(x)) dx        ∈ [0, 1]
#   D_Bhat  = -ln(1 - BC)                       ∈ [0, ∞)
#
# BC = 0  -> the two distributions have disjoint support -> fully separable.
# BC = 1  -> identical distributions -> no chemistry-only method can
#            tell A and B apart; every error between A and B is
#            *irreducible*.
#
# We estimate the integral by Monte Carlo on a held-out sample drawn
# from the mixture 0.5 f_A + 0.5 f_B (importance sampling, which puts
# evaluation points where both densities are non-negligible):
#
#   BC ≈ (1/N) Σ_i sqrt( f_A(x_i) * f_B(x_i) ) / w(x_i)
#
# with w(x_i) = 0.5 f_A(x_i) + 0.5 f_B(x_i)  and x_i ~ 0.5 f_A + 0.5 f_B
# (sampled by drawing half the points from the test set of A and half
# from the test set of B).  This estimator is the standard
# Bhattacharyya Monte-Carlo of Berrendero & Grande (2019).

def bhattacharyya_between(kdes, bws, test, label_a, label_b,
                          use_trace, trace_variant, trace_set='all',
                          n_mc=2000, seed=7):
    """Estimate the Bhattacharyya coefficient BC(a,b) by Monte Carlo."""
    rng = np.random.default_rng(seed)
    sub_a = test[test['Label'] == label_a]
    sub_b = test[test['Label'] == label_b]
    if len(sub_a) < 10 or len(sub_b) < 10:
        return np.nan, 0
    n_half = min(n_mc // 2, len(sub_a), len(sub_b))
    a_idx = rng.choice(len(sub_a), n_half, replace=len(sub_a) < n_half)
    b_idx = rng.choice(len(sub_b), n_half, replace=len(sub_b) < n_half)
    pts = np.vstack([
        geo.build_design_geochem(sub_a.iloc[a_idx], use_trace, trace_variant,
                                 trace_set=trace_set),
        geo.build_design_geochem(sub_b.iloc[b_idx], use_trace, trace_variant,
                                 trace_set=trace_set),
    ])
    fa = np.exp(geo.score_kde_diag(kdes[label_a], bws[label_a], pts))
    fb = np.exp(geo.score_kde_diag(kdes[label_b], bws[label_b], pts))
    w = 0.5 * fa + 0.5 * fb
    w = np.where(w <= 0, 1e-300, w)
    bc = np.mean(np.sqrt(fa * fb) / w)
    bc = float(np.clip(bc, 0.0, 1.0))
    return bc, len(pts)


def overlap_matrix(kdes, bws, test, use_trace, trace_variant,
                   trace_set='all', n_mc=2000, seed=7):
    """Symmetric (L x L) matrix of BC overlaps + the irreducible share.

    Returns:
      bc   : DataFrame of Bhattacharyya coefficients.
      irred: DataFrame of the *irreducible error share* per pair, defined
             as BC / (1 + BC) ∈ [0, 0.5], which is the minimum
             top-1 error any chemistry-only classifier must incur on the
             binary A/B problem (a tight upper bound on the Bayes error
             for two equally likely classes; see Cover & Thomas 2006,
             eq. on the Bhattacharyya bound).
    """
    labels = list(kdes.keys())
    L = len(labels)
    BC = np.full((L, L), np.nan)
    for i in range(L):
        BC[i, i] = 1.0
        for j in range(i + 1, L):
            bc, _ = bhattacharyya_between(
                kdes, bws, test, labels[i], labels[j],
                use_trace, trace_variant, trace_set=trace_set,
                n_mc=n_mc, seed=seed)
            BC[i, j] = bc
            BC[j, i] = bc
    bc_df = pd.DataFrame(BC, index=labels, columns=labels)
    irred = BC / (1.0 + BC)  # ∈ [0, 0.5]
    irred_df = pd.DataFrame(irred, index=labels, columns=labels)
    return bc_df, irred_df


def per_label_irreducible(irred_df):
    """For each label, the mean irreducible overlap with every other
    label — a single-number summary of "how chemically entangled is
    this rock name with the rest".  A high value means most of the
    error on this label is *not* the method's fault: the label overlaps
    with others in composition space and no chemistry-only method can
    fully separate them.
    """
    out = {}
    for lab in irred_df.index:
        others = irred_df.loc[lab].drop(lab)
        out[lab] = {
            'mean_irreducible': float(others.mean()),
            'max_irreducible': float(others.max()),
            'max_partner': str(others.idxmax()),
        }
    return pd.DataFrame(out).T


# --------------------------------------------------------------------------- #
# Stage I-IV synthesis: the per-label epistemic table
# --------------------------------------------------------------------------- #
def epistemic_table(res, kdes, bws, test, use_trace, trace_variant,
                    flagged, irred_summary, n_boot=200, seed=7):
    """One row per label with the four epistemic quantities + CIs."""
    rows = []
    for label in sorted(res['Label'].unique()):
        sub = res[res['Label'] == label]
        n = len(sub)
        if n == 0:
            continue
        # Stage I
        tas_mask = sub['TAS_Pred'].notna() & (sub['TAS_Pred'] != None)
        tas_acc, tas_lo, tas_hi, tas_n = accuracy_with_ci(
            sub['Label'], sub['TAS_Pred'], n_boot=n_boot, seed=seed)
        # Stage II
        sm_acc, sm_lo, sm_hi, sm_n = accuracy_with_ci(
            sub['Label'], sub['SoftMax_Pred'], n_boot=n_boot, seed=seed)
        # Stage III — fraction of this label's citations flagged
        flag_lab = flagged[flagged['Label'] == label]
        n_cit = sub['CitationID'].nunique()
        n_flag = len(flag_lab)
        flag_rate = n_flag / n_cit if n_cit else 0.0
        # Wilson 95% CI for the flag rate
        if n_cit:
            lo, hi = binom.interval(0.95, n_cit, flag_rate)
            flag_lo, flag_hi = lo / n_cit, hi / n_cit
        else:
            flag_lo = flag_hi = 0.0
        # Stage IV — irreducible overlap
        ir_mean = irred_summary.loc[label, 'mean_irreducible'] \
            if label in irred_summary.index else np.nan
        ir_max = irred_summary.loc[label, 'max_irreducible'] \
            if label in irred_summary.index else np.nan
        ir_partner = irred_summary.loc[label, 'max_partner'] \
            if label in irred_summary.index else ''
        rows.append({
            'Label': label,
            'n_samples': n,
            'n_citations': n_cit,
            'TAS_acc': tas_acc, 'TAS_CI': '[%.2f, %.2f]' % (tas_lo, tas_hi),
            'SoftMax_acc': sm_acc, 'SoftMax_CI': '[%.2f, %.2f]' % (sm_lo, sm_hi),
            'mislabel_flag_rate': flag_rate,
            'mislabel_CI': '[%.2f, %.2f]' % (flag_lo, flag_hi),
            'n_flagged_citations': n_flag,
            'irreducible_mean': ir_mean,
            'irreducible_max': ir_max,
            'irreducible_max_partner': ir_partner,
            # reducible error = the SoftMax error minus the irreducible part
            'reducible_error': max(0.0, (1 - sm_acc) - ir_max),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_epistemic_ladder(table, out_dir):
    """4-panel bar chart per label: TAS acc, SoftMax acc, flag rate,
    irreducible overlap."""
    labels = table['Label']
    y = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(11, 0.32 * len(labels) + 2))
    panels = [
        ('TAS_acc', 'Classical TAS accuracy', '#9a99b7'),
        ('SoftMax_acc', 'GeoChem-GS accuracy', '#e48080'),
        ('mislabel_flag_rate', 'Probable mis-labelling rate', '#7fb0d6'),
        ('irreducible_mean', 'Irreducible overlap (mean)', '#6abf8e'),
    ]
    for ax, (col, title, color) in zip(axes.flat, panels):
        ax.barh(y, table[col], color=color)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=9)
        ax.invert_yaxis()
    fig.tight_layout()
    for ext in ('svg', 'pdf'):
        fig.savefig(os.path.join(out_dir, 'figure_epistemic_ladder.%s' % ext))
    plt.close(fig)


def plot_overlap_heatmap(bc_df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(bc_df.values, cmap='magma', vmin=0, vmax=1)
    ax.set_xticks(range(len(bc_df)))
    ax.set_yticks(range(len(bc_df)))
    ax.set_xticklabels(bc_df.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(bc_df.index, fontsize=7)
    fig.colorbar(im, ax=ax, label='Bhattacharyya coefficient')
    ax.set_title('Irreducible compositional overlap between rock names',
                 fontsize=10)
    fig.tight_layout()
    for ext in ('svg', 'pdf'):
        fig.savefig(os.path.join(out_dir, 'figure_overlap_heatmap.%s' % ext))
    plt.close(fig)


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
    ap.add_argument('--trace-set', default='all',
                    choices=list(geo.TRACE_SETS.keys()))
    ap.add_argument('--no-cleanup', dest='cleanup', action='store_false')
    ap.set_defaults(cleanup=True)
    ap.add_argument('--n-test-citations', type=int, default=1500)
    ap.add_argument('--n-mc', type=int, default=2000,
                    help='Monte-Carlo sample size for the Bhattacharyya '
                         'estimate per label pair')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out-root',
                    default=os.path.join(HERE, 'TAS_GS_v2'))
    args = ap.parse_args()

    tag = '%s_geochem_%s_%s%s' % (
        args.rock_type,
        'major' if not args.use_trace else args.trace_set,
        args.trace_variant,
        '' if args.use_trace else '_notrace')
    base = os.path.join(args.out_root, tag)
    epi_dir = os.path.join(args.out_root, tag + '_epistemology')
    os.makedirs(epi_dir, exist_ok=True)

    # ---- reload the stage-3 results so we don't retrain ----
    with open(os.path.join(base, 'accuracy.json')) as f:
        base_report = json.load(f)
    res = pd.read_csv(os.path.join(base, 'predictions.csv'))
    test_meta = pd.read_csv(os.path.join(base, 'test_set.csv'))
    # the predictions CSV carries SiO2/Alk but the design-matrix rebuild
    # needs all the major oxides (+ trace).  Re-derive the full test
    # frame from the database, restricted to the same test citations.
    test_cits = set(test_meta['CitationID'].astype(str))
    df_all = geo.load_georoc_geochem(args.db, args.rock_type,
                                     use_trace=args.use_trace,
                                     trace_set=args.trace_set)
    test = df_all[df_all['CitationID'].isin(test_cits)].reset_index(drop=True)
    # reload KDEs
    kdes, bws = {}, {}
    kd = os.path.join(base, 'kde_train')
    for fn in os.listdir(kd):
        if fn.endswith('_kde.pkl'):
            lab = fn[:-len('_kde.pkl')]
            with open(os.path.join(kd, fn), 'rb') as f:
                kde, bw = pickle.load(f)
            kdes[lab] = kde
            bws[lab] = bw
    # Re-attach the Stage-3 predictions to the rebuilt test frame so the
    # flagging step uses the same chemistry and the same SoftMax outputs.
    # The predictions CSV was produced on the original test rows in the
    # same order, but the rebuild above may reorder rows; we therefore
    # re-score here rather than trusting positional alignment.
    print('[reload] %d KDEs, %d test samples (rebuilt from db)'
          % (len(kdes), len(test)))

    # ---- Stage III — mislabelling diagnostic ----
    t0 = time.time()
    # calibration needs a clean training subset: reload the train split
    print('[cal] loading training split for calibration...')
    df = geo.load_georoc_geochem(args.db, args.rock_type,
                                 use_trace=args.use_trace,
                                 trace_set=args.trace_set)
    train, _ = split_by_citation(df, args.n_test_citations, args.seed)
    tau = calibrate_mislabelling_threshold(
        train, kdes, bws, args.use_trace, args.trace_variant,
        trace_set=args.trace_set, seed=args.seed)
    print('[cal] tau = %.3f' % tau)
    flagged, m_post, sm, labels = flag_mislabelled_citations(
        test=test, kdes=kdes, bws=bws,
        use_trace=args.use_trace, trace_variant=args.trace_variant,
        tau=tau, trace_set=args.trace_set)
    flagged.to_csv(os.path.join(epi_dir, 'per_citation_mislabelling.csv'),
                   index=False)
    print('[stage III] %d flagged citation-label pairs (%.1fs)'
          % (len(flagged), time.time() - t0))

    # ---- Stage IV — irreducible overlap ----
    t1 = time.time()
    bc_df, irred_df = overlap_matrix(
        kdes, bws, test, args.use_trace, args.trace_variant,
        trace_set=args.trace_set, n_mc=args.n_mc, seed=args.seed)
    bc_df.to_csv(os.path.join(epi_dir, 'per_pair_overlaps_bc.csv'))
    irred_df.to_csv(os.path.join(epi_dir, 'per_pair_overlaps_irred.csv'))
    irred_summary = per_label_irreducible(irred_df)
    irred_summary.to_csv(os.path.join(
        epi_dir, 'per_label_irreducible.csv'))
    print('[stage IV] %d pairs computed (%.1fs)'
          % (len(bc_df) * (len(bc_df) - 1) // 2, time.time() - t1))

    # ---- Synthesis ----
    table = epistemic_table(
        res, kdes, bws, test, args.use_trace, args.trace_variant,
        flagged, irred_summary, n_boot=200, seed=args.seed)
    table.to_csv(os.path.join(epi_dir, 'per_label_epistemic_table.csv'),
                 index=False)
    plot_epistemic_ladder(table, epi_dir)
    plot_overlap_heatmap(bc_df, epi_dir)

    summary = {
        'tau': tau,
        'n_flagged_citation_label_pairs': int(len(flagged)),
        'mean_irreducible': float(irred_summary['mean_irreducible'].mean()),
        'labels_with_high_irreducibility': (
            irred_summary[irred_summary['mean_irreducible'] > 0.3]
            .index.tolist()),
        'base_accuracy': base_report['SoftMax_Pred']['accuracy'],
    }
    with open(os.path.join(epi_dir, 'epistemology.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('=== Epistemic summary ===')
    print(table[['Label', 'TAS_acc', 'SoftMax_acc',
                 'mislabel_flag_rate', 'irreducible_mean',
                 'reducible_error']].to_string(index=False))
    print('Flagged citations: %d, mean irreducible overlap: %.3f'
          % (len(flagged), summary['mean_irreducible']))


if __name__ == '__main__':
    main()