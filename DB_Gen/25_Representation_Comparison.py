"""
25_Representation_Comparison.py
=================================
Systematic comparison of data representations (raw wt%, log, ilr)
× classifiers (KDE, LDA, RF) × validation strategies (hold-out, K-fold).

This script answers three questions:
1. Does ilr help or hurt each classifier?
2. What is the optimal K for GroupKFold by-citation CV?
3. Which representation × classifier × K combination is best?

Outputs:
  representation_comparison_table.csv
  figure_representation_comparison.{svg,pdf}
  validation_strategy_table.csv
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'm', os.path.join(HERE, '23_Classifiers_Compare.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.modules['m'] = m

from tasgs import close_composition, ilr_transform, DEFAULT_DB, TAS_COMPARABLE

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'svg.fonttype': 'none', 'pdf.fonttype': 'truetype',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def build_representations(df, use_trace=False, trace_set='all'):
    """Build RAW, LOG, and ILR design matrices from the same data."""
    major = m.geo.MAJOR_OXIDES
    Z_raw = df[major].values.astype(float)
    Z_raw = np.where(np.isfinite(Z_raw) & (Z_raw > 0), Z_raw, 1e-6)
    Z_log = np.log10(Z_raw)

    # ilr with rest part closing to 100
    rest = 100.0 - Z_raw.sum(axis=1)
    comp = close_composition([Z_raw[:, i] for i in range(
        Z_raw.shape[1])] + [rest])
    Z_ilr = ilr_transform(comp)

    if use_trace:
        te = m.geo.TRACE_SETS[trace_set]
        T = df[te].values.astype(float)
        T = np.where(np.isfinite(T) & (T > 0), T, 1e-3)
        tr_rest = 1e6 - T.sum(axis=1)
        tr_comp = close_composition(list(T.T) + [tr_rest])
        Z_ilr_trace = ilr_transform(tr_comp)
        Z_log_trace = np.log10(T)
        Z_raw_trace = T
        Z_ilr = np.hstack([Z_ilr, Z_ilr_trace])
        Z_log = np.hstack([Z_log, Z_log_trace])
        Z_raw = np.hstack([Z_raw, Z_raw_trace])

    return {'RAW': Z_raw, 'LOG': Z_log, 'ILR': Z_ilr}


def run_kfold(Z, y_enc, groups, le, y_raw, K, clf_fn):
    """Run K-fold GroupKFold CV, return mean/std accuracy."""
    gkf = GroupKFold(n_splits=K)
    accs = []
    mask_global = np.array([l in TAS_COMPARABLE for l in y_raw])
    accs_comp = []
    for tr_idx, te_idx in gkf.split(Z, y_enc, groups):
        clf = clf_fn()
        clf.fit(Z[tr_idx], y_enc[tr_idx])
        p = le.inverse_transform(clf.predict(Z[te_idx]))
        yt = y_raw[te_idx]
        accs.append(np.mean(p == yt))
        m = mask_global[te_idx]
        if m.any():
            accs_comp.append(np.mean(p[m] == yt[m]))
    return np.mean(accs), np.std(accs), np.mean(accs_comp) if accs_comp else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--rock-type', default='VOL')
    ap.add_argument('--no-trace', dest='use_trace', action='store_false')
    ap.set_defaults(use_trace=False)
    ap.add_argument('--trace-set', default='core8')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'TAS_GS_v2',
                     'representation_comparison'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('[load] %s / trace=%s' % (args.rock_type, args.use_trace))
    df = m.geo.load_georoc_geochem(args.db, args.rock_type,
                                   use_trace=args.use_trace,
                                   trace_set=args.trace_set)
    print('[load] %d rows, %d labels, %d citations'
          % (len(df), df['Label'].nunique(), df['CitationID'].nunique()))

    groups = df['CitationID'].values
    y = df['Label'].values
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    reps = build_representations(df, args.use_trace, args.trace_set)
    print('[reps] RAW=%d LOG=%d ILR=%d'
          % (reps['RAW'].shape[1], reps['LOG'].shape[1],
             reps['ILR'].shape[1]))

    classifiers = {
        'RF': lambda: RandomForestClassifier(
            n_estimators=200, max_depth=20,
            random_state=args.seed, n_jobs=-1),
        'LDA': lambda: LDA(),
    }

    # ---- Part 1: representation × classifier (5-fold) ----
    print('\n=== Part 1: representation × classifier (5-fold GroupKFold) ===')
    rows = []
    for rep_name, Z in reps.items():
        for clf_name, clf_fn in classifiers.items():
            t = time.time()
            acc, std, acc_comp = run_kfold(
                Z, y_enc, groups, le, y, K=5, clf_fn=clf_fn)
            row = {'Representation': rep_name, 'Classifier': clf_name,
                   'K': 5, 'Accuracy_all': acc, 'Std': std,
                   'Accuracy_TAScomp': acc_comp,
                   'D': Z.shape[1], 'time_sec': time.time() - t}
            rows.append(row)
            print('  %-4s %-4s  all=%.3f±%.3f  comp=%.3f  (%.0fs)'
                  % (rep_name, clf_name, acc, std, acc_comp,
                     time.time() - t))

    # ---- Part 2: K-fold effect (RF on RAW only) ----
    print('\n=== Part 2: effect of K (RF on RAW) ===')
    for K in [3, 5, 10]:
        t = time.time()
        acc, std, acc_comp = run_kfold(
            reps['RAW'], y_enc, groups, le, y, K=K,
            clf_fn=classifiers['RF'])
        row = {'Representation': 'RAW', 'Classifier': 'RF',
               'K': K, 'Accuracy_all': acc, 'Std': std,
               'Accuracy_TAScomp': acc_comp,
               'D': reps['RAW'].shape[1], 'time_sec': time.time() - t}
        rows.append(row)
        print('  K=%2d  all=%.3f±%.3f  comp=%.3f  (%.0fs)'
              % (K, acc, std, acc_comp, time.time() - t))

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir,
                 'representation_comparison_table.csv'), index=False)

    # ---- Figure: representation × classifier heatmap ----
    pivot = table[table['K'] == 5].pivot_table(
        index='Representation', columns='Classifier',
        values='Accuracy_all')
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(pivot.values, cmap='YlGnBu', vmin=0.5, vmax=0.85,
                   aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.iloc[i, j]
            ax.text(j, i, '%.3f' % v, ha='center', va='center',
                    fontsize=10,
                    color='white' if v > 0.7 else 'black')
    fig.colorbar(im, ax=ax, label='Top-1 accuracy')
    ax.set_title('Representation × classifier accuracy (5-fold CV)',
                 fontsize=9)
    fig.tight_layout()
    for ext in ('svg', 'pdf'):
        fig.savefig(os.path.join(args.out_dir,
                    'figure_representation_comparison.%s' % ext),
                    bbox_inches='tight')
    plt.close(fig)

    # ---- Figure: K-fold effect ----
    k_rows = table[(table['Representation'] == 'RAW') &
                  (table['Classifier'] == 'RF')]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(k_rows['K'], k_rows['Accuracy_all'],
                yerr=k_rows['Std'], fmt='o-', color='#D55E00', lw=1.5,
                capsize=3, ms=6, label='All labels')
    ax.errorbar(k_rows['K'], k_rows['Accuracy_TAScomp'],
                yerr=k_rows['Std'], fmt='s--', color='#0072B2', lw=1.5,
                capsize=3, ms=6, label='TAS-comparable')
    ax.set_xlabel('Number of folds K')
    ax.set_ylabel('Top-1 accuracy')
    ax.set_title('Effect of K in GroupKFold by-citation CV '
                 '(RF on RAW wt%)', fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xticks(k_rows['K'])
    fig.tight_layout()
    for ext in ('svg', 'pdf'):
        fig.savefig(os.path.join(args.out_dir,
                    'figure_kfold_effect.%s' % ext),
                    bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump({'rows': rows, 'config': vars(args)}, f, indent=2,
                  default=str)
    print('\nDone. Results saved to %s' % args.out_dir)


if __name__ == '__main__':
    main()