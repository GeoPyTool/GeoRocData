"""
24_Make_Figures.py
==================
Generate all publication-quality figures for the revised manuscript.
SVG + PDF for each, colorblind-safe, font >= 8pt, Nature Geoscience style.

Figures:
  Fig 1 — Classic TAS diagram with GEOROC data (updated from original)
  Fig 2 — Bandwidth comparison (Silverman / Median / Adaptive) with LOO-CV
  Fig 3 — Method comparison bar chart (11 methods, all + TAS-comparable)
  Fig 4 — Dimension vs accuracy curve (D=10, 18, 36)
  Fig 5 — Epistemic ladder: 4-panel per-label bar chart
  Fig 6 — Irreducible overlap heatmap (Bhattacharyya coefficient matrix)
  Fig 7 — Confusion matrix for the best method (Random Forest)

Usage:
  python 24_Make_Figures.py --out-dir ../文章撰写/未发表/TAS-GS/figures/new
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib import cm

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
for mod, path in [('tasgs', '20_TAS_GS_v2.py'),
                  ('geo', '21_GeoChem_GS.py')]:
    spec = importlib.util.spec_from_file_location(
        mod, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.modules[mod] = m

from tasgs import (  # noqa: E402
    load_tas_polygons, split_by_citation, classify_tas, DEFAULT_DB,
    TAS_COMPARABLE, TAS_FIELD_TO_ROCKNAME,
)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'svg.fonttype': 'none',
    'pdf.fonttype': 'truetype',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-safe palette (Wong 2011, Nature Methods)
C_BLUE = '#0072B2'
C_RED = '#D55E00'
C_GREEN = '#009E73'
C_PURPLE = '#CC79A7'
C_GRAY = '#9a99b7'
C_ORANGE = '#E69F00'
C_CYAN = '#56B4E9'
C_YELLOW = '#F0E442'


def save_fig(fig, out_dir, name):
    """Save as SVG + PDF."""
    for ext in ('svg', 'pdf'):
        fig.savefig(os.path.join(out_dir, '%s.%s' % (name, ext)),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  saved %s' % name)


# --------------------------------------------------------------------------- #
# Figure 1 — Classic TAS diagram with GEOROC volcanic data
# --------------------------------------------------------------------------- #
def make_fig1(out_dir):
    """Classic TAS diagram with field polygons and GEOROC volcanic data,
    colored by rock name using the same color scheme as the original
    Figure 3/4 (Color_Config/VOL_color_dict.json)."""
    import sqlite3
    import json
    # load color dict
    cd_path = os.path.join(HERE, 'Color_Config', 'VOL_color_dict.json')
    with open(cd_path, 'r') as f:
        color_dict = json.load(f)

    conn = sqlite3.connect(DEFAULT_DB)
    df = pd.read_sql_query(
        'SELECT "SIO2(WT%)" as sio2, "NA2O(WT%)" as na2o, "K2O(WT%)" as k2o, '
        '"ROCK NAME" as rockname FROM georoc_data WHERE "ROCK TYPE"="VOL"',
        conn)
    conn.close()
    for c in ['sio2', 'na2o', 'k2o']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['sio2', 'na2o', 'k2o'])
    df = df[(df['sio2'] > 30) & (df['sio2'] < 85)]
    df['alk'] = df['na2o'] + df['k2o']
    df = df[(df['alk'] >= 0) & (df['alk'] < 20)]
    # parse rock name labels (same as 20_TAS_GS_v2.parse_rock_name)
    spec = importlib.util.spec_from_file_location(
        'tasgs', os.path.join(HERE, '20_TAS_GS_v2.py'))
    tasgs_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasgs_mod)
    df['Label'] = df['rockname'].map(tasgs_mod.parse_rock_name)
    df = df[df['Label'].notna()].copy()

    polys, cord = load_tas_polygons()
    fig, ax = plt.subplots(figsize=(10, 7))
    # plot each label group with its color from the original color dict
    for label, grp in df.groupby('Label'):
        if label not in color_dict:
            continue
        rgba = tuple(color_dict[label])
        n = len(grp)
        # alpha scales with log(data_amount) as in original 8_DB_TAS_Base.py
        base = 0.28
        data_scale = n * n
        data_range = (grp['sio2'].max() - grp['sio2'].min()) * \
                     (grp['alk'].max() - grp['alk'].min())
        if data_range <= 0:
            alpha = 0.05
        else:
            alpha = base / (np.log10(max(data_scale, 10)) *
                            np.log10(max(data_range, 10)))
            alpha = np.clip(alpha, 0.02, 0.3)
        ax.scatter(grp['sio2'].values, grp['alk'].values,
                   color=rgba, edgecolors='none', alpha=alpha,
                   s=12, rasterized=True, label=label)
    # draw TAS field boundaries
    for code, path in polys.items():
        verts = path.vertices
        ax.plot(verts[:, 0], verts[:, 1], 'k-', lw=0.5, alpha=0.6)
    # field labels at polygon centroids
    for code, path in polys.items():
        verts = path.vertices
        cx, cy = verts[:, 0].mean(), verts[:, 1].mean()
        ax.text(cx, cy, code, fontsize=7, ha='center', va='center',
                alpha=0.7, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.3, pad=1,
                          edgecolor='none'))
    ax.set_xlabel(r'SiO$_2$ (wt%, anhydrous basis)')
    ax.set_ylabel(r'Na$_2$O + K$_2$O (wt%, anhydrous basis)')
    ax.set_xlim(30, 85)
    ax.set_ylim(0, 19)
    ax.set_title('Classic TAS diagram with GEOROC volcanic data '
                 '(colored by rock name)', fontsize=9)
    # compact legend
    ax.legend(loc='upper left', fontsize=5, ncol=2, markerscale=2,
              framealpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, 'Figure_1_TAS_classic')


# --------------------------------------------------------------------------- #
# Figure 3 — Method comparison bar chart
# --------------------------------------------------------------------------- #
def make_fig3(out_dir):
    """Grouped bar chart of all methods."""
    csv_path = os.path.join(HERE, 'TAS_GS_v2', 'VOL_compare_core8_ilr',
                            'master_accuracy_table.csv')
    if not os.path.exists(csv_path):
        print('  [skip] master_accuracy_table.csv not found')
        return
    table = pd.read_csv(csv_path)
    # shorten method names
    table['Method_short'] = table['Method'].str.replace(' (on ilr)', '',
                                                         regex=False)
    table['Method_short'] = table['Method_short'].str.replace(
        ' (adaptive bw)', '', regex=False)
    table['Method_short'] = table['Method_short'].str.replace(
        ' (major-only, ', ' (maj ', regex=False)
    table['Method_short'] = table['Method_short'].str.replace(
        ' (core8, ', ' (c8 ', regex=False)
    table['Method_short'] = table['Method_short'].str.replace(
        ' (shrinkage, ', ' (shr ', regex=False)
    table['Method_short'] = table['Method_short'].str.replace(
        ' (polygon)', '', regex=False)
    table['Method_short'] = table['Method_short'].str.replace(
        '+SoftMax', '+SM', regex=False)

    methods = table['Method_short'].tolist()
    all_acc = table['Accuracy_all'].tolist()
    tas_acc = table['Accuracy_TAScomp'].tolist()
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    bars_all = ax.bar(x - w/2, all_acc, w, label='All labels',
                      color=C_BLUE, alpha=0.8, edgecolor='white', lw=0.5)
    bars_tas = ax.bar(x + w/2, tas_acc, w, label='TAS-comparable',
                      color=C_ORANGE, alpha=0.8, edgecolor='white', lw=0.5)
    # highlight best
    best_idx = int(np.argmax(all_acc))
    bars_all[best_idx].set_edgecolor(C_RED)
    bars_all[best_idx].set_linewidth(2)
    bars_tas[best_idx].set_edgecolor(C_RED)
    bars_tas[best_idx].set_linewidth(2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Top-1 accuracy')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.set_title('Method comparison on GEOROC volcanic rocks '
                 '(core8 ilr, $D{=}18$, 19\,986 test samples)',
                 fontsize=9)
    for i, v in enumerate(all_acc):
        ax.text(i - w/2, v + 0.01, '%.2f' % v, ha='center', fontsize=6.5,
                rotation=90, va='bottom')
    for i, v in enumerate(tas_acc):
        ax.text(i + w/2, v + 0.01, '%.2f' % v, ha='center', fontsize=6.5,
                rotation=90, va='bottom')
    ax.axhline(y=0.37, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.text(len(methods) - 0.5, 0.38, 'TAS baseline\n(all labels, 0.37)',
            fontsize=6, ha='right', color='gray')
    fig.tight_layout()
    save_fig(fig, out_dir, 'Figure_3_method_comparison')


# --------------------------------------------------------------------------- #
# Figure 4 — Dimension vs accuracy curve
# --------------------------------------------------------------------------- #
def make_fig4(out_dir):
    """Dimension versus accuracy curve (KDE vs LDA vs RF)."""
    dims = [2, 10, 18, 36]
    # KDE+SoftMax
    kde_all = [0.55, 0.559, 0.560, 0.339]
    kde_comp = [0.53, 0.551, 0.580, 0.359]
    # LDA
    lda_all = [np.nan, 0.674, 0.694, np.nan]  # LDA on D=2 not tested; D=36 not stable
    lda_comp = [np.nan, 0.728, 0.737, np.nan]
    # RF
    rf_all = [np.nan, 0.767, 0.767, np.nan]
    rf_comp = [np.nan, 0.807, 0.808, np.nan]
    # TAS baseline (constant, dimension-independent)
    tas_all = [0.37] * len(dims)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (title, kde, lda, rf, tas) in zip(axes, [
        ('All labels', kde_all, lda_all, rf_all, tas_all),
        ('TAS-comparable labels', kde_comp, lda_comp, rf_comp,
         [0.43] * len(dims))]):
        ax.plot(dims, tas, 'k--o', ms=4, lw=1, label='Classical TAS',
                alpha=0.5)
        ax.plot(dims, kde, 'o-', color=C_PURPLE, lw=1.5, ms=5,
                label='KDE+SoftMax')
        valid = ~np.isnan(lda)
        ax.plot(np.array(dims)[valid], np.array(lda)[valid], 's-',
                color=C_GREEN, lw=1.5, ms=5, label='LDA')
        valid = ~np.isnan(rf)
        ax.plot(np.array(dims)[valid], np.array(rf)[valid], 'D-',
                color=C_RED, lw=1.5, ms=5, label='Random Forest')
        ax.set_xlabel('Feature dimension $D$')
        ax.set_ylabel('Top-1 accuracy')
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0.25, 0.9)
        ax.set_xticks(dims)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Dimension versus accuracy: the curse of dimensionality '
                 'strikes KDE but not RF', fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, 'Figure_4_dimension_vs_accuracy')


# --------------------------------------------------------------------------- #
# Figure 5 — Epistemic ladder (4-panel per-label bar chart)
# --------------------------------------------------------------------------- #
def make_fig5(out_dir):
    """4-panel bar chart: TAS acc, SoftMax acc, mislabel flag, irreducible."""
    csv_path = os.path.join(HERE, 'TAS_GS_v2',
                            'VOL_geochem_major_notrace_epistemology',
                            'per_label_epistemic_table.csv')
    if not os.path.exists(csv_path):
        print('  [skip] epistemic table not found')
        return
    table = pd.read_csv(csv_path)
    # drop rows with NaN irreducible or n<5
    table = table[table['n_samples'] >= 5].copy()
    table = table.dropna(subset=['irreducible_mean'])
    table = table.sort_values('SoftMax_acc', ascending=True)

    labels = table['Label'].tolist()
    y = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 0.35 * len(labels) + 2))
    panels = [
        ('TAS_acc', 'Classical TAS accuracy', C_GRAY),
        ('SoftMax_acc', 'KDE+SoftMax accuracy', C_BLUE),
        ('mislabel_flag_rate', 'Probable mis-labelling rate', C_ORANGE),
        ('irreducible_mean', 'Irreducible overlap (mean)', C_GREEN),
    ]
    for ax, (col, title, color) in zip(axes.flat, panels):
        vals = table[col].fillna(0).values
        ax.barh(y, vals, color=color, alpha=0.8, edgecolor='white', lw=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=8)
        ax.axvline(x=0.5, color='gray', ls=':', lw=0.5, alpha=0.5)
    fig.suptitle('Per-label epistemic decomposition (major-only, 200 test '
                 'citations, seed 7)', fontsize=9, y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, 'Figure_5_epistemic_ladder')


# --------------------------------------------------------------------------- #
# Figure 6 — Irreducible overlap heatmap
# --------------------------------------------------------------------------- #
def make_fig6(out_dir):
    """Bhattacharyya coefficient heatmap between rock-name pairs."""
    csv_path = os.path.join(HERE, 'TAS_GS_v2',
                            'VOL_geochem_major_notrace_epistemology',
                            'per_pair_overlaps_bc.csv')
    if not os.path.exists(csv_path):
        print('  [skip] overlap matrix not found')
        return
    bc = pd.read_csv(csv_path, index_col=0)
    # order by cluster on the diagonal for readability
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(bc.values, cmap='magma', vmin=0, vmax=1,
                   aspect='equal')
    n = len(bc)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(bc.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(bc.index, fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Bhattacharyya coefficient BC(A,B)', fontsize=8)
    ax.set_title('Irreducible compositional overlap between rock names',
                 fontsize=9)
    # mark high-overlap cells
    for i in range(n):
        for j in range(i + 1, n):
            v = bc.iloc[i, j]
            if v > 0.3:
                ax.text(j, i, '%.2f' % v, ha='center', va='center',
                        fontsize=5, color='white' if v > 0.5 else 'black')
    fig.tight_layout()
    save_fig(fig, out_dir, 'Figure_6_overlap_heatmap')


# --------------------------------------------------------------------------- #
# Figure 7 — Confusion matrix for Random Forest
# --------------------------------------------------------------------------- #
def make_fig7(out_dir):
    """Row-normalised confusion matrix for the best method (RF)."""
    csv_path = os.path.join(HERE, 'TAS_GS_v2', 'VOL_compare_core8_ilr',
                            'per_method_predictions', 'RandomForest.csv')
    if not os.path.exists(csv_path):
        print('  [skip] RF predictions not found')
        return
    df = pd.read_csv(csv_path)
    labels = sorted(df['Label'].unique())
    mat = pd.DataFrame(0, index=labels, columns=labels, dtype=float)
    for _, row in df.iterrows():
        if row['Pred'] in mat.columns:
            mat.loc[row['Label'], row['Pred']] += 1
    # row-normalise
    mat = mat.div(mat.sum(axis=1).replace(0, 1), axis=0)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat.values, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Recall (row-normalised)', fontsize=8)
    ax.set_title('Confusion matrix: Random Forest on ilr (core8)',
                 fontsize=9)
    # annotate diagonal
    for i in range(n):
        v = mat.iloc[i, i]
        color = 'white' if v > 0.5 else 'black'
        ax.text(i, i, '%.2f' % v, ha='center', va='center',
                fontsize=5, color=color)
    fig.tight_layout()
    save_fig(fig, out_dir, 'Figure_7_confusion_matrix_RF')


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir',
                    default=os.path.join(os.path.dirname(HERE),
                    '文章撰写', '未发表', 'TAS-GS', 'figures', 'new'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('Generating publication-quality figures...')
    make_fig1(args.out_dir)
    make_fig3(args.out_dir)
    make_fig4(args.out_dir)
    make_fig5(args.out_dir)
    make_fig6(args.out_dir)
    make_fig7(args.out_dir)
    print('Done. Figures saved to %s' % args.out_dir)


if __name__ == '__main__':
    main()