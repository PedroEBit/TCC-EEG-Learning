"""Regenera as figuras do post a partir dos modelos finais salvos por train_final.py."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from train_final import load_subject, prepare, SUBJECT_ID, MODELS_DIR, ROOT

OUT = ROOT / 'linkedin_post'
OUT.mkdir(exist_ok=True)
sns.set_style('whitegrid')

SPECS = [
    (2, 'eegnet_a01_2class_seed0.keras', ['Left Hand', 'Right Hand']),
    (4, 'eegnet_a01_4class_seed0.keras', ['Left Hand', 'Right Hand', 'Feet', 'Tongue']),
]

data = {}
for n_classes, fname, names in SPECS:
    model = tf.keras.models.load_model(MODELS_DIR / fname, compile=False)
    X_te, y_te = load_subject(SUBJECT_ID, 'E', n_classes)
    proba = model.predict(prepare(X_te), verbose=0)
    y_pred = np.argmax(proba, axis=1)
    data[n_classes] = dict(y=y_te, pred=y_pred, proba=proba, names=names,
                           acc=(y_pred == y_te).mean())

# --- Figura 1: matrizes de confusao lado a lado --------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, n_classes in zip(axes, (2, 4)):
    d = data[n_classes]
    cm = np.zeros((n_classes, n_classes))
    for t, p in zip(d['y'], d['pred']):
        cm[t, p] += 1
    cm = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, ax=ax,
                xticklabels=d['names'], yticklabels=d['names'], cbar=False,
                annot_kws={'size': 11})
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f"{n_classes}-class — A01, held-out session "
                 f"(acc {d['acc']:.1%})", fontsize=12)
fig.suptitle('EEGNet + Gaussian noise augmentation — BCI Competition IV-2a, subject A01',
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / 'fig1_confusion_matrices.png', dpi=150, bbox_inches='tight')
print('fig1_confusion_matrices.png')

# --- Figura 2: distribuicao de confianca do softmax ----------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, n_classes in zip(axes, (2, 4)):
    d = data[n_classes]
    conf = d['proba'].max(axis=1)
    hit = d['pred'] == d['y']
    bins = np.linspace(1.0 / n_classes, 1.0, 25)
    ax.hist(conf[hit], bins=bins, alpha=.75, label=f'Correct (n={hit.sum()})',
            color='#2a9d8f', edgecolor='white')
    ax.hist(conf[~hit], bins=bins, alpha=.75, label=f'Incorrect (n={(~hit).sum()})',
            color='#e76f51', edgecolor='white')
    ax.axvline(conf[hit].mean(), color='#2a9d8f', ls='--', lw=2)
    ax.axvline(conf[~hit].mean(), color='#e76f51', ls='--', lw=2)
    ax.set_xlabel('Softmax confidence (max probability)')
    ax.set_ylabel('Trials')
    ax.set_title(f"{n_classes}-class — mean {conf[hit].mean():.0%} correct "
                 f"vs {conf[~hit].mean():.0%} incorrect", fontsize=12)
    ax.legend()
fig.suptitle('Confidence separates by correctness — a usable rejection threshold for online BCI',
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / 'fig2_confidence_distribution.png', dpi=150, bbox_inches='tight')
print('fig2_confidence_distribution.png')

# --- Figura 3: filtros temporais aprendidos ------------------------------
from scipy.fft import rfft, rfftfreq

# Os dois modelos lado a lado: com 232 trials (4-class) quase todos os filtros
# convergem para mu/beta; com 116 (2-class) metade degenera para quase-DC.
# O tamanho de treino vem dos runs, para o rotulo nunca divergir do pipeline.
N_TRAIN = {json.loads(f.read_text())['n_classes']: json.loads(f.read_text())['n_train_orig']
           for f in (ROOT / 'runs').glob('run_*_seed0.json')}
fig, axes = plt.subplots(4, 4, figsize=(14, 11))
peaks = {}
for row_pair, (n_classes, fname, _) in zip(((0, 1), (2, 3)), reversed(SPECS)):
    model = tf.keras.models.load_model(MODELS_DIR / fname, compile=False)
    Wt = model.get_layer('temporal_conv').get_weights()[0][0, :, 0, :]  # (125, 8)
    pk_list = []
    for i in range(8):
        ax = axes[row_pair[i // 4]][i % 4]
        spec = np.abs(rfft(Wt[:, i], 512))
        freqs = rfftfreq(512, 1 / 250)
        band = (freqs >= 1) & (freqs <= 45)
        f, s = freqs[band], spec[band]
        pk = float(f[np.argmax(s)])
        pk_list.append(pk)
        in_band = 8 <= pk <= 30
        ax.axvspan(8, 13, color='#2a9d8f', alpha=.18)
        ax.axvspan(13, 30, color='#457b9d', alpha=.12)
        ax.plot(f, s / s.max(), color='#1d3557' if in_band else '#adb5bd', lw=1.6)
        ax.axvline(pk, color='#e76f51' if in_band else '#adb5bd', ls='--', lw=1.4)
        ax.set_title(f'filter {i} — peak {pk:.1f} Hz', fontsize=9,
                     color='#1d3557' if in_band else '#868e96')
        ax.set_xlim(1, 45); ax.set_yticks([])
        if i >= 4:
            ax.set_xlabel('Hz', fontsize=8)
    n_in = sum(1 for p in pk_list if 8 <= p <= 30)
    peaks[n_classes] = pk_list
    n_trials = N_TRAIN[n_classes]
    axes[row_pair[0]][0].set_ylabel(
        f'{n_classes}-class model\n({n_trials} train trials)\n'
        f'{n_in}/8 filters in 8–30 Hz', fontsize=10)

fig.suptitle('Learned temporal filters vs. training set size\n'
             'mu (8–13 Hz, green) and beta (13–30 Hz, blue) — network was fed the full 4–40 Hz band',
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / 'fig3_temporal_filters.png', dpi=150, bbox_inches='tight')
print(f'fig3_temporal_filters.png  (picos: { {k: [round(p,1) for p in v] for k,v in peaks.items()} })')

summary = {str(k): {'accuracy': float(v['acc']),
                    'mean_conf_correct': float(v['proba'].max(1)[v['pred'] == v['y']].mean()),
                    'mean_conf_incorrect': float(v['proba'].max(1)[v['pred'] != v['y']].mean())}
           for k, v in data.items()}
summary['temporal_filter_peaks_hz'] = {
    str(k): [round(float(p), 1) for p in v] for k, v in peaks.items()}
print(json.dumps(summary, indent=2))


# --- Figura 4: membros individuais vs ensemble ---------------------------
ens = json.loads((ROOT / 'results_ensemble.json').read_text())
runs_by_nc = {}
for f in sorted((ROOT / 'runs').glob('run_*.json')):
    r = json.loads(f.read_text())
    runs_by_nc.setdefault(r['n_classes'], []).append(r)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, nc in zip(axes, (2, 4)):
    e = ens[f'{nc}class']
    accs = sorted(r['test_accuracy'] for r in runs_by_nc[nc])
    ax.scatter(accs, [0] * len(accs), s=140, color='#adb5bd',
               zorder=3, label=f'individual seeds (n={len(accs)})')
    for name, y, color in [('soft_vote_all', 1, '#2a9d8f'),
                           ('hard_vote_all', 2, '#457b9d')]:
        ax.scatter([e[name]['accuracy']], [y], s=200, marker='D',
                   color=color, zorder=3, label=name.replace('_', ' '))
    ax.axvline(np.mean(accs), color='#adb5bd', ls=':', lw=1.5)
    ax.axvline(1.0 / nc, color='#e76f51', ls='--', lw=1.5)
    ax.text(1.0 / nc, 2.6, ' chance', color='#e76f51', fontsize=9, va='top')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['members', 'soft vote', 'hard vote'])
    ax.set_ylim(-0.7, 3.4)
    ax.set_xlim(min(1.0 / nc, min(accs)) - 0.05, 1.0)
    ax.set_xlabel('Test accuracy (held-out session A01E)')
    ax.set_title(f'{nc}-class — members span {min(accs):.1%} to {max(accs):.1%}',
                 fontsize=12)
    ax.legend(loc='upper left', fontsize=8, framealpha=.95)
fig.suptitle('Averaging seeds removes the run-to-run lottery\n'
             'ensemble members are never selected using the test set', fontsize=13)
fig.tight_layout()
fig.savefig(OUT / 'fig4_ensemble_vs_members.png', dpi=150, bbox_inches='tight')
print('fig4_ensemble_vs_members.png')
