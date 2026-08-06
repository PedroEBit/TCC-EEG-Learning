"""
Pipeline final EEGNet / BCI Competition IV-2a, sujeito A01.

Diferenca em relacao ao notebook de exploracao: o split de validacao acontece
ANTES do augmentation. No notebook, `validation_split=0.2` do Keras pegava os
ultimos 20% do array ja aumentado -- copias ruidosas de trials cujos originais
estavam no treino. A val_accuracy chegava a 1.000, o que tornava o EarlyStopping
inutil. As metricas de teste nao eram afetadas (A01E e uma sessao separada),
mas a selecao de epoca era guiada por ruido.

Protocolo: within-subject, cross-session. Treino = A01T, teste = A01E, o split
oficial do BCI Competition IV-2a. Nenhum augmentation toca treino->teste.
"""

import json
from pathlib import Path

import numpy as np
import mne
from mne.io import read_raw_gdf
from scipy.io import loadmat

import tensorflow as tf
tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'))

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, SeparableConv2D,
    BatchNormalization, Activation, AveragePooling2D,
    Dropout, Flatten, Dense
)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

mne.set_log_level('ERROR')

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / 'data' / 'bcic_iv_2a'
LABELS_DIR = ROOT / 'data' / 'true_labels'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

SFREQ, N_CHANNELS, SUBJECT_ID = 250, 22, 1
CODE_TO_IDX = {769: 0, 770: 1, 771: 2, 772: 3}
EVENT_ID = {'left_hand': 769, 'right_hand': 770, 'feet': 771, 'tongue': 772}


def load_subject(subject_id, split='T', n_classes=4, l_freq=4.0, h_freq=40.0):
    raw = read_raw_gdf(str(DATA_DIR / f'A0{subject_id}{split}.gdf'),
                       preload=True, verbose=False)
    raw.pick_types(eeg=True)
    raw.pick(raw.ch_names[:N_CHANNELS])
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', verbose=False)

    if split == 'T':
        events, _ = mne.events_from_annotations(
            raw, event_id={str(k): k for k in CODE_TO_IDX}, verbose=False)
        events = events[np.isin(events[:, 2], list(CODE_TO_IDX))]
        epochs = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=0.0, tmax=4.0,
                            baseline=None, preload=True, verbose=False)
        X = epochs.get_data().astype(np.float32)
        y = np.array([CODE_TO_IDX[c] for c in epochs.events[:, 2]], dtype=np.int32)
    else:
        events, _ = mne.events_from_annotations(raw, event_id={'783': 783}, verbose=False)
        epochs = mne.Epochs(raw, events, event_id={'cue': 783}, tmin=0.0, tmax=4.0,
                            baseline=None, preload=True, verbose=False)
        X = epochs.get_data().astype(np.float32)
        y = (loadmat(str(LABELS_DIR / f'A0{subject_id}E.mat'))['classlabel']
             .flatten() - 1).astype(np.int32)[:len(X)]

    if n_classes == 2:
        mask = np.isin(y, [0, 1])
        X, y = X[mask], y[mask]
    return X, y


def build_eegnet(n_channels=22, n_times=1001, n_classes=2, sfreq=250,
                 F1=8, D=2, F2=16, dropout_rate=0.5):
    kern_len = sfreq // 2
    inputs = Input(shape=(1, n_channels, n_times), name='eeg_input')

    x = Conv2D(F1, (1, kern_len), padding='same', use_bias=False,
               data_format='channels_first', name='temporal_conv')(inputs)
    x = BatchNormalization(axis=1)(x)

    x = DepthwiseConv2D((n_channels, 1), depth_multiplier=D,
                        depthwise_constraint=max_norm(1.0), use_bias=False,
                        data_format='channels_first', name='spatial_conv')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4), data_format='channels_first')(x)
    x = Dropout(dropout_rate)(x)

    x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False,
                        data_format='channels_first', name='separable_conv')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8), data_format='channels_first')(x)
    x = Dropout(dropout_rate)(x)

    x = Flatten()(x)
    outputs = Dense(n_classes, activation='softmax',
                    kernel_constraint=max_norm(0.25), name='classifier')(x)
    return Model(inputs, outputs, name='EEGNet')


def prepare(X):
    """z-score por epoca (nao usa estatistica do dataset) + eixo de profundidade."""
    mu = X.mean(axis=(1, 2), keepdims=True)
    sd = X.std(axis=(1, 2), keepdims=True) + 1e-8
    return ((X - mu) / sd)[:, np.newaxis, :, :]


def augment_gaussian_noise(X, y, n_copies=5, noise_std=0.1, rng=None):
    rng = rng or np.random
    out_X, out_y = [X], [y]
    epoch_std = X.std(axis=(1, 2), keepdims=True)
    for _ in range(n_copies):
        noise = rng.standard_normal(X.shape).astype(np.float32) * (noise_std * epoch_std)
        out_X.append(X + noise)
        out_y.append(y)
    return np.concatenate(out_X), np.concatenate(out_y)


def stratified_split(y, val_frac=0.2, rng=None):
    """Indices de treino/validacao, proporcao de classes preservada."""
    tr, va = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(idx))))
        va.extend(idx[:n_val])
        tr.extend(idx[n_val:])
    return np.sort(np.array(tr)), np.sort(np.array(va))


def run(n_classes, seed, epochs, batch_size, patience, save_path=None):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    X_tr_full, y_tr_full = load_subject(SUBJECT_ID, 'T', n_classes)
    X_te, y_te = load_subject(SUBJECT_ID, 'E', n_classes)

    # --- split ANTES do augmentation ---
    tr_idx, va_idx = stratified_split(y_tr_full, 0.2, rng)
    X_tr, y_tr = X_tr_full[tr_idx], y_tr_full[tr_idx]
    X_va, y_va = X_tr_full[va_idx], y_tr_full[va_idx]

    # augmentation so no treino; validacao e teste ficam limpos
    X_tr_aug, y_tr_aug = augment_gaussian_noise(X_tr, y_tr, n_copies=5,
                                                noise_std=0.1, rng=rng)

    model = build_eegnet(N_CHANNELS, X_tr.shape[-1], n_classes, SFREQ)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    hist = model.fit(
        prepare(X_tr_aug), y_tr_aug,
        validation_data=(prepare(X_va), y_va),
        epochs=epochs, batch_size=batch_size, verbose=0,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=patience,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5),
        ],
    )

    proba = model.predict(prepare(X_te), verbose=0)
    y_pred = np.argmax(proba, axis=1)
    acc = float((y_pred == y_te).mean())
    kappa = float(cohen_kappa_score(y_te, y_pred))
    hit = y_pred == y_te
    conf_ok = float(proba[hit].max(axis=1).mean()) if hit.any() else 0.0
    conf_no = float(proba[~hit].max(axis=1).mean()) if (~hit).any() else 0.0

    if save_path:
        model.save(save_path)

    return {
        'seed': seed, 'n_classes': n_classes,
        'n_train_orig': int(len(X_tr)), 'n_train_aug': int(len(X_tr_aug)),
        'n_val': int(len(X_va)), 'n_test': int(len(X_te)),
        'epochs_run': len(hist.history['loss']),
        'best_val_acc': float(max(hist.history['val_accuracy'])),
        'final_val_acc': float(hist.history['val_accuracy'][-1]),
        'test_accuracy': acc, 'test_kappa': kappa,
        'mean_confidence_correct': conf_ok,
        'mean_confidence_incorrect': conf_no,
        'confusion_matrix': confusion_matrix(y_te, y_pred).tolist(),
        'report': classification_report(y_te, y_pred, output_dict=True, zero_division=0),
    }


RUNS_DIR = ROOT / 'runs'
CONFIGS = (
    [(2, s, dict(epochs=300, batch_size=32, patience=25)) for s in range(5)] +
    [(4, s, dict(epochs=150, batch_size=64, patience=25)) for s in range(3)]
)


def collect():
    """Agrega os runs individuais em results_final.json."""
    results = {'2class': [], '4class': []}
    for f in sorted(RUNS_DIR.glob('run_*.json')):
        r = json.loads(f.read_text())
        results[f"{r['n_classes']}class"].append(r)
    for key, runs in results.items():
        runs.sort(key=lambda r: r['seed'])
        if not runs:
            continue
        a = np.array([r['test_accuracy'] for r in runs])
        k = np.array([r['test_kappa'] for r in runs])
        print(f'{key} (n={len(runs)}): acc {a.mean():.4f} +/- {a.std(ddof=1):.4f} '
              f'[min {a.min():.4f}, mediana {np.median(a):.4f}, max {a.max():.4f}] | '
              f'kappa {k.mean():.4f} +/- {k.std(ddof=1):.4f}')
    (ROOT / 'results_final.json').write_text(json.dumps(results, indent=2))
    return results


def model_path(n_classes, seed):
    return MODELS_DIR / f'eegnet_a01_{n_classes}class_seed{seed}.keras'


if __name__ == '__main__':
    # Cada run e gravado assim que termina, entao a execucao e retomavel:
    # rodar de novo so refaz o que estiver faltando. Todo seed salva seu modelo,
    # porque o ensemble precisa de todos eles (ver explicando_ensemble.ipynb).
    RUNS_DIR.mkdir(exist_ok=True)
    for n_classes, seed, kw in CONFIGS:
        out = RUNS_DIR / f'run_{n_classes}c_seed{seed}.json'
        save = model_path(n_classes, seed)
        if out.exists() and save.exists():
            print(f'[{n_classes}-class seed {seed}] ja existe, pulando', flush=True)
            continue
        r = run(n_classes, seed, save_path=save, **kw)
        out.write_text(json.dumps(r, indent=2))
        print(f"[{n_classes}-class seed {seed}] acc={r['test_accuracy']:.4f} "
              f"kappa={r['test_kappa']:.4f} val={r['best_val_acc']:.3f} "
              f"({r['epochs_run']} ep)", flush=True)

    print()
    collect()
    print('\nresults_final.json salvo.')
