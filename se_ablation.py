"""
Ablacao do bloco Squeeze-and-Excitation (Exercicio 8) sob o protocolo final.

Pergunta: o SE block do eeg_fundamentos_e_arquitetura.ipynb (+148 parametros)
melhora de verdade, ou os +8.6 pontos daquele notebook eram ruido de semente?

O numero do notebook nao serve como resposta porque (a) era max(val_accuracy),
que o README ja marca como otimista em ate 10 pontos, e (b) era uma rodada
contra uma rodada, e a dispersao entre sementes no 2-class vai de 57.6% a 92.4%.

Aqui: mesma arquitetura base, mesmo split, mesmo augmentation, mesmo batch,
mesma paciencia, mesmo teste (A01E). So o bloco SE e ligado. Cinco sementes por
tarefa, pareadas com os runs ja existentes em runs/run_*.json.

Uso:
    python se_ablation.py            # roda tudo que falta, depois agrega
    python se_ablation.py --collect  # so agrega o que ja existe
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization, Activation,
    AveragePooling2D, Dropout, Flatten, Dense,
    GlobalAveragePooling2D, Multiply, Reshape,
)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from train_final import (
    load_subject, prepare, augment_gaussian_noise, stratified_split,
    SFREQ, N_CHANNELS, SUBJECT_ID, ROOT, RUNS_DIR,
)

SE_DIR = RUNS_DIR / 'se'


def build_eegnet_se(n_channels=22, n_times=1001, n_classes=2, sfreq=250,
                    F1=8, D=2, F2=16, dropout_rate=0.5, se_ratio=4):
    """EEGNet + Squeeze-and-Excitation sobre os filtros espaciais.

    Identico a build_eegnet do train_final.py, exceto pelo bloco SE inserido
    depois da ELU do spatial_conv e antes do primeiro AveragePooling, que e
    onde ele esta no Exercicio 8. Custo: 2 Dense pequenas.
        Dense(F1*D -> F1*D//se_ratio) = 16*4 + 4  = 68
        Dense(F1*D//se_ratio -> F1*D) =  4*16 + 16 = 80
                                                  ---
                                                   148
    """
    kern_len = sfreq // 2
    n_spatial = F1 * D
    inputs = Input(shape=(1, n_channels, n_times), name='eeg_input')

    x = Conv2D(F1, (1, kern_len), padding='same', use_bias=False,
               data_format='channels_first', name='temporal_conv')(inputs)
    x = BatchNormalization(axis=1)(x)

    x = DepthwiseConv2D((n_channels, 1), depth_multiplier=D,
                        depthwise_constraint=max_norm(1.0), use_bias=False,
                        data_format='channels_first', name='spatial_conv')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('elu')(x)

    # --- Squeeze-and-Excitation ---
    se = GlobalAveragePooling2D(data_format='channels_first')(x)
    se = Dense(n_spatial // se_ratio, activation='relu', name='se_squeeze')(se)
    se = Dense(n_spatial, activation='sigmoid', name='se_excite')(se)
    se = Reshape((n_spatial, 1, 1))(se)
    x = Multiply(name='se_scale')([x, se])

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
    return Model(inputs, outputs, name='EEGNet_SE')


def run_se(n_classes, seed, epochs, batch_size, patience):
    """Copia fiel de train_final.run(), trocando so o construtor do modelo."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    X_tr_full, y_tr_full = load_subject(SUBJECT_ID, 'T', n_classes)
    X_te, y_te = load_subject(SUBJECT_ID, 'E', n_classes)

    tr_idx, va_idx = stratified_split(y_tr_full, 0.2, rng)
    X_tr, y_tr = X_tr_full[tr_idx], y_tr_full[tr_idx]
    X_va, y_va = X_tr_full[va_idx], y_tr_full[va_idx]
    X_tr_aug, y_tr_aug = augment_gaussian_noise(X_tr, y_tr, n_copies=5,
                                                noise_std=0.1, rng=rng)

    model = build_eegnet_se(N_CHANNELS, X_tr.shape[-1], n_classes, SFREQ)
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

    val_acc_restored = float(
        (model.predict(prepare(X_va), verbose=0).argmax(1) == y_va).mean())
    proba = model.predict(prepare(X_te), verbose=0)
    y_pred = np.argmax(proba, axis=1)

    return {
        'seed': seed, 'n_classes': n_classes, 'variant': 'se',
        'n_params_total': int(model.count_params()),
        'epochs_run': len(hist.history['loss']), 'epochs_cap': int(epochs),
        'val_acc_restored': val_acc_restored,
        'best_val_acc': float(max(hist.history['val_accuracy'])),
        'test_accuracy': float((y_pred == y_te).mean()),
        'test_kappa': float(cohen_kappa_score(y_te, y_pred)),
        'confusion_matrix': confusion_matrix(y_te, y_pred).tolist(),
        'y_pred': y_pred.tolist(),
        'y_true': y_te.tolist(),
    }


def collect():
    """Pareia SE contra os runs base ja existentes, por semente."""
    base = {}
    for f in RUNS_DIR.glob('run_*.json'):
        r = json.loads(f.read_text())
        base[(r['n_classes'], r['seed'])] = r

    summary = {}
    for n_classes in (2, 4):
        rows = []
        for seed in range(5):
            f = SE_DIR / f'se_{n_classes}c_seed{seed}.json'
            if not f.exists() or (n_classes, seed) not in base:
                continue
            se = json.loads(f.read_text())
            rows.append({'seed': seed,
                         'base': base[(n_classes, seed)]['test_accuracy'],
                         'se': se['test_accuracy'],
                         'base_kappa': base[(n_classes, seed)]['test_kappa'],
                         'se_kappa': se['test_kappa']})
        if not rows:
            continue
        b = np.array([r['base'] for r in rows])
        s = np.array([r['se'] for r in rows])
        improved = int((s > b).sum())
        # teste do sinal: P(>= improved sucessos em n, p=0.5), bilateral
        from math import comb
        n = len(rows)
        k = max(improved, n - improved)
        p_sign = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
        summary[f'{n_classes}class'] = {
            'n_seeds': n,
            'base_mean': float(b.mean()), 'base_std': float(b.std(ddof=1)),
            'se_mean': float(s.mean()), 'se_std': float(s.std(ddof=1)),
            'delta_mean': float(s.mean() - b.mean()),
            'delta_median': float(np.median(s - b)),
            'seeds_improved': improved,
            'sign_test_p': float(p_sign),
            'per_seed': rows,
        }
        print(f'{n_classes}-class  base {b.mean():.4f} +/- {b.std(ddof=1):.4f}  |  '
              f'SE {s.mean():.4f} +/- {s.std(ddof=1):.4f}  |  '
              f'delta {s.mean()-b.mean():+.4f}  |  melhora em {improved}/{n}  |  '
              f'sinal p={p_sign:.3f}')
        for r in rows:
            print(f'    seed {r["seed"]}: {r["base"]:.4f} -> {r["se"]:.4f} '
                  f'({r["se"]-r["base"]:+.4f})')

    (ROOT / 'results_se_ablation.json').write_text(json.dumps(summary, indent=2))
    print('\nresults_se_ablation.json salvo.')
    return summary


CONFIGS = (
    [(2, s, dict(epochs=300, batch_size=32, patience=25)) for s in range(5)] +
    [(4, s, dict(epochs=150, batch_size=64, patience=25)) for s in range(5)]
)

if __name__ == '__main__':
    if '--collect' not in sys.argv:
        SE_DIR.mkdir(parents=True, exist_ok=True)
        print(f'params SE (2-class): {build_eegnet_se(n_classes=2).count_params()}', flush=True)
        print(f'params SE (4-class): {build_eegnet_se(n_classes=4).count_params()}', flush=True)
        for n_classes, seed, kw in CONFIGS:
            out = SE_DIR / f'se_{n_classes}c_seed{seed}.json'
            if out.exists():
                print(f'[SE {n_classes}c seed {seed}] ja existe, pulando', flush=True)
                continue
            r = run_se(n_classes, seed, **kw)
            out.write_text(json.dumps(r, indent=2))
            print(f"[SE {n_classes}c seed {seed}] acc={r['test_accuracy']:.4f} "
                  f"kappa={r['test_kappa']:.4f} ({r['epochs_run']} ep)", flush=True)
        print(flush=True)
    collect()
