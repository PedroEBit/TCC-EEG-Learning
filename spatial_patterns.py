"""
Padroes espaciais (Haufe et al., 2014) da convolucao depthwise do EEGNet.

Por que nao ler os pesos direto: os pesos da depthwise sao *filtros* de um modelo
discriminativo. Um peso grande pode servir para CANCELAR ruido num canal, nao para
extrair sinal dele, entao ler o filtro como topografia e invalido. A transformacao
de Haufe converte filtro em padrao:

    A = Cov(X) . w

onde X e o sinal que entra naquela convolucao. Na EEGNet isso nao e o EEG bruto: a
depthwise atua sobre a SAIDA da convolucao temporal, entao cada filtro espacial tem
que ser pareado com a covariancia do sinal ja filtrado pela sua propria banda.

Pergunta: o modelo redescobre C3/C4, os eletrodos sobre o cortex motor de mao?
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from train_final import load_subject, prepare, model_path, ROOT, RUNS_DIR

# Ordem oficial do montage do BCI Competition IV-2a (22 canais). O MNE nomeia
# so os cinco rotulados no GDF; os demais viram EEG-0..EEG-16 na mesma ordem.
MONTAGE = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
           'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
           'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
           'P1', 'Pz', 'P2', 'POz']

# Faixa sensoriomotora: linhas C e CP, que cobrem M1 e S1. Definida aqui de forma
# explicita porque o "esperado por acaso" depende so do tamanho deste conjunto.
SENSORIMOTOR = ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4']
SM_IDX = [MONTAGE.index(c) for c in SENSORIMOTOR]
CHANCE = len(SM_IDX) / len(MONTAGE)


def temporal_maps(model, X):
    """Saida da convolucao temporal: (n_trials, F1, n_channels, n_times)."""
    sub = tf.keras.Model(model.inputs, model.get_layer('temporal_conv').output)
    return sub.predict(prepare(X), verbose=0)


def patterns_for_model(model, X):
    """Um padrao de Haufe por par (filtro temporal f, multiplicador de profundidade d)."""
    W = model.get_layer('spatial_conv').get_weights()[0]   # (n_ch, 1, F1, D)
    n_ch, _, F1, D = W.shape
    maps = temporal_maps(model, X)                          # (n, F1, n_ch, T)
    out = []
    for f in range(F1):
        Xf = maps[:, f]                                     # (n, n_ch, T)
        Xf = Xf.transpose(1, 0, 2).reshape(n_ch, -1)        # (n_ch, n*T)
        Xf = Xf - Xf.mean(axis=1, keepdims=True)
        cov = np.cov(Xf)
        for d in range(D):
            out.append(cov @ W[:, 0, f, d])
    return np.array(out)                                    # (F1*D, n_ch)


def analyse(n_classes):
    X, _ = load_subject(1, 'T', n_classes)
    seeds = sorted(json.loads(f.read_text())['seed']
                   for f in RUNS_DIR.glob(f'run_{n_classes}c_seed*.json'))
    per_seed, ranks = [], []
    for s in seeds:
        p = model_path(n_classes, s)
        if not p.exists():
            continue
        A = np.abs(patterns_for_model(tf.keras.models.load_model(p, compile=False), X))
        A = A / A.sum(axis=1, keepdims=True)                # cada padrao soma 1
        mass = float(A[:, SM_IDX].sum(axis=1).mean())       # massa na faixa sensoriomotora
        mean_pat = A.mean(axis=0)
        order = np.argsort(-mean_pat)                       # 0 = canal mais forte
        rank = {c: int(np.where(order == MONTAGE.index(c))[0][0]) + 1 for c in ('C3', 'C4')}
        per_seed.append({'seed': s, 'sensorimotor_mass': mass,
                         'rank_C3': rank['C3'], 'rank_C4': rank['C4'],
                         'top3_channels': [MONTAGE[i] for i in order[:3]]})
        ranks += [rank['C3'], rank['C4']]
        print(f"  {n_classes}c seed{s}: massa sensoriomotora {mass:.1%} "
              f"(acaso {CHANCE:.1%}) | C3 em {rank['C3']}o, C4 em {rank['C4']}o de 22 "
              f"| top3 {per_seed[-1]['top3_channels']}")
    m = np.array([d['sensorimotor_mass'] for d in per_seed])
    return {
        'n_classes': n_classes,
        'sensorimotor_channels': SENSORIMOTOR,
        'chance_level': CHANCE,
        'sensorimotor_mass_mean': float(m.mean()),
        'sensorimotor_mass_min': float(m.min()),
        'sensorimotor_mass_max': float(m.max()),
        'mean_rank_C3_C4': float(np.mean(ranks)),
        'per_seed': per_seed,
    }


if __name__ == '__main__':
    out = {}
    for nc in (2, 4):
        print(f'\n{nc} classes — {len(SENSORIMOTOR)} canais sensoriomotores '
              f'de {len(MONTAGE)} (acaso {CHANCE:.1%})')
        out[f'{nc}class'] = analyse(nc)
    (ROOT / 'results_spatial_patterns.json').write_text(json.dumps(out, indent=2))
    print()
    for k, v in out.items():
        print(f"{k}: massa {v['sensorimotor_mass_mean']:.1%} "
              f"[{v['sensorimotor_mass_min']:.1%}, {v['sensorimotor_mass_max']:.1%}] "
              f"vs acaso {v['chance_level']:.1%} | rank medio de C3/C4 "
              f"{v['mean_rank_C3_C4']:.1f} de 22")
    print()
    print('results_spatial_patterns.json salvo.')
