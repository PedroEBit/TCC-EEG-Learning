"""
Ensemble entre sementes para EEGNet / BCI IV-2a A01.

Motivacao: uma unica execucao nao e um resultado. Em 2 classes a acuracia de
teste varia de 57.6% a 92.4% entre cinco sementes. Reportar 88.9% era reportar
um sorteio favoravel. O ensemble ataca exatamente a componente de variancia
desse erro.

Cada semente difere em quatro coisas: inicializacao dos pesos, mascaras de
dropout, ordem dos batches e -- o mais importante aqui -- a particao
treino/validacao, porque stratified_split e derivado da mesma seed. Ou seja,
cada modelo viu 115 dos 144 trials, um subconjunto diferente. Isso e o que
descorrelaciona os erros e e a condicao para o ensemble funcionar.

REGRA METODOLOGICA: nenhuma decisao sobre quais modelos entram no ensemble
pode olhar para o teste. A regra de descarte aqui usa apenas best_val_acc,
medida no split de validacao limpo de cada semente.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

from train_final import load_subject, prepare, MODELS_DIR, ROOT, RUNS_DIR, model_path

SUBJECT_ID = 1


def load_seed_models(n_classes):
    """Modelos e metadados de validacao de cada semente disponivel."""
    members = []
    for f in sorted(RUNS_DIR.glob(f'run_{n_classes}c_seed*.json')):
        meta = json.loads(f.read_text())
        p = model_path(n_classes, meta['seed'])
        if not p.exists():
            print(f'  ! modelo da seed {meta["seed"]} ausente, ignorando')
            continue
        members.append({
            'seed': meta['seed'],
            'val_acc': meta['best_val_acc'],
            'solo_acc': meta['test_accuracy'],
            'model': tf.keras.models.load_model(p, compile=False),
        })
    return members


def metrics(y_true, proba):
    y_pred = proba.argmax(1)
    hit = y_pred == y_true
    return {
        'accuracy': float(hit.mean()),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
        'mean_confidence_correct': float(proba[hit].max(1).mean()) if hit.any() else 0.0,
        'mean_confidence_incorrect': float(proba[~hit].max(1).mean()) if (~hit).any() else 0.0,
        'y_pred': y_pred,
    }


def soft_vote(probas):
    """Media das probabilidades. Preserva a incerteza de cada membro."""
    return np.mean(probas, axis=0)


def hard_vote(probas, n_classes):
    """Voto majoritario nos argmax. Empate resolvido pela soma das probabilidades.

    Retorna a fracao de votos por classe, nao a contagem bruta, para que a
    "confianca" do hard voting seja comparavel a do soft voting: 0.8 passa a
    significar "4 dos 5 membros concordaram".
    """
    votes = np.stack([p.argmax(1) for p in probas])            # (n_models, n_trials)
    counts = np.stack([(votes == c).sum(0) for c in range(n_classes)], axis=1)
    tie_break = soft_vote(probas) * 1e-6
    return (counts + tie_break) / len(probas)


def disagreement_matrix(probas):
    """Fracao de trials em que cada par de modelos discorda. Proxy de diversidade."""
    preds = [p.argmax(1) for p in probas]
    n = len(preds)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = (preds[i] != preds[j]).mean()
    return D


def mcnemar(y_true, pred_a, pred_b):
    """Teste exato de McNemar pareado entre dois classificadores nos mesmos trials."""
    from scipy.stats import binomtest
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n01 = int((~a_ok & b_ok).sum())   # a erra, b acerta
    n10 = int((a_ok & ~b_ok).sum())   # a acerta, b erra
    if n01 + n10 == 0:
        return n01, n10, 1.0
    p = binomtest(n10, n01 + n10, 0.5).pvalue
    return n01, n10, float(p)


def evaluate(n_classes, verbose=True):
    members = load_seed_models(n_classes)
    if len(members) < 2:
        print(f'{n_classes}-class: menos de 2 modelos disponiveis, pulando')
        return None

    X, y = load_subject(SUBJECT_ID, 'E', n_classes)
    Xp = prepare(X)
    for m in members:
        m['proba'] = m['model'].predict(Xp, verbose=0)

    probas = [m['proba'] for m in members]
    solo = np.array([m['solo_acc'] for m in members])
    val = np.array([m['val_acc'] for m in members])

    # Regra de descarte fixada a priori e calculada SO na validacao:
    # descarta membros a mais de 1 desvio-padrao abaixo da media de val_acc.
    cut = val.mean() - val.std(ddof=1)
    keep = val >= cut
    kept_seeds = [m['seed'] for m, k in zip(members, keep) if k]

    res = {
        'n_classes': n_classes,
        'n_members': len(members),
        'seeds': [m['seed'] for m in members],
        'solo_accuracy_mean': float(solo.mean()),
        'solo_accuracy_std': float(solo.std(ddof=1)),
        'solo_accuracy_min': float(solo.min()),
        'solo_accuracy_max': float(solo.max()),
        'solo_accuracy_median': float(np.median(solo)),
        'val_cutoff': float(cut),
        'seeds_kept_by_val_rule': kept_seeds,
        'mean_pairwise_disagreement': float(
            disagreement_matrix(probas)[np.triu_indices(len(probas), 1)].mean()),
    }

    variants = {
        'soft_vote_all': soft_vote(probas),
        'hard_vote_all': hard_vote(probas, n_classes),
    }
    if keep.sum() >= 2 and keep.sum() < len(members):
        variants['soft_vote_val_filtered'] = soft_vote(
            [p for p, k in zip(probas, keep) if k])

    for name, proba in variants.items():
        m = metrics(y, proba)
        res[name] = {k: v for k, v in m.items() if k != 'y_pred'}

    # Comparacao pareada contra o membro mediano (nao contra o melhor: escolher o
    # melhor olhando o teste seria exatamente o erro que estamos tentando evitar).
    median_idx = int(np.argsort(solo)[len(solo) // 2])
    pred_median = probas[median_idx].argmax(1)
    pred_ens = variants['soft_vote_all'].argmax(1)
    n01, n10, p = mcnemar(y, pred_median, pred_ens)
    res['mcnemar_vs_median_member'] = {
        'median_member_seed': members[median_idx]['seed'],
        'median_only_correct': n10, 'ensemble_only_correct': n01, 'p_value': p,
    }

    res['confusion_matrix_soft_vote'] = confusion_matrix(y, pred_ens).tolist()
    res['report_soft_vote'] = classification_report(
        y, pred_ens, output_dict=True, zero_division=0)

    if verbose:
        print(f'\n{"="*66}\n{n_classes} CLASSES — {len(members)} membros '
              f'(seeds {res["seeds"]})\n{"="*66}')
        print(f'  membro individual : {solo.mean():.4f} +/- {solo.std(ddof=1):.4f} '
              f'[min {solo.min():.4f}, mediana {np.median(solo):.4f}, max {solo.max():.4f}]')
        print(f'  discordancia media entre pares : {res["mean_pairwise_disagreement"]:.3f}')
        for name in variants:
            r = res[name]
            print(f'  {name:24s}: acc {r["accuracy"]:.4f} | kappa {r["kappa"]:.4f} '
                  f'| conf ok/erro {r["mean_confidence_correct"]:.3f}/'
                  f'{r["mean_confidence_incorrect"]:.3f}')
        print(f'  regra de val: corte {cut:.3f} -> mantem seeds {kept_seeds}')
        mc = res['mcnemar_vs_median_member']
        print(f'  McNemar vs membro mediano (seed {mc["median_member_seed"]}): '
              f'ensemble acerta sozinho {mc["ensemble_only_correct"]}, '
              f'membro acerta sozinho {mc["median_only_correct"]}, p={mc["p_value"]:.4f}')

    return res


if __name__ == '__main__':
    out = {}
    for nc in (2, 4):
        r = evaluate(nc)
        if r:
            out[f'{nc}class'] = r
    (ROOT / 'results_ensemble.json').write_text(json.dumps(out, indent=2))
    print('\nresults_ensemble.json salvo.')
