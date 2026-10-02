"""Within-subject nos 9 sujeitos: treina em A0xT, testa em A0xE, para cada x.

Isto e REPLICACAO, nao transferencia. Cada sujeito ganha o seu proprio modelo,
treinado do zero no pipeline identico ao do A01 -- mesma arquitetura, mesmo
augmentation, mesmo batch, mesma paciencia, mesma banda. A unica coisa que muda
e de quem sao os dados. E a condicao do CLAUDE.md secao 3, regra 2: ablacao
pareada muda so a coisa medida.

Por que vem antes da Fase 3
---------------------------
O zero-shot de A01 da 0.4931 no A05 (2-class, no acaso). Isso e transferencia
falhando, ou e um sujeito que nenhum modelo decodifica? Sem o within do A05 nao
da para saber, e todo numero de transferencia fica sem denominador. O numero que
se reporta na Fase 3 e transferencia / teto within do PROPRIO sujeito.

E os 90 modelos salvos aqui sao os mesmos que a Fase 3 usa como fonte no
leave-one-subject-out. Nada aqui e descartavel.

Ordem do loop: SEMENTE por fora, sujeito por dentro
---------------------------------------------------
De proposito. Se a execucao for interrompida, sobra cobertura COMPLETA dos 9
sujeitos com menos sementes -- e ai os testes pareados entre sujeitos
(Wilcoxon sobre os 8 de teste) ainda rodam, so com n de sementes menor, que se
reporta honestamente. Com sujeito por fora sobraria "5 sementes dos sujeitos 2 a
5", que reintroduz exatamente a limitacao de cobertura que esta fase existe para
matar.

Resumivel: pula qualquer (sujeito, tarefa, semente) cujo JSON e .keras existam.
O A01 ja esta completo em runs/within/, entao ele e pulado inteiro -- e por isso
que os 10 modelos do A01 nao correm risco de ser retreinados.

Uso:  python within9.py            # treina o que falta, depois agrega
      python within9.py --collect  # so agrega o que ja existe
"""

import json
import sys
import time

import numpy as np

import paths
from train_final import run, model_path, ROOT, MODELS_DIR

SUBJECTS = range(1, 10)
SEEDS = range(5)
BAND = (4.0, 40.0)

# Identicos aos do A01 (train_final.CONFIGS). Mudar qualquer um destes quebra a
# comparabilidade com os resultados ja publicados no README.
TASK_KW = {
    2: dict(epochs=300, batch_size=32, patience=25),
    4: dict(epochs=150, batch_size=64, patience=25),
}
TASKS = (2, 4)

# Chaves do retorno de run() que pertencem ao `config`, nao ao `metrics`.
_CONFIG_KEYS = ('seed', 'n_classes', 'augmented', 'subject')


def already_done(subject, n_classes, seed):
    """True se o run E o modelo existem. Os dois, como no train_final.py."""
    p = paths.run_path(experiment='within', subject=subject, n_classes=n_classes,
                       arch='base', seed=seed, augmented=True, band=BAND,
                       align=None)
    return p.exists() and model_path(n_classes, seed, subject=subject).exists()


def train_one(subject, n_classes, seed):
    """Treina um (sujeito, tarefa, semente) e grava via paths.save_run.

    Devolve (metrics, segundos) ou (None, 0.0) se ja existia.
    """
    if already_done(subject, n_classes, seed):
        return None, 0.0

    t0 = time.time()
    r = run(n_classes, seed, subject=subject,
            save_path=model_path(n_classes, seed, subject=subject),
            **TASK_KW[n_classes])
    elapsed = time.time() - t0

    preds = r.pop('predictions')
    config = paths.make_config(
        experiment='within', subject=subject, n_classes=n_classes, arch='base',
        seed=seed, augmented=True, band=BAND, align=None,
        predictions_complete=True)
    metrics = {k: v for k, v in r.items() if k not in _CONFIG_KEYS}
    paths.save_run(config, metrics, preds)
    return metrics, elapsed


def collect():
    """Agrega os within dos 9 em results_within9.json.

    Reporta POR SUJEITO. A media dos 9 existe para comparar com a literatura,
    mas nao e o resultado: a dispersao entre sujeitos neste dataset e enorme.
    """
    out = {}
    for n_classes in TASKS:
        subs = {}
        for s in SUBJECTS:
            runs = paths.load_runs('within', subject=s, n_classes=n_classes,
                                   arch='base', augmented=True)
            if not runs:
                continue
            runs.sort(key=lambda r: r['config']['seed'])
            acc = np.array([r['metrics']['test_accuracy'] for r in runs])
            kap = np.array([r['metrics']['test_kappa'] for r in runs])
            subs[f'A{s:02d}'] = {
                'subject': s,
                'n_seeds': len(runs),
                'seeds': [r['config']['seed'] for r in runs],
                'accuracy_mean': float(acc.mean()),
                # ddof=1 e dp amostral; com n=1 da nan, e nan e honesto aqui
                'accuracy_std': float(acc.std(ddof=1)) if len(acc) > 1 else float('nan'),
                'accuracy_min': float(acc.min()),
                'accuracy_median': float(np.median(acc)),
                'accuracy_max': float(acc.max()),
                'kappa_mean': float(kap.mean()),
                'per_seed': [{'seed': r['config']['seed'],
                              'accuracy': r['metrics']['test_accuracy'],
                              'kappa': r['metrics']['test_kappa'],
                              'val_acc_restored': r['metrics'].get('val_acc_restored'),
                              'epochs_run': r['metrics'].get('epochs_run')}
                             for r in runs],
            }
        if not subs:
            continue
        means = np.array([v['accuracy_mean'] for v in subs.values()])
        out[f'{n_classes}class'] = {
            'subjects': subs,
            'across_subjects': {
                'n_subjects': len(subs),
                'mean_of_means': float(means.mean()),
                'median_of_means': float(np.median(means)),
                'min': float(means.min()), 'max': float(means.max()),
                'std_across_subjects': float(means.std(ddof=1)) if len(means) > 1
                                       else float('nan'),
            },
        }

    for n_classes in TASKS:
        key = f'{n_classes}class'
        if key not in out:
            continue
        blk = out[key]
        print(f'\n{"=" * 78}')
        print(f'WITHIN-SUBJECT  {n_classes} classes   (treino A0xT -> teste A0xE)')
        print(f'{"=" * 78}')
        print(f'{"sujeito":8s} {"n":3s} {"media":8s} {"dp":8s} '
              f'{"min":8s} {"mediana":8s} {"max":8s} {"kappa":8s}')
        print('-' * 78)
        for name, v in sorted(blk['subjects'].items()):
            print(f'{name:8s} {v["n_seeds"]:<3d} {v["accuracy_mean"]:.4f}   '
                  f'{v["accuracy_std"]:.4f}   {v["accuracy_min"]:.4f}   '
                  f'{v["accuracy_median"]:.4f}   {v["accuracy_max"]:.4f}   '
                  f'{v["kappa_mean"]:.4f}')
        a = blk['across_subjects']
        print('-' * 78)
        print(f'{a["n_subjects"]} sujeitos: media das medias {a["mean_of_means"]:.4f}, '
              f'mediana {a["median_of_means"]:.4f}, '
              f'faixa [{a["min"]:.4f}, {a["max"]:.4f}], '
              f'dp entre sujeitos {a["std_across_subjects"]:.4f}')

    (ROOT / 'results_within9.json').write_text(
        json.dumps(out, indent=2), encoding='utf-8')
    print('\nresults_within9.json salvo.')
    return out


if __name__ == '__main__':
    if '--collect' not in sys.argv:
        MODELS_DIR.mkdir(exist_ok=True)
        todo = [(seed, s, nc) for seed in SEEDS for s in SUBJECTS for nc in TASKS
                if not already_done(s, nc, seed)]
        print(f'{len(todo)} treinos a fazer '
              f'(de {len(list(SEEDS)) * len(list(SUBJECTS)) * len(TASKS)}); '
              f'o resto ja esta no disco.', flush=True)

        times = []
        for i, (seed, subject, n_classes) in enumerate(todo, 1):
            m, dt = train_one(subject, n_classes, seed)
            if m is None:
                continue
            times.append(dt)
            eta = np.mean(times) * (len(todo) - i) / 60
            print(f'[{i}/{len(todo)}] A{subject:02d} {n_classes}c seed {seed}: '
                  f'acc={m["test_accuracy"]:.4f} kappa={m["test_kappa"]:.4f} '
                  f'val={m["val_acc_restored"]:.3f} ({m["epochs_run"]} ep, '
                  f'{dt/60:.1f} min)  ETA {eta:.0f} min', flush=True)
    collect()
