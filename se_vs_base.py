"""Ablacao pareada do bloco Squeeze-and-Excitation sobre os 9 sujeitos.

A pergunta
----------
No A01 o SE nao muda acuracia (Wilcoxon p = 1.00 e 0.81) e corta o desvio entre
sementes em 33.7x e 11.4x (Pitman-Morgan p = 0.016 e 0.030). Esse e o unico
efeito que atinge significancia, e ele tem **3 graus de liberdade**: um sujeito,
cinco sementes.

Aqui o mesmo teste roda com 9 sujeitos. Cada sujeito contribui um par
(desvio do base, desvio do SE), e o teste pareado passa a ter 9 pares.

Pareamento
----------
Base e SE compartilham sujeito, tarefa, semente, split de validacao (derivado da
semente), augmentation, batch, paciencia e banda. Desde a refatoracao de
03/10/2026 compartilham tambem o MESMO caminho de codigo: train_final.run()
recebe o construtor do modelo. Antes, se_ablation.run_se era uma copia da
funcao, e copia e como uma ablacao pareada deixa de ser pareada.

Dois niveis de teste, e eles respondem a coisas diferentes
----------------------------------------------------------
1. DENTRO de um sujeito, 5 sementes pareadas -> Pitman-Morgan, 3 g.l. E o teste
   do A01, repetido nos 9. Fraco por construcao.
2. ENTRE sujeitos, 9 pares de desvios -> Wilcoxon e teste do sinal. E o teste
   que a Fase 2 existe para viabilizar.

Uso:  python se_vs_base.py
"""

import json

import numpy as np

import paths
from train_final import ROOT

TASKS = (2, 4)
SUBJECTS = range(1, 10)


def pitman_morgan(x, y):
    """Teste de igualdade de variancias para amostras PAREADAS.

    Nao se usa teste F aqui: F assume amostras independentes, e base e SE da
    mesma semente nao sao independentes -- compartilham split e inicializacao.
    Pitman-Morgan explora isso: sob H0 (variancias iguais), a soma e a diferenca
    dos pares sao nao-correlacionadas. Testa-se a correlacao entre x+y e x-y
    como um t com n-2 graus de liberdade.
    """
    from scipy.stats import pearsonr
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float('nan'), float('nan'), n - 2
    r, _ = pearsonr(x + y, x - y)
    if abs(r) >= 1.0:
        return r, 0.0, n - 2
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(t), n - 2))
    return float(r), float(p), n - 2


def chance_threshold(n, n_classes, alpha=0.05):
    """Menor acuracia que rejeita o acaso, binomial exato unilateral."""
    from scipy.stats import binom
    ks = np.arange(n + 1)
    hit = ks[binom.sf(ks - 1, n, 1.0 / n_classes) <= alpha]
    return float(hit[0]) / n if len(hit) else 1.0


def paired_by_seed(subject, n_classes):
    """Acuracias base e SE do mesmo sujeito, alinhadas por semente.

    Devolve (seeds, base, se) so com as sementes presentes NOS DOIS bracos.
    Parear por posicao em vez de por semente seria o erro classico aqui.
    """
    def load(arch):
        return {r['config']['seed']: r for r in
                paths.load_runs('within', subject=subject, n_classes=n_classes,
                                arch=arch, augmented=True)}
    b, s = load('base'), load('se')
    seeds = sorted(set(b) & set(s))
    return (seeds,
            np.array([b[k]['metrics']['test_accuracy'] for k in seeds]),
            np.array([s[k]['metrics']['test_accuracy'] for k in seeds]),
            [b[k] for k in seeds], [s[k] for k in seeds])


def build():
    out = {}
    for n_classes in TASKS:
        rows = {}
        for subj in SUBJECTS:
            seeds, acc_b, acc_s, runs_b, runs_s = paired_by_seed(subj, n_classes)
            if len(seeds) < 2:
                continue
            n_test = runs_b[0]['metrics'].get('n_test') or len(
                runs_b[0]['predictions']['y_true'])
            thr = chance_threshold(n_test, n_classes)
            sd_b = float(acc_b.std(ddof=1))
            sd_s = float(acc_s.std(ddof=1))
            r, p, df = pitman_morgan(acc_b, acc_s)
            rows[f'A{subj:02d}'] = {
                'subject': subj, 'n_seeds': len(seeds), 'seeds': seeds,
                'base_acc': acc_b.tolist(), 'se_acc': acc_s.tolist(),
                'base_mean': float(acc_b.mean()), 'se_mean': float(acc_s.mean()),
                'delta_mean': float(acc_s.mean() - acc_b.mean()),
                'base_sd': sd_b, 'se_sd': sd_s,
                # razao de desvios: >1 significa que o SE reduziu a dispersao
                'sd_ratio_base_over_se': (sd_b / sd_s) if sd_s > 1e-12
                                         else float('inf'),
                'pitman_morgan': {'r': r, 'p_value': p, 'df': df},
                'chance_threshold': thr,
                'base_runs_below_chance': int((acc_b < thr).sum()),
                'se_runs_below_chance': int((acc_s < thr).sum()),
                'base_min': float(acc_b.min()), 'se_min': float(acc_s.min()),
                'base_max': float(acc_b.max()), 'se_max': float(acc_s.max()),
            }
        if not rows:
            continue

        # ---- teste ENTRE sujeitos: um par por sujeito ----
        from scipy.stats import wilcoxon, binomtest
        R = list(rows.values())
        d_mean = np.array([r['delta_mean'] for r in R])
        sd_b = np.array([r['base_sd'] for r in R])
        sd_s = np.array([r['se_sd'] for r in R])

        def paired_test(a, b, alt):
            """Wilcoxon mais teste do sinal. Com n=9 o sinal e mais honesto:
            Wilcoxon assume simetria da distribuicao das diferencas."""
            imp = int((a > b).sum())
            try:
                w = float(wilcoxon(a, b, alternative=alt).pvalue)
            except ValueError:
                w = 1.0
            return {'n_subjects': len(a), 'n_favouring': imp,
                    'wilcoxon_p': w,
                    'sign_test_p': float(binomtest(imp, len(a), 0.5,
                                                   alternative='greater').pvalue)}

        out[f'{n_classes}class'] = {
            'subjects': rows,
            'across_subjects': {
                'accuracy': {
                    'median_delta': float(np.median(d_mean)),
                    'mean_delta': float(d_mean.mean()),
                    **paired_test(np.array([r['se_mean'] for r in R]),
                                  np.array([r['base_mean'] for r in R]),
                                  'greater'),
                },
                'dispersion': {
                    'median_sd_base': float(np.median(sd_b)),
                    'median_sd_se': float(np.median(sd_s)),
                    'median_sd_ratio': float(np.median(sd_b / np.maximum(sd_s, 1e-12))),
                    # alternativa 'greater' em (sd_base, sd_se): testa se o SE
                    # REDUZ a dispersao, que e a hipotese do artigo
                    **paired_test(sd_b, sd_s, 'greater'),
                },
                'collapses': {
                    'base': int(sum(r['base_runs_below_chance'] for r in R)),
                    'se': int(sum(r['se_runs_below_chance'] for r in R)),
                    'n_runs_per_arm': int(sum(r['n_seeds'] for r in R)),
                },
            },
        }
    return out


def report(out):
    for n_classes in TASKS:
        key = f'{n_classes}class'
        if key not in out:
            continue
        blk = out[key]
        print(f'\n{"=" * 96}')
        print(f'SE vs BASE  -  {n_classes} classes  -  pareado por semente, '
              f'dentro de cada sujeito')
        print(f'{"=" * 96}')
        print(f'{"sub":5s} {"n":3s} {"base media":11s} {"SE media":10s} {"delta":8s} '
              f'{"dp base":9s} {"dp SE":9s} {"razao":7s} {"P-M p":8s} {"colapsos":9s}')
        print('-' * 96)
        for name in sorted(blk['subjects']):
            r = blk['subjects'][name]
            ratio = r['sd_ratio_base_over_se']
            rs = 'inf' if np.isinf(ratio) else f'{ratio:.2f}'
            print(f'{name:5s} {r["n_seeds"]:<3d} {r["base_mean"]:.4f}      '
                  f'{r["se_mean"]:.4f}    {r["delta_mean"]:+.4f}  '
                  f'{r["base_sd"]:.4f}    {r["se_sd"]:.4f}    {rs:>6s}  '
                  f'{r["pitman_morgan"]["p_value"]:.4f}   '
                  f'{r["base_runs_below_chance"]} -> {r["se_runs_below_chance"]}')
        a = blk['across_subjects']
        print('-' * 96)
        acc, disp, col = a['accuracy'], a['dispersion'], a['collapses']
        print(f'ACURACIA   : delta mediano {acc["median_delta"]:+.4f}, '
              f'SE melhor em {acc["n_favouring"]}/{acc["n_subjects"]} sujeitos, '
              f'Wilcoxon p={acc["wilcoxon_p"]:.4f}, sinal p={acc["sign_test_p"]:.4f}')
        print(f'DISPERSAO  : dp mediano {disp["median_sd_base"]:.4f} -> '
              f'{disp["median_sd_se"]:.4f} (razao mediana {disp["median_sd_ratio"]:.2f}x), '
              f'SE menor em {disp["n_favouring"]}/{disp["n_subjects"]}, '
              f'Wilcoxon p={disp["wilcoxon_p"]:.4f}, sinal p={disp["sign_test_p"]:.4f}')
        print(f'COLAPSOS   : {col["base"]} -> {col["se"]} '
              f'de {col["n_runs_per_arm"]} runs por braco '
              f'(abaixo do limiar binomial do acaso)')


if __name__ == '__main__':
    out = build()
    if not out:
        print('nenhum par base/SE completo ainda.')
        raise SystemExit
    report(out)
    (ROOT / 'results_se_9subjects.json').write_text(
        json.dumps(out, indent=2), encoding='utf-8')
    print('\nresults_se_9subjects.json salvo.')
