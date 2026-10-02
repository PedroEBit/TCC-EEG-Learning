"""Transferencia NORMALIZADA pelo teto do proprio sujeito.

Por que este arquivo existe
---------------------------
A tabela de zero-shot em acuracia absoluta e enganosa, e enganou. Ela diz que
A03 e A08 "transferem" e que A05 "nao transfere". Mas A03 e A08 sao os sujeitos
mais faceis do dataset (within 0.965 e 0.979 no 2-class), e A05 e dos mais
dificeis. Acuracia absoluta de transferencia confunde duas coisas:

    (1) quanto o modelo-fonte transfere para aquela pessoa, e
    (2) quao decodificavel aquela pessoa e para qualquer modelo.

A razao transferencia / teto within separa as duas. Sem ela, "a transferencia
funcionou" e uma afirmacao sobre a sorte do sujeito.

Dois baselines, e o padrao e o justo
------------------------------------
O soft vote do zero-shot e um ensemble de 5 modelos; o within aqui tem n=1 ou
n=5 sementes, ou seja, UM modelo por semente. Comparar ensemble contra modelo
unico infla o numerador. Entao:

  ratio_single   = media dos membros zero-shot / media within   <- PRINCIPAL,
                   modelo unico contra modelo unico
  ratio_ensemble = soft vote zero-shot / media within           <- secundario,
                   ensemble contra modelo unico, infla

A01 fica na tabela mas a razao dele NAO e transferencia: ali fonte e alvo sao a
mesma pessoa, entao a razao mede o ganho do ensemble, nao transporte. Marcado.

Uso:  python transfer_ratio.py
"""

import json

import numpy as np

from train_final import ROOT

TASKS = (2, 4)
DEV_SUBJECTS = (1,)


def load():
    zs = json.loads((ROOT / 'results_zeroshot.json').read_text(encoding='utf-8'))
    wi = json.loads((ROOT / 'results_within9.json').read_text(encoding='utf-8'))
    return zs, wi


def build():
    zs, wi = load()
    out = {}
    for n_classes in TASKS:
        key = f'{n_classes}class'
        if key not in zs or key not in wi:
            continue
        rows = {}
        for name, z in zs[key]['subjects'].items():
            w = wi[key]['subjects'].get(name)
            if w is None:
                continue
            rows[name] = {
                'subject': z['subject'],
                'role': z['role'],
                'is_transfer': z['subject'] not in DEV_SUBJECTS,
                'within_n_seeds': w['n_seeds'],
                'within_mean': w['accuracy_mean'],
                'within_max': w['accuracy_max'],
                'zeroshot_member_mean': z['member_mean'],
                'zeroshot_soft_vote': z['soft_vote']['accuracy'],
                'chance_level': z['chance_level'],
                # Razao crua. Nao corrige pelo acaso de proposito: ver
                # ratio_above_chance abaixo, que corrige, e dizer as duas e mais
                # honesto que escolher uma.
                'ratio_single': z['member_mean'] / w['accuracy_mean'],
                'ratio_ensemble': z['soft_vote']['accuracy'] / w['accuracy_mean'],
            }
            # Fracao do ganho ACIMA DO ACASO que a transferencia reteve. Uma
            # razao crua de 0.81 parece boa em 4 classes, mas se within=0.347 e
            # zero-shot=0.258 com acaso 0.25, o que foi retido do que havia para
            # reter e so (0.258-0.25)/(0.347-0.25) = 8%.
            c = z['chance_level']
            head = w['accuracy_mean'] - c
            rows[name]['ratio_above_chance'] = (
                (z['member_mean'] - c) / head if head > 1e-9 else float('nan'))

            # O denominador `head` e o espaco que havia para reter. Quando o
            # within do proprio sujeito nao esta acima do acaso, head ~ 0 e a
            # razao explode: A02 2-class deu 590% so porque within=0.514 contra
            # acaso 0.50, head=0.014. Isso nao e "transferiu 6x melhor", e
            # divisao por quase zero. Marcado e excluido das medianas.
            from scipy.stats import binomtest
            k_within = int(round(w['accuracy_mean'] * z['n_test']))
            bt = binomtest(k_within, z['n_test'], c, alternative='greater')
            rows[name]['within_above_chance_p'] = float(bt.pvalue)
            rows[name]['ratio_unstable'] = bool(bt.pvalue >= 0.05)
        out[key] = rows
    return out


def report(out):
    for n_classes in TASKS:
        key = f'{n_classes}class'
        if key not in out:
            continue
        rows = out[key]
        print(f'\n{"=" * 94}')
        print(f'TRANSFERENCIA NORMALIZADA  {n_classes} classes   '
              f'(fonte A01, acaso {1/n_classes:.2f})')
        print(f'{"=" * 94}')
        print(f'{"alvo":6s} {"n":3s} {"within":8s} {"zs-membro":10s} {"zs-soft":9s} '
              f'{"razao":7s} {"razao-ens":10s} {"retido>acaso":12s}')
        print('-' * 94)
        trans = []
        for name in sorted(rows):
            r = rows[name]
            tag = '' if r['is_transfer'] else '  <- dev, mesma pessoa: nao e transferencia'
            if r.get('ratio_unstable') and r['is_transfer']:
                tag = '  ! within no acaso, razao instavel'
            print(f'{name:6s} {r["within_n_seeds"]:<3d} {r["within_mean"]:.4f}   '
                  f'{r["zeroshot_member_mean"]:.4f}     '
                  f'{r["zeroshot_soft_vote"]:.4f}    '
                  f'{r["ratio_single"]:.3f}   {r["ratio_ensemble"]:.3f}      '
                  f'{r["ratio_above_chance"]*100:6.1f}%{tag}')
            if r['is_transfer']:
                trans.append(r)
        rs = np.array([r['ratio_single'] for r in trans])
        print('-' * 94)
        print(f'{len(trans)} sujeitos de transferencia: razao mediana {np.median(rs):.3f} '
              f'[{rs.min():.3f}, {rs.max():.3f}]')

        # "Retido acima do acaso" so faz sentido onde o teto do sujeito ESTA
        # acima do acaso. Onde nao esta, o denominador e ruido.
        ok = [r for r in trans if not r['ratio_unstable']]
        skipped = [f'A{r["subject"]:02d}' for r in trans if r['ratio_unstable']]
        ra = np.array([r['ratio_above_chance'] for r in ok])
        print(f'  retido acima do acaso ({len(ok)} sujeitos): '
              f'mediana {np.median(ra)*100:.1f}% '
              f'[{ra.min()*100:.1f}%, {ra.max()*100:.1f}%]')
        if skipped:
            print(f'  ! excluidos (within nao acima do acaso, razao instavel): '
                  f'{", ".join(skipped)}')
        best = max(ok, key=lambda r: r['ratio_above_chance'])
        worst = min(ok, key=lambda r: r['ratio_above_chance'])
        print(f'  melhor A{best["subject"]:02d} ({best["ratio_above_chance"]*100:.1f}%), '
              f'pior A{worst["subject"]:02d} ({worst["ratio_above_chance"]*100:.1f}%)')


if __name__ == '__main__':
    out = build()
    report(out)
    (ROOT / 'results_transfer_ratio.json').write_text(
        json.dumps(out, indent=2), encoding='utf-8')
    print('\nresults_transfer_ratio.json salvo.')
