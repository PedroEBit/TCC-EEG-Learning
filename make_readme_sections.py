"""Gera as secoes do README que dependem de numeros, a partir dos JSON de resultado.

Por que isto existe: os numeros mudam a cada semente que termina. Escrever a
tabela a mao significa transcrever 36 valores toda vez, e transcricao manual de
numero e exatamente como um paper ganha um erro que ninguem acha depois.

As secoes sao delimitadas por marcadores HTML no README e reescritas no lugar,
entao rodar de novo e idempotente.

Uso:  python make_readme_sections.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
SUBJECTS = [f'A{i:02d}' for i in range(1, 10)]


def _seeds_cell(v):
    """Valores individuais, nao media +- dp: com n<=3 o dp e o gap rescalado."""
    xs = [p['accuracy'] for p in v['per_seed']]
    if len(xs) >= 5:
        return (f"{v['accuracy_mean']:.3f} "
                f"[{v['accuracy_min']:.3f}–{v['accuracy_max']:.3f}]")
    return ', '.join(f'{x:.3f}' for x in xs)


def within_table(wi):
    n_by_task = {}
    rows = ['| Subject | 2-class, per seed | 4-class, per seed |',
            '|---|---|---|']
    for name in SUBJECTS:
        cells = []
        for nc in (2, 4):
            v = wi.get(f'{nc}class', {}).get('subjects', {}).get(name)
            cells.append(_seeds_cell(v) if v else '—')
            if v:
                n_by_task.setdefault(nc, []).append(v['n_seeds'])
        tag = ' *(dev)*' if name == 'A01' else ''
        rows.append(f'| {name}{tag} | {cells[0]} | {cells[1]} |')
    means = []
    for nc in (2, 4):
        a = wi.get(f'{nc}class', {}).get('across_subjects')
        means.append(f"{a['mean_of_means']:.3f}" if a else '—')
    rows.append(f'| **mean of the nine** | **{means[0]}** | **{means[1]}** |')
    ns = {nc: (min(v), max(v)) for nc, v in n_by_task.items()}
    note = '; '.join(
        f'{nc}-class n = ' + (f'{lo}' if lo == hi else f'{lo}–{hi}')
        for nc, (lo, hi) in sorted(ns.items()))
    rows.append('')
    rows.append(f'Seeds per cell: {note} (A01 carries five from the '
                f'single-subject phase). Values are individual runs.')
    return '\n'.join(rows)


def ratio_table(tr):
    rows = ['| Target | Within ceiling | Zero-shot | Ratio | Retained above chance |',
            '|---|---|---|---|---|']
    for nc in (2, 4):
        blk = tr.get(f'{nc}class')
        if not blk:
            continue
        rows.append(f'| **{nc}-class** | | | | |')
        for name in SUBJECTS:
            r = blk.get(name)
            if not r:
                continue
            if not r['is_transfer']:
                rows.append(f'| {name} *(dev, same person — not transfer)* | '
                            f'{r["within_mean"]:.3f} | '
                            f'{r["zeroshot_member_mean"]:.3f} | 1.00 | — |')
                continue
            flag = ' ⚠' if r.get('ratio_unstable') else ''
            rows.append(f'| {name} | {r["within_mean"]:.3f} | '
                        f'{r["zeroshot_member_mean"]:.3f} | '
                        f'{r["ratio_single"]:.3f} | '
                        f'{r["ratio_above_chance"]*100:.1f}%{flag} |')
    return '\n'.join(rows)


def medians(tr):
    import numpy as np
    out = {}
    for nc in (2, 4):
        blk = tr.get(f'{nc}class')
        if not blk:
            continue
        t = [r for r in blk.values() if r['is_transfer']]
        ok = [r for r in t if not r.get('ratio_unstable')]
        out[nc] = {
            'ratio': float(np.median([r['ratio_single'] for r in t])),
            'ratio_lo': float(min(r['ratio_single'] for r in t)),
            'ratio_hi': float(max(r['ratio_single'] for r in t)),
            'retained': float(np.median([r['ratio_above_chance'] for r in ok])),
            'ret_lo': float(min(r['ratio_above_chance'] for r in ok)),
            'ret_hi': float(max(r['ratio_above_chance'] for r in ok)),
        }
    return out


def splice(text, marker, body):
    a = f'<!-- AUTOGEN:{marker} -->'
    b = f'<!-- /AUTOGEN:{marker} -->'
    i, j = text.index(a), text.index(b)
    return text[:i] + a + '\n' + body + '\n' + text[j:]


if __name__ == '__main__':
    wi = json.loads((ROOT / 'results_within9.json').read_text(encoding='utf-8'))
    tr = json.loads((ROOT / 'results_transfer_ratio.json').read_text(encoding='utf-8'))
    med = medians(tr)

    p = ROOT / 'README.md'
    s = p.read_text(encoding='utf-8')
    s = splice(s, 'within-table', within_table(wi))
    s = splice(s, 'ratio-table', ratio_table(tr))
    s = splice(s, 'ratio-medians', '\n'.join(
        f'- **{nc}-class:** median ratio **{m["ratio"]:.2f}** '
        f'(range {m["ratio_lo"]:.2f}–{m["ratio_hi"]:.2f}); median retained above chance '
        f'**{m["retained"]*100:.1f}%** '
        f'(range {m["ret_lo"]*100:.1f}% to {m["ret_hi"]*100:.1f}%).'
        for nc, m in sorted(med.items())))
    p.write_text(s, encoding='utf-8')
    print('README.md: tabelas regeneradas a partir dos JSON.')
    for nc, m in sorted(med.items()):
        print(f'  {nc}-class: razao mediana {m["ratio"]:.3f}, '
              f'retido mediana {m["retained"]*100:.1f}%')
