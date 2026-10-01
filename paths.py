"""Convencao unica de nomes e de schema para todos os runs do projeto.

Por que isto existe
-------------------
Ate a fase A01 bastava `runs/run_4c_seed0.json`: so variavam tarefa e semente.
A partir dos 9 sujeitos variam tambem sujeito, arquitetura, banda, augmentation,
sujeito-fonte, politica de congelamento e orcamento de calibracao. Nome plano
nao aguenta isso, e "de qual run veio este numero" vira impossivel de responder.

Aqui o nome do arquivo carrega a configuracao, em estilo BIDS (`chave-valor`
separados por `_`), e **todo JSON carrega um bloco `config` completo**. O nome e
conveniencia para o olho humano; o `config` dentro do arquivo e a fonte de
verdade, e e por ele que se filtra e se agrega.

Exemplos
--------
runs/within/sub-A01_task-4c_arch-base_aug-on_band-4-40_seed-0.json
runs/within/sub-A01_task-2c_arch-se_aug-on_band-4-40_seed-3.json
runs/zeroshot/src-A01_sub-A05_task-4c_arch-base_band-4-40_seed-2.json
runs/transfer/src-A01_sub-A05_task-4c_arch-base_freeze-temporal_budget-20_seed-1.json
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / 'runs'

# Ordem canonica das chaves no nome do arquivo. Chave ausente (None) e omitida,
# entao um run within-subject nao carrega `src-` nem `budget-`.
KEY_ORDER = ('src', 'sub', 'task', 'arch', 'aug', 'band', 'freeze', 'budget', 'seed')

EXPERIMENTS = ('within', 'zeroshot', 'transfer')


def _fmt_band(band):
    """(4.0, 40.0) -> '4-40'. Inteiro quando possivel, para o nome ficar legivel."""
    def one(v):
        return str(int(v)) if float(v).is_integer() else str(v)
    return f'{one(band[0])}-{one(band[1])}'


def run_key(*, subject, n_classes, arch='base', seed, augmented=True,
            band=(4.0, 40.0), source_subject=None, freeze=None, budget=None,
            experiment='within'):
    """Monta o nome canonico (sem extensao) a partir da configuracao.

    subject / source_subject : int (1..9)
    n_classes                : 2 ou 4
    arch                     : 'base' | 'se' | 'msem1' | 'msem3' | ...
    augmented                : bool
    band                     : (l_freq, h_freq)
    freeze                   : None | 'temporal' | 'temporal+spatial' | 'allbutclf'
    budget                   : None | int, trials do alvo liberados para fine-tuning
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f'experimento desconhecido: {experiment}')

    parts = {
        'src':    f'A{source_subject:02d}' if source_subject is not None else None,
        'sub':    f'A{subject:02d}',
        'task':   f'{n_classes}c',
        'arch':   arch,
        # zero-shot nao treina, entao augmentation nao se aplica e sai do nome
        'aug':    None if experiment == 'zeroshot' else ('on' if augmented else 'off'),
        'band':   _fmt_band(band),
        'freeze': freeze,
        'budget': None if budget is None else str(budget),
        'seed':   str(seed),
    }
    return '_'.join(f'{k}-{parts[k]}' for k in KEY_ORDER if parts[k] is not None)


def run_path(**kw):
    """Caminho completo do JSON. Aceita os mesmos argumentos de run_key()."""
    experiment = kw.get('experiment', 'within')
    return RUNS_DIR / experiment / f'{run_key(**kw)}.json'


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _versions():
    out = {}
    for name, mod in (('tensorflow', 'tensorflow'), ('keras', 'keras'),
                      ('mne', 'mne'), ('numpy', 'numpy')):
        try:
            out[name] = __import__(mod).__version__
        except Exception:
            out[name] = None
    return out


def make_config(*, subject, n_classes, arch='base', seed, augmented=True,
                band=(4.0, 40.0), source_subject=None, freeze=None, budget=None,
                experiment='within', **extra):
    """Bloco `config` que vai dentro de todo JSON de run.

    Inclui commit e versoes de biblioteca porque este projeto ja foi mordido por
    mudanca de ambiente antes (ver README, secao de reprodutibilidade).
    """
    cfg = {
        'experiment': experiment,
        'subject': subject,
        'source_subject': source_subject,
        'n_classes': n_classes,
        'arch': arch,
        'augmented': augmented,
        'band': list(band),
        'freeze': freeze,
        'budget': budget,
        'seed': seed,
        'git_commit': _git_commit(),
        'versions': _versions(),
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    cfg.update(extra)
    return cfg


def save_run(config, metrics, predictions):
    """Grava um run no schema canonico e devolve o caminho.

    predictions DEVE conter y_true, y_pred e proba. O `proba` nao e opcional:
    sem ele nao da para refazer soft vote, calibracao ou ECE sem recarregar os
    modelos, o que ja custou retrabalho duas vezes neste projeto.
    """
    for k in ('y_true', 'y_pred', 'proba'):
        if k not in predictions:
            raise ValueError(f'predictions precisa conter {k!r}')

    p = run_path(**{k: config[k] for k in
                    ('subject', 'n_classes', 'arch', 'seed', 'augmented',
                     'source_subject', 'freeze', 'budget', 'experiment')},
                 band=tuple(config['band']))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {'config': config, 'metrics': metrics, 'predictions': predictions},
        indent=2), encoding='utf-8')
    return p


def load_runs(experiment=None, include_excluded=False, **filters):
    """Carrega todos os runs que casam com os filtros, lendo o bloco `config`.

    O filtro e pelo conteudo do JSON, nunca pelo nome do arquivo.

        load_runs('within', subject=1, n_classes=4, arch='base')
        load_runs('zeroshot', source_subject=1, n_classes=4)
    """
    globs = (RUNS_DIR / experiment).glob('*.json') if experiment \
        else RUNS_DIR.glob('*/*.json')
    out = []
    for f in sorted(globs):
        r = json.loads(f.read_text(encoding='utf-8'))
        if 'config' not in r:
            continue          # run anterior ao schema; ignorado de proposito
        cfg = r['config']
        if cfg.get('exclude_from_aggregation') and not include_excluded:
            continue
        if all(cfg.get(k) == v for k, v in filters.items()):
            r['_path'] = str(f.relative_to(ROOT))
            out.append(r)
    return out


def index(experiment=None):
    """Tabela de uma linha por run, para responder 'o que eu ja rodei?'."""
    rows = []
    for r in load_runs(experiment):
        c, m = r['config'], r['metrics']
        rows.append({
            'path': r['_path'],
            'experiment': c['experiment'], 'src': c.get('source_subject'),
            'sub': c['subject'], 'task': f"{c['n_classes']}c", 'arch': c['arch'],
            'aug': c['augmented'], 'band': '-'.join(str(int(b)) for b in c['band']),
            'freeze': c.get('freeze'), 'budget': c.get('budget'), 'seed': c['seed'],
            'acc': round(m['test_accuracy'], 4), 'kappa': round(m['test_kappa'], 4),
        })
    return rows


if __name__ == '__main__':
    import sys
    exp = sys.argv[1] if len(sys.argv) > 1 else None
    rows = index(exp)
    if not rows:
        print('nenhum run encontrado.')
        raise SystemExit
    cols = ['experiment', 'src', 'sub', 'task', 'arch', 'aug', 'band',
            'freeze', 'budget', 'seed', 'acc', 'kappa']
    w = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    print('  '.join(c.ljust(w[c]) for c in cols))
    print('  '.join('-' * w[c] for c in cols))
    for r in sorted(rows, key=lambda r: (r['experiment'], str(r['src']), r['sub'],
                                         r['task'], r['arch'], r['seed'])):
        print('  '.join(str(r[c]).ljust(w[c]) for c in cols))
    print(f'\n{len(rows)} runs.')


# Campos do config que o codigo legado espera encontrar no nivel de cima.
_FLAT_FROM_CONFIG = ('seed', 'n_classes', 'augmented', 'subject', 'arch')


def load_flat(experiment='within', **filters):
    """Runs com `config` e `metrics` achatados num dict unico.

    Existe para o codigo escrito antes do schema (train_final.collect,
    ensemble.load_seed_models, se_ablation.collect), que acessa
    r['test_accuracy'] direto. Codigo novo deve usar load_runs().
    """
    out = []
    for r in load_runs(experiment, **filters):
        d = dict(r['metrics'])
        d.update({k: r['config'][k] for k in _FLAT_FROM_CONFIG if k in r['config']})
        d['predictions'] = r['predictions']
        d['_path'] = r['_path']
        out.append(d)
    return sorted(out, key=lambda d: (d.get('n_classes', 0), d.get('seed', 0)))
