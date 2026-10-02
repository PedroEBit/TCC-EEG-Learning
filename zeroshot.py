"""Transferencia zero-shot: modelos treinados em A01, avaliados nos 9 sujeitos.

Nenhum treino acontece aqui. Os modelos de A01 ja estao em models/, salvos pelo
train_final.py. Este script so carrega, prediz e mede.

Por que este experimento vem primeiro: ele e o piso. Antes de perguntar "o
congelamento de camadas ajuda?" e preciso saber quanto o modelo transfere sem
adaptacao nenhuma. Sem esse numero, nenhum resultado de fine-tuning e
interpretavel.

Protocolo
---------
Fonte  : A01 (5 sementes por tarefa, arquitetura base, com augmentation)
Alvo   : A0xE, x = 1..9  -- SEMPRE a sessao E, que e o teste trancado
Controle: x = 1 tem que reproduzir os runs within de A01 SEMENTE A SEMENTE, em
          igualdade exata. Ver check_control().

A sessao A0xT NAO e tocada. Ela e o dado de desenvolvimento do sujeito x, e vai
ser usada no fine-tuning da Fase 3. Manter o A0xE fixo em todas as condicoes e o
que torna zero-shot, fine-tuned e congelado comparaveis trial a trial (McNemar).

Decisoes registradas (01/10/2026)
---------------------------------
1. Split de sujeitos: desenvolvimento = A01, teste = A02..A09.
   A01 ja esta irreversivelmente queimado -- o projeto inteiro foi feito nele --
   entao usa-lo como unico sujeito de desenvolvimento nao gasta nada de novo, e
   deixa 8 sujeitos limpos para o teste pareado das Fases 3 e 4.
   Este script roda nos 9 porque e DESCRITIVO: ele nao escolhe nada. Toda decisao
   de projeto (politica de congelamento, orcamento, arquitetura) se toma olhando
   A01 e validacao, nunca esta tabela.
2. align=None: nenhum alinhamento de dominio. Vale lembrar que o z-score do
   prepare() divide por UM escalar por epoca (std sobre canais e tempo juntos),
   logo corrige ganho global e NAO alinha a covariancia entre canais -- que e a
   unica coisa que o filtro espacial consome. O braco Euclidean Alignment da
   Fase 3 entra como align='ea-T' / 'ea-E', pareado contra estes runs.
3. augmented=True no config: aqui o campo descreve a PROCEDENCIA do modelo-fonte
   (A01 foi treinado com augmentation), nao este run, que nao treina. O run_key
   omite `aug-` do nome quando experiment='zeroshot', de proposito.

Referencias para ler a tabela
-----------------------------
Limite do acaso (binomial, 95% unilateral): 2-class n=144 -> 0.569;
4-class n=288 -> 0.292. Com Bonferroni para 18 testes: 0.615 e 0.321.
Desacordo entre dois modelos que chutam uniforme: 0.50 (2c) e 0.75 (4c).
Dentro do A01 o desacordo medido e 0.231 e 0.242.

Uso:  python zeroshot.py            # roda o que falta e agrega
      python zeroshot.py --collect  # so agrega o que ja existe
"""

import json
import sys

import numpy as np

import paths
from train_final import load_subject, prepare, ROOT, N_CHANNELS
from ensemble import (
    load_seed_models, metrics, soft_vote, hard_vote, disagreement_matrix, mcnemar,
)

SOURCE_SUBJECT = 1
DEV_SUBJECTS = (1,)                       # onde se decide
TEST_SUBJECTS = tuple(range(2, 10))       # trancados; olhados uma vez
TARGETS = range(1, 10)
TASKS = (2, 4)
BAND = (4.0, 40.0)


def evaluate_member(model, Xp, y):
    """Roda um modelo num conjunto ja preparado e devolve (proba, metricas).

    `metrics` vem do ensemble.py e devolve accuracy, kappa, as duas confiancas
    medias, e y_pred. Duas armadilhas, as duas tratadas no run_target: o y_pred
    dele e ndarray, que nao serializa em JSON; e as chaves sao 'accuracy'/'kappa'
    enquanto paths.index() le 'test_accuracy'/'test_kappa'.
    """
    proba = model.predict(Xp, verbose=0)
    return proba, metrics(y, proba)


def run_target(n_classes, target, members, X4, y4):
    """Avalia os modelos-fonte em A0{target}E e grava um run por semente.

    Recebe o 4-class ja carregado (X4, y4) e deriva o 2-class por mascara, em vez
    de chamar load_subject de novo: reler o GDF e refazer o filtro IIR custa ~15s,
    e seriam 18 leituras em vez de 9. A mascara isin([0,1]) e exatamente a que o
    load_subject aplica internamente.

    Resumivel: pula qualquer semente cujo JSON ja exista.
    Devolve a lista de probas na ordem das sementes -- mas o collect le do disco,
    que e a fonte de verdade e sobrevive a interrupcao.
    """
    if n_classes == 2:
        mask = np.isin(y4, [0, 1])
        X, y = X4[mask], y4[mask]
    else:
        X, y = X4, y4

    Xp = prepare(X)
    # Sem numero magico: o shape esperado sai do proprio modelo salvo. Se algum
    # GDF vier com sfreq diferente, quebra aqui e nao 40 linhas adiante.
    expected = tuple(members[0]['model'].input_shape[1:])
    assert Xp.shape[1:] == expected, (
        f'A0{target}E tem shape {Xp.shape[1:]}, modelo espera {expected}')

    probas = []
    for mem in members:
        # `seed` identifica o modelo-FONTE. `subject` e o alvo, `source_subject`
        # e de onde o modelo veio: e o par (src, sub) que define o run.
        cfg_kw = dict(experiment='zeroshot', subject=target,
                      source_subject=SOURCE_SUBJECT, n_classes=n_classes,
                      arch='base', augmented=True, band=BAND, align=None,
                      freeze=None, budget=None, seed=mem['seed'])

        out = paths.run_path(**cfg_kw)
        if out.exists():
            r = json.loads(out.read_text(encoding='utf-8'))
            probas.append(np.array(r['predictions']['proba'], dtype=np.float32))
            continue

        proba, m = evaluate_member(mem['model'], Xp, y)
        paths.save_run(
            paths.make_config(**cfg_kw, predictions_complete=True),
            {
                'test_accuracy': m['accuracy'],
                'test_kappa': m['kappa'],
                'mean_confidence_correct': m['mean_confidence_correct'],
                'mean_confidence_incorrect': m['mean_confidence_incorrect'],
                'n_test': int(len(y)),
            },
            {
                'y_true': y.tolist(),
                'y_pred': m['y_pred'].tolist(),
                # 6 casas mantem o arquivo pequeno e nao muda argmax nenhum
                'proba': np.round(proba, 6).tolist(),
            },
        )
        probas.append(proba)

    acc = np.array([(p.argmax(1) == y).mean() for p in probas])
    print(f'  A0{target}E {n_classes}c: por semente '
          f'[{", ".join(f"{a:.4f}" for a in acc)}]  media {acc.mean():.4f}',
          flush=True)
    return probas


def check_control(n_classes):
    """Alvo = A01 e o controle. Igualdade EXATA, semente a semente.

    Nao e "aproximadamente igual": e o mesmo .keras, no mesmo A01E, pela mesma
    load_subject e o mesmo prepare, em CPU, deterministico. Qualquer diferenca
    significa que alguma coisa mudou no carregamento desde que os runs within
    foram gravados, e nesse caso a tabela toda esta suspeita.

    Semente a semente, e nao na media, porque media casa por compensacao: dois
    erros de sinal oposto passam batido. Foi por isso que a constante CONTROL
    (que guardava so a media) saiu deste arquivo.
    """
    within = {r['config']['seed']: r['metrics']['test_accuracy']
              for r in paths.load_runs('within', subject=SOURCE_SUBJECT,
                                       n_classes=n_classes, arch='base',
                                       augmented=True)}
    zs = {r['config']['seed']: r['metrics']['test_accuracy']
          for r in paths.load_runs('zeroshot', subject=SOURCE_SUBJECT,
                                   source_subject=SOURCE_SUBJECT,
                                   n_classes=n_classes)}
    assert within, f'nenhum run within de A01 {n_classes}c para comparar'

    bad = {s: (within[s], zs.get(s)) for s in sorted(within)
           if zs.get(s) is None or abs(zs[s] - within[s]) > 1e-12}
    if bad:
        raise AssertionError(
            f'controle A01 {n_classes}c FALHOU (within vs zeroshot): ' +
            '; '.join(f'seed {s}: {w:.10f} vs {z}' for s, (w, z) in bad.items()))
    return within, zs


ALPHA = 0.05
# Testes de hipotese efetivamente reportados: 8 sujeitos de teste x 2 tarefas.
# A01 nao conta: ele e o sujeito de desenvolvimento e entra como CONTROLE, nao
# como hipotese. O p bruto tambem e gravado, para quem quiser outra correcao.
N_TESTS = len(TEST_SUBJECTS) * len(TASKS)


def critical_accuracy(n, n_classes, alpha=ALPHA, n_tests=1):
    """Menor acuracia que rejeita o acaso num binomial unilateral EXATO.

    Nao usa aproximacao normal: com n=144 e p=0.5 a normal erra o limiar por
    quase um trial inteiro, e o limiar e justamente onde varios sujeitos caem.
    Devolve k_crit/n, onde k_crit e o menor numero de acertos com
    P(X >= k_crit) <= alpha/n_tests sob chute uniforme.
    """
    from scipy.stats import binom
    ks = np.arange(n + 1)
    p_ge = binom.sf(ks - 1, n, 1.0 / n_classes)      # P(X >= k)
    hit = ks[p_ge <= alpha / n_tests]
    return float(hit[0]) / n if len(hit) else 1.0


def select_reference_member(n_classes):
    """Membro de referencia do McNemar, escolhido SO na validacao de A01.

    REGRA METODOLOGICA: o ensemble.py compara o ensemble contra o membro
    MEDIANO por acuracia de teste. Mediana é melhor que maximo, mas ainda e
    uma escolha que olha o teste. Aqui a referencia e o membro de maior
    val_acc_restored no split de validacao de A01T -- decidido antes de qualquer
    predicao em sujeito alheio, e portanto valido num paper.

    Empate resolvido pela menor semente, para ser deterministico.
    """
    runs = paths.load_runs('within', subject=SOURCE_SUBJECT, n_classes=n_classes,
                           arch='base', augmented=True)
    best = max(runs, key=lambda r: (r['metrics']['val_acc_restored'],
                                    -r['config']['seed']))
    return best['config']['seed'], float(best['metrics']['val_acc_restored'])


def _clean(m):
    """metrics() devolve y_pred como ndarray, que json.dumps nao serializa."""
    return {k: float(v) for k, v in m.items() if k != 'y_pred'}


def summarize_target(n_classes, target, ref_seed):
    """Tudo o que se sabe de UM alvo numa tarefa. Devolve o dict do sujeito.

    Esta funcao existe separada de collect() de proposito: funcao que nao se
    consegue chamar com um caso so e funcao grande demais. Da para inspecionar
    um sujeito no REPL com summarize_target(2, 3, 1).
    """
    from scipy.stats import binomtest

    runs = paths.load_runs('zeroshot', n_classes=n_classes, subject=target,
                           source_subject=SOURCE_SUBJECT)
    assert runs, f'nenhum run zero-shot para A0{target} {n_classes}c'

    # load_runs ordena por NOME de arquivo. Com seed-0..4 da certo por acidente;
    # com seed-10 nao daria. Ordena pelo config.
    runs.sort(key=lambda r: r['config']['seed'])

    probas = [np.array(r['predictions']['proba']) for r in runs]
    y = np.array(runs[0]['predictions']['y_true'])
    seeds = [r['config']['seed'] for r in runs]

    # y_true tem que ser identico nas 5 sementes: e o MESMO A0xE, so o modelo
    # muda. Se divergir, alguma epoca foi descartada numa das leituras e o
    # alinhamento rotulo/epoca quebrou em silencio. Sem este assert, soft_vote
    # mediaria probabilidades de trials diferentes e devolveria um numero
    # plausivel e errado.
    for r in runs[1:]:
        assert np.array_equal(np.array(r['predictions']['y_true']), y), (
            f'y_true divergente em A0{target} {n_classes}c, '
            f'seed {r["config"]["seed"]}')

    n = len(y)

    # --- membros individuais ---
    per_seed = [metrics(y, p) for p in probas]
    acc = np.array([m['accuracy'] for m in per_seed])

    # --- os dois ensembles ---
    m_soft = metrics(y, soft_vote(probas))
    m_hard = metrics(y, hard_vote(probas, n_classes))

    # --- diversidade ---
    # Triangulo superior sem a diagonal (k=1): a diagonal e zero por construcao
    # e puxaria a media para baixo.
    D = disagreement_matrix(probas)
    disagreement = float(D[np.triu_indices(len(probas), 1)].mean())

    # --- acima do acaso? ---
    # Testado no SOFT VOTE, que e o resultado reportado. Nao na media das 5
    # acuracias: media de acuracias nao tem distribuicao binomial.
    k_correct = int((m_soft['y_pred'] == y).sum())
    bt = binomtest(k_correct, n, 1.0 / n_classes, alternative='greater')

    # --- McNemar pareado, mesmos trials ---
    ref_idx = seeds.index(ref_seed)
    median_idx = int(np.argsort(acc)[len(acc) // 2])
    out = {
        'subject': target,
        'role': 'dev' if target in DEV_SUBJECTS else 'test',
        'n_test': n,
        'chance_level': 1.0 / n_classes,
        'critical_accuracy_uncorrected': critical_accuracy(n, n_classes),
        'critical_accuracy_bonferroni': critical_accuracy(n, n_classes,
                                                          n_tests=N_TESTS),
        'seeds': seeds,
        'per_seed': [dict(seed=s, **_clean(m)) for s, m in zip(seeds, per_seed)],
        'member_mean': float(acc.mean()),
        'member_std': float(acc.std(ddof=1)),
        'member_min': float(acc.min()),
        'member_median': float(np.median(acc)),
        'member_max': float(acc.max()),
        'soft_vote': _clean(m_soft),
        'hard_vote': _clean(m_hard),
        'ensemble_gain_over_member_mean': float(m_soft['accuracy'] - acc.mean()),
        'mean_pairwise_disagreement': disagreement,
        # Teto do desacordo sob chute independente e uniforme: 1 - 1/k.
        # Desacordo subindo PARA esse teto com acuracia no acaso nao e
        # diversidade util, e cinco modelos chutando.
        'chance_disagreement': 1.0 - 1.0 / n_classes,
        'above_chance': {
            'n_correct': k_correct, 'n_test': n,
            'p_value': float(bt.pvalue),
            'significant_uncorrected': bool(bt.pvalue < ALPHA),
            'significant_bonferroni': bool(bt.pvalue < ALPHA / N_TESTS),
            'n_tests_corrected': N_TESTS,
        },
    }

    for label, idx, how in (('val_selected', ref_idx, 'maior val_acc_restored em A01T'),
                            ('median', median_idx, 'mediana da acuracia NESTE alvo (olha teste)')):
        n01, n10, p = mcnemar(y, probas[idx].argmax(1), m_soft['y_pred'])
        out[f'mcnemar_soft_vs_{label}_member'] = {
            'member_seed': seeds[idx],
            'member_accuracy': float(acc[idx]),
            'selection': how,
            'member_only_correct': n10,
            'ensemble_only_correct': n01,
            'p_value': p,
        }
    return out


def collect():
    """Agrega os runs zero-shot em results_zeroshot.json.

    Reporta POR SUJEITO, nunca so a media: a variancia entre sujeitos neste
    dataset e enorme e uma media esconde tudo. A01 aparece separado, porque e o
    sujeito de desenvolvimento e ali o modelo esta em casa -- e o controle, nao
    um resultado de transferencia.

    Limiares: o acaso nao e 0.50 e 0.25. Com n finito o limiar binomial exato
    unilateral fica perto de 0.57 (2c, n=144) e 0.29 (4c, n=288); com Bonferroni
    sobre os 16 testes reportados, perto de 0.61 e 0.32. Os valores exatos sao
    calculados por critical_accuracy() e gravados em cada sujeito.
    """
    from scipy.stats import binomtest, wilcoxon

    out = {}
    for n_classes in TASKS:
        task_key = f'{n_classes}class'
        ref_seed, ref_val = select_reference_member(n_classes)

        subjects = {f'A{t:02d}': summarize_target(n_classes, t, ref_seed)
                    for t in TARGETS}

        # --- agregacao SO sobre os 8 sujeitos de teste ---
        # A01 fica fora: ele e dev e esta em casa, incluir inflaria tudo.
        rows = [subjects[f'A{t:02d}'] for t in TEST_SUBJECTS]
        soft_acc = np.array([r['soft_vote']['accuracy'] for r in rows])
        mem_mean = np.array([r['member_mean'] for r in rows])

        # Pareado por SUJEITO: cada sujeito contribui um par (ensemble, membros).
        # Wilcoxon para locacao; e tambem o teste do sinal, que com n=8 e mais
        # honesto porque nao assume simetria da distribuicao das diferencas.
        def paired(a, b):
            """Teste pareado por sujeito entre dois vetores de acuracia."""
            imp = int((a > b).sum())
            try:
                w = float(wilcoxon(a, b, alternative='greater').pvalue)
            except ValueError:             # todas as diferencas nulas
                w = 1.0
            return {'n_subjects_improved': imp, 'wilcoxon_p': w,
                    'sign_test_p': float(binomtest(imp, len(a), 0.5,
                                                   alternative='greater').pvalue),
                    'mean_gain': float((a - b).mean())}

        # O baseline MUDA a conclusao, entao os dois sao reportados.
        # Contra a media dos membros, o ensemble parece forte. Mas na pratica
        # ninguem usa "a media dos membros": usa-se UM modelo, e escolhido na
        # validacao. Contra esse, o ensemble tem que provar o seu valor de novo.
        val_acc = np.array([r['mcnemar_soft_vs_val_selected_member']
                            ['member_accuracy'] for r in rows])
        vs_mean = paired(soft_acc, mem_mean)
        vs_val = paired(soft_acc, val_acc)
        n_improved, w_p, sign_p = (vs_mean['n_subjects_improved'],
                                   vs_mean['wilcoxon_p'], vs_mean['sign_test_p'])

        out[task_key] = {
            'reference_member': {
                'seed': ref_seed, 'val_acc_restored': ref_val,
                'selected_on': 'val_acc_restored em A01T (nunca no teste)'},
            'subjects': subjects,
            'test_subject_summary': {
                'subjects': list(TEST_SUBJECTS),
                'n_subjects': len(rows),
                'soft_vote_mean': float(soft_acc.mean()),
                'soft_vote_median': float(np.median(soft_acc)),
                'soft_vote_min': float(soft_acc.min()),
                'soft_vote_max': float(soft_acc.max()),
                'member_mean_mean': float(mem_mean.mean()),
                'n_above_chance_bonferroni': int(sum(
                    r['above_chance']['significant_bonferroni'] for r in rows)),
                'n_above_chance_uncorrected': int(sum(
                    r['above_chance']['significant_uncorrected'] for r in rows)),
                'ensemble_vs_member_mean': vs_mean,
                'ensemble_vs_val_selected_member': vs_val,
            },
        }

    # ---------------- relatorio ----------------
    for n_classes in TASKS:
        task_key = f'{n_classes}class'
        blk = out[task_key]
        any_sub = blk['subjects'][f'A{TEST_SUBJECTS[0]:02d}']
        print(f'\n{"=" * 92}')
        print(f'ZERO-SHOT  fonte A0{SOURCE_SUBJECT}  ->  {n_classes} classes   '
              f'(n={any_sub["n_test"]} trials, acaso={1/n_classes:.2f})')
        print(f'  limiar binomial exato: {any_sub["critical_accuracy_uncorrected"]:.4f} '
              f'| com Bonferroni({N_TESTS}): '
              f'{any_sub["critical_accuracy_bonferroni"]:.4f}')
        print(f'  membro de referencia (por validacao em A01T): '
              f'seed {blk["reference_member"]["seed"]}')
        print(f'{"=" * 92}')
        print(f'{"alvo":6s} {"papel":5s} {"membros (media+-dp)":22s} '
              f'{"soft":8s} {"hard":8s} {"ganho":7s} {"desac":7s} '
              f'{"p(acaso)":10s} sig')
        print('-' * 92)
        for t in TARGETS:
            r = blk['subjects'][f'A{t:02d}']
            sig = ('**' if r['above_chance']['significant_bonferroni']
                   else '*' if r['above_chance']['significant_uncorrected'] else '--')
            print(f'A{t:02d}    {r["role"]:5s} '
                  f'{r["member_mean"]:.4f} +- {r["member_std"]:.4f}      '
                  f'{r["soft_vote"]["accuracy"]:.4f}   '
                  f'{r["hard_vote"]["accuracy"]:.4f}   '
                  f'{r["ensemble_gain_over_member_mean"]:+.4f} '
                  f'{r["mean_pairwise_disagreement"]:.3f}'
                  f'/{r["chance_disagreement"]:.2f} '
                  f'{r["above_chance"]["p_value"]:.2e}  {sig}'
                  + ('   <- CONTROLE, em casa' if r['role'] == 'dev' else ''))
        s = blk['test_subject_summary']
        print('-' * 92)
        print(f'8 sujeitos de teste: soft vote mediana {s["soft_vote_median"]:.4f} '
              f'[{s["soft_vote_min"]:.4f}, {s["soft_vote_max"]:.4f}], '
              f'media {s["soft_vote_mean"]:.4f}')
        print(f'  acima do acaso: {s["n_above_chance_bonferroni"]}/8 com Bonferroni, '
              f'{s["n_above_chance_uncorrected"]}/8 sem correcao')
        for label, key in (('media dos membros   ', 'ensemble_vs_member_mean'),
                           ('membro val-selected ', 'ensemble_vs_val_selected_member')):
            e = s[key]
            print(f'  ensemble vs {label}: ganho {e["mean_gain"]:+.4f}, '
                  f'melhora em {e["n_subjects_improved"]}/8, '
                  f'Wilcoxon p={e["wilcoxon_p"]:.4f}, sinal p={e["sign_test_p"]:.4f}')

    (ROOT / 'results_zeroshot.json').write_text(
        json.dumps(out, indent=2), encoding='utf-8')
    print('\nresults_zeroshot.json salvo.')
    return out


if __name__ == '__main__':
    if '--collect' not in sys.argv:
        # load_seed_models le runs/within/ filtrando A01 base aug-on e carrega os
        # .keras correspondentes. E especifico de A01: o model_path() no
        # train_final.py tem 'a01' no nome. Isso vira um problema na Fase 3, nao agora.
        members = {}
        for n_classes in TASKS:
            members[n_classes] = load_seed_models(n_classes)
            print(f'{n_classes}-class: {len(members[n_classes])} modelos-fonte '
                  f'carregados', flush=True)

        for target in TARGETS:
            # Uma leitura de GDF por sujeito; as duas tarefas saem dela.
            X4, y4 = load_subject(target, 'E', n_classes=4)
            for n_classes in TASKS:
                run_target(n_classes, target, members[n_classes], X4, y4)
            if target == SOURCE_SUBJECT:
                for n_classes in TASKS:
                    check_control(n_classes)
                print('  controle A01 OK: igualdade exata semente a semente',
                      flush=True)
            del X4, y4
    collect()
