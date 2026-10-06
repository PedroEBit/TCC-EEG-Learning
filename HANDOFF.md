# Handoff — estado do projeto em 06/10/2026

Para o proximo agente que o Pedro trouxer. Este arquivo e o **estado de trabalho**: o que
esta feito, o que nos mordeu, e por onde pegar. Ele nao substitui os outros tres.

## Leia nesta ordem

1. **`CLAUDE.md`** — como trabalhar com o Pedro (secao 0, leia inteira e obedeca), as regras
   metodologicas (secao 3), o roteiro (secao 6) e a convencao de runs (secao 8).
2. **`CLAIMS.md`** — o que o dado sustenta e, explicitamente, o que nao sustenta. Em ingles.
3. **`README.md`** — a versao longa, com as tabelas. Elas sao **geradas por script**, nunca
   transcritas a mao.
4. Este arquivo.

---

## 1. Onde o projeto esta

TCC virando paper. EEGNet, imagética motora, BCI Competition IV-2a, 9 sujeitos.

**Fase 1 (A01 sozinho):** fechada ha tempo.
**Fase 2 (os 9 sujeitos):** fechada em 03/10/2026. Tres bracos completos, n=5 em todos:

| braco | runs | o que e |
|---|---|---|
| `within`, `arch-base` | 100 | A0xT -> A0xE, cada sujeito com o proprio modelo |
| `within`, `arch-se` | 92 | o mesmo com o bloco Squeeze-and-Excitation |
| `zeroshot` | 90 | modelos do A01 aplicados aos 9, sem treino nenhum |

**Fase 3 (congelamento/transferencia):** nao comecada.
**Fase 4 (EEGNet-mSEM):** nao comecada.

### Os numeros que importam

- **Within, media dos 9:** 0.801 (2 classes) e 0.630 (4 classes).
- **Zero-shot normalizado** pelo teto de cada sujeito: razao mediana 0.751 e 0.559; retido
  acima do acaso 33.3% e 14.4%. Acima do acaso corrigido: 4/8 e 2/8 sujeitos.
- **Ensemble entre sementes:** contra a media dos membros ganha 8/8 (2c, p=0.0039); contra o
  membro escolhido por validacao, 4/8 (p=0.30) e ganho negativo em 4c. **Nao se sustenta.**
- **SE nos 9:** **nao replica.** Reduz dispersao em 4/9 e 3/9; razao mediana de desvios 0.87x
  e 0.67x (abaixo de 1 = piorou). O A01 era outlier.
- **Dispersao vs acuracia** (hipotese pre-registrada): Spearman -0.667 (p=0.050) e -0.867
  (p=0.0025).

---

## 2. O que nos mordeu — leia antes de confiar em qualquer numero

Isto nao esta no CLAUDE.md porque aconteceu nesta rodada.

### 2.1 Variancia com n pequeno mente, e mentiu duas vezes

A correlacao de Spearman entre acuracia media e dispersao, no 4-class, deu **rho = -0.183,
p = 0.637 com 3 sementes** e **rho = -0.867, p = 0.0025 com 5**. Conclusao oposta. Com 3
pontos o desvio por sujeito e ruido demais para correlacionar com qualquer coisa.

**Regra pratica:** nao reporte, nem conclua, nada sobre dispersao com menos de 5 sementes por
celula. Com n=2 o "desvio padrao" e literalmente a distancia entre os dois valores dividida
por raiz(2) — nao carrega informacao nova e o simbolo `±` convida a uma leitura
distribucional que o dado nao sustenta. Com n<5, reporte valores individuais ou min-max.

### 2.2 Um numero central estava com o rotulo errado

O README dizia "the across-seed **standard deviation** falls by a factor of 33.7". Sao razoes
de **variancia**. O desvio padrao cai 5.80x e 3.38x (33.7 = 5.80²). Numeros certos, rotulo
errado, efeito inflado em 5.8x na escala que as tabelas usam.

Foi encontrado porque o `se_vs_base.py` **recalculou do zero** em vez de copiar. Faca isso
sempre que for citar um numero antigo.

### 2.3 Codigo duplicado quebra ablacao pareada

`se_ablation.run_se` se descrevia no proprio docstring como "copia fiel de
`train_final.run()`". Copia fiel e exatamente como uma ablacao pareada deixa de ser pareada.
Foi unificado: `train_final.run()` recebe `build_fn`. O `se_ablation.py` continua no repo como
artefato historico do A01; **nao o use para codigo novo.**

### 2.4 Afirmacoes que estavam no README sem dado por tras

- "A01 e um dos sujeitos mais fortes do dataset" — falso, e mediano. Corrigido.
- "O colapso de treino e uma limitacao do A01" — falso, e do pipeline. A02, A05 e A06
  colapsam, em sementes diferentes. Corrigido.

Se voce encontrar outra afirmacao sem numero atras, trate como suspeita.

### 2.5 Nao existe semente azarada

A semente 3 e a pior de cinco para o A01 (0.576) e a melhor de cinco para o A02. O
`stratified_split` deriva a particao treino/validacao da semente, entao a mesma semente gera
um split ruim para um sujeito e bom para outro. **A instabilidade nao se conserta evitando
uma semente.**

---

## 3. Compromissos metodologicos ja assumidos — nao os desfaca

1. **Split de sujeitos, declarado no README antes de usar:** desenvolvimento = **A01**,
   teste = **A02–A09**. Toda decisao de projeto (congelamento, orcamento, arquitetura, banda)
   se toma olhando A01 e validacao. Os oito nunca foram usados para selecionar nada.
2. **Nenhuma media dos 9 como resultado de transferencia** — A01 e dev e esta em casa.
3. **Baseline de ensemble e o membro escolhido por validacao**, nao a media dos membros nem o
   mediano por acuracia de teste. Media dos membros inclui os colapsados: e boneco de palha.
4. **Acaso e binomial exato**, nao 0.50/0.25. Limiares com Bonferroni sobre os 16 testes
   reportados: 0.618 (2c, n=144) e 0.326 (4c, n=288).

---

## 4. Como rodar

Ambiente: `.venv\Scripts\python.exe`, **CPU only** (TF >= 2.11 nao tem GPU em Windows nativo).
Custo medido: **~2.3 s/epoca no 2-class e ~4.1 s/epoca no 4-class**; um run vai de 1.5 a 10
minutos conforme quantas epocas o EarlyStopping deixa correr. 90 runs levam ~6 h.

```
.venv\Scripts\python.exe paths.py              # lista tudo que ja foi rodado
.venv\Scripts\python.exe within9.py            # treina base nos 9 (resumivel)
.venv\Scripts\python.exe within9.py se         # treina SE nos 9 (resumivel)
.venv\Scripts\python.exe within9.py --collect  # agrega
.venv\Scripts\python.exe zeroshot.py --collect
.venv\Scripts\python.exe transfer_ratio.py
.venv\Scripts\python.exe se_vs_base.py
.venv\Scripts\python.exe make_readme_sections.py   # regenera as tabelas do README
```

Treino longo: `nohup .venv/Scripts/python.exe -W ignore within9.py se > se_train.log 2>&1 &`
e acompanhe o `.log`. **Nao canalize por `grep` numa tool de background** — o buffer segura a
saida e voce fica cego. Conte arquivos em `runs/within/` em vez disso.

### Armadilhas operacionais

- **Heredoc do bash quebra** com aspas/acentos nestes scripts. Escreva o script Python num
  arquivo e execute, em vez de `python - <<'PY'`.
- **`git push` demora minutos** neste repo (varios JSONs). Use timeout generoso; ele completa.
- **Edicoes por `str.replace` com assert:** o README tem quebras de linha que nao sao onde
  voce espera. Sempre `assert s.count(old) == 1` e escreva o arquivo so no fim, para a
  edicao ser atomica.
- **Dois arquivos com nome nao derivavel do config:** os `_envcheck` em `runs/within/`. Sao
  os unicos; estao marcados `exclude_from_aggregation`.

---

## 5. O que fazer a seguir, em ordem de retorno sobre custo

### 5.1 Banda 8–30 Hz — barato, e corrige um pecado documentado

A unica decisao do projeto **nunca reexaminada**. O README registra que 8–30 Hz foi testada,
deu 50.7% (acaso) e por isso 4–40 Hz foi mantida. Mas: nao existe artefato daquele
experimento, e **50.7% esta dentro do regime de colapso que agora foi medido em tres
sujeitos**. A rejeicao foi quase certamente n=1.

E 8–30 Hz contem mu e beta, ou seja, o sinal inteiro de imagetica motora.

**Como fazer:** `load_subject` ja aceita `l_freq`/`h_freq`; `paths` ja carrega `band` no nome
e no config. Falta so passar a banda pelo `within9.py`. Ablacao pareada, 5 sementes, **so no
A01** (e o sujeito de desenvolvimento — decidir olhando os outros oito violaria o compromisso
da secao 3). Custo: 10 runs, ~40 min.

### 5.2 Euclidean Alignment — um revisor vai cobrar

O `prepare()` divide cada epoca por **um escalar** (std sobre canais e tempo juntos). Isso
corrige ganho global e **nao alinha a covariancia entre canais**, que e a unica coisa que o
filtro espacial consome. EA branqueia por `R^(-1/2)`, com `R` a covariancia espacial media do
sujeito, e iguala as estatisticas de segunda ordem de todo mundo.

O campo `align` ja existe no schema (`None` | `'ea-T'` | `'ea-E'`), entao o braco e um filtro,
nao uma migracao. **A escolha de em qual sessao estimar `R` nao e inocente:** `A0xT` e
indutivo e realista, `A0xE` e transdutivo e vai dar numero mais bonito. Declare qual.

### 5.3 Fase 3 — congelamento e transferencia

Ver CLAUDE.md secao 6. Tres pontos que o trabalho desta rodada acrescenta:

- O piso ja existe: zero-shot retem mediana de 33.3% e 14.4% do headroom. **Qualquer metodo
  de transferencia tem que bater isso**, e a comparacao e por sujeito, nao na media.
- **A09 e A07 em 4 classes sao os melhores alvos**: within 0.783 e 0.756, zero-shot retendo
  so ~11%. Muito a recuperar.
- Com um sujeito-fonte so, tudo mede semelhanca com o A01. Leave-one-subject-out (treinar em
  8, testar no nono) e outro regime e provavelmente o que o paper precisa.

### 5.4 Fase 4 — mSEM

CLAUDE.md secao 6. **A Fig. 3(b) do artigo precisa ser lida pelo Pedro** — mSEM-2 e mSEM-3
nao estao reconstruidas e reconstrucao de agente nao serve ali. Dado que o SE nao replicou,
calibre a expectativa: o ganho reportado do mSEM no IV-2a vai de +0.15 a +1.09 pontos, e o
ruido entre sementes neste pipeline e maior que isso em varios sujeitos.

### 5.5 Pendencias menores

- O soft vote do braco SE **agora e calculavel** (os runs tem `proba`). A pendencia da secao 7
  do CLAUDE.md esta fechada, mas o calculo em si nao foi feito.
- `se_ablation.py` continua com `SUBJECT_ID` fixo e sem `proba`. Nao foi consertado de
  proposito: e artefato historico. Use `within9.py se`.

---

## 6. Como trabalhar com o Pedro

**Leia a secao 0 do CLAUDE.md inteira.** O resumo: ele quer aprender a fazer, nao receber
pronto. De especificacao, decisoes de projeto, armadilhas e desenho experimental; deixe o
codigo com ele. Prefira esqueleto com `TODO` vazios a solucao completa. Contradiga quando a
premissa estiver errada — ele pede isso e tem sido util.

**O que aconteceu nesta rodada, e que voce deve corrigir:** sob prazo (ele tinha entrega no
mesmo dia), ele pediu que eu implementasse e eu implementei quase tudo. Funcionou para o
prazo e **violou a secao 0**. Em certo momento ele disse, textualmente, "estou perdido no que
esta acontecendo no projeto, e eu nao deveria estar". **Volte ao modo professor por padrao.**

Diagnostico util, do mesmo episodio: a lacuna dele **nao e sintaxe nem conceito de ML** — ele
escreveu a parte dificil do `collect()` certa de primeira, incluindo a armadilha do `sort` por
config. O que trava e **decomposicao**: dada uma lista de requisitos, por onde comecar e qual
funcao ja existente resolve a maior parte. Ensine isso, nao o resto. Ele pediu "aulas do zero"
depois do trabalho em andamento; o diagnostico acima vale mais que um curso generico.

---

## 7. Mapa dos arquivos novos desta rodada

| arquivo | o que e |
|---|---|
| `zeroshot.py` | transferencia zero-shot A01 -> 9; nao treina, so prediz e mede |
| `within9.py` | within nos 9, `base` e `se`; loop semente-por-fora, resumivel |
| `transfer_ratio.py` | razao transferencia / teto within; o numero que o paper reporta |
| `se_vs_base.py` | ablacao pareada do SE em dois niveis (dentro e entre sujeitos) |
| `make_readme_sections.py` | regenera as tabelas do README a partir dos JSON |
| `CLAIMS.md` | o que o dado sustenta e o que nao sustenta |
| `TESES_PAPER.md` | resumo de mao em portugues, **fora do git**, para enviar a colaborador |

**Por que o loop e semente-por-fora:** se a execucao for interrompida, sobra cobertura
completa dos 9 sujeitos com menos sementes, e os testes pareados entre sujeitos ainda rodam
com n menor e declarado. Sujeito-por-fora deixaria "5 sementes dos sujeitos 2 a 5", que
reintroduz a limitacao de cobertura que a Fase 2 existe para matar.
