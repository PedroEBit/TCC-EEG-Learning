# CLAUDE.md

Instruções para qualquer agente que trabalhe neste repositório. Leia inteiro antes
de agir. Escrito em 30/09/2026.

---

## 0. Como trabalhar comigo (Pedro)

**Seja professor, não implementador.** O objetivo deste projeto não é o código
pronto, é eu aprender a fazer. Portanto:

- **Não implemente o que eu posso implementar.** Dê a especificação, as decisões de
  projeto, as armadilhas e o desenho experimental. Deixe o código para mim.
- Quando eu pedir código, prefira **esqueleto com `TODO` vazios** a solução completa.
- Menos dicas explícitas. Se eu estou perto, me deixe chegar sozinho.
- Explique o **mecanismo**, não só o resultado. "Por que quebrou", não "quebrou".
- Pode e deve **me contradizer**. Se a minha premissa está errada, diga na hora, com
  o dado que mostra. Já aconteceu várias vezes e foi útil todas.
- Números exatos, sempre. Se um número não está num arquivo deste repo, ele não
  entra em conclusão nenhuma.

Exceções em que você pode executar direto: higiene de git, refatoração mecânica,
gerar figuras, e rodar treinos longos que são só espera.

---

## 1. O que é este projeto

TCC virando paper. Decodificação de imagética motora (motor imagery) em EEG com
EEGNet, sobre o **BCI Competition IV dataset 2a** (9 sujeitos, 22 eletrodos, 4
classes, duas sessões em dias diferentes).

**Protocolo:** within-subject, cross-session. Treina em `A0xT`, testa em `A0xE`.
É o split oficial da competição. Não existe split aleatório, então augmentation
não consegue vazar estruturalmente.

O README.md é a fonte de verdade sobre resultados. Este arquivo é a fonte de
verdade sobre **o que vamos fazer a seguir**.

---

## 2. Ambiente

- **Use `.venv`**, nunca o Python do sistema (que não tem TensorFlow):
  `C:\Users\pedro\Workspaces\EEG_learning\.venv\Scripts\python.exe`
- TF 2.21.0 / Keras 3.14.0 / MNE 1.12.1 / NumPy 2.4.4. É o que está no
  `requirements.txt` e o que reproduz os resultados salvos.
- **CPU only.** TF ≥ 2.11 não tem GPU em Windows nativo, e o plugin DirectML não
  implementa backprop `channels_first`, que a depthwise do EEGNet exige.
  Um treino de 5 sementes leva ~90 min.
- Os `.gdf` (~600 MB) estão em `data/` e são gitignored. **Os 9 sujeitos estão no
  disco**, T e E, com os labels verdadeiros em `data/true_labels/`.

---

## 3. Regras metodológicas deste projeto

Estas regras são o que o projeto tem de mais valioso. Não as relaxe.

1. **Uma rodada não é um resultado.** Toda configuração roda com múltiplas
   sementes. A dispersão entre sementes no 2-class vai de 57.6% a 92.4%.
2. **Ablação pareada.** Para medir um componente: mesma arquitetura, mesmo split,
   mesmo augmentation, mesmo batch, mesma paciência, mesmo teste. **Muda só a coisa
   medida**, e pareia semente a semente contra os runs existentes.
3. **Teste estatístico pareado**, não comparação de médias no olho. Wilcoxon ou
   teste do sinal para locação; Pitman–Morgan para variância pareada; McNemar para
   dois classificadores nos mesmos trials.
4. **Nunca decida olhando o teste.** O README documenta que isso já foi violado no
   passado (banda, número de classes, adoção do augmentation). Daqui para frente,
   decisão se toma em validação ou em sujeitos de desenvolvimento.
5. **Reportado ≠ adotado.** Medir um componente não obriga a colocá-lo no pipeline.
   O bloco SE está medido e deliberadamente não adotado, pelo motivo em (4).
6. **Escreva a limitação.** Se um resultado tem n=5, diga n=5.

---

## 4. Estado atual: o que já está estabelecido

Todos os números abaixo são do **sujeito A01 apenas**, 5 sementes por tarefa.

| | 2-class | 4-class |
|---|---|---|
| EEGNet base | 82.6% ± 14.1% | 73.9% ± 3.2% |
| ensemble soft vote (5 sementes) | 91.0% (κ 0.819) | 79.2% (κ 0.722) |
| McNemar vs membro mediano | p = 0.29 | **p = 0.013** |

**Augmentation (ruído gaussiano, σ=0.1 por época, 5 cópias):** +17.4 e +12.7
pontos, melhora em 5/5 sementes nas duas tarefas. O ganho **não** é um deslocamento
uniforme: sem ele, três de cinco sementes 2-class colapsam para o acaso. Compra
confiabilidade, não acurácia.

**Bloco Squeeze-and-Excitation (+148 parâmetros), medido, não adotado:**
- Acurácia média: +4.4 pts (2-class) e +0.6 pts (4-class). Wilcoxon p = 1.00 e 0.81.
  **Não há ganho de acurácia.** O +8.6 pts do notebook não se reproduz.
- Variância: desvio cai **33.7×** (2-class) e **11.4×** (4-class).
  Pitman–Morgan p = 0.016 e p = 0.030. É o único efeito que atinge significância.
- Corta os dois rabos: piso 57.6→84.7 e teto 92.4→91.0 (2-class).
- O braço 4-class é o convincente, porque lá não havia colapso para resgatar.
- Custo para o ensemble: desacordo entre membros cai 0.231→0.128 e 0.242→0.214.
  No 4-class o ensemble piora (hard vote 77.4→75.7).

**Interpretabilidade:**
- A conv temporal **acha a fisiologia sozinha**: 7 de 8 filtros com pico entre 10.7
  e 17.6 Hz no 4-class. No 2-class só 4 de 8; os outros degeneram para picos abaixo
  de 4 Hz, fora da banda de entrada.
- A conv espacial **não** redescobre C3/C4. Padrões de Haufe: massa sensório-motora
  em 49.0% (2-class) e 51.2% (4-class), contra **54.5%** esperado sob indiferença
  espacial. C3 e C4 em 17.5º e 15.7º de 22. Replicado nos dez modelos.

**Calibração** (estudo em `C:\Users\pedro\Workspaces\Estudos\calibracao\`):
- Os membros individuais já estão **bem calibrados** (gaps de ~1 pp no 4-class).
- O ensemble soft vote é **subconfiante**: acerta 91.0% com confiança média 78.8%.
  ECE piora de 0.076 (média dos membros) para 0.126. Média de softmax encolhe o máximo.
- Temperature scaling **não ajuda**: T ajustado em 28 ou 56 trials overfita.
  Piorou o ECE em 2/5 e 3/5 membros.

---

## 5. Armadilhas conhecidas

- **Nomes de eletrodo.** O GDF só nomeia 5 dos 22: `EEG-Fz`, `EEG-C3`, `EEG-Cz`,
  `EEG-C4`, `EEG-Pz`. Os outros são `EEG-0`..`EEG-16`. A montagem padrão do IV-2a
  na ordem do arquivo é, e os cinco nomeados **confirmam** esse mapa:
  ```
   0 Fz
   1 FC3   2 FC1   3 FCz   4 FC2   5 FC4
   6 C5    7 C3    8 C1    9 Cz   10 C2   11 C4   12 C6
  13 CP3  14 CP1  15 CPz  16 CP2  17 CP4
  18 P1   19 Pz   20 P2
  21 POz
  ```
- **`validation_split` do Keras.** Pega os últimos 20% do array **sem embaralhar**.
  Num array `[originais, cópias...]` isso é validar em duplicata do treino. Já
  quebrou este projeto uma vez. O split de validação acontece **antes** do
  augmentation, e é assim que tem que continuar.
- **`train_final.py` é resumível e não atualiza artefato velho.** Ele pula qualquer
  semente cujo `runs/*.json` e `models/*.keras` existam. Para forçar retreino,
  apague os dois.
- **Alinhamento de rótulo no split E.** Os labels vêm de `.mat` separado, alinhados
  pela **ordem** dos cues. Se o MNE descartar uma época, quebra em silêncio. Há
  `assert` para isso em `load_subject`; não remova.
- **Deleção de arquivo pode ser bloqueada** pelo classificador do harness. Prefira
  `git rm --cached`, renomear, ou escrever em arquivo novo.

---

## 6. Roadmap: o que vamos fazer, nesta ordem

### Fase 1 — Preservar o estado atual ✅
Feito em 30/09/2026. Repositório limpo (só material do paper), tag
`v1-single-subject`.

### Fase 2 — Os 9 sujeitos (interpessoal) ← **PARCIAL, 02/10/2026**

**Feito:**
- **Split declarado no README antes de usar:** dev = A01, teste = A02..A09. A01 ja estava
  queimado, entao nomea-lo dev nao custa nada e deixa 8 sujeitos limpos.
- **Zero-shot completo** (`zeroshot.py`, 90 runs). Controle A01 reproduz os runs within em
  igualdade exata, semente a semente.
- **Razao normalizada** (`transfer_ratio.py`): transferencia / teto within do proprio
  sujeito. Sem ela a tabela absoluta engana, e enganou -- "A03 e A08 transferem" era
  artefato de serem os sujeitos mais faceis do dataset.
- **(a) Within nos 9** (`within9.py`): em andamento, ~3.3 min/run, loop semente-por-fora
  para que uma interrupcao deixe cobertura completa dos 9 com menos sementes.

**Achados que mudaram o README:**
- **O A01 e sujeito MEDIANO, nao forte.** A03, A08 e A09 o superam nas duas tarefas. A
  frase do protocolo que afirmava o contrario nunca teve dado por tras; foi corrigida.
- **O colapso de treino e do PIPELINE, nao do A01.** A02 2c (0.514 vs 0.674), A05 2c
  (0.535 vs 0.708), A06 4c (0.257 vs 0.396, val 0.232 abaixo do acaso). Assinatura
  consistente: parada precoce + validacao tao ruim quanto o teste, logo detectavel sem
  tocar no teste. Isso e a precondicao do braco SE.
- **O ensemble de sementes nao se sustenta contra o baseline certo.** Contra a media dos
  membros ganha 8/8 (2c); contra o membro escolhido por validacao, 4/8 (p=0.30), e no
  4-class o ganho e negativo. O achado do A01 (McNemar p=0.013 no 4c) NAO replicou.

**Falta:**
- Terminar as 5 sementes within nos 9 (roda sozinho, `python within9.py`).
- **Braco SE nos 9, com n=5.** NAO fazer com n=2: o achado do SE e sobre variancia, e com
  duas sementes "variancia" e o gap dividido por raiz(2). Pre-requisito: `se_ablation.py`
  tem `SUBJECT_ID` fixo em 1 e nao salva `proba` (secao 7).
- Euclidean Alignment como braco pareado (`align='ea-T'` ja existe no schema).
- Hipotese pre-registrada, a testar com n=5: a dispersao entre sementes e inversamente
  relacionada a acuracia media do sujeito? Spearman sobre 9 pares, por tarefa.

---

### Fase 2 (plano original, mantido para referencia)

O projeto inteiro é A01. O README lista isso como limitação. Deixou de ser
limitação: os 9 sujeitos estão no disco.

Objetivo: descobrir se os achados (augmentation, SE, ensemble) são **gerais** ou
são artefato do A01. Com 9 sujeitos × 5 sementes vira um desenho pareado entre
sujeitos, e é isso que torna o achado do SE defensável num paper. Hoje o
Pitman–Morgan tem 3 graus de liberdade.

Duas coisas diferentes, **não confundir**:
- **(a) Within-subject nos 9 sujeitos.** Treina em `A0xT`, testa em `A0xE`, para
  cada x. É replicação, não transferência. É a base de tudo.
- **(b) Cross-subject de verdade.** Treina num conjunto de sujeitos, testa noutro
  sujeito nunca visto. É aqui que entra o congelamento de camadas.

Fazer (a) antes de (b), sempre. Sem a linha de base por sujeito, nenhum número de
transferência é interpretável.

**Antes de rodar qualquer coisa:** definir o split de sujeitos de desenvolvimento
e de teste, e **escrever no README**. Toda decisão de projeto se toma nos de
desenvolvimento. Essa é a chance de consertar o pecado documentado nas Limitações.

### Fase 3 — Congelamento e transferência

Testar quais partes do EEGNet transferem entre pessoas.

**A convenção é congelar as camadas INICIAIS**, não as finais: a conv temporal
aprende essencialmente filtros de banda, e mu/beta são a mesma fisiologia em todo
mundo; a conv espacial depende de anatomia e posicionamento de eletrodo, que são
individuais.

Mas o nosso próprio resultado complica isso, e é por isso que vale rodar: se a
camada espacial não está achando C3/C4, não está claro que ela seja a parte
individual. E há a pista de que 4 dos 8 filtros temporais do modelo 2-class
degeneram; **transferir filtros temporais de um modelo treinado com mais dados pode
consertar isso.**

Rodar o espectro, não uma configuração: congelar nada / só temporal /
temporal+espacial / tudo menos o classificador.

Três controles obrigatórios:
- **Orçamento de calibração** como eixo explícito (0, 10, 20, 40 trials do alvo).
  Sem isso "a transferência funcionou" não quer dizer nada.
- **Alinhamento de domínio.** O z-score por época já faz alinhamento parcial. Num
  paper vamos ser cobrados por comparar com Euclidean Alignment.
- **Sementes em tudo.** Transferência tem variância maior que within-subject.

### Fase 4 — EEGNet-mSEM

Artigo: Wang, Ju, Sun, Yu, Li, Hu (2025), *"Improved EEGNet With a Multilevel
Spatial Feature Extraction Module for EEG Decoding"*, IEEE Trans. Instrum. Meas.,
vol. 74. **mSEM = multilevel Spatial feature Extraction Module. Não tem nada de
squeeze-and-excitation** (erro cometido uma vez, não repetir).

Substitui o `spatial_conv` inteiro por dois níveis:
1. **Local:** divide os eletrodos em k regiões cerebrais; cada região recebe uma
   `DepthwiseConv2D` de kernel `(n_g, 1)`, `depth_multiplier=1`. Saída
   `(batch, F1, 1, T)` por região.
2. **Concatena** as k saídas locais **com a entrada original inteira** →
   `(batch, F1, 22+k, T)`. Reter os 22 originais importa (ablação, Tabela III).
3. **Global:** `DepthwiseConv2D` kernel `(22+k, 1)`, `depth_multiplier=D=2` →
   `(batch, 2*F1, 1, T)`.

Custo com F1=8, k=4: 176 + 416 = 592 params, contra 352 do depthwise atual. **+240.**

Divisões reconstruídas do texto (índices da montagem da seção 5):
- **mSEM-1 (áreas funcionais, por letra, singleton funde com vizinho), k=4:**
  `[0-5] (Fz+FC*)`, `[6-12] (C*)`, `[13-17] (CP*)`, `[18-21] (P*+POz)`
- **mSEM-4 (hemisférios), k=3:**
  `[0,3,9,15,19,21]` sagital, `[1,2,6,7,8,13,14,18]` esquerda,
  `[4,5,10,11,12,16,17,20]` direita

**mSEM-2 e mSEM-3 exigem ler a Fig. 3(b) do artigo**, que é imagem. A mSEM-3 usa 6
regiões e é a que dá o SOTA deles. Não aceitar reconstrução de agente aí.

**Reality check.** No IV-2a, o ganho do mSEM sobre o EEGNet vai de +0.15 a +1.09
pontos (Fig. 6b: EEGNet 77.56%, mSEM-4 78.65%). O +1.38% do abstract é do dataset
próprio deles. **Um efeito de +1 ponto é invisível no nosso ruído de ±3.2% com um
sujeito só.** Por isso a Fase 2 vem antes. E eles não reportam desvio entre
execuções nem teste de significância para essa comparação; nós vamos reportar.

**O experimento que só nós podemos fazer.** O artigo *assume* que impor estrutura
anatômica faz a rede usar melhor a informação espacial, e nunca verifica. Nós temos
a medição: `spatial_patterns.py`, com a referência de 49.0% / 51.2% de massa
sensório-motora contra 54.5% de indiferença.

E funciona de forma exata, porque **entre o conv local e o global não há
não-linearidade**. Os dois são lineares e sequenciais, então o filtro espacial
efetivo sobre os 22 eletrodos originais é composição linear fechada: peso global
dos 22 canais originais, mais a soma dos pesos globais das k linhas locais
multiplicados pelos filtros locais reinseridos nas posições dos seus eletrodos.
Recupera-se um vetor exato de 22 dimensões por filtro, e aplica-se Haufe igual.

Pergunta binária e honesta: **o mSEM melhora porque recuperou estrutura anatômica,
ou melhora sem recuperar?** Qualquer das duas respostas vale publicar.

### Fase 5 — Escrever o paper

---

## 7. Pendências e dúvidas abertas

- **A banda 8–30 Hz precisa ser reexaminada.** O README registra que ela foi testada,
  deu 50.7% (acaso) e por isso a banda larga 4–40 Hz foi mantida. Mas: (i) não existe
  nenhum artefato daquele experimento, nem `runs/`, nem registro de banda em
  `results_*.json`; a banda é argumento default de `load_subject`; (ii) 50.7% está
  dentro do regime de colapso já medido neste projeto, onde uma semente azarada deu
  57.6%. **A rejeição da banda foi provavelmente n=1.** E 8–30 Hz contém mu e beta,
  ou seja, o sinal inteiro de imagética motora. Rejeitar isso por uma rodada é
  exatamente o erro que o projeto inteiro combate, aplicado à única decisão que nunca
  foi reexaminada. Refazer como ablação pareada de 5 sementes, igual à do SE.
- Não existe conjunto limpo para calibrar o ensemble: cada semente treina num 80%
  diferente do `A0xT`, então todo trial do T foi treino de algum membro.
- O soft vote do SE não é calculável: `run_se()` salva `y_pred` mas não as
  probabilidades. Corrigir se formos comparar ensembles de SE.

---

## 8. Como os runs são salvos (LEIA ANTES DE CRIAR QUALQUER RUN)

Tudo passa por **`paths.py`**. Não invente nome de arquivo à mão.

**Nome, estilo BIDS** (`chave-valor` separados por `_`), em `runs/<experimento>/`:

```
runs/within/sub-A01_task-4c_arch-base_aug-on_band-4-40_seed-0.json
runs/within/sub-A01_task-2c_arch-se_aug-on_band-4-40_seed-3.json
runs/zeroshot/src-A01_sub-A05_task-4c_arch-base_band-4-40_seed-2.json
runs/transfer/src-A01_sub-A05_task-4c_arch-base_freeze-temporal_budget-20_seed-1.json
```

Experimentos: `within` | `zeroshot` | `transfer`. Chave que não se aplica é omitida
(um run within não tem `src-` nem `budget-`; zero-shot não tem `aug-`, porque não treina).

**Schema de todo JSON**, três blocos:

```json
{ "config": {...}, "metrics": {...}, "predictions": {"y_true": [], "y_pred": [], "proba": []} }
```

O `config` carrega experimento, sujeito, sujeito-fonte, n_classes, arquitetura,
augmentation, banda, política de congelamento, orçamento, semente, **commit do git**
e **versões de biblioteca**. O nome do arquivo é conveniência para o olho; **o
`config` dentro do arquivo é a fonte de verdade**, e é por ele que se filtra.

**`proba` não é opcional em run novo.** Sem ele não dá para refazer soft vote,
ECE ou calibração sem recarregar modelo, e isso já custou retrabalho duas vezes
neste projeto.

**API:**

```python
import paths
paths.run_key(subject=5, n_classes=4, arch='base', seed=0, source_subject=1,
              experiment='zeroshot')          # monta o nome
paths.save_run(config, metrics, predictions)  # grava no lugar certo
paths.load_runs('within', subject=1, arch='se')   # filtra pelo config
paths.load_flat('within', arch='base')            # view achatada (código legado)
```

**Para ver o que já foi rodado:**

```bash
.venv\Scripts\python.exe paths.py            # tudo
.venv\Scripts\python.exe paths.py zeroshot   # só um experimento
```

**Nota histórica:** os runs da fase A01 nasceram num esquema plano
(`runs/run_4c_seed0.json`, `runs/ablation/`, `runs/se/`) e foram migrados para cá.
Os 10 runs base tiveram `proba` recuperado recarregando os modelos salvos, e a
migração foi verificada reproduzindo acurácia e matriz de confusão exatamente. Os
runs `aug-off` e `arch-se` não têm `proba`, porque seus modelos nunca foram salvos
(`config.predictions_complete` diz quais estão completos). Dois runs marcados
`exclude_from_aggregation` são retreinos de checagem de ambiente.

**Pendência:** o lado de *escrita* do `train_final.py` ainda grava no formato antigo
e tem `SUBJECT_ID` fixo em 1. Parametrizar o sujeito e passar a usar
`paths.save_run()` é tarefa do Pedro, na Fase 2.

---

## 9. Mapa dos arquivos

| Arquivo | O que é |
|---|---|
| `train_final.py` | pipeline final, semeado e reprodutível; runs principais + ablação de augmentation |
| `ensemble.py` | ensemble de sementes: soft/hard voting, McNemar, diversidade |
| `spatial_patterns.py` | padrões de Haufe da depthwise (a pergunta do C3/C4) |
| `se_ablation.py` | ablação do bloco Squeeze-and-Excitation, pareada por semente |
| `make_figures.py` | regenera as figuras em `figures/` a partir dos modelos salvos |
| `paths.py` | **convenção de nomes e schema dos runs**; rodar direto lista o que já existe |
| `zeroshot.py` | transferência zero-shot A01 -> 9 sujeitos; não treina, só prediz e mede |
| `within9.py` | within-subject nos 9; loop semente-por-fora, resumível |
| `transfer_ratio.py` | razão transferência / teto within; o número que o paper reporta |
| `make_readme_sections.py` | regenera as tabelas do README a partir dos JSON (idempotente) |
| `results_*.json` | métricas por semente, ensemble, ablações, padrões espaciais |
| `runs/`, `models/` | um JSON e um `.keras` por semente |
| `*.ipynb` | notebooks de aula; é de onde veio o bloco SE (Exercício 8). Não são entregáveis |

Estudo de calibração fica fora deste repo, em
`C:\Users\pedro\Workspaces\Estudos\calibracao\`.
