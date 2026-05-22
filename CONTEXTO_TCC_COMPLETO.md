# CONTEXTO COMPLETO PARA TCC — EEG Motor Imagery com EEGNet

## Documento de referencia para outra IA escrever o TCC
**Autor:** Pedro | **Data de compilacao:** 2026-05-22
**Nota:** Este documento contem TUDO que foi feito no projeto — arquiteturas, justificativas camada a camada, resultados, evolucao estatistica, decisoes de design e reflexoes. Use-o como base para um TCC extremamente detalhado.

---

## 1. VISAO GERAL DO PROJETO

### 1.1 Objetivo
Classificar padroes de imaginacao motora (Motor Imagery - MI) a partir de sinais de EEG usando redes neurais convolucionais, especificamente a arquitetura EEGNet. O projeto evoluiu de fundamentos teoricos ate uma implementacao pratica com data augmentation e integracao online com jogo via TCP.

### 1.2 Dataset
- **Nome:** BCI Competition IV Dataset 2a (Graz University of Technology)
- **Referencia:** Brunner et al., 2008
- **Sujeitos:** 9 (A01 a A09)
- **Classes:** 4 — Left Hand (769), Right Hand (770), Feet (771), Tongue (772)
- **Sessoes:** 2 por sujeito — T (treino) e E (avaliacao)
- **Trials por sessao:** 288 (72 por classe, perfeitamente balanceado)
- **Canais EEG:** 22 (sistema 10-20): Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
- **Taxa de amostragem:** 250 Hz
- **Formato:** GDF (General Data Format) + MAT (labels para sessao E)
- **Duracao do trial:** 4 segundos pos-cue (0 a 4s)
- **Estrutura do tensor:** (n_trials, n_channels, n_times) = (288, 22, 1001)

### 1.3 Stack Tecnologico
- **Linguagem:** Python
- **Processamento EEG:** MNE 1.11/1.12
- **Deep Learning:** TensorFlow 2.21 / Keras
- **Computacao:** CPU-only (DirectML nao suporta channels_first backprop)
- **Metricas:** scikit-learn (confusion_matrix, classification_report, cohen_kappa_score, r2_score)
- **Visualizacao:** Matplotlib, Seaborn

### 1.4 Referencia Principal
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.

---

## 2. FUNDAMENTOS NEUROFISIOLOGICOS

### 2.1 O que e EEG
Eletroencefalografia (EEG) mede diferencas de potencial eletrico na superficie do cranio, geradas pela atividade sincronizada de neuronios corticais. Cada eletrodo e como um microfone posicionado sobre a orquestra cerebral.

### 2.2 Bandas de Frequencia

| Banda | Frequencia | Estado / Significado |
|-------|-----------|---------------------|
| Delta | 1-4 Hz | Sono profundo, lesoes |
| Theta | 4-8 Hz | Sonolencia, memoria de trabalho |
| **Alpha/Mu** | **8-13 Hz** | **Repouso motor, olhos fechados** |
| **Beta** | **13-30 Hz** | **Atividade motora, concentracao** |
| Gamma | 30-100 Hz | Processamento cognitivo elevado |

### 2.3 Motor Imagery e ERD/ERS
- **Motor Imagery (MI):** quando o sujeito *imagina* mover uma parte do corpo (sem movimento real)
- **ERD (Event-Related Desynchronization):** supressao das oscilacoes alpha/beta durante MI — os neuronios param de oscilar em sincronia
- **ERS (Event-Related Synchronization):** aumento de oscilacoes (beta rebound) apos MI
- **Contralateralidade:** imaginar mao esquerda -> ERD no hemisferio direito (C4), imaginar mao direita -> ERD no hemisferio esquerdo (C3)
- **Ritmo Mu (~8-12 Hz):** principal marcador de MI, centrado sobre o cortex motor

### 2.4 Por que o sinal bruto nao basta
A informacao discriminativa nao esta na amplitude bruta — esta nas frequencias especificas. O sinal bruto mistura todas as frequencias juntas, e o ERD some nessa mistura. E necessario decompor o sinal em frequencias para detectar a modulacao.

### 2.5 Filtragem: 4-40 Hz
- **Abaixo de 4 Hz (descartado):** movimento corporal, respiracao (~0.2 Hz), drift de eletrodo
- **Acima de 40 Hz (descartado):** EMG (atividade muscular do couro cabeludo, amplitude 10x maior que sinal neural), ruido da rede eletrica (50/60 Hz)

---

## 3. PIPELINE DE PRE-PROCESSAMENTO

### 3.1 Carregamento dos Dados
1. Leitura do arquivo GDF com MNE (`read_raw_gdf`)
2. Selecao apenas de canais EEG (drop EOG)
3. Selecao dos 22 canais de interesse
4. Sessao T: labels embutidos nas anotacoes (769-772)
5. Sessao E: onsets marcados como 783, labels reais no arquivo .mat

### 3.2 Filtragem
- Filtro bandpass IIR: 4-40 Hz (padrao) ou 8-30 Hz (experimento de banda)
- Justificativa: isolar bandas mu/alpha e beta relevantes para MI

### 3.3 Epoching
- Janela: 0 a 4 segundos pos-cue
- Sem correcao de baseline
- Resultado: (288, 22, 1001) por sujeito/sessao

### 3.4 Normalizacao
- Z-score por epoch independentemente (eixos canais x tempo)
- `mu = X.mean(axis=(1,2), keepdims=True)`
- `std = X.std(axis=(1,2), keepdims=True) + 1e-8`
- Justificativa: cada epoch tem sua propria escala de amplitude; normalizar garante que o modelo nao dependa de amplitude absoluta

### 3.5 Reshaping para EEGNet
- `(n, C, T) -> (n, 1, C, T)` — adiciona dimensao de "profundidade" (como imagem P&B)
- Formato channels_first: batch, feature_maps, canais, tempo

---

## 4. ARQUITETURA EEGNet — CAMADA A CAMADA

### 4.1 Visao Geral
Total de parametros: **4,012** (15.67 KB) — extremamente compacto

```
Entrada:         (batch,  1, 22, 1001)
Apos temporal:   (batch,  8, 22, 1001)   1,000 params
Apos spatial:    (batch, 16,  1, 1001)     352 params
Apos AvgPool(4): (batch, 16,  1,  250)
Apos SepConv:    (batch, 16,  1,  250)     512 params
Apos AvgPool(8): (batch, 16,  1,   31)
Apos Flatten:    (batch, 496)
Apos Dense:      (batch,   4)            1,988 params
Total:                                   4,012 params
```

### 4.2 Block 1 — Convolucao Temporal
```python
Conv2D(filters=F1=8, kernel_size=(1, 125), padding='same', use_bias=False,
       data_format='channels_first', name='temporal_conv')
BatchNormalization(axis=1)
```

**Parametros:** 1 x 125 x 1 x 8 = **1,000**

**O que faz:** Cada um dos 8 filtros "varre" 0.5 segundos de sinal (125 amostras a 250 Hz) em cada canal independentemente. O kernel (1, 125) toca apenas 1 canal por vez no eixo vertical.

**O que aprende:** Filtros de frequencia — essencialmente descobre sozinha que as bandas mu e beta sao as mais relevantes para MI. E como fazer Fourier de forma aprendida (data-driven).

**Por que kernel de 125 (0.5s):** Um ciclo de 10 Hz (Alpha) dura 0.1s = 25 amostras. Com 125 amostras (0.5s), a rede enxerga ~5 ciclos, suficiente para discriminar oscilacao real de ruido.

**Por que use_bias=False:** Combinado com BatchNorm imediatamente depois, o bias seria redundante (BN faz shift proprio).

**Por que padding='same':** Preserva a dimensao temporal — a saida tem o mesmo comprimento da entrada.

### 4.3 Block 1 — Convolucao Espacial (Depthwise)
```python
DepthwiseConv2D(kernel_size=(22, 1), depth_multiplier=D=2,
                depthwise_constraint=max_norm(1.0), use_bias=False,
                data_format='channels_first', name='spatial_conv')
BatchNormalization(axis=1)
Activation('elu')
```

**Parametros:** 22 x 1 x 8 x 2 = **352**

**O que faz:** Para cada um dos 8 filtros temporais, aplica 2 filtros espaciais que combinam todos os 22 canais em um unico valor. O kernel (22, 1) "engole" todos os canais — saida vai de (8, 22, 1001) para (16, 1, 1001).

**O que aprende:** Combinacoes lineares de canais — descobre quais eletrodos (C3 vs C4) carregam informacao discriminativa, de forma analoga ao CSP (Common Spatial Patterns). E como beamforming aprendido.

**Por que DepthwiseConv2D e nao Conv2D padrao:**
- DepthwiseConv aplica um filtro **por canal de entrada** (nao global)
- Conv2D naive (22x125, 8 filtros) teria 22,000 parametros — **16.3x mais** que a separacao temporal->espacial (1,352 params)
- A separacao forca a rede a aprender **o que** (frequencia) e **onde** (espaco) independentemente

**Por que depth_multiplier=2:** Cada filtro temporal ganha 2 "visoes" espaciais diferentes. Total: 8 x 2 = 16 feature maps. Captura mais diversidade de padroes espaciais sem explosao parametrica.

**Por que max_norm(1.0):** Com apenas 288 trials, pesos podem explodir — um unico eletrodo dominaria com peso absurdo (ex: C3=50, todos outros ~0). max_norm(1.0) forca pesos numa esfera de raio 1 — regularizacao com interpretacao fisiologica (plausibilidade biologica).

### 4.4 Ativacao ELU
```python
Activation('elu')  # ELU(x) = x se x > 0, alpha*(e^x - 1) se x < 0
```

**Por que ELU e nao ReLU:** ReLU zera valores negativos -> gradiente zero -> "dying neurons". O sinal EEG tem componentes negativos com significado fisiologico. ELU mantem gradiente nao-zero para valores negativos, preservando aprendizado. Os filtros espaciais produzem combinacoes lineares que naturalmente geram valores negativos.

### 4.5 AveragePooling2D
```python
AveragePooling2D(pool_size=(1, 4), data_format='channels_first')  # Block 1
AveragePooling2D(pool_size=(1, 8), data_format='channels_first')  # Block 2
```

**O que faz:** Block 1 reduz tempo de 1001 para 250 (downsampling 4x). Block 2 reduz de 250 para 31 (downsampling 8x).

**Por que AveragePooling e nao MaxPooling:**
1. MaxPooling pegaria artefatos biologicos (spike de EMG, piscar de olho com valor maximo enorme)
2. EEG carrega informacao na **energia das oscilacoes**, nao em picos isolados
3. Uma onda de 10 Hz tem picos positivos e negativos alternados — MaxPooling pegaria so o pico positivo, descartando a estrutura oscilatoria
4. AveragePooling preserva a energia media da janela inteira

**Resultado do ablation study:** MaxPooling atingiu val_acc=0.7759 vs AveragePooling 0.7069 (neste run especifico), mas a justificativa neurofisiologica favorece AveragePooling para generalizacao.

### 4.6 Dropout
```python
Dropout(0.5)  # Em ambos os blocos
```

**Justificativa:** Com apenas 288 trials por sujeito, overfitting e severo. Dropout de 50% e agressivo mas necessario.

### 4.7 Block 2 — SeparableConv2D
```python
SeparableConv2D(filters=F2=16, kernel_size=(1, 16), padding='same',
                use_bias=False, data_format='channels_first', name='separable_conv')
BatchNormalization(axis=1)
Activation('elu')
AveragePooling2D(pool_size=(1, 8))
Dropout(0.5)
```

**Parametros:** **512** (depthwise 16 + pointwise 256 x 2)

**O que faz:** Captura interacoes tempo-espaco de ordem superior. Opera sobre os 16 feature maps do Block 1, refinando padroes temporais e misturando filtros entre si.

**Por que SeparableConv e nao Conv2D:** Muito mais eficiente — depthwise processa cada feature map separadamente, pointwise combina. Menos parametros, menos overfitting.

### 4.8 Classificador
```python
Flatten()  # (batch, 16, 1, 31) -> (batch, 496)
Dense(n_classes, activation='softmax', kernel_constraint=max_norm(0.25))
```

**Parametros:** 496 x 4 + 4 = **1,988** (para 4 classes)

**Por que max_norm(0.25) mais restritivo que max_norm(1.0):** Dense recebe 496 features e mapeia para 4 classes — e a decisao final. Com 496 pesos por neuronio de saida, facilmente um unico peso dominaria. 0.25 forca a decisao a ser distribuida entre muitas features.

### 4.9 BatchNormalization(axis=1) — Detalhamento
- Normaliza ao longo do eixo 1 (feature maps/filtros)
- Sem BatchNorm, cada filtro temporal produziria ativacoes em escalas completamente diferentes
- Um filtro sensivel a Alpha pode gerar valores 10x maiores que um sensivel a Beta
- Resultado do ablation study: **Sem BatchNorm val_acc=0.3276** vs com BatchNorm 0.7069 — essencial

---

## 5. ABLATION STUDY

Realizado com sujeito 1, 4 classes, split 80/20.

| Variante | Val Accuracy | Observacao |
|----------|-------------|------------|
| EEGNet original | 0.7069 | Baseline |
| Sem BatchNormalization | 0.3276 | Colapso total — pior que chance |
| MaxPooling (em vez de AvgPool) | 0.7759 | Ligeiramente melhor neste run |

**Conclusoes:**
- BatchNorm e absolutamente critico — sem ele o modelo nao converge
- MaxPooling vs AvgPooling: diferenca pequena no within-subject, mas AvgPool e preferido por razoes neurofisiologicas (preserva energia oscilatoria)

---

## 6. VISUALIZACAO DE FILTROS APRENDIDOS

### 6.1 Filtros Temporais
- Shape dos pesos: (1, 125, 1, 8) — 8 filtros de 125 amostras cada
- Visualizados via FFT para ver resposta em frequencia
- Varios filtros se concentraram nas bandas Alpha (8-13 Hz) e Beta (13-30 Hz)
- Confirma que a rede aprendeu a importancia dessas bandas para MI

### 6.2 Filtros Espaciais
- Shape: (22, 1, 8, 2) — 16 filtros com pesos sobre os 22 canais
- Interpretaveis como "padroes espaciais" ou "filtros de beamforming"
- Canais C3 e C4 tendiam a ter pesos altos — consistente com lateralizacao hemisferica de MI
- Nomes dos canais: Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz

---

## 7. DOMAIN SHIFT E TRANSFER LEARNING

### 7.1 O Problema
Cada sujeito EEG e um **dominio diferente**:
- Anatomia craniana unica -> eletrodos captam de angulos diferentes
- Impedancia de pele variavel -> amplitude e ruido diferentes
- Estrategias cognitivas individuais -> padroes temporais distintos
- Artefatos especificos (piscar, tensao muscular)

### 7.2 Abordagens Testadas

| Abordagem | Descricao | Vantagem | Desvantagem |
|-----------|-----------|----------|-------------|
| Within-subject | Treina/testa no mesmo sujeito | Alta acuracia | Requer calibracao |
| Cross-subject (LOSO) | Treina em N-1, testa em 1 | Sem calibracao | Menor acuracia |
| Fine-tuning | Pre-treina em N-1, adapta ao novo | Equilibrado | Requer poucos dados |

### 7.3 Experimento de Fine-tuning (Sujeito 1 -> Sujeito 2)
- **Estrategia:** Congelar camada temporal (Conv2D) + Fine-tune espacial (DepthwiseConv) + Classifier
- **Intuicao:** Filtros temporais (frequencias) sao mais universais; filtros espaciais (anatomia) sao individuais
- **Learning rate:** 0.0001 (10x menor que treino normal) para nao destruir filtros pre-treinados

**Resultados Sujeito 2 (4 classes):**
- Fine-tuning: melhor val_acc = **0.3966** (39.7%)
- Do zero: melhor val_acc = **0.5862** (58.6%)

**Interpretacao:** O fine-tuning **nao** ajudou neste caso — as diferencas entre sujeitos 1 e 2 sao tao grandes que os filtros espaciais pre-treinados atrapalham mais do que ajudam. O modelo do zero, com learning rate maior, se adapta melhor ao novo sujeito. Isso demonstra a severidade do domain shift em EEG.

---

## 8. VARIANTE PROPRIA — SE Block (Squeeze-and-Excitation)

### 8.1 Motivacao
Pedro propôs adicionar um mecanismo de atencao (SE Block) apos a DepthwiseConv2D para ponderar a relevancia de cada filtro espacial. A intuicao: nem todos os 16 filtros espaciais sao igualmente uteis para cada trial — um mecanismo de atencao pode aprender a suprimir filtros ruidosos e amplificar filtros informativos.

### 8.2 Implementacao
```python
# Apos spatial_conv + BN + ELU:
se = GlobalAveragePooling2D(data_format='channels_first')(x)  # Squeeze: (batch, 16)
se = Dense(4, activation='relu')(se)      # Bottleneck: reducao 16->4
se = Dense(16, activation='sigmoid')(se)  # Excitation: 4->16, sigmoid para pesos [0,1]
se = Reshape((16, 1, 1))(se)             # Reshape para broadcast
x = Multiply()([x, se])                   # Reescala feature maps
```

### 8.3 Parametros Adicionais
- SE Block: 16*4 + 4 + 4*16 + 16 = **148 params**
- Total modelo: **4,160** (vs 4,012 original — apenas +3.7%)

### 8.4 Resultado
| Modelo | Val Accuracy (4 classes, Sujeito 1) |
|--------|-------------------------------------|
| EEGNet original | 0.7069 (70.7%) |
| **EEGNet + SE Block** | **0.7931 (79.3%)** |
| **Delta** | **+0.0862 (+8.6 pontos percentuais)** |

### 8.5 Por que funcionou
O SE Block aprende a ponderar a relevancia dos filtros espaciais trial-a-trial. Quando o sujeito imagina mao esquerda, o bloco amplifica filtros que capturam o ERD no hemisferio direito e suprime filtros que captam ruido. E um mecanismo de atencao por canal (channel attention) adaptado ao contexto de EEG.

---

## 9. EVOLUCAO ESTATISTICA — MELHORIAS INCREMENTAIS (NOTEBOOK 3)

### 9.1 Contexto
Todas as metricas abaixo sao para **Sujeito 1**, avaliadas no **test set** (sessao E, dados nunca vistos durante treino). O ponto de partida e o baseline de 4 classes do notebook 2.

### 9.2 Progressao Completa

| # | Experimento | Classes | Banda | Augmentation | N Treino | **Accuracy** | **Kappa** | Observacao |
|---|-------------|---------|-------|--------------|----------|-------------|-----------|------------|
| 0 | Baseline 4-class (notebook 2) | 4 | 4-40 Hz | Nenhum | 288 | **66.7%** | **0.556** | Ponto de partida |
| 1 | Baseline 2-class | 2 | 4-40 Hz | Nenhum | 144 | **50.7%** | **0.014** | Chance! Overfitting severo com 144 trials |
| 2 | 2-class + banda 8-30 Hz | 2 | 8-30 Hz | Nenhum | 144 | **50.7%** | **0.014** | Nenhuma melhora — EEGNet ja aprende a ignorar bandas irrelevantes internamente |
| 3 | **2-class + Gaussian Noise Aug** | 2 | 4-40 Hz | 5 copias, std=0.1 | 864 | **88.9%** | **0.778** | **SALTO ENORME — de 50% para 89%!** |
| 4 | 4-class + Gaussian Noise Aug | 4 | 4-40 Hz | 5 copias, std=0.1 | 1728 | **80.2%** | **0.736** | Tambem excelente para 4 classes |

### 9.3 Detalhamento: Baseline 2-class (Exercicio 1)
- **Accuracy: 0.5069** (basicamente chance level de 50%)
- **Kappa: 0.0139** (quase zero — nenhum poder preditivo real)
- **Classification Report:**
  - Left Hand: precision=0.55, recall=0.08, f1=0.14 (praticamente nao detecta!)
  - Right Hand: precision=0.50, recall=0.93, f1=0.65 (modelo enviesado para Right)
- **Confusion Matrix:** Left predicted as Right em 66/72 casos
- **Diagnostico:** O modelo decorou os dados de treino (144 trials muito poucos para 4,012 params) e nao generalizou

### 9.4 Detalhamento: Banda 8-30 Hz (Exercicio 2)
- **Accuracy: 0.5069** — identica ao baseline
- **Kappa: 0.0139** — identico
- **Conclusao:** Restringir a banda nao ajudou porque a primeira camada convolucional da EEGNet JA e um filtro temporal aprendido — a rede aprende sozinha a ignorar frequencias irrelevantes. Isso demonstra a capacidade adaptativa da arquitetura.

### 9.5 Detalhamento: Gaussian Noise Augmentation (Exercicio 3) — O DIVISOR DE AGUAS
- **Dataset:** 144 trials originais + 5 copias ruidosas = **864 trials**
- **Noise std:** 0.1 relativo ao std de cada epoch
- **Accuracy: 0.8889 (88.9%)**
- **Kappa: 0.7778**
- **Classification Report (2 classes):**
  - Left Hand: precision=0.86, recall=0.93, f1=0.89
  - Right Hand: precision=0.92, recall=0.85, f1=0.88
  - Macro avg: precision=0.89, recall=0.89, f1=0.89
- **Por que funcionou tao bem:** EEG e inerentemente ruidoso — cada trial tem ruido de fundo diferente. O noise injection simula essa variabilidade natural. O sinal de MI (ERD/ERS) e uma modulacao relativamente lenta na potencia de mu/beta, robusta a ruido aditivo. Cada copia augmentada e como um "trial alternativo" com ruido de fundo diferente.

### 9.6 Detalhamento: 4 Classes + Noise Augmentation
- **Dataset:** 288 trials x 6 = **1,728 trials**
- **Accuracy: 0.8021 (80.2%)**
- **Kappa: 0.7361**
- **Classification Report (4 classes):**
  - Left Hand: precision=0.92, recall=0.79, f1=0.85
  - Right Hand: precision=0.90, recall=0.93, f1=0.91
  - Feet: precision=0.65, recall=0.74, f1=0.69
  - Tongue: precision=0.77, recall=0.75, f1=0.76

---

## 10. REPORT COMPARATIVO POS-AUGMENTATION

### 10.1 Tabela Completa de Metricas (4 Classes vs 2 Classes, ambos com Noise Aug)

| Metrica | 4 Classes | 2 Classes |
|---------|-----------|-----------|
| **Accuracy** | 0.8021 | 0.8889 |
| **R2** | 0.6202 | 0.6741 |
| **Cohen's Kappa** | 0.7361 | 0.7778 |
| **F1 Macro** | 0.8022 | 0.8887 |
| **Precision Macro** | 0.8088 | 0.8916 |
| **Recall Macro** | 0.8021 | 0.8889 |
| **Confianca Media (Acertos)** | 0.8270 | 0.9402 |
| **Confianca Media (Erros)** | 0.6698 | 0.7510 |
| Train size (augmented) | 1,728 | 864 |
| Test size | 288 | 144 |
| Chance level | 25.0% | 50.0% |

### 10.2 Analise de Confianca do Softmax
- **2 classes:** confianca media nos acertos = 94.02% vs erros = 75.10%
  - Gap de ~19% permite usar threshold para filtrar predicoes incertas em tempo real
  - Viavel para BCI online com threshold de confianca
- **4 classes:** confianca media nos acertos = 82.70% vs erros = 66.98%
  - Gap menor (~16%), mas ainda discriminativo

### 10.3 Sobre o Cohen's Kappa
- Cohen's Kappa e a metrica padrao do BCI Competition porque desconta o acerto por acaso
- Kappa = 0.778 (2 classes) e 0.736 (4 classes) indicam performance robusta e **comparavel**
- A diferenca absoluta em accuracy (88.9% vs 80.2%) e amplificada pelo chance level diferente
- Os Kappas proximos mostram que o poder discriminativo real e similar

### 10.4 Sobre o R2
- R2 aplicado as probabilidades do softmax revela a confianca real do modelo
- 0.674 no 2-class: o modelo acerta, mas nem sempre com conviccao total
- Util para entender se o modelo esta "confiante" ou "chutando certo"

---

## 11. TREINAMENTO — DETALHES TECNICOS

### 11.1 Hiperparametros
```python
optimizer = Adam(learning_rate=1e-3)
loss = 'sparse_categorical_crossentropy'
epochs = 300 (com EarlyStopping)
batch_size = 32
validation_split = 0.2
```

### 11.2 Callbacks
```python
EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
```

### 11.3 Comportamento do Treinamento
- **4 classes, sem augmentation (288 trials):** Treino converge rapido (~80% train acc em ~20 epochs), mas val_acc satura em ~55-65%. Gap grande = overfitting.
- **2 classes, sem augmentation (144 trials):** Overfitting extremo — train acc sobe, test acc ~50%.
- **2 classes, com noise augmentation (864 trials):** Convergencia mais lenta mas estavel. Val_acc sobe gradualmente, sem gap excessivo com train_acc. Early stopping por volta de epoch ~50-80.

---

## 12. DATA AUGMENTATION — DETALHES

### 12.1 Gaussian Noise Injection
```python
def augment_gaussian_noise(X, y, n_copies=5, noise_std=0.1):
    epoch_std = X.std(axis=(1, 2), keepdims=True)
    noise = np.random.randn(*X.shape) * (noise_std * epoch_std)
    X_augmented = X + noise
```

- **n_copies=5:** cada trial gera 5 copias ruidosas
- **noise_std=0.1:** 10% do std da epoch — calibrado para nao destruir o sinal
- **Aplicado antes de normalizar:** importante — augmentar dados brutos, normalizar depois
- **Nunca aplicado ao test set**

### 12.2 Sliding Window Augmentation (Planejado mas nao completado)
- Extrai sub-janelas sobrepostas de cada trial de 4s
- Window de 2s, step de 0.5s -> 5 janelas por trial
- Proposito duplo: multiplica dados + treina para janelas mais curtas (menor latencia online)
- O ERD tipicamente comeca ~500ms pos-cue e pico em 1-2s -> janela de 2-3s captura maior parte

### 12.3 Pipeline Completo Planejado (Nao finalizado)
1. Carregar dados 2-class, banda 8-30 Hz
2. Sliding window no treino
3. Noise augmentation nos windows
4. Normalizar e reshape
5. Treinar e avaliar
- Potencial: 144 trials x 5 windows x 5 noise copies = 3,600 trials

---

## 13. SEPARACAO TEMPORAL vs ESPACIAL — POR QUE FUNCIONA

### 13.1 O Problema do Conv1D para EEG
Conv1D com kernel (k,) mistura informacao espacial e temporal indiscriminadamente — trata todos os canais como features independentes na mesma dimensao temporal. Perde a informacao espacial: qual canal captura o que, qual hemisferio esta suprimido.

### 13.2 A Solucao do EEGNet
Tratar sinal como "imagem" (canais, tempo) e usar Conv2D em duas etapas:
- **Etapa 1 — Temporal:** kernel (1, T) varre so o eixo tempo, nao "ve" canais adjacentes
- **Etapa 2 — Espacial:** kernel (C, 1) varre so o eixo canais, nao "ve" tempo adjacente

### 13.3 Reducao Parametrica
- Conv2D "naive" (22x125, 8 filtros): **22,000** parametros
- Separacao temporal (1,000) + espacial (352): **1,352** parametros
- **Reducao de 16.3x** com capacidade expressiva preservada

### 13.4 Diferenca entre "espacial" em imagens e em EEG
Em imagens, pixels vizinhos sao vizinhos no espaco fisico de forma uniforme. No EEG, proximidade entre eletrodos nao garante similaridade de sinal — depende da profundidade e geometria do cranio (que varia entre pessoas). Nao existe vizinhanca uniforme como numa grade de pixels.

---

## 14. APLICACAO ONLINE — JOGO BCI

### 14.1 Arquitetura do Sistema
Projeto OpenBCIOnlineProtocol — jogo estilo Pac-Man controlado por BCI:
- **Servidor TCP:** recebe comandos do classificador EEG (2=LeftArrow, 3=RightArrow, 10=Quit)
- **Cliente pygame:** renderiza personagem que se move left/right coletando cristais
- **Comunicacao:** TCP localhost (127.0.0.1:8123)
- **Protocolo:** comandos codificados como inteiros de 1 byte (big-endian)

### 14.2 Versoes
1. **OnlineProtocol.py:** versao com input de teclado via multiprocessing (prototipo)
2. **OnlineProtocol - TCP.py:** versao com servidor TCP para receber comandos externos (BCI)

### 14.3 Consideracoes para BCI Online
- **Latencia:** com janela de 4s, resposta minima de 4s. Com sliding window de 2s, cai para 2s.
- **Threshold de confianca:** em BCI para jogos, alta Precision importa mais — errar ativamente (ir pro lado errado) e pior que nao responder. Threshold de softmax > 0.7 para filtrar incerteza.
- **Confianca do modelo 2-class:** 94% nos acertos vs 75% nos erros — gap suficiente para threshold efetivo.

---

## 15. DECISOES DE DESIGN E JUSTIFICATIVAS COMPILADAS

| Decisao | Justificativa Neurofisiologica/Tecnica |
|---------|---------------------------------------|
| Conv2D temporal (1, 125) | Captura ~5 ciclos de Alpha (10 Hz), suficiente para discriminacao |
| DepthwiseConv2D (22, 1) | Combinacao espacial eficiente, analogia com CSP |
| depth_multiplier=2 | 2 visoes espaciais por filtro temporal, mais diversidade |
| AveragePooling | Preserva energia oscilatoria, nao picos de artefatos |
| ELU | Preserva gradientes para componentes negativos do EEG |
| max_norm(1.0) | Previne explosao de pesos espaciais com poucos dados |
| max_norm(0.25) no Dense | Forca decisao distribuida entre features |
| Dropout 0.5 | Regularizacao agressiva necessaria com 288 trials |
| BatchNorm axis=1 | Estabiliza escalas entre filtros — CRITICO (sem ele, acc cai para 33%) |
| Z-score por epoch | Remove dependencia de amplitude absoluta |
| Noise augmentation std=0.1 | Simula variabilidade natural do EEG sem destruir ERD |
| Banda 4-40 Hz | Remove artefatos lentos e EMG/ruido de rede |
| SE Block (variante propria) | Pondera relevancia dos filtros espaciais por trial |

---

## 16. METRICAS E COMO INTERPRETAR

### 16.1 Cohen's Kappa
- Metrica padrao do BCI Competition
- Desconta acerto por acaso: kappa=0 e chance, kappa=1 e perfeito
- Para 2 classes: chance level 50%, kappa=0.778 indica poder discriminativo forte
- Para 4 classes: chance level 25%, kappa=0.736 indica poder discriminativo comparavel

### 16.2 Accuracy vs Kappa
- Accuracy de 88.9% em 2 classes vs 80.2% em 4 classes parece grande
- Mas kappas proximos (0.778 vs 0.736) mostram que o poder real e similar
- Accuracy e enganosa quando chance levels sao diferentes

### 16.3 R2 nas Probabilidades Softmax
- Mede quanto da variancia nas predicoes e explicada pelo modelo
- 0.674 (2 classes): modelo acerta mas nem sempre com conviccao total
- Complementa accuracy — revela confianca real

### 16.4 Precision vs Recall em BCI
- Para jogos: **Precision** importa mais — errar ativamente e pior que nao responder
- Para reabilitacao: **Recall** importa mais — nao detectar intencao de movimento e pior
- No projeto: equilibrado (precision=0.89, recall=0.89 para 2 classes)

---

## 17. CRONOLOGIA DO APRENDIZADO

### 17.1 Pre-requisito: DeepLearningStudy
Notebook completo com 10 exercicios: Dense, LSTM, GRU, Conv1D, Dropout, BatchNorm, Attention, hibridos, multi-step, arquitetura propria.

### 17.2 Notebook 1: eeg_fundamentos_e_arquitetura.ipynb (8 exercicios)
1. **Sinal EEG:** tensor, canais, ERD — entender que informacao esta nas frequencias, nao amplitude bruta
2. **Bandas de frequencia:** Alpha, Beta, PSD — visualizar ERD contralateral
3. **Tensor 2D:** espaco + tempo, Conv2D vs Conv1D — por que tratar como imagem
4. **EEGNet desmontado:** cada camada com shapes anotados — entender fluxo de dados
5. **Ablation study:** sem BN (colapso), MaxPool vs AvgPool
6. **Filtros aprendidos:** visualizar filtros temporais (resposta em frequencia) e espaciais (pesos por canal)
7. **Fine-tuning entre sujeitos:** domain shift, congelar temporal + adaptar espacial
8. **Variante propria (SE Block):** 79.3% vs 70.7% original — +8.6 pontos

### 17.3 Notebook 2: eegnet_motor_imagery.ipynb (Pipeline completo)
- Pipeline end-to-end: carregamento, preprocessing, EEGNet, treino, avaliacao
- Within-subject (Sujeito 1): 4 classes, resultado baseline

### 17.4 Notebook 3: eegnet_mi_improvements.ipynb (Melhorias)
1. Baseline 2 classes: 50.7% (chance level)
2. Ajuste de banda 8-30 Hz: sem melhora (rede ja filtra)
3. **Gaussian noise augmentation: 88.9%** — divisor de aguas
4. 4 classes + augmentation: 80.2%
5. Report comparativo com R2, kappa, confianca softmax
6. Sliding window (planejado, nao finalizado)
7. Pipeline completo (planejado, nao finalizado)

---

## 18. FIGURAS GERADAS (pasta linkedin_post/)

1. `01_confusion_matrix_baseline_2class.png` — Matriz de confusao baseline 2 classes
2. `02_confusion_matrix_8-30Hz.png` — Matriz de confusao com banda 8-30 Hz
3. `03_confusion_matrix_noise_aug_2class.png` — Matriz de confusao com noise augmentation
4. `04_confusion_matrices_4c_vs_2c.png` — Comparacao lado a lado 4 classes vs 2 classes
5. `05_barras_comparativas_metricas.png` — Barras de Accuracy, R2, Kappa, F1
6. `06_distribuicao_confianca_softmax.png` — Histograma de confianca (acertos vs erros)

---

## 19. TRABALHO FUTURO / NAO FINALIZADO

1. **Sliding window augmentation:** implementacao da funcao mas nao treino
2. **Pipeline completo combinando todas as tecnicas:** planejado mas nao executado
3. **Export do modelo para uso online:** codigo de salvamento pronto mas nao executado
4. **Integracao EEGNet -> TCP -> Jogo:** arquitetura pronta, falta conectar
5. **Multi-subject training:** treinar com todos os 9 sujeitos
6. **Ensemble de modelos:** diferentes augmentations
7. **Threshold de confianca online:** so enviar comando se softmax > 0.7

---

## 20. RESPOSTAS DO PEDRO AOS EXERCICIOS (demonstracao de entendimento)

### Exercicio 1 — Por que 22 canais mudam as coisas
"Existe um assigning de peso que sera descoberto pela rede para dizer quais os canais mais importantes a serem ouvidos."

### Exercicio 1 — Qual regiao para mao esquerda
"No C4, pois e no cortex motor que se encontra a area do cerebro responsavel pelo movimento, e especificamente na regiao do centro e direita, afinal o cerebro e um sistema contralateralizado."

### Exercicio 2 — ERD no espectro
"Left Hand mantem maior potencia em C3 e menor em C4. Right Hand mantem maior potencia em C4 e menor em C3. A logica e contralateral: imaginar mao direita -> ERD no hemisferio esquerdo (C3 cai)."

### Exercicio 3 — Por que Conv2D e nao Conv1D
"O Conv1D mistura todos os 22 canais num unico conglomerado. A informacao relevante nao esta em cada canal isolado, mas na combinacao deles."

### Exercicio 4 — Por que AveragePooling
"MaxPooling pegaria artefatos biologicos. O EEG carrega informacao na energia das oscilacoes, nao em picos isolados."

### Exercicio 4 — Por que max_norm
"Com 288 trials, os pesos podem explodir — um unico eletrodo dominaria com peso absurdo. max_norm(1.0) forca os pesos a ficarem numa esfera de raio 1."

---

## 21. RESUMO EXECUTIVO PARA O TCC

Um estudante sem experiencia previa em redes neurais completou uma trajetoria de aprendizado que vai desde fundamentos de Deep Learning (Dense, LSTM, Conv1D, Attention) ate a implementacao e otimizacao de um sistema BCI completo usando EEGNet para classificacao de Motor Imagery.

**Resultado principal:** Accuracy de **88.9%** (kappa=0.778) em classificacao binaria (left hand vs right hand) e **80.2%** (kappa=0.736) em 4 classes, usando o BCI Competition IV Dataset 2a.

**Contribuicao original:** Variante do EEGNet com SE Block (Squeeze-and-Excitation) que superou o original em +8.6 pontos percentuais (79.3% vs 70.7%) no cenario de 4 classes.

**Tecnica chave:** Gaussian Noise Augmentation foi o divisor de aguas, elevando accuracy de 50.7% (chance) para 88.9% no cenario de 2 classes, multiplicando o dataset de 144 para 864 trials.

**Stack:** Python, MNE, TensorFlow/Keras, scikit-learn.

**Aplicacao pratica:** Prototipo de jogo controlado por BCI via TCP, com analise de confianca do softmax para threshold online.
