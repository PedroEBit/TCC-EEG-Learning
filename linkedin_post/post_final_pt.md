<!--
Versao em portugues, para publicar ~1 semana depois da inglesa.
Todo numero aqui sai de results_final.json ou da analise de pesos.
-->

Meu classificador de EEG tinha aprendido a chutar "mão direita" sempre.

144 trials de treino, duas classes, e o modelo parou em 50.7% de acurácia — Cohen's κ de 0.01. Ele não estava confuso. Tinha encontrado a estratégia mais preguiçosa possível: responder sempre a mesma classe e acertar metade. Respondeu "direita" em 133 dos 144 trials.

A tarefa: classificar imaginação motora a partir de EEG. Quando você imagina mover a mão esquerda, o ritmo mu (8–13 Hz) dessincroniza sobre o córtex motor contralateral. Essa modulação existe, e está enterrada em ruído. Eu tinha 144 trials para encontrá-la, com uma rede de 4 mil parâmetros — a acurácia de treino subiu a 84% enquanto a validação ficava em 45%.

**O que resolveu**

Não foi rede maior. Foi data augmentation por injeção de ruído gaussiano — ruído aditivo escalado pelo desvio padrão de cada época, em σ = 0.1, simulando a variabilidade natural do EEG de fundo entre trials. O sinal de ERD/ERS é uma modulação lenta na potência de mu/beta, então ele sobrevive a um ruído aditivo que só destrói o fundo.

Cinco cópias ruidosas por trial. O augmentation toca apenas o split de treino.

Sobre vazamento, porque é a primeira coisa que vale perguntar: não existe split aleatório aqui. O BCI Competition IV-2a entrega cada sujeito em duas sessões gravadas em dias diferentes — eu treino na sessão T e testo na sessão E. O augmentation é estruturalmente incapaz de cruzar essa fronteira.

Aí eu rodei cinco vezes com sementes diferentes, e o resultado desmontou.

Acurácia de 2 classes nas cinco sementes: 88.2%, 88.2%, 92.4%, **57.6%**, 86.8%. Mesmos dados, mesma arquitetura, mesmos hiperparâmetros. Uma execução em cinco simplesmente não converge. Qualquer número único desse pipeline — inclusive os 88.9% que eu vinha citando — é um sorteio, não uma medição.

**O que eu fiz a respeito**

Parei de escolher um modelo. Passei a fazer a média das saídas softmax de todas as sementes. E o ponto decisivo: cada semente sorteia também a própria partição treino/validação, então cada membro viu 116 dos 144 trials, um subconjunto diferente — e eles erram em trials diferentes. A discordância média entre pares é 0.23. É essa descorrelação que faz a média funcionar.

Resultados na sessão de teste, within-subject, sujeito A01:
· 2 classes (mão esquerda vs. direita): **91.0% de acurácia, κ = 0.819**
· 4 classes (esquerda, direita, pés, língua): **78.5%, κ = 0.713**

O ensemble de 4 classes supera *todos* os membros individuais — 78.5% contra 75.7% da melhor semente — e o teste de McNemar contra o membro mediano dá p = 0.023. O de 2 classes chega a 91.0%, acima do membro mediano, mas ali p = 0.29: a melhora aponta na direção certa e 144 trials de teste não conseguem prová-la. Reporto os dois p-valores, inclusive o que falha.

Nenhum membro foi selecionado ou descartado olhando o teste. A semente que colapsou continua lá dentro, diluída em vez de apagada.

Reporto κ porque é o padrão do BCI Competition — ele desconta o acerto por acaso. E reporto o sujeito, porque são resultados within-subject num dos sujeitos mais fáceis do dataset. Não são comparáveis às médias de nove sujeitos que costumam ser citadas nesse benchmark.

**O bug que eu escrevi e tive que corrigir**

O `validation_split=0.2` do Keras reserva os últimos 20% do array **sem embaralhar**. Meu array aumentado era `[originais, cópia1, ..., cópia5]` — ou seja, a validação era feita em cópias ruidosas de trials que já estavam no treino. A acurácia de validação marcava 1.000. Os números de teste nunca foram afetados, já que o teste é uma sessão fisicamente separada, mas o early stopping estava escolhendo época por um sinal sem significado.

Separar a validação **antes** de aumentar custa 20% dos dados de treino, e custou acurácia também: em execução única, o 4 classes caiu de 80.2% para 73.2%. Aquele número estava escorado justamente no early stopping quebrado, que deixava o modelo treinar muito além de onde uma validação honesta teria parado.

Ver um número piorar porque você consertou algo é o sinal mais útil que existe. E foi o que tornou o ensemble necessário, em vez de opcional.

**Por que as camadas da EEGNet mapeiam neurofisiologia**

A arquitetura (Lawhern et al., 2018) não é uma CNN genérica, e os pesos mostram isso.

A convolução temporal aprende filtros de frequência. Eu dei a ela a banda inteira de 4–40 Hz e fui olhar no que ela convergiu. No modelo de 4 classes, 7 dos 8 filtros aprendidos têm pico entre 10.7 e 17.6 Hz — a faixa mu/beta — e a energia em theta fica abaixo de 1% em todos eles, sem exceção. Ninguém disse a ela qual banda importava.

No modelo de 2 classes, treinado com metade dos trials, só 4 dos 8 fazem isso. Os outros quatro degeneram para picos abaixo de 4 Hz, fora da banda de entrada, onde não existe sinal para responder. Metade dos dados, metade dos filtros que aprendem alguma coisa. Esse contraste é a imagem mais clara que eu tenho do que 144 trials custam de verdade.

A convolução espacial depthwise aprende combinações de canais — e aqui a versão de livro-texto não sobreviveu ao contato com os pesos. Pesos de depthwise são **filtros**, não **padrões**, então lê-los como topografia é inválido; é preciso convertê-los antes em padrões de ativação (Haufe et al., 2014). Feito isso, os canais sensorimotores carregam 44–47% da massa do padrão — *abaixo* dos 50% que se obteria por acaso. C3 e C4 ficam por volta do 13º lugar entre 22. A lateralização que existe está em CP3/CP4, alguns centímetros posterior.

Eu tinha escrito que a rede "redescobre que C3 e C4 carregam o sinal discriminativo". Ela não redescobre. Conferi nos três modelos treinados e o resultado negativo se repetiu em todos.

Retirar essa afirmação é a parte que eu não saberia fazer seis meses atrás.

**O achado que eu não esperava**

A confiança do softmax separa por acerto: 87.8% de confiança média nos acertos contra 71.0% nos erros. Esse gap não é curiosidade — é um threshold de rejeição utilizável. Num BCI online, recusar agir sobre uma predição incerta é melhor que agir sobre uma errada. Em controle de jogo, precisão importa mais que revocação: ir para o lado errado é pior que não responder.

Próximos passos: sliding window augmentation para cortar pela metade a janela de decisão de 4s, e integração online via TCP para controlar um jogo em tempo real.

Há alguns meses eu não sabia o que era uma convolução. O número de acurácia não é do que eu me orgulho — é de conseguir dizer quais das minhas afirmações os pesos sustentam, e quais eu tive que retirar.

Dataset: BCI Competition IV-2a (Graz) · Arquitetura: EEGNet · Stack: Python, MNE, TensorFlow/Keras
Código: github.com/PedroEBit/TCC-EEG-Learning

Se você já trabalhou com decodificação de imaginação motora — o que mais fez diferença no seu caso com poucos trials?

#BCI #EEG #DeepLearning #MachineLearning #Neurociência
