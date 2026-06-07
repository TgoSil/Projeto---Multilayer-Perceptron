from Camada import Camada
from Logger import Logger
import numpy as np

##################################################################################################################
#  Classe Gerenciador                                                                                            #
#                                                                                                                #
# - A classe Gerenciador é responsável por gerenciar a criação e o funcionamento da rede neural, coordenando     #
#   as operações entre as diferentes camadas e controlando o processo de treinamento, teste e validação.         #
#                                                                                                                #
# - É na classe gerenciador que as funções FeedFoward, BackPropagation e Update são chamadas, na ordem correta,  #
# para realizar o treinamento da rede neural.                                                                    #
#                                                                                                                #
##################################################################################################################

class Gerenciador:

    # Construtor
    # No construtor do Gerenciador são inicializadas as variáveis de controle da rede, os hiperparâmetros, a lista de camadas,
    # as entradas e saídas, além de criar um objeto da classe Logger para realizar os logs do processo.

    # def __init__ (float taxaAprendizado, list entradas, list saidas, float tolerancia, int paciencia)
    # | taxaAprendizado: Um número flutuante que representa o hiperparâmetro que controla a velocidade com que a rede neural 
    # | ajusta seus pesos durante o treinamento.
    # | entradas: Uma lista de listas, onde cada sublista representa um conjunto de entradas para a rede neural. Essas entradas
    # | são usadas para treinar a rede e fazer previsões.
    # | saidas: Uma lista de listas, onde cada sublista representa um conjunto de saídas esperadas correspondentes às entradas.
    # | tolerancia: Um número flutuante que define a tolerância para o critério de parada do treinamento.
    # | paciencia: Um número inteiro que define o número de épocas consecutivas sem melhoria mínima no 
    # | erro para acionar o critério de parada do treinamento.
    # | epocas_executadas: Um número inteiro que representa o número de épocas realizadas durante o treinamento, inicializado como 0 e atualizado ao longo do processo.
    def __init__(self, taxaAprendizado:float, entradas:list, saidas:list, tolerancia:float, paciencia:int):
        self.logger = Logger()
        self.taxaAprendizado = taxaAprendizado
        self.entradas = np.insert(entradas, 0, 1, axis=1)
        self.targets = saidas
        self.camadas = []
        self.tol = tolerancia
        self.qntEpocasSemErro = paciencia
        self.cont = 0
        self.menorErro = float('inf')
        self.epocas_executadas = 0 

    # Criação de camadas
    # A função criaCamada é responsável por criar uma nova camada na rede neural, utilizando a classe Camada para isso.
    # Ela recebe como parâmetro a quantidade de neurônios desejada para a nova camada e, dependendo se é a primeira camada ou não, 
    # define a quantidade de entradas para os neurônios da nova camada, que será igual à quantidade de neurônios da última camada 
    # criada mais um (para o bias). A nova camada é então adicionada à lista de camadas do Gerenciador.

    # def criaCamada (int qtdNeurons)
    # | qtdNeurons: Um número inteiro que representa a quantidade de neurônios desejada para a nova camada a ser criada.

    def criaCamada(self, qtdNeurons:int):
        if len(self.camadas) > 0:
            self.camadas.append(Camada(qtdNeurons, (self.camadas[-1].camada+1)))
        else:
            self.camadas.append(Camada(qtdNeurons, len(self.entradas[0])))

    # Exibição da rede
    # A função printaRede é responsável por exibir a estrutura da rede neural, mostrando a quantidade de neurônios em cada camada.
    # Ela percorre a lista de camadas do Gerenciador e imprime a quantidade de neurônios de cada camada, facilitando a visualização 
    # da arquitetura da rede.

    def printaRede(self):
        for i, camada in enumerate(self.camadas):
            print(f"{i+1}ª camada: {camada.camada} neurônios")

    # Iniciação da rede
    # A função iniciaRede é responsável por iniciar a rede neural, criando as camadas necessárias de acordo com a quantidade de camadas
    # desejada. Ela recebe como parâmetro a quantidade de camadas que a rede deve ter e, utilizando a função criaCamada, cria a primeira
    # camada oculta com a quantidade de neurônios igual à quantidade de entradas menos um (para o bias), as camadas ocultas seguintes
    # com uma quantidade fixa de neurônios (150) e a camada de saída com a quantidade de neurônios igual à quantidade de saídas.

    # def iniciaRede (int qtdCamadas)
    # | qtdCamadas: Um número inteiro que representa a quantidade total de camadas que a rede neural deve ter.
    # | qtdNeuronsCamadaOculta: Um número inteiro que representa a quantidade de neurônios na camada oculta.
    # | return: Um valor booleano que indica se a rede foi iniciada com sucesso (True) ou se houve um erro devido à quantidade de camadas 
    # | ser menor que 2 (False).

    def iniciaRede(self, qtdCamadas:int, qtdNeuronsCamadaOculta:int):
        if (qtdCamadas < 2): 
            print("VALORES MENORES QUE 2 NÃO PERMITIDOS!!\n(Não é possível criar uma rede sem pelo menos uma camada oculta e uma de saída)")
            return False
        self.criaCamada(qtdNeuronsCamadaOculta) #Cria primeira camada oculta
        for i in range(qtdCamadas-2):
            # self.criaCamada(random.randint(2, 10)) #Cria camada oculta com numero aleatoria de neuronios
            self.criaCamada(qtdNeuronsCamadaOculta) #Cria camada oculta com 150 neuronios
        self.criaCamada(len(self.targets[0]))  #Cria camada de saida
        return True

    # Parada Antecipada (Early Stopping)
    # A função earlyStopping é responsável por verificar se o processo de treinamento deve ser interrompido antecipadamente 
    # com base no erro da época atual. Ela compara o erro atual com o menor erro registrado até o momento e verifica se a melhoria 
    # é maior do que a tolerância definida. Se a melhoria for menor ou igual à tolerância, o contador de épocas sem melhoria é 
    # incrementado. Se o erro atual for menor do que o menor erro registrado, o menor erro é atualizado. A função retorna True se o
    # contador de épocas sem melhoria atingir ou ultrapassar a quantidade definida (paciência), indicando que o treinamento deve ser 
    # interrompido, ou False caso contrário.
    # A parada antecipada é uma técnica utilizada para evitar o overfitting, interrompendo o treinamento quando o modelo começa a 
    # apresentar um desempenho pior em dados de validação, mesmo que continue melhorando nos dados de treinamento.

    # def earlyStopping (float erro)
    # | erro: Um número flutuante que representa o erro da época atual, utilizando o MSE do conjunto de validação.
    # | return: Um valor booleano que indica se o treinamento deve ser interrompido antecipadamente (True) ou se deve continuar (False).

    def earlyStopping(self, erro:float): 
        delta = self.menorErro - erro
        self.cont = 0 if delta > self.tol else self.cont + 1 # número de epocas seguidas sem melhora minima
        if erro < self.menorErro: self.menorErro = erro
        return self.cont >= self.qntEpocasSemErro

    # Matriz de confusão
    # A função geraMatrizDeConfusao é responsável por gerar a matriz de confusão com base nas saídas da rede neural 
    # e nas saídas esperadas (gabarito). Ela lê as saídas da rede a partir de um arquivo, compara cada resposta com 
    # o gabarito correspondente e preenche a matriz de confusão de acordo com as classes previstas e esperadas.
    # A matriz de confusão é uma ferramenta importante para avaliar o desempenho de um modelo de classificação, 
    # permitindo visualizar os acertos e erros do modelo em relação às diferentes classes.

    # def geraMatrizDeConfusao (str saidas, list gabarito)
    # | saidas: Uma string que representa o caminho para o arquivo que contém as saídas da rede neural, 
    # | onde cada linha corresponde a uma saída para um conjunto de entradas.
    # | gabarito: Uma lista de listas, onde cada sublista representa um conjunto de saídas esperadas correspondentes às entradas.
    # | return: Uma matriz de confusão, que é uma matriz bidimensional onde as linhas representam as classes esperadas e as colunas 
    # | representam as classes previstas pela rede neural.
     
    def geraMatrizDeConfusao(self, saidas, gabarito):
        
        matriz_confusao = np.zeros((26,27))
        matriz_saidas = np.loadtxt(saidas, dtype=float)

        for resposta, gabarito in zip(matriz_saidas, gabarito):
            classe_predita = np.argmax(resposta)
            classe_esperada = np.argmax(gabarito)
            if (resposta[classe_predita] < 0):
                matriz_confusao[classe_esperada][26] += 1
            else:
                matriz_confusao[classe_esperada][classe_predita] += 1
        
        return matriz_confusao

    # Avaliação de métricas
    # As funções avaliaAcuracia, avaliaRecall, avaliaPrecisao e avaliaF1Score são responsáveis por calcular as métricas 
    # de avaliação do modelo com base na matriz de confusão gerada.
    # - A função avaliaAcuracia calcula a acurácia do modelo, que é a proporção de acertos em relação ao total de amostras.
    # - A função avaliaRecall calcula o recall para cada classe, que é a proporção de acertos em relação ao total de amostras 
    # da classe esperada.
    # - A função avaliaPrecisao calcula a precisão para cada classe, que é a proporção de acertos em relação ao total de amostras.
    # - A função avaliaF1Score calcula o F1 Score para cada classe, que é a média harmônica entre precisão e recall, fornecendo uma
    # métrica balanceada para avaliar o desempenho do modelo, especialmente em casos de classes desbalanceadas.

    # def avaliaAcuracia (matriz_confusao)
    # | matriz_confusao: Uma matriz de confusão, que é uma matriz bidimensional onde as linhas representam as classes 
    # | esperadas e as colunas representam as classes previstas pela rede neural.
    # | return: Um número flutuante que representa a acurácia do modelo, calculada como a proporção de acertos em relação ao 
    # | total de amostras.

    def avaliaAcuracia(self, matriz_confusao):
        total_valores = np.sum(matriz_confusao)
        total_acertos = np.trace(matriz_confusao)
        return total_acertos / total_valores
    
    # def avaliaRecall (matriz_confusao)
    # | matriz_confusao: Uma matriz de confusão, que é uma matriz bidimensional onde as linhas representam as classes
    # | esperadas e as colunas representam as classes previstas pela rede neural.
    # | return: Um array de números flutuantes que representa o recall para cada classe, calculado como a proporção de acertos
    # | em relação ao total de amostras da classe esperada.

    def avaliaRecall(self, matriz_confusao):
        resultado_recall = np.empty(len(matriz_confusao))
        for i, linha in enumerate(matriz_confusao):
            total_classe = np.sum(linha)
            resultado_recall[i] = linha[i] / total_classe
        return resultado_recall
    
    # def avaliaPrecisao (matriz_confusao)
    # | matriz_confusao: Uma matriz de confusão, que é uma matriz bidimensional onde as linhas representam as classes
    # | esperadas e as colunas representam as classes previstas pela rede neural.
    # | return: Um array de números flutuantes que representa a precisão para cada classe, calculado como a proporção de acertos
    # | em relação ao total de amostras da classe prevista.

    def avaliaPrecisao(self, matriz_confusao):
        resultado_precisao = np.zeros(matriz_confusao.shape[0])
        soma_colunas = np.sum(matriz_confusao, axis=0)
        for i in range(len(matriz_confusao)):
            if (soma_colunas[i] == 0): soma_colunas[i] = 1
            precisao_classe = matriz_confusao[i][i] / soma_colunas[i]
            resultado_precisao[i] = precisao_classe
        return resultado_precisao
    
    # def avaliaF1Score (precisoes, recalls)
    # | precisoes: Um array de números flutuantes que representa a precisão para cada classe, calculado como a proporção de acertos
    # | em relação ao total de amostras da classe prevista.
    # | recalls: Um array de números flutuantes que representa o recall para cada classe, calculado como a proporção de acertos
    # | em relação ao total de amostras da classe esperada.
    # | return: Um array de números flutuantes que representa o F1 Score para cada classe, calculado como a média harmônica entre 
    # | precisão e recall.

    def avaliaF1Score(self, precisoes, recalls):
        denominador = (precisoes + recalls)
        resultado_F1 = np.divide(2 * (precisoes * recalls), denominador, out=np.zeros_like(precisoes), where=denominador!=0)
        return resultado_F1

    # Avaliação completa
    # A função avalicaoCompleta é responsável por realizar uma avaliação completa do modelo, calculando a matriz de confusão 
    # e as métricas de avaliação (acurácia, precisão, recall e F1 Score) com base nas saídas da rede neural e no gabarito.
    # Ela chama as funções geraMatrizDeConfusao, avaliaAcuracia, avaliaRecall, avaliaPrecisao e avaliaF1Score para realizar 
    # os cálculos necessários e, em seguida, utiliza o objeto Logger para criar um log simples com os resultados da avaliação.
    # Essa avaliação completa é essencial para entender o desempenho do modelo em diferentes aspectos, fornecendo uma visão detalhada 
    # dos acertos e erros do modelo em relação às classes previstas e esperadas.

    # def avalicaoCompleta (str saidas, list gabarito)
    # | saidas: Uma string que representa o caminho para o arquivo que contém as saídas da rede neural, 
    # | onde cada linha corresponde a uma saída para um conjunto de entradas.
    # | gabarito: Uma lista de listas, onde cada sublista representa um conjunto de saídas esperadas correspondentes às entradas.
    
    def avalicaoCompleta(self, saidas, gabarito):
        m = self.geraMatrizDeConfusao(saidas, gabarito)
        a = self.avaliaAcuracia(m)
        r = self.avaliaRecall(m)
        p = self.avaliaPrecisao(m)
        f1 = self.avaliaF1Score(p, r)
        self.logger.criaLogSimples(a, p, r, f1, m)

    # Treinamento da rede
    # A função MLP_treinamento é responsável por realizar o processo de treinamento da rede neural, utilizando o algoritmo de 
    # retropropagação (backpropagation) para ajustar os pesos das camadas com base nos erros calculados.
    # Ela recebe como parâmetros o número de épocas para o treinamento e as entradas e saídas de validação. O processo de 
    # treinamento envolve a realização de feedforward para calcular as saídas da rede, o cálculo dos deltas para cada camada
    # utilizando backpropagation, a atualização dos pesos das camadas com base nos deltas e na taxa de aprendizado, e a validação do
    # modelo a cada época utilizando o conjunto de validação. O processo de treinamento é interrompido antecipadamente se o critério 
    # de parada definido pela função earlyStopping for atendido, evitando o overfitting e garantindo um modelo mais generalizável.

    # def MLP_treinamento (int numEpocas, list entradasValidacao, list targetsValidacao)
    # | numEpocas: Um número inteiro que representa o número máximo de épocas para o treinamento da rede neural.
    # | entradasValidacao: Uma lista de listas, onde cada sublista representa um conjunto de entradas para o conjunto de validação.
    # | targetsValidacao: Uma lista de listas, onde cada sublista representa um conjunto de saídas esperadas correspondentes às entradas
    # | de validação.

    def MLP_treinamento(self, numEpocas:int, entradasValidacao:list, targetsValidacao:list):
        i = 0
        entradasValidacao = np.insert(entradasValidacao, 0, 1, axis=1) #Coloca o Bias
        self.logger.logPesos(self.camadas,  "log/pesos_iniciais.txt")
        self.logger.abrirLogErro() # Abre arquivo de log de erro
        for epoca in range(numEpocas): #Roda por um número definido de épocas
            saidasFinal = []
            for linha_entrada, linha_saida  in zip(self.entradas, self.targets):
                saidasCamadas = [] # Cada linha é uma camada e cada coluna é a resposta de um neurônio
                deltasCamadas = [] # Cada linha é uma camada e cada coluna é o delta de um neurônio

                ## Inicia FeedFoward e armazena as saidas de cada camada em um array
                saidasCamadas.append(self.camadas[0].camadaFeedFoward(linha_entrada)) # Feedforward na primeira camada

                for i_camada in range(1, len(self.camadas)): # Feedfoward nas outras camadas
                    saidasCamadas.append(self.camadas[i_camada].camadaFeedFoward(saidasCamadas[i_camada-1]))  # Realiza feedfoward com os resultados da camada anterior

                deltasCamadas.insert(0, # Armazena deltas da camada de output
                    self.camadas[-1].camadaOutputBackPropagation(
                        saidasCamadas[-1], linha_saida
                    )
                )
    
                for i_deltas in range(len(self.camadas) -2, -1, -1): # Armazena deltas das outras camadas
                    deltasCamadas.insert(0, 
                        self.camadas[i_deltas].camadaBackPropagation(
                            deltasCamadas[0], np.array(self.camadas[i_deltas+1].pesos).T
                        )
                    )
                
                self.camadas[0].camadaUpdate(deltasCamadas[0], linha_entrada, self.taxaAprendizado)
                for i in range(1, len(self.camadas)):
                    self.camadas[i].camadaUpdate(deltasCamadas[i], saidasCamadas[i-1], self.taxaAprendizado)
                saidasFinal.append(saidasCamadas[len(saidasCamadas)-1]) # Pega valores da ultima camada.

            # Calcula o MSE de Treino
            erro_quadratico_treino = np.sum(np.square(self.targets - np.array(saidasFinal)[:, 1:]))
            MSE_treino = erro_quadratico_treino / len(self.entradas)

            # Calcula o MSE de Validação 
            MSE_validacao = self.MLP_Validacao(entradasValidacao, targetsValidacao) # Validação da época
            print(f"ÉPOCA {epoca + 1} | MSE Treino: {MSE_treino:.6f} | MSE Validação: {MSE_validacao:.6f} | ME: {(self.menorErro-MSE_validacao):.6f}") # Exibe que época está 

            self.logger.logErroEpoca(epoca + 1, MSE_treino, MSE_validacao) # Log do erro da época, tanto de treino quanto de validação
            self.epocas_executadas = epoca + 1 # Atualiza o número de épocas executadas

            if self.earlyStopping(MSE_validacao):
                print("Early Stopping")
                break # Verificação se o erro da época ultrapassa a tolerância

        self.logger.abrirLogSaidas("log/saidas_treinamento.txt")
        for count in range(len(saidasFinal)):
            self.logger.logSaidas(saidasFinal[count][1:], "log/saidas_treinamento.txt")
        self.logger.logPesos(self.camadas, "log/pesos_finais.txt")


    # Validação da rede
    # A função MLP_Validacao é responsável por realizar a validação da rede neural, calculando o erro quadrático médio (MSE) 
    # com base nas saídas da rede e nas saídas esperadas (targets). Ela recebe como parâmetros as entradas e as saídas de validação, 
    # realiza o processo de feedforward para calcular as saídas da rede com base nas entradas de validação, e em seguida calcula o MSE 
    # comparando as saídas da rede com as saídas esperadas. 
    # O MSE é uma métrica importante para avaliar o desempenho do modelo durante o processo de treinamento, permitindo monitorar a 
    # evolução do erro e ajustar os hiperparâmetros ou interromper o treinamento antecipadamente se o modelo começar a apresentar 
    # um desempenho pior em dados de validação.

    # def MLP_Validacao (list entradas, list saidas)
    # | entradas: Uma lista de listas, onde cada sublista representa um conjunto de entradas para o conjunto de validação.
    # | saidas: Uma lista de listas, onde cada sublista representa um conjunto de saídas esperadas correspondentes às entradas 
    # | de validação.
    # | return: Um número flutuante que representa o erro quadrático médio (MSE).

    def MLP_Validacao(self, entradas, saidas):
        saidasFinal = []

        for linha_entrada, linha_saida in zip(entradas, saidas):
            saidasCamadas = []
            saidasCamadas.append(self.camadas[0].camadaFeedFoward(linha_entrada)) #Feedfoward na primeira camada
            
            for i_camada in range(1, len(self.camadas)): # Feedfoward nas outras camadas
                    saidasCamadas.append(self.camadas[i_camada].camadaFeedFoward(saidasCamadas[i_camada-1]))

            saidasFinal.append(saidasCamadas[len(saidasCamadas)-1])

        erro_quadratico_total = np.sum(np.square(saidas - np.array(saidasFinal)[:, 1:])) 
        N_amostra = len(entradas)

        return erro_quadratico_total / N_amostra # Calculo do MSE

    # Execução da rede
    # A função MLP_execucao é responsável por executar a rede neural, utilizando as entradas de teste para calcular 
    # as saídas da rede e compará-las com as saídas esperadas (targets). Ela recebe como parâmetros as entradas e 
    # as saídas de teste, realiza o processo de feedforward para calcular as saídas da rede com base nas entradas de teste, 
    # e em seguida armazena as saídas calculadas. As saídas da rede são então escritas em um arquivo de log utilizando o 
    # objeto Logger, permitindo a análise dos resultados do teste e a comparação com as saídas esperadas para avaliar o 
    # desempenho do modelo em dados não vistos durante o treinamento.
    
    # def MLP_execucao (list entradas, list saidas)
    # | entradas: Uma lista de listas, onde cada sublista representa um conjunto de entradas para o conjunto de teste.
    # | saidas: Uma lista de listas, onde cada sublista representa um conjunto de saídas esperadas correspondentes às entradas de teste.
    
    def MLP_execucao(self, entradas, saidas:list):
        saidasFinal = []
        entradas = np.insert(entradas, 0, 1, axis=1) #Coloca o Bias
        for linha_entrada, linha_saida in zip(entradas, saidas):
            saidasCamadas = []
            saidasCamadas.append(self.camadas[0].camadaFeedFoward(linha_entrada)) #Feedfoward na primeira camada
            
            for i_camada in range(1, len(self.camadas)): # Feedfoward nas outras camadas
                    saidasCamadas.append(self.camadas[i_camada].camadaFeedFoward(saidasCamadas[i_camada-1]))
                    
            saidasFinal.append(saidasCamadas[len(saidasCamadas)-1])

        self.logger.abrirLogSaidas("log/saidas_teste.txt")
        for count in range(len(saidasFinal)):
            self.logger.logSaidas(saidasFinal[count][1:], "log/saidas_teste.txt") # Escreve as saidas
       
