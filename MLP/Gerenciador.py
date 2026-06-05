from Camada import Camada
import numpy as np

##################################################################################################################
#  Classe Gerenciador                                                                                            #
#                                                                                                                #
# - A classe Camada é onde estão implementadas as funções que realizam os cálculos matemáticos principais,       #
# são esses os calculos de y_in, deltas, entre outras variáveis importantes para o processamento dos neurônios.  #
#                                                                                                                #
# - Para cálculos como estes é na classe camada que está presente a matriz de pesos referente ao seus neurônios, #
# dessa forma, os neurônios são representados como as linhas da matriz de pesos presente na camada.              #
#                                                                                                                #
# P.S.: Em diversos cálculos foram utilizadas função do numpy, principalmente em cálculos envolvendo matrizes,   #
# essa decisão foi tomada para otimização do código, mas cada cálculo será exemplificado na função referente.    #
#                                                                                                                #
##################################################################################################################

class Gerenciador:

    def __init__(self, taxaAprendizado:float, entradas, saidas:list, tolerancia:float, paciencia:int):
        self.taxaAprendizado = taxaAprendizado
        self.entradas = np.insert(entradas, 0, 1, axis=1)
        self.targets = saidas
        self.camadas = []
        self.tol = tolerancia
        self.qntEpocasSemErro = paciencia
        self.cont = 0
        self.menorErro = float('inf')

    def criaCamada(self, qtdNeurons:int):
        if len(self.camadas) > 0:
            self.camadas.append(Camada(qtdNeurons, (self.camadas[-1].camada+1))) #Pega quantidade de neuronios da ultima camada e adiciona mais 1 como entrada (por causa do BIAS)
        else:
            self.camadas.append(Camada(qtdNeurons, len(self.entradas[0])))

    def printaRede(self):
        for i, camada in enumerate(self.camadas):
            print(f"{i+1}ª camada: {camada.camada} neurônios")

    def iniciaRede(self, qtdCamadas:int):
        if (qtdCamadas < 2): 
            print("VALORES MENORES QUE 2 NÃO PERMITIDOS!!\n(Não é possível criar uma rede sem pelo menos uma camada oculta e uma de saída)")
            return False
        self.criaCamada(len(self.entradas[0])-1) #Cria primeira camada oculta
        for i in range(qtdCamadas-2):
            # self.criaCamada(random.randint(2, 10)) #Cria camada oculta com numero aleatoria de neuronios
            self.criaCamada(150) #Cria camada oculta com 150 neuronios
        self.criaCamada(len(self.targets[0]))  #Cria camada de saida
        return True
        
    def logIniciais(self):
        with open("log/pesos_iniciais.txt", "w", encoding="utf-8") as pesos_iniciais:
            for i, camada in enumerate(self.camadas):
                # pesos_iniciais.write(f"Camada {i+1}:\n")
                np.savetxt(pesos_iniciais, camada.pesos, fmt='%.6f', delimiter=',')
    
    def abrirLogErro(self):
        with open("log/log_erro.txt", "w", encoding="utf-8") as arquivo_log:
            arquivo_log.write("Epoca,MSE\n") 

    def logErroEpoca(self, epoca: int, mse: float):
        with open("log/log_erro.txt", "a", encoding="utf-8") as arquivo_log:
            arquivo_log.write(f"{epoca},{mse:.6f}\n")
    
    def logFinais(self):
        with open("log/pesos_finais.txt", "w", encoding="utf-8") as pesos_finais:
            for i, camada in enumerate(self.camadas):
                # pesos_finais.write(f"Camada {i+1}:\n")
                np.savetxt(pesos_finais, camada.pesos, fmt='%.6f', delimiter=',')

    def logSaidas(self, saidas:list, nomeArq):
        with open(nomeArq, "a", encoding="utf-8") as saidas_teste:
            np.savetxt(saidas_teste, saidas, fmt='%.6f', delimiter=',', newline=' ')
            saidas_teste.write("\n")

    def earlyStopping(self, erro:float): 
        delta = self.menorErro - erro
        self.cont = 0 if delta > self.tol else self.cont + 1 # número de epocas seguidas sem melhora minima
        if erro < self.menorErro: self.menorErro = erro
        return self.cont >= self.qntEpocasSemErro

    def geraMatrizDeConfusao(self, saidas):
        # Matriz de confusão
        matriz_confusao = np.zeros((26,27))
        matriz_saidas = np.loadtxt(saidas, dtype=float)

        for resposta, gabarito in zip(matriz_saidas, self.targets):
            classe_predita = np.argmax(resposta)
            classe_esperada = np.argmax(gabarito)
            if (resposta[classe_predita] < 0):
                matriz_confusao[classe_esperada][26] += 1
            else:
                matriz_confusao[classe_esperada][classe_predita] += 1
        
        return matriz_confusao

    def avaliaAcuracia(self, matriz_confusao):
        total_valores = np.sum(matriz_confusao)
        total_acertos = np.trace(matriz_confusao)
        return total_acertos / total_valores
    
    def avaliaRecall(self, matriz_confusao):
        resultado_recall = np.empty(len(matriz_confusao))
        for i, linha in enumerate(matriz_confusao):
            total_classe = np.sum(linha)
            resultado_recall[i] = linha[i] / total_classe
        return resultado_recall
    
    def avaliaPrecisao(self, matriz_confusao):
        resultado_precisao = np.zeros(matriz_confusao.shape[0])
        soma_colunas = np.sum(matriz_confusao, axis=0)
        for i in range(len(matriz_confusao)):
            if (soma_colunas[i] == 0): soma_colunas[i] = 1
            precisao_classe = matriz_confusao[i][i] / soma_colunas[i]
            resultado_precisao[i] = precisao_classe
        return resultado_precisao
    
    def avaliaF1Score(self, precisoes, recalls):
        denominador = (precisoes + recalls)
        resultado_F1 = np.divide(2 * (precisoes * recalls), denominador, out=np.zeros_like(precisoes), where=denominador!=0)
        return resultado_F1
    
    def criaLogSimples(self, acuracia, vetor_precisao, vetor_recall, vetor_f1, matriz_confusao):
        import numpy as np

        np.set_printoptions(linewidth=np.inf)

        with open("log/log_avaliacao.txt", "w", encoding="utf-8") as f:
            f.write(f"Acuracia Global: {acuracia}\n\n")

            f.write("Precisao por classe:\n")
            f.write(str(vetor_precisao) + "\n\n")

            f.write("Recall por classe:\n")
            f.write(str(vetor_recall) + "\n\n")

            f.write("F1-Score por classe:\n")
            f.write(str(vetor_f1) + "\n\n")

            f.write("Matriz de Confusao:\n")
            f.write(str(matriz_confusao) + "\n")

        print("Log salvo com sucesso!")

    def avalicaoCompleta(self, saidas):
        m = self.geraMatrizDeConfusao(saidas)
        a = self.avaliaAcuracia(m)
        r = self.avaliaRecall(m)
        p = self.avaliaPrecisao(m)
        f1 = self.avaliaF1Score(p, r)
        self.criaLogSimples(a, p, r, f1, m)

    def criaArrayPesosDoBackPropagation(self, pesosCamadas): #"Virando" - tranposta da matriz de pesos da camada de saida para que seja mais fácil de calcular o deltinha da camada atual
        pesosParaCalcularDelta = []
        for j in range(len(pesosCamadas[0])): 
            pesosParaCalcularDelta.append([])
            for i in range(len(pesosCamadas)):
                pesosParaCalcularDelta[j].append(pesosCamadas[i][j]) 
        return pesosParaCalcularDelta

    def MLP_treinamento(self, numEpocas:int):
        i = 0
        self.logIniciais()
        self.abrirLogErro()# Abri arquivo de log de erro
        for epoca in range(numEpocas): #Roda por um número definido de épocas (condição de parada)
            saidasFinal = []
            erro_quadratico_total = 0.0 # Acumula o erro da época
            for linha_entrada, linha_saida  in zip(self.entradas, self.targets):
                saidasCamadas = [] # Cada linha é uma camada e cada coluna é a resposta de um neurônio
                deltasCamadas = [] # Cada linha é uma camada e cada coluna é o delta de um neurônio

                ## Inicia FeedFoward e armazena as saidas de cada camada em um array
                saidasCamadas.append(self.camadas[0].camadaFeedFoward(linha_entrada)) # Feedforward na primeira camada

                for i_camada in range(1, len(self.camadas)): # Feedfoward nas outras camadas
                    saidasCamadas.append(self.camadas[i_camada].camadaFeedFoward(saidasCamadas[i_camada-1]))  #faz feedfoward com os resultados da camada anterior

                # Calculo do erro da amostra

                saida_rede = saidasCamadas[-1] # Resposta dos neuronios da camada de saída             
                
                if len(saida_rede) > len(linha_saida): # Verifica se a camada retorna o bias. Se sim, ignoramos ele para o calculo do erro
                    saida_rede = saida_rede[1:] 

                erro_amostra = 0.0 # Erro da amostra atual/imagem atual
                for t_k, y_k in zip(linha_saida, saida_rede): # Calcula o erro da amostra atual ou seja da imagem específica
                    erro_amostra += (t_k - y_k) ** 2
                
                erro_quadratico_total += erro_amostra # Erro Quadrático Total (uma vez que não estamos divindo pelo número de amostras ainda!)

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
                    # print(self.camadas[i].pesos)
                saidasFinal.append(saidasCamadas[len(saidasCamadas)-1]) #pega valores da ultima camadas

            N_amostra = len(self.entradas) # Numero de amostras, vai ter que muda dps dependendo do dataset usado (se é de treino, validacao ou teste)
            MSE_epoca = erro_quadratico_total / N_amostra # Calculo do MSE
            self.logErroEpoca(epoca + 1, MSE_epoca) # Log do erro da época

            print(f"ÉPOCA {epoca + 1} | MSE: {MSE_epoca:.6f} | ME: {(self.menorErro-MSE_epoca):.6f}") # imprimi que época está 

            if self.earlyStopping(MSE_epoca):
                print("Early Stopping")
                break # Verificação se o erro da época ultrapassa a tolerância


        with open("log/saidas_teste.txt", "w", encoding="utf-8") as saidas_teste:
            saidas_teste.write("")
        for count in range(len(saidasFinal)):
            self.logSaidas(saidasFinal[count][1:], "log/saidas_treinamento.txt")
        self.logFinais()



    def MLP_teste(self):
        matriz = np.loadtxt('log/pesos_finais.txt', delimiter=',')
        ##print(matriz)
        for camada in self.camadas:
            for i in range(len(camada.pesos)):
                for j in range(len(camada.pesos[0])):
                    camada.pesos[i][j] = matriz[i][j]
        
    #     saidasFinal = []
    #     for linha_entrada in self.entradas:
    #         saidasCamadas = [] # Cada linha é uma camada e cada coluna é a resposta de um neurônio
    #         ## Inicia FeedFoward e armazena as saidas de cada camada em um array
    #         saidasCamadas.append(self.camadas[0].camadaFeedFoward(linha_entrada)) # Feedforward na primeira camada

    #         for i_camada in range(1, len(self.camadas)): # Feedfoward nas outras camadas
    #             saidasCamadas.append(self.camadas[i_camada].camadaFeedFoward(saidasCamadas[i_camada-1]))
            
    #         saidasFinal.append(saidasCamadas[len(saidasCamadas)-1])
        
    #     ##for count in range(len(saidasFinal)):
    #             ##print(f"{saidasFinal[count][1:]}", end="")

    #     self.logIniciais()

    
    # 1326 caracteres 266
    def MLP_execucao(self, entradas, saidas:list):
        saidasFinal = []
        entradas = np.insert(entradas, 0, 1, axis=1) #Coloca o Bias
        for linha_entrada, linha_saida in zip(entradas, saidas):
            saidasCamadas = []
            saidasCamadas.append(self.camadas[0].camadaFeedFoward(linha_entrada)) #Feedfoward na primeira camada
            
            for i_camada in range(1, len(self.camadas)): # Feedfoward nas outras camadas
                    saidasCamadas.append(self.camadas[i_camada].camadaFeedFoward(saidasCamadas[i_camada-1]))
                    
            saidasFinal.append(saidasCamadas[len(saidasCamadas)-1])            
            #print("terminou o feedfoward")

        with open("log/saidas_teste.txt", "w", encoding="utf-8") as saidas_teste: # "log/saidas_teste.txt"
            saidas_teste.write("")
        for count in range(len(saidasFinal)):
            self.logSaidas(saidasFinal[count][1:], "log/saidas_teste.txt") # Escreve as saidas
       