from Camada import Camada
import numpy as np
import random

class Gerenciador:

    def __init__(self, taxaAprendizado:float, entradas, saidas:list):
        self.taxaAprendizado = taxaAprendizado
        self.entradas = np.insert(entradas, 0, 1, axis=1)
        self.targets = saidas
        self.camadas = []

    def criaCamada(self, qtdNeurons:int):
        if len(self.camadas) > 0:
            self.camadas.append(Camada(qtdNeurons, len(self.camadas[-1].camada)+1))
        else:
            self.camadas.append(Camada(qtdNeurons, len(self.entradas[0])))

    def printaRede(self):
        for i, camada in enumerate(self.camadas):
            print(f"{i+1}ª camada: {len(camada.camada)} neurônios")

    def iniciaRede(self, qtdCamadas:int):
        if (qtdCamadas < 2): 
            print("VALORES MENORES QUE 2 NÃO PERMITIDOS!!\n(Não é possível criar uma rede sem pelo menos uma camada oculta e uma de saída)")
            return False
        self.criaCamada(len(self.entradas[0])-1) #Cria primeira camada oculta
        for i in range(qtdCamadas-2):
            # self.criaCamada(random.randint(2, 10)) #Cria camada oculta com numero aleatoria de neuronios
            self.criaCamada(120) #Cria camada oculta com 2 neuronios
        self.criaCamada(len(self.targets[0]))  #Cria camada de saida
        return True
        
    def logIniciais(self):
        with open("log/pesos_iniciais.txt", "w", encoding="utf-8") as pesos_iniciais:
            for i, camada in enumerate(self.camadas):
                # pesos_iniciais.write(f"Camada {i+1}:\n")
                np.savetxt(pesos_iniciais, camada.pesos, fmt='%.6f', delimiter=',')
    
    def logFinais(self):
        with open("log/pesos_finais.txt", "w", encoding="utf-8") as pesos_finais:
            for i, camada in enumerate(self.camadas):
                # pesos_finais.write(f"Camada {i+1}:\n")
                np.savetxt(pesos_finais, camada.pesos, fmt='%.6f', delimiter=',')

    def logSaidas(self, saidas:list):
        with open("log/saidas_teste.txt", "a", encoding="utf-8") as saidas_teste:
            np.savetxt(saidas_teste, saidas, fmt='%.6f', delimiter=',', newline=' ')
            saidas_teste.write("\n")

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
        for epoca in range(numEpocas): #Roda por um número definido de épocas (condição de parada)
            saidasFinal = []
            for linha_entrada, linha_saida  in zip(self.entradas, self.targets):
                saidasCamadas = [] # Cada linha é uma camada e cada coluna é a resposta de um neurônio
                deltasCamadas = [] # Cada linha é uma camada e cada coluna é o delta de um neurônio
                ## Inicia FeedFoward e armazena as saidas de cada camada em um array
                saidasCamadas.append(self.camadas[0].camadaFeedFoward(linha_entrada)) # Feedforward na primeira camada

                for i_camada in range(1, len(self.camadas)): # Feedfoward nas outras camadas
                    saidasCamadas.append(self.camadas[i_camada].camadaFeedFoward(saidasCamadas[i_camada-1]))  #faz feedfoward com os resultados da camada anterior

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
                saidasFinal.append(saidasCamadas[len(saidasCamadas)-1]) #pega valores da ultima camada
            print(f"ÉPOCA {epoca + 1}") # imprimi que época está 
            ##print(f"Saidas: ", end="") #imprime as saidas encontradas ao final dessa época
            ##for count in range(len(saidasFinal)):
            ##    print(f"{saidasFinal[count][1:]}", end="") #imprime as saidas encontradas ao final dessa época
            ##print()
        with open("log/saidas_teste.txt", "w", encoding="utf-8") as saidas_teste:
            saidas_teste.write("")
        for count in range(len(saidasFinal)):
            self.logSaidas(saidasFinal[count][1:])
        self.logFinais()

    def MLP_teste(self):
        matriz = np.loadtxt('log/pesos_finais.txt', delimiter=',')
        ##print(matriz)
        for camada in self.camadas:
            for i in range(len(camada.pesos)):
                for j in range(len(camada.pesos[0])):
                    camada.pesos[i][j] = matriz[i][j]
        
        saidasFinal = []
        for linha_entrada in self.entradas:
            saidasCamadas = [] # Cada linha é uma camada e cada coluna é a resposta de um neurônio
            ## Inicia FeedFoward e armazena as saidas de cada camada em um array
            saidasCamadas.append(self.camadas[0].camadaFeedFoward(linha_entrada)) # Feedforward na primeira camada

            for i_camada in range(1, len(self.camadas)): # Feedfoward nas outras camadas
                saidasCamadas.append(self.camadas[i_camada].camadaFeedFoward(saidasCamadas[i_camada-1]))
            
            saidasFinal.append(saidasCamadas[len(saidasCamadas)-1])
        
        ##for count in range(len(saidasFinal)):
                ##print(f"{saidasFinal[count][1:]}", end="")

        self.logIniciais()

    
    def MLP_execucao(self, ):
        pass