from Camada import Camada
import numpy as np

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
            self.camadas.append(Camada(qtdNeurons, len(self.entradas[0]))) # Parametro incorreto, faz todos os neuronios terem 3 pesos

    def criaArrayPesosDoBackPropagation(self, pesosCamadas): #"Virando" - tranposta da matriz de pesos da camada de saida para que seja mais fácil de calcular o deltinha da camada atual
        pesosParaCalcularDelta = []
        for j in range(len(pesosCamadas[0])): 
            pesosParaCalcularDelta.append([])
            for i in range(len(pesosCamadas)):
                pesosParaCalcularDelta[j].append(pesosCamadas[i][j]) 
        return pesosParaCalcularDelta

    def MLP(self):
        saidasCamadas = [] # Cada linha é uma camada e cada coluna é a resposta de um neurônio
        deltasCamadas = [] # Cada linha é uma camada e cada coluna é o delta de um neurônio
        for linha_entrada, linha_saida  in zip(self.entradas, self.targets):
            # print()
            # print(linha)
            ## Inicia FeedFoward e armazena as saidas de cada camada em um array
            saidasCamadas.append(self.camadas[0].camadaFeedFoward(linha_entrada)) # Feedforward na primeira camada

            for i_camada in range(1, len(self.camadas)): # Feedfoward nas outras camadas
                saidasCamadas.append(self.camadas[i_camada].camadaFeedFoward(saidasCamadas[i_camada-1])) 
            # print("Saidas:")
            # print(saidasCamadas)

            deltasCamadas.insert(0, # Armazena deltas da camada de output
                self.camadas[-1].camadaOutputBackPropagation(
                    saidasCamadas[-1], linha_saida
                ) #NÃO DEVERIA SER -2??
            )
 
            for i in range(len(self.camadas) -2, -1, -1): # Armazena deltas das outras camadas
                deltasCamadas.insert(0, 
                    self.camadas[i].camadaBackPropagation(
                        deltasCamadas[0], self.criaArrayPesosDoBackPropagation(self.camadas[i+1].pesos)
                    )
                )
            # print("Deltas:")
            # print(saidasCamadas)
            
            # print("Pesos e Bías:")
            self.camadas[0].camadaUpdate(deltasCamadas[0], linha_entrada, self.taxaAprendizado)
            # print(self.camadas[0].pesos)
            for i in range(1, len(self.camadas)):
                self.camadas[i].camadaUpdate(deltasCamadas[i], saidasCamadas[i-1], self.taxaAprendizado)
                # print(self.camadas[i].pesos)