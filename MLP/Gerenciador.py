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
            self.camadas.append(Camada(qtdNeurons, len(self.entradas[0]))) # Parametro incorreto, faz todos os neuronios terem 3 pesos

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
            self.criaCamada(2) #Cria camada oculta com numero aleatoria de neuronios
        self.criaCamada(len(self.targets[0]))  #Cria camada de saida
        return True
        
    def log(self, saidas:list):
        print("Saidas rede neural: ")
        df = pd.DataFrame(saidas)
        print(df)
        # df.to_csv(path_escolhido\log.csv, index=False) Guarda o log de saidas
        for i, saida in enumerate(saidas):
            print(f"{i}ª saida: {saida}")
            # taxaAprendizado, entradas, saidas: ENTRADA GERENCIADOR
            # qtdNeurons, qtdEntradas: ENTRADA CAMADA
            # pesos: ENTRADA NEURONIO
            # return delta SAIDA ?

    def criaArrayPesosDoBackPropagation(self, pesosCamadas): #"Virando" - tranposta da matriz de pesos da camada de saida para que seja mais fácil de calcular o deltinha da camada atual
        pesosParaCalcularDelta = []
        for j in range(len(pesosCamadas[0])): 
            pesosParaCalcularDelta.append([])
            for i in range(len(pesosCamadas)):
                pesosParaCalcularDelta[j].append(pesosCamadas[i][j]) 
        return pesosParaCalcularDelta

    def MLP_treinamento(self, numEpocas:int):
        i = 0
        for epoca in range(numEpocas): #Roda por um número definido de épocas (condição de parada)
            saidasFinal = []
            for linha_entrada, linha_saida  in zip(self.entradas, self.targets):
                saidasCamadas = [] # Cada linha é uma camada e cada coluna é a resposta de um neurônio
                deltasCamadas = [] # Cada linha é uma camada e cada coluna é o delta de um neurônio
                # print()
                # print(linha)
                ## Inicia FeedFoward e armazena as saidas de cada camada em um array
                saidasCamadas.append(self.camadas[0].camadaFeedFoward(linha_entrada)) # Feedforward na primeira camada

                for i_camada in range(1, len(self.camadas)): # Feedfoward nas outras camadas
                    saidasCamadas.append(self.camadas[i_camada].camadaFeedFoward(saidasCamadas[i_camada-1]))  #faz feedfoward com os resultados da camada anterior
                # print("Saidas:")
                # print(saidasCamadas)

                deltasCamadas.insert(0, # Armazena deltas da camada de output
                    self.camadas[-1].camadaOutputBackPropagation(
                        saidasCamadas[-1], linha_saida
                    )
                )
    
                for i_deltas in range(len(self.camadas) -2, -1, -1): # Armazena deltas das outras camadas
                    deltasCamadas.insert(0, 
                        self.camadas[i_deltas].camadaBackPropagation(
                            deltasCamadas[0], self.criaArrayPesosDoBackPropagation(self.camadas[i_deltas+1].pesos)
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
                saidasFinal.append(saidasCamadas[len(saidasCamadas)-1]) #pega valores da ultima camada
            print(f"ÉPOCA {epoca + 1}") # imprimi que época está 
            print(f"Saidas: ", end="") #imprime as saidas encontradas ao final dessa época
            for count in range(len(saidasFinal)):
                print(f"{saidasFinal[count][1:]}", end="") #imprime as saidas encontradas ao final dessa época
            print()
        # self.log(saidasFinal)
    def MLP_execucao(self, ):
        pass