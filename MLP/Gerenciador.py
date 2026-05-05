from Camada import Camada

class Gerenciador:

    def __init__(taxaAprendizado:float, entradas, saidas:list):
        taxaAprendizado = taxaAprendizado
        entradas = entradas
        saidas = saidas
        camadas = []

    def criaCamada(self, qtdNeurons:int):
        self.camadas.append(Camada(qtdNeurons, len(self.entradas[0])))

    def MLP(self):
        saidasCamadas = []
        for linha in self.entradas:
            ## Inicia FeedFoward e armazena as saidas de cada camada em um array
            saidasCamadas.append(self.camadas[0].camadaFeedFoward(linha)) #Pega os valores da 1 camada oculta.
            for i in range(1, len(self.camadas)):
                saidasCamadas.append(self.camadas[i].camadaFeedFoward(saidasCamadas[i-1]))
            
        
