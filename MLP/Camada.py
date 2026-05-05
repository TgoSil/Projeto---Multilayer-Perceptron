from neuron import Neuron 

class Camada:

    def __init__(self, tamanho:int, qtdEntradas:int):
        self.camada = [Neuron(qtdEntradas) for i in range(tamanho)] #Recebe a quantidade de entradas e cria a lista de neuronios

    def camadaFeedFoward(self, entradas:list):
        saidasCamada = []
        for neuronio in self.camada: ## Para cada neuronio 
            saidasCamada.append(neuronio.neuron_FeedFoward(entradas)) ## Processa 1 linha e salva no array de saidas da camada
        #print (f"Saidas: {saidasCamada}")
        return saidasCamada

    def camadaBackPropagation(self, deltas:list, entradas:list, taxa:float):
        deltasCamada = []
        pesosAnterior = []
        for i in range(len(self.camada)):
            pesos_antigos, delta = self.camada[i].neuronBackPropagation(deltas, entradas, taxa)
            pesosAnterior.append(pesos_antigos[i])
            deltasCamada.append(delta)
        return deltasCamada, pesosAnterior

    def camadaOutputBackPropagation(self, y_k:list, t_k:list, entradas:list, taxa:float): #Entrada nesse caso é a saida da camada_oculta
        deltasCamada = []
        for i in range(len(self.camada)):
            deltasCamada.append(self.camada[i].output_Neuron_BackPropagation(y_k[i], t_k[i], entradas, taxa))
        
        return deltasCamada