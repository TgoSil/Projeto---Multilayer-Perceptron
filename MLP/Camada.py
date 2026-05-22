from neuron import Neuron 
import random as rd
import numpy as np
class Camada:
    
    def __init__(self, qtdNeurons:int, qtdEntradas:int):
        self.pesos = np.random.uniform(-0.5, 0.5, size=(qtdNeurons, qtdEntradas))
        # self.pesos = []
        # for i in range(qtdNeurons):
        #     self.pesos.append([round(rd.uniform(-0.5,0.5),4) for j in range (qtdEntradas)]) #Para cada neuronia cria seus pesos para cada entrada (E "+1" por causa do bias)
        self.camada = [Neuron(self.pesos[k]) for k in range(qtdNeurons)] #Recebe a quantidade de entradas e cria a lista de neuronios
    
    def camadaFeedFoward(self, entradas:list):
        saidasCamada = [1]
        for neuronio in self.camada: ## Para cada neuronio 
            saidasCamada.append(neuronio.neuron_FeedFoward(entradas)) ## Processa 1 linha e salva no array de saidas da camada
        #print (f"Saidas: {saidasCamada}")

        return saidasCamada

    def camadaBackPropagation(self, deltas:list, pesosCamadaFrente:list):
        deltasCamada = []
        for i in range(len(self.camada)):
            deltasCamada.append(self.camada[i].neuron_BackPropagation(deltas, pesosCamadaFrente[i+1]))
        
        return deltasCamada

    def camadaOutputBackPropagation(self, y_k:list, t_k:list): #Entrada nesse caso é a saida da camada_oculta
        deltasCamada = []
        for i in range(len(self.camada)):
            deltasCamada.append(self.camada[i].output_Neuron_BackPropagation(y_k[i+1], t_k[i]))
        return deltasCamada
    
    def camadaUpdate(self, deltas:list, listaEntradas:list, taxaAprendizado:float):
        for i, neuronio in enumerate(self.camada):
            neuronio.neuron_Update(deltas[i], listaEntradas, taxaAprendizado)