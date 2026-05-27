import numpy as np
class Camada:
    
    def __init__(self, qtdNeurons:int, qtdEntradas:int):
        self.pesos = np.random.uniform(-0.5, 0.5, size=(qtdNeurons, qtdEntradas))
        self.values_in = []
        self.camada = qtdNeurons
    
    def func_ativ_tanh(self, y_in):
        return np.tanh(y_in)

    def derivada_func_ativ_tanh(self, y_in): #depois pensa em generalizar para outras funcoes de ativação
        return 1.0 - np.tanh(y_in)**2

    def camadaFeedFoward(self, entradas:list):
        self.values_in = np.dot(self.pesos, entradas)
        saidasCamada = self.func_ativ_tanh(self.values_in)
        saidasCamada = np.insert(saidasCamada, 0, 1)
        return saidasCamada

    def camadaBackPropagation(self, deltas:list, pesosCamadaFrente:list):
        deltas_in = np.dot(pesosCamadaFrente[1:], deltas)
        deltasCamada = deltas_in * self.derivada_func_ativ_tanh(self.values_in)
        return deltasCamada

    def camadaOutputBackPropagation(self, y_k:list, t_k:list): #Entrada nesse caso é a saida da camada_oculta
        erro = np.array(t_k) - np.array(y_k[1:]) # em y_k tiramos a entrada do Bias
        deltasCamada = erro * self.derivada_func_ativ_tanh(self.values_in)
        return deltasCamada #Retorna lista de deltas
    
    def camadaUpdate(self, deltas:list, listaEntradas:list, taxaAprendizado:float):
        self.pesos += np.outer(deltas * taxaAprendizado, listaEntradas)
