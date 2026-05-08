

import numpy as np
import random as rd

class Neuron:


    def __init__(self, pesos:list): #Recebe lista com seus pesos da camada
        self.pesos = pesos
        self.value_in = None

    def func_ativ_tanh(self, y_in):
        return 2 / (1 + np.exp(-2*y_in)) -1
    
    def derivada_func_ativ_tanh(self, y_in): #depois pensa em generalizar para outras funcoes de ativação
        return (1 / np.cosh(y_in))**2

    def calcula_value_in(self, listaEntrada:list):
        sum_x = 0.0
        for i in range(len(listaEntrada)):
            sum_x = sum_x + listaEntrada[i]*self.pesos[i]
        return sum_x
        
    def neuron_Update(self, delta, listaEntradas:list, taxa_aprendizado:float): #Do neuronio
        for i in range(len(self.pesos)): # i indice, entrada (a própria entrada)
            self.pesos[i] = self.pesos[i] + (delta * taxa_aprendizado * listaEntradas[i])      

    def neuron_BackPropagation(self, deltas:list, pesosCamadaFrente:list):
        delta_in = 0.00
        for i in range(len(pesosCamadaFrente)): 
            delta_in += deltas[i]*pesosCamadaFrente[i]
        return delta_in * self.derivada_func_ativ_tanh(self.value_in) #Retorna o delta do próprio neurônio

    def neuron_FeedFoward(self, listaEntradas:list): ## Recebe array de valores de entrada; 
        self.value_in = self.calcula_value_in(listaEntradas)
        return float(self.func_ativ_tanh(self.value_in)) #Devolve a saída 

    def output_Neuron_BackPropagation(self, y_k:float, t_k:float):
        erro = (t_k - y_k)             
        delta = erro * self.derivada_func_ativ_tanh(self.value_in)  # (target - gerado) x derivada(y_in) -> regra da cadeia
        return delta
