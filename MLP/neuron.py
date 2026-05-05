import numpy as np
import random as rd

class Neuron:
    
    taxa_de_aprendizado = 0.5

####2 / (1 + np.exp(-2*y_in)) -1
    def __init__(self, entradas:int):
        self.pesos = [round(rd.uniform(-1,1),4) for i in range(entradas)] 
        self.bias = round(rd.uniform(-1,1),4)
        self.value_in = None

    def func_ativ_tanh(self, y_in):
        return 2 / (1 + np.exp(-2*y_in)) -1
    
    def derivada_func_ativ_tanh(self, y_in): ##depois pensa em generalizar para outras funcoes de ativação
        return (1 / np.cosh(y_in))**2

    def calcula_value_in(self, x:list):
        sum_x = 0.0
        for i in range(len(x)):
            sum_x = sum_x + x[i]*self.pesos[i]
        return self.bias + sum_x
        
    def corrige_pesos_e_bias(self, delta, lista_entradas:list, taxa_aprendizado:float):
        pesos_corrigidos = []
        for entrada in lista_entradas:
            pesos_corrigidos.append(taxa_aprendizado * delta * entrada)
        self.bias = self.bias + (delta * taxa_aprendizado)
        return pesos_corrigidos # Ver se por retornar ponteiro não vai dar algum problema doido
            

    def neuron_FeedFoward(self, x:list): ## Recebe array de valores de entrada; 
        self.value_in = self.calcula_value_in(x)
        return float(self.func_ativ_tanh(self.value_in))
    

    def output_Neuron_BackPropagation(self, y_k:float, t_k:float, lista_entradas:list, taxa_aprendizado:float):
        pesos_antigos = self.pesos.copy()
        erro = (t_k - y_k)             
        delta = erro * self.derivada_func_ativ_tanh(self.value_in)  # (target - gerado) x derivada(y_in)
        
        correcao_pesos = self.corrige_pesos_e_bias(delta, lista_entradas,taxa_aprendizado)
        for i in range(len(self.pesos)):
            self.pesos[i] = self.pesos[i] + correcao_pesos[i]
        return delta, pesos_antigos #confia no pai
    

    def neuron_BackPropagation(self, lista_deltas:list, lista_pesos_frente:list, lista_entradas:list, taxa_aprendizado:float):
        pesos_antigos = self.pesos.copy()
        delta_in = 0
        for i in range(len(lista_pesos_frente)):
            delta_in += lista_deltas[i] * lista_pesos_frente[i]

        delta = delta_in * self.derivada_func_ativ_tanh(self.value_in)
        correcao_pesos = self.corrige_pesos_e_bias(delta, lista_entradas,taxa_aprendizado)
        for i in range(len(self.pesos)):
            self.pesos[i] = self.pesos[i] + correcao_pesos[i]
        return pesos_antigos

        #print (f"delta_neuron: {delta_neuron}")
        

# a = Neuron(1)

y_in = 1
a = Neuron(1)
print(a.derivada_func_ativ_tanh(y_in))

