import numpy as np
import random

class Neuron:
    
    def __init__(self, pesos:list, bias:float):
        self.pesos = pesos
        self.bias = bias

    def func_ativ_tanh(self, y_in):
        return 2 / (1 + np.exp(-2*y_in)) -1

    def neuron_iteration(self, x:list): ## Recebe 1 linha; 
        sum_x = 0.0
        for i in range(len(x)): 
            sum_x = sum_x + x[i]*self.pesos[i]
        print(f"sum_x: {sum_x}")
        y_in = self.bias + sum_x
        print(f"y_in: {y_in}")
        return float(self.func_ativ_tanh(y_in))