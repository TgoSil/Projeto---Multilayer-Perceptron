## Projeto MLP  
## Importa csv
from math import tanh
#from Neuron import Neuron 
import pandas as pd
import numpy as np
from pathlib import Path
import random as rd
from Gerenciador import Gerenciador


def main():
    #Abrir arquivo csv
    df_AND = pd.read_csv("portas_logicas/problemAND.csv", sep=",", header=None)
    df_OR = pd.read_csv("portas_logicas/problemOR.csv", sep=",", header=None)
    df_XOR = pd.read_csv("portas_logicas/problemXOR.csv", sep=",", header=None)

    df_X = pd.read_csv("caracteres_completo/X.txt", sep=",", header=None)
    entradas_X = pd.DataFrame(df_X.iloc[:, 0:-1]).values
    df_Y = np.load("caracteres_completo/Y_classe.npy")
    df_Y = np.where(df_Y == 0, -1, df_Y)
    saidas_Y = df_Y
    print(df_Y)

    # #Extrair arrays/list da entradas e saídas
    # entradas_AND = pd.DataFrame(df_AND.iloc[:, 0:2]).values
    # saidas_AND = pd.DataFrame(df_AND.iloc[:, 2]).values

    # entradas_OR = pd.DataFrame(df_OR.iloc[:, 0:2]).values
    # saidas_OR = pd.DataFrame(df_OR.iloc[:, 2]).values

    # entradas_XOR = pd.DataFrame(df_XOR.iloc[:, 0:2]).values
    # saidas_XOR = pd.DataFrame(df_XOR.iloc[:, 2]).values

    # camadas = iniciaRede(10, len(entradas_AND[0]), len(saidas_AND[0]))
    gere = Gerenciador(0.05, entradas_X, saidas_Y, 0.00001, 100)
    innit = gere.iniciaRede(8)
    if not innit: return
    gere.printaRede()
    gere.MLP_treinamento(100000) #nro de epocas

if __name__ == "__main__":
    main()