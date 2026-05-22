## Projeto MLP  
## Importa csv
from math import tanh
from neuron import Neuron 
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
    # np.set_printoptions(threshold=np.inf)
    df_Y = np.load("caracteres_completo/Y_classe.npy")
    df_Y = np.where(df_Y == 0, -1, df_Y)
    print(df_Y)

    # #Extrair arrays/list da entradas e saídas
    entradas_AND = pd.DataFrame(df_AND.iloc[:, 0:2]).values
    saidas_AND = pd.DataFrame(df_AND.iloc[:, 2]).values

    entradas_OR = pd.DataFrame(df_OR.iloc[:, 0:2]).values
    saidas_OR = pd.DataFrame(df_OR.iloc[:, 2]).values

    entradas_XOR = pd.DataFrame(df_XOR.iloc[:, 0:2]).values
    saidas_XOR = pd.DataFrame(df_XOR.iloc[:, 2]).values

    # camadas = iniciaRede(10, len(entradas_AND[0]), len(saidas_AND[0]))
    gere = Gerenciador(0.01, entradas_XOR, saidas_XOR)
    innit = gere.iniciaRede(2) 
    if not innit: return
    gere.printaRede()
    gere.MLP_treinamento(10000) #nro de epocas

    ## Passo 0: Inicializa pesos, bias, taxa de aprendizado, número de épocas, etc 
    ## Passo 1: Enquanto a condição de parada é falsa, execute mais uma época

    ## Passo 3: Cada unidade de entrada (Xi, i = 1.. n) recebe um sinal de entrada xi e o dissipa para todas as unidades na próxima camada.

    ## Passo 4: Cada unidade escondida (Zj, j = 1.. p ) soma as suas entradas ponderadas,
    ## aplica a função de ativação para computar seu sinal de saída, e o envia para a próxima camada

    ## Passo 5: Cada unidade de saída (Yk, k=1.. m) soma suas entradas poderadas, aplica a função de ativação para computar seu sinal, de saída

    ## Passo 6: Cada unidade de saída (Yk, k = 1..m) considera a sua saída e a saída esperada para o dado de entrada para então computar o termo de informação
    ## de erro &k. Então calcula a correção de pesos e bias (DeltaWjk, e DeltaW0k) e envia o termo0 de correção de erro para a camada abaixo(anterior).

    ## Passo 7: Cada unidade de saída (Zj, j = 1..p ) soma suas entradas &k (as informações de erro vinda da camada acima (posterior)

if __name__ == "__main__":
    main()