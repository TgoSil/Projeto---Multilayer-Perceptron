## Projeto MLP  
## Importa csv
from math import tanh
from neuron import Neuron 
import pandas as pd
import numpy as np
import random as rd

#Abrir arquivo csv
df_AND = pd.read_csv("portas_logicas\\problemAND.csv", sep=",", header=None)
df_OR = pd.read_csv("portas_logicas\\problemOR.csv", sep=",", header=None)
df_XOR = pd.read_csv("portas_logicas\\problemXOR.csv", sep=",", header=None)

#Extrair arrays/list da entradas e saídas
entradas_AND = df_AND.iloc[:, 0:2].values
saidas_AND = df_AND.iloc[:, 2].values
print("CSV AND")
print(entradas_AND)
print(saidas_AND)


entradas_OR = df_OR.iloc[:, 0:2].values
saidas_OR = df_OR.iloc[:, 2].values
print()
print("CSV OR")
print(entradas_OR)
print(saidas_OR)

entradas_XOR = df_XOR.iloc[:, 0:2].values
saidas_XOR = df_XOR.iloc[:, 2].values
print()
print("CSV XOR")
print(entradas_XOR)
print(saidas_XOR)



## Inicializa pesos e bias
##Camada oculta (lista de neuronios)
tamanhoCamadaOculta = 2
camadaOculta = []
for i in range(tamanhoCamadaOculta):
    # camadaOculta.append(Neuron(
    #                 [round(rd.uniform(-1,1), 4), round(rd.uniform(-1,1), 4)],
    #                 round(rd.uniform(-1, 1), 4))
    #               )

   camadaOculta.append(Neuron([0.5, 0.5], 0.5))

#neuronioSaida = Neuron([round(rd.uniform(-1,1), 4), round(rd.uniform(-1,1), 4)],
#                  round(rd.uniform(-1, 1), 4))

neuronioSaida = Neuron([0.8, 0.1], 0.3)


# print("Neuronio 1: ")
# print(f"Pesos: {camadaOculta[0].pesos}")
# print(f"Bias: {camadaOculta[0].bias}")
# print("Neuronio 2: ")
# print(f"Pesos: {camadaOculta[1].pesos}")
# print(f"Bias: {camadaOculta[1].bias}")

# Taxa de aprendizado
# taxa_aprendizado = 0.5000


## ÉPOCAS - Definir condições de parada (minimiza alguma funcao de custo - informcao de erro, MSE (erro quadratico medio)) e taxa de aprendizado:
    ## Foward Propagation
saidasOcultas = []
listaMaluca = []

for i in range(len(entradas_AND)): ##Para cada linha do dataset
    print (f"Linha {i}: {entradas_AND[i]}") 
    for j in range(tamanhoCamadaOculta): ## Para cada neuronio 
        saidasOcultas.append(camadaOculta[j].neuronFeedFoward(entradas_AND[i])) ## Para cada neuronio da camada oculta processa 1 linha e salva no array de saidas da camada ocultas
    print (f"Saidas ocultas: {saidasOcultas}")
    SaidaFinal = neuronioSaida.neuronFeedFoward(saidasOcultas) ##Pega a quantidade de saidas ocultas e processa no neuronio de saida 
    listaMaluca.append(saidasOcultas.copy())  # Insere as saídas da camada oculta no final da lista de saídas
    saidasOcultas.clear() #Limpa lista Aux os parametros do neurônio de saida.
    
    print (f"Camada saida: {SaidaFinal}")
    print (f"Lista Maluca (Salva saidas): {listaMaluca}")
    
    ## BackWard Propagation (Gradiente Descente...) 

        # derivada --> np.gradient()


##dsa


## Arquitetura da MLP
 # 1 camada de entradas (2 neuronios)
 # 1 camada oculta (2 neuronios)
 # 1 camada de saida (1 neuronio)


## Define a funcao de ativacao (vai receber o valor da Funcao Somatorio) #tanh (x) = 2 / (1 + e ^ (- 2x)) -1 

# def func_ativ_tanh (y_in):
#     return 2 / (1 + np.exp(-2*y_in)) -1


# camada de entrada  (x)-> camada oculta funcao de ativacao ((x , w) + bias) ->   camada de saida



## Passo 0: Inicializa pesos, bias, taxa de aprendizado, número de épocas, etc 
## Passo 1: Enquanto a condição de parada é falsa, execute mais uma época

## Passo 3: Cada unidade de entrada (Xi, i = 1.. n) recebe um sinal de entrada xi e o dissipa para todas as unidades na próxima camada.

## Passo 4: Cada unidade escondida (Zj, j = 1.. p ) soma as suas entradas ponderadas,
## aplica a função de ativação para computar seu sinal de saída, e o envia para a próxima camada

## Passo 5: Cada unidade de saída (Yk, k=1.. m) soma suas entradas poderadas, aplica a função de ativação para computar seu sinal, de saída

## Passo 6: Cada unidade de saída (Yk, k = 1..m) considera a sua saída e a saída esperada para o dado de entrada para então computar o termo de informação
## de erro &k. Então calcula a correção de pesos e bias (DeltaWjk, e DeltaW0k) e envia o termo0 de correção de erro para a camada abaixo(anterior).

## Passo 7: Cada unidade de saída (Zj, j = 1..p ) soma suas entradas &k (as informações de erro vinda da camada acima (posterior)