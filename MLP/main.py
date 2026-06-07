## Projeto MLP  
## Importa csv
import pandas as pd
import numpy as np
import time
from Gerenciador import Gerenciador
from pathlib import Path
from Benchmark import roda_benchmarks

def main():

    # Abre arquivos csv para os problemas lógicos.
    # df_AND = pd.read_csv("portas_logicas/problemAND.csv", sep=",", header=None)
    # df_OR = pd.read_csv("portas_logicas/problemOR.csv", sep=",", header=None)
    # df_XOR = pd.read_csv("portas_logicas/problemXOR.csv", sep=",", header=None)

    dir_path = Path("log").mkdir(parents=True, exist_ok=True) # Cria pasta log caso ela não exista.

    # Dataset de caracteres_completo
    df_X = pd.read_csv("caracteres_completo/X.txt", sep=",", header=None)
    entradas_X = pd.DataFrame(df_X.iloc[:, 0:-1]).values
    df_Y = np.load("caracteres_completo/Y_classe.npy")
    df_Y = np.where(df_Y == 0, -1, df_Y)
    targets_Y = df_Y

    # Divisão dos dados em treino, validação e teste
    X_teste = entradas_X[-130:, :]
    Y_teste = targets_Y[-130:, :]

    X_restante = entradas_X[:-130, :] # Os dados restantes para treino e validação, retirando os últimos 130 exemplos para teste.
    Y_restante = targets_Y[:-130, :]
    classes_restantes = np.argmax(Y_restante, axis=1) 

    modo_validacao = 1 # altere o valor para 1 ou 2 para escolher o modo de amostragem do conjunto de validação, sendo 1 para estratificado e 2 para sequencial.
    if modo_validacao == 1: 
        print("Usando modo ESTRATIFICADO (5 amostras por classe).")
        classes_restantes = np.argmax(Y_restante, axis=1) 
        indices_validacao = []
        indices_treino = []
        
        np.random.seed(42) 
        for letra_classe in range(26): 
            idx_letra = np.where(classes_restantes == letra_classe)[0]
            np.random.shuffle(idx_letra)
            indices_validacao.extend(idx_letra[:5])
            indices_treino.extend(idx_letra[5:])

        X_validacao = X_restante[indices_validacao]
        Y_validacao = Y_restante[indices_validacao]
        X_treino = X_restante[indices_treino]
        Y_treino = Y_restante[indices_treino]

    elif modo_validacao == 2:
        print("Usando modo SEQUENCIAL (Fatiamento direto do final).")
        # Penultimos 130 exemplos para validação, garantindo que haja exemplos de todas as classes, e os demais para treino.
        X_validacao = X_restante[-130:, :]
        Y_validacao = Y_restante[-130:, :]

        # Os dados restantes para treino, retirando os últimos 130 exemplos para validação.
        X_treino = X_restante[:-130, :]
        Y_treino = Y_restante[:-130, :]


    # Gerenciador da rede neural
    np.random.seed(42)
    # Execução 1: 150 neurônios na camada oculta, taxa de aprendizado de 0.02, limite de épocas de 3000, paciência de 300 e tolerância de 0.000001.
    gere = Gerenciador(0.02, X_treino, Y_treino, 0.000001, 300) # taxa de aprendizado, entradas, targets, erro mínimo, paciência
    innit = gere.iniciaRede(2, 150) # nro de camadas da rede neural, nro de neurônios na camada oculta
    if not innit: return
    gere.printaRede() # Imprime a estrutura da rede neural
    gere.MLP_treinamento(3000, X_validacao, Y_validacao) # nro de epocas, entradas de validação, targets de validação

    print(f"Tamanho dos testes {len(X_teste)} e {len(Y_teste)}") # Imprime o tamanho dos dados de teste para verificar se estão corretos
    gere.MLP_execucao(X_teste, Y_teste) # Executa a rede neural com os dados de teste

    gere.avalicaoCompleta("log/saidas_teste.txt", Y_teste) # Avaliação completa da rede neural, gerando o log de avaliação com acurácia, precisão, recall, f1-score e matriz de confusão.


    # Execução 2: 120 neurônios na camada oculta, taxa de aprendizado de 0.02, limite de épocas de 3000, paciencia de 100 e tolerância de 0.000001.
    # gere = Gerenciador(0.02, X_treino, Y_treino, 0.000001, 100) # taxa de aprendizado, entradas, targets, erro mínimo, paciência
    # innit = gere.iniciaRede(2, 120) # nro de camadas da rede neural, nro de neurônios na camada oculta
    # if not innit: return
    # gere.printaRede() # Imprime a estrutura da rede neural
    # gere.MLP_treinamento(3000, X_validacao, Y_validacao) # nro de epocas, entradas de validação, targets de validação

    # print(f"Tamanho dos testes {len(X_teste)} e {len(Y_teste)}") # Imprime o tamanho dos dados de teste para verificar se estão corretos
    # gere.MLP_execucao(X_teste, Y_teste) # Executa a rede neural com os dados de teste

    # gere.avalicaoCompleta("log/saidas_teste.txt", Y_teste) # Avaliação completa da rede neural, gerando o log de avaliação com acurácia, precisão, recall, f1-score e matriz de confusão.


    # Testes de benchmark
    # roda_benchmarks(X_treino, Y_treino, X_validacao, Y_validacao) # Descomentar para rodar os benchmarks, que testam o impacto da quantidade de neurônios, taxa de aprendizado e quantidade de épocas no tempo de execução e número de épocas necessárias para o treinamento, tanto com quanto sem early stopping.
if __name__ == "__main__":
    main()
