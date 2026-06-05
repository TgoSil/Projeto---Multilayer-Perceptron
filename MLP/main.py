## Projeto MLP  
## Importa csv
import pandas as pd
import numpy as np
from Gerenciador import Gerenciador
from pathlib import Path

def main():
    #Abrir arquivo csv
    # df_AND = pd.read_csv("portas_logicas/problemAND.csv", sep=",", header=None)
    # df_OR = pd.read_csv("portas_logicas/problemOR.csv", sep=",", header=None)
    # df_XOR = pd.read_csv("portas_logicas/problemXOR.csv", sep=",", header=None)

    dir_path = Path("log").mkdir(parents=True, exist_ok=True)

    #Dataset de caracteres
    df_X = pd.read_csv("caracteres_completo/X.txt", sep=",", header=None)
    entradas_X = pd.DataFrame(df_X.iloc[:, 0:-1]).values
    df_Y = np.load("caracteres_completo/Y_classe.npy")
    df_Y = np.where(df_Y == 0, -1, df_Y)
    targets_Y = df_Y
    print(df_Y)
    print(entradas_X)
    print(targets_Y)

    X_treino = entradas_X[:1196, :]
    X_teste = entradas_X[-130:, :]
    Y_treino = targets_Y[:1196, :]
    Y_teste = targets_Y[-130:, :]

    gere = Gerenciador(0.02, X_treino, Y_treino, 0.00001, 100)
    innit = gere.iniciaRede(2) 
    if not innit: return
    gere.printaRede()
    gere.MLP_treinamento(3000) #nro de epocas

    print(f"Tamanho dos testes {len(X_teste)} e {len(Y_teste)}")
    gere.MLP_execucao(X_teste, Y_teste)

    gere.avalicaoCompleta("log/saidas_teste.txt")

if __name__ == "__main__":
    main()