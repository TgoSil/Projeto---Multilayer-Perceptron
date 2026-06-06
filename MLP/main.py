## Projeto MLP  
## Importa csv
import pandas as pd
import numpy as np
from Gerenciador import Gerenciador
from pathlib import Path

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
    X_treino = entradas_X[:1066, :]
    X_validacao = entradas_X[-260:-130, :]
    X_teste = entradas_X[-130:, :]

    Y_treino = targets_Y[:1066, :]
    Y_validacao = targets_Y[-260:-130, :]
    Y_teste = targets_Y[-130:, :]

    # Gerenciador da rede neural
    gere = Gerenciador(0.02, X_treino, Y_treino, 0.00001, 100) # taxa de aprendizado, entradas, targets, erro mínimo, paciência
    innit = gere.iniciaRede(2) # nro de camadas da rede neural
    if not innit: return
    gere.printaRede() # Imprime a estrutura da rede neural
    gere.MLP_treinamento(3000, X_validacao, Y_validacao) # nro de epocas, entradas de validação, targets de validação

    print(f"Tamanho dos testes {len(X_teste)} e {len(Y_teste)}") # Imprime o tamanho dos dados de teste para verificar se estão corretos
    gere.MLP_execucao(X_teste, Y_teste) # Executa a rede neural com os dados de teste

    gere.avalicaoCompleta("log/saidas_teste.txt", Y_teste) # Avaliação completa da rede neural, gerando o log de avaliação com acurácia, precisão, recall, f1-score e matriz de confusão.

if __name__ == "__main__":
    main()