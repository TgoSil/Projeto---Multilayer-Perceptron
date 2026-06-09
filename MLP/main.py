## Projeto MLP  
## Importa csv
import pandas as pd
import numpy as np
import time
import string
from Gerenciador import Gerenciador
from pathlib import Path
from Benchmark import roda_benchmarks

def mesclarY(Y, letra_origem, letra_destino):
    Y_modificado = Y.copy()
    idx_origem = ord(letra_origem.upper()) - ord('A')
    idx_destino = ord(letra_destino.upper()) - ord('A')
    
    linhas_alvo = (Y_modificado[:, idx_origem] == 1)
    Y_modificado[linhas_alvo, idx_origem] = -1.0
    Y_modificado[linhas_alvo, idx_destino] = 1.0
    return Y_modificado

def saltPepper(X, taxa_ruido):
    X_ruido = X.copy()
    probabilidades = np.random.rand(*X.shape)
    X_ruido[probabilidades < (taxa_ruido / 2)] = -1.0
    X_ruido[(probabilidades >= (taxa_ruido / 2)) & (probabilidades < taxa_ruido)] = 1.0
    return X_ruido

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

    # SETUP
    modo_validacao = 3
    ruidoSaltPepper = 1
    mesclar = False
    np.random.seed(42) # Semente para reprodutibilidade

    if mesclar:
        print("Agrupando classes")
        targets_Y = mesclarY(targets_Y, 'O', 'D')
        targets_Y = mesclarY(targets_Y, 'I', 'J')
        targets_Y = mesclarY(targets_Y, 'U', 'V')
        targets_Y = mesclarY(targets_Y, 'M', 'N')
        targets_Y = mesclarY(targets_Y, 'P', 'R')
        

    # Divisão dos dados em treino, validação e teste
    X_teste = entradas_X[-130:, :]
    Y_teste = targets_Y[-130:, :]

    X_restante = entradas_X[:-130, :] # Os dados restantes para treino e validação, retirando os últimos 130 exemplos para teste.
    Y_restante = targets_Y[:-130, :]
    classes_restantes = np.argmax(Y_restante, axis=1)

    if modo_validacao == 1: 
        print("Usando modo ESTRATIFICADO (Exatamente 5 amostras por classe).")
        indices_validacao = []
        indices_treino = []

        for letra_classe in range(26): 
            idx_letra = np.where(classes_restantes == letra_classe)[0]
            if len(idx_letra) == 0: continue
            np.random.shuffle(idx_letra)
            
            indices_validacao.extend(idx_letra[:5]) # 5 para validação
            indices_treino.extend(idx_letra[5:])   # O resto para treino

        # Divisão dos conjuntos por índices
        X_validacao = X_restante[indices_validacao]
        Y_validacao = Y_restante[indices_validacao]
        X_treino = X_restante[indices_treino]
        Y_treino = Y_restante[indices_treino]

    elif modo_validacao == 2:
        print("Usando modo SEQUENCIAL (Fatiamento direto do final).")
        # Penúltimos/últimos 130 exemplos para validação e o restante para treino.
        X_validacao = X_restante[-130:, :]
        Y_validacao = Y_restante[-130:, :]
        X_treino = X_restante[:-130, :]
        Y_treino = Y_restante[:-130, :]

    elif modo_validacao == 3:
        print("-> Amostragem selecionada: ALEATÓRIA PURA (Sem nenhuma garantia de classe).")
        
        # Define a semente para garantir que o sorteio seja reprodutível
        
        # Gera uma lista com todos os índices válidos dos dados restantes
        todos_indices = np.arange(len(X_restante))
        
        # Sorteia aleatoriamente 130 índices únicos diretamente do bolo total
        indices_validacao = np.random.choice(todos_indices, size=130, replace=False)
        
        # O conjunto de treino recebe tudo o que NÃO foi escolhido para a validação
        indices_treino = np.setdiff1d(todos_indices, indices_validacao)
        
        # Fatiamento final dos conjuntos de dados
        X_validacao = X_restante[indices_validacao, :]
        Y_validacao = Y_restante[indices_validacao, :]
        X_treino = X_restante[indices_treino, :]
        Y_treino = Y_restante[indices_treino, :]

    if ruidoSaltPepper == 1:
        taxa = 0.2
        print(f"SALT e PEPPER (taxa de {taxa * 100}%)")
        X_treino = saltPepper(X_treino, taxa)
        X_validacao = saltPepper(X_validacao, taxa)

    print(f"Treino: {X_treino.shape[0]} | Validação: {X_validacao.shape[0]} | Teste: {X_teste.shape[0]}")

    # Gerenciador da rede neural
    # Execução 1: 150 neurônios na camada oculta, taxa de aprendizado de 0.02, limite de épocas de 3000, paciência de 300 e tolerância de 0.000001.
    gere = Gerenciador(0.12, X_treino, Y_treino, 0.00001, 100) # taxa de aprendizado, entradas, targets, erro mínimo, paciência
    innit = gere.iniciaRede(2, 35) # nro de camadas da rede neural, nro de neurônios na camada oculta
    if not innit: return
    gere.printaRede() # Imprime a estrutura da rede neural
    gere.MLP_treinamento(1000, X_validacao, Y_validacao) # nro de epocas, entradas de validação, targets de validação

    print(f"Tamanho dos testes {len(X_teste)} e {len(Y_teste)}") # Imprime o tamanho dos dados de teste para verificar se estão corretos
    gere.MLP_execucao(X_teste, Y_teste) # Executa a rede neural com os dados de teste

    gere.avalicaoCompleta("log/saidas_teste.txt", Y_teste) # Avaliação completa da rede neural, gerando o log de avaliação com acurácia, precisão, recall, f1-score e matriz de confusão.

    # # Execução 2: 150 neurônios na camada oculta, taxa de aprendizado de 0.02, limite de épocas de 3000, paciencia de 100 e tolerância de 0.000001.
    # gere = Gerenciador(0.02, X_treino, Y_treino, 0.000001, 100) # taxa de aprendizado, entradas, targets, erro mínimo, paciência
    # innit = gere.iniciaRede(2, 150) # nro de camadas da rede neural, nro de neurônios na camada oculta
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
