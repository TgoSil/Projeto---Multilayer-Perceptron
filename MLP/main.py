## Projeto MLP  
## Importa csv
import pandas as pd
import numpy as np
import time
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
    ruidoSaltPepper = False # Altere para True ou False para ativar ou desativar o ruído do tipo Salt e Pepper
    mesclar = False # Altere para True ou False para ativar ou desativar a mesclagem de classes, reduzindo o número de classes de 26 para 21.
    np.random.seed(42) # Semente para reprodutibilidade

    # Mesclagem de classes (opcional)
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

    modo_validacao = 2 # altere o valor para 1, 2 ou 3 para escolher o modo de amostragem do conjunto de validação, sendo 1 para estratificado, 2 para sequencial e 3 para aleatório.
    if modo_validacao == 1: 
        print("Usando modo ESTRATIFICADO (5 amostras por classe).")
        classes_restantes = np.argmax(Y_restante, axis=1) 
        indices_validacao = []
        indices_treino = []
        
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

    elif modo_validacao == 3:
        print("Usando modo ALEATÓRIO (Sem nenhuma garantia de classe).")
        
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

    # Adição de ruído do tipo Salt e Pepper (opcional)
    if ruidoSaltPepper:
        taxa = 0.2
        print(f"SALT e PEPPER (taxa de {taxa * 100}%)")
        X_treino = saltPepper(X_treino, taxa)
        X_validacao = saltPepper(X_validacao, taxa)

    # Gerenciador da rede neural
    np.random.seed(42)

    gere = Gerenciador(0.02, X_treino, Y_treino, 0.00001, 1000) # taxa de aprendizado, entradas, targets, erro mínimo, paciência
    innit = gere.iniciaRede(2, 35) # nro de camadas da rede neural, nro de neurônios na camada oculta
    if not innit: return
    gere.printaRede() # Imprime a estrutura da rede neural
    gere.MLP_treinamento(6000, X_validacao, Y_validacao) # nro de epocas, entradas de validação, targets de validação

    print(f"Tamanho dos testes {len(X_teste)} e {len(Y_teste)}") # Imprime o tamanho dos dados de teste para verificar se estão corretos
    gere.MLP_execucao(X_teste, Y_teste) # Executa a rede neural com os dados de teste

    gere.avalicaoCompleta("log/saidas_teste.txt", Y_teste) # Avaliação completa da rede neural, gerando o log de avaliação com acurácia, precisão, recall, f1-score e matriz de confusão.

if __name__ == "__main__":
    main()
