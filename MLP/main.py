## Projeto MLP  
## Importa csv
import pandas as pd
import numpy as np
from Gerenciador import Gerenciador

def main():
    #Abrir arquivo csv
    # df_AND = pd.read_csv("portas_logicas/problemAND.csv", sep=",", header=None)
    # df_OR = pd.read_csv("portas_logicas/problemOR.csv", sep=",", header=None)
    # df_XOR = pd.read_csv("portas_logicas/problemXOR.csv", sep=",", header=None)

    #Dataset de caracteres
    df_X = pd.read_csv("caracteres_completo/X.txt", sep=",", header=None)
    entradas_X = pd.DataFrame(df_X.iloc[:, 0:-1]).values
    df_Y = np.load("caracteres_completo/Y_classe.npy")
    df_Y = np.where(df_Y == 0, -1, df_Y)
    targets_Y = df_Y
    print(df_Y)
    print(entradas_X)
    print(targets_Y)

    #Separação de dados de treino e de teste
    
    def meu_train_test_split(X, Y, test_size=0.2, random_state=None):
        # Garente q tenham o mesmo tamanho
        if len(X) != len(Y):
            print("Erro: X e Y devem ter a mesma quantidade de linhas (amostras).")

        #Travar a semente aleatória (se random_state for passado)
        if random_state is not None:
            np.random.seed(random_state)
            
        # 3. Descobrir o total de amostras e criar uma lista de índices [0, 1, 2, ..., N-1]
        num_amostras = len(X)
        indices = np.arange(num_amostras)
        
        # 4. Embaralhar os índices aleatoriamente
        np.random.shuffle(indices)
        
        # 5. Calcular onde fazer o "corte" para separar os dados
        qtd_teste = int(num_amostras * test_size)
        
        # 6. Dividir os índices
        indices_teste = indices[:qtd_teste]   # Pega os primeiros índices até o corte
        indices_treino = indices[qtd_teste:]  # Pega do corte até o final
        
        # 7. Converter X e Y para NumPy arrays (caso sejam listas do Python) 
        # O NumPy permite puxar várias linhas de uma vez só passando uma lista de índices
        X_np = np.array(X)
        Y_np = np.array(Y)
        
        # 8. Extrair os dados usando os índices embaralhados
        X_treino = X_np[indices_treino]
        X_teste = X_np[indices_teste]
        Y_treino = Y_np[indices_treino]
        Y_teste = Y_np[indices_teste]
        
        return X_treino, X_teste, Y_treino, Y_teste

    # Extrair arrays/list da entradas e saídas
    # entradas_AND = pd.DataFrame(df_AND.iloc[:, 0:2]).values
    # saidas_AND = pd.DataFrame(df_AND.iloc[:, 2]).values

    # entradas_OR = pd.DataFrame(df_OR.iloc[:, 0:2]).values
    # saidas_OR = pd.DataFrame(df_OR.iloc[:, 2]).values

    # entradas_XOR = pd.DataFrame(df_XOR.iloc[:, 0:2]).values
    # saidas_XOR = pd.DataFrame(df_XOR.iloc[:, 2]).values

    X_treino, X_teste, Y_treino, Y_teste = meu_train_test_split(entradas_X, targets_Y, test_size=0.2, random_state=42)
    gere = Gerenciador(0.02, X_treino, Y_treino, 0.00001, 100)
    innit = gere.iniciaRede(2) 
    if not innit: return
    gere.printaRede()
    gere.MLP_treinamento(3000) #nro de epocas

    print(f"Tamanho dos testes {len(X_teste)} e {len(Y_teste)}")
    gere.MLP_execucao(X_teste, Y_teste)

    m = gere.geraMatrizDeConfusao("log/saidas_teste.txt")
    acuracia = gere.avaliaAcuracia(m)
    recalls = gere.avaliaRecall(m)
    precisoes = gere.avaliaPrecisao(m)
    f1 = gere.avaliaF1Score(precisoes, recalls)
    gere.criaLogSimples(acuracia, precisoes, recalls, f1, m)

if __name__ == "__main__":
    main()