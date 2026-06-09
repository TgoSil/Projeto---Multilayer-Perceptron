import time
import numpy as np
from Gerenciador import Gerenciador
from Logger import Logger


##################################################################################################################
#  Função Benchmarking                                                                                           #
#                                                                                                                #
# - A função roda_benchmarks é responsável por automatizar a execução de testes de estresse e performance        #
# na arquitetura da rede neural. Ela instacia a rede múltiplas vezes com diferentes configurações para           #
# coletar dados sobre o custo computacional e a capacidade de convergência do modelo.                            #
#                                                                                                                #
# - Os testes são divididos em execuções com e sem a atuação do mecanismo de Early Stopping.                     #
# Os experimentos avaliam três dimensões principais:                                                             #
#   1. Impacto da quantidade de neurônios na camada oculta sobre o tempo de execução.                            #
#   2. Impacto da Taxa de Aprendizado sobre o numero de epocas e o tempo de execução.                            #
#   3. Impacto do limite máximo de épocas sobre o tempo de execucao                                              #
#                                                                                                                #
# - Todos os resultados de métricas e de tempo (usando a biblioteca 'time') são persistidos em arquivos TXT      #
# gerados através da classe Logger para posterior análise crítica.                                               #
#                                                                                                                #
##################################################################################################################

# def roda_benchmarks(list X_treino, list Y_treino, list X_validacao, list Y_validacao)
# | X_treino: Matriz (lista de listas) contendo os dados de entrada (features) para o treinamento.
# | Y_treino: Matriz contendo o gabarito (targets) correspondente aos dados de treinamento.
# | X_validacao: Matriz contendo os dados de entrada separados via amostragem estratificada para validação.
# | Y_validacao: Matriz contendo o gabarito correspondente aos dados de validação.

def roda_benchmarks(X_treino, Y_treino, X_validacao, Y_validacao):
    logger = Logger()
    
    # 1. BENCHMARK SEM EARLY STOPPING (Paciência = 99999)
    logger.inicializa_log_benchmark("resultados_sem_early_stopping.txt", "RESULTADOS DOS TESTES DE BENCHMARK (SEM EARLY STOPPING)")

    # TESTE 1: Neurônios X Tempo
    resultados_t1_sem = []
    for neuronios in [10, 60, 120, 150, 300, 500]: # Testa diferentes quantidades de neurônios na camada oculta, mantendo a taxa de aprendizado fixa em 0.02 e um limite de épocas de 1000, sem early stopping.
        np.random.seed(42) # Define uma semente para garantir a reprodutibilidade dos resultados.
        gere = Gerenciador(0.02, X_treino, Y_treino, 0.0000001, 99999) 
        gere.iniciaRede(2, neuronios) 
        
        inicio = time.time()
        gere.MLP_treinamento(1000, X_validacao, Y_validacao)
        tempo = time.time() - inicio
        
        resultados_t1_sem.append({'parametro': neuronios, 'epocas': gere.epocas_executadas, 'minutos': int(tempo // 60), 'segundos': tempo % 60})
    logger.log_benchmark_resultado("resultados_sem_early_stopping.txt", "\nTeste 1: Neurônios X Tempo X Épocas (Taxa Fixa: 0.02)", resultados_t1_sem)

    # TESTE 3: Quantidade de épocas X Tempo
    resultados_t3_sem = [] # Testa diferentes limites de épocas para o treinamento, mantendo a quantidade de neurônios fixa em 150 e a taxa de aprendizado fixa em 0.02, sem early stopping. 
    for epocas_limite in [100, 500, 1000, 2000, 4000, 5000]: 
        np.random.seed(42) # Define uma semente para garantir a reprodutibilidade dos resultados.
        gere = Gerenciador(0.02, X_treino, Y_treino, 0.0000001, 99999)
        gere.iniciaRede(2, 150)
        
        inicio = time.time()
        gere.MLP_treinamento(epocas_limite, X_validacao, Y_validacao)
        tempo = time.time() - inicio
        
        resultados_t3_sem.append({'parametro': epocas_limite, 'epocas': gere.epocas_executadas, 'minutos': int(tempo // 60), 'segundos': tempo % 60})
    logger.log_benchmark_resultado("resultados_sem_early_stopping.txt", "\nTeste 3: Quantidade de épocas x Tempo de execução (Neurônios: 150, Taxa: 0.02)", resultados_t3_sem)


    # 2. BENCHMARK COM EARLY STOPPING (Paciência = 100)
    logger.inicializa_log_benchmark("resultados_benchmark.txt", "RESULTADOS DOS TESTES DE BENCHMARK (COM EARLY STOPPING)")

    # TESTE 1: Neurônios X Tempo
    resultados_t1_com = [] # Testa diferentes quantidades de neurônios na camada oculta, mantendo a taxa de aprendizado fixa em 0.02 e um limite de épocas de 3000, com early stopping ativado (paciência = 100).
    for neuronios in [10, 60, 120, 150, 300, 500]: 
        np.random.seed(42) # Define uma semente para garantir a reprodutibilidade dos resultados.
        gere = Gerenciador(0.02, X_treino, Y_treino, 0.0000001, 100)
        gere.iniciaRede(2, neuronios)
        
        inicio = time.time()
        gere.MLP_treinamento(3000, X_validacao, Y_validacao)
        tempo = time.time() - inicio
        
        resultados_t1_com.append({'parametro': neuronios, 'epocas': gere.epocas_executadas, 'minutos': int(tempo // 60), 'segundos': tempo % 60})
    logger.log_benchmark_resultado("resultados_benchmark.txt", "\nTeste 1: Neurônios X Tempo X Épocas (Taxa Fixa: 0.02)", resultados_t1_com)

    # TESTE 2: Taxa de Aprendizado X Tempo
    resultados_t2_com = [] # Testa diferentes taxas de aprendizado, mantendo a quantidade de neurônios fixa em 150 e um limite de épocas de 3000, com early stopping ativado (paciência = 100).
    for taxa_aprendizado in [0.01, 0.05, 0.1, 0.3, 0.7, 1.0]: 
        np.random.seed(42) # Define uma semente para garantir a reprodutibilidade dos resultados.
        gere = Gerenciador(taxa_aprendizado, X_treino, Y_treino, 0.0000001, 100)
        gere.iniciaRede(2, 150)
        
        inicio = time.time()
        gere.MLP_treinamento(3000, X_validacao, Y_validacao)
        tempo = time.time() - inicio
        
        resultados_t2_com.append({'parametro': taxa_aprendizado, 'epocas': gere.epocas_executadas, 'minutos': int(tempo // 60), 'segundos': tempo % 60})
    logger.log_benchmark_resultado("resultados_benchmark.txt", "\nTeste 2: Taxa de Aprendizado X Tempo X Épocas (Neurônios Fixos: 150)", resultados_t2_com)

    print("\n -> Benchmarks concluídos! Resultados salvos na pasta 'log'.")
