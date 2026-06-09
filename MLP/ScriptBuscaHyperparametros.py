import numpy as np
import pandas as pd
import multiprocessing
import os
from Gerenciador import Gerenciador

##################################################################################################################
#  Script de Busca de Hiperparâmetros                                                                            #
#                                                                                                                #
# - Este script é responsável por automatizar a execução e o teste de múltiplas configurações de hiperparâmetros #
#   para a rede neural. Ele utiliza processamento paralelo (multiprocessing) para agilizar o treinamento e a     #
#   validação de diferentes arquiteturas e taxas de aprendizado simultaneamente.                                 #
#                                                                                                                #
# - É neste arquivo que definimos alguns "perfis de teste" e executamos a classe Gerenciador em múltiplos     #
#   múltiplos núcleos lógicos do processador. O script garante o isolamento de pastas, logs e resultados         #
#   (arquivos CSV) para cada processo trabalhador (worker), evitando conflitos de escrita.                       #
#                                                                                                                #
##################################################################################################################

# Nome da pasta onde todos os CSVs serão organizados
PASTA_SAIDAS = "saidas_script"
PASTA_RESULTADOS = "resultados_automacao"


# Gerador de configurações de hiperparâmetros
# A função gera_configs_unicas é responsável por criar combinações (perfis) de teste contendo diferentes valores
# para a quantidade de neurônios, taxa de aprendizado, paciência e épocas limite. Ela sorteia esses valores com base
# em perfis pré-definidos (4: rápido/cirurgião, 5: médio/equilibrista, 6: longo/maratonista) e garante que nenhuma
# configuração repetida seja enviada para os processos de treinamento.

# def gera_configs_unicas (int total_configs)
# | total_configs: Um número inteiro que representa o limite máximo de combinações únicas a serem geradas.
# | return: Uma lista de dicionários, onde cada dicionário contém uma configuração única de hiperparâmetros a ser testada.

def gera_configs_unicas(total_configs=1200):
    configs_unicas = set()
    lista_configs = []
    
    print(f"Gerando {total_configs} combinações únicas de hiperparâmetros...")
    
    while len(configs_unicas) < total_configs:
        perfil = np.random.choice([4, 5, 6])

        # if perfil == 1:
        #     neuronios = int(np.random.choice([30, 45, 60, 75, 90]))
        #     taxa = round(float(np.random.uniform(0.005, 0.008)), 5)
        #     epocas = 3000
        #     paciencia = int(np.random.choice([400, 500, 600]))
        # elif perfil == 2:
        #     neuronios = int(np.random.choice([100, 150, 200, 250]))
        #     taxa = round(float(np.random.uniform(0.002, 0.004)), 5)
        #     epocas = 5000
        #     paciencia = int(np.random.choice([600, 700, 800]))
        # else: # 3
        #     neuronios = int(np.random.choice([30, 40, 50])) 
        #     taxa = round(float(np.random.uniform(0.001, 0.002)), 5)
        #     epocas = 8000
        #     paciencia = 1000

        if perfil == 4:
            neuronios = int(np.random.choice([15, 20, 25, 30]))
            taxa = round(float(np.random.choice([0.05, 0.01, 0.005, 0.001])), 5)
            epocas = 5000
            paciencia = int(np.random.choice([500, 600]))
            
        elif perfil == 5:
            neuronios = int(np.random.choice([35, 45, 55, 65, 75]))
            taxa = round(float(np.random.choice([0.02, 0.008, 0.004, 0.001])), 5)
            epocas = 6000
            paciencia = int(np.random.choice([800, 1000]))
            
        else: # 6
            neuronios = int(np.random.choice([80, 100, 120])) 
            taxa = round(float(np.random.choice([0.0005, 0.0001, 0.00005])), 6)
            epocas = 10000
            paciencia = 1500
            
        erro_min = 0.00001
        config_tupla = (taxa, erro_min, paciencia, neuronios, epocas)
        
        if config_tupla not in configs_unicas:
            configs_unicas.add(config_tupla)
            lista_configs.append({
                "taxa_aprendizado": taxa,
                "erro_minimo": erro_min,
                "paciencia": paciencia,
                "nro_neuronios": neuronios,
                "nro_total_epocas": epocas
            })
            
    return lista_configs


# Worker de Treinamento (Processo Isolado)
# A função worker_treinamento representa a unidade de trabalho isolada que será executada por cada núcleo de processamento.
# Ela recebe os parâmetros e o dataset, instancia um novo Gerenciador, treina a rede neural, realiza a validação 
# e extrai as métricas finais (acurácia e épocas rodadas). Por fim, grava o resultado em um arquivo CSV exclusivo 
# do núcleo que executou a tarefa, prevenindo a colisão de gravação em disco (Race Condition).

# def worker_treinamento (tuple params)
# | params: Uma tupla contendo um dicionário com a configuração de hiperparâmetros atual e os arrays NumPy 
# | correspondentes aos conjuntos de dados (X_treino, Y_treino, X_validacao, Y_validacao, X_teste, Y_teste).
# | return: Uma string formatada contendo o resumo do resultado da execução (Neurônios, Taxa e Acurácia) para ser exibida no console principal.

def worker_treinamento(params):
    # Semente travada para garantir reprodutibilidade
    np.random.seed(42)

    config, X_treino, Y_treino, X_validacao, Y_validacao, X_teste, Y_teste = params
    
    nome_processo = multiprocessing.current_process().name
    
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    nome_arquivo_csv = os.path.join(PASTA_RESULTADOS, f"resultado_{nome_processo}.csv")
    
    # Cria o cabeçalho do arquivo CSV caso ele ainda não exista para o trabalhador atual
    if not os.path.exists(nome_arquivo_csv):
        with open(nome_arquivo_csv, "w", encoding="utf-8") as f:
            f.write("acuracia,nro_epocas_executada,nro_neuronios,taxa_aprendizado,erro_minimo,paciencia,nro_total_epocas_limite\n")
    
    gere = Gerenciador(config["taxa_aprendizado"], X_treino, Y_treino, config["erro_minimo"], config["paciencia"])
    innit = gere.iniciaRede(2, config["nro_neuronios"])
    
    if not innit:
        return f"[{nome_processo}] Erro ao inicializar rede."
    
    pasta_processo = os.path.join("log", PASTA_SAIDAS, nome_processo)
    os.makedirs(pasta_processo, exist_ok=True)
    
    # Garante a barra no final (os.sep) para o Gerenciador poder concatenar os nomes
    caminho_logs_treino = pasta_processo + os.sep 
    
    # 1. O Treinamento usa a pasta exclusiva do núcleo
    gere.MLP_treinamento(config["nro_total_epocas"], X_validacao, Y_validacao, caminho_logs=caminho_logs_treino)
    
    # 2. A Execução gera o arquivo dentro da pasta do núcleo
    caminho_txt_saidas = os.path.join(pasta_processo, "saidas_teste.txt")
    if os.path.exists(caminho_txt_saidas):
        try: os.remove(caminho_txt_saidas) 
        except: pass

    gere.MLP_execucao(X_teste, Y_teste, caminho_txt_saidas)
    
    # 3. A Avaliação lê o arquivo isolado
    matriz_de_confusao = gere.geraMatrizDeConfusao(caminho_txt_saidas, Y_teste)
    acuracia_final = round(gere.avaliaAcuracia(matriz_de_confusao), 4)
    
    # Apaga apenas o TXT de saídas provisório para economizar espaço em disco
    try: os.remove(caminho_txt_saidas)
    except: pass

    epocas_rodadas = gere.epocas_executadas
    
    # Prepara a linha de resultado a ser anexada no CSV
    linha_resultado = (
        f"{acuracia_final},{epocas_rodadas},{config['nro_neuronios']},"
        f"{config['taxa_aprendizado']},{config['erro_minimo']},"
        f"{config['paciencia']},{config['nro_total_epocas']}\n"
    )
    
    with open(nome_arquivo_csv, "a", encoding="utf-8") as f:
        f.write(linha_resultado)
        
    return f"Neurônios: {config['nro_neuronios']} | Taxa: {config['taxa_aprendizado']} -> Acurácia: {acuracia_final}"


# Execução Principal (Main)
# O bloco principal é o ponto de entrada do script. Ele é isolado sob a condição __name__ == "__main__" para evitar
# que instâncias filhas (workers) re-executem este código ao serem importadas pelo multiprocessing.
# É aqui que os dados são lidos, segmentados em Treino, Validação e Teste, e onde o Pool de processos é
# instanciado para distribuir a carga de trabalho.

if __name__ == "__main__":
    print("Carregando datasets igual à main.py...")
    
    df_X = pd.read_csv("caracteres_completo/X.txt", sep=",", header=None)
    entradas_X = pd.DataFrame(df_X.iloc[:, 0:-1]).values
    df_Y = np.load("caracteres_completo/Y_classe.npy")
    df_Y = np.where(df_Y == 0, -1, df_Y)
    targets_Y = df_Y

    # 1. Separando o conjunto de Teste (Últimos 130 exemplos)
    X_teste = entradas_X[-130:, :]
    Y_teste = targets_Y[-130:, :]

    # 2. O Restante para separar Treino e Validação
    X_restante = entradas_X[:-130, :]
    Y_restante = targets_Y[:-130, :]

    # 3. Separando Validação (Penúltimos 130) e Treino (O resto que sobrou da base)
    X_validacao = X_restante[-130:, :]
    Y_validacao = Y_restante[-130:, :]
    X_treino = X_restante[:-130, :]
    Y_treino = Y_restante[:-130, :]
    # ------------------------------------------

    num_nucleos = multiprocessing.cpu_count()
    print(f"Detectados {num_nucleos} núcleos lógicos.")
    
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    print(f"Diretório '{PASTA_RESULTADOS}' preparado para armazenar os logs.")
    
    quantidade_testes = 1200
    lista_configs = gera_configs_unicas(quantidade_testes)
    
    # Adicionando X_teste e Y_teste no empacotamento, enviando as listas de hiperparâmetros e bases de dados
    pacotes_de_trabalho = [
        (config, X_treino, Y_treino, X_validacao, Y_validacao, X_teste, Y_teste) 
        for config in lista_configs
    ]
    
    print(f"Iniciando a piscina com {num_nucleos} processos (Pressione Ctrl+C para interromper).")
    print("-" * 60)
    
    testes_concluidos = 0
    
    try:
        # Inicia a execução concorrente. O imap_unordered consome a fila à medida que os processos ficam ociosos.
        with multiprocessing.Pool(processes=num_nucleos) as pool:
            for retorno_worker in pool.imap_unordered(worker_treinamento, pacotes_de_trabalho):
                testes_concluidos += 1
                print(f"[{testes_concluidos}/{quantidade_testes}] {retorno_worker}")
                
    except KeyboardInterrupt:
        print(f"\n\nExecução interrompida! Paramos no teste: {testes_concluidos}.")