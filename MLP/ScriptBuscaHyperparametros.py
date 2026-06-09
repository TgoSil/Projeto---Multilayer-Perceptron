import numpy as np
import pandas as pd
import multiprocessing
import os
from Gerenciador import Gerenciador

# Nome da pasta onde todos os CSVs serão organizados
PASTA_SAIDAS = "saidas_script"
PASTA_RESULTADOS = "resultados_automacao"

# Gerador de configurações de hyperparâmetros únicas
def gera_configs_unicas(total_configs=1200):
    configs_unicas = set()
    lista_configs = []
    
    print(f"Gerando {total_configs} combinações únicas de hiperparâmetros...")
    
    while len(configs_unicas) < total_configs:
        perfil = np.random.choice(["compacta", "media", "complexa"])
        
        if perfil == "compacta":
            neuronios = int(np.random.choice([30, 45, 60, 75, 90]))
            taxa = round(float(np.random.uniform(0.005, 0.008)), 5)
            epocas = 3000
            paciencia = int(np.random.choice([400, 500, 600]))
        elif perfil == "media":
            neuronios = int(np.random.choice([100, 150, 200, 250]))
            taxa = round(float(np.random.uniform(0.002, 0.004)), 5)
            epocas = 5000
            paciencia = int(np.random.choice([600, 700, 800]))
        else: # complexa
            neuronios = int(np.random.choice([30, 40, 50])) 
            taxa = round(float(np.random.uniform(0.001, 0.002)), 5)
            epocas = 8000
            paciencia = 1000
            
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

def worker_treinamento(params):
    # Semente travada para garantir reprodutibilidade
    np.random.seed(42)

    config, X_treino, Y_treino, X_validacao, Y_validacao, X_teste, Y_teste = params
    
    nome_processo = multiprocessing.current_process().name
    
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    nome_arquivo_csv = os.path.join(PASTA_RESULTADOS, f"resultado_{nome_processo}.csv")
    
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

    # Lembrete: O seu Gerenciador agora espera um 'caminho_arquivo' (corrigimos isso na última mensagem)
    gere.MLP_execucao(X_teste, Y_teste, caminho_txt_saidas)
    
    # 3. A Avaliação lê o arquivo isolado
    matriz_de_confusao = gere.geraMatrizDeConfusao(caminho_txt_saidas, Y_teste)
    acuracia_final = round(gere.avaliaAcuracia(matriz_de_confusao), 4)
    
    # Apaga apenas o TXT de saídas para economizar espaço
    try: os.remove(caminho_txt_saidas)
    except: pass

    epocas_rodadas = gere.epocas_executadas
    
    linha_resultado = (
        f"{acuracia_final},{epocas_rodadas},{config['nro_neuronios']},"
        f"{config['taxa_aprendizado']},{config['erro_minimo']},"
        f"{config['paciencia']},{config['nro_total_epocas']}\n"
    )
    
    with open(nome_arquivo_csv, "a", encoding="utf-8") as f:
        f.write(linha_resultado)
        
    return f"Neurônios: {config['nro_neuronios']} | Taxa: {config['taxa_aprendizado']} -> Acurácia: {acuracia_final}"
# Main
if __name__ == "__main__":
    print("Carregando bases de dados na memória principal...")
    
    df_X = pd.read_csv("caracteres_completo/X.txt", sep=",", header=None)
    entradas_X = pd.DataFrame(df_X.iloc[:, 0:-1]).values
    df_Y = np.load("caracteres_completo/Y_classe.npy")
    df_Y = np.where(df_Y == 0, -1, df_Y)
    targets_Y = df_Y

    # 1. Separando o conjunto de Teste (Últimos 130)
    X_teste = entradas_X[-130:, :]
    Y_teste = targets_Y[-130:, :]

    # 2. O Restante para separar Treino e Validação
    X_restante = entradas_X[:-130, :]
    Y_restante = targets_Y[:-130, :]

    # 3. Separando Validação (Penúltimos 130) e Treino (O resto que sobrou)
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
    
    # Adicionando X_teste e Y_teste no empacotamento
    pacotes_de_trabalho = [
        (config, X_treino, Y_treino, X_validacao, Y_validacao, X_teste, Y_teste) 
        for config in lista_configs
    ]
    
    print(f"Iniciando a piscina com {num_nucleos} processos (Pressione Ctrl+C para interromper).")
    print("-" * 60)
    
    testes_concluidos = 0
    
    try:
        with multiprocessing.Pool(processes=num_nucleos) as pool:
            for retorno_worker in pool.imap_unordered(worker_treinamento, pacotes_de_trabalho):
                testes_concluidos += 1
                print(f"[{testes_concluidos}/{quantidade_testes}] {retorno_worker}")
                
    except KeyboardInterrupt:
        print(f"\n\nExecução interrompida! Paramos no teste: {testes_concluidos}.")