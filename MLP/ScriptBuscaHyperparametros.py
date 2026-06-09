import numpy as np
import pandas as pd
import multiprocessing
import os
from Gerenciador import Gerenciador

# Nome da pasta onde todos os CSVs serão organizados
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

# Função de treinamento
def worker_treinamento(params):
    # Semente travada para garantir reprodutibilidade na inicialização
    np.random.seed(42)

    # Agora recebemos X_teste e Y_teste também
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
    
    # O Treinamento e o Early Stopping usam X_validacao
    gere.MLP_treinamento(config["nro_total_epocas"], X_validacao, Y_validacao)
    
    # A Validação (Acurácia Final) agora usa RIGOROSAMENTE X_teste (Igual a main.py)
    acertos = 0
    total = len(X_teste)
    X_teste_bias = np.insert(X_teste, 0, 1, axis=1)
    
    for linha_entrada, linha_saida in zip(X_teste_bias, Y_teste):
        saida_camada = gere.camadas[0].camadaFeedFoward(linha_entrada)
        saida_final = gere.camadas[1].camadaFeedFoward(saida_camada)
        
        classe_predita = np.argmax(saida_final[1:])
        classe_esperada = np.argmax(linha_saida)
        
        if classe_predita == classe_esperada:
            acertos += 1
            
    acuracia_final = round((acertos / total) * 100, 2)
    epocas_rodadas = gere.epocas_executadas
    
    linha_resultado = (
        f"{acuracia_final},{epocas_rodadas},{config['nro_neuronios']},"
        f"{config['taxa_aprendizado']},{config['erro_minimo']},"
        f"{config['paciencia']},{config['nro_total_epocas']}\n"
    )
    
    with open(nome_arquivo_csv, "a", encoding="utf-8") as f:
        f.write(linha_resultado)
        
    return f"Neurônios: {config['nro_neuronios']} | Taxa: {config['taxa_aprendizado']} -> Acurácia: {acuracia_final}%"

# Main
if __name__ == "__main__":
    print("Carregando bases de dados na memória principal...")
    
    df_X = pd.read_csv("caracteres_completo/X.txt", sep=",", header=None)
    entradas_X = pd.DataFrame(df_X.iloc[:, 0:-1]).values
    df_Y = np.load("caracteres_completo/Y_classe.npy")
    df_Y = np.where(df_Y == 0, -1, df_Y)
    targets_Y = df_Y

    # --- NOVO FATIAMENTO IDÊNTICO À MAIN.PY ---
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