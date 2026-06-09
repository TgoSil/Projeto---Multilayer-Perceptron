import numpy as np
import string
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from Camada import Camada

##################################################################################################################
#  Classe Logger                                                                                                 #
#                                                                                                                #
# - A classe Logger é responsável por gerenciar a criação e o funcionamento dos arquivos de log, coordenando     #
#   as operações de escrita de informações durante o processo de treinamento, teste e validação.                 #
#                                                                                                                #
# - É na classe Logger que as funções de log são chamadas, na ordem correta, para realizar o registro das        #
# informações durante o processo de treinamento, teste e validação.                                              #
#                                                                                                                #
##################################################################################################################

class Logger:
    
    def criaLogSimples(self, acuracia, vetor_precisao, vetor_recall, vetor_f1, matriz_confusao):
        import numpy as np

        np.set_printoptions(linewidth=np.inf)

        with open("log/log_avaliacao.txt", "w", encoding="utf-8") as f:
            f.write(f"Acuracia Global: {acuracia}\n\n")

            f.write("Precisao por classe:\n")
            f.write(str(vetor_precisao) + "\n\n")

            f.write("Recall por classe:\n")
            f.write(str(vetor_recall) + "\n\n")

            f.write("F1-Score por classe:\n")
            f.write(str(vetor_f1) + "\n\n")

            f.write("Matriz de Confusao:\n")
            f.write(str(matriz_confusao) + "\n")

        print("Log salvo com sucesso!")

        # Chama função que plota a matriz de confusão
        self.plotarMatriz(matriz_confusao)
        self.plotarCurva("log/log_erro.txt")

    def logPesos(self, camadas:list, nomeArq:str):
        os.makedirs(os.path.dirname(nomeArq), exist_ok=True)
        with open(nomeArq, "w", encoding="utf-8") as pesos:
            for camada in camadas:
                np.savetxt(pesos, camada.pesos, fmt='%.6f', delimiter=',')

    def abrirLogErro(self):
        with open("log/log_erro.txt", "w", encoding="utf-8") as arquivo_log:
            arquivo_log.write("Epoca,MSE_Treino,MSE_Validacao\n")

    def logErroEpoca(self, epoca: int, mse_treino: float, mse_validacao: float):
        with open("log/log_erro.txt", "a", encoding="utf-8") as arquivo:
            arquivo.write(f"{epoca},{mse_treino:.6f},{mse_validacao:.6f}\n")

    def abrirLogSaidas(self, nomeArq):
        os.makedirs(os.path.dirname(nomeArq), exist_ok=True)
        with open(nomeArq, "w", encoding="utf-8") as saidas:
            saidas.write("")

    def logSaidas(self, saidas:list, nomeArq):
        os.makedirs(os.path.dirname(nomeArq), exist_ok=True)
        with open(nomeArq, "a", encoding="utf-8") as saidas_teste:
            np.savetxt(saidas_teste, saidas, fmt='%.6f', delimiter=',', newline=' ')
            saidas_teste.write("\n")

    def inicializa_log_benchmark(self, arquivo_nome: str, titulo: str):
         with open(f"log/{arquivo_nome}", "w", encoding="utf-8") as arquivo_log:
            arquivo_log.write(titulo + "\n")

    def log_benchmark_resultado(self, arquivo_nome: str, titulo_secao: str, resultados: list):
        with open(f"log/{arquivo_nome}", "a", encoding="utf-8") as arquivo_log:
            print(titulo_secao)
            arquivo_log.write(titulo_secao + "\n")
            for resultado in resultados:
                linha = f"Neurônios/Taxa/Limite: {resultado['parametro']:<5} | Épocas: {resultado['epocas']:<5} | Tempo: {resultado['minutos']}m {resultado['segundos']:.2f}s"
                print(linha)
                arquivo_log.write(linha + "\n")

    def plotarMatriz(self, matriz):
        num_linhas, num_colunas = matriz.shape

        alfabeto = list(string.ascii_uppercase)
        labelsY = alfabeto.copy()
        labelsX = alfabeto.copy()
        if num_colunas == 27:
            labelsX.append("Inválido")

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            matriz,
            annot=True,
            cmap="YlOrBr",
            fmt="g",
            xticklabels=labelsX,
            yticklabels=labelsY,
            linewidths=0.5,
            linecolor="lightgray"
        )
        plt.xlabel("Saída", fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel("Target", fontsize=12, fontweight='bold', labelpad=10)
        plt.title("Matriz de Confusão", fontsize=16, pad=20)
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig("log/matriz_confusao.png", dpi=300)

        plt.close()

    def plotarCurva(self, caminho):
        dados_erro = pd.read_csv(caminho)
        
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.edgecolor'] = '#e0e0e0'
        plt.rcParams['axes.linewidth'] = 0.8
        
        fig, ax = plt.subplots(figsize=(11, 6), facecolor='#fcfcfc')
        ax.set_facecolor('#f4f5f7')
        
        # Pega os valores da última época
        ultima_epoca = dados_erro['Epoca'].iloc[-1]
        ultimo_mse_treino = dados_erro['MSE_Treino'].iloc[-1]
        ultimo_mse_validacao = dados_erro['MSE_Validacao'].iloc[-1]
        
        # Plot da curva de Treino
        ax.plot(
            dados_erro['Epoca'],
            dados_erro['MSE_Treino'],
            color='#2b5c8f',
            linewidth=2.0,
            drawstyle='steps-post',       
            markersize=3
        )
        ax.text(
            ultima_epoca, 
            ultimo_mse_treino + 0.1,
            'MSE (Treino)', 
            color='#2b5c8f', 
            fontsize=10, 
            fontweight='bold',
            ha='right',
            va='bottom'
        )
        
        # Plot da curva de Validação
        ax.plot(
            dados_erro['Epoca'],
            dados_erro['MSE_Validacao'],
            color='#d9534f',
            linewidth=2.0,    
            drawstyle='steps-post',
            markersize=4
        )
        ax.text(
            ultima_epoca, 
            ultimo_mse_validacao + 0.1,
            'MSE (Validação)', 
            color='#d9534f', 
            fontsize=10, 
            fontweight='bold',
            ha='right', 
            va='bottom'
        )

        # Ajustes de layout e eixos
        ax.set_xlim(left=0, right=ultima_epoca * 1.02)
        ax.set_ylim(bottom=0)
        ax.set_title("Curva de Aprendizado", fontsize=15, fontweight='600', pad=20, color='#333333')
        ax.set_xlabel("Épocas", fontsize=11, fontweight='500', color='#555555', labelpad=10)
        ax.set_ylabel("Erro Quadrático Médio (MSE)", fontsize=11, fontweight='500', color='#555555', labelpad=10)
        ax.grid(True, color='white', linestyle='-', linewidth=1.2)
        ax.set_axisbelow(True) 
        ax.tick_params(colors='#666666', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')        
        plt.tight_layout()

        plt.savefig("log/curva_aprendizado.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()