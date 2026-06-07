import numpy as np
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

    def logPesos(self, camadas:list, nomeArq:str):
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
        with open(nomeArq, "w", encoding="utf-8") as saidas:
            saidas.write("")

    def logSaidas(self, saidas:list, nomeArq):
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
