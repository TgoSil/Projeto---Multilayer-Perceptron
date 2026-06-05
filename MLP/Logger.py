import numpy as np
from Camada import Camada

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
            arquivo_log.write("Epoca,MSE\n") 

    def logErroEpoca(self, epoca: int, mse: float):
        with open("log/log_erro.txt", "a", encoding="utf-8") as arquivo_log:
            arquivo_log.write(f"{epoca},{mse:.6f}\n")

    def abrirLogSaidas(self, nomeArq):
        with open(nomeArq, "w", encoding="utf-8") as saidas:
            saidas.write("")

    def logSaidas(self, saidas:list, nomeArq):
        with open(nomeArq, "a", encoding="utf-8") as saidas_teste:
            np.savetxt(saidas_teste, saidas, fmt='%.6f', delimiter=',', newline=' ')
            saidas_teste.write("\n")
