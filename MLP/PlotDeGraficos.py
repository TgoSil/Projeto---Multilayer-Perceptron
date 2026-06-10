import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##################################################################################################################
#  Script de Análise e Geração de Gráficos                                                                       #
#                                                                                                                #
# - Este script é responsável por ler os resultados extraídos do treinamento da rede neural, identificar a       #
#   melhor configuração de hiperparâmetros (maior acurácia) e gerar um painel visual (dashboard) interativo.     #
#                                                                                                                #
# - O painel inclui gráficos de dispersão multivariados e um mapa de calor (heatmap) para analisar a correlação  #
#   entre o número de neurônios, taxa de aprendizado, épocas executadas e a acurácia final do modelo.            #
#                                                                                                                #
##################################################################################################################

# Leitura de Dados
# Esta seção é responsável por definir o caminho do arquivo CSV contendo os resultados e carregá-lo em um 
# DataFrame utilizando a biblioteca pandas. Caso o arquivo não seja encontrado, a execução é interrompida.

ARQUIVO_CSV = "resultados_Finais.csv" 

try:
    df = pd.read_csv(ARQUIVO_CSV)
except FileNotFoundError:
    print(f"Erro: O arquivo '{ARQUIVO_CSV}' não foi encontrado na pasta atual.")
    exit()


# Busca da Melhor Configuração
# Esta seção identifica a linha do DataFrame que contém o maior valor na coluna de acurácia. 
# Em seguida, extrai e exibe no console os hiperparâmetros que geraram esse resultado ótimo,
# servindo como um resumo rápido do melhor modelo encontrado durante a automação de testes.

indice_melhor = df['acuracia'].idxmax()
melhor_resultado = df.loc[indice_melhor]

print(f"Acurácia:             {melhor_resultado['acuracia']:.4f} ({(melhor_resultado['acuracia']*100):.2f}%)")
print(f"Neurônios:            {int(melhor_resultado['nro_neuronios'])}")
print(f"Taxa de Aprendizado:  {melhor_resultado['taxa_aprendizado']:.5f}")
print(f"Épocas Executadas:    {int(melhor_resultado['nro_epocas_executada'])}")
print(f"Paciência:            {int(melhor_resultado['paciencia'])}")


# Geração do Painel Visual (Dashboard)
# Esta seção configura uma figura do matplotlib subdividida em 4 eixos (2x2). Ela plota três gráficos de 
# dispersão cruzando diferentes variáveis (Neurônios, Taxa de Aprendizado em escala logarítmica e Tempo de 
# Treinamento) contra a acurácia, utilizando as escalas de cores para representar dependências multivariadas.
# O quarto quadrante exibe um mapa de calor (Heatmap) utilizando a correlação de Pearson. Em todos os gráficos
# de dispersão, a configuração com a maior acurácia absoluta é destacada com um marcador vermelho (estrela).

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análise Comportamental dos Hiperparâmetros', fontsize=18, weight='bold')

sc1 = axes[0, 0].scatter(df['nro_neuronios'], df['acuracia'], 
                         c=df['taxa_aprendizado'], cmap='viridis', s=60, alpha=0.8, edgecolors='k')
axes[0, 0].set_title('Impacto do Número de Neurônios', fontsize=14)
axes[0, 0].set_xlabel('Número de Neurônios (Capacidade)')
axes[0, 0].set_ylabel('Acurácia')
axes[0, 0].grid(True, linestyle='--', alpha=0.6)
axes[0, 0].scatter(melhor_resultado['nro_neuronios'], melhor_resultado['acuracia'], color='red', s=150, marker='*', edgecolor='k')
cbar1 = fig.colorbar(sc1, ax=axes[0, 0])
cbar1.set_label('Taxa de Aprendizado')

sc2 = axes[0, 1].scatter(df['taxa_aprendizado'], df['acuracia'], 
                         c=df['nro_neuronios'], cmap='coolwarm', s=60, alpha=0.8, edgecolors='k')
axes[0, 1].set_title('Impacto da Taxa de Aprendizado', fontsize=14)
axes[0, 1].set_xlabel('Taxa de Aprendizado (Escala Log)')
axes[0, 1].set_ylabel('Acurácia')
axes[0, 1].set_xscale('log') 
axes[0, 1].grid(True, linestyle='--', alpha=0.6)
axes[0, 1].scatter(melhor_resultado['taxa_aprendizado'], melhor_resultado['acuracia'], color='red', s=150, marker='*', edgecolor='k')
cbar2 = fig.colorbar(sc2, ax=axes[0, 1])
cbar2.set_label('Número de Neurônios')

sc3 = axes[1, 0].scatter(df['nro_epocas_executada'], df['acuracia'], 
                         c=df['nro_neuronios'], cmap='plasma', s=60, alpha=0.8, edgecolors='k')
axes[1, 0].set_title('Tempo de Estagnação (Épocas antes do Stop)', fontsize=14)
axes[1, 0].set_xlabel('Épocas Executadas')
axes[1, 0].set_ylabel('Acurácia')
axes[1, 0].grid(True, linestyle='--', alpha=0.6)
axes[1, 0].scatter(melhor_resultado['nro_epocas_executada'], melhor_resultado['acuracia'], color='red', s=150, marker='*', edgecolor='k')
cbar3 = fig.colorbar(sc3, ax=axes[1, 0])
cbar3.set_label('Número de Neurônios')

axes[1, 1].set_title('Relevância dos Parâmetros para a Acurácia', fontsize=14)
correlacoes = df.corr()[['acuracia']].sort_values(by='acuracia', ascending=False).drop('acuracia')
sns.heatmap(correlacoes, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, ax=axes[1, 1], cbar=False, linewidths=1, edgecolor='k')
axes[1, 1].set_ylabel('Hiperparâmetros')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()