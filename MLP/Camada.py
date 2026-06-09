import numpy as np
np.random.seed(42)

##################################################################################################################
#  Classe Camada                                                                                                 #
#                                                                                                                #
# - A classe Camada é onde estão implementadas as funções que realizam os cálculos matemáticos principais,       #
# são esses os calculos de y_in, deltas, entre outras variáveis importantes para o processamento dos neurônios.  #
#                                                                                                                #
# - Para cálculos como estes é na classe camada que está presente a matriz de pesos referente ao seus neurônios, #
# dessa forma, os neurônios são representados como as linhas da matriz de pesos presente na camada.              #
#                                                                                                                #
# P.S.: Em diversos cálculos foram utilizadas função do numpy, principalmente em cálculos envolvendo matrizes,   #
# essa decisão foi tomada para otimização do código, mas cada cálculo será exemplificado na função referente.    #
#                                                                                                                #
##################################################################################################################

class Camada:
    
    # Construtor
    # No construtor da Camada cria-se os neurônios (a matriz de pesos) utilizando como parâmetros a quantidade de neurônio
    # que define a quantidade de linhas da matriz de pesos e a quantidade de entradas que define quantos pesos cada neurônio
    # terá (as colunas).
    # Iniciam-se também as variáveis values_in que será um array representando a entrada de cada neurônio, que será utilizado
    # para o cálculo de deltas no BackPropagation. Além da variável camada que armazenará a quantidade de neurônios apenas para
    # facilitar cálculos no Gerenciador.

    # def __init__ (int qtdNeurons, int qtdEntradas)
    # | qtdNeurons: inteiro representando a quantidade de neurônios desejados nesta camada.
    # | qtdEntradas: inteiro representando a quantidade de entradas que a camada receberá (incluindo o bias).

    def __init__(self, qtdNeurons:int, qtdEntradas:int):
        self.pesos = np.random.uniform(-0.5, 0.5, size=(qtdNeurons, qtdEntradas))
        self.values_in = []
        self.camada = qtdNeurons

    # Função de ativação
    # A função de ativação será responsável por processar o y_in e transformá-lo na entrada do próximo neurônio.
    # A função utilizada nessa implementação foi a tangente hiperbólica, isso porque não é uma função linear,
    # além de ser derivável em todos os seus pontos, o que permite a resolução de problemas mais complexos e não-lineares,
    # como é o caso do XOR e do CARACTERES_COMPLETO.
    # P.S.: A função do numpy utilizada realiza o cálculo senh(y_in)/cosh(y_in)

    # def func_ativ_tanh (float y_in)
    # | y_in: valor flutuante que representa a saída do neurônio pré-processamento.
    # | return: valor flutuante que representa a saída já processada na função de ativação.

    def func_ativ_tanh(self, y_in):
        return np.tanh(y_in)

    # Derivada da função de ativação
    # A derivada da função de ativação é a função que será utilizada para o cálculo de delta para o BackPropagation.
    # Ela representa o processo contrário feito na função de ativação, e receberá também um y_in referente ao values_in,
    # que será o valor ainda não processado pela função de ativação.

    # def derivada_func_ativ_tanh (float y_in)
    # | y_in: valor flutuante que representa a saída do neurônio pré-processamento.
    # | return: valor flutuante que representa a saída já processada na derivada da função de ativação.

    def derivada_func_ativ_tanh(self, y_in):
        return 1.0 - np.tanh(y_in)**2

    # FeedFoward
    # A função camadaFeedFoward é responsável por calcular as saídas da camada, que são salvas em um arranjo,
    # neste arranjo cada posição representa a resposta de um neurônio, primeiro é calculado o y_in ou values_in,
    # calculado fazendo-se o somatório de todos os pesos multiplicados pelas entradas de cada neurônio.
    # Para esse calculo utiliza-se a função dot do numpy, que para cada linha da matriz de pesos, multiplica os
    # valores das entradas em seu indíce referente. Ou seja, o somatório pesos[i][j] * entradas[j] é o retorno
    # para cada uma das posições i do arranjo final.
    
    # def camadaFeedFoward (list entradas)
    # | entradas: arranjo floats representando as entradas da camada.
    # | return: arranjo de floats com as saídas de cada neurônio da camada para as respectivas entradas.
      
    def camadaFeedFoward(self, entradas:list):
        self.values_in = np.dot(self.pesos, entradas)
        saidasCamada = self.func_ativ_tanh(self.values_in)
        saidasCamada = np.insert(saidasCamada, 0, 1)
        return saidasCamada

    # BackPropagation
    # A função camadaBackPropagation é responsável por calcular o caminho contrário da função FeedFoward,
    # isso é, ela calcula os deltas que serão utilizados na atualização dos pesos da camada. Para isso a mesma
    # função dot do numpy é utilizada, porém dessa vez utilizando os pesos da camada da frente como matriz
    # e os deltas calculados também na camada da frente.
    # Esses deltas calculados são multiplicados pelos valores y_in calculados na etapa feed foward após passarem
    # na derivada da função de ativação. Esse processo é feito para considerar o erro nas respostas na hora de 
    # atualizar os pesos.

    # def camadaBackPropagation (list deltas, list pesosCamadaFrente)
    # | deltas: arranjo de floats contendo os deltas calculados na função BackPropagation da camada à frente.
    # | pesosCamadaFrente: matriz de floats que representam os pesos da camada à frente.
    # | return: array de floats que representam os deltas desta camada, os quais serão utilizados na atualização dos pesos.

    def camadaBackPropagation(self, deltas:list, pesosCamadaFrente:list):
        deltas_in = np.dot(pesosCamadaFrente[1:], deltas)
        deltasCamada = deltas_in * self.derivada_func_ativ_tanh(self.values_in)
        return deltasCamada

    # Output BackPropagation
    # A função camadaOutputBackPropagation é usada com o mesmo princípio da função anterior, porém na camada de saída.
    # Uma função diferente é utilizada pois, na camada de saída, o erro não estará nos deltas da camada à frente
    # (já que não há uma camada à frente), mas sim na comparação direta da saída da camada com as variáveis
    # target.

    # def camadaOutputbackPropagation (list y_k, list t_k)
    # | y_k: arranjo de floats com as saídas calculadas no método feedFoward da camada de saída.
    # | t_k: arranjo de float com as variáveis target do banco de dados.
    # | return: array de floats com os deltas da camada de saída.

    def camadaOutputBackPropagation(self, y_k:list, t_k:list):
        erro = np.array(t_k) - np.array(y_k[1:])
        deltasCamada = erro * self.derivada_func_ativ_tanh(self.values_in)
        return deltasCamada
    
    # Update
    # A função camadaUpdate irá basicamente atualizar a matriz de pesos da camada, utilizando como
    # base os deltas (onde o erro é considerado), as entradas e a taxa de aprendizado (que definirá
    # o quanto essa atualização alterará os pesos). Dessa forma os pesos são atualizados considerando
    # todas as variáveis com as quais eles interagem.

    # def camadaUpdate (list deltas, list listaEntradas, float taxaAprendizado)
    # | deltas: arranjo de floats cntendo os deltas calculados na etapa BackPropagation dessa camada.
    # | listaEntradas: arranjo de floats contendo as entradas dessa camada.
    # | taxaAprendizado: valor flutuante representando um hiper-parâmetro que definirá o quão relevante será cada atualização.

    def camadaUpdate(self, deltas:list, listaEntradas:list, taxaAprendizado:float):
        self.pesos += np.outer(deltas * taxaAprendizado, listaEntradas)