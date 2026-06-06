# Projeto---Multilayer-Perceptron

Projeto realizado na disciplina de IA ministrada por Sarajane Marques Peres, no quinto semestre do curso de Sistemas de Informação.

### Integrantes
- Bruno Tenorio Park
- Gustavo Gouveia Franca do Nascimento
- Lucas Giovani Santos Ross
- Tiago Silveira Almeida
- Vinicius Chirnev Panhoca

## Objetivo
O projeto tem como objetivo o estudo da arquitetura de Multilayer Perceptrons, utilizando como base para construção da arquitetura os datasets de problemas lógicos AND, OR e XOR. E depois aprofundando a arquitetura e seu potencial usando como testes os conjuntos de dados caracteres_completos.

O algoritmo atual já foi capaz de atingir 83% de acurácia no conjunto de testes, se usados os hiperparâmetros corretos. O algoritmo conta com um classe de Log, que exibe nos arquivos de log após a execução, avaliações completas da rede neural executada, para fins de regulagem dos hiperparâmetros.

## Arquitetura
O diretório principal é o MLP, dentro dele será criada a pasta de log e todos os arquivos de log após a execução do main.py que é o arquivo onde podem ser definidos os hiperparâmetros desejados.
Os cálculos mais complexos são realizados pela classe Camada e o gerenciamento da rede e chamada das funções é feita na classe Gerenciador. Os logs são criados na classe Logger.

Para checagem da avaliação e avaliação dos hiperparâmetros deve-se analisar principalmente o arquivo log_avaliacao, onde está a avaliação completa da rede neural após treinamento e teste.
