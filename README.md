# Processo seletivo Bio Chaves 2019
##### Proposta de solução para resolver o problema apresentado no processo seletivo de 2019 do grupo de pesquisa Bio Chaves <http://biochaves.com/processo-seletivo/>

### Introdução
A solução proposta é baseada na identificação da mão a partir da assinatura do seu formato (*shape signature*) utilizando os descritores de fourier da função distância do centroide (*Centroid Distance Function*) em conjunto com redes neurais.\
A função distância do centroide  é formada pelas distâncias euclidianas entre o centroide de um objeto e os pontos que formam o seu contorno. Em processamento de imagens a função distância do centroide, às vezes chamada assinatura da forma, é considerada um descritor do formato de objetos. O trabalho de COSGRIFF (1960) foi o primeiro a propor o uso dos descritores de fourier para identificar objetos a partir da sua assinatura da forma. A ideia é representar o contorno do objeto por um conjunto de números que representem as frequências que formam o contorno do objeto. No caso dos descritores de fourier este conjunto é formado pelos coeficientes da transformada discreta de fourier da função distância do centroide (ALHILAL et al., 2015) (UCI DEPARTMENT OFMATHEMATICS, 2011).\
Os descritores de fourier podem ser utilizados em conjuntos com métodos de classificação para produzir classificadores capazes de classificar contornos *out-of-samples*, na solução proposta o método de classificação utilizado para solução do problema foram as redes neurais.\

[1] COSGRIFF, R. L. Identification of shape.OHIO STATE UNIVERSITY RESEARCHFOUNDATION, COLUMBUS, REP. 820-11,. ASTIA AD 254 792, 1960\
[2] ALHILAL, M. S.; SOUDANI, A.; AL-DHELAAN, A. Image-based object identification forefficient event-driven sensing in wireless multimedia sensor networks.International Journal ofDistributed Sensor Networks, SAGE Publications, v. 11, n. 3, p. 850869, jan. 2015. Disponível em: <https://doi.org/10.1155/2015/850869>.\
[3]UCI DEPARTMENT OF MATHEMATICS.Centroid Distance Function and theFourier Descriptor with Applications to Cancer Cell Clustering. 2011. Disponível em: <https://www.math.uci.edu/icamp/summer/research_11/klinzmann/cdfd.pdf>\


### Metodologia
Todos os códigos foram escritos na linguagem Python v3.6 com os pacotes NumPy v1.16.2, SciPy v1.2.1, Pandas v0.24.1, Tensorflow-gpu v1.12.0 e Keras v2.2.4.

A solução proposta é composta por 3 etapas:
1 - Preparar os dados de treinamento e avaliação
2 - Treinar os classificadores com os dados de treinamento
3 - Utilizar os classificadores com os dados de avaliação para resolver o problema

O conjunto de treinamento é composto por um conjunto de 18 contornos de mãos que pertencem a 9 indivíduos distintos. Cada contorno é definido por 14 pontos chaves da mão. De cada contorno é extraída a função distância do centroide e, em seguida, é realizada a transformada discreta de fourier desta função. Os coeficientes das frequências que formam o contorno de cada mão são armazenadas no arquivo "trainData.csv". Este processo é realizado em "A_processRawData.py". Os 14 pontos que definem o contorno da mão em "marcaMao.jpg" foram identificados manualmente a partir de sua imagem. Estes pontos também deram origem a uma função distância do centroide cujos descritores de fourier foram armazenados em "targetHand.csv". Este processo é realizado em "B_processTargetHand.py"

A informação contida em "trainData.csv" é utilizada para treinar redes neurais que utilizam ReLu, tangente hiperbólica e sigmoide como funções de ativação de seus neurônios. As redes variam em tamanho e quantidade de neurônios por camada. O processo de treinamento é realizado em "C_trainClassifiers".

Após treinadas as redes neurais que obtiveram acurácia de treinamento maior do que 90% foram utilizadas para construção de um classificador de votação por maioria (majority voting ou Majority rule) que utiliza os coeficientes de fourier extraídos do contorno da mão em "marcaMao.jpg" pra inferir o nome de quem deixou à marca . O processo de predição é realizado em "D_evaluateModels.py".

### Resultados
Foram treinadas 72 redes neurais, destas apenas 14 obtiveram acurácia de treinamento superior a 90%. Quando submetidas aos descritores de fourier extraídos do contorno da mão em "marcaMao.jpg" todas as redes apontaram que **Thiago, de quem foram coletadas as impressões "thi01" e "thi02", é o responsável por deixar à marca da mão em "marcaMao.jpg"**.