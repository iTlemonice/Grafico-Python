#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import seaborn as sns

uri = 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'
dados = pd.read_csv(uri)

mapa = {
    'unfinished': 'nao_finalizado',
    'expected_hours': 'horas_esperadas',
    'price': 'preco'
}

dados = dados.rename(columns=mapa)

troca = {   # INVERTENDO OS VALORES DENTRO DA COLUNA
    1: 0,
    0: 1
}

dados['finalizado'] = dados.nao_finalizado.map(troca)   # INVERTENDO OS VALORES DENTRO DA COLUNA

print(dados)

tabela = sns.scatterplot(x='horas_esperadas', y='preco', hue='finalizado',data=dados)

print(tabela)   # IMPRIMIRIA A TABELA, PORÉM, O PYCHARM N IMPRIME DESSE MODO, NO COLABORATORY IMPRIME.

tabela = sns.scatterplot(x='horas_esperadas', y='preco', hue='finalizado', data=dados)
    # AQUI ELE PEGA A COLUNA 'FINALIZADO' E PINTA DE OUTRA COR, PARA A MELHOR VIZUALIZAÇÃO.

print(tabela)

x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']


# In[41]:


sns.relplot(x='horas_esperadas', y='preco', hue='finalizado',col='finalizado', data = dados)
# AQUI, A GENTE USA O RELPLOTO PARA A GENTE PODER IMPRIMIR DOIS GRAFICOS DIFERENTES, UM MOSTRANDO NO CASO, OS VALORES DOS
# PROJETOS FINALIZADOS E O OUTRO COM OS PROJETOS NÃO FINALIZADOS


# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np

SEED = 5
np.random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

print(f'Treinaremos com {len(treino_x)} e testaremos com {len(teste_x)}')

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes)*100
print(f'A nossa acuraccy foi de: {acuracia:.2f}%')


# In[43]:


import numpy as np
chute_mateus = np.ones(540)
acuracia = accuracy_score(teste_y, chute_mateus)*100
print(f'A nossa acuraccy de Mateus foi de: {acuracia:.2f}%')
# OU SEJA, A GENTE CHUTANDO COM TODAS AS RESPOSTAS SIM, TEMOS UMA TAXA DE ACERTO PARECIDA COM A DA NOSSA INTELIGENCIA. E ISSO É 
# PÉSSIMO, POIS PRECISAMOS DE UMA TAXA DE ACERTOS MUITO MAIOR, BEIRANDO 100%


# In[44]:


sns.scatterplot(x='horas_esperadas', y='preco', hue=teste_y, data=teste_x)
# A NOSSA TABELA SUPORTA UMA ARRAY EM HUE, PARA A GENTE MUDAR A COR


# In[45]:


# COM ESSE PROBLEMA DE UM ALGORITMO QUE NÁO TEM UMA TAXA DE CONFIANÇA ELEVADA, A GENTE VAI VARRER ELE INTEIRO, E TENTAR PERCEBER
# AONDE ESTÁ ESSE ERRO.

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()

print(x_min, x_max, y_min, y_max)


# In[46]:


import numpy as np
pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)


# In[47]:


# O MESHGRID USADO PARA MESCLAR OS DOIS EIXOS
xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()] 


# In[48]:


z = modelo.predict(pontos)
z = z.reshape(xx.shape)
z


# In[49]:


import matplotlib.pyplot as plt

plt.contourf(xx, yy, z, alpha=0.3)
# O CONTOURF SERVE PARA MOSTRAR UMA RETA, QUE SERIA A NOSSA CURVA DE PROJETOS FINALIZADOS. MUDANDO O SEED LA EM CIMA PARA 5
# A GENTE OBTEM OUTRO RESULTADO, QUE NO CASO SERIA A RETA CONTANDO O GRAFICO PERTO DO MEIO.

plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1.5)
# FIZEMOS UM GRAFICO UM POUCO DIFERENTE, NA VERDADE ELE É BEM PARECIDO COM O OUTRO, PORÉM COM FUNCIONALIDADES DIFERENTES
# O VALOR DE S SERÁ O TAMANHO DOS PONTOS


# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SEED = 5
np.random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

print(f'Treinaremos com {len(treino_x)} e testaremos com {len(teste_x)}')

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes)*100
print(f'A nossa acuraccy foi de: {acuracia:.2f}%')

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()] 

z = modelo.predict(pontos)
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, alpha=0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1.5)


# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

SEED = 5
np.random.seed(SEED)

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

print(f'Treinaremos com {len(treino_x)} e testaremos com {len(teste_x)}')

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes)*100
print(f'A nossa acuraccy foi de: {acuracia:.2f}%')


# In[52]:


data_x = teste_x[:,0]
data_y = teste_x[:,0]

x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()] 

z = modelo.predict(pontos)
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, alpha=0.3)
plt.scatter(data_x, data_y, c=teste_y, s=1.5)
# DEU ERRADO, PORÉM, DÁ PARA ENTENDER COMO FUNCIONA A MANIPULAÇÃO DA BIBLIOTECA DE PLOT E TAMBEM DO SKLEARN


# In[ ]:




