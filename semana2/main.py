#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[171]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[26]:


black_friday.shape


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[27]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return(black_friday.shape)


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[30]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return (black_friday[(black_friday['Age'] == '26-35') & (black_friday['Gender']=="F")].shape[0])


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[157]:


def q5():
    # Retorne aqui o resultado da questão 5.
    result =((black_friday.isna().sum() / black_friday.shape[0]).max())
    return float(result)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[69]:


black_friday.isna().sum().max()


# In[9]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return int(black_friday.isna().sum().max())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday['Product_Category_3'].value_counts().index[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[94]:


black_friday.info()


# In[191]:


def q8():
    # Retorne aqui o resultado da questão 8.
    normalized_df=((black_friday['Purchase'] - black_friday['Purchase'].min()) / (black_friday['Purchase'].max() - black_friday['Purchase'].min())).mean()
    return float(normalized_df)


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[103]:


def q9():
    # Retorne aqui o resultado da questão 9.
    from sklearn.preprocessing import StandardScaler
    data = black_friday[['Purchase']]
    scaler = StandardScaler()
    x_stand = scaler.fit_transform(data)
    count = 0
    for x in x_stand:
        if -1<= x <= 1:
            count += 1

    return int(count)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[130]:


def q10():
    # Retorne aqui o resultado da questão 10.
    a = black_friday[(black_friday['Product_Category_3'].isna()) & (black_friday['Product_Category_2'].isna())].shape[0]    == black_friday[(black_friday['Product_Category_2'].isna())].shape[0]
    return a

