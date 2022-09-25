#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model


# In[3]:


dp = pd.read_csv('C:/Users/lokes/OneDrive/Desktop/Analytics/Salary_data_set.csv')


# In[4]:


dp


# In[5]:


dp.Test_score.median()


# In[7]:


dp.Test_score = dp.Test_score.fillna(dp.Test_score.median())


# In[8]:


dp


# In[9]:


dp.Experience = dp.Experience.fillna(0)


# In[10]:


dp


# In[14]:


reg = linear_model.LinearRegression()
reg.fit(dp[['Experience','Test_score','Interview_score']],dp.Salary)


# In[15]:


reg.coef_


# In[16]:


reg.intercept_


# In[17]:


plt.figure(figsize=(10,7))
sns.heatmap(dp.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()


# In[19]:


reg.predict([[2,9,6]])


# In[20]:


reg.predict([[12,10,10]])


# In[ ]:




