#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")
import pickle


# In[2]:


df=pd.read_csv(r"C:\Users\VAISHNAVI\Downloads\Breast_cancer_data.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


x=df.iloc[:,:5]
y=df.iloc[:,5]


# In[6]:


x


# In[7]:


y


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=40,random_state=42)


# In[9]:


X_train.head()


# In[10]:


X_test.head()


# In[11]:


y_train.head()


# In[12]:


y_test.head()


# In[13]:


from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train = sc.fit_transform(X_train)  
x_test = sc.transform(X_test)


# In[14]:


x_train


# In[15]:


x_test


# In[16]:


from sklearn.naive_bayes import GaussianNB  
clf = GaussianNB()  
clf.fit(x_train, y_train)  


# In[17]:


y_pred = clf.predict(x_test)


# In[18]:


y_pred


# In[19]:


with open('model.pickle', 'wb') as f:
    pickle.dump(clf, f)


# In[ ]:





# In[20]:


from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)  


# In[21]:


cm


# In[ ]:





# In[ ]:




