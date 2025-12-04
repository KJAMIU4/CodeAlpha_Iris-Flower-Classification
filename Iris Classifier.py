#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split


# #  Load Dataset

# In[2]:


df = pd.read_csv("C:\\Desktop\\PROJECT\\Iris.csv")
df


# In[6]:


#Divide into Features and Target(X,y)


X = df.drop("Species",axis=1)
y = df["Species"]


# In[19]:


#Split into Train/Test 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[21]:


#Load Classifier

model= DecisionTreeClassifier()
model.fit(X_train,y_train)


# In[25]:


# Make Prediction
y_pred = model.predict(X_test)


# In[29]:


accuracy_score(y_test,y_pred)


# In[33]:


print(classification_report(y_test,y_pred))


# In[ ]:




