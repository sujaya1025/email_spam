#!/usr/bin/env python
# coding: utf-8

# # spam mail classification

# Apply Naive Bayes to perform predictions on given dataset

# In[50]:


#import required libraries
import pandas as pd
import numpy as np


# In[32]:


#load the dataset
ds=pd.read_csv("Downloads\spam.csv")


# In[33]:


ds.head(10)


# In[34]:


#replacing spam with 1 and ham with 0
def new(column):
    ds[column].replace({'spam':1,'ham':0},inplace=True)


# In[35]:


for column in ds:
    new(column)


# In[36]:


ds.head()


# In[37]:


#load text to X variable
X=ds['v2']


# In[38]:


X.head()


# In[39]:


#load dependent attribute(v1) to y variable
y=ds['v1']


# In[40]:


y.head()


# In[41]:


#split test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y) 


# In[42]:


#convert the text data into numerical features using countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X_trainCV=cv.fit_transform(X_train.values)
X_trainCV.toarray()[:3]


# In[43]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_trainCV,y_train)


# In[44]:


#transforming test data
X_testCV=cv.transform(X_test)


# In[46]:


#Predicting the result of test set  
y_pred= model.predict(X_testCV)


# In[47]:


#Create Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test,y_pred) 


# In[51]:


cm


# In[52]:


#calculate the accuracy of the model
accuracy=(cm[0][0]+cm[1][1])/np.sum(cm)


# In[54]:


print(accuracy*100)


# In[55]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




