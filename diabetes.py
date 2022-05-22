#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv("diabetes.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df['Outcome'].value_counts()


# In[7]:


df.groupby('Outcome').mean()


# In[8]:


X=df.drop(columns='Outcome',axis=1)
y=df['Outcome']


# In[9]:


print(X)


# In[10]:


print(y)


# In[11]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)


# In[12]:


print(X.shape,X_train.shape,X_test.shape)


# In[13]:


classifier=svm.SVC(kernel='linear')


# In[14]:


classifier.fit(X_train,y_train)


# In[15]:


X_pred=classifier.predict(X_test)


# In[16]:


acc=accuracy_score(X_pred,y_test)


# In[17]:


print(acc)


# In[18]:


input_data=(5,106,72,19,175,25.0,0.587,51)


# In[19]:


input_data_as_numpy_array=np.asarray(input_data)


# In[20]:


input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


# In[21]:


prediction=classifier.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
    


# In[25]:


import pickle


# In[26]:


file='trained_model'
pickle.dump(classifier,open(file,'wb'))


# In[27]:


load_model=pickle.load(open('trained_model','rb'))


# In[28]:


input_data=(5,106,72,19,175,25.0,0.587,51)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=load_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")


# In[ ]:





# In[ ]:




