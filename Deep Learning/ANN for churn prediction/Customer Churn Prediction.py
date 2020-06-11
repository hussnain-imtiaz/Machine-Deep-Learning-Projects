#!/usr/bin/env python
# coding: utf-8

# # Bank Customer Churn Rate Prediction
# This notebook does end-to-end customer churn rate prediction using Artifacial Neural Networks (Deep Learning). 

# ## 1. Problem Definintion:
# Taking customer's data as input, we have to tell that whether customer will quit from the bank or not.
#   
# ## 2. Data:  
# Data comes from a toy data set, but its really close to original one.  
# Csv file of dataset can be found in same folder.
#   
# ## 3. Evaluation:  
# Model is evaluated by accuracy, and cross validation score.
# 
# ## 4. Features:
# Data Contains many features about the customer that can contribute towards its exit/not exit.  
#   
# Following are the important features.  
#   
# 1. CreditScore  
# 2. Geography
# 3. Gender 
# 4. Age
# 5. Tenure 
# 6. Balance 
# 7. NumOfProducts 
# 8. HasCrCard
# 9. IsActiveMember 
# 10. EstimatedSalary

# ### Getting the Tools ready

# In[51]:


data.columns


# In[1]:


#deep learning packages
import tensorflow as tf
#regular EDA and plotting libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#for data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ### Getting Data Ready:

# In[2]:


data=pd.read_csv("Churn_Modelling.csv")


# In[3]:


data.head()


# ### Data preprocessing:

# In[4]:


X=data.iloc[:,3:13]
y=data.iloc[:,13]


# ### Dealing with categorical values:

# In[5]:


def preprocessing(df):
    """
    Do label encoding and imputation.
    """
    for label,content in df.items():
        if not pd.api.types.is_numeric_dtype(content):
            df[label]=content.astype('category').cat.as_ordered()
            df[label]=pd.Categorical(content).codes
    


# In[6]:


preprocessing(X)


# In[7]:


X.info()


# In[8]:


X.head()


# ### Checking Missing Values:

# In[9]:


X.isna().any()


# ### Splitting into train,test set

# In[10]:


X=X.values
y=y.values


# In[11]:


y


# In[12]:


X_train,X_test,y_train,y_test=train_test_split(X,y,
                                              random_state=22,
                                              test_size=0.2)


# In[ ]:





# ### Feature Scalling

# In[13]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[14]:


len(X_test)


# ## 5. Modelling:
# Our data is preprocessed, now it's the time to do modeling.

# In[54]:


#importing modeling,avaluation and tunning packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix


# In[58]:


def model_building():
    # initializing the model
    ANN_clf=Sequential()

    # creating the input layer and first hidden layer
    ANN_clf.add(Dense(input_dim=10,output_dim=6,init="uniform",activation="relu"))

    #creating the second hidden layer

    ANN_clf.add(Dense(units=6,init="uniform",activation="relu"))

    #creating the output layer

    ANN_clf.add(Dense(units=1,init="uniform",activation="sigmoid"))

    #compiling the model
    ANN_clf.compile(optimizer=keras.optimizers.Adam(),
                loss="binary_crossentropy",
               metrics=["accuracy"])
    ANN_clf.build()
    return ANN_clf


# In[26]:


#building model
model=model_building()


# In[28]:


#fitting model
model.fit(x=X_train,
          y=y_train,
          nb_epoch=100,
         batch_size=30)


# ### 6. Evaluating the model 

# In[29]:


##evaluating the model with k-folds
classifier=KerasClassifier(build_fn=model_building,nb_epoch=100,batch_size=10)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,n_jobs=-1,cv=10)


# In[33]:


accuracies


# In[45]:


cross_val_accuracy=np.mean(accuracies)
cross_val_accuracy


# In[46]:


varience=np.var(cross_val_accuracy)
varience


# In[36]:


y_preds=model.predict(X_test)
y_preds


# In[37]:


y_preds=(y_preds > 0.5)
y_preds


# In[50]:


model.evaluate(X_test,y_test)


# #### Visualizing Model Confusions

# In[41]:


cm=confusion_matrix(y_test,y_preds)


# In[42]:


cm


# In[43]:


import seaborn as sns


# In[44]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues');


# ### 7. Tunning Our Model

# In[63]:


params_grid={
    'batch_size':[20,32],
    'epochs':[250,500]
    }

classifier=KerasClassifier(build_fn=model_building)
grid_model=GridSearchCV(estimator=classifier,
                       param_grid=params_grid,
                       cv=5)
grid_model.fit(X_train,y_train)


# In[64]:


grid_model.best_params_


# In[65]:


grid_model.best_score_


# ### 8. Training Our Model with the best Parameters Found

# In[67]:


model=model_building()
model.fit(X_train,
         y_train,
         nb_epoch=150,
         batch_size=32,)


# In[68]:


model.evaluate(X_test,y_test)


# Ops! Our model is not improved.....  
# But we can do more experiments do get high results.

# ## What we can do more to get high accuracy?  
#   
# Well, we can :  
# 1. Change ANN Artitechure.  
# 
# 2. Do more search for hyperparameters.  
# 
# 3. Increase data. 
#   
# and lot more.
#   
