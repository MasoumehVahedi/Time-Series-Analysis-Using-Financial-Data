#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# In[4]:


df = pd.read_csv('D:/finance-vix.txt',
                     sep=',',
                     parse_dates=[0],
                     infer_datetime_format=True,
                     index_col=False,
                     na_values=['nan', '?'])

#Fill na values with 0
df.fillna(0, inplace=True)

#Drop date from dataset
df = df.drop('Date', axis=1)
display(df.head())


# In[5]:


#To change the name of VIX Open column to OPEN 
df = df.rename(columns={'VIX Open':'Open'})
display(df.describe())


# In[6]:


#Here we need to specify features and labels
X = df.drop('Open', axis=1)
Y = df['Open']


# In[7]:


#Now split our data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

print('X_train shape {}'.format(X_train.shape))
print('Y_train shape {}'.format(Y_train.shape))
print('X_test shape {}'.format(X_test.shape))
print('Y_test shape {}'.format(Y_test.shape))


# In[8]:


#Scale data, otherwise model will fail
scaler = StandardScaler()
scaler = scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Set Decision Tree

# In[10]:


#Create an instance
model = DecisionTreeRegressor()
#Fit the DT model
model.fit(X_train_scaled, Y_train)
#Predict the model
y_pred = model.predict(X_test_scaled)

DT_mse = mean_squared_error(Y_test, y_pred)
DT_mae = mean_absolute_error(Y_test, y_pred)

print('Mean Squared Error from Decision Tree: {}'.format(DT_mse))
print('Mean Absolute Error from Decision Tree: {}'.format(DT_mae))


# # Set Random Forest

# In[11]:


model = RandomForestRegressor(n_estimators=30, random_state=30)
model.fit(X_train_scaled, Y_train)
y_pred = model.predict(X_test_scaled)
RF_mse = mean_squared_error(Y_test, y_pred)
RF_mae = mean_absolute_error(Y_test, y_pred)

print('Mean squared error using Random Forest: {}'.format(RF_mse))
print('Mean absolute error Using Random Forest: {}'.format(RF_mae))


# ### To investigate which parameters are most importance in random forest decision

# In[12]:


features = list(X.columns)
important_features = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(important_features)


# In[ ]:




