#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[76]:


df = pd.read_csv('D:/finance-vix.txt',
                sep=',',
                infer_datetime_format=True,
                index_col=False)

#convert'Date' into 'to_datetime'
df['Date'] = pd.to_datetime(df['Date'])

#To change the name of MEDV column to PRICE 
df = df.rename(columns={'VIX Close':'Close'})
df = df.rename(columns={'VIX Open':'Open'})

print(df.head())
print(df.describe())


# In[38]:


plt.xlabel('Open')
plt.ylabel('Close')
plt.scatter(df.Open[:100], df.Close[:100], color='red', marker='+')
plt.show()


# # Set Linear Regression

# ## Univariate linear regression

# In[50]:


# X independent (Open)
# Y dependent - we are predictiong Y
X = df[['Open']]
Y = df['Close']

#Now split our data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

print('X_train shape {}'.format(X_train.shape))
print('Y_train shape {}'.format(Y_train.shape))
print('X_test shape {}'.format(X_test.shape))
print('Y_test shape {}'.format(Y_test.shape))


# In[40]:


#Scale data, otherwise model will fail
scaler = StandardScaler()
scaler = scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[46]:


#Create an instance of the model
reg = linear_model.LinearRegression()
#Training model (fit a line)
reg.fit(X_train_scaled, Y_train)
#Prediction model
y_pred_reg = reg.predict(X_test_scaled)


# In[42]:


mse_reg = mean_squared_error(Y_test, y_pred_reg)
mae_reg = mean_absolute_error(Y_test, y_pred_reg)

print('Mean Squared Error from linear regression: ', mse_reg)
print('Mean Absolute Error from linear regression: ', mae_reg)


# ## Multivariate linear regression

# In[73]:


#Drop 'Date' column
df = df.drop('Date', axis=1)

# X independent 
# Y dependent - we are predictiong Y (Close)
X = df.drop('Close', axis=1)
Y = df['Close']

#Now split our data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

print('X_train shape {}'.format(X_train.shape))
print('Y_train shape {}'.format(Y_train.shape))
print('X_test shape {}'.format(X_test.shape))
print('Y_test shape {}'.format(Y_test.shape))


# In[74]:


#Scale data, otherwise model will fail
scaler = StandardScaler()
scaler = scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[75]:


#Set linear regression
model_lr = linear_model.LinearRegression()
model_lr.fit(X_train_scaled, Y_train)
y_pred = model_lr.predict(X_test_scaled)
mse_lr = mean_squared_error(Y_test, y_pred)
mae_lr = mean_absolute_error(Y_test, y_pred)

print('Mean Squared Error from linear regression: {}'.format(mse_lr))
print('Mean Absolute Error from linear regression: {}'.format(mae_lr))


# ## Regression_statsmodel

# ## Univariate linear regression

# In[60]:


# X independent (Date)
# Y dependent - we are predictiong Y
X = df[['Open']]
Y = df['Close']

#Add constant value to independent variable
X_ = sm.add_constant(X)
#Fit the model
model = sm.OLS(Y, X_)
results = model.fit()
print(results.summary())


# In[61]:


display("rsquared is : {}".format(results.rsquared))
display("params is : {}".format(results.params))
display("fittedvalues is : {}".format(results.fittedvalues))


# ## Multivariate linear regression

# In[77]:


#Drop 'Date' column
df = df.drop('Date', axis=1)

# X independent 
# Y dependent - we are predictiong Y (Close)
X = df[['Open', 'VIX High', 'VIX Low']]
Y = df['Close']

#Add constant to independent variables
x = sm.add_constant(X)

#Fit the model
model = sm.OLS(Y, x)
results = model.fit()
print(results.summary())


# In[78]:


display("rsquared is : {}".format(results.rsquared))
display("params is : {}".format(results.params))
display("fittedvalues is : {}".format(results.fittedvalues))


# In[ ]:




