#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


#Download of dataset : https://datahub.io/zelima1/finance-vix
df = pd.read_csv('D:/finance-vix.txt',
                sep=',',
                infer_datetime_format=True,
                index_col=False)
df.fillna(0, inplace=True)
#Drop date from dataset
df.drop('Date', axis=1, inplace=True)

print(df.head())
print(df.describe())


# In[4]:


#Here we need to specify Features and Label (split dataset into features and target (VIX Close))
X = df.drop("VIX Close", axis=1)
Y = df["VIX Close"]

#Split dataset to train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

print('X_train shape is :', X_train.shape)
print('Y_train shape is :', Y_train.shape)
print('X_test shape is :', X_test.shape)
print('Y_test shape is :', Y_test.shape)


# In[5]:


#Scale our data to prevent failing the model
scaler = StandardScaler()
scaler.fit(X_train)

Xtrain_scaled = scaler.transform(X_train)
Xtest_scaled = scaler.transform(X_test)


# # Set Neural Network

# In[6]:


#In this case, we do not need convolutuonal layer as there are one dimension, while in image we have two dimensions
model = Sequential()
model.add(Dense(128, input_dim=3, activation='relu'))
model.add(Dense(64, activation='relu'))
#Output layer
model.add(Dense(1,  activation='linear'))
#Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()
#Fit the model
history = model.fit(Xtrain_scaled, Y_train, epochs=100, validation_split=0.2)


# In[7]:


#Plotting training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validataion loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[8]:


acc = history.history['mae']
val_acc = history.history['val_mae']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[9]:


#Predict the model using test data
pred = model.predict(Xtest_scaled[:10])
print('Predicted values are {}'.format(pred))
print('Real Values are {}'.format(Y_test[:10]))


# In[10]:


#Comparison with other models 
#mse = mean squared error, 
#mae = mean absolute error
mse_neural, mae_neural = model.evaluate(Xtest_scaled, Y_test)
print('Mean Squared Error from neural net: {}'.format(mse_neural))
print('Mean Absolute Error from neural net: {}'.format(mae_neural))


# In[ ]:




