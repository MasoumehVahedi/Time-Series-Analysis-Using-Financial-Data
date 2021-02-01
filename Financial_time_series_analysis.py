#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler


# In[3]:


dataset = pd.read_csv('D:/finance-vix.txt',
                     sep=',',
                     parse_dates=[0],
                     infer_datetime_format=True,
                     index_col=False,
                     na_values=['nan', '?'])

#Fill na values with 0
dataset.fillna(0, inplace=True)
display(dataset.head())

#Separate 'Date' from dataframe and convert to 'to_datetime'
date_col = dataset['Date'] 
date_col = pd.to_datetime(date_col)

#Drop date from dataset
dataset = dataset.drop('Date', axis=1)

#variable for training
values = dataset.values
values = values.astype('float32')


# In[4]:


#All variables without 'Date'
dataset.head()


# In[5]:


#Plotting values
values_plot = dataset.tail(3000)
values_plot.plot.line()


# In[6]:


#Because of using tanh and sigmoid in LSTM that are sensitive to magnitade, values have to be normalized
#Normalizing variables
scaler = StandardScaler()
scaler = scaler.fit(values)
values_scaled = scaler.transform(values)
scaled = pd.DataFrame(values_scaled)


# In[7]:


scaled.head()


# In[8]:


#For LSTM network, input data should be shaped into n_sample * timesteps
# In this case, n_features is 2. we will make timesteps= 3
# So, n_samples is equal to 5

x_train = []   #training series
y_train = []   #prediction

#Number of days that we want to predict to the future
num_future = 1
#Number of past days that we plan to predict the future based on it
num_past = 15

for i in range(num_past, len(values) - num_future + 1):
    x_train.append(values_scaled[i - num_past:i, 0:values.shape[1]])
    y_train.append(values_scaled[i + num_future - 1:i + num_future, 0])
    
X_train, Y_train = np.array(x_train), np.array(y_train)

#Print train X and Y shape
print("X_train shape is: {}".format(X_train.shape))
print("Y_train shape is: {}".format(Y_train.shape))


# ## Set LSTM model

# In[11]:


model = Sequential()
model.add(LSTM(units=64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(Y_train.shape[1]))

#Compile the model
model.compile(loss='mse', optimizer='adam')
model.summary()


# In[12]:


#fit the model
history_lstm = model.fit(X_train, Y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=1)


# In[13]:


plt.plot(history_lstm.history['loss'], label='Training loss', color='red')
plt.plot(history_lstm.history['val_loss'], label='Validation loss', color= 'green')
plt.legend()
plt.show()


# ## Predicting

# In[15]:


#We want to predict for 70 days
#So, we can change the number of future for prediction
num_future = 70
#To predict time from the last date in dataset: date_col[-1]
predict_dates = pd.date_range(list(date_col)[-1], periods=num_future, freq='1d').tolist()

#Predicting
prediction = model.predict(X_train[-num_future:])
print(prediction[:5])


# In[20]:


#Now, we need to inverse transformation to rescale back to original range
#Here, 5 variables was used for transform
#So, we copy the values 5 times and get rid of them after inverse transform
copy_pred = np.repeat(prediction, values.shape[1], axis=-1)
y_pred = scaler.inverse_transform(copy_pred)[:,0]
print(y_pred[:10])


# In[21]:


#Read original dataset again
df = pd.read_csv('D:/finance-vix.txt',
                     sep=',',
                     parse_dates=[0],
                     infer_datetime_format=True,
                     index_col=False,
                     na_values=['nan', '?'])


# In[22]:


#Convert timestamp to date
pred_date = []
for time_i in predict_dates:
    pred_date.append(time_i.date())
    
df_predict = pd.DataFrame({'Date':np.array(pred_date), 'VIX Open':y_pred})
df_predict['Date'] = pd.to_datetime(df_predict['Date'])

original = df[['Date', 'VIX Open']]
original['Date'] = pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2015-5-1']

sns.lineplot(original['Date'], original['VIX Open'])
sns.lineplot(df_predict['Date'], df_predict['VIX Open'])


# In[ ]:




