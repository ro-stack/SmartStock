#!/usr/bin/env python
# coding: utf-8

# # LightGBM

# In[1]:


# Import modules
import numpy as np 
from sklearn.model_selection import train_test_split
import pandas_datareader as web
import datetime as dt
import pandas as pd
import talib


# In[2]:


# Retrieve Data from Yahoo & View it

start = dt.datetime(1970,1,1).date()
end = pd.to_datetime("today").date()
company = "NFLX" # Anything from Yahoo - check ticker on website - NFLX (Stock) GBP=X (GBP/USD) GBPEUR=X(GBP/EUR)
dataset = web.DataReader(company,"yahoo",start,end)

df=pd.DataFrame(dataset)

print('Newest Data:')
print(df.tail())
print('Earliest Data:')
print(df.head())

df.shape


# In[3]:


# Add Technical Indicators to Dataframe 
high = np.array(df.High)
low = np.array(df.Low)
close = np.array(df.Close)
 
df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
 
df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
 
df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
 
df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
 
df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
 
df['CMO'] = talib.CMO(close, timeperiod=14)
 
df['DX'] = talib.DX(high, low, close, timeperiod=14)
 
df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
 
df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
 
df['MOM'] = talib.MOM(close, timeperiod=10)
 
df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
 
df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
 
df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
 
df['ROC'] = talib.ROC(close, timeperiod=10)
 
df['ROCP'] = talib.ROCP(close, timeperiod=10)
 
df['ROCR100'] = talib.ROCR100(close, timeperiod=10)
 
df['RSI'] = talib.RSI(close, timeperiod=14)
df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
 
df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

# Added more

df['DEMA'] = talib.DEMA(close, timeperiod=30)

df['EMA'] = talib.EMA(close, timeperiod=30)

df['MA'] = talib.MA(close, timeperiod=30, matype=0)

df['SAR'] = talib.SAR(high, low, acceleration=0, maximum=0)

df['WMA'] = talib.WMA(close, timeperiod=30)


# In[4]:


# Drop any rows with empty features
df.dropna(inplace = True)
df.isnull().sum()
print(df.shape)


# In[5]:


# Make a copy of the last row (today) so we can predict tomorrows price
# Drop the columns we won't be using
tomorrow = df.copy()
drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
tomorrow = tomorrow.drop(drop_cols, 1)
tomorrow.tail(1)


# In[6]:


#Shift Close back 1 so we can predict next days price
df['Close'] = df['Close'].shift(-1)


# In[7]:


# Drop first 58 columns (Performs better for SVR)
df = df.iloc[58:] # Because of moving averages and MACD line
df = df[:-1]      # Because of shifting close price - This removes last column this is why its not todays prediction 
#  Moreover, after shifting Close price column, last row price is equal to 0 which is not true.

# COnvert index into number range
df.index = range(len(df))


# In[8]:


# Set training/test data - and visualize them
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go

test_size  = 0.25

test_split_idx  = int(df.shape[0] * (1-test_size))

train_df  = df.loc[:test_split_idx].copy()
test_df   = df.loc[test_split_idx+1:].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df.index, y=train_df.Close, name='Training'))
fig.add_trace(go.Scatter(x=test_df.index, y=test_df.Close, name='Test'))
fig.show()


# In[9]:


# Check test/train data correctly split
print(train_df.head(1))
print(train_df.tail(1))

print(test_df.head(1))
print(test_df.tail(1))


# In[10]:


#Normalize the data
from sklearn.preprocessing import MinMaxScaler

def normalize_sets(scaler, train_df, test_df, features):
    for feature in features:
        train_df[feature] = scaler.fit_transform(train_df[feature].values.reshape(-1,1))
        test_df[feature] = scaler.fit_transform(test_df[feature].values.reshape(-1,1))
    
# Drop
drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
train_df = train_df.drop(drop_cols, 1)
test_df  = test_df.drop(drop_cols, 1)

# Normalize
scaler = MinMaxScaler()
normalize_sets(scaler, train_df, test_df, features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM', 
                                                              'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                                                              , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA' ]) 

# Split x y

label_column = 'Close'
normalize_sets(scaler, train_df, test_df, features=[label_column]) 


y_train = train_df[label_column].copy()
X_train = train_df.drop([label_column], 1)
y_test  = test_df[label_column].copy()
X_test  = test_df.drop([label_column], 1)


# In[11]:


# View normalized data
X_train.tail(1)


# In[12]:


X_test.tail(1)


# In[13]:


from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

#best_params = {'C': 1.5, 'epsilon': 0.1, 'gamma': 1e-07, 'kernel': 'linear'}


model = LGBMRegressor()
model.fit(X_train, y_train)


# In[14]:


y_pred = model.predict(X_test)
print(f'y_true = {np.array(y_test)[:5]}')
print(f'y_pred = {y_pred[:5]}')


# In[15]:


y_true_unnorm = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
y_pred_unnorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

print(f'y_true = {y_true_unnorm[:5]}')
print(f'y_pred = {y_pred_unnorm[:5]}')

print(f'MSE on normalized values = {mean_squared_error(y_test, y_pred)}')
print(f'MSE on un-normalized values = {mean_squared_error(y_true_unnorm, y_pred_unnorm)}')


# In[16]:


print(f'Most recent predictions = {y_pred_unnorm[-5:]}')
print('vs')
print(f'Most recent true = {np.array(y_true_unnorm)[-5:]}')


# In[17]:


# Visualize Predictions
predicted_prices = df.loc[test_split_idx+1:].copy()
predicted_prices['Close'] = y_pred_unnorm

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df.Close, name='Ground truth'))
fig.add_trace(go.Scatter(x=predicted_prices.index, y=predicted_prices.Close, marker_color='Orange', name='Prediction'))
fig.show()


# In[18]:


# Evaluation Metrics (MSE, MAE, RMSE, r2)
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print('Normalized Results')
print(f'MSE = {mean_squared_error(y_test, y_pred)}')

rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE = {rmse}')


mae = mean_absolute_error(y_test, y_pred)
print(f'MAE = {mae}')


r2 = r2_score(y_test, y_pred)
print(f'R2 = {r2}')


# In[19]:


print('Non-Normalized Results')
print(f'MSE = {mean_squared_error(y_true_unnorm, y_pred_unnorm)}')

rmse = sqrt(mean_squared_error(y_true_unnorm, y_pred_unnorm))
print(f'RMSE = {rmse}')


mae = mean_absolute_error(y_true_unnorm, y_pred_unnorm)
print(f'MAE = {mae}')


r2 = r2_score(y_true_unnorm, y_pred_unnorm)
print(f'R2 = {r2}')


# In[20]:


# Closer look at the predictions 
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

predicted_prices = df.loc[test_split_idx+1:].copy()
predicted_prices['Close'] = y_pred_unnorm


fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=df.index, y=df.Close,
                         name='Truth',
                         marker_color='LightSkyBlue'), row=1, col=1)


fig.add_trace(go.Scatter(x=test_df.index, 
                         y=y_true_unnorm, 
                         name='Test',
                         marker_color='LightPink'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.index,
                         y=predicted_prices.Close,
                         name='Prediction',
                         marker_color='MediumPurple'), row=1, col=1)


fig.add_trace(go.Scatter(x=predicted_prices.index,
                         y=y_test,
                         name='Truth',
                         marker_color='LightPink',
                         showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.index,
                         y=y_pred,
                         name='Prediction',
                         marker_color='MediumPurple',
                         showlegend=False), row=2, col=1)

fig.update_layout(
    title={
        'text': "Closer look at Predictions",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.show()


# In[21]:


# Predicting Tomorrows Price
print(tomorrow[-1:])
tomorrow.dropna(inplace = True)
tomorrow.index = range(len(tomorrow))
print(tomorrow)


# In[22]:


def normalize_new(scaler2, tomorrow, features):
    for feature in features:
        tomorrow[feature] = scaler2.fit_transform(tomorrow[feature].values.reshape(-1,1))


# Normalize
scaler2 = MinMaxScaler()
normalize_new(scaler2, tomorrow, features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM', 
                                                              'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                                                              , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA' ]) 

# Split x y

label_column = 'Close'
normalize_new(scaler2, tomorrow, features=[label_column]) 

predict = tomorrow.drop([label_column], 1)
predict = predict[-1:]
print(predict)


# In[23]:


#Data used to predict tomorrow
pred_tomorrow = predict[-1:]
pred_tomorrow.tail()


# In[24]:


#Call the model to predict tomorrows price
prediction = model.predict(pred_tomorrow)


# In[25]:


# Show tomorrows predicted price
prediction = scaler2.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()

print(f'Tomorrows Predicted Price:  ${prediction[-1]}')


# In[26]:


#Show Todays Predicted vs True price
start = pd.to_datetime("today").date()
today_price = web.DataReader(company,"yahoo",start,end)['Close']

print(f'Todays Predicted Price:')
print(y_pred_unnorm[-1:])

print(f'Todays Current Price:')
print(today_price)

# try to get just the number 


# In[ ]:




