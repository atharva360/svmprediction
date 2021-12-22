import matplotlib.pyplot as plt
import streamlit as st
#Import data
import pandas_datareader as data
# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
  
# For data manipulation
import pandas as pd
import numpy as np
  
# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
  
# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


st.title('STOCK MARKET PREDICTION USING SUPPORT VECTOR MACHINE')
user_input = st.text_input('Enter The Stock Ticker', 'AAPL')
start = '2010-01-01'
end = '2021-12-15'
df = data.DataReader(user_input,'yahoo',start,end)
#set the date as the index
# df = df.set_index(data.DatetimeIndex(df['Date'].values))

#describing data
st.subheader('Data From 2010-2021')
st.write(df.describe())


# Create predictor variables
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low
  
# Store all predictor variables in a variable X
X = df[['Open-Close', 'High-Low']]

# Target variables
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# We will split data into training and test data sets. 
# This is done so that we can evaluate the effectiveness of the model in the test dataset

split_percentage = 0.8
split = int(split_percentage*len(df))
  
# Train data set
X_train = X[:split]
y_train = y[:split]
  
# Test data set
X_test = X[split:]
y_test = y[split:]

# Support vector classifier
cls = SVC().fit(X_train, y_train)

df['Predicted_Signal'] = cls.predict(X)

# Calculate daily returns
df['Return'] = df.Close.pct_change()

# Calculate strategy returns
df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)

# Calculate Cumulutive returns
df['Cum_Ret'] = df['Return'].cumsum()

# Plot Strategy Cumulative returns 
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
st.subheader('Data with Cumulative returns and strategy returns') 
st.write(df)

st.subheader('Support Vector Machine Prediction') 
figsvm = plt.figure(figsize = (12,6))
plt.plot(df['Cum_Ret'],color='red')
plt.plot(df['Cum_Strategy'],color='blue')
st.pyplot(figsvm)