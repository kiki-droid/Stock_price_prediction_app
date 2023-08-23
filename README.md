# Stock price prediction app with lstm

* In this project, machine learning model LSTM or Long Short Term Memory is used for making predictions.Long short-term memory (LSTM) belongs to the complex areas of Deep Learning and is studied to give best predictions in time series models in machine learning.

* Average price of a stock for the next whole day is calculated using averages prices from previous dates fed in the model. Here, average price for the day is calculated by taking into consideration **OHLC i.e., Open, High, Low and Close** prices for that day. 

* The LSTM used here is a stacked univariate LSTM.

* The data is splited into train and test with 80:20 ratio where, the most previous dates are used for training and the nearer dates are used for testing and predicting. The average price of last 50 and 100 days is used as a window period for training, testing and predicting the next day's average price. 

* Root mean squared error (RMSE) is the square root of the mean of the square of all of the error. Based on a rule of thumb, it can be said that RMSE values between 0.2 and 0.5 shows that the model can relatively predict the data accurately. RMSE of training and testing data both is calculted to see model's quality.

* The data from 8 years before the date of running the application is analysed and presented. For extraction of data, yfinance library of python is used which can be installed using-
**pip install yfinance** 

* Prices for various cryptocurrencies in addition to company stocks can be analysed in the .ipynb notebook provided you know the ticker symbol.

* For technical Analysis, the focus has been on the most prominent indicators that can be efficiently operationalized and are intuitive in interpretation, including: Moving average convergence & divergence, Daily Closing Volume and Exponential Moving averages over 50, 100 and 200 days.

* All Moving averages calculated here are exponential and the formula used is- 
**EMA = (K x (C - P)) + P**
Where:
C = Current Price
P = Previous periods EMA (A SMA is used for the first periods calculations)
K = Exponential smoothing constant

*  **Moving average convergence & divergence** :- It is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.
* MACD Formula:- **MACD=12-Period EMA − 26-Period EMA**
* The signal line is a 9-Period EMA of the MACD line.
*  If the MACD crosses above its signal line following a brief correction within a longer-term uptrend, it qualifies as **bullish** confirmation.
*  If the MACD crosses below its signal line following a brief move higher within a longer-term downtrend, traders would consider that a **bearish** confirmation.
*  MACD is best used with daily periods, where the traditional settings of 26/12/9 days is the norm.

Deployed app for analysing 5 major companies using streamlit-
link - https://stocks-price-prediction.streamlit.app/
