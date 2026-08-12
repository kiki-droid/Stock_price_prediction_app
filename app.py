# ============================================================
# IMPORT PACKAGES
# ============================================================

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly

import streamlit as st
import yfinance as yf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras import backend as K

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from streamlit_option_menu import option_menu
from ta.trend import MACD


# ============================================================
# STREAMLIT PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Stock Market Analysis and Prediction",
    layout="wide"
)


# Hide Streamlit menu, footer and header
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)

px.defaults.width = 1000


# ============================================================
# SIDEBAR - STOCK SELECTION
# ============================================================

with st.sidebar:

    stock = option_menu(
        menu_title="Stock market analysis and prediction",
        options=["GOOGL", "AAPL", "AMZN", "MSFT", "TSLA"],
        icons=["google", "apple", "check-square", "microsoft", "robot"],
        menu_icon="bar-chart-fill",
        default_index=0,
    )


# ============================================================
# STOCK TITLE
# ============================================================

if stock == "GOOGL":
    st.subheader("GOOGLE Stocks")

if stock == "AAPL":
    st.subheader("APPLE Stocks")

if stock == "AMZN":
    st.subheader("AMAZON Stocks")

if stock == "MSFT":
    st.subheader("MICROSOFT Stocks")

if stock == "TSLA":
    st.subheader("TESLA Stocks")


# ============================================================
# DOWNLOAD STOCK DATA
# ============================================================

df = yf.download(
    tickers=stock,
    period="8y",
    interval="1d",
    auto_adjust=False
)

# Keep only the most recent 4 years of data
four_years_ago = pd.Timestamp(date.today() - timedelta(days=4 * 365))

df = df[df.index >= four_years_ago]

# ============================================================
# FIX YFINANCE MULTIINDEX
# ============================================================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)


# Make sure OHLCV columns are 1-dimensional
for col in ["Open", "High", "Low", "Close", "Volume"]:

    if isinstance(df[col], pd.DataFrame):
        df[col] = df[col].iloc[:, 0]


# Check whether data was downloaded
if df.empty:

    st.error(f"No data found for {stock}")

    st.stop()


# ============================================================
# DATE VARIABLES
# ============================================================

end = date.today()

start = date.today() - timedelta(days=30)

new = date.today() + timedelta(days=1)

last = date.today() + timedelta(days=30)


# ============================================================
# STOCK INFORMATION
# ============================================================

st.subheader("Stock information")

d = str(
    date(
        date.today().year - 4,
        date.today().month,
        date.today().day
    )
)

st.write("Stock Data till today from", d)

st.write(df)


# ============================================================
# CREATE PLOTLY FIGURES
# ============================================================

fig = go.Figure()

fig1 = go.Figure()

fig2 = go.Figure()


# ============================================================
# MOVING AVERAGES
# ============================================================

df["MA50"] = df["Close"].ewm(
    span=50,
    adjust=False
).mean()

df["MA100"] = df["Close"].ewm(
    span=100,
    adjust=False
).mean()

df["MA200"] = df["Close"].ewm(
    span=200,
    adjust=False
).mean()


# ============================================================
# MACD
# ============================================================

macd = MACD(
    close=df["Close"],
    window_slow=26,
    window_fast=12,
    window_sign=9
)


# ============================================================
# STOCK PRICE / CANDLESTICK CHART
# ============================================================

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Market Data"
    )
)


# MA50

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA50"],
        opacity=0.7,
        line=dict(
            color="blue",
            width=2
        ),
        name="MA50"
    )
)


# MA100

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA100"],
        opacity=0.7,
        line=dict(
            color="orange",
            width=2
        ),
        name="MA100"
    )
)


# MA200

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA200"],
        opacity=0.7,
        line=dict(
            color="yellow",
            width=2
        ),
        name="MA200"
    )
)


# ============================================================
# PRICE CHART LAYOUT
# ============================================================

fig.update_layout(
    title=str(stock) + " Live Share Price:",
    yaxis_title="Stock Price (USD per Share)"
)


fig.update_xaxes(
    rangeslider_visible=True,

    rangeselector=dict(
        buttons=list([

            dict(
                count=15,
                label="15d",
                step="day",
                stepmode="backward"
            ),

            dict(
                count=30,
                label="30d",
                step="day",
                stepmode="backward"
            ),

            dict(
                count=1,
                label="HTD",
                step="month",
                stepmode="todate"
            ),

            dict(
                count=3,
                label="3mo",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=6,
                label="6mo",
                step="month",
                stepmode="backward"
            ),

            dict(step="all")

        ])
    )
)


# ============================================================
# VOLUME CHART
# ============================================================

colors = np.where(
    df["Close"].values >= df["Open"].values,
    "green",
    "red"
)


fig1.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color=colors,
        name="Volume"
    )
)


# ============================================================
# MACD HISTOGRAM
# ============================================================

fig2.add_trace(
    go.Bar(
        x=df.index,
        y=macd.macd_diff(),
        marker_color="orange",
        name="MACD Histogram"
    )
)


# MACD line

fig2.add_trace(
    go.Scatter(
        x=df.index,
        y=macd.macd(),
        line=dict(
            color="black",
            width=2
        ),
        name="MACD line (EMA-12/FAST - EMA-26/SLOW)"
    )
)


# Signal line

fig2.add_trace(
    go.Scatter(
        x=df.index,
        y=macd.macd_signal(),
        line=dict(
            color="blue",
            width=1
        ),
        name="Signal line (EMA-9)"
    )
)


# ============================================================
# VOLUME / MACD LABELS
# ============================================================

fig1.update_yaxes(
    title_text="Volume"
)

fig2.update_yaxes(
    title_text="MACD",
    showgrid=False
)


fig1.update_layout(
    title="Volume"
)

fig2.update_layout(
    title="MACD(12,26,9)"
)


# ============================================================
# VOLUME X-AXIS
# ============================================================

fig1.update_xaxes(
    rangeslider_visible=True,

    rangeselector=dict(
        buttons=list([

            dict(
                count=15,
                label="15d",
                step="day",
                stepmode="backward"
            ),

            dict(
                count=30,
                label="30d",
                step="day",
                stepmode="backward"
            ),

            dict(
                count=1,
                label="HTD",
                step="month",
                stepmode="todate"
            ),

            dict(
                count=3,
                label="3mo",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=6,
                label="6mo",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=12,
                label="12mo",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=24,
                label="24mo",
                step="month",
                stepmode="backward"
            )

        ])
    )
)


# ============================================================
# MACD X-AXIS
# ============================================================

fig2.update_xaxes(
    rangeslider_visible=True,

    rangeselector=dict(
        buttons=list([

            dict(
                count=15,
                label="15d",
                step="day",
                stepmode="backward"
            ),

            dict(
                count=30,
                label="30d",
                step="day",
                stepmode="backward"
            ),

            dict(
                count=1,
                label="HTD",
                step="month",
                stepmode="todate"
            ),

            dict(
                count=3,
                label="3mo",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=6,
                label="6mo",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=12,
                label="12mo",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=24,
                label="24mo",
                step="month",
                stepmode="backward"
            )

        ])
    )
)


# ============================================================
# GRAPHICAL ANALYSIS
# ============================================================

st.subheader("Graphical Analysis")


tab1, tab2, tab3 = st.tabs(
    [
        "Stock Prices",
        "Volume",
        "Moving average convergence/divergence (MACD)"
    ]
)


with tab1:

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with tab2:

    st.plotly_chart(
        fig1,
        use_container_width=True
    )


with tab3:

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# LSTM PREDICTION SECTION
# ============================================================

st.subheader(
    "Stock average price prediction using LSTM for next 30 days from today"
)


st.caption(
    "The LSTM used here is a stacked univariate LSTM.\n"
    "The data from 4 years till today is split into train and test "
    "with 80:20 ratio where, the most previous dates are used for "
    "training and the nearer dates are used for testing and predicting.\n"
    "The average price of last 100 days is used as a window period for "
    "training, testing and predicting the next day's average price. "
    "Root mean squared error (RMSE) is the square root of the mean "
    "of the square of all of the error."
)


# ============================================================
# RESET INDEX
# ============================================================

df = df.reset_index()


# ============================================================
# TRAINING DATES
# ============================================================

train_dates = pd.to_datetime(
    df["Date"]
).dt.date


# ============================================================
# CREATE AVERAGE PRICE DATA
# ============================================================

df1 = df[
    [
        "Close",
        "Open",
        "High",
        "Low"
    ]
].copy()


df2 = df1.mean(axis=1)


# ============================================================
# SCALE DATA
# ============================================================

df3 = np.reshape(
    df2.values,
    (len(df2), 1)
)


scaler = MinMaxScaler(
    (0, 1)
)


df4 = scaler.fit_transform(df3)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_d = int(
    len(df4) * 0.8
)

test_d = len(df4) - train_d


train_d, test_d = (
    df4[0:train_d, :],
    df4[train_d:len(df4), :]
)


# ============================================================
# CREATE DATASET
# ============================================================

def new_dataset(dataset, step_size):

    data_X = []

    data_Y = []

    for i in range(
        len(dataset) - step_size - 1
    ):

        a = dataset[
            i:(i + step_size),
            0
        ]

        data_X.append(a)

        data_Y.append(
            dataset[
                i + step_size,
                0
            ]
        )

    return (
        np.array(data_X),
        np.array(data_Y)
    )


# ============================================================
# WINDOW SIZE
# ============================================================

step_size = 100


X_train, Y_train = new_dataset(
    train_d,
    step_size
)


X_test, Y_test = new_dataset(
    test_d,
    step_size
)


# ============================================================
# RESHAPE FOR LSTM
# ============================================================

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)


X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)


# ============================================================
# LSTM MODEL
# ============================================================

K.clear_session()

model = Sequential()

model.add(
    Input(shape=(step_size, 1))
)

model.add(
    LSTM(
        units=50,
        return_sequences=True
    )
)

model.add(
    LSTM(
        units=50
    )
)

model.add(
    Dense(1)
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    loss="mean_squared_error",
    optimizer="adam"
)


# ============================================================
# TRAIN MODEL
# ============================================================

history = model.fit(
    X_train,
    Y_train,
    epochs=10,
    batch_size=32
)


# ============================================================
# PREDICTIONS
# ============================================================

trainPredict = model.predict(
    X_train
)


testPredict = model.predict(
    X_test
)


# ============================================================
# RMSE
# ============================================================

trainScore = math.sqrt(
    mean_squared_error(
        Y_train,
        trainPredict
    )
)


st.write(
    "LSTM Train RMSE: %.2f"
    % trainScore
)


testScore = math.sqrt(
    mean_squared_error(
        Y_test,
        testPredict
    )
)


st.write(
    "LSTM Test RMSE: %.2f"
    % testScore
)


# ============================================================
# INVERSE TRANSFORM
# ============================================================

trainPredictY = scaler.inverse_transform(
    trainPredict
)


trainY = scaler.inverse_transform(
    [Y_train]
)


testPredictY = scaler.inverse_transform(
    testPredict
)


testY = scaler.inverse_transform(
    [Y_test]
)


# ============================================================
# FUTURE 30-DAY PREDICTION
# ============================================================

n_future = 30

y_future = []


x_pred = X_test[-1:, :]


y_pred = testPredict[-1]


for i in range(n_future):

    x_pred = np.append(
        x_pred[:, 1:],
        y_pred.reshape(1, 1),
        axis=1
    )

    y_pred = model.predict(
        x_pred
    )

    y_future.append(
        y_pred.flatten()[0]
    )


# ============================================================
# INVERSE SCALE FUTURE PREDICTIONS
# ============================================================

y_future = np.array(
    y_future
).reshape(-1, 1)


y_future_S = scaler.inverse_transform(
    y_future
)


# ============================================================
# FUTURE PREDICTION DATAFRAME
# ============================================================

dff = pd.DataFrame(
    columns=[
        "Date",
        "Average price predicted for the next 30 days with 100 days window period"
    ]
)


dff["Date"] = pd.date_range(
    date.today() + pd.Timedelta(days=1),
    periods=n_future
)


dff[
    "Average price predicted for the next 30 days with 100 days window period"
] = y_future_S.flatten()


st.write(dff)


# ============================================================
# ACTUAL AVERAGE PRICE
# ============================================================

data = df1.copy()


data["Average price (actual)"] = data.mean(
    numeric_only=True,
    axis=1
)


ext_col = df["Date"]


data.insert(
    0,
    "Date",
    ext_col
)


data.drop(
    [
        "Close",
        "Open",
        "High",
        "Low"
    ],
    axis=1,
    inplace=True
)


# ============================================================
# LAST 30 DAYS
# ============================================================

mask = (
    train_dates >
    (
        date.today() -
        timedelta(days=30)
    )
)


data1 = data.loc[mask].copy()


# ============================================================
# COMBINE ACTUAL + PREDICTED DATA
# ============================================================

frames = [
    data1,
    dff
]


result = pd.concat(
    frames,
    ignore_index=True
)


# ============================================================
# PREDICTION CHART
# ============================================================

figx = px.scatter(
    result,
    x="Date",
    y=[
        "Average price (actual)",
        "Average price predicted for the next 30 days with 100 days window period"
    ]
)


figx.update_yaxes(
    title_text="Price (in $)"
)


figx.update_layout(
    width=1000
)


# ============================================================
# DISPLAY PREDICTION CHART
# ============================================================

st.plotly_chart(
    figx,
    width=1000
)
