""" function to visualize results from prediction"""
import copy
import pandas as pd
from math import isnan
import plotly.graph_objects as go

def visualize_predict(suffix = "", code = "", tick = ""):
    """
    function to visualize the results from a prediction
    Input : 
    - suffix : string to identify the csv file to load
    """
    # load data
    df = pd.read_csv("df_eval_"+suffix+code+".csv")
    # to normalize prices and balance
    idx_init = df['balance'].notna().idxmax()
    init_Close = df['Close'].values[idx_init]
    init_balance = next(x for x in df['balance'].values if not isnan(x))
    # infor for sell
    dfsell = copy.deepcopy(df)
    dfsell = dfsell[dfsell["sell"] > 0]
    xsell = dfsell["Date"]
    ysell = dfsell["Close"]
    sizesell = dfsell["volume_order"].values*20
    # info for buy
    dfbuy = copy.deepcopy(df)
    dfbuy = dfbuy[dfbuy["buy"] > 0]
    xbuy = dfbuy["Date"]
    ybuy = dfbuy["Close"]
    sizebuy = dfbuy["volume_order"].values*20
    # Create figure
    fig = go.Figure(
        data=[go.Candlestick(x=df['Date'],
                    open=(df['Open']/init_Close-1)*100,
                    high=(df['High']/init_Close-1)*100,
                    low=(df['Low']/init_Close-1)*100,
                    close=(df['Close']/init_Close-1)*100,
                    name=tick),
                go.Scatter(x=xsell,
                    y=(ysell/init_Close-1)*100,
                    mode='markers',
                    name="buy",
                    marker=dict(
                        color="red",
                        size=sizesell,
                        opacity = 0.5,
                        sizemin=0)),
                go.Scatter(x=xbuy,
                    y=(ybuy/init_Close-1)*100,
                    mode='markers',
                    name="sell",
                    marker=dict(
                        color="green",
                        size=sizebuy,
                        opacity = 0.5,
                        sizemin=0)),
                go.Scatter(x=df['Date'],
                    y=(df['balance']/init_balance-1)*100,
                    mode='lines',
                    name="account balance")
            ])
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(title=suffix+" "+code,
                    xaxis_title="Date",
                    yaxis_title="Relative Scale %")
    fig.show()

#visualize_predict(suffix = "train", code = "", tick = "AAPL")
#visualize_predict(suffix = "valid", code = "", tick = "AAPL")   




