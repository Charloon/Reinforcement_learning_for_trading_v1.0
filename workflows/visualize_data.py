""" script to visualize training data"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# load dataframe
df = pd.read_csv("df_original.csv")

# to normalize prices
init_Close = df['Close'].values[0]

# df for tops and bottom
df_tops = df[df["tops_bottoms_20MA"]==1]
df_bottoms = df[df["tops_bottoms_20MA"]==-1]
df_LL = df[df["LL"]==1]
df_HH = df[df["HH"]==1]
df_LH = df[df["LH"]==1]
df_HL = df[df["HL"]==1]
# Create figure 1
fig = make_subplots(rows=2, cols=1)
fig = go.Figure(
    data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']),
            go.Scatter(x=df_tops['Date'],
                y=df_tops['High'],
                mode='markers',
                marker=dict(
                    color="red",
                    size=10,
                    opacity = 0.5,
                    #sizemode='area',
                    #sizeref=2.*max(dfi['count'])/(40.**2),
                    sizemin=0)),
            go.Scatter(x=df_bottoms['Date'],
                y=df_bottoms['Low'],
                mode='markers',
                marker=dict(
                    color="green",
                    size=10,
                    opacity = 0.5,
                    #sizemode='area',
                    #sizeref=2.*max(dfi['count'])/(40.**2),
                    sizemin=0)),
            go.Scatter(x=df_HH['Date'],
                y=df_HH['High'],
                text=["HH"]*df_HH.shape[0],
                mode="text",
                textposition="top center"),
            go.Scatter(x=df_LH['Date'],
                y=df_LH['High'],
                text=["LH"]*df_HH.shape[0],
                mode="text",
                textposition="top center"),
            go.Scatter(x=df_LL['Date'],
                y=df_LL['Low'],
                text=["LL"]*df_HH.shape[0],
                mode="text",
                textposition="bottom center"),
            go.Scatter(x=df_HL['Date'],
                y=df_HL['Low'],
                text=["HL"]*df_HL.shape[0],
                mode="text",
                textposition="bottom center"),
            go.Scatter(x=df['Date'],
                y=df['beta'],
                mode='lines',
                xaxis="x",
                yaxis="y2"),
          ], 
    layout = go.Layout(
            xaxis=dict(
                domain=[0, 1]
            ),
            yaxis2=dict(
                domain=[0, 0.45]
            ),
            yaxis=dict(
                domain=[0.55, 1]
            )))
fig.update_xaxes(rangeslider_visible=False)
fig.show()

# Create figure2
# df for tops and bottom
df_tops = df[df["higherTF_tops_bottoms_20MA"]==1]
df_bottoms = df[df["higherTF_tops_bottoms_20MA"]==-1]
fig = make_subplots(rows=2, cols=1)
fig = go.Figure(
    data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']),
            go.Scatter(x=df_tops['Date'],
                y=df_tops['High'],
                mode='markers',
                marker=dict(
                    color="red",
                    size=10,
                    opacity = 0.5,
                    #sizemode='area',
                    #sizeref=2.*max(dfi['count'])/(40.**2),
                    sizemin=0)),
            go.Scatter(x=df_bottoms['Date'],
                y=df_bottoms['Low'],
                mode='markers',
                marker=dict(
                    color="green",
                    size=10,
                    opacity = 0.5,
                    #sizemode='area',
                    #sizeref=2.*max(dfi['count'])/(40.**2),
                    sizemin=0)),
            go.Scatter(x=df['Date'],
                y=df['higherTF_beta'],
                mode='lines',
                xaxis="x",
                yaxis="y2"),
          ], 
    layout = go.Layout(
            xaxis=dict(
                domain=[0, 1]
            ),
            yaxis2=dict(
                domain=[0, 0.45]
            ),
            yaxis=dict(
                domain=[0.55, 1]
            )))
fig.update_xaxes(rangeslider_visible=False)
fig.show()

