""" Utilities for the environment """
import copy
import os

def TimeStampDiff(n_candle, TimeFrame):
    """ dictionnary to connect strings and timestamps for timeframes """
    TimestampMin = {"1m": n_candle*60*1000, "3m": n_candle*3*60*1000,
                 "5m": n_candle*5*60*1000, "15m": n_candle*15*60*1000,
                 "30m": n_candle*30*60*1000, "1h": n_candle*3600*1000,
                 "2h": n_candle*2*3600*1000, "4h": n_candle*4*3600*1000,
                 "6h": n_candle*3*3600*1000, "8h": n_candle*8*3600*1000,
                 "12h": n_candle*12*3600*1000, "1d": n_candle*24*3600*1000,
                 "3d": n_candle*3*24*3600*1000, "1w": n_candle*7*24*3600*1000,
                 "1M":n_candle*30*24*3600*1000}
    return TimestampMin[TimeFrame]

def cross_detection_oscillator(df, rsi):
    """ function to detect key crossing for oscillator between 0 and 100 """
    oversold = 70
    overbought = 30
    list_name = []

    #### detect into oversold ####
    name = rsi+"intoOverSold"
    df[name] = 0
    for i in range(1,df.shape[0]):
        if df[rsi+""].values[i] - df[rsi+""].values[i-1] > 0:
            if df[rsi+""].values[i-1] <= oversold and df[rsi+""].values[i] >  oversold:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect out of oversold ####
    name = rsi+"outOfOverSold"
    df[name] = 0
    for i in range(1,df.shape[0]):
        if df[rsi].values[i] - df[rsi].values[i-1] < 0:
            if df[rsi].values[i-1] > oversold and df[rsi].values[i] <=  oversold:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect in oversold ####
    name = rsi+"inOverSold"
    df[name] = 0
    for i in range(0,df.shape[0]):
        if df[rsi].values[i] >= oversold:
            df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect into overbought ####
    name = rsi+"intoOverBought"
    df[name] = 0
    for i in range(1,df.shape[0]):
        if df[rsi].values[i] - df[rsi].values[i-1] < 0:
            if df[rsi].values[i-1] > overbought and df[rsi].values[i] <=  overbought:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect out of overbought ####
    name = rsi+"outOfOverBought"
    df[name] = 0
    for i in range(1,df.shape[0]):
        if df[rsi].values[i] - df[rsi].values[i-1] > 0:
            if df[rsi].values[i-1] < overbought and df[rsi].values[i] >=  overbought:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect in overbought ####
    name = rsi+"inOverBought"
    df[name] = 0
    for i in range(0,df.shape[0]):
        if df[rsi].values[i] <= oversold:
            df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect cross below 50 ####
    name = rsi+"above50"
    df[name] = 0
    for i in range(1,df.shape[0]):
        if df[rsi].values[i] - df[rsi].values[i-1] > 0:
            if df[rsi].values[i-1] < 50 and df[rsi].values[i] >=  50:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect cross above 50 ####
    name = rsi+"below50"
    df[name] = 0
    for i in range(1,df.shape[0]):
        if df[rsi].values[i] - df[rsi].values[i-1] < 0:
            if df[rsi].values[i-1] > 50 and df[rsi].values[i] <=  50:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    return df, list_name

def clean_results_folder():
    """ function to empy the folder to store results for each episode"""
    for i in range(1000):
        myfile = "./tmp/result_"+str(i)+".csv"
        try:
            os.remove(myfile)
        except:
            pass
