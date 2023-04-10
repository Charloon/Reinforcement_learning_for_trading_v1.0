""" All functions used in preprocessing the data """
import copy
from datetime import datetime
import numpy as np
import pandas as pd
from env.RSI import add_RSI_feat
from env.moving_average import moving_average
from env.tops_and_bottoms import find_tops_and_bottom

def find_time_index(df, current_time, time_feat):
    """ find the index of the current time in the dataframe"""
    # change dates to datetime
    dt = np.array([(x - pd.to_datetime(current_time)).total_seconds() \
                    for x in pd.to_datetime(df[time_feat])])
    return int(np.argmin(np.abs(np.where(dt > 1e-6, 1e20, dt))))

def add_day_month_weekday(df_temp, absolute_time):
    """ add columns to the dataframe for day, month and weekday"""
    time = [datetime.strptime(x, '%Y-%m-%d') for x in df_temp[absolute_time].values]
    df_temp["day"] = [x.day for x in time]
    df_temp["month"] = [x.month for x in time]
    df_temp["weekday"] = [x.weekday() for x in time]
    list_feature = ["day", "month", "weekday"]
    return df_temp, list_feature

def describe_candle(df, open = "Open", close = "Close", high = "High", low = "Low"):
    """ add columns to the dataframe to describe candles"""
    # initialize
    list_feat = []
    list_feat_diffPrice = []
    # body size
    df["body_size"] = df[close] - df[open]
    list_feat.append("body_size")
    list_feat_diffPrice.append("body_size")

    # top mesh
    df["top_mesh"] = df[high] - np.maximum(df[open], df[close])
    list_feat.append("top_mesh")
    list_feat_diffPrice.append("top_mesh")

    # bottom mesh
    df["bot_mesh"] = np.minimum(df[open], df[close]) - df[low]
    list_feat.append("bot_mesh")
    list_feat_diffPrice.append("bot_mesh")

    # total mesh size
    df["mesh_size"] = df[high] - df[low]
    list_feat.append("mesh_size")
    list_feat_diffPrice.append("mesh_size")

    # body/mesh ratio
    df["body_mesh_ratio"] = np.divide(df["body_size"],df["mesh_size"])
    list_feat.append("body_mesh_ratio")

    return df, list_feat, list_feat_diffPrice 


class Generate_df():
    """Class to preprocess the dataframe, and do feature engineering"""
    # to initilaize
    def __init__(self, df, TF, time_feat):
        self.price_feat = ['Open', 'High', 'Low', 'Close', 'Volume']
        list_target = []
        # save dataframe in a dict
        dt = 0
        # define normalization
        self.normalize = dict(list_feat_absPrice = ["Open", "Close", "High", "Low"],
                              list_feat_diffPrice = [],
                              list_feat_absVolume = ["Volume"],
                              list_feat_diffVolume = [], 
                              list_feat_percent = [],
                              list_feat_day = [],
                              list_feat_week = [],
                              list_feat_month = [],
                              list_feat_year = [],
                              list_feat_weekday = []
                              )
        ## preprocess ## 
        # get date info
        df, list_feat = add_day_month_weekday(df, time_feat)
        self.price_feat.extend(list_feat)
        self.normalize["list_feat_day"].append("day")
        self.normalize["list_feat_week"].append("week")
        self.normalize["list_feat_month"].append("month")
        self.normalize["list_feat_year"].append("year")
        self.normalize["list_feat_weekday"].append("weekday")
        # get candle description
        df, list_feat, list_feat_diffPrice = describe_candle(df, open = "Open",
                                             close = "Close", high = "High", low = "Low")
        self.normalize["list_feat_diffPrice"].extend(list_feat_diffPrice)
        self.price_feat.extend(list_feat)
        # RSI
        df, list_feat_RSI, list_feat_percent = add_RSI_feat(df)
        self.normalize["list_feat_percent"].extend(list_feat_percent)
        self.price_feat.extend(list_feat_RSI)
        # moving average
        df, list_feat, list_feat_absPrice, list_feat_diffPrice = moving_average(df, "Close",
                                                                [20, 50, 100, 200])
        self.price_feat.extend(list_feat)
        self.normalize["list_feat_absPrice"].extend(list_feat_absPrice)
        self.normalize["list_feat_diffPrice"].extend(list_feat_diffPrice)
        # add tops and bottom
        df, list_feat = find_tops_and_bottom(df, period=20)
        self.price_feat.extend(list_feat)
        # remove rows with nans
        df = df.dropna(axis=0)
        #df.to_csv("df.csv")
        # save dataframe and dataframe information for training and prediction
        self.dict_df = {TF: {"df": df, "list_feat": self.price_feat, "dt": dt}}
        # save original dataframe
        self.df_init = df
        # reset index
        df = df.reset_index(drop=True)           
        # save dataframe
        self.dict_df[TF]["df"] = df 
        df.to_csv("df_original.csv")

def add_reward_limits(init_cash, df, idx):
    """ to add some features to the dataframe:
        - balance: track the balance of the account
        - max_return: track the maximum return with optimal trading
        - min_return: track the minimum return with worst trading
        - all_asset: track the return if the account all in asset all the time
    """
    # add initialize
    df["balance"] = init_cash
    df["max_return"] = init_cash
    df["min_return"] = init_cash
    df["all_asset"] = init_cash
    # update for every time step
    for i in range(idx+1,df.shape[0]):
        penalty = 0.0
        coef_max = max(1.0 , (df["Close"].values[i] - penalty)/df["Close"].values[i-1])
        df["max_return"].values[i] = df["max_return"].values[i-1] * coef_max
        coef_min = min(1.0 , (df["Close"].values[i] - penalty)/df["Close"].values[i-1])
        df["min_return"].values[i] = df["min_return"].values[i-1] * coef_min
        coef = (df["Close"].values[i] - penalty)/df["Close"].values[i-1]
        df["all_asset"].values[i] = df["all_asset"].values[i-1] * coef
    df["balance"] = init_cash
    return df

def split_train_val(data, metadata, perct_train = 0.7, embargo = 0):
    """ Split the dataset into a training and a validation dataset"""
    # duplicate the data
    data_train =  copy.deepcopy(data)
    data_val = copy.deepcopy(data)
    # find index to split
    idx_end_train = int((data.dict_df[metadata.TF]["df"].shape[0]-embargo) * perct_train)
    idx_start_valid = min((idx_end_train + embargo), data.dict_df[metadata.TF]["df"].shape[0])
    # create new dataframe
    df_train = data.dict_df[metadata.TF]["df"].iloc[:idx_end_train]
    df_val = data.dict_df[metadata.TF]["df"].iloc[idx_start_valid-1:]
    # reset index
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    # update data structure
    data_train.dict_df[metadata.TF]["df"] = df_train
    data_val.dict_df[metadata.TF]["df"] = df_val
    data_train.df_init = df_train
    data_val.df_init = df_val
    return data_train, data_val, df_train, df_val











































































































































        
