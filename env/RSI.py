""" function to create features based on RSI indicator """
import copy
import numpy as np
from env.utils import cross_detection_oscillator

def RSI_formula(df_org, close, n_period = 14):
    """ Calculate a feature for the RSI
    input:
    - df_org: dataframe to update
    - close: string of the feature for close price
    - n_period: number of days used for the RSI calclation
    Output:
    - df_org: updated dataset
    - list with the strings of created features
    """
    df = copy.deepcopy(df_org)
    df["U"] = np.where(df[close].diff() < 0, 0,  df[close].diff())
    df["D"] = np.where(-df[close].diff() < 0, 0,  -df[close].diff())
    df["U"] = df["U"].rolling(window = n_period).mean()
    df["D"] = df["D"].rolling(window = n_period).mean()
    df["RS"] = df["U"]/df["D"]
    df_org["RSI"] = 100 - 100/(1+df["RS"])
    return df_org, ["RSI"]

def add_RSI_feat(df):
    """
    add features related to the RSI
    Input: 
    - df: dataframe to update
    Output:
    - df: updated dataframe
    - list_new_feat: list of new features (strings)
    - list_feat_percent: list of new feature which dimension is percentage (for normalization)
    """
    list_new_feat = []
    # calculate RSI
    df, list_feat_RSI = RSI_formula(df, "Close", 14)
    list_new_feat.extend(list_feat_RSI)
    list_feat_percent = list_feat_RSI
            
    # detect crossing of RSI
    df, list_feat = cross_detection_oscillator(df, list_feat_RSI[0])
    list_new_feat.extend(list_feat)

    return df, list_new_feat, list_feat_percent

