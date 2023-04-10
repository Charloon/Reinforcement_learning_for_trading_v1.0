""" functions to detect tops and bottoms on different timeframe, 
    used in the reward function"""
import copy

def filter_tops_bottoms(df, feat_name):
    """
    Filter feature for tops and bottom to make sure it alternate tops and bottoms
    Input:
    - df: dataframe to update
    - feat_name: name of the feature with tops and bottoms
    Output:
    - df: updated dataframe
    """
    df_temp = copy.deepcopy(df[df[feat_name]!=0])
    for i in range(df_temp.shape[0]-1):
        j = df_temp.index.values[i]
        j2 = df_temp.index.values[i+1]
        if df_temp[feat_name].values[i]*df_temp[feat_name].values[i+1] > 0:
            if df_temp[feat_name].values[i] == 1:
                if df_temp["High"].values[i] < df_temp["High"].values[i+1]:
                    df[feat_name].values[j] = 0
                else:
                    df[feat_name].values[j2] = 0
            elif df_temp[feat_name].values[i] == -1:
                if df_temp["Low"].values[i] > df_temp["Low"].values[i+1]:
                    df[feat_name].values[j] = 0
                else:
                    df[feat_name].values[j2] = 0
    return df

def calculate_beta(df, tops_bottom_feat):
    """
    Calculate the coefficient beta used in the reward function. 
    Input:
    - df: dataframe to update
    - tops_bottoms_feat: string of the feature to update
    Output:
    - values of beta
    """
    df_temp = copy.deepcopy(df)
    df_temp["beta0"] = 0.
    last_value = 0
    for i in range(1, df_temp.shape[0]):
        if df_temp[tops_bottom_feat].values[i] != 0:
            last_value = df_temp[tops_bottom_feat].values[i]
            i_start = i
            if last_value == 1:
                price_start = df_temp["High"].values[i]
            else:
                price_start = df_temp["Low"].values[i]
            price_start = df_temp["Close"].values[i]
            j=i+1
            while (df_temp[tops_bottom_feat].values[j] == 0) and (j < (df_temp.shape[0]-2)):
                j+=1
            i_end = j
            if last_value == -1:
                price_end = df_temp["High"].values[j]
            else:
                price_end = df_temp["Low"].values[j]
            for k in range(i_start, i_end+1):
                df_temp["beta0"].values[k] = -last_value
                #### alpha #### -1 at start and 1 at end
                #df_temp["alpha"].values[k] = (float(k-i_start)/float(i_end-i_start)*2-1)*last_value
                #### beta ####
                #df_temp["beta0"].values[k] = (last_value*((df_temp["Close"].values[k]-price_start)/ \
                # (price_end-price_start)*2-1))**n
                #df_temp["beta0"].values[k] = -last_value * np.abs((df_temp["Close"].values[k]-price_end)/ \
                # (price_start-price_end))
                #df_temp["gamma"].values[k] = ((df_temp["Close"].values[k]-price_start)/ \
                # (price_end-price_start)*2-1)
    return  df_temp["beta0"]

def find_tops_and_bottom(df, period=14):
    """ to create a features related to tops and bottoms of the price according to a moving average.
    Input:
    - df : dataframe to update
    - period : number of days to be used in moving average
    Output:
    - df : updated dataframe
    - list_feat :  list of added features to the dataframe
    """

    # Initialize
    list_feat = []
    df["Typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["Typical_price_"+str(period)+"MA"] = df["Typical_price"].rolling(period, center=True).mean()
    df_temp = copy.deepcopy(df)
    df_temp["dTypical_price"] = df_temp["Typical_price"].diff()
    df_temp["dTypical_price_"+str(period)+"MA"] = df_temp["Typical_price_"+str(period)+"MA"].diff()
    df_temp["dHigh"] = df_temp["dTypical_price"] + (df_temp["High"] - df_temp["Typical_price"])
    df_temp["dLow"] = df_temp["dTypical_price"] + (df_temp["Low"] - df_temp["Typical_price"])

    # apply centered moving average
    df_temp["dHigh_"+str(period)+"MA"] = df_temp["dHigh"].rolling(period, center=True).mean()
    df_temp["dLow_"+str(period)+"MA"] = df_temp["dLow"].rolling(period, center=True).mean()
    # find tops
    df_temp["tops_"+str(period)+"MA"] = 0 
    df_temp["bottoms_"+str(period)+"MA"] = 0
    df_temp["tops_bottoms_"+str(period)+"MA"] = 0
    feat = "dTypical_price_"+str(period)+"MA"
    delta_index = int(float(period)/2)
    for i in range(2, df_temp.shape[0]):
        i_start = max(0,i-delta_index)
        i_end = min(df_temp.shape[0]-1, i+delta_index+1)
        if df_temp[feat].values[i]*df_temp[feat].values[i-1] < 0:
            if df_temp[feat].values[i] > 0:
                id = df_temp["Low"].values[i_start:i_end].argmin()
                df_temp["tops_bottoms_"+str(period)+"MA"].values[i_start+id] = -1
            else:
                id = df_temp["High"].values[i_start:i_end].argmax()
                df_temp["tops_bottoms_"+str(period)+"MA"].values[i_start+id] = 1
    
    # filter for ducplicated tops or bottoms
    df_temp = filter_tops_bottoms(df_temp, "tops_bottoms_"+str(period)+"MA")
        
    # remove buy that are higher than the next sell
    last_value = 0
    last_price = 0
    last_index = 0
    for i in range(1, df_temp.shape[0]):
        if df_temp["tops_bottoms_"+str(period)+"MA"].values[i] == 1:
            if last_value == -1 and df_temp["Close"].values[i] < last_price:
                df_temp["tops_bottoms_"+str(period)+"MA"].values[i] = 0
                df_temp["tops_bottoms_"+str(period)+"MA"].values[last_index] = 0
            else:
                last_value = 1
                last_price = df_temp["Close"].values[i]
                last_index = i 
        elif df_temp["tops_bottoms_"+str(period)+"MA"].values[i] == -1:
            if last_value == 1 and df_temp["Close"].values[i] > last_price:
                df_temp["tops_bottoms_"+str(period)+"MA"].values[i] = 0
                df_temp["tops_bottoms_"+str(period)+"MA"].values[last_index] = 0
            else:
                last_value = -1
                last_price = df_temp["Close"].values[i]
                last_index = i 

    # add beta
    df_temp["alpha"] = 0.
    df_temp["beta0"] = calculate_beta(df = df_temp, tops_bottom_feat = "tops_bottoms_"+str(period)+"MA")

    # update dataframe
    df["beta0"] = df_temp["beta0"]
    df["alpha"] = 0.
    df["tops_bottoms_"+str(period)+"MA"] = df_temp["tops_bottoms_"+str(period)+"MA"]

    # add LL, HH, LH and HL
    df["HH"] = 0
    df["HL"] = 0
    df["LL"] = 0
    df["LH"] = 0
    df["higherTF_tops_bottoms_"+str(period)+"MA"] = 0
    df_temp = copy.deepcopy(df[df["tops_bottoms_"+str(period)+"MA"] != 0])
    for i in range(df_temp.shape[0]):
        # find LL, HH, HL, LH
        j = df_temp.index.values[i]
        if df_temp["tops_bottoms_"+str(period)+"MA"].values[i] == 1:
            if i > 1:
                if df_temp["High"].values[i] > df_temp["High"].values[i-2]:
                    df["HH"].values[j] = 1
                else:
                    df["LH"].values[j] = 1
        elif df_temp["tops_bottoms_"+str(period)+"MA"].values[i] == -1:
            if i > 1:
                if df_temp["Low"].values[i] < df_temp["Low"].values[i-2]:
                    df["LL"].values[j] = 1
                else:
                    df["HL"].values[j] = 1
        # find tops and bottom at the higher time frame
        if i > 1 and i < df_temp.shape[0]-2:
            if df_temp["tops_bottoms_"+str(period)+"MA"].values[i] == 1:
                if df_temp["High"].values[i] > max(df_temp["High"].values[i-2], df_temp["High"].values[i+2]):
                    df["higherTF_tops_bottoms_"+str(period)+"MA"].values[j] = 1
            elif df_temp["tops_bottoms_"+str(period)+"MA"].values[i] == -1:
                if df_temp["Low"].values[i] < min(df_temp["Low"].values[i-2], df_temp["Low"].values[i+2]):
                    df["higherTF_tops_bottoms_"+str(period)+"MA"].values[j] = -1

    # remove two consecutive tops or bottoms at higher timeframe 
    df = filter_tops_bottoms(df, "higherTF_tops_bottoms_"+str(period)+"MA")

    # calculate beta at higher timeframe    
    df["higherTF_beta"] = calculate_beta(df = df, tops_bottom_feat = "higherTF_tops_bottoms_"+str(period)+"MA")
    list_feat.append("alpha")

    return df, list_feat