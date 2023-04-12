""" Utilities used for training, prediction and parametr optimisation"""
import pickle
import copy
import numpy as np
import pandas as pd
from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.trading_environment import make_env

def organise_hyperparam(dict_param):
    """ function to reorganise parameter from the Optuna optimisation into
     specific shape or dict.
    Input:
    - dict_param : dict of the parameter received from Optuna
    Output:
    - dict_new : updated dict
    """
    # list of parameters going into the kwargs_policy
    param_kwarg_policy = ["optimizer_class", "net_arch", "activation_fn",
                         "ortho_init", "lstm_hidden_size", "n_lstm_layers"]
    dict_new = {}
    kwargs_policy = {}
    for key in dict_param.keys():
        if key in param_kwarg_policy:
            value = dict_param[key]
            # special treatment for net_arch
            if key == "net_arch":
                value = [{"pi": [64], "vf": [64]} if value == "tiny" else {"pi": [64, 64], "vf": [64, 64]}]
            # special treatment for activation_fn
            elif key == "activation_fn":
                value = {"tanh": nn.Tanh, "relu": nn.ReLU}[value]
            kwargs_policy.update({key:value})
        else:
            dict_new.update({key:dict_param[key]})
    # update the policy_kwargs
    if len(kwargs_policy.keys()) > 0:
        dict_new.update({"policy_kwargs":kwargs_policy})
    return dict_new

def organise_env_param(dict_param, metadata):
    """ update metada during environment optimization 
    Input:
    - dict_param : dict of the parameter received from Optuna
    - metadata : Object to be updated
    Output:
    - metadata: updated object
    """
    list_param_env = ["n_step"]
    for key in dict_param.keys():
        if key in list_param_env:
            metadata.n_step = dict_param[key]
        else:
            metadata.kwargs_reward[key] = dict_param[key]
    return metadata

def get_metrics(suffix, code):
    """
    Function to extract some performance metrics from predictions.
    Input: 
    - suffix : string to indicate if the prediction is for training or validation
    - code : unique code to identify the model and training environment
    Output:
    - cumulative_return : normalized cumulative return
    - mean_relative_variation : average relative variation of the balance from one step to the other
    - price_balance_penalty : penalty of -1 if balance finishes below the price
    - df_eval : dataframe with prediction results
    """
    # fetch results from the prediction
    file = open("df_eval_"+suffix+code+".csv", 'rb')
    df_eval = pd.read_csv(file)
    df_eval = df_eval[df_eval["balance"].notna()]
    # calculate cumulative return
    cumulative_return = df_eval["balance"].values[-1]/df_eval["balance"].values[0]
    # calculate mean relative increase
    mean_relative_variation = np.nanmean(np.divide(df_eval["balance"].diff().values[1:-1],
                                 df_eval["balance"].values[2:]))*100
    # penalty if balance finishes below the price
    price_balance_penalty = 0.
    A = df_eval["balance"].values[-1]/df_eval["balance"].values[0]
    B = df_eval["Close"].values[-1]/df_eval["Close"].values[0]
    if A < B:
        price_balance_penalty = -1

    return cumulative_return, mean_relative_variation, price_balance_penalty, df_eval

def export_study_to_csv(study, name = 'study_env.pkl', suffix = "env"):
    """ function to save and export results from Optima study into a csv
    Input:
    - study : Obejct from Optuna optimization
    - name : string to the saved file of the study
    - suffix : string to indicate if this is for env or hyperparam
    """
    # save study
    with open(name, 'wb') as handle:
        pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # export study to csv
    with open(name, 'rb') as handle:
        b = pickle.load(handle)
        df_opt = b.trials_dataframe()
        df_opt.dropna(inplace=True)
        df_opt.reset_index(inplace=True)
        df_name = "df_opt"+suffix+".csv"
        df_opt.to_csv(df_name)
        print("Study exported to", df_name)

def align_env_with_training(metadata, df, data, stats_path):
    """ to update the metada for training to be aligned with the metada used in training
    Input:
    - metadata : object to be updated
    - df : dataframe to be used for the environment
    - data : information about the dataframe
    - stats_path :  path to the saved training environment
    Output:
    - metadata : updated metadata
    """
    # create a dummy environment
    env_temp = DummyVecEnv([make_env(copy.deepcopy(df),
                            copy.deepcopy(metadata),
                            copy.deepcopy(data), 0)])
    # transfer info from the training env to the dummy env
    env_temp = VecNormalize.load(stats_path, env_temp)
    # seach through levels of wrapping for n_step parameter in dummy env to update new metadata
    for i in range(10):
        if "n_step" in dir(env_temp.metadata):
            metadata.n_step = env_temp.metadata.n_step
            return metadata
        elif "unwrapped" in dir(env_temp):
            env_temp = env_temp.unwrapped
    return metadata