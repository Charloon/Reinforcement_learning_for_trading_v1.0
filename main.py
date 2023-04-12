""" Main file to run the different workflows to train and predict, tune model hyperparameters and reward parameters."""
import json
import os
import pandas as pd
from pathlib import Path
from env.account import Account
from env.metadata import Metadata
from env.data import Generate_df, split_train_val
from workflows.hyperparam_opt import run_hyperparam_opt
from workflows.env_opt import run_envparam_opt
from workflows.training import run_training
from workflows.predict import run_predict
from workflows.utils import organise_env_param
from workflows.visualize_predict import visualize_predict

if __name__ == '__main__':
    ################ initialize ##################
    # workflows
    FLAG_HYPERPARAMOPT = False      # flag to run hyperparameter optimization
    FLAG_OPT_REWARD = False         # flag to run environment and reward optimization
    FLAG_LOAD_STUDY = False         # flag to load existing parametr optimisation study if available
    FLAG_USE_OPT_HYPERPARAM = True  # flag to use previously optimized hyperparameters
    FLAG_USE_OPT_ENV = True         # flag to use previously optimized environment and reward function parameters
    FLAG_TRAIN = True               # flag to train
    FLAG_PREDICT = True             # flag to predict 
    # parameters
    NUM_CPU = 8                     # number of environment to run in parallel
    TOTAL_TIMESTEPS = 30000         # number for time steps for each training 
    ALGO = "RecurrentPPO"           # selected RL algorithm
    NAME_ASSET = "AAPL"             # name of the asset
    INFO_KEYWORDS = tuple([])
    LOG_DIR = "tmp/"
    STATS_PATH = os.path.join(LOG_DIR, "vec_normalize.pkl")
    
    # create directory for temp file
    os.makedirs(LOG_DIR, exist_ok=True)

    ################ create gym environment ################
    # create Account
    Account = Account(init_cash = 1000.0, name = NAME_ASSET, n_asset = 0)

    # create metadata
    metadata = Metadata(market = "Crypto",
                        render_mode = None,
                        account = Account,
                        time_feat = "Date",
                        trading_type ="spot0",
                        TF = "1d",
                        n_step = 20,
                        max_step = 100, 
                        mode = "training")

    # load optimum env parameter
    if FLAG_USE_OPT_ENV and Path("best_envparams.json").is_file():
        with open("best_envparams.json", "r") as fp:
            metadata = organise_env_param(json.load(fp), metadata)

    # import data from csv
    df = pd.read_csv('./data/'+NAME_ASSET+".csv").sort_values('Date')

    # get preprocessed data for training and validation
    # preprocess data 
    data = Generate_df(df, metadata.TF, metadata.time_feat)
    # split data into training and validation
    data_train, data_val, df_train, df_val = split_train_val(data, metadata,
                                                            perct_train = 0.8, embargo = 0)

    # define default parameter for RL algorithm
    default_param = {"policy": "MlpLstmPolicy", "verbose":2, "seed":1, "n_steps": metadata.max_step,
                    "target_kl": 0.1, "clip_range": 0.3}

    ################## Optimze hyperparameter of the rl agorithm #######################        
    if FLAG_HYPERPARAMOPT:
        run_hyperparam_opt(df_train, df_val, metadata, data_train, data_val, ALGO, default_param,
                        NUM_CPU, info_keywords = INFO_KEYWORDS, timeout = 3600*4, n_trials = 50,
                        flag_load_study = FLAG_LOAD_STUDY, flag_use_opt_env = FLAG_USE_OPT_ENV,
                        total_timesteps = TOTAL_TIMESTEPS)

    ################# Optimize parameter of the environment and reward function ########
    if FLAG_OPT_REWARD:  
        run_envparam_opt(df_train, df_val, metadata, data_train, data_val, ALGO, default_param,
                        NUM_CPU, info_keywords = INFO_KEYWORDS, timeout = 3600*4, n_trials = 50,
                        flag_load_study = FLAG_LOAD_STUDY, total_timesteps = TOTAL_TIMESTEPS)

    ################# train final model ###############################################  
    if FLAG_TRAIN:
        run_training(df_train, metadata, data_train, ALGO, default_param, NUM_CPU, TOTAL_TIMESTEPS,
                    info_keywords = INFO_KEYWORDS, flag_use_opt_hyperparam = FLAG_USE_OPT_HYPERPARAM,
                    flag_use_opt_env = FLAG_USE_OPT_ENV, stats_path = STATS_PATH)

    ################# predict with final model ###################
    if FLAG_PREDICT:
        # predict on training data
        run_predict(df_train, metadata, data_train, ALGO, STATS_PATH, "train")    
        visualize_predict(suffix = "train", tick = Account.name)
        # predict on training data
        run_predict(df_val, metadata, data_val, ALGO, STATS_PATH, "valid")
        visualize_predict(suffix = "valid", tick = Account.name)   