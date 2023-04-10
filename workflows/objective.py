""" Objective functions for the hyperparameter and environment parameters optimization"""
import binascii
import copy
import os
import optuna
from workflows.training import run_training
from workflows.predict import run_predict
from workflows.utils import get_metrics, export_study_to_csv
from workflows.visualize_predict import visualize_predict

def sample_rPPO_params(trial: optuna.Trial):
    """Sampler for PPO hyperparameters."""
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    n_lstm_layers = trial.suggest_categorical("n_lstm_layers", [1, 2])
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256, 512])
    return {
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "policy_kwargs": {
            "lstm_hidden_size": lstm_hidden_size,
            "n_lstm_layers": n_lstm_layers,
        },
    }

def objective_hyperparam(trial, default_param, df_train, df_val, metadata, data_train, data_val,
                         num_cpu, study, algo, total_timesteps, info_keywords, flag_use_opt_env):
    """ objecive function for the hyperparameter optimization
    Input: 
    - trial : Optuna sampling of parameters
    - default_param : dict of default parameter for the RL algorithm 
    - df_train : dataframe for training
    - df_val : dataframe for validation
    - metadata : information to setup the environment
    - data_train: information specific to the training
    - data_val :  information specific to the validation
    - num_cpu :  number of cpus to parallelise the environment
    - study :  Optuna study
    - algo : string to specify the type of RL algorithm
    - total_timesteps : total number of steps for each training
    - info_keywords : list of string for additional output to the parameter optimisation 
    - flag_use_opt_env : boolean to use optimise parameter for the environment
    """
    # export study to follow on optimization progress
    export_study_to_csv(study, name = 'study.pkl', suffix = "hyperparam")

    # define parameters for selected algorithm
    kwargs = default_param
    if algo == "RecurrentPPO":
        kwargs.update(sample_rPPO_params(trial))

    # define unique code for each model
    code = binascii.b2a_hex(os.urandom(5))
    code = code.decode('ascii')

    # define path to save the environment
    log_dir = "tmp/"
    stats_path = os.path.join(log_dir, "vec_normalize"+code+".pkl")

    # train the model
    run_training(copy.deepcopy(df_train), copy.deepcopy(metadata), copy.deepcopy(data_train),
                algo, default_param, num_cpu, total_timesteps, info_keywords = info_keywords,
                flag_use_opt_hyperparam = False, flag_use_opt_env = flag_use_opt_env,
                stats_path = stats_path, run_code = code)

    # predict on validation data
    run_predict(copy.deepcopy(df_val), copy.deepcopy(metadata), copy.deepcopy(data_val),
                algo, stats_path, "valid", run_code = code)

    # plot results
    visualize_predict(suffix = "valid", code = code, tick = metadata.account.name)

    # get metrics
    _ , mean_relative_var_valid, price_balance_penalty_valid, _ \
                = get_metrics("valid", code)

    # value to maximize
    metric = mean_relative_var_valid + price_balance_penalty_valid

    return metric

def sample_env_params(trial: optuna.Trial):
    """ Sampler for environment parameters. 
    the parameter sampled here are n_step, the number of previous time
    step consider in training, and coefficient that are weighting the different 
    component of the reward function 
    """
    n_step = trial.suggest_int("n_step", 5, 30, 1)
    coef_0 = trial.suggest_float("coef_0", 0., 1.)
    coef_1 = trial.suggest_float("coef_1", 0., 1.)
    coef_2 = trial.suggest_float("coef_2", 0., 1.)
    coef_3 = trial.suggest_float("coef_3", 0., 1.)
    coef_4 = trial.suggest_float("coef_4", 0., 1.)
    coef_5 = trial.suggest_float("coef_5", 0., 1.)
    coef_6 = trial.suggest_float("coef_6", 0., 1.)
    coef_7 = trial.suggest_float("coef_7", 0., 1.)

    return {
        "n_step": n_step,
        "kwargs_reward":{ 
        "coef_0": coef_0,
        "coef_1": coef_1,
        "coef_2": coef_2,
        "coef_3": coef_3,
        "coef_4": coef_4,
        "coef_5": coef_5,
        "coef_6": coef_6,
        "coef_7": coef_7}
        }

def objective_envparam(trial, default_param, df_train, df_val, metadata, data_train, data_val,
            num_cpu, study, algo, total_timesteps, info_keywords, flag_use_opt_hyperparam):
    """ objecive function for the hyperparameter optimization
    Input: 
    - trial : Optuna sampling of parameters
    - default_param : dict of default parameter for the RL algorithm 
    - df_train : dataframe for training
    - df_val : dataframe for validation
    - metadata : information to setup the environment
    - data_train: information specific to the training
    - data_val :  information specific to the validation
    - num_cpu :  number of cpus to parallelise the environment
    - study :  Optuna study
    - algo : string to specify the type of RL algorithm
    - total_timesteps : total number of steps for each training
    - info_keywords : list of string for additional output to the parameter optimisation 
    - flag_use_opt_hyperparam : boolean to use optimise hyperparameter of the RL algorithm
    """
    # export study to csv
    export_study_to_csv(study, name = 'study_env.pkl', suffix = "env")

    # update environment with new parameters
    sample = sample_env_params(trial)
    metadata.kwargs_reward = sample["kwargs_reward"]
    metadata.n_step = sample["n_step"]

    # define unique code for each model
    code = binascii.b2a_hex(os.urandom(5))
    code = code.decode('ascii')
    # path = "tmp/TestMonitor_"+code

    # define path to save the environment
    log_dir = "tmp/"
    stats_path = os.path.join(log_dir, "vec_normalize"+code+".pkl")

    # train the model
    run_training(copy.deepcopy(df_train), copy.deepcopy(metadata), copy.deepcopy(data_train),
                algo, default_param, num_cpu, total_timesteps, info_keywords = info_keywords,
                flag_use_opt_hyperparam = flag_use_opt_hyperparam,
                flag_use_opt_env = False, stats_path = stats_path, run_code = code)

    # predict on validation data
    run_predict(copy.deepcopy(df_val), copy.deepcopy(metadata), copy.deepcopy(data_val),
                algo, stats_path, "valid", run_code = code)

    # plot results
    visualize_predict(suffix = "valid", code = code, tick = metadata.account.name)

    # get metrics
    _, mean_relative_var_valid, price_balance_penalty_valid, _ \
                 = get_metrics("valid", code)

    # value to maximize
    metric = mean_relative_var_valid + price_balance_penalty_valid

    return metric