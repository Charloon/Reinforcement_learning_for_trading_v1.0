""" to train RL model"""
import json
import copy
from pathlib import Path
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from env.trading_environment import make_env
from env.utils import clean_results_folder
from workflows.utils import organise_hyperparam, organise_env_param

def run_training(df, metadata, data, algo, default_param, num_cpu, total_timesteps,
                 info_keywords = [], flag_use_opt_hyperparam = True,
                flag_use_opt_env = True, stats_path = "./", run_code = ""):
    """ function to train RL models
    Input:
    - df : dataframe use in the prediction
    - metadata : information to setup the environment
    - data : information about the data
    - algo : string describing the RL algorithm
    - default_param : dict of default parameter for the RL algorithm 
    - num_cpu :  number of cpus to parallelise the environment
    - total_timesteps : total number of steps for each training
    - info_keywords : list of string for additional output to the parameter optimisation
    - flag_use_opt_hyperparam : boolean to use optimise hyperparameter for the RL algorithm
    - flag_use_opt_env : boolean to use optimise hyperparameter for the environment
    - stats_path : path to save the environment
    - run_code :  unique code to identify the model"""

    # Initialize
    clean_results_folder()
    kwargs = default_param

    # try load existing saved best param
    if flag_use_opt_hyperparam and Path("best_params.json").is_file():
        with open("best_params.json", "r") as fp:
            kwargs.update(organise_hyperparam(json.load(fp)))

    # load optimum env parameter
    if flag_use_opt_env and Path("best_envparams.json").is_file():
        with open("best_envparams.json", "r") as fp:
            metadata = organise_env_param(json.load(fp), metadata)

    # create environment
    env = DummyVecEnv([make_env(copy.deepcopy(df),
                                copy.deepcopy(metadata),
                                copy.deepcopy(data), i) for i in range(num_cpu)])
    env = VecNormalize(env)
    env = VecMonitor(env, filename = "tmp/TestMonitor",
                            info_keywords = info_keywords)
    kwargs["env"] = env

    # run traning
    if algo == "RecurrentPPO":
        model = RecurrentPPO(**kwargs, tensorboard_log="./tensorboard/")
    model.learn(total_timesteps=total_timesteps, log_interval=True)

    # save trained model and environment
    model.save("model_"+run_code)
    env.save(stats_path)
