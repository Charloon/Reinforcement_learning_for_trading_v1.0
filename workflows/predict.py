""" Function run prediction with an RL model """
import numpy as np
import copy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from env.trading_environment import make_env
from env.utils import clean_results_folder
from workflows.utils import align_env_with_training

def run_predict(df, metadata, data, algo, stats_path, suffix, run_code = ""):
    """ function to run prediction
    Input:
    - df : dataframe use in the prediction
    - metadata : information to setup the environment
    - data : information about the data
    - algo : string describing the RL algorithm
    - stats_path : path to save the environment
    - suffix : string to indicate what type of data we are predicting (train, valid)
    - run_code :  unique code to identify the model"""

    clean_results_folder()
    # load model
    model = RecurrentPPO.load("model_"+run_code)
    # update the number of step to the length of the dataset
    metadata.max_step = df.shape[0]-metadata.n_step
    # set the model to predict and update the suffix
    metadata.mode = "predict"
    metadata.suffix = suffix+run_code

    # update metadata with critical info from training environment
    metadata = align_env_with_training(metadata, df, data, stats_path)

    # create the gym environment
    env = DummyVecEnv([make_env(copy.deepcopy(df),
                                copy.deepcopy(metadata),
                                copy.deepcopy(data), 0)] )
    env = VecNormalize.load(stats_path, env)

    # do not update the moving avergae of the vector normalization during prediction 
    env.training = False
    # reward normalization in te vector normalization is not needed during prediction
    env.norm_reward = False

    # prepare episode for prediction
    obs = env.reset()
    done = False
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while not done:
        try:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, _ , dones , _ = env.step(action)
            episode_starts = dones
        except Exception as error:
            print(error)
            print("Done!")
            done = True
            pass