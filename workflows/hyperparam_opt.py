""" script to optimize hyperparameter of the RL algorithm """
import json
import os
import pickle
import optuna
from env.utils import clean_results_folder
from workflows.utils import export_study_to_csv
from workflows.objective import objective_hyperparam

def run_hyperparam_opt(df_train, df_val, metadata, data_train, data_val, algo,
     default_param, num_cpu, info_keywords = [], timeout = 3600*8, n_trials = 50,
     flag_load_study = False, flag_use_opt_env = True, total_timesteps = 30000):
    """ function to run an optimization of some environment parameter.
    Input: 
    - df_train : dataframe for training
    - df_val : dataframe for validation
    - metadata : information to setup the environment
    - data_train: information specific to the training
    - data_val :  information specific to the validation
    - algo : string to specify the type of RL algorithm
    - default_param : dict of default parameter for the RL algorithm 
    - num_cpu :  number of cpus to parallelise the environment
    - info_keywords : list of string for additional output to the parameter optimisation 
    - timeout : seconds before forcing a stop to the optimization
    - n_trials : number of test for the Optuna search
    - flag_load_study : boolean to start from a preexisting study 
    - flag_use_opt_env : boolean to use optimise hyperparameter for the environment
    - total_timesteps : total number of steps for each training"""

    ### run hyperparameter optimization
    # create or load a study
    if os.path.isfile('study.pkl') and flag_load_study:
        with open('study.pkl', 'rb') as handle:
            study = pickle.load(handle)
    else:
        study = optuna.create_study(direction="maximize")

    # run Optuna search
    clean_results_folder()
    study.optimize(lambda trial: objective_hyperparam(trial, default_param, df_train,
                                                df_val, metadata, data_train, data_val,
                                                num_cpu, study, algo, total_timesteps,
                                                info_keywords, flag_use_opt_env),
                                n_trials=n_trials, timeout=timeout)

    # save best parameter in file
    with open("best_params.json", "w") as fp:
        json.dump(study.best_params, fp)

    # analyze study
    export_study_to_csv(study, name = 'study.pkl', suffix = "hyperparam")