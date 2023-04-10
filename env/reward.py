""" We define here the reward function. """
import numpy as np

def calculate_reward(env):
    """ calculate the reward:
        Input:
        - env: the gym environment that contains all the info required.
        Output:
        - reward: value of the reward
        - early_stop: boolean to say if the episode should stop based on the reward
        - info: a dict used to output information during reward design optimization
        """
    # initialize
    idx = env.current_index
    base_value = max(env.df_reward["all_asset"].values[idx],
                     env.metadata.account.init_cash) # reference potential profit
    balance = env.metadata.account.balance
    min_step = 5 # number of step before considering early stop
    # weigths for the different type of rewards
    coef_0 = env.metadata.kwargs_reward["coef_0"] if "coef_0" in \
         env.metadata.kwargs_reward.keys() else 1.
    coef_1 = env.metadata.kwargs_reward["coef_1"] if "coef_1" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_2 = env.metadata.kwargs_reward["coef_2"] if "coef_2" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_3 = env.metadata.kwargs_reward["coef_3"] if "coef_3" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_4 = env.metadata.kwargs_reward["coef_4"] if "coef_4" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_5 = env.metadata.kwargs_reward["coef_5"] if "coef_5" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_6 = env.metadata.kwargs_reward["coef_6"] if "coef_6" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_7 = env.metadata.kwargs_reward["coef_7"] if "coef_7" in \
        env.metadata.kwargs_reward.keys() else 1.
    
    # reward saying portfolio should be made of cash or asset depending on where we
    # are between tops and bottoms
    alpha = (1-env.metadata.account.cash/env.metadata.account.balance)*2-1.0 # no cash,
                                                          # alpha=1, only cash, alpha=-1
    beta = env.dfs.dict_df[env.metadata.TF]["df"]["beta0"].values[env.current_index]
    reward0 = alpha * beta

    # same as reward0, but for tops and bottoms defined at a higher timeframe
    higherTF_beta = env.dfs.dict_df[env.metadata.TF]["df"]["beta0"].values[env.current_index]
    reward1 = alpha * higherTF_beta

    # reward depening if balance improved from previous time step
    balance_previous = env.dfs.dict_df[env.metadata.TF]["df"]["balance"].values[env.current_index-1]
    print(balance, env.df_reward["balance"].values[env.current_index])
    relative_profit = (balance-balance_previous)/balance_previous
    std_relative_Close = np.nanstd(np.divide(np.diff(env.df_reward["Close"]) ,
                            env.df_reward["Close"].values[:-1]))

    # reward2 is based in on the profit normalized by the price volatility
    reward2 = relative_profit/(2*max(std_relative_Close, 1e-10))

    # reward3 is based on the length of the episode, to reward avoiding early stops
    reward3 = float(env.current_step)/env.metadata.max_step*2-1

    # reward4 gives penaty if too few trades
    max_step_without_trade = 20
    reward4 = min(0.,(max_step_without_trade - env.step_since_last_trade)/(env.metadata.max_step - max_step_without_trade))

    # reward5 based on a simple sharpe ratio
    profits = np.divide(np.diff(env.df_reward["balance"].values[env.init_index:env.current_index+1]) ,
                            env.df_reward["balance"].values[env.init_index:env.current_index])
    sharpe_ratio = np.mean(profits)/max(np.std(profits), 1e-10)
    reward5 = 0.
    if env.current_step > min_step:
        reward5 = sharpe_ratio

    # reward6 is a penalty if balance is too low
    reward6 = 0.
    if max(-1., min(1., balance/base_value - 1.)) < -0.2:
        reward6 = -1.
        
    # early stopping base on reward 6
    early_stop = False
    if reward6 == -1. and env.current_step > min_step:
        early_stop = True

    # extra reward reach the end with a satifactory PnL
    reward7 = 0.
    if env.current_step == env.metadata.max_step-1 and not early_stop:
        reward7 = 1.

    # define final reward
    reward = np.mean([reward0*coef_0, reward1*coef_1, reward2*coef_2, reward3*coef_3,
                      reward4*coef_4, reward5*coef_5, reward6*coef_6, reward7*coef_7])

    # add output infos
    info = {}
    """for i in range(6):
    info.update({str(i): list_reward}) """
  
    return reward, early_stop, info
