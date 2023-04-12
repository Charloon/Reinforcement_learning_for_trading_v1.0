""" Gym envirnment for trading """
import gym
import copy
import random
from gym import spaces
from gym.utils import seeding
import numpy as np
from env.data import find_time_index, add_reward_limits
from env.reward import calculate_reward
from stable_baselines3.common.utils import set_random_seed

def make_env(df, metadata, data, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = TradingEnvironment(metadata, data, seed)
        #check_env(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

class TradingEnvironment(gym.Env):
    """ Gym environment for trading """
    # Environment for trading, with obervations, action, step
    def __init__(self, metadata, data, seed):
        super(TradingEnvironment, self).__init__()

        # initialize
        random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        self.iteration = 0
        self.metadata = metadata
        self.dfs = data
        self.current_step = 0
        self.current_time = self.dfs.dict_df[self.metadata.TF]["df"][self.metadata.time_feat].values[self.metadata.n_step+1]
        self.current_index = find_time_index(self.dfs.dict_df[self.metadata.TF]["df"],
                                            self.current_time, self.metadata.time_feat)
        self.init_index = self.current_index
        self.current_price = self.dfs.dict_df[self.metadata.TF]["df"]["Close"].values[self.metadata.n_step+1]
        self.step_since_last_trade = 0
        # define action space
        self.define_action_space = {
            # for spot, we only buy, sell or hold, and here only a fixed % each time of the Account balance
            # (% to be defined separately). Here the volume is a float, meaning that you can defien a precise 
            # number of asset to buy/sold, which is right for currencies, but stock would be converted to 
            # the closest integer
            "spot0" : spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
                                   }
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        #self.action_space = flatten_space(self.define_action_space[self.metadata.trading_type])
        self.action_space = self.define_action_space[self.metadata.trading_type]
        
        # Define observations
        # first add Account information columns to the dataframe
        self.dfs.dict_df[self.metadata.TF]["df"]["asset"] = 0.
        self.dfs.dict_df[self.metadata.TF]["df"]["cash"] = 0.
        self.dfs.dict_df[self.metadata.TF]["df"]["balance"] = 0.
        self.dfs.dict_df[self.metadata.TF]["df"]["asset_balance_ratio"] = 0.
        self.dfs.dict_df[self.metadata.TF]["df"]["norm_balance"] = 0.
        self.dfs.dict_df[self.metadata.TF]["list_feat"].extend(["asset_balance_ratio", "norm_balance"])
        # define space for observation
        self.observation_space = spaces.Box(low=0, high=1,
            shape=(self.metadata.n_step+1, len(self.dfs.dict_df[self.metadata.TF]["list_feat"])),
                                                                                 dtype=np.float32)

        # update data with reward precalculation
        self.df_reward = copy.deepcopy(self.dfs.dict_df[self.metadata.TF]["df"])
        self.df_reward = add_reward_limits(self.metadata.account.init_cash, self.df_reward, self.current_index)
        #self.df_reward.to_csv("df_reward.csv")
        # initialize dataframe for results evaluation
        self.df_eval = copy.deepcopy(self.dfs.df_init)
        self.df_eval[["reward", "balance", "cash", "sell", "buy", "volume_order"]] = np.nan
        self.df_eval["sell"] = 0.0
        self.df_eval["buy"] = 0.0

    def _next_observation(self):
        """ update observation """
        idx = self.current_index 
        # update Account information in dataframe
        self.dfs.dict_df[self.metadata.TF]["df"]["asset_balance_ratio"].values[idx] = \
                self.metadata.account.n_asset*self.current_price / self.metadata.account.balance
        self.dfs.dict_df[self.metadata.TF]["df"]["norm_balance"].values[idx] = \
                    (self.metadata.account.balance - self.metadata.account.init_cash)/ \
                        self.metadata.account.init_cash
        self.dfs.dict_df[self.metadata.TF]["df"]["asset"].values[idx] = \
            self.metadata.account.n_asset*self.current_price
        self.dfs.dict_df[self.metadata.TF]["df"]["cash"].values[idx] = self.metadata.account.init_cash
        self.dfs.dict_df[self.metadata.TF]["df"]["balance"].values[idx] = self.metadata.account.balance
        alpha = (1-self.metadata.account.cash/self.metadata.account.balance)*2-1.0
        self.dfs.dict_df[self.metadata.TF]["df"]["alpha"].values[idx] = alpha
        # extract time steps and features to be used for the observation
        df = self.dfs.dict_df[self.metadata.TF]["df"]
        list_feat = self.dfs.dict_df[self.metadata.TF]["list_feat"]
        df_temp = copy.deepcopy(df[list_feat].iloc[idx-self.metadata.n_step:idx+1])
        # normalize observation
        minPrice = np.nanmin(df_temp["Low"])
        maxPrice = np.nanmax(df_temp["High"])
        minVol = np.nanmin(df_temp["Volume"])
        maxVol = np.nanmax(df_temp["Volume"])
        for x in self.dfs.normalize["list_feat_absPrice"]:
            df_temp[x] = (df_temp[x]-minPrice)/(maxPrice-minPrice)
        for x in self.dfs.normalize["list_feat_diffPrice"]:
            df_temp[x] = df_temp[x]/(maxPrice-minPrice)
        for x in self.dfs.normalize["list_feat_absVolume"]:
            df_temp[x] = (df_temp[x]-minVol)/(maxVol-minVol)
        for x in self.dfs.normalize["list_feat_diffVolume"]:
            df_temp[x] = df_temp[x]/(maxVol-minVol)
        for x in self.dfs.normalize["list_feat_percent"]:
            df_temp[x] = df_temp[x]/100
        for x in self.dfs.normalize["list_feat_day"]:
            df_temp[x] = df_temp[x]/31
        for x in self.dfs.normalize["list_feat_month"]:
            df_temp[x] = df_temp[x]/12
        for x in self.dfs.normalize["list_feat_weekday"]:
            df_temp[x] = df_temp[x]/7

        return np.array(df_temp, dtype=np.float32)

    def _take_action(self, action):
        """ take action based on updated observations"""
        # initialize
        final_order = None
        final_volume = None

        # modify the environment (account) based on the selected action
        if self.metadata.trading_type == "spot0":

            # find order and volume
            order = action[0]
            volume = action[1]
            # rescale
            order = (order+1)/2.0*3.0 # rescale from [-1,1] to [0,3]
            volume = (volume+1)/2.0 # rescale from [-1,1] to [0,1]
            # test on possible actions: need enough cash to buy stock or have enough stocks to sell
            shares_bought = int(self.metadata.account.cash / self.current_price * volume)
            shares_sold = int(self.metadata.account.n_asset * volume)

            # if enough cash to share and order to buy
            if order < 1 and shares_bought > 0 : 
                print("Buy")
                # Buy
                self.metadata.account.n_asset += shares_bought
                cost = shares_bought * self.current_price
                print("cost :", cost)
                self.metadata.account.cash -= cost
                # update logs of trades
                self.trades.append({'step': self.current_step, "time": self.current_time,
                    "balance" : self.metadata.account.balance, 
                    "shares": shares_bought, "cost": cost,
                    "order": "buy", "price": self.current_price})
                # save final order status
                final_order = "buy"
                final_volume = volume

            # if enough cash to share and order to buy
            elif order > 2 and shares_sold > 0:
                print("Sell")
                # Sell amount % of shares held
                self.metadata.account.n_asset -= shares_sold
                gain = shares_sold * self.current_price
                self.metadata.account.cash += gain
                # update logs of trades
                self.trades.append({'step': self.current_step, "time": self.current_time,
                    "value" : self.metadata.account.balance,  
                    "shares": shares_sold, "gain": shares_sold * self.current_price,
                    "order": "sell", "price": self.current_price})
                # save final order status
                final_order = "sell"
                final_volume = volume
            else: # no action was taken
                self.step_since_last_trade += 1

            # update balance
            self.metadata.account.update(self.current_price)

        return (final_order, final_volume)

    def step(self, action):
        """ actions and updates to the new step """
        info = ()
        print("##################")
        # Execute one time step within the environment
        final_action = self._take_action(action)

        # update current situation to the next time step
        self.current_step += 1
        df = self.dfs.dict_df[self.metadata.TF]["df"]
        self.current_index += 1
        idx = self.current_index
        print("idx:", idx, "step:", self.current_step)
        self.current_price = df["Close"].values[self.current_index]
        self.current_time = df[self.metadata.time_feat].values[self.current_index]

        # update Account
        self.metadata.account.update(self.current_price)
        self.df_reward["balance"].values[self.current_index] = self.metadata.account.balance

        # reward
        reward, early_stop, info = calculate_reward(self)
            
        # check if episode is done
        # liquidated
        done = self.metadata.account.balance <= 0
        if early_stop:
            done = True
        # exceed max step
        if not done: done = self.current_step >= self.metadata.max_step

        # update observations
        obs = self._next_observation()

        # save results for evaluation
        self.df_eval["reward"].values[idx] = reward
        self.df_eval["balance"].values[idx] = self.metadata.account.balance
        self.df_eval["cash"].values[idx] = self.metadata.account.cash
        self.df_eval["n_asset"].values[idx] = self.metadata.account.n_asset
        if final_action[0] == "buy":
            self.df_eval["buy"].values[idx] = 1
        if final_action[0] == "sell":
            self.df_eval["sell"].values[idx] = 1
        self.df_eval["volume_order"].values[idx] = final_action[1]

        # to print current situation
        if final_action[0] is not None:
            print("reward: ", reward)
            print("balance: ", self.metadata.account.balance)
            print("cash: ", self.metadata.account.cash)
            print("n_asset: ", self.metadata.account.n_asset)
            print("current_price: ", self.current_price)
            print("current_time: ", self.current_time)
        print("current_step: ", self.current_step)
        print("iteration: ", self.iteration)
        if done: 
            print("DONE!!")
            # save on file recorded behavior every episode
            """try:
                self.df_eval.to_csv('./tmp/result_'+str(self.iteration)+'.csv')
            except:
                pass"""

        # for prediction, save results in a separate file and cancel stop
        if self.metadata.mode == "predict":
            self.df_eval.to_csv("df_eval_"+self.metadata.suffix+".csv")
            done = False

        return obs, reward, done, info

    def reset(self):
        print("start reset")
        # save on file recorded behavior every episode
        """try:
            self.df_eval.to_csv('./tmp/result_'+str(self.iteration)+'.csv')
        except:
            pass"""

        # Reset the state of the environment to an initial state
        self.metadata.account.cash = self.metadata.account.init_cash
        self.metadata.account.n_asset = 0
        self.metadata.account.balance = self.metadata.account.cash
        self.trades = []

        # Set the current step back to 0
        self.iteration += 1
        self.current_step = 0
        self.step_since_last_trade = 0
        # for training randomly select the initail step of the episode
        if self.metadata.mode == "training":
            self.init_index = random.randint(self.metadata.n_step+1, 
                self.dfs.dict_df[self.metadata.TF]["df"].shape[0]-self.metadata.max_step-1)
        # for predict start the episode at the first time step
        elif self.metadata.mode == "predict":
            self.init_index = self.metadata.n_step+1
        self.current_index = self.init_index
        self.current_time = self.dfs.dict_df[self.metadata.TF]["df"][self.metadata.time_feat].values[self.init_index]
        self.current_price = self.dfs.dict_df[self.metadata.TF]["df"]["Close"].values[self.init_index]
        # reset account information
        self.dfs.dict_df[self.metadata.TF]["df"]["asset"] = 0.
        self.dfs.dict_df[self.metadata.TF]["df"]["cash"] = self.metadata.account.init_cash
        self.dfs.dict_df[self.metadata.TF]["df"]["balance"] = self.metadata.account.init_cash
        self.dfs.dict_df[self.metadata.TF]["df"]["asset_balance_ratio"] = 0.
        self.dfs.dict_df[self.metadata.TF]["df"]["norm_balance"] = 1.
        # update initial account with a random buy of asset
        if self.metadata.trading_type == "spot0":
            action = [-1, random.uniform(-1, 1)]
        _ = self._take_action(action)
        print("final_action end")        

        # prepare df for results evaluation to record behavior
        self.df_eval = copy.deepcopy(self.dfs.df_init)
        self.df_eval[["reward", "balance", "cash", "n_asset", "sell", "buy", "volume_order"]] = np.nan
        self.df_eval["sell"] = 0.0
        self.df_eval["buy"] = 0.0

        # print info on reset
        print("n_asset", self.metadata.account.n_asset)
        print("price", self.current_price)
        print(self.metadata.account.n_asset*self.current_price)
        print("cash", self.metadata.account.cash)
        print("balance", self.metadata.account.balance)
        print("n_step", self.metadata.n_step)

        return self._next_observation()

    def render(self, mode='human'):
        # Render the environment to the screen
        profit = self.metadata.account.balance - self.metadata.account.init_cash
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.metadata.account.balance}')
        print(f'Profit: {profit}')

