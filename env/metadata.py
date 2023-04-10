""" To define the class Metadata that will store all inforpation necessary 
for the gym environment"""
from env.account import Account

class Metadata():
    """
    This class is to store information to build the environment:
    - market: string to describe the type of market
    - render_mode = ununsed for now
    - account = Object to contain account related information and methods
    - time_feat = string for the name of the time feature
    - trading_type = string to select among different trading style and account managment style
    - TF = string to describe the timeframe
    - n_step = number of previous time step to consider
    - max_step = maximum number of step for each episodes
    - mode = string to descrive what the environment will be used for
    - suffix = string to save and identify outputs from this environment
    - kwargs_reward = dict that contains weights for the reward function
    """
    def __init__(self, market = "stock", render_mode = ['human'], account = Account(),
                time_feat = "Date", trading_type ="spot0", TF = "3d", n_step = 5, max_step = 5,
                mode = "training", suffix = "", kwargs_reward = {}):
        self.render_modes = render_mode
        self.market = market
        self.account = account
        self.time_feat = time_feat
        self.trading_type = trading_type
        self.TF = TF
        self.n_step = n_step
        self.trades = []
        self.max_step = max_step
        self.mode = mode
        self.suffix = suffix
        self.kwargs_reward = kwargs_reward