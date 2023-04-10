"""The class Account is to store and update any information related to the account
 containing assets and cash"""

class Account():
    """ Contains information about the account balance """
    def __init__(self, init_cash = 1000, name = "APPL", n_asset = 0):
        self.cash = init_cash
        self.init_cash = init_cash
        self.name = name
        self.n_asset = n_asset
        self.cost = 0.0
        self.balance = self.cash

    def dict_features(self, current_price = 0):
        """ create a dictionary of info about the account for the observation"""
        dict_temp = {}
        dict_temp.update({"cash": self.cash})
        dict_temp.update({self.name: self.n_asset*current_price})
        return dict_temp

    def n_feat(self):
        """ get number of features from account """
        return len(self.dict_features().keys())

    def update(self, current_price):
        """ update account information with current price """
        self.balance = self.cash + self.n_asset * current_price
        return