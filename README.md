# Reinforcement_learning_for_trading_v1.0
A lean and simple implementation of Reinforcement Learning for trading stocks.

## Introduction

The repo is a personnal project to explore the possibilities of Reinforcement Learning (RL). I was curious to learn about its concepts and see how it would apply on a real life use case such as trading. I could not find any really applicable open source library, notebook or github, so I started my own from scratch. 

Why RL for trading? Past experiences taught me that Supervised Learning for trading could provide valuable trade signals, but wasn't removing the pitfalls coming from the trader's emotions, such as FOMO, confirmation bias, revenge trade, etc ... Therefore, my ML approach to trading needed to include decision making in the process.

In a nutshell, RL is creating a decision maker (the agent) capable, through try and error, to learn how to navigate a Markov decision process (policy) for the best outcome (to maximize rewards), based on available information (state or observations). Which seems to make it the right approach.

## Overview

The present code is built around two main components:
- A custom environment for trading using the Gym framework from Open AI. The environment is responsible to update the account balance when it receives an action, sends back observations (like information about the price action and status of the account), and a reward.  
- A RL model (agent) from the Stable Baseline 3 library, also from Open AI. In the present implementation, it is a recurrent PPO, which is a kind of Actor-Critic algorithm that uses one LSTM network to predict the new best action (actor), and a second to evaluate the value of a action in a specific context. 

![A2C_animation3](https://user-images.githubusercontent.com/55462061/231509301-3a6ef1cc-81ec-43d2-9de2-d4be84b1a6f8.gif)

The code also includes two types of parameter tuning with the Optuna library:
- For the parameter of the agent.
- For the parameter of the environment and weigths of the reward function.
The metrics for the tuning comes from the application to the agent to validation data. 

Below are examples of the performance of the model trained in AAPL stock data on both the training and validation dataset.

![image](https://user-images.githubusercontent.com/55462061/231503942-239bedd3-2907-4fac-b678-4b83826668be.png)

![image](https://user-images.githubusercontent.com/55462061/231504139-8c0c4e31-9223-4cc1-a426-b67921c46a20.png)

## Installation

The libraries required are in the requirement.txt. The main ones are Stable Baselines3 Contrib, Optuna, and Tensorboard. 
To launch tensorboard, run this command:
```
tensorboard --logdir ./tensorboard/
```
Then run the script main.py. In this file, several flags to define the type of computation:
```python
FLAG_HYPERPARAMOPT = True       # flag to run hyperparameter optimization
FLAG_OPT_REWARD = True          # flag to run environment and reward optimization
FLAG_LOAD_STUDY = False         # flag to load existing parametr optimisation study if available
FLAG_USE_OPT_HYPERPARAM = True  # flag to use previously optimized hyperparameters
FLAG_USE_OPT_ENV = True         # flag to use previously optimized environment and reward function parameters
FLAG_TRAIN = True               # flag to train
FLAG_PREDICT = True             # flag to predict 
```

To visualize data, a basic file in workflows/vizualize_data.py is available. It is to be expanded and customized depending on the feature engineering of the code. Similarly, prediction results on training and validation data are visualized with the file worflows/visualize_predict.py. 

## Description

This code is to be considered as a framework to built upon, and some components that are currently limited, are intended to be expended.

### Features
The feature creation is centralized in the file env/data.py, and each group of features has its own file for computation. The current version has a limited and basic set of features:
- OCHL and volume
- RSI
- moving average
Obviously, adding more features based on your personal experience of trading should improve the performance of the agent. 

### Data
The example data provided with the code is the AAPL stock price history in daily. This is likely not enough to generate a well performing generalized trading agent, as it seems prone to overfitting. One extension of the code would be to enable training and validation on multiple tickers. 

### Trading Oders
This version only has basic orders for stocks, with buy and sell a number of shares. This should first be extended to add stop losses, cost and slippage. The next step would be to consider orders for currencies (crytos and FX), as well as futures and options trading.

### Rewards
The design of reward function is said to be an art, and after taking a few steps into the rabbit hole, I confirm. What I have implemented so far is a weighted average of severals rewards that can be put in two categories:
- Rewards to penalize or confirm some specific behaviors or situation. For example, negative rewards if we consider the agent is not trading enough (or too much), or positive rewards if the accound balance is satisfactory.
- For algorithms that uses policy gradient (such as PPO), the reward needs to be constructed in a way that incentivize the policy toward the optimimum behavior in all situations, so that it can find its path toward the maximumn reward. In a nutshell, the approach taken here can be broken into 3 steps:
  1. Create features on the training data that would indicate the optimum behavior (I'll call them solution features). For example, some of these features are based on identified tops and bottoms. 
  2. Test that the agent does find this optimum behavior when these extra features are provided for learning. Then remove from the training dataset the solution features that won't be available in validation (like tops and bottoms).
  3. Adapt the form of the legitimate features (RSI, moving averages, ect..) to mimic and approximate these removed solution features.
This method is only partially used in this code, but preliminary test were encouraging.

# Conclusion
I hope you will find this code usefull for your own journey into RL trading, and any feedback or suggestion is welcome. 
And if you like this project, contributions are appreciated:
ETH, BNB : 0x009E9e4090906ED58696d44595658cE45E2d114f

