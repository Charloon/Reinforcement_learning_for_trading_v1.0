# Reinforcement_learning_for_trading_v1.0
A lean and simple implementation of Reinforcement Learning for trading stocks.

## Introduction

This is a personnal project to explore the possibilities of Reinforcement Learning (RL). I was curious to learn about its concept and see how it would apply on a real life use case such as trading. I could not find any really applicable open source library, notebook or github, so I started my own from scratch. 

Why RL for trading? Past experiences taught me that Supervised Learning for trading could provide valuable trade signals, but wasn't removing the pitfalls coming from the trader's emotions, such as FOMO, confirmation bias, revenge trade, etc ... Therefore, my ML approach to trading needed to include decision making in the process.

In a nutshell, RL is creating a decision maker (the agent) capable, through try and error, to learn how to navigate a Markov decision process (policy) for the best outcome (to maximize rewards), based on available information (state or observations). Which seems to make it the right appraoch.

## Overview

The present code is built around two main components:
- A custom environment for trading using the Gym framework from Open AI. The environment is responsible to receive update the account balance when it receives an action, sends back observations (like information about the price action and status of the account), and a reward.  
- A RL model (agent) from the Stable Baseline 3 library, also from Open AI. In the present implementation, it is a recurrent PPO, which is a kind Actor-Critic algorithm that uses one LSTM network to predict the newt best action (actor) and a second to evaluate the value of actions in specific context. 

![A2C_animation3](https://user-images.githubusercontent.com/55462061/231509301-3a6ef1cc-81ec-43d2-9de2-d4be84b1a6f8.gif)

The code also include two types of parameter tuning with the Optuna library:
- For the parameter of the agent.
- For the parameter of the environment and weigths of the reward function.
The metrics for the tuning comes from the application to the agent to validation data. 

Below are examples of the performance of the model trained in AAPL stock data on both the training and validation dataset.

![image](https://user-images.githubusercontent.com/55462061/231503942-239bedd3-2907-4fac-b678-4b83826668be.png)

![image](https://user-images.githubusercontent.com/55462061/231504139-8c0c4e31-9223-4cc1-a426-b67921c46a20.png)

## Instalation 

The libraries in the requirement.txt file are required to run. The main ones are Stable Baselines3 and Stable Baselines-contrib, Optuna, and Tensorboard. 
To launch tensorboard, run this command
```
tensorboard --logdir ./tensorboard/
```

Then to run the script main.py. In this file there are several flags to define the type of calculation:
```python
FLAG_HYPERPARAMOPT = True       # flag to run hyperparameter optimization
FLAG_OPT_REWARD = True          # flag to run environment and reward optimization
FLAG_LOAD_STUDY = False         # flag to load existing parametr optimisation study if available
FLAG_USE_OPT_HYPERPARAM = True  # flag to use previously optimized hyperparameters
FLAG_USE_OPT_ENV = True         # flag to use previously optimized environment and reward function parameters
FLAG_TRAIN = True               # flag to train
FLAG_PREDICT = True             # flag to predict 
```

To visualize data, a basic file exist in workflows/vizualize_data.py. It is to be expanded and custom depending of the feature engineering of the code. Similarly, prediction results on training and validation data are visualized with the file worflows/visualize_predict.py. 

## Description

This code is to be consider as aframework to built upon, and some components that are currently limited are aimed to be expended.

### Features
The feature creation is centralized in the file env/data.py, and each group of feature as its own calculation file. The current version has a limited and basic set of features:
- OCHL and volume
- RSI
- moving averages
Obviously, adding more features based on your personal experience of trading should improve the performance of the agent. 

### Data
The example provided with the code is the AAPL stock price history in daily. This is likely not enough to generate a performant general trading agent, as it seems prone to overfitting. One extension of the code would be to enable training and validation on multiple tickers. 

### Oders
This version only as basic orders for stocks, with buy and sell a number fo share. This should first be extended to add stop losses, cost and slippage. The next step would be to consider orders for currencies (crytos and FX), as well as futurs and options. 

### Rewards
The reward system is a weighted average of severals rewards that can be put in two categories:
- Rewards to penalize or confirm some specific behaviors or situation. For example, negative reward if we consider the agent is not trading enough or too much, or positive rewards if the accound balance is satisfactory.
- For algorithm that calculate policy gradient, the reward need to be constructed in a way to incentivize the policy toward the optimimum behavior in all situation, so that it can find its path toward maximimzin the reward. In a nutshell, the approach taken here is to create the features on the training data that would indicate the optimum behavior (I'll call them solution features), test that the agent does find this behavior when these extra features are provided for learning, and after removing these solution features that would not be available in validation, adapt the form of the legitimate features (RSI, moving averages, ect..) to mimic and approximate these solution features. The method is only partially used in this code, but preliminary test were encouraging.

# Conclusion
I hope some will find this code hopefull ffor then own endeavior into Rl trading, and any feedback and suggestion is welcomed. 
If youlike this project, contribution are welcomed as well!
ETH, BNB : 0x009E9e4090906ED58696d44595658cE45E2d114f

