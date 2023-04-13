# Reinforcement_learning_for_trading_v1.0
A lean and simple implementation of Reinforcement Learning for trading stocks.

## Introduction

This is a personnal project to explore the possibilities of Reinforcement Learning (RL). I was curious to learn about its concept and see how it would apply on a real life use case such as trading. I could not find any really applicable open source library, notebook or github, so I started my own from scratch. 

Why RL for trading? Past experiences taught me that Supervised Learning for trading could provide valuable trade signals, but wasn't removing the pitfalls coming from the trader's emotions, such as FOMO, confirmation bias, revenge trade, etc ... Therefore, my ML approach to trading needed to include decision making in the process.

In a nutshell, RL is creating a decision maker (the agent) capable, through try and error, to learn how to navigate a Markov decision process (policy) for the best outcome (to maximize rewards), based on available information (state or observations). Which seems to make it the right appraoch.

## Description

The present code is built around two main components:
- A custom environment for trading using the Gym framework from Open AI.  
- A RL model from the Stable Baseline 3 library, also from Open AI. 

![A2C_animation3](https://user-images.githubusercontent.com/55462061/231509301-3a6ef1cc-81ec-43d2-9de2-d4be84b1a6f8.gif)

![image](https://user-images.githubusercontent.com/55462061/231503942-239bedd3-2907-4fac-b678-4b83826668be.png)

![image](https://user-images.githubusercontent.com/55462061/231504139-8c0c4e31-9223-4cc1-a426-b67921c46a20.png)
