#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            s = states[t]
            a = actions[t]
            
            self.Q_sa[s, a] += self.learning_rate * (G - self.Q_sa[s, a])

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    s = env.reset()
    states = [s]
    actions = []
    rewards = []
    
    for t in range(n_timesteps):
        if t % eval_interval == 0:
            mean_return = pi.evaluate(eval_env, max_episode_length=max_episode_length)
            eval_returns.append(mean_return)
            eval_timesteps.append(t)
            
        a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
        s_next, r, done = env.step(a)
        
        actions.append(a)
        rewards.append(r)
        states.append(s_next)
           
        if done or len(actions) == max_episode_length:
            pi.update(states, actions, rewards)            
            s = env.reset()
            states = [s]
            actions = []
            rewards = []
        else:
            s = s_next

        if plot:
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1)

    return np.array(eval_returns), np.array(eval_timesteps)
def test():
    n_timesteps = 100000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.01

    policy = 'egreedy'
    epsilon = 0.1
    temp = 1.0
    plot = True

    eval_returns, eval_timesteps = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    print(eval_returns, eval_timesteps)
    
if __name__ == '__main__':
    test()