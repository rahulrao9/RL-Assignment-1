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

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        T_ep = len(actions)
        for t in range(T_ep):
            m = min(n, T_ep - t)
            target = sum(self.gamma ** i * rewards[t + i] for i in range(m))
            if not (done and t + m == T_ep):
                target += self.gamma ** m * np.max(self.Q_sa[states[t + m]])
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (target - self.Q_sa[states[t], actions[t]])
            
def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=False, n=5, eval_interval=500):
    ''' runs a single repetition of an n-step Q-learning agent ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    t = 0
    while t < n_timesteps:
        s = env.reset()            
        states  = [s]              
        actions = []               
        rewards = []              
        done = False              

        for _ in range(max_episode_length):
            if t >= n_timesteps:
                break
                
            if t % eval_interval == 0:
                mean_return = pi.evaluate(eval_env)        
                eval_returns.append(mean_return)
                eval_timesteps.append(t)
            
            a = pi.select_action(s, policy, epsilon, temp)   
            s_next, r, done = env.step(a)
            if plot:
                env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
            actions.append(a)
            rewards.append(r)
            states.append(s_next)
            
            t += 1
            
            if done:
                break
                
            s = s_next            
        pi.update(states, actions, rewards, done, n)                
        
    return np.array(eval_returns), np.array(eval_timesteps)

def test():
    n_timesteps = 15000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()