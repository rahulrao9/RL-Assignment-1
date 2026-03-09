#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""
import argparse
import numpy as np
import time

from Q_learning import q_learning
from SARSA import sarsa
from Nstep import n_step_Q
from MonteCarlo import monte_carlo
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', 
                    epsilon=None, temp=None, smoothing_window=None, plot=False, n=5, eval_interval=500):

    returns_over_repetitions = []
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        if backup == 'q':
            returns, timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
        elif backup == 'sarsa':
            returns, timesteps = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
        elif backup == 'nstep':
            returns, timesteps = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n, eval_interval)
        elif backup == 'mc':
            returns, timesteps = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, eval_interval)

        returns_over_repetitions.append(returns)
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions  
    if smoothing_window is not None: 
        learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, timesteps  

def experiment(args):
    ####### Settings
    n_repetitions = args.n_repetitions
    smoothing_window = 35 # Must be an odd number. Use 'None' to switch smoothing off!
    plot = args.plot # Plotting is very slow, switch it off when we run repetitions
    
    # MDP    
    n_timesteps = args.n_timesteps # Set one extra timestep to ensure evaluation at start and end
    eval_interval = args.eval_interval
    max_episode_length = 100
    gamma = 1.0
    
    # Nice labels for plotting
    backup_labels = {'q': 'Q-learning',
                  'sarsa': 'SARSA',
                  'mc': 'Monte Carlo',
                  'nstep': 'n-step Q-learning'}
    
    ####### Experiments
    optimal_episode_return = 83.68 # set the optimal return per episode you found in the DP assignment here
    
    if args.run_exp2:
        print(f"--- Running Assignment 2: Effect of exploration ({n_repetitions} repetitions) ---")
        policy = 'egreedy'
        epsilons = [0.03,0.1,0.3]
        learning_rate = 0.1
        backup = 'q'
        n = 5
        temp = 1.0
        Plot = LearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax exploration')    
        Plot.set_ylim(-100, 100) 
        for epsilon in epsilons:        
            learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                  gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
            Plot.add_curve(timesteps,learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))    
        
        policy = 'softmax'
        temps = [0.01,0.1,1.0]
        epsilon = 0.05 # placeholder
        for temp in temps:
            learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                  gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
            Plot.add_curve(timesteps,learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))
        Plot.add_hline(optimal_episode_return, label="DP optimum")
        Plot.save('exploration.png')
        print("Saved exploration.png")
            
    if args.run_exp3:
        print(f"--- Running Assignment 3: Q-learning versus SARSA ({n_repetitions} repetitions) ---")
        policy = 'egreedy'
        epsilon = 0.1 
        temp = 1.0
        n = 5
        learning_rates = [0.03,0.1,0.3]
        backups = ['q','sarsa']
        Plot = LearningCurvePlot(title = 'Back-up: on-policy versus off-policy')    
        Plot.set_ylim(-100, 100) 
        for backup in backups:
            for learning_rate in learning_rates:
                learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                  gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
                Plot.add_curve(timesteps,learning_curve,label=r'{}, $\alpha$ = {} '.format(backup_labels[backup],learning_rate))
        Plot.add_hline(optimal_episode_return, label="DP optimum")
        Plot.save('on_off_policy.png')
        print("Saved on_off_policy.png")
        
    if args.run_exp4:
        print(f"--- Running Assignment 4: Back-up depth ({n_repetitions} repetitions) ---")
        policy = 'egreedy'
        epsilon = 0.05 
        temp = 1.0
        learning_rate = 0.1
        backup = 'nstep'
        ns = [1,3,10]
        Plot = LearningCurvePlot(title = 'Back-up: depth')   
        Plot.set_ylim(-100, 100) 
        for n in ns:
            learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                  gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
            Plot.add_curve(timesteps,learning_curve,label=r'{}-step Q-learning'.format(n))
        
        backup = 'mc'
        n = 5 # placeholder
        learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
        Plot.add_curve(timesteps,learning_curve,label='Monte Carlo')        
        Plot.add_hline(optimal_episode_return, label="DP optimum")
        Plot.save('depth.png')
        print("Saved depth.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Tabular RL Experiments')
    parser.add_argument('--n_repetitions', type=int, default=20, help='Number of repetitions per experiment')
    parser.add_argument('--n_timesteps', type=int, default=50001, help='Total timesteps per run')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Interval between evaluations')
    parser.add_argument('--plot', action='store_true', help='Toggle environment rendering on')
    
    # Flags to run specific experiments. If none are provided, it runs all of them!
    parser.add_argument('--run_exp2', action='store_true', help='Run Assignment 2: Exploration')
    parser.add_argument('--run_exp3', action='store_true', help='Run Assignment 3: SARSA vs Q-learning')
    parser.add_argument('--run_exp4', action='store_true', help='Run Assignment 4: Backup Depth')

    args = parser.parse_args()

    # If the user just ran "python Experiments.py" with no flags, turn them all on automatically
    if not (args.run_exp2 or args.run_exp3 or args.run_exp4):
        args.run_exp2 = True
        args.run_exp3 = True
        args.run_exp4 = True

    experiment(args)