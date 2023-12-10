from typing import Optional
import argparse
import fire
import os
import sys

import ai_economist.foundation as foundation
import numpy as np
import matplotlib.pyplot as plt
import yaml
from time import time
from collections import defaultdict
import re
from simulate_utils import *
import pickle as pkl
from itertools import product
from dateutil.relativedelta import relativedelta

with open('config.yaml', "r") as f:
    run_configuration = yaml.safe_load(f)
env_config = run_configuration.get('env')
        
def gpt_actions(env, obs, dialog_queue, dialog4ref_queue, gpt_path, gpt_error, total_cost):
    # To update after acceptance
    pass
    # return actions, gpt_error, total_cost

def complex_actions(env, obs, beta=0.1, gamma=0.1, h=1):

    def consumption_len(price, wealth, curr_income, last_income, interest_rate):
        c = (price/(1e-8+wealth+curr_income))**beta
        c = min(max(c//0.02, 0), 50)
        return c
    def consumption_cats(price, wealth, curr_income, last_income, interest_rate):
        h1 = h / (1 + interest_rate)
        g = curr_income/(last_income+1e-8) - 1
        d = wealth/(last_income+1e-8) - h1
        c = 1 + (d - h1*g)/(1 + g + 1e-8)
        c = min(max(c*curr_income/(wealth+curr_income+1e-8)//0.02, 0), 50)
        return c
    def work_income_wealth(price, wealth, curr_income, last_income, expected_income, interest_rate):
        return int(np.random.uniform() < (curr_income/(wealth*(1 + interest_rate)+1e-8))**gamma)
    
    consumption_funs = [consumption_len, consumption_cats]
    work_funs = [work_income_wealth]

    actions = {}
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        price = env.world.price[-1]
        wealth = this_agent.inventory['Coin']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        max_income = max_l * this_agent.state['skill']
        last_income = this_agent.income['Coin']
        expected_income = max_l * this_agent.state['expected skill']
        interest_rate = env.world.interest_rate[-1]
        if 'consumption_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['consumption_fun_idx'] = np.random.choice(range(len(consumption_funs)))
        if 'work_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['work_fun_idx'] = np.random.choice(range(len(work_funs)))
        work_fun = work_funs[this_agent.endogenous['work_fun_idx']]
        l = work_fun(price, wealth, max_income, last_income, expected_income, interest_rate)
        curr_income = l * max_income
        consumption_fun = consumption_funs[this_agent.endogenous['consumption_fun_idx']]
        c = consumption_fun(price, wealth, curr_income, last_income, interest_rate)
        actions[str(idx)] = [l, c]
    actions['p'] = [0]
    return actions
    

def main(policy_model='complex', num_agents=100, episode_length=240, dialog_len=3, beta=0.1, gamma=0.1, h=1, max_price_inflation=0.1, max_wage_inflation=0.05):
    env_config['n_agents'] = num_agents
    env_config['episode_length'] = episode_length
    if policy_model == 'gpt':
        total_cost = 0
        env_config['flatten_masks'] = False
        env_config['flatten_observations'] = False
        env_config['components'][0]['SimpleLabor']['scale_obs'] = False
        env_config['components'][1]['PeriodicBracketTax']['scale_obs'] = False
        env_config['components'][3]['SimpleSaving']['scale_obs'] = False
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation
        
        gpt_error = 0
        from collections import deque
        dialog_queue = [deque(maxlen=dialog_len) for _ in range(env_config['n_agents'])]
        dialog4ref_queue = [deque(maxlen=7) for _ in range(env_config['n_agents'])]

    elif policy_model == 'complex':
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation

    t = time()
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    actions = {}
    if policy_model == 'complex':
        policy_model_save = f'{policy_model}-{beta}-{gamma}-{h}-{max_price_inflation}-{max_wage_inflation}'
    if policy_model == 'gpt':
        policy_model_save = f'{policy_model}-{dialog_len}-noperception-reflection-1'
    policy_model_save = f'{policy_model_save}-{num_agents}agents-{episode_length}months'
    if not os.path.exists(f'{save_path}data/{policy_model_save}'):
        os.makedirs(f'{save_path}data/{policy_model_save}')
    if not os.path.exists(f'{save_path}figs/{policy_model_save}'):
        os.makedirs(f'{save_path}figs/{policy_model_save}')
    for epi in range(env.episode_length):
        if policy_model == 'gpt':
            actions, gpt_error, total_cost = gpt_actions(env, obs, dialog_queue, dialog4ref_queue, f'{save_path}data/{policy_model_save}/dialogs', gpt_error, total_cost)
        elif policy_model == 'complex':
            actions = complex_actions(env, obs, beta=beta, gamma=gamma, h=h)
        obs, rew, done, info = env.step(actions)
        if (epi+1) % 3 == 0:
            print(f'step {epi+1} done, cost {time()-t:.1f}s')
            if policy_model == 'gpt':
                print(f'#errors: {gpt_error}, cost ${total_cost:.1f} so far')
            t = time()
        if (epi+1) % 6 == 0 or epi+1 == env.episode_length:
            with open(f'{save_path}data/{policy_model_save}/actions_{epi+1}.pkl', 'wb') as f:
                pkl.dump(actions, f)
            with open(f'{save_path}data/{policy_model_save}/obs_{epi+1}.pkl', 'wb') as f:
                pkl.dump(obs, f)
            with open(f'{save_path}data/{policy_model_save}/env_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env, f)
            if policy_model == 'gpt':
                with open(f'{save_path}data/{policy_model_save}/dialog_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog_queue, f)
                with open(f'{save_path}data/{policy_model_save}/dialog4ref_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog4ref_queue, f)
            with open(f'{save_path}data/{policy_model_save}/dense_log_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env.dense_log, f)
                
    with open(f'{save_path}data/{policy_model_save}/dense_log.pkl', 'wb') as f:
        pkl.dump(env.dense_log, f)
        
    if policy_model == 'gpt':
        print(f'#gpt errors: {gpt_error}')

if __name__ == "__main__":
    fire.Fire(main)