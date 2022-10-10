import gym
import numpy as np
from time import time

def evaluate(agent, env: gym.Env, eval_outcome: float,
                num_episodes: int, subtract_return=False):
    t0 = time()
    stats = {'return': [], 'length': []}
    successes = None

    eval_outcome = np.array([[eval_outcome]])

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        observation = np.expand_dims(observation, 0)
        
        cum_return = 0.0
        while not done:
            if subtract_return:
                eval_return = eval_outcome - cum_return
            else:
                eval_return = eval_outcome

            action = agent.sample_actions(observation, eval_return)
            
            observation, reward, done, info = env.step(action[0])
            
            observation = np.expand_dims(observation, 0)
            cum_return += reward
        
        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    t1 = time()
    stats['eval_time'] = t1 - t0
    return stats


def collect_traj(env, agent, eval_ret, subtract_return):
    if eval_ret is not None:
        eval_ret =  np.array([[eval_ret]])
   
    obses, actions, rewards = [], [], []
    
    observation, done = env.reset(), False
    observation = np.expand_dims(observation, 0)
    obses.append(observation)

    cum_return = 0.0
    eval_return = eval_ret
    while not done:
        if eval_return is not None:
            if subtract_return:
                eval_return = eval_ret - cum_return
            action = agent.sample_actions(observation, eval_return)
        else:
            action = agent.sample_actions(observation)
        observation, reward, done, info = env.step(action[0])

        observation = np.expand_dims(observation, 0)
        
        obses.append(observation)
        actions.append(action)
        rewards.append(reward)
        
        cum_return += reward
    
    return np.array(obses), np.array(actions), np.array(rewards)