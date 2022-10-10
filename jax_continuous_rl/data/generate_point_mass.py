from re import L
from jaxrl.environments import point_mass
import numpy as np

def flat_obs(observation):
    flat = []
    for v in observation.values():
        flat.append(v)
    return np.array(flat).flatten()

def gen_traj(env, policy, initializer, n_latents, probs):
    obses, actions, rewards, next_obses, dones  = [], [], [], [], []
    
    latent = np.random.choice(n_latents, p=probs)
    
    timestep = env.reset()
    if initializer is not None:
        init_obs = initializer(latent)
        env.physics.set_state(init_obs)
        obs = init_obs
    else:
        obs = flat_obs(timestep.observation)

    while not timestep.last():
        obses.append(obs)
        action = policy(timestep, latent)
        timestep = env.step(action)
        
        actions.append(action)
        obs = flat_obs(timestep.observation)
        next_obses.append(obs)
        dones.append(float(timestep.last()))
        rewards.append(timestep.reward)

    obses = np.array(obses)
    return obses, actions, rewards, next_obses, dones

def stitch_policy(timestep, latent):
    if latent == 0:
        pos = timestep.observation['position']
        action = - pos / np.linalg.norm(pos) 
    elif latent == 1:
        action = [1.0, 0.0]
    elif latent == 2:
        action = [0.0, 1.0]
            
    noise = np.random.normal(size = 2)
    action = np.array(action) + noise
    return np.clip(action, -1, 1)

def stitch_easy_policy(timestep, latent):
    if latent in [0]:
        pos = timestep.observation['position']
        action = - pos / np.linalg.norm(pos)      
    if latent in [1,2]:
        action = [0.0, 1.0]
            
    noise = np.random.normal(size = 2)
    action = np.array(action) + noise
    return np.clip(action, -1, 1)

def stitch_initializer(latent):
    inits = np.array([[-0.28, 0, 0, 0], 
                [-0.14, -0.28, 0, 0],
                [-0.14, -0.28, 0, 0]])
    noise = 0.02 * (np.random.uniform(size = 4) - 0.5) 
    return np.clip(inits[latent] + noise, -0.29, 0.29)


def offset_normal_policy(timestep, latent):
    offset = np.array([0.13, 0.26])
    noise = np.random.normal(size = 2)
    return np.clip(offset + noise, -1, 1)

def smalloffset_normal_policy(timestep, latent):
    offset = np.array([0.07, 0.14])
    noise = np.random.normal(size = 2)
    return np.clip(offset + noise, -1, 1)

def normal_policy(timestep, latent):
    offset = np.array([0.0, 0.0])
    noise = np.random.normal(size = 2)
    return np.clip(offset + noise, -1, 1)

def uniform_policy(timestep, latent):
    offset = np.array([0.0, 0.0])
    noise = 2  * np.random.uniform(size = 2)  - 1.0
    return np.clip(offset + noise, -1, 1)


def collect_data(n_traj, env_name, data_name, seed):
    env = point_mass.make_env(env_name)
    
    initializer = None
    n_latents = 1
    probs = [1.0]

    if data_name == 'stitch':
        policy = stitch_policy
        initializer = stitch_initializer
        n_latents = 3
        probs = [0.2,  0.6, 0.2]
    elif data_name == 'easy':
        policy = stitch_easy_policy
        initializer = stitch_initializer
        n_latents = 3
        probs = [0.2,  0.6, 0.2]
    elif data_name == 'offset':
        policy = offset_normal_policy
    elif data_name == 'smalloffset':
        policy = smalloffset_normal_policy
    elif data_name == 'normal':
        policy = normal_policy
    elif data_name == 'uniform':
        policy = uniform_policy
        
    else:
        raise NotImplementedError

    np.random.seed(seed)
    dataset = {'observations':[], 'actions':[], 
                'rewards':[], 'next_observations': [], 'terminals': []}
    many_obses = []
    for _ in range(n_traj):
        obses, actions, rewards, next_obses, dones = gen_traj(env, policy, initializer, n_latents, probs)
        dataset['observations'].append(obses)
        dataset['next_observations'].append(next_obses)
        dataset['actions'].append(actions)
        dataset['rewards'].append(rewards)
        dataset['terminals'].append(dones)
        
    for k,v in dataset.items():
        dataset[k] = np.concatenate(v, axis = 0)

    np.savez(f'./point_mass-{env_name}-{data_name}', **dataset)

if __name__ == '__main__':

    n_traj = 100
    seed = 1

    env_data_pairs = [
                        ('stitch', 'stitch'), 
                        ('stitch', 'easy'),
                        ('open', 'offset'), 
                        ('wideinit', 'normal'),
                        ('widedense', 'normal'),
                        ('dense', 'offset'),
                        ('ring_of_fire', 'offset'),
                        ('bandit', 'uniform')
                    ]

    for e,d in env_data_pairs:
        collect_data(n_traj, e, d, seed)

   