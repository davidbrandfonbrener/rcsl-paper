import numpy as np

import collections
import pickle


datasets = ['point_mass-stitch-easy', 'point_mass-stitch-stitch',
                    'point_mass-open-offset', 'point_mass-dense-offset',
					'point_mass-ring_of_fire-offset',
                    'point_mass-wideinit-normal',
					'point_mass-widedense-normal']

for env_name in datasets:
	dataset =  dict(np.load(f'../../../jax_continuous_rl/data/{env_name}.npz'))

	N = dataset['rewards'].shape[0]
	data_ = collections.defaultdict(list)

	use_timeouts = False
	if 'timeouts' in dataset:
		use_timeouts = True

	episode_step = 0
	paths = []
	for i in range(N):
		done_bool = bool(dataset['terminals'][i])

		for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
			data_[k].append(dataset[k][i])
		if done_bool:  # or final_timestep:
			episode_step = 0
			episode_data = {}
			for k in data_:
				episode_data[k] = np.array(data_[k])
			paths.append(episode_data)
			data_ = collections.defaultdict(list)
		episode_step += 1

	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	with open(f'{env_name}.pkl', 'wb') as f:
		pickle.dump(paths, f)
