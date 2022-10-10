import os

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from jaxrl import agents
from jaxrl.datasets import make_env_and_dataset
from jaxrl.evaluation import evaluate
from jaxrl.rvs_evaluation import evaluate as rvs_evaluate

from utils import Logger, generate_name

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'bc', 'Algorithm name.')
flags.DEFINE_string('env_name', 'point_mass-dense-offset', 'Environment name.')

flags.DEFINE_enum('dataset_name', 'mine', ['d4rl', 'awac', 'rl_unplugged',
                                            'mine'],
                  'Dataset name.')
flags.DEFINE_string('save_dir', 'logs', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 100,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of training steps.')

flags.DEFINE_float('discount', 1.0, 'Discount factor for RvS')
flags.DEFINE_boolean('avg_returns', False, 'use avg returns to go')
flags.DEFINE_float('eval_outcome', 1.0, 'Fraction of max return in the data to condition on for eval')

flags.DEFINE_float(
    'percentile', 100.0,
    'Dataset percentile (see https://arxiv.org/abs/2106.01345).')
flags.DEFINE_float('percentage', 100.0,
                   'Pencentage of the dataset to use for training.')

flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('save_model', False, 'Save model after training.')
flags.DEFINE_float('train_frac', 0.9,
                   'Fraction of the dataset to use for training.')
config_flags.DEFINE_config_file(
    'config',
    'configs/bc_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    job_name = generate_name(FLAGS.flag_values_dict())
    save_dir = os.path.join(FLAGS.save_dir, 'logs', job_name)

    flags_dict = FLAGS.flag_values_dict()
    flags_dict.update(dict(FLAGS.config))
    logger = Logger(flags_dict, save_dir, FLAGS.seed)

    video_save_folder = None if not FLAGS.save_video else os.path.join(
        save_dir, 'video', 'eval')

    if FLAGS.name in ['rvs', 'gen_rvs', 'joint_rvs']:
        rvs_alg = True
    else:
        rvs_alg = False
    env, dataset = make_env_and_dataset(rvs_alg, 
                                        FLAGS.env_name, FLAGS.seed,
                                        FLAGS.dataset_name,
                                        use_avg_returns=FLAGS.avg_returns,
                                        use_max_length=True,
                                        discount=FLAGS.discount,
                                        transform_reward=False,
                                        video_save_folder=video_save_folder)
        
    if not rvs_alg:
        if FLAGS.percentage < 100.0:
            dataset.take_random(FLAGS.percentage)

        if FLAGS.percentile < 100.0:
            dataset.take_top(FLAGS.percentile)

    dataset, val_dataset = dataset.train_validation_split(FLAGS.train_frac)

    if rvs_alg:
        return_spread = np.max(dataset.outcomes) - np.min(dataset.outcomes)
        eval_outcome = return_spread * FLAGS.eval_outcome + np.min(dataset.outcomes)
        print('Eval outcome: ', eval_outcome)

    kwargs = dict(FLAGS.config)
    if FLAGS.name == 'bc':
        kwargs['num_steps'] = FLAGS.max_steps
        agent = agents.BCLearner(FLAGS.seed,
                      env.observation_space.sample()[np.newaxis],
                      env.action_space.sample()[np.newaxis], **kwargs)
    elif FLAGS.name == 'ddpg':
        agent = agents.DDPGLearner(FLAGS.seed,
                      env.observation_space.sample()[np.newaxis],
                      env.action_space.sample()[np.newaxis], **kwargs)
    elif FLAGS.name == 'iql':
        agent = agents.IQLLearner(FLAGS.seed,
                      env.observation_space.sample()[np.newaxis],
                      env.action_space.sample()[np.newaxis], **kwargs)
    elif FLAGS.name == 'rvs':
        agent = agents.RvsLearner(FLAGS.seed,
                      env.observation_space.sample()[np.newaxis],
                      np.array([[0.0]]),
                      env.action_space.sample()[np.newaxis], **kwargs)
    elif FLAGS.name == 'gen_rvs':
        agent = agents.GenRvsLearner(FLAGS.seed,
                      env.observation_space.sample()[np.newaxis],
                      env.action_space.sample()[np.newaxis], **kwargs)
    elif FLAGS.name == 'joint_rvs':
        agent = agents.JointRvsLearner(FLAGS.seed,
                      env.observation_space.sample()[np.newaxis],
                      env.action_space.sample()[np.newaxis], 
                      n_bins=11, 
                      v_min=np.min(dataset.outcomes),
                      v_max=np.max(dataset.outcomes),
                      **kwargs)
    else:
        raise NotImplementedError



    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.write(update_info, 'train', i)

        if i % FLAGS.eval_interval == 0:

            if rvs_alg:
                if FLAGS.name == 'rvs' or FLAGS.name == 'joint_rvs':
                    if  FLAGS.avg_returns:
                        subtract = False
                    else:
                        subtract = True
                elif FLAGS.name == 'gen_rvs':
                    subtract = False

                eval_stats = rvs_evaluate(agent, env, 
                                        eval_outcome,
                                        FLAGS.eval_episodes,
                                        subtract)
            else:
                eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            eval_batch = val_dataset.sample(FLAGS.batch_size)
            eval_stats.update(agent.eval(eval_batch))

            logger.write(eval_stats, 'eval', i)
    
    logger.close()
    if FLAGS.save_model:
        save_dir = os.path.join(FLAGS.save_dir, 'models', 
                                job_name, str(FLAGS.seed), 'actor')
        agent.actor.save(save_dir)

        try: 
            save_dir = os.path.join(FLAGS.save_dir, 'models', 
                                    job_name, str(FLAGS.seed), 'critic')
            agent.critic.save(save_dir)
        except:
            pass

if __name__ == '__main__':
    app.run(main)
