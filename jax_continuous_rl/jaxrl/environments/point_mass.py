
   
# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Point-mass domain."""

import collections
import pathlib

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()


def get_model_and_assets(xml_name):
  """Returns a tuple containing the model XML string and a dict of assets."""
  current_dir = pathlib.Path(__file__).parent.absolute()
  return (resources.GetResource(f'{current_dir}/point_mass_{xml_name}.xml'),
          common.ASSETS)

def make_env(name):
  time_limit = _DEFAULT_TIME_LIMIT 
  model = 'open'
  if name == 'stitch':
    model = 'stitch'
    task_kwargs = {}
  elif name == 'wideinit':
    task_kwargs = {'wide_s0': True}
  elif name == 'widedense':
    task_kwargs = {'wide_s0': True, 'dense_reward': True}
  elif name == 'open':
    task_kwargs = {}
  elif name == 'dense':
    task_kwargs = {'dense_reward': True}
  elif name == 'ring_of_fire':
    task_kwargs = {'ring_of_fire': True}
  elif name == 'bandit':
    time_limit = 0.05
    task_kwargs = {'wide_s0': True, 'bandit_reward': True}
  else:
    raise NotImplementedError(f'{name} is not implemented')
  
  return _make_env(time_limit=time_limit,
                     model=model, 
                     task_kwargs=task_kwargs)


def _make_env(time_limit=_DEFAULT_TIME_LIMIT, 
            random=None, 
            model="open",
            task_kwargs=None,
            environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets(model))
  task_kwargs = task_kwargs or {}
  task = PointMass(random=random, **task_kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """physics for the point_mass domain."""

  def mass_to_target(self):
    """Returns the vector from mass to target in global coordinate."""
    return (self.named.data.geom_xpos['target'] -
            self.named.data.geom_xpos['pointmass'])

  def mass_to_target_dist(self):
    """Returns the distance from mass to the target."""
    return np.linalg.norm(self.mass_to_target())


class PointMass(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self,  wide_s0=False, 
                      stochastic_transitions=False,
                      dense_reward=False,
                      ring_of_fire=False,
                      bandit_reward=False,
                      random=None):
    """Initialize an instance of `PointMass`.
    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._wide_s0 = wide_s0
    self._stochastic_transitions = stochastic_transitions
    self._dense_reward  = dense_reward
    self._ring_of_fire  =  ring_of_fire
    self._bandit_reward  =  bandit_reward
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.
       If _randomize_gains is True, the relationship between the controls and
       the joints is randomized, so that each control actuates a random linear
       combination of joints.
    Args:
      physics: An instance of `mujoco.Physics`.
    """
    if self._wide_s0:
      noise = 0.6 * np.random.uniform(size = 4) - 0.3
      init_loc = np.array([0, 0, 0, 0])
    else:
      noise = 0.02 * (np.random.uniform(size = 4) - 0.5)
      init_loc = np.array([-0.14, -0.28, 0, 0])
    
    init_state = np.clip(init_loc + noise, -0.29, 0.29)
    physics.set_state(init_state) 
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    if self._bandit_reward:
      x = physics.control()[0]
      y = physics.control()[1]
      fx = -3 * (x-0.1)**2 + np.sin(15*(x-2))
      return fx + y
    
    else:
      target_size = physics.named.model.geom_size['target', 0]
      if self._dense_reward:
        margin = 0.3
      else:
        margin = 0.0

      if  self._ring_of_fire:
        penalty = rewards.tolerance(physics.mass_to_target_dist(),
                                      bounds=(target_size, 3*target_size), 
                                      margin=0.0)
      else:
        penalty = 0.0
      
      near_target = rewards.tolerance(physics.mass_to_target_dist(),
                                      bounds=(0, target_size), margin=margin)
      control_reward = rewards.tolerance(physics.control(), margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic').mean()
      small_control = (control_reward + 4) / 5
      return near_target * small_control  - penalty