""" This code will help you create your robot, load demonstrations from an
expert robot, and let your robot imitate the demonstrations.

Look over the class functions below to figure out how!
"""
import gym
import moviepy.editor as mpy
from multiprocessing import Process, Queue
import numpy as np
import pickle

from gym_torcs import TorcsEnv
from utils import Robot

class ImitationRobot(Robot):
  """ Robot that can imitate. """
  def __init__(self, robot_name):
    self.robot_name = robot_name
    self.loss = None
    if 'hopper' in robot_name.lower():
      env = gym.envs.make('Hopper-v1')
      self.name = 'hopper'
    elif 'walker' in robot_name.lower():
      env = gym.envs.make('Walker2d-v1')
      self.name = 'walker'
    elif 'ant' in robot_name.lower():
      env = gym.envs.make('Ant-v1')
      self.name = 'ant'
    elif 'cheetah' in robot_name.lower():
      env = gym.envs.make('HalfCheetah-v1')
      self.name = 'cheetah'
    else:
      raise ValueError('Unknown robot name')
    self.env = env
    self.max_timesteps = min(env.spec.timestep_limit, 1000)
    self.demos_loaded = False
    super(ImitationRobot, self).__init__(env)

  def run(self):
    """ Runs the robot in the environment and returns a video clip.
    """
    obs = self.env.reset()
    video = []
    for t in range(self.max_timesteps):
      video.append(self.get_image())
      action = self.get_action(obs)
      obs, _, _, _ = self.env.step(action)
    video_clip = mpy.ImageSequenceClip(video, fps=20*2)
    return video_clip


  def load_demonstrations(self, num_demos=50):
    """ Loads the specified number of expert demonstrations. """
    if num_demos > 50:
        raise ValueError('Specified number of demos must be at most 50.')
    self.num_demos = num_demos
    with open('experts/' + self.name + '_demos.pkl', 'rb') as f:
      self.expert_data = pickle.load(f)
    self.expert_data['observations'] = self.expert_data['observations'][:num_demos*self.max_timesteps]
    self.expert_data['actions'] = self.expert_data['actions'][:num_demos*self.max_timesteps]
    self.expert_video_clip = mpy.ImageSequenceClip(self.expert_data['video'], fps=20*2)
    self.expert_data['video'] = None
    self.demos_loaded = True

  def show_demonstrations(self):
    """ Returns a video of the expert demonstration trajectories. """
    return self.expert_video_clip

  def set_loss(self, loss_func):
    """ Sets the loss function for imitation. """
    self._set_loss(loss_func)

  def train_step(self):
    """ Runs one training step and returns the current error. """
    if self.loss == None:
      print('Loss needs to be set before training')
      return
    if self.demos_loaded == False:
      print('Expert demonstrations need to be loaded before training')
      return
    return self._train_step(self.expert_data['observations'], self.expert_data['actions'])

class ImitationCar(Robot):
  """ Car that can imitate. """
  def __init__(self, port=3101):
    self.loss = None
    self.name = 'car'
    self.env = TorcsEnv(vision=False, throttle=False, port=port)
    ob = self.env.reset(relaunch=False)
    obs_shape = self.process_obs(ob)
    self.max_timesteps = 1000
    self.demos_loaded = False
    super(ImitationCar, self).__init__(self.env, dim_action=1, dim_obs=2)

  def run(self, timesteps=None):
    """ Runs the car in the environment and prints how long the car drove without crashing.
    """
    if timesteps is None:
      timesteps = self.max_timesteps

    obs = self.env.reset(relaunch=False)
    print('The car is driving.')
    for i in range(timesteps):
      distance_traveled = round(obs.distRaced, 2)
      if i == 0:
        action = np.array([0.0])
      else:
        action = self.get_action(obs)
      obs, _, done, _ = self.env.step(action)
      if done:
        print('The car drove ' + str(distance_traveled) + ' feet in ' + str(i) + ' timesteps, and then crashed.')
        break
      elif i > 0 and i % 100 == 0:
        print('The car has driven ' + str(distance_traveled) + ' feet in ' + str(i) + ' timesteps.')

    if not done:
      print('Congrats!! The car drove without crashing.')
      print('The car drove ' + str(distance_traveled) + ' feet in ' + str(i) + ' timesteps.')


  def load_demonstrations(self, num_demos=10):
    """ Loads the specified number of expert demonstrations. """
    if num_demos > 10:
        raise ValueError('Specified number of demos must be at most 10.')
    self.num_demos = num_demos
    with open('experts/' + self.name + '_demos.pkl', 'rb') as f:
      self.expert_data = pickle.load(f)
    self.expert_data['observations'] = self.expert_data['observations'][:num_demos*self.max_timesteps]
    self.expert_data['actions'] = self.expert_data['actions'][:num_demos*self.max_timesteps]

    self.expert_video_clip = mpy.VideoFileClip('experts/car_demo.gif')
    self.expert_video_clip = self.expert_video_clip.cutout(0,5)
    self.demos_loaded = True

  def show_demonstrations(self):
    """ Returns a video of the expert demonstration trajectories. """
    return self.expert_video_clip

  def set_loss(self, loss_func):
    """ Sets the loss function for imitation. """
    self._set_loss(loss_func)

  def train_step(self):
    """ Runs one training step and returns the current error. """
    if self.loss == None:
      print('Loss needs to be set before training')
      return
    if self.demos_loaded == False:
      print('Expert demonstrations need to be loaded before training')
      return
    return self._train_step(self.expert_data['observations'], self.expert_data['actions'])


