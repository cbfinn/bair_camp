""" This code will help you create your robot, load demonstrations from an
expert robot, and let your robot imitate the demonstrations.

Look over the class functions below to figure out how!
"""
import gym
import moviepy.editor as mpy
import numpy as np
import pickle

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
    self.max_timesteps = min(env.spec.timestep_limit, 400)
    self.demos_loaded = False
    super(ImitationRobot, self).__init__(env)

  def run(self):
    """ Runs the robot in the environment and returns a video clip.
    """
    obs = self.env.reset()
    video = []
    for _ in range(self.max_timesteps):
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


