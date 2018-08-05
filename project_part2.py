""" This code will help you create your robot, load demonstrations from an
expert robot, and let your robot imitate the demonstrations.

Look over the class functions below to figure out how!
"""
import gym
import moviepy.editor as mpy
from multiprocessing import Process, Queue
import numpy as np
import pickle

from utils import Robot
from project_part1 import CustomRobot

class ImitationRobot(CustomRobot):
  """ Robot that can imitate. """
  def __init__(self, robot_name, params=None, linear=True):
    """Initialize the robot.
    Args:
      robot_name: name of the robot (e.g. hopper, ant, walker, cheetah)
      params: dictionary of robot body parameters. if None, use defaults.
      linear: if True, use a linear policy. if False, use a neural network.
    """
    self.loss = None
    self.expert_video_clip = None
    self.demos_loaded = False

    super(ImitationRobot, self).__init__(robot_name, params, linear)

  def run(self):
    """ Runs the robot in the environment and returns a video clip.
    """
    obs = self.env.reset()
    video = []
    for t in range(self.max_timesteps):
      if t % 2 == 0:
        video.append(self.get_image())
      action = self.get_action(obs)
      obs, _, _, _ = self.env.step(action)

    center_of_mass = self.env.get_body_com("torso")[0]
    if center_of_mass > 0:
      print('Done running. Your robot went ' + '{0:.2f}'.format(center_of_mass)
              + ' meters forward.')
    else:
      print('Done running. Your robot went ' + '{0:.2f}'.format(abs(center_of_mass))
              + ' meters backward.')

    video_clip = mpy.ImageSequenceClip(video, fps=20)
    return video_clip

  def load_demonstrations(self, num_demos):
    """ Loads the specified number of expert demonstrations, and returns a video of the expert demonstration trajectories. 
        The maximum number of demonstrations that can be loaded is 40.
    """
    if num_demos > 40:
        raise ValueError('Specified number of demos must be at most 40.')
    self.num_demos = num_demos
    with open('experts/' + self.robot_name + '_demos.pkl', 'rb') as f:
      self.expert_data = pickle.load(f)
    self.expert_data['observations'] = self.expert_data['observations'][:num_demos*self.max_timesteps]
    self.expert_data['actions'] = self.expert_data['actions'][:num_demos*self.max_timesteps]
    self.expert_video_clip = mpy.ImageSequenceClip(self.expert_data['video'], fps=20)
    self.expert_data['video'] = None
    self.demos_loaded = True
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


