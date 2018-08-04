import gym
import moviepy.editor as mpy
import numpy as np
import random
import string
import pickle

from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv

from mjc_models import cheetah, ant, hopper, walker
from utils import Robot

class CustomRobot(Robot):
  def __init__(self, robot_name):
    self.robot_name = robot_name.lower()
    self.env = None
    self.params = self.get_default_body_params()
    # change_robot_body initializes the env
    self.change_robot_body(self.params)
    self.max_timesteps = 500
    self.action_dim = self.env.action_space.shape[0]
    super(CustomRobot, self).__init__(self.env)

  def get_default_body_params(self):
    """Returns dictionary of default body parameters. Can be modified and passed to
    change_robot_body to modify the body of the robot.
    """
    params = {}
    if 'cheetah' in self.robot_name:
      # Color of body and feet, expressed as RGB intensities
      params['body_rgb'] = (0.8, 0.6, 0.4)
      params['feet_rgb'] = (0.9, 0.6, 0.6)

      # Scaling factor for different body parts.
      param_names = ['torso', 'head', 'back_foot', 'front_foot', 'back_thigh',
                     'front_thigh', 'back_shin', 'front_shin', 'limb_width']
      for key in param_names:
        params[key] = 1.0
    elif 'ant' in self.robot_name:
      # Color of body and feet, expressed as RGB intensities
      params['torso_rgb'] = (0.8, 0.6, 0.4)
      params['legs_rgb'] = (0.8, 0.6, 0.4)

      # Scaling factor for different body parts.
      param_names = ['torso', 'front_hips', 'back_hips', 'front_legs',
                     'back_legs', 'front_feet', 'back_feet', 'limb_width']
      for key in param_names:
        params[key] = 1.0
    elif 'hopper' in self.robot_name:
      # Color of body and feet, expressed as RGB intensities
      params['torso_rgb'] = (0.8, 0.6, 0.4)
      params['leg_rgb'] = (0.7, 0.3, 0.6)

      # Scaling factor for different body parts.
      param_names = ['torso', 'thigh', 'leg', 'foot', 'limb_width']
      for key in param_names:
        params[key] = 1.0
    elif 'walker' in self.robot_name:
      # Color of body and feet, expressed as RGB intensities
      params['torso_rgb'] = (0.8, 0.6, 0.4)
      params['right_leg_rgb'] = (0.8, 0.6, 0.4)
      params['left_leg_rgb'] = (0.7, 0.3, 0.6)

      # Scaling factor for different body parts.
      param_names = ['torso', 'thighs', 'legs', 'feet', 'limb_width']
      for key in param_names:
        params[key] = 1.0
    return params

  def change_robot_body(self, params):
    """
    Args:
      params: dictionary containing the parameters for changing the robot body.
    """

    # Check to make sure params are all within correct range
    for key in params.keys():
      if 'rgb' in key:
        if len(params[key]) != 3:
          print('Color parameter ' + key + ' must be list of length 3')
          return
        if any([value > 1 or value < 0 for value in params[key]]):
          print('RGB color values must be between 0 and 1.')
          return
      else:
        if params[key] < 0.5 or params[key] > 2.0:
          print('Scaling value ' + key + ' must be between 0.5 and 2.0')
          return

    if self.env:
      self.env.close()

    if 'cheetah' in self.robot_name:
      # generate random filename for xml
      xml = 'half_cheetah_' + ''.join(random.choice(string.ascii_lowercase
                                      + string.digits) for i in range(8)) + '.xml'
      cheetah(xml_name=xml, **params)
      if self.env:
        self.env.close()
      self.env = HalfCheetahEnv(xml)
    elif 'ant' in self.robot_name:
      # generate random filename for xml
      xml = 'ant_' + ''.join(random.choice(string.ascii_lowercase
                             + string.digits) for i in range(8)) + '.xml'
      ant(xml_name=xml, **params)
      if self.env:
        self.env.close()
      self.env = AntEnv(xml)
    elif 'hopper' in self.robot_name:
      # generate random filename for xml
      xml = 'hopper_' + ''.join(random.choice(string.ascii_lowercase
                                + string.digits) for i in range(8)) + '.xml'
      hopper(xml_name=xml, **params)
      self.env = HopperEnv(xml)
    elif 'walker' in self.robot_name:
      # generate random filename for xml
      xml = 'walker_' + ''.join(random.choice(string.ascii_lowercase
                                + string.digits) for i in range(8)) + '.xml'
      walker(xml_name=xml, **params)
      self.env = Walker2dEnv(xml)
    else:
      print("Illegal Robot Name. Must contain 'cheetah', 'ant', 'hopper', or 'walker'")


  def get_action_size(self):
    """ Returns the dimensionality of the robot's action space.
    """
    return self.action_dim

  def run(self, actions=None, action_durations=None):
    """ Runs the robot in the environment and returns a video clip.
    Params:
      actions:
        if actions is None, use random actions.
        if actions is a list of scalar numbers, execute the specified action continuously.
            The length of the list must be the number of joints of the robot. Get the
            number of joints by calling get_action_size()
        if actions is a list of lists and action_durations is a list,
            periodically execute each action in the list actions for the duration
            specified in action_durations. The length of actions and action_durations
            should be the same.
    """
    if actions is not None and action_durations is None:
      if len(actions) != self.action_dim:
        print('Must specify an action of size ' + str(self.action_dim))
        print('Size of passed in action is ' + str(len(actions)))
        return
      # With a constant action, only simulate for a short amount of time.
      max_timesteps = 100
    else:
      max_timesteps = self.max_timesteps

    if actions is not None and action_durations is not None:
      if len(actions) != len(action_durations):
        print('Number of actions must be the same as the number of action_durations.')
        print('len(actions) is ' + str(len(actions)))
        print('len(action_durations) is ' + str(len(action_durations)))
        return
      cur_action_id = 0
      cur_action_timestep = 0

    print('Running.')
    obs = self.env.reset()
    video = []
    for t in range(max_timesteps):
      if t % 2 == 0:
        video.append(self.get_image())
      if actions == None:
        # Sample random actions.
        action = 0.1 * np.random.normal(size=(self.action_dim, 1))
      elif actions is not None and action_durations == None:
        action = actions
      else:
        action = actions[cur_action_id]
        cur_action_timestep += 1
        if cur_action_timestep > action_durations[cur_action_id]:
          cur_action_id += 1
          cur_action_timestep = 0
        if cur_action_id >= len(actions):
          cur_action_id = 0
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

