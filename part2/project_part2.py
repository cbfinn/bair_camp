import gym
import moviepy.editor as mpy

from utils import Robot

# TODO - add docs.

def collect_demonstrations(robot_name):
  # TODO

def show_demonstrations(demo_trajectories):
  # TODO - get video of demo trajectories

class ImitationRobot(Robot):
  def __init__(self, robot_name):
    self.robot_name = robot_name
    self.loss = None
    if 'hopper' in robot_name.lower():
      env = gym.envs.make('Hopper-v1')
    elif 'walker' in robot_name.lower():
      env = gym.envs.make('Walker2d-v1')
    elif 'swimmer' in robot_name.lower():
      env = gym.envs.make('Swimmer-v1')
    else:
      raise ValueError('Unknown robot name')
    self.env = env
    super(ImitationRobot, self).__init__(env)  # initialize neural network with random parameters

  def run(self):
    # TODO - don't hardcode.
    obs = self.env.reset()
    video = []
    for _ in range(100):
      video.append(self.env.env.viewer.get_array())
      action = self.get_action(obs)
      obs = self.env.step(action)
    video_clip = mpy.ImageSequenceClip(video, fps=20)
    video_clip.ipython_display(width=280)

    # include env.render() to get images?
    # TODO - roll out current policy and play video
    pass

  def set_loss(self, loss_func):
    self._set_loss(loss_func)

  def train_step(self, demo_trajectories):
    if self.loss == None:
      print('Loss needs to be set before training')
      return
    demo_observations = demo_trajectories['observations']
    demo_actions = demo_trajectories['actions']
    # Call tensorflow training step.
    self._train_step(demo_observations, demo_actions)


