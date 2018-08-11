from group_params import *
from mjc_models import cheetah, ant, hopper, walker

def get_default_body_params(robot_name):
    """Returns dictionary of default body parameters. Can be modified and passed to
    change_robot_body to modify the body of the robot.
    """
    params = {}
    if 'cheetah' in robot_name:
      # Color of body and feet, expressed as RGB intensities
      params['body_rgb'] = (0.8, 0.6, 0.4)
      params['feet_rgb'] = (0.9, 0.6, 0.6)

      # Scaling factor for different body parts.
      param_names = ['torso', 'head', 'back_foot', 'front_foot', 'back_thigh',
                     'front_thigh', 'back_shin', 'front_shin', 'limb_width']
      for key in param_names:
        params[key] = 1.0
    elif 'ant' in robot_name:
      # Color of body and feet, expressed as RGB intensities
      params['torso_rgb'] = (0.8, 0.6, 0.4)
      params['legs_rgb'] = (0.8, 0.6, 0.4)

      # Scaling factor for different body parts.
      param_names = ['torso', 'front_hips', 'back_hips', 'front_legs',
                     'back_legs', 'front_feet', 'back_feet', 'limb_width']
      for key in param_names:
        params[key] = 1.0
    elif 'hopper' in robot_name:
      # Color of body and feet, expressed as RGB intensities
      params['torso_rgb'] = (0.8, 0.6, 0.4)
      params['leg_rgb'] = (0.7, 0.3, 0.6)

      # Scaling factor for different body parts.
      param_names = ['torso', 'thigh', 'leg', 'foot', 'limb_width']
      for key in param_names:
        params[key] = 1.0
    elif 'walker' in robot_name:
      # Color of body and feet, expressed as RGB intensities
      params['torso_rgb'] = (0.8, 0.6, 0.4)
      params['right_leg_rgb'] = (0.8, 0.6, 0.4)
      params['left_leg_rgb'] = (0.7, 0.3, 0.6)

      # Scaling factor for different body parts.
      param_names = ['torso', 'thighs', 'legs', 'feet', 'limb_width']
      for key in param_names:
        params[key] = 1.0
    return params



robot = g1robot1
params = g1robot1_params

if 'cheetah' in robot.lower():
  xml_name = 'half_cheetah.xml'
  cheetah(xml_name=xml_name, **params)
if 'hopper' in robot.lower():
  xml_name = 'hopper.xml'
  hopper(xml_name=xml_name, **params)
if 'walker' in robot.lower():
  xml_name = 'walker2d.xml'
  walker(xml_name=xml_name, **params)
if 'ant' in robot.lower():
  xml_name = 'ant.xml'
  ant(xml_name=xml_name, **params)

robot = g1robot2
params = g1robot2_params

if 'cheetah' in robot.lower():
  xml_name = 'half_cheetah.xml'
  cheetah(xml_name=xml_name, **params)
if 'hopper' in robot.lower():
  xml_name = 'hopper.xml'
  hopper(xml_name=xml_name, **params)
if 'walker' in robot.lower():
  xml_name = 'walker2d.xml'
  walker(xml_name=xml_name, **params)
if 'ant' in robot.lower():
  xml_name = 'ant.xml'
  ant(xml_name=xml_name, **params)


