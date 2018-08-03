import gym
import time
from contextlib import contextmanager
import random
import tempfile
import os
import numpy as np

#from shutil import copyfile, copy2

GYM_PATH = gym.__path__[0][:-4]   #'/home/cfinn/code/gym'

class MJCModel(object):
    def __init__(self, name):
        self.name = name
        self.root = MJCTreeNode("mujoco").add_attr('model', name)

    @contextmanager
    def asfile(self):
        """
        Usage:

        model = MJCModel('reacher')
        with model.asfile() as f:
            print f.read()  # prints a dump of the model

        """
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.xml', delete=True) as f:
            self.root.write(f)
            f.seek(0)
            yield f

    def open(self):
        self.file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.xml', delete=True)
        self.root.write(self.file)
        self.file.seek(0)
        return self.file

    def save(self, path):
        with open(path, 'w') as f:
            self.root.write(f)

    def close(self):
        self.file.close()


class MJCModelRegen(MJCModel):
    def __init__(self, name, regen_fn):
        super(MJCModelRegen, self).__init__(name)
        self.regen_fn = regen_fn

    def regenerate(self):
        self.root = self.regen_fn().root



class MJCTreeNode(object):
    def __init__(self, name):
        self.name = name
        self.attrs = {}
        self.children = []

    def add_attr(self, key, value):
        if isinstance(value, str):  # should be basestring in python2
            pass
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            value = ' '.join([str(val) for val in value])

        self.attrs[key] = value
        return self

    def __getattr__(self, name):
        def wrapper(**kwargs):
            newnode =  MJCTreeNode(name)
            for (k, v) in kwargs.items(): # iteritems in python2
                newnode.add_attr(k, v)
            self.children.append(newnode)
            return newnode
        return wrapper

    def dfs(self):
        yield self
        if self.children:
            for child in self.children:
                for node in child.dfs():
                    yield node

    def write(self, ostream, tabs=0):
        contents = ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        if self.children:

            ostream.write('\t'*tabs)
            ostream.write('<%s %s>\n' % (self.name, contents))
            for child in self.children:
                child.write(ostream, tabs=tabs+1)
            ostream.write('\t'*tabs)
            ostream.write('</%s>\n' % self.name)
        else:
            ostream.write('\t'*tabs)
            ostream.write('<%s %s/>\n' % (self.name, contents))

    def __str__(self):
        s = "<"+self.name
        s += ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        return s+">"


def hopper(xml_name='hopper.xml', torso_rgb=(0.8, 0.6, .4), leg_rgb=(.7, .3, .6), limb_width=1.0, torso=1.0, thigh=1.0, leg=1.0, foot=1.0):

    torso_size = 1.25
    actual_torso_size = 0.4 * torso
    leg_size = 0.5 * leg
    thigh_size = .45 * thigh
    foot_size = 0.39 * foot
    limb_widths = 0.05 * limb_width  # the width of the head, torso, and all limbs
    torso_rgba = [torso_rgb[0], torso_rgb[1], torso_rgb[2], 1.0]
    leg_rgba = [leg_rgb[0], leg_rgb[1], leg_rgb[2], 1.0]


    mjcmodel = MJCModel('hopper')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="global")
    mjcmodel.root.option(timestep="0.002", integrator="RK4")

    asset = mjcmodel.root.asset()
    asset.texture(builtin="gradient",height="100",rgb1="1 1 1",rgb2="0 0 0",type="skybox",width="100")
    asset.texture(builtin="flat",height="1278",mark="cross",markrgb="1 1 1",name="texgeom",random="0.01",rgb1="0.8 0.6 0.4",rgb2="0.8 0.6 0.4",type="cube",width="127")
    asset.texture(builtin="checker",height="100",name="texplane",rgb1="0 0 0",rgb2="0.8 0.8 0.8",type="2d",width="100")
    asset.material(name="MatPlane",reflectance="0.5",shininess="1",specular="1",texrepeat="60 60",texture="texplane")
    asset.material(name="geom",texture="texgeom",texuniform="true")

    default = mjcmodel.root.default()
    default.joint(armature=1., damping=.1, limited='true')
    default.geom(conaffinity="1", condim="1", contype="1", margin="0.001", material="geom", rgba=leg_rgba, solref=".02 1", solimp="0.8 0.8 0.01")
    default.motor(ctrllimited="true", ctrlrange="-.4 .4")

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(cutoff="100",diffuse=[.8,.8,.8],dir="-0 0 -1.3",directional="true",exponent="1",pos="0 0 1.3",specular=".1 .1 .1")
    worldbody.geom(conaffinity=1, condim=3, material="MatPlane",name="floor",pos="0 0 0",rgba="0.8 0.9 0.8 1",size="40 40 40",type="plane")

    hopper = worldbody.body(name='torso') #, pos=[0, 0, 1.25])
    hopper.joint(armature="0", axis="1 0 0", damping="0", limited="false", name="rootx", pos="0 0 0", stiffness="0", type='slide')
    hopper.joint(armature="0", axis="0 0 1", damping="0", limited="false", name="rootz", pos="0 0 0", ref=1.25, stiffness="0", type="slide")
    hopper.joint(armature="0", axis="0 1 0", damping="0", limited="false", name="rooty", pos=[0, 0, 1.25], stiffness="0", type="hinge")
    hopper.geom(friction="0.9", fromto=[0, 0, thigh_size + leg_size + actual_torso_size + 0.1, 0, 0, thigh_size + leg_size + 0.1], name="torso_geom", size=limb_widths, type="capsule", rgba=torso_rgba)
    thigh = hopper.body(name="thigh") #, pos=[0, 0, thigh_size + leg_size + 0.1])
    thigh.joint(axis="0 -1 0", name="thigh_joint", pos=[0, 0, leg_size+thigh_size+0.1], range="-150 0", type="hinge")
    thigh.geom(friction="0.9", fromto=[0, 0, thigh_size + leg_size + 0.1, 0, 0, leg_size+0.1], name="thigh_geom", size=limb_widths, type="capsule")
    leg = thigh.body(name="leg") #, pos=[0, 0, 0.35])
    leg.joint(axis="0 -1 0", name="leg_joint", pos=[0, 0, leg_size + 0.1], range="-150 0", type="hinge")
    leg.geom(friction="0.9", fromto=[0, 0, leg_size + 0.1, 0, 0, 0.1], name="leg_geom", size=0.04/0.05*limb_widths, type="capsule")
    foot = leg.body(name="foot", pos="0.13/2 0 0.1")
    foot.joint(axis="0 -1 0", name="foot_joint", pos="0 0 0.1", range="-45 45", type="hinge")
    foot.geom(friction="2.0", fromto=[-foot_size/3.0, 0, 0.1, 2.0/3.0*foot_size, 0, 0.1], name="foot_geom", size=0.06/0.05*limb_widths, type="capsule")

    actuator = mjcmodel.root.actuator()
    gear = 200
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="thigh_joint", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="leg_joint", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="foot_joint", gear=gear)

    mjcmodel.save(GYM_PATH + '/gym/envs/mujoco/assets/' + xml_name)
    return mjcmodel

def walker(xml_name='walker2d.xml', torso_rgb=(0.8, 0.6, .4), left_leg_rgb=(.7, .3, .6), right_leg_rgb=(0.8, 0.6, .4), limb_width=1.0, torso=1.0, thighs=1.0, legs=1.0, feet=1.0):

    limb_widths = 0.05 * limb_width  # the width of the head, torso, and all limbs
    torso_size = 0.4 * torso
    leg_size = 0.5 * legs
    thigh_size = .45 * thighs
    foot_size = 0.2 * feet

    torso_rgba = [torso_rgb[0], torso_rgb[1], torso_rgb[2], 1.0]
    rgba_left = [left_leg_rgb[0], left_leg_rgb[1], left_leg_rgb[2], 1.0]
    rgba_right = [right_leg_rgb[0], right_leg_rgb[1], right_leg_rgb[2], 1.0]

    mjcmodel = MJCModel('walker2d')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="global")
    mjcmodel.root.option(timestep="0.002", integrator="RK4")

    asset = mjcmodel.root.asset()
    asset.texture(builtin="gradient",height="100",rgb1="1 1 1",rgb2="0 0 0",type="skybox",width="100")
    asset.texture(builtin="flat",height="1278",mark="cross",markrgb="1 1 1",name="texgeom",random="0.01",rgb1="0.8 0.6 0.4",rgb2="0.8 0.6 0.4",type="cube",width="127")
    asset.texture(builtin="checker",height="100",name="texplane",rgb1="0 0 0",rgb2="0.8 0.8 0.8",type="2d",width="100")
    asset.material(name="MatPlane",reflectance="0.5",shininess="1",specular="1",texrepeat="60 60",texture="texplane")
    asset.material(name="geom",texture="texgeom",texuniform="true")

    default = mjcmodel.root.default()
    default.joint(armature=.01, damping=.1, limited='true')
    default.geom(friction=[0.7,0.1,0.1], density=1000.0, margin=0.01, condim=3, contype=1, conaffinity=0, rgba=rgba_right)

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(cutoff="100",diffuse=[.8,.8,.8],dir="-0 0 -1.3",directional="true",exponent="1",pos="0 0 1.3",specular=".1 .1 .1")
    worldbody.geom(conaffinity=1, condim=3, material="MatPlane",name="floor",pos="0 0 0",rgba="0.8 0.9 0.8 1",size="40 40 40",type="plane")

    walker = worldbody.body(name='torso') #, pos=[0, 0, torso_size])
    walker.geom(name='torso_geom', friction=0.9, fromto=[0, 0, 0.1+leg_size+thigh_size+torso_size, 0, 0, 0.1+leg_size+thigh_size], size=limb_widths, type="capsule", rgba=torso_rgba)
    walker.joint(armature="0", axis=[1,0,0], stiffness="0", limited="false", name="rootx", pos=[0, 0, 0], type="slide")
    walker.joint(armature="0", axis=[0,0,1], stiffness="0", limited="false", name="rootz", pos=[0, 0, 0], type="slide", ref=1.25)
    walker.joint(armature="0", axis=[0,1,0], stiffness="0", limited="false", name="rooty", pos=[0, 0, 1.25], type="slide")

    thigh = walker.body(name="thigh") #, pos=[0, 0, 1.05])  # 1.05 = leg_size + thigh_size + 0.1
    thigh.joint(axis="0 -1 0", name="thigh_joint", pos=[0, 0, 0.1+leg_size+thigh_size], range="-150 0", type="hinge")
    thigh.geom(friction="0.9", fromto=[0, 0, 0.1+leg_size+thigh_size, 0, 0, 0.1+leg_size], name="thigh_geom", size=limb_widths, type="capsule")
    leg = thigh.body(name="leg", pos="0 0 0.35")
    leg.joint(axis="0 -1 0", name="leg_joint", pos=[0, 0, 0.1+leg_size], range="-150 0", type="hinge")
    leg.geom(friction="0.9", fromto=[0, 0, 0.1+leg_size, 0, 0, 0.1], name="leg_geom", size=0.04/0.05*limb_widths, type="capsule")
    foot = leg.body(name="foot", pos="0.2/2 0 0.1")
    foot.joint(axis="0 -1 0", name="foot_joint", pos="0 0 0.1", range="-45 45", type="hinge")
    foot.geom(friction="0.9", fromto=[-0.0, 0, 0.1, foot_size, 0, 0.1], name="foot_geom", size=0.06/0.05*limb_widths, type="capsule")

    thigh_left = walker.body(name="thigh_left") #, pos=[0, 0, 1.05])
    thigh_left.joint(axis="0 -1 0", name="thigh_joint_left", pos=[0, 0, 0.1+leg_size+thigh_size], range="-150 0", type="hinge")
    thigh_left.geom(friction="0.9", fromto=[0, 0, 0.1+leg_size+thigh_size, 0, 0, 0.1+leg_size], name="thigh_geom_left", size=limb_widths, type="capsule", rgba=rgba_left)
    leg_left = thigh_left.body(name="leg_left", pos="0 0 0.35")
    leg_left.joint(axis="0 -1 0", name="leg_joint_left", pos=[0, 0, 0.1+leg_size], range="-150 0", type="hinge")
    leg_left.geom(friction="0.9", fromto=[0, 0, 0.1+leg_size, 0, 0, 0.1], name="leg_geom_left", size=0.04/0.05*limb_widths, type="capsule", rgba=rgba_left)
    foot_left = leg_left.body(name="foot_left", pos="0.2/2 0 0.1")
    foot_left.joint(axis="0 -1 0", name="foot_joint_left", pos="0 0 0.1", range="-45 45", type="hinge")
    foot_left.geom(friction="0.9", fromto=[-0.0, 0, 0.1, foot_size, 0, 0.1], name="foot_geom_left", size=0.06/0.05*limb_widths, type="capsule", rgba=rgba_left)


    actuator = mjcmodel.root.actuator()
    gear = 100
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="thigh_joint", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="leg_joint", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="foot_joint", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="thigh_joint_left", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="leg_joint_left", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="foot_joint_left", gear=gear)

    mjcmodel.save(GYM_PATH + '/gym/envs/mujoco/assets/' + xml_name)
    return mjcmodel


def ant(xml_name='ant.xml', torso=1.0, limb_width=1.0, front_hips=1.0, back_hips=1.0, front_legs=1.0, back_legs=1.0, front_feet=1.0, back_feet=1.0, torso_rgb=(0.8, 0.6, 0.4), legs_rgb=(0.8, 0.6, 0.4)):

    torso_size = 0.25 * torso
    limb_widths = 0.08 * limb_width  # the width of the head, torso, and all limbs
    hip1 = hip4 = 0.2 * front_hips
    hip2 = hip3 = 0.2 * back_hips
    ank1 = ank4 = 0.2 * front_legs
    ank2 = ank3 = 0.2 * back_legs
    ft1 = ft4 = 0.4 * front_feet
    ft2 = ft3 = 0.4 * back_feet
    torso_rgba = [torso_rgb[0], torso_rgb[1], torso_rgb[2], 1.0]
    legs_rgba = [legs_rgb[0], legs_rgb[1], legs_rgb[2], 1.0]

    gear = 150

    mjcmodel = MJCModel('ant_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(timestep="0.01", integrator="RK4")
    mjcmodel.root.custom().numeric(data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0",name="init_qpos")
    asset = mjcmodel.root.asset()
    asset.texture(builtin="gradient",height="100",rgb1="1 1 1",rgb2="0 0 0",type="skybox",width="100")
    asset.texture(builtin="flat",height="1278",mark="cross",markrgb="1 1 1",name="texgeom",random="0.01",rgb1="0.8 0.6 0.4",rgb2="0.8 0.6 0.4",type="cube",width="127")
    asset.texture(builtin="checker",height="100",name="texplane",rgb1="0 0 0",rgb2="0.8 0.8 0.8",type="2d",width="100")
    asset.material(name="MatPlane",reflectance="0.5",shininess="1",specular="1",texrepeat="60 60",texture="texplane")
    asset.material(name="geom",texture="texgeom",texuniform="true")

    default = mjcmodel.root.default()
    default.joint(armature=1, damping=1, limited='true')
    default.geom(friction=[1.5,0.5,0.5], density=5.0, margin=0.01, condim=3, conaffinity=0, rgba=legs_rgba)

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(cutoff="100",diffuse=[.8,.8,.8],dir="-0 0 -1.3",directional="true",exponent="1",pos="0 0 1.3",specular=".1 .1 .1")
    worldbody.geom(conaffinity=1, condim=3, material="MatPlane",name="floor",pos="0 0 0",rgba="0.8 0.9 0.8 1",size="40 40 40",type="plane")

    ant = worldbody.body(name='torso', pos=[0, 0, 0.75])
    ant.geom(name='torso_geom', pos=[0, 0, 0], size=torso_size, type="sphere", rgba=torso_rgba)
    ant.joint(armature="0", damping="0", limited="false", margin="0.01", name="root", pos=[0, 0, 0], type="free")

    front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
    front_left_leg.geom(fromto=[0.0, 0.0, 0.0, hip1, hip1, 0.0], name="aux_1_geom", size=limb_widths, type="capsule")
    aux_1 = front_left_leg.body(name="aux_1", pos=[hip1, hip1, 0])
    aux_1.joint(axis=[0, 0, 1], name="hip_1", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_1.geom(fromto=[0.0, 0.0, 0.0, ank1, ank1, 0.0], name="left_leg_geom", size=limb_widths, type="capsule")
    ankle_1 = aux_1.body(pos=[ank1, ank1, 0])
    ankle_1.joint(axis=[-1, 1, 0], name="ankle_1", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_1.geom(fromto=[0.0, 0.0, 0.0, ft1, ft1, 0.0], name="left_ankle_geom", size=limb_widths, type="capsule")

    front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
    front_right_leg.geom(fromto=[0.0, 0.0, 0.0, -hip2, hip2, 0.0], name="aux_2_geom", size=limb_widths, type="capsule")
    aux_2 = front_right_leg.body(name="aux_2", pos=[-hip2, hip2, 0])
    aux_2.joint(axis=[0, 0, 1], name="hip_2", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_2.geom(fromto=[0.0, 0.0, 0.0, -ank2, ank2, 0.0], name="right_leg_geom", size=limb_widths, type="capsule")
    ankle_2 = aux_2.body(pos=[-ank2, ank2, 0])
    ankle_2.joint(axis=[1, 1, 0], name="ankle_2", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_2.geom(fromto=[0.0, 0.0, 0.0, -ft2, ft2, 0.0], name="right_ankle_geom", size=limb_widths, type="capsule")

    back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
    back_left_leg.geom(fromto=[0.0, 0.0, 0.0, -hip3, -hip3, 0.0], name="aux_3_geom", size=limb_widths, type="capsule")
    aux_3 = back_left_leg.body(name="aux_3", pos=[-hip3, -hip3, 0])
    aux_3.joint(axis=[0, 0, 1], name="hip_3", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_3.geom(fromto=[0.0, 0.0, 0.0, -ank3, -ank3, 0.0], name="backleft_leg_geom", size=limb_widths, type="capsule")
    ankle_3 = aux_3.body(pos=[-ank3, -ank3, 0])
    ankle_3.joint(axis=[-1, 1, 0], name="ankle_3", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_3.geom(fromto=[0.0, 0.0, 0.0, -ft3, -ft3, 0.0], name="backleft_ankle_geom", size=limb_widths, type="capsule")

    back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
    back_right_leg.geom(fromto=[0.0, 0.0, 0.0, hip4, -hip4, 0.0], name="aux_4_geom", size=limb_widths, type="capsule")
    aux_4 = back_right_leg.body(name="aux_4", pos=[hip4, -hip4, 0])
    aux_4.joint(axis=[0, 0, 1], name="hip_4", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_4.geom(fromto=[0.0, 0.0, 0.0, ank4, -ank4, 0.0], name="backright_leg_geom", size=limb_widths, type="capsule")
    ankle_4 = aux_4.body(pos=[ank4, -ank4, 0])
    ankle_4.joint(axis=[1, 1, 0], name="ankle_4", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_4.geom(fromto=[0.0, 0.0, 0.0, ft4, -ft4, 0.0], name="backright_ankle_geom", size=limb_widths, type="capsule")

    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3", gear=gear)

    mjcmodel.save(GYM_PATH + '/gym/envs/mujoco/assets/' + xml_name)
    return mjcmodel


def cheetah(torso=1.0, head=1.0, limb_width=1.0, back_foot=1.0, front_foot=1.0, back_thigh=1.0, front_thigh=1.0, back_shin=1.0, front_shin=1.0, body_rgb=(0.8, 0.6, 0.4), feet_rgb=(0.9, 0.6, 0.6), gravity=9.81, xml_name='half_cheetah.xml'):
    """
    Scaling factor.
    """

    mjcmodel = MJCModelRegen('half_cheetah', regen_fn=lambda: cheetah(gravity))
    mjcmodel = MJCModel('half_cheetah')

    # defaults
    torso_length = 1.0 * torso
    head_length = 0.15 * head
    bfoot_length = 0.094 * back_foot
    ffoot_length = 0.07 * front_foot
    bthigh_length = 0.145 * back_thigh
    fthigh_length = 0.133 * front_thigh
    bshin_length = 0.15 * back_shin
    fshin_length = 0.106 * front_shin
    limb_widths = 0.046 * limb_width  # the width of the head, torso, and all limbs
    total_mass = 14  # do we want this to be larger with wider limbs? (e.g. modulate with limb_widths)
    cheetah_rgba = [body_rgb[0], body_rgb[1], body_rgb[2], 1]
    feet_rgba = [feet_rgb[0], feet_rgb[1], feet_rgb[2], 1]

    root = mjcmodel.root
    root.compiler(angle="radian", coordinate="local", inertiafromgeom="true", settotalmass=total_mass)
    default = root.default()
    default.joint(armature=".1", damping=".01", limited="true", solimplimit="0 .8 .03", solreflimit=".02 1",
                  stiffness="8")
    default.geom(conaffinity="0", condim="3", contype="1", friction=".4 .1 .1", rgba=cheetah_rgba,
                 solimp="0.0 0.8 0.01", solref="0.02 1")
    default.motor(ctrllimited="true", ctrlrange="-1 1")

    root.size(nstack="300000", nuser_geom="1")
    root.option(gravity=[0,0,-gravity], timestep="0.01")
    asset = root.asset()
    asset.texture(builtin="gradient", height="100", rgb1="1 1 1", rgb2="0 0 0", type="skybox", width="100")
    asset.texture(builtin="flat", height="1278", mark="cross", markrgb="1 1 1", name="texgeom", random="0.01",
                                              rgb1="0.8 0.6 0.4", rgb2="0.8 0.6 0.4", type="cube", width="127")
    asset.texture(builtin="checker", height="100", name="texplane", rgb1="0 0 0", rgb2="0.8 0.8 0.8", type="2d",
                                                  width="100")
    asset.material(name="MatPlane", reflectance="0.5", shininess="1", specular="1", texrepeat="60 60", texture="texplane")
    asset.material(name="geom", texture="texgeom", texuniform="true")
    worldbody = root.worldbody()
    worldbody.light(cutoff="100", diffuse="1 1 1", dir="-0 0 -1.3", directional="true", exponent="1", pos="0 0 1.3",
                                            specular=".1 .1 .1")
    worldbody.geom(conaffinity="1", condim="3", material="MatPlane", name="floor", pos="0 0 0", rgba="0.8 0.9 0.8 1",
                                               size="40 40 40", type="plane")
    torso = worldbody.body(name="torso", pos="0 0 .7")
    torso.joint(armature="0", axis="1 0 0", damping="0", limited="false", name="rootx", pos="0 0 0", stiffness="0",
                            type="slide")
    torso.joint(armature="0", axis="0 0 1", damping="0", limited="false", name="rootz", pos="0 0 0", stiffness="0",
                                type="slide")
    torso.joint(armature="0", axis="0 1 0", damping="0", limited="false", name="rooty", pos="0 0 0", stiffness="0",
                                    type="hinge")
    torso.geom(fromto=[-.5, 0, 0, -.5+torso_length, 0, 0], name="torso", size=str(limb_widths), type="capsule")
    torso.geom(axisangle=[0, 1, 0, .87], name="head", pos=[-0.4+torso_length, 0, .1], size=[limb_widths, head_length], type="capsule")

    bthigh = torso.body(name="bthigh", pos="-.5 0 0")
    bthigh.joint(axis="0 1 0", damping="6", name="bthigh", pos="0 0 0", range="-.52 1.05", stiffness="240", type="hinge")
    bthigh.geom(axisangle="0 1 0 -3.8", name="bthigh", pos=[(bthigh_length+0.01)*np.sin(-3.8), 0, (bthigh_length+0.01)*np.cos(-3.8)], size=[limb_widths, bthigh_length], type="capsule")

    bshin = bthigh.body(name="bshin", pos=[2*(bthigh_length)*np.sin(-3.8), 0, 2*(bthigh_length)*np.cos(-3.8)])
    bshin.joint(axis="0 1 0", damping="4.5", name="bshin", pos="0 0 0", range="-.785 .785", stiffness="180", type="hinge")
    bshin.geom(axisangle="0 1 0 -2.03", name="bshin", pos=[(bshin_length+0.01)*np.sin(-2.03), 0, (bshin_length+0.01)*np.cos(-2.03)], rgba=feet_rgba, size=[limb_widths, bshin_length],
                                               type="capsule")
    bfoot = bshin.body(name="bfoot", pos=[2*(bshin_length)*np.sin(-2.03), 0, 2*(bshin_length)*np.cos(-2.03)])
    bfoot.joint(axis="0 1 0", damping="3", name="bfoot", pos="0 0 0", range="-.4 .785", stiffness="120", type="hinge")
    bfoot.geom(axisangle="0 1 0 -.27", name="bfoot", pos=[-(bfoot_length+0.01)*np.sin(-0.27), 0, -(bfoot_length+0.01)*np.cos(-0.27)], rgba=feet_rgba, size=[limb_widths , bfoot_length],
                                   type="capsule")

    fthigh = torso.body(name="fthigh", pos=[-.5+torso_length, 0, 0])
    fthigh.joint(axis="0 1 0", damping="4.5", name="fthigh", pos="0 0 0", range="-1 .7", stiffness="180", type="hinge")
    fthigh.geom(axisangle="0 1 0 .52", name="fthigh", pos=[-(fthigh_length+0.01)*np.sin(.52), 0, -(fthigh_length+0.01)*np.cos(.52)], size=[limb_widths, fthigh_length], type="capsule")
    fshin = fthigh.body(name="fshin", pos=[-2*(fthigh_length)*np.sin(.52), 0, -2*(fthigh_length)*np.cos(.52)])
    fshin.joint(axis="0 1 0", damping="3", name="fshin", pos="0 0 0", range="-1.2 .87", stiffness="120", type="hinge")
    fshin.geom(axisangle="0 1 0 -.6", name="fshin", pos=[-(fshin_length+0.01)*np.sin(-0.6), 0, -(fshin_length+0.01)*np.cos(-0.6)], rgba=feet_rgba, size=[limb_widths, fshin_length],
                                               type="capsule")
    ffoot = fshin.body(name="ffoot", pos=[-2*(fshin_length)*np.sin(-.6), 0, -2*(fshin_length)*np.cos(-.6)])
    ffoot.joint(axis="0 1 0", damping="1.5", name="ffoot", pos="0 0 0", range="-.5 .5", stiffness="60", type="hinge")
    ffoot.geom(axisangle="0 1 0 -.6", name="ffoot", pos=[-(ffoot_length+0.01)*np.sin(-0.6), 0, -(ffoot_length+0.01)*np.cos(-0.6)], rgba=feet_rgba, size=[limb_widths, ffoot_length],
                                                           type="capsule")

    actuator = root.actuator()
    actuator.motor(gear="120", joint="bthigh", name="bthigh")
    actuator.motor(gear="90", joint="bshin", name="bshin")
    actuator.motor(gear="60", joint="bfoot", name="bfoot")
    actuator.motor(gear="120", joint="fthigh", name="fthigh")
    actuator.motor(gear="60", joint="fshin", name="fshin")
    actuator.motor(gear="30", joint="ffoot", name="ffoot")

    mjcmodel.save(GYM_PATH + '/gym/envs/mujoco/assets/' + xml_name)
    return mjcmodel



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Could edit this to be the path to the object file instead
    parser.add_argument('--xml_filepath', type=str, default='None')
    parser.add_argument('--obj_filepath', type=str, default='None')
    args = parser.parse_args()

    # TODO - could call code to autogenerate xml file here
    #env_name = 'Pusher-v0'
    #model = pusher(mesh_file=args.obj_filepath, mesh_file_path=args.obj_filepath)
    #model.save(GYM_PATH + '/gym/envs/mujoco/assets/pusher.xml')
    #if args.obj_filepath != 'None':
    #    copy2(args.obj_filepath, GYM_PATH+'/gym/envs/mujoco/assets')

    #env_name = 'HalfCheetah-v1'
    #cheetah()

    #model = cheetah()
    #model.save(GYM_PATH + '/gym/envs/mujoco/assets/half_cheetah.xml')

    #env_name = 'Ant-v1'
    #ant()

    env_name = 'Walker2d-v1'
    walker()

    #env_name = 'Hopper-v1'
    #hopper()

    env = gym.envs.make(env_name)
    env.reset()
    for _ in range(100000):
        env.render()
        env.step(0.1*np.random.normal(size=(7,1)))
        time.sleep(0.01)
