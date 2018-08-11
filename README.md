### Set-up

Used the `bair_camp` branch of [this gym repo](https://github.com/cbfinn/gym/tree/bair_camp).

Installed `mujoco_py==0.5.7`.

I can't remember all of the additional python packages that I installed, but you definitely need jupyter, moviepy, and tensorflow.

### For rendering on EC2 instances

As recommended by gym README, I used xvfb for rendering.

This [thread](https://github.com/IntelVCL/Open3D/issues/17) and this [thread](https://github.com/openai/gym/issues/366) is very useful for figuring out how to get it to work.

This command can be used to test if rendering is working:
`xvfb-run -e /tmp/xvfb.err -a -s "-screen 0 1400x900x24 +extension RANDR" -- glxinfo`

The following error can be ignored:
`GLFW error: 65544, desc: X11: RandR gamma ramp support seems broken`

I think the latest version of glfw (3.3) might need to be installed. To do that, you can follow the instructions [here](https://github.com/mikeseven/node-glfw/blob/master/README.md). Though, in the future, 3.3 will probably be available more directly.


### Code for running jupyter notebook

First, set up a password for the notebooks by running `jupyter notebook password`.

More info [here](http://jupyter-notebook.readthedocs.io/en/stable/public_server.html)

For running the notebook, the following command can be used:
`xvfb-run -e /tmp/xvfb.err -a -s "-screen 0 1400x900x24 +extension RANDR" -- jupyter notebook --no-browser --port=8889 --NotebookApp.iopub_data_rate_limit=10000000000 --ip='*'`

Then connect by going to instanceurl.com:8889 and entering the password. Need to be on secure network (e.g. Airbears2) to connect.

### Commands used for Optimizing Experts

Used codebase [here](https://github.com/openai/imitation).

First ran `create_xmls.py` to change the default xml files used to be the robots designed by the students.

Ran this command, depending on the agent type:
`python scripts/run_rl_mj.py --tiny_policy --use_tanh=1 --env_name=Walker2d-v1 --log=walker.h5`
`python scripts/run_rl_mj.py --tiny_policy --use_tanh=1 --env_name=HalfCheetah-v1 --log=cheetah.h5`
`python scripts/run_rl_mj.py --tiny_policy --use_tanh=1 --env_name=Ant-v1 --log=ant.h5`
`python scripts/run_rl_mj.py --tiny_policy --use_tanh=1 --env_name=Hopper-v1 --log=hopper.h5`

For some of the ant agents, I needed to up the survival bonus (e.g. to around 2-4 instead of 1) and decrease the constraints on done (e.g. changing the upper limit to 1.5 instead of 1.0) to get it to work. It depended on the designed agent though. Some of the hoppers and walkers did not optimize well. One of the cheetahs learned a very funny gait.

### Command for Collecting Demonstrations

Once the experts are optimized, the following code will collect demonstrations.

`xvfb-run -e /tmp/xvfb.err -a -s "-screen 0 1400x900x24 +extension RANDR" -- python project_part1.py`


