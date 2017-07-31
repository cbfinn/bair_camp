# BAIR Camp Jupyter Notebooks
This repository contains the Jupyter notebooks for the Python introduction ([`Intro_Python`](https://github.com/cbfinn/bair_camp/tree/master/Intro_Python)), as well as for the 2 projects we worked through ([`part1`](https://github.com/cbfinn/bair_camp/tree/master/part1) and [`part2`](https://github.com/cbfinn/bair_camp/tree/master/part2)).

If you just want to read the notebooks, you can browse through this webpage. Clicking on a notebook should load a preview in your browser, so you can read it. You won't be able to edit this preview, though.

## Downloading this repository
Click the green "Clone or Download" button, then click "Download ZIP" to download a ZIP file of the repository. Unzip this file once it is finished downloading.

## Installing Dependencies
If you want to actually play with the notebooks, you will need to install some software, including Python 2.7 and Jupyter. To install Python 2.7, we highly recommend installing [Anaconda (Python 2.7)](https://www.continuum.io/downloads). Jupyter should be included with Anaconda.

In addition to Python you will need to install some Python libraries to run some of the notebooks. Below is a list of libraries you will need for each part. If you have Anaconda, installing most packages should be as easy as typing `conda install _____` into your terminal, where the blank spaces are replaced by the package name. If you don't have Anaconda, you should type `pip install _____`. For example, to install "seaborn" you should type `conda install seaborn` (Anaconda) or `pip install seaborn` (non-Anaconda).

### Intro_Python
`Intro_Python` does not require installing any additional Python libraries.

### part1
`part1` requires installing:
* numpy
* matplotlib
* seaborn

### part2
NOTE: `part2` may be difficult to get working on Windows, compared to Mac or Linux.
`part2` requires all the `part1` libraries, as well as:
* tensorflow
* moviepy
* gym
* mujoco-py (for this you should run `pip install mujoco-py==0.5.7`)

For `part2` you will also need [Mujoco 131](http://www.mujoco.org/index.html). You can get a student license for free.
