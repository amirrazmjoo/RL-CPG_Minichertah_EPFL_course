# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
import sys
cur_dir = sys.path[0]
sys.path.append('/home/amir/SWITCH/')
from my_lib import ploty
my_plot = ploty()
from scipy.spatial.transform import Rotation as R
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "PPO"
interm_dir = cur_dir + "/env/logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '121321105810'
log_dir = '/home/amir/Git/RL-CPG_Minichertah_EPFL_course/env/logs/intermediate_models/CPG_RL_FWD_FULL_VEL_iter_dep_01_10_19_06/'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {"motor_control_mode":"CPG","observation_space_mode": "LR_COURSE_OBS"}
env_config['render'] = True
env_config['record_video'] = True
env_config['add_noise'] = True 
env_config['competition_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
# plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
# plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

Base_vel = []
Base_orn = []
CPG_main = []

for i in range(10000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    # episode_reward += rewards
    # Base_vel.append(info[0]['base_vel_lin'])
    # r = R.from_quat(info[0]['base_orn'])
    # Base_orn.append(r.as_euler("ZYX"))
    # CPG_main.append(info[0]['cpg_main'])
    # if dones:
    #     break

# Base_vel = np.array(Base_vel)
# fig, ax = plt.subplots()
# ax.plot(Base_vel[:,0],label = '$V_x$',linewidth = 2)
# ax.plot(Base_vel[:,1],label = '$V_y$',linewidth = 2)
# ax.plot(Base_vel[:,2],label = '$V_z$',linewidth = 2)

# ax = my_plot.plot_legend(ax,size = 32)
# ax.set_xlabel("time steps",size = 32)
# ax.set_ylabel(r"Vel. [$\frac{m}{s}$]",size = 32)
# fig.set_size_inches(16,9)
# ax.tick_params(direction='out',labelsize = 32)
# fig.savefig(cur_dir + '/FWD_adpatation_0_6_base_vel.png')

# Base_orn = np.array(Base_orn)
# fig, ax = plt.subplots()
# ax.plot(Base_orn[:,0],label = '$Z$',linewidth = 2)
# ax.plot(Base_orn[:,1],label = '$Y$',linewidth = 2)
# ax.plot(Base_orn[:,2],label = '$X$',linewidth = 2)

# ax = my_plot.plot_legend(ax,size = 32)
# ax.set_xlabel("time steps",size = 32)
# ax.set_ylabel(r"Ang. [rad]",size = 32)
# fig.set_size_inches(16,9)

# ax.tick_params(direction='out',labelsize = 32)
# fig.savefig(cur_dir + '/FWD_adpatation_0_6_base_orn.png')

# CPG_main = np.array(CPG_main)
# print(CPG_main.shape)
# fig, ax = plt.subplots()
# ax.plot(CPG_main[:,0,0],linewidth = 2)
# # ax.plot(CPG_main[:,0,1],linewidth = 2)
# # ax.plot(CPG_main[:,0,2],linewidth = 2)
# # ax.plot(CPG_main[:,0,3],linewidth = 2)

# # ax = my_plot.plot_legend(ax,size = 32)
# ax.set_xlabel("time steps",size = 32)
# ax.set_ylabel("CPG Amp. [m]",size = 32)
# fig.set_size_inches(16,9)

# ax.tick_params(direction='out',labelsize = 32)
# fig.savefig(cur_dir + '/FWD_adpatation_0_6_CPG_Amp.png')

# fig, ax = plt.subplots()
# ax.plot(CPG_main[:,1,0],linewidth = 2)

# ax.set_xlabel("time steps",size = 32)
# ax.set_ylabel("CPG Phase [rad].",size = 32)
# fig.set_size_inches(16,9)

# ax.tick_params(direction='out',labelsize = 32)
# fig.savefig(cur_dir + '/FWD_adpatation_0_6_CPG_Phase.png')


# fig, ax = plt.subplots()
# ax.plot(CPG_main[:,2,0],linewidth = 2)


# ax.set_xlabel("time steps",size = 32)
# ax.set_ylabel("CPG Steering Angle [rad].",size = 32)
# fig.set_size_inches(16,9)

# ax.tick_params(direction='out',labelsize = 32)
# fig.savefig(cur_dir + '/FWD_adpatation_0_6_CPG_Steering_angle.png')
# plt.show()