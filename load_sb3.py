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
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '121321105810'
log_dir = interm_dir + '121723032126'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {"motor_control_mode":"CPG_PSI",
               "task_env":"FLAGRUN",
               "observation_space_mode":"LR_COURSE_OBS",
               "test_env": False}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
#env_config['competition_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

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

NUM_STEPS = 1000

# [TODO] initialize arrays to save data from simulation 
cpg_states_hist = np.zeros((4, 6, NUM_STEPS))
foot_pos_hist = np.zeros((4, 3, NUM_STEPS))
joint_pos_hist = np.zeros((4, 3, NUM_STEPS))
joint_vel_hist = np.zeros((4, 3, NUM_STEPS))
body_pose_hist = np.zeros((3,NUM_STEPS))
current_body_pose_hist = []
body_vel_hist = np.zeros((3, NUM_STEPS))
body_ang_vel_hist = np.zeros((3, NUM_STEPS))
body_ort_hist = np.zeros((3,NUM_STEPS)) # Cartesian angles (heading)

for i in range(NUM_STEPS):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards

    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    # Get data from each leg
    for j in range(4):
        #J, foot_pos = env.envs[0].env.robot.ComputeJacobianAndPosition(i)
        #foot_pos_hist[j,:,i] = foot_pos
        joint_vel_hist[j,:,i] = env.envs[0].env.robot.GetMotorVelocities()[3*j:3*(j+1)]
        joint_pos_hist[j,:,i] = env.envs[0].env.robot.GetMotorAngles()[3*j:3*(j+1)]

    
    # Get body data
    body_pose_hist[:,i] = env.envs[0].env.robot.GetBasePosition()
    body_vel_hist[:,i] = env.envs[0].env.robot.GetBaseLinearVelocity()
    body_ang_vel_hist[:,i] = env.envs[0].env.robot.GetBaseAngularVelocity()
    body_ort_hist[:,i] = env.envs[0].env.robot.GetBaseOrientationRollPitchYaw()

    cpg_states_hist[:,:,i] = np.transpose(np.vstack((env.envs[0].env._cpg.get_r(), 
                                                     env.envs[0].env._cpg.get_theta(), 
                                                     env.envs[0].env._cpg.get_dr(),
                                                     env.envs[0].env._cpg.get_dtheta(),
                                                     env.envs[0].env._cpg.get_phi(),
                                                     env.envs[0].env._cpg.get_dphi())))
    #time.sleep(0.1)
# [TODO] make plots:
order_indices = [(0,1), (0,0), (1,1), (1,0)]
# Desired vs actual foot pose
fig, ax1 = plt.subplots(2,2,sharey=True)
fig.suptitle("Desired vs actual foot trajectories")
titles = ['FR', 'FL', 'HR', 'HL']
for i in range(4):
  ax1[order_indices[i]].plot(foot_pos_hist[i,0,:], foot_pos_hist[i,1,:], 'b-')
  ax1[order_indices[i]].set_title(titles[i])

ax1[0,0].legend(['actual_foot_pos'])
ax1[1,0].set_xlabel("x")
ax1[order_indices[1]].set_ylabel("z")

# CPG parameters
begin_range = 0
plot_range = 500
fig2, ax2 = plt.subplots(2,2, sharey=True)
fig2.suptitle("CPG parameters")
for i in range(4):
  ax2[order_indices[i]].plot(cpg_states_hist[i,1,begin_range:begin_range+plot_range])
  ax2[order_indices[i]].set_title(titles[i])
  ax2[order_indices[i]].plot(cpg_states_hist[i,3,begin_range:begin_range+plot_range])
  ax2y = ax2[order_indices[i]].secondary_yaxis('right', ylim=(0,2))
  ax2[order_indices[i]].plot(cpg_states_hist[i,0,begin_range:begin_range+plot_range])
  ax2[order_indices[i]].plot(cpg_states_hist[i,2,begin_range:begin_range+plot_range])

ax2[0,0].legend([r'$\theta$', r'$\dot{\theta}$', 'r', r'$\dot{r}$'])
ax2[1,0].set_xlabel("Timesteps")
ax2[0,0].set_ylabel(r'$\theta$')
ax2y = ax2[0,0].secondary_yaxis('right', ylim=(0,2))
ax2y.set_ylabel("r")

# Body info (pose, velocity, oritentation)
fig3, ax3 = plt.subplots(2,2)
fig3.suptitle('Body parameters')
ax3[0,0].plot(body_vel_hist[0,:], 'r-')
ax3[0,0].plot(body_vel_hist[1,:], 'g-')
ax3[0,0].plot(body_vel_hist[2,:], 'b-')
ax3[0,0].set_xlabel('Timestep')
ax3[0,0].set_ylabel('Velocity (m/s)')
ax3[0,0].legend([r'$v_x$', r'$v_y$', r'$v_z$'])

ax3[0,1].plot(body_ang_vel_hist[0,:], 'r-')
ax3[0,1].plot(body_ang_vel_hist[1,:], 'g-')
ax3[0,1].plot(body_ang_vel_hist[2,:], 'b-')
ax3[0,1].yaxis.tick_right()
ax3[0,1].yaxis.set_label_position('right')
ax3[0,1].set_ylabel('Angular velocity (rad/s)')
ax3[0,1].legend([r'$\dot{\phi}$', r'$\dot{\theta}$', r'$\dot{\psi}$'])

legend = []
#for i in range(len(body_pose_hist)):
#    b_pos = np.array(body_pose_hist[i])
#    ax3[1,0].plot(b_pos[0,:], b_pos[1,:])
#    legend.append(f'run_{i+1}')
ax3[1,0].plot(body_pose_hist[0,:], body_pose_hist[1,:])
ax3[1,0].set_xlabel('x')
ax3[1,0].set_ylabel('y')
ax3[1,0].legend(legend)

ax3[1,1].plot(body_ort_hist[0,:], 'r-')
ax3[1,1].plot(body_ort_hist[1,:], 'g-')
ax3[1,1].plot(body_ort_hist[2,:], 'b-')
ax3[1,1].set_ylabel('Angle (rad)')
ax3[1,1].yaxis.tick_right()
ax3[1,1].yaxis.set_label_position('right')
ax3[1,1].legend([r'$\phi$', r'$\theta$', r'$\psi$'])

# CPG phi and dphi
fig4, ax4 = plt.subplots(2,1)
fig4.suptitle(r'CPG $\phi$ and $\dot{\phi}}$ for each leg')
ax4[0].set_ylabel(r'$\phi$')
ax4[1].set_ylabel(r'$\dot{\phi}}$')
ax4[1].set_xlabel('Timesteps')
for i in range(4):
    ax4[0].plot(cpg_states_hist[i,4,begin_range:begin_range+plot_range])
    ax4[1].plot(cpg_states_hist[i,5,begin_range:begin_range+plot_range])
ax4[0].legend([f'Leg {i}' for i in range(1,5)])
ax4[1].legend([f'Leg {i}' for i in range(1,5)])

plt.show()