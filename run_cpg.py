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

""" Run CPG """
import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv


ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
omega_stance = 12*2*np.pi
omega_swing = 18*2*np.pi
cpg = HopfNetwork(time_step=TIME_STEP, omega_stance=omega_stance, omega_swing=omega_swing)

TEST_STEPS = int(2 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
cpg_states_hist = np.zeros((4, 4, TEST_STEPS))
des_foot_pos_hist = np.zeros((4, 2, TEST_STEPS))
real_foot_pos_hist = np.zeros((4, 2, TEST_STEPS))
foot_vel_hist = np.zeros((4,3, TEST_STEPS))
des_joint_pos_hist = np.zeros((4, 3, TEST_STEPS))
real_joint_pos_hist = np.zeros((4, 3, TEST_STEPS))
real_joint_vel_hist = np.zeros((4, 3, TEST_STEPS))
normal_force_hist = np.zeros((4, TEST_STEPS))
body_vel_hist = np.zeros((3,TEST_STEPS))
foot_contact_hist = np.zeros((4,TEST_STEPS))

############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])
# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()
  # Save foot_pos
  des_foot_pos_hist[:,:,j] = np.transpose(np.vstack((xs, zs)))


  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities() 
  #print("Shape of motor angles: ", np.shape(q))
  #print("Shape of motor velocities: ", np.shape(dq))

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i,leg_xyz) # [TODO] 
    # Add joint PD contribution to tau for leg i (Equation 4)
    tau += kp*(leg_q - q[3*i:3*i+3]) + kd*(-dq[3*i:3*i+3]) # [TODO] 

    # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
    J, foot_pos = env.robot.ComputeJacobianAndPosition(i)
    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get current foot velocity in leg frame (Equation 2)
      foot_vel = J @ dq[3*i:3*i+3]
      foot_vel_hist[i,:,j] = foot_vel
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += J.T @ (kpCartesian @ (leg_xyz - foot_pos) + kdCartesian @ (-foot_vel)) # [TODO]



    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

    # Save joints pos and vel
    real_joint_pos_hist[i,:,j] = q[3*i:3*i+3]
    real_joint_vel_hist[i,:,j] = dq[3*i:3*i+3]
    des_joint_pos_hist[i,:,j] = leg_q
    # Get real pos
    real_foot_pos_hist[i,:,j] = foot_pos[[0,2]]
    body_vel_hist[:,j] = env.robot.GetBaseLinearVelocity()

    normal_force_hist[:,j] = env.robot.GetContactInfo()[2]
    foot_contact_hist[:,j] = env.robot.GetContactInfo()[3]

    #time.sleep(0.001)

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO] save any CPG or robot states
  cpg_states_hist[:,:,j] = np.transpose(np.vstack((cpg.get_r(), cpg.get_theta(), cpg.get_dr(), cpg.get_dtheta())))


##################################################### 
# PLOTS
#####################################################
order_indices = [(0,1), (0,0), (1,1), (1,0)]
# Desired vs actual foot pose
fig, ax1 = plt.subplots(2,2,sharey=True)
fig.suptitle("Desired vs actual foot trajectories")
titles = ['FR', 'FL', 'HR', 'HL']
for i in range(4):
  ax1[order_indices[i]].plot(des_foot_pos_hist[i,0,:], des_foot_pos_hist[i,1,:], 'b-')
  ax1[order_indices[i]].plot(real_foot_pos_hist[i,0,:], real_foot_pos_hist[i,1,:], 'r')
  ax1[order_indices[i]].set_title(titles[i])

ax1[0,0].legend(['des_foot_pos', 'actual_foot_pos'])
ax1[1,0].set_xlabel("x")
ax1[order_indices[1]].set_ylabel("z")

# CPG parameters
begin_range = (int)(0/TIME_STEP)
plot_range = (int)((2*np.pi/omega_stance + 2*np.pi/omega_swing)/TIME_STEP)
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


fig3, ax3 = plt.subplots(2,2, sharey=True)
fig3.suptitle("Joints desired vs actual position")
for i in range(4):
  ax3[order_indices[i]].plot(des_joint_pos_hist[i,0,begin_range:begin_range+plot_range], 'r--')
  ax3[order_indices[i]].plot(des_joint_pos_hist[i,1,begin_range:begin_range+plot_range], 'b--')
  ax3[order_indices[i]].plot(des_joint_pos_hist[i,2,begin_range:begin_range+plot_range], 'm--')
  ax3[order_indices[i]].plot(real_joint_pos_hist[i,0,begin_range:begin_range+plot_range], 'r-')
  ax3[order_indices[i]].plot(real_joint_pos_hist[i,1,begin_range:begin_range+plot_range], 'b-')
  ax3[order_indices[i]].plot(real_joint_pos_hist[i,2,begin_range:begin_range+plot_range], 'm-')

ax3[0,0].legend([r'$q_{0,des}$', r'$q_{1,des}$', r'$q_{2,des}$', 
                                r'$q_{0,act}$', r'$q_{1,act}$', r'$q_{2,act}$'])
ax3[0,0].set_xlabel('Timesteps')
ax3[0,0].set_ylabel('angle (q)')

fig4, ax4 = plt.subplots(2,2)
fig4.suptitle('Feet velocities')
for i in range(3):
  ax4[0,0].plot(foot_vel_hist[i,0,:])
  ax4[0,1].plot(foot_vel_hist[i,1,:])
  ax4[1,0].plot(foot_vel_hist[i,2,:])
ax4[0,0].legend([f'Leg {i}' for i in range(4)])

fig5, ax5 = plt.subplots(1,1)
fig5.suptitle('Foot normal contact force')
ax5.set_ylabel('Force (N)')
ax5.set_xlabel('Timesteps')
for i in range(4):
  ax5.plot(normal_force_hist[i,:])
ax5.legend([f'Foot {k}' for k in range(1,5)])

fig6, ax6 = plt.subplots(2,2, sharey=True)
fig6.suptitle('Foot vs body vel')
ax6[0,0].set_ylabel('m/s')
ax6[0,1].set_xlabel('Timesteps')
ax6[0,0].set_title('y')
ax6[0,1].set_title('x')
ax6[1,1].set_title('z')

for i in range(3):
  ax6[order_indices[i]].plot(foot_vel_hist[0,i,:])
  ax6[order_indices[i]].plot(body_vel_hist[i,:])
  ax6[order_indices[i]].plot(foot_vel_hist[0,i,:] - body_vel_hist[i,:])
  ax6[order_indices[i]].plot(foot_contact_hist[0,:])
  ax6[order_indices[i]].legend(['Foot', 'Base', 'Diff', 'Contact'])

plt.show()

