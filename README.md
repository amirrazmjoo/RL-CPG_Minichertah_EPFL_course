# RL-CPG_Minichertah_EPFL_course

## Progress

|Observation Space|Reward|NN structure|Method-Hyper params|state|Results|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Default|Time-step dependent|Default|Default|(?)|(?)|
|GLM_paper + foot normal force + prev_action|Iteration dependent velocity (sampling around zero -> sampling around 2)|Default|Default|Done|Better learning curve (eps_length = 980 at around 500 thousand iters)|
|GLM_paper + foot normal force + prev_action|Iteration dependent velocity (sampling around one with zero variation -> sampling around 1 with 1 variation)|Default|Default|Underconstruction|The learning curve seems to be worse than the previous one, but the motion seems (visually) better. Needs more test with fixed speed|
|Default|Default|Default|andom Initialization of the robot state|(?)|(?)|
|Default|Foot slipage|Default|Default|work in progress (by alex :))|(?)|
|Default|air-time|Default|Default|Done|Converged quickly also (around 400 000 steps), difficult to say improvement, but locomotion looks good|
|Default|Smooth action (difference between actions in time-steps)|Default|Default|(?)|(?)|

## LR Course

1. Observation space

    - Body: {orientation, velocity, angular velocity}
    - Joints: {position, velocity}
    - Robot height (z)
    - CPG feedback: {$\theta$, $r$, $\phi$, $\dot{\phi}$}
    - History: {last_action}
    - Distance to goal
    - Heading to goal

2. Reward

    - Distance increment to goal: $10 * ||d_{last} - d_{current}||^2$
    - Direction to goal: $-0.01 * |\theta_{goal}|$
    - Survival reward: $0.01t$
    - velocity in z: $-||v_z||^2$
    - Roll/pitch: $-0.005*||[\omega_{\theta},\omega_{\phi}]||^2$
    - Smooth action: $-0.05*||a_t - a_{t-1}||^2$
    - Energy: $-0.001*energy*dt$

3. Curriculum learning

    - New coefficient $k_l = min(1,e^\frac{T-500000}{50000})$
    - $k_l$ multiplies following reward to make them 0 in the beginning: velocity z, roll/pitch, direction, last action
    - $(1-k_l)$ multiplies this survival to make it near zero after 500000 timesteps
