# RL-CPG_Minichertah_EPFL_course

## Progress

|Observation Space|Reward|NN structure|Method-Hyper params|state|Results|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Default|Time-step dependent|Default|Default|(?)|(?)|
|GLM_paper + foot normal force + prev_action|Iteration dependent velocity (sampling around zero -> sampling around 2)|Default|Default|Done|Better learning curve (eps_length = 980 at around 500 thousand iters)|
|GLM_paper + foot normal force + prev_action|Iteration dependent velocity (sampling around one with zero variation -> sampling around 1 with 1 variation)|Default|Default|Underconstruction|The learning curve seems to be worse than the previous one, but the motion seems (visually) better. Needs more test with fixed speed|
|Default|Default|Default|andom Initialization of the robot state|(?)|(?)|
|Default|Foot slipage|Default|Default|(?)|(?)|
|Default|air-time|Default|Default|(?)|(?)|
|Default|Smooth action (difference between actions in time-steps)|Default|Default|(?)|(?)|