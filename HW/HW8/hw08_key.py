#%% 
import kinematics_key_hw08 as kin
from dynamics_key import SerialArmDyn
import numpy as np
from numpy import pi
np.set_printoptions(precision=4,suppress=True)


# %% [markdown]
# # Problem 1:

# Given the DH parameters for a Puma 560 robot, and the offset from the robot tip to a tool frame, find the 
# jacobian at the tool, and in the tool frame. 

# %%
dh = [[0, 0,        0,      pi/2.0],
      [0, 0,        0.4318, 0], 
      [0, 0.15,     0.02,   -pi/2.0], 
      [0, 0.4318,   0,      pi/2.0],
      [0, 0,        0,      -pi/2.0],
      [0, 0.4,      0,      0]]

jt_types = ['r']*6

# making the 2 DoF arm
arm = kin.SerialArm(dh, jt=jt_types)

# defining joint angles
q = [0]*6

# from problem definition
T_tool_in_6 = kin.se3(R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), p = np.array([0, 0, 0.2]))

# using FK to get rotation from 0 to 6 
T_6_in_0 = arm.fk(q)

# finding rotation to describe transfrom to take Jacobian in frame zero, to the tool frame 
# (indices should cancel to give J a "tool" superscript)
R_0_in_6 = T_6_in_0[0:3,0:3].T
R_6_in_tool = T_tool_in_6[0:3,0:3].T

# finding the vector from frame 6 to tool, buit in the tool frame
p_from_6_to_tool_in_frame_6 = T_tool_in_6[0:3,3]

Z_0_in_6_from_6_to_6 = arm.Z_shift(R=R_0_in_6)
Z_6_in_tool_from_6_to_tool = arm.Z_shift(R=R_6_in_tool, p=p_from_6_to_tool_in_frame_6, p_frame='i')

J_at_6_in_frame_0 = arm.jacob(q)

# shifting the Jacobian, in two steps because it's easier this way.
J_at_tool_in_tool_frame = Z_6_in_tool_from_6_to_tool @ Z_0_in_6_from_6_to_6 @ J_at_6_in_frame_0

print('Shifted Jacobian is:')
print(J_at_tool_in_tool_frame)


# %% [markdown]
# # Problem 3, part a)

#%%
dh = [[0, 0, 0.4, 0],
    [0, 0, 0.4, 0],
    [0, 0, 0.4, 0]]

joint_type = ['r', 'r', 'r']

link_masses = [1, 1, 1]

# defining three different centers of mass, one for each link
r_coms = [np.array([-0.2, 0, 0]), np.array([-0.2, 0, 0]), np.array([-0.2, 0, 0])]

link_inertias = []
for i in range(len(joint_type)):
    iner = 0.01
    # this inertia tensor is only defined as having Izz non-zero
    link_inertias.append(np.array([[0, 0, 0], [0, 0, 0], [0, 0, iner]]))


arm = SerialArmDyn(dh,
                    jt=joint_type,
                    mass=link_masses,
                    r_com=r_coms,
                    link_inertia=link_inertias)

# once implemented, you can call arm.RNE and it should work. 
q = [pi/4.0]*3
qd = [pi/6.0, -pi/4.0, pi/3.0]
qdd = [-pi/6.0, pi/3.0, pi/6.0]
tau, wrenches = arm.rne(q, qd, qdd, g=np.array([0, -9.81, 0]))

print('joints torques are:\n', tau)

# %% [markdown]
# # Problem 3, part b)

M = np.zeros((arm.n, arm.n))

# calculating the mass matrix by iterating through RNE "n" times, and changing the location of the "1" entry in qdd
for i in range(arm.n):
    qdd = np.zeros((arm.n, ))
    qdd[i] = 1
    tau, _ = arm.rne(q, np.zeros((arm.n, )), qdd)
    M[:,i] = tau

print("generalized mass matrix is:\n", M) 


# %% [markdown]
# # Problem 3, part c)
# %%

# finding C(q, q_dot)@q_dot by setting qdd = 0, (gravity is zero by default)
Cq_dot, _ = arm.rne(q, qd, np.zeros((arm.n,)))

print("Coriolis vector is:\n", Cq_dot)


# %%


