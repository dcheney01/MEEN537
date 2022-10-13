# %% [markdown] 
# # Homework 6 key
# %%
import kinematics_key_hw06 as kin
# from visualization import VizScene
import numpy as np
import time


# %% [markdown]
# # Problem 2:

# %%
dh = [[0, 0.2, 0, -np.pi/2.0],
      [0, 0, 0.2, 0],
      [np.pi/2.0, 0, 0, np.pi/2.0],
      [np.pi/2, 0.4, 0, -np.pi/2.0],
      [0, 0, 0, np.pi/2.0],
      [0, 0.4, 0, 0]]

jt_types = ['r', 'r', 'r', 'r', 'r', 'r']

# making the 6 DoF arm 
arm = kin.SerialArm(dh, jt=jt_types, tip=kin.se3(kin.roty(-np.pi/2.0)))

# defining joint angles
q = [np.pi/4.0]*6
T = arm.fk(q)


# # show the robot and a goal (just for demo's sake)
# viz = VizScene()
# viz.add_arm(arm, draw_frames=True)
# viz.add_marker(T[0:3,3], size = 20)
# viz.update(qs = [q])

# time_to_run = 1
# refresh_rate = 60

# for i in range(refresh_rate * time_to_run):
#     viz.update()
#     time.sleep(1.0/refresh_rate)


# viz.close_viz()




# %% [markdown]
# # Problem 3:

# %%
q_set1 = np.array([0]*6)
q_set2 = np.array([np.pi/4]*6)

# if this runs for too long or bogs down your computer, we can do 1 goal at a time instead
goals = [[-0.149, 0.364, 1.03],
         [-0.171, -0.682, -0.192],
         [0.822, -0.1878, 0.533],
         [-0.336, 0.095, 0.931],
         [0.335, 0.368, 0.88]]

# part a)i)
sln_pinv_q_set1 = []
sln_pinv_q_set2 = []

for goal in goals:
    qf, error_f, iter, reached_max_iter, status_msg = arm.ik_position(np.array(goal), 
                                                                      q0=q_set1, 
                                                                      method='pinv', 
                                                                      K = np.eye(3),
                                                                      max_iter = 1000)
    sln_pinv_q_set1.append(qf)

    qf, error_f, iter, reached_max_iter, status_msg = arm.ik_position(np.array(goal), 
                                                                      q0=q_set2, 
                                                                      method='pinv', 
                                                                      K = np.eye(3),
                                                                      max_iter = 1000)
    sln_pinv_q_set2.append(qf)


# %% 
# part a)ii)
sln_J_T_q_set1 = []
sln_J_T_q_set2 = []

for goal in goals:
    qf, error_f, iter, reached_max_iter, status_msg = arm.ik_position(np.array(goal), 
                                                                      q0=q_set1, 
                                                                      method='J_T', 
                                                                      K = np.eye(3),
                                                                      max_iter = 1000)
    sln_J_T_q_set1.append(qf)

    qf, error_f, iter, reached_max_iter, status_msg = arm.ik_position(np.array(goal), 
                                                                      q0=q_set2, 
                                                                      method='J_T', 
                                                                      K = np.eye(3),
                                                                      max_iter = 1000)
    sln_J_T_q_set2.append(qf)


#%%
# part a)iii)
viz = VizScene()

# this arm with joints that are darker red is for the pinv solutions
viz.add_arm(arm) 

# this arm with joints that are brighter/lighter red is for the J_T solutions
viz.add_arm(arm, joint_colors=[np.array([1.0, 51.0/255.0, 51.0/255.0, 1])]*arm.n)
viz.add_marker([0,0,0], size=20)

counter = 0

time_to_run = 15
refresh_rate = 60

for goal in goals:
    for i in range(refresh_rate * time_to_run):
        viz.update(qs=[sln_pinv_q_set1[counter], sln_J_T_q_set1[counter]], poss=[goal])
        time.sleep(1.0/refresh_rate)
    counter += 1

input('press Enter when ready to see next set starting from q_set2')

counter = 0
for goal in goals:
    for i in range(refresh_rate * time_to_run):
        viz.update(qs=[sln_pinv_q_set2[counter], sln_J_T_q_set2[counter]], poss=[goal])
        time.sleep(1.0/refresh_rate)
    counter += 1



# %%
