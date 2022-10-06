# %% [markdown] 
# # Homework 5
# * Copy the contents of the file "hw05_jacobian.py" into the SerialArm class definition in "kinematics.py".
# * Now complete the blank sections that will allow you to calculate the Geometric Jacobian for any robot based on a DH parameter description. 
# * After completion, check that your answers match the the answers for the Jacobian test below. 

# %%
import sys
sys.path.append("C:/Users/danie/Documents/School/MEEN537/HW")
sys.path.append("/home/daniel/Documents/MEEN537/HW/")
# import kinematics_key_hw05 as kin
import kinematics as kin
# from visualization import VizScene
import numpy as np
import time
np.set_printoptions(precision=4, suppress=True)


# %% [markdown]
# # Problem 1:

# Check your implementation of "jacob" using the DH parameters shown below. For the given vectors of q's, you should get the following:
# $$ q = \left[\begin{matrix}0 \\ 0 \\ 0 \\0 \end{matrix}\right]$$
# $$ J = \left[\begin{matrix}0 & 0 & 0 & 0\\0.5 & 0 & 0.1 & -1.0\\0 & 0.3 & 0 & 0\\0 & 0 & 0 & 0\\0 & -1.0 & 0 & 0\\1.0 & 0 & 1.0 & 0\end{matrix}\right]$$

# While for $$ q = \left[\begin{matrix}\pi/4 \\ \pi/4 \\ \pi/4 \\ 0.10 \end{matrix}\right]$$
# $$ J = \left[\begin{matrix}-0.3121 & -0.1707 & -0.1 & 0.8536\\0.3121 & -0.1707 & 0.1 & -0.1464\\0 & 0.2414 & -6.939 \cdot 10^{-18} & 0.5\\0 & 0.7071 & -0.5 & 0\\0 & -0.7071 & -0.5 & 0\\1.0 & 0 & 0.7071 & 0\end{matrix}\right] $$


# %%
dh = [[0, 0, 0.2, np.pi/2.0],
      [0, 0, 0.2, -np.pi/2.0],
      [0, 0, 0.1, np.pi/2.0],
      [0, 0, 0.0, 0.0]]

# An example of defining joint types which we may not have done yet. 
# The 4th joint, and 4th row of the DH parameters correspond to a prismatic joint. 
jt_types = ['r', 'r', 'r', 'p']

# notice how "jt_types" can be passed as an argument into "SerialArm"
arm = kin.SerialArm(dh, jt=jt_types)

# defining two different sets of joint angles
q_set1 = [0, 0, 0, 0]
q_set2 = [np.pi/4, np.pi/4, np.pi/4, 0.10]

# calculating two different jacobians for the two different joint configurations. 
J1 = arm.jacob(q_set1)
J2 = arm.jacob(q_set2)

print("from first set of q's, J is:")
print(J1)

print("now look at the configuration of the arm for q_set1 to understand J")

# # making a visualization
# viz = VizScene()

# # adding a SerialArm to the visualization, and telling it to draw the joint frames. 
# viz.add_arm(arm, draw_frames=True)

# # setting the joint angles to draw
# viz.update(qs=[q_set1])


# time_to_run = 30
# refresh_rate = 60

# for i in range(refresh_rate * time_to_run):
#     viz.update()
#     time.sleep(1.0/refresh_rate)


print("from second set of q's, J is:")
print(J2)

# # updating position of the arm in visualization
# viz.update(qs=[q_set2])

# print("now look at the configuration of the arm for q_set2 to understand J")
# for i in range(refresh_rate * time_to_run):
#     viz.update()
#     time.sleep(1.0/refresh_rate)

# viz.close_viz()

   


# %% [markdown]
# # Problem 2:

# %%
import sympy as sp
from sympy.physics.vector.printing import vlatex
from IPython.display import Math, display

# part a)
d1, a2 = sp.symbols('d1, a2')

dh_prob_2 = [[0, d1, 0, sp.pi/2.0],
            [0, 0, a2, 0]]

# proceeding from here, you could do this by hand, or calculate the FK with code, but the
# Jacobian by hand. This was pretty open ended, but the idea was that if you used code, you 
# recreated something simpler than in "kinematics.py", but that gave you undersanding and
# the same overall answer. 

# borrowing these rotations from HW 02 key:

def rotx_sym(th):
    R = sp.Matrix([[1, 0, 0],
                  [0, sp.cos(th), -sp.sin(th)],
                  [0, sp.sin(th), sp.cos(th)]])
    return R


def roty_sym(th):
    R = sp.Matrix([[sp.cos(th), 0, sp.sin(th)],
                   [0, 1, 0],
                   [-sp.sin(th), 0, sp.cos(th)]])
    return R


def rotz_sym(th):
    R = sp.Matrix([[sp.cos(th), -sp.sin(th), 0],
                   [sp.sin(th), sp.cos(th), 0],
                   [0, 0, 1]])
    return R

# making a symbolic SE3 function
def se3_sym(R = sp.eye(3), p = sp.Matrix([[0], [0], [0]])):
    T = sp.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    
    return T

# using previous rotation and se3 symbolic versions to 
# generate an A matrix for each joint
def get_A_sym(dh, jt_type = 'r', q=sp.Symbol('q')):
    if jt_type == 'r':
        A = se3_sym(R = rotz_sym(dh[0]+q)) @\
            se3_sym(p = sp.Matrix([[0], [0], [dh[1]]])) @\
            se3_sym(p = sp.Matrix([[dh[2]], [0], [0]])) @\
            se3_sym(R = rotx_sym(dh[3]))
    elif jt_type == 'p':
        A = se3_sym(R = rotz_sym(dh[0])) @\
            se3_sym(p = sp.Matrix([[0], [0], [dh[1]+q]])) @\
            se3_sym(p = sp.Matrix([[dh[2]], [0], [0]])) @\
            se3_sym(R = rotx_sym(dh[3]))
    else:
        A = None

    return A

# defining joint variables
q1, q2 = sp.symbols('q1 q2')

# calculating kinematics to each joint
T_01 = get_A_sym(dh_prob_2[0], 'r', q1)
T_12 = get_A_sym(dh_prob_2[1], 'r', q2)

# finding fk to robot tip
T_02 = T_01 @ T_12

# axis of rotation and position of joint 1
z0 = sp.Matrix([[0], [0], [1]])
p0 = sp.Matrix([[0], [0], [0]])

# axis of rotation and position of joint 2
z1 = T_01[0:3, 2]
p1 = T_01[0:3, 3]


# position of end effector (or point of interest)
p2 = T_02[0:3, 3]

J_hand = sp.zeros(6,2)

J_hand[0:3,0] = z0.cross(p2-p0)
J_hand[3:, 0] = z0

J_hand[0:3, 1] = z1.cross(p2-p1)
J_hand[3:, 1]  = z1


display(Math(vlatex(J_hand)))

#%%
# part b)
dh_prob_2_partb = [[0, 0.3, 0, np.pi/2.0],
                  [0, 0, 0.3, 0]]

q_test1 = [0, 0]
q_test2 = [np.pi/4.0, np.pi/4.0]

arm_prob2_partb = kin.SerialArm(dh_prob_2_partb, jt = ['r', 'r'])

J_hand1 = J_hand.subs([(a2, 0.3), (d1, 0.3), ('q1', q_test1[0]), 
                        ('q2', q_test1[1])])
J_1 = arm_prob2_partb.jacob(q_test1)
print("comparing the two matrices")
display(Math(vlatex(J_hand1)))
print(J_1)


J_hand2 = J_hand.subs([(a2, 0.3), (d1, 0.3), ('q1', sp.pi/4.0), 
                        ('q2', sp.pi/4.0)])

J_2 = arm_prob2_partb.jacob(q_test2)

print("comparing the two matrices")
display(Math(vlatex(J_hand2.evalf(4))))
print(J_2)




# %% [markdown]
# # Problem 4:

# %%

dh_stanf = [[0, 0, 0, -np.pi/2.0],
            [0, 0.154, 0, np.pi/2.0],
            [0, 0.25, 0, 0],
            [-np.pi/2.0, 0, 0, -np.pi/2.0],
            [-np.pi/2.0, 0, 0, np.pi/2.0],
            [np.pi/2.0, 0.263, 0, 0]]

jt_types = ['r', 'r', 'p', 'r', 'r', 'r']

stanf_arm = kin.SerialArm(dh_stanf, jt=jt_types, tip=kin.se3(R=kin.roty(-np.pi/2)))

q = [0]*6

# # making a visualization
# viz = VizScene()
# viz.add_arm(stanf_arm, draw_frames=True)
# viz.update(qs=[q])


# time_to_run = 30
# refresh_rate = 60

# for i in range(refresh_rate * time_to_run):
#     viz.update()
#     time.sleep(1.0/refresh_rate)



J_q_zeros = stanf_arm.jacob([0, 0, 0, 0, 0, 0])
J_q_set2 = stanf_arm.jacob([0, 0, 0.10, 0, 0, 0])

print("for q = 0, J is:")
print(J_q_zeros)

print("for q = [0, 0, 0.1, 0, 0, 0], J is:")
print(J_q_set2)

# viz.close_viz()
# %%
