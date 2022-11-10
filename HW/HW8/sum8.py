
import sys
sys.path.append("C:/Users/danie/Documents/School/MEEN537/HW")
sys.path.append("/home/daniel/Documents/MEEN537/HW/")
import kinematics as kin
# from visualization import VizScene
import sympy as sp
import numpy as np
import dynamics as dyn

dh = [[0, 0, 0.4, 0],
      [0, 0, 0.4, 0],
      [0, 0, 0.4, 0]]

n = 3
jt_types = ['r'] * n
link_mass = [1] * n   # kg per link
r_coms = [np.array([-0.2, 0, 0])] * n
Izz = 0.01
link_inertias = [np.array([[0, 0, 0], [0, 0, 0], [0, 0, Izz]])] * n

arm = dyn.SerialArmDyn(dh,
                        jt=jt_types,
                        mass=link_mass,
                        r_com=r_coms,
                        link_inertia=link_inertias)

g = np.array([0, -9.81, 0])

q = [np.pi/4.0]*n
qd = [np.pi/6, -np.pi/4, np.pi/3]
qdd = [-np.pi/6, np.pi/3, np.pi/6]



# # Problem 3, part b)
M = np.zeros((arm.n, arm.n))

# calculating the mass matrix by iterating through RNE "n" times, and changing the location of the "1" entry in qdd
for i in range(arm.n):
    qdd = np.zeros((arm.n, ))
    qdd[i] = 1
    tau, _ = arm.rne(q, np.zeros((arm.n, )), qdd)
    M[:,i] = tau

print("generalized mass matrix is:\n", M) 


# finding C(q, q_dot)@q_dot by setting qdd = 0, (gravity is zero by default)
Cq_dot, _ = arm.rne(q, qd, np.zeros((arm.n,)))

print("Coriolis vector is:\n", Cq_dot)
