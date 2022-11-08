
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
Izz = 0.1

link_inertias = []
for i in range(n):
      link_inertias.append(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0.01]]))

arm = dyn.SerialArmDyn(dh,
                        jt=jt_types,
                        mass=link_mass,
                        r_com=r_coms,
                        link_inertia=link_inertias)

q = [np.pi/4.0]*n
qd = [np.pi/6, -np.pi/4, np.pi/3]
qdd = [-np.pi/6, np.pi/3, np.pi/6]
arm.rne(q, qd, qdd)