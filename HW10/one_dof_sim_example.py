# %% [markdown] 
# # One DOF Numerical Simulation Example

# %%
import sys
sys.path.append("C:/Users/danie/OneDrive/Documents/School/MEEN537/HW/")
import dynamics as dyn
from visualization import VizScene
from scipy.integrate import solve_ivp
import time
from matplotlib import pyplot as pl

import numpy as np
from numpy import pi

# %%
# set up model 

# defining kinematic parameters for three-link planar robot
dh = [[0, 0, 0.2, 0]]

joint_type = ['r']

link_masses = [1]

# defining one COM, but must be in a list to match expectations of code
r_coms = [np.array([-0.1, 0, 0])]

# all terms except Izz are zero because they don't matter in the 
# equations, Ixx, and Iyy are technically non-zero, we just don't 
# rotate about those axes so it doesn't matter. 
link_inertias = [np.diag([0.1, 0.1, 0.1])]

arm = dyn.SerialArmDyn(dh,
                    jt=joint_type,
                    mass=link_masses,
                    r_com=r_coms,
                    link_inertia=link_inertias)

# defining gravity as a global variable for simulation purposes
gravity = np.array([0, -9.81, 0])


# %% 
# define functions to calculate torque (or u) at each time step, and to calculate x_dot = f(x,u)
# where x = [q_dot, q] and x_dot = [q_ddot, q_dot]

# this will be our input function "u" below
def control(q, qd, q_des):
    # we will eventually define controllers like these. 

    torque = np.zeros((arm.n,))

    # calculate the applied torque for gravity compensation only right now
    # torque = arm.get_G(q, gravity)

    return torque 


# Define the dynamics function (x_dot = f(x,u)) for integration
def one_dof_eom(t,x,u,q_des):
    qd = x[0:arm.n]
    q = x[arm.n:]

    # making an empty vector for x_dot that is the right size
    x_dot = np.zeros((arm.n*2,))


    M = arm.get_M(q)
    C_q_dot = arm.get_C(q,qd)
    G = arm.get_G(q, gravity)

    #adding viscous friction 
    B = np.array([0.1])

    # calculating torque from our "controller" function u
    torque = u(q, qd, q_des)

    # solving for q_ddot by moving all other terms to right hand side and inverting M(q)
    qdd = np.linalg.inv(M) @ (torque - C_q_dot - G - B@qd)

    # filling the x_dot vector 
    x_dot[0:arm.n] = qdd
    x_dot[arm.n:] = qd

    return x_dot

# %%

# set up problem parameters 
num_steps = 1000
tf = 10.0
t = np.linspace(0, tf, num=num_steps)

# we aren't using this yet, but will in future examples/homework
q_des = np.array([0])

# this is the numerical integration performed in one function call. 
# it looks complicated but we'll talk through it in lecture. Here are brief descriptions to help:
#    fun=lambda t,x - this is just to tell the integration the state variables and time variable
#    one_dof_eom(t, x, control, q_des) - this is the f(x,u) function, it requires t, and x as inputs, but can take other inputs as shown.
#    t_span - is the length of time for the integration
#    t_eval - if defined, this specifies specific time steps where we want a solution from the integration (rather than letting the step size be determined by the algorithm)
#    y0 - is the initial condition for the integration
sol = solve_ivp(fun=lambda t, x: one_dof_eom(t, x, control, q_des), t_span=[0,tf], t_eval=t, y0=np.array([0, 0]))


# %%
# making an empty figure
fig = pl.figure()

# plotting the time vector "t" versus the solution vector for joint position
pl.plot(sol.t, sol.y[arm.n:].T)
pl.show()

# %%
# visualizing the single-link robot acting under gravity 
viz = VizScene()
time_to_run = tf
refresh_rate = num_steps/tf
viz.add_arm(arm)

qs = sol.y[arm.n:].T.flatten()

for i in range(int(refresh_rate * time_to_run)):
    viz.update(qs=[qs[i]])
    time.sleep(1.0/refresh_rate)
    
viz.hold()