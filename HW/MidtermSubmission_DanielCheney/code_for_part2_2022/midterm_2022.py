# %% [markdown]
# # Midterm 2022
# * Copy this file to your homework workspace to have access to your other kinematic functions
# * Make sure to also copy a new "visualization.py" file to replace the old file (this new file can plot obstacles to scale)

# %%
# To test your setup, after defining the robot arm as described below, (but nothing else)
# you can run this file directly to make sure it is plotting the arm, obstacle, and goal 
# as expected. 

import sys
sys.path.append("/home/daniel/Documents/MEEN537/HW/")
import kinematics as kin  #this is your kinematics file that you've been developing all along
from visualization import VizScene # this is the newest visualization file updated on Oct 12
import time
import numpy as np
np.set_printoptions(precision=4)

# Define your kinematics and an "arm" variable here using DH parameters so they
# are global variables that are available in your function below:

dh = [[-np.pi/2, 4, 0, -np.pi/2.0],
      [-np.pi/6, 0, 0, -np.pi/2],
      [0, 4, 0, np.pi/2.0],
      [np.pi/6, 0, 2, np.pi/2.0]]
arm = kin.SerialArm(dh)

# let's also plot robot to make sure it matches what we think it should
# (this will look mostly like the pictures on part 1 if your DH parameters
# are correct
# viz_check = VizScene()
# viz_check.add_arm(arm, joint_colors=[np.array([0.95, 0.13, 0.13, 1])]*arm.n)
# viz_check.update(qs = [[0, 0,0,0]])
# viz_check.hold()


def compute_robot_path(q_init, goal, obst_location, obst_radius):
      def o(q, i=None):
            # returns position of the given joint
            return arm.fk(q,index=i)[:3, -1]

      def calc_error(q):
            return goal - o(q)

      def get_next_qf(q):
            # Returning the direction that we want to step in
            e = calc_error(q)
            K = np.eye(3)*.01
            kd = 0.01
            J = arm.jacob(q)[:3, :]
            J_dag = J.T @ np.linalg.inv(J @ J.T + (kd**2 * np.eye(3)))
            qdot = J_dag @ (K @ e) # SAME AS QDOT IN PINV
            return qdot
      

      def calc_del_U_att(q, qf):
            # Potential Field that describes the goal (goal position returns 0, everything else is non-zero)
            zeta = 2
            Us = []
            for i in range(1,len(q)+1):
                  U_att = (-zeta * (o(q, i) - o(qf,i)))
                  Us.append(U_att)
            return np.asarray(Us)

      def calc_min_dist(pos):
            distance = pos - obst_location
            unit_vec = distance / np.linalg.norm(distance) # get the unit vector for the direction 
            # Check if the robot hits the obstacle
            if np.any([np.linalg.norm(i) < 0 for i in np.asarray(distance - unit_vec * obst_radius)]):
                  print(np.linalg.norm(distance - unit_vec * obst_radius))
            return np.asarray(distance - unit_vec * obst_radius)

      def calc_del_U_rep(q):
            rho_0 = obst_radius # distance of influence of the obstacle (outside the radius of the obstacle itself)
            nu = 2

            Us = []
            for i in range(1, len(q)+1):
                  rho = np.linalg.norm(calc_min_dist(o(q,i)))
                  if rho_0 < rho:
                        U_rep = np.asarray([0,0,0])
                  else:
                        del_rho = calc_min_dist(o(q,i)) / np.linalg.norm(calc_min_dist(o(q,i)))
                        U_rep = (nu * (1/rho - 1/rho_0) * (1/rho**2) * del_rho)
                  Us.append(U_rep)

            return np.asarray(Us)

      count = 0
      tol = 1e-3
      q = q_init
      q_s = []

      while count < 1000 and np.linalg.norm(calc_error(q)) > tol:   
            # Find next goal configuration
            qdot_next = get_next_qf(q)
            qf_next = qdot_next + q

            # find U_att and U_rep
            U_att = calc_del_U_att(q,qf_next)
            U_rep = calc_del_U_rep(q)

            # take a step towards the goal configuration based on U
            tau = 0
            for i in range(len(q)): # find the force for each joint
                  JT = arm.jacob(q,i+1)[:3, :].T
                  U_tot = (U_rep[i] + U_att[i])

                  tau += JT @ U_tot

            # Calculate the next q
            q = q + tau
            q_s.append(q)
            
            count += 1
      return q_s

if __name__ == "__main__":

      # if your function works, this code should show the goal, the obstacle, and your robot moving towards the goal.
      q_0 = [0, 0, 0, 0]
      goal = [0, 2, 4]
      obst_position = [0, 3, 2]

      obst_rad = 1.0
      q_ik_slns = compute_robot_path(q_0, goal, obst_position, obst_rad)

      # depending on how you store q_ik_slns inside your function, you may need to change this for loop
      # definition. However if you store q as I've done above, this should work directly.
      viz = VizScene()
      viz.add_arm(arm, joint_colors=[np.array([0.95, 0.13, 0.13, 1])]*arm.n)
      viz.add_marker(goal, size=20)
      viz.add_obstacle(obst_position, rad=obst_rad)

      for q in q_ik_slns:
            viz.update(qs=[q])
            # if your step in q is very small, you can shrink this time, or remove it completely to speed up your animation
            time.sleep(0.1)
      viz.hold()

# %%
