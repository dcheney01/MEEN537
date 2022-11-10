"""
dynamics Module - Contains code for:
- Dynamic SerialArm class
- RNE Algorithm
- Euler - Lagrange formulation

John Morrell, Jan 28 2022
Tarnarmour@gmail.com

modified by: 
Marc Killpack, October 25, 2022
"""

import numpy as np
from kinematics import SerialArm
from utility import skew
import transforms as tr

eye = np.eye(4)


class SerialArmDyn(SerialArm):
    """
    SerialArmDyn class represents serial arms with dynamic properties and is used to calculate forces, torques, accelerations,
    joint forces, etc. using the Newton-Euler and Euler-Lagrange formulations. It inherits from the previously defined kinematic
    robot arm class "SerialArm". 
    """

    def __init__(self, 
                 dh, 
                 jt=None, 
                 base=eye, 
                 tip=eye, 
                 joint_limits=None,
                 mass=None,
                 r_com=None,
                 link_inertia=None,
                 motor_inertia=None,
                 joint_damping=None):

        SerialArm.__init__(self, dh, jt, base, tip, joint_limits)
        self.mass = mass
        self.r_com = r_com
        self.link_inertia = link_inertia
        self.motor_inertia = motor_inertia
        if joint_damping is None:
            self.B = np.zeros((self.n, self.n))
        else:
            self.B = np.diag(joint_damping)

    def rne(self, q, qd, qdd, 
            Wext=np.zeros((6,1)),
            g=np.zeros((3, )),
            omega_base=np.zeros((3, 1)),
            alpha_base=np.zeros((3, 1)),
            v_base=np.zeros((3, 1)),
            acc_base=np.zeros((3, 1))):

        """
        tau, W = RNE(q, qd, qdd):
        returns the torque in each joint (and the full wrench at each joint) given the joint configuration, velocity, and accelerations
        Args:
            q:
            qd:
            qdd:

        Returns:
            tau: torques or forces at joints (assuming revolute joints for now though)
            wrenches: force and torque at each joint, and for joint i, the wrench is in frame i


        We start with the velocity and acceleration of the base frame, v0 and a0, and the joint positions, joint velocities,
        and joint accelerations (q, qd, qdd).

        For each joint, we find the new angular velocity, w_i = w_(i-1) + z * qdot_(i-1)
        v_i = v_(i-1) + w_i x r_(i-1, com_i)


        if motor inertia is None, we don't consider it. Solve for now without motor inertia. The solution will provide code for motor inertia as well. 
        """

        omegas = []
        alphas = []
        v_ends = []
        v_coms = []
        acc_ends = []
        acc_coms = []

                # First we'll define some additional terms that we'll use in each iteration of the algorithm
        Rs = []  # List of Ri-1_i, rotation from i-1 to i in the i-1 frame
        R0s = []  # List of R0_i, rotation from 0 to i in the 0 frame
        rp2cs = []  # List of pi-1_i-1_i, vector from i-1 to i frame in frame i-1
        forces = []  # List of fi_i, force applied to link i at frame i-1, expressed w.r.t frame i
        moments = []  # List of Mi_i, moment applied to link i expressed w.r.t frame i
        rp2coms = []  # List of r_i_i-1,com, the vector from the origin of frame i-1 to the COM of link i in the i frame
        zaxes = []  # List of z axis of frame i-1, expressed in frame i

        # Lets generate all of the needed transforms now to simplify code later and save unnecessary calls to self.fk
        for i in range(self.n):
            T = self.fk(q, [i, i+1])  # Find the transform from link i to link i+1
            R = T[0:3, 0:3]
            p = T[0:3, 3]

            Rs.append(R)
            rp2cs.append(R.T @ p)
            rp2coms.append(R.T @ p + self.r_com[i])
            zaxes.append(Rs[i-1].T[0:3, 2])

            R0 = self.fk(q, i+1)[0:3, 0:3]  # Find the transform from base to link i
            R0s.append(R0)

        ## Solve for needed angular velocities, angular accelerations, and linear accelerations
        ## If helpful, you can define a function to call here so that you can debug the output more easily. 
        for i in range(0, self.n):
            if i == 0:  # If this is the first link instead of using the previous values we use the movement of the base
                w_prev = (Rs[i].T @ omega_base).flatten()
                alph_prev = (Rs[i].T @ alpha_base).flatten()
                a_prev = (Rs[i].T @ acc_base).flatten()
            else:  # Else, we just transform the values from the previous step
                w_prev = (Rs[i].T @ omegas[i-1]).flatten()
                alph_prev = (Rs[i].T @ alphas[i-1]).flatten()
                a_prev = (Rs[i].T @ acc_ends[i-1]).flatten()

            # Find kinematics of the current link
            if self.jt[i] == 'r':
                w_cur = w_prev + qd[i] * zaxes[i]
                alph_cur = alph_prev + qdd[i] * zaxes[i] + np.cross(w_cur, zaxes[i]) * qd[i]
                a_com = a_prev + np.cross(alph_cur, rp2coms[i]) + np.cross(w_cur, np.cross(w_cur, rp2coms[i]))
                a_end = a_prev + np.cross(alph_cur, rp2cs[i]) + np.cross(w_cur, np.cross(w_cur, rp2cs[i]))
            else:
                print("you need to implement kinematic equations for joint type:\t", self.jt[i])

            # Append values to our lists
            omegas.append(w_cur)
            alphas.append(alph_cur)
            acc_coms.append(a_com)
            acc_ends.append(a_end)

    ## Now solve Kinetic equations by starting with forces at last link and going backwards
        ## If helpful, you can define a function to call here so that you can debug the output more easily. 
        Wrenches = np.zeros((6, self.n,))
        tau = np.zeros((self.n,))

        for i in range(self.n - 1, -1, -1):  # Index from n-1 to 0
            if i == self.n-1:  # If we are at the last link, instead of the previous forces we use the external wrench
                Rn_0 = R0s[i].T

                # These are both positive assuming that we know the force applied to our end effector. If the
                # wrench is what our robot is applying to the world, we need to negate these or Wext.
                f_prev = (Rn_0 @ Wext[0:3]).flatten()
                M_prev = (Rn_0 @ Wext[3:]).flatten()
                g_cur = Rn_0 @ g  # Convert the gravity to the right frame
            else:  # Use the previous forces in this case
                Ri_0 = R0s[i].T
                f_prev = (Rs[i+1] @ Wrenches[0:3,i+1]).flatten()
                M_prev = (Rs[i+1] @ Wrenches[3:,i+1]).flatten()
                g_cur = Ri_0 @ g

            # Sum of forces and mass * acceleration to find forces
            # m*a = f_cur - f_prev + m*g --> f_cur = m * (a - g) + f_prev
            f_cur = f_prev + self.mass[i] * (acc_coms[i] - g_cur)

            # Using the sum of moments and d/dt(angular momentum) to find moment at joint
            # Be very careful with the r x f terms here; easy to mess up
            M_cur = self.link_inertia[i] @ alphas[i] + np.cross(omegas[i], self.link_inertia[i] @ omegas[i]) \
                        + M_prev + np.cross(self.r_com[i], -f_prev) + np.cross(rp2coms[i], f_cur)  

            Wrenches[0:3,i] = f_cur
            Wrenches[3:, i] = M_cur

        for i in range(self.n):
            if self.jt[i] == 'r':
                # this is the same as doing R_(i-1)^i @ tau_i^i and taking only the third element. 
                tau[i] = zaxes[i] @ Wrenches[3:, i] 
            else:
                print("you need to implement generalized for calculation for joint type:\t", self.jt[i])


        return tau, Wrenches



if __name__ == '__main__':

    ## this just gives an example of how to define a robot, this is a planar 3R robot.
    dh = [[0, 0, 1, 0],
          [0, 0, 1, 0],
          [0, 0, 1, 0]]

    joint_type = ['r', 'r', 'r']

    link_masses = [1, 1, 1]

    # defining three different centers of mass, one for each link
    r_coms = [np.array([-0.5, 0, 0]), np.array([-0.5, 0, 0]), np.array([-0.5, 0, 0])]

    link_inertias = []
    for i in range(len(joint_type)):
        iner = link_masses[i] / 12 * dh[i][2]**2

        # this inertia tensor is only defined as having Iyy, and Izz non-zero
        link_inertias.append(np.array([[0, 0, 0], [0, iner, 0], [0, 0, iner]]))


    arm = SerialArmDyn(dh,
                       jt=joint_type,
                       mass=link_masses,
                       r_com=r_coms,
                       link_inertia=link_inertias)

    # once implemented, you can call arm.RNE and it should work. 
    q = [np.pi/4.0]*3
    qd = [0.2]*3
    qdd = [0.05]*3
    arm.rne(q, qd, qdd)
