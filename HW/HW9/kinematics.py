"""
Kinematics Module - Contains code for:
- Forward Kinematics, from a set of DH parameters to a serial linkage arm with callable forward kinematics
- Inverse Kinematics
- Jacobian

Written by: 
John Morrell, August 24 2021
Tarnarmour@gmail.com

modified by: 
Marc Killpack, Sept 21, 2021
"""

import numpy as np
import sympy as sp
import mpmath as mp
import transforms as tr

def dh2A(dh, joint_type="r", q=sp.Symbol("q"), convention='standard', radians=True):
    """
    A = dh2A(dh, joint_type="r", q=sp.Symbol("q"), convention='standard')
    Description:
    Accepts one link of dh parameters and returns a homogeneous transform representing the transform from link i to link i+1

    Parameters:
    dh - 1 x 4 list or iterable of floats or sympy symbols, dh parameter table for one transform from link i to link i+1,
    in the order [theta d a alpha] - THIS IS NOT THE CONVENTION IN THE BOOK!!! But it is the order of operations. 
    q - sympy symbol, sympy symbol representing actuator input
    convention - string, 'standard' for standard dh convention, 'modified' for modified dh convention, 
    !!! modified not yet implemented !!!
    radians - bool, if false will assume theta and alpha are in degrees

    Returns:
    A - 4x4 sympy matrix representing the transform from one link to the next
    """
    # Convert to radians if needed
    if not radians:
        dh = [mp.radians(dh[0]), dh[1], dh[2], mp.radians(dh[3])]

    # If the joint is revolute, the actuator will change theta, while if it is prismatic it will affect d
    if joint_type == 'p':
        d = dh[1] + q
        theta = dh[0]
    else:
        d = dh[1]
        theta = dh[0] + q

    a = dh[2]
    alpha = dh[3]
    
    # See eq. (2.52), pg. 64
    A = sp.Matrix([[sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
                [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
                [0, sp.sin(alpha), sp.cos(alpha), d],
                [0, 0, 0, 1]])
    
    return A

class SerialArm:
    """
    Serial Arm - A class designed to simulate a serial linkage robotic arm
    Attributes:
    n - int, the number of links in the arm
    joint_type - list, a list of strings ('r' or 'p') representing whether each joint is prismatic or revolute (e.g. joint_type = ['r', 'r', 'p'])
    base - Sympy Matrix, Transform from the base to the first joint frame, defaults to identity
    tip - Sympy Matrix 4x4, Transform from the last joint frame to the tool tip
    is_symbolic - bool, if True indicates the arm's dh parameters have at least some symbolic elements defining the arm

    Methods:
    fk - returns the transform from one link to another
    jacob - returns the Jacobian at a given point on the arm
    jacoba - returns analytic jacobian at a given point
    ik - returns the joint coordinates to move the end effector to a specific point
    """
    def __init__(self, dh, joint_type=None, base=sp.eye(4), tip=sp.eye(4), radians=True):
        """
        arm = SerialArm(dh, joint_type=None, base=sp.eye(4), tip=sp.eye(4))
        Description:
        Constructor

        Parameters:
        dh - list or iterable of length 4 lists or iterables, each element is a list holding dh parameters for one link of the robot arm in the order [theta, d, a, alpha]
        joint_type - list of strings, each string is either 'r' for revolute joint or 'p' for prismatic joint, for each link
        base - 4x4 sympy matrix representing the transform from base to first joint
        tip - 4x4 sympy matrix representing the transfrom from last joint to tool tip
        radians - bool, if False will assume dh parameters are given in degrees when valid

        Returns:
        arm - Instance of the SerialArm class
        """
        self.n = len(dh)
        self.joint_type = joint_type
        self.is_symbolic = False
        self.dh = []
        # Check if any values in DH are symbolic, and remake as a nested list for consistency
        for i in range(self.n):
            self.dh.append([dh[i][0], dh[i][1], dh[i][2], dh[i][3]])
            for j in range(4):
                if isinstance(dh[i][j], sp.Symbol):
                    self.is_symbolic = True

        self.transforms = []
        self.base = base
        self.tip = tip
        # Check if any values in base or tip are symbolic
        for i in range(4):
            for j in range(4):
                if isinstance(self.base[i,j], sp.Symbol) or isinstance(self.tip[i,j], sp.Symbol):
                    self.is_symbolic = True
        
        if joint_type is None:
            self.joint_type = ['r'] * self.n
        
        for i in range(self.n):
            symbolic_tf = dh2A(dh[i], self.joint_type[i], q=sp.symbols("q"+str(i+1)))
            self.transforms.append(symbolic_tf)

    def fk(self, q, index=-1, base=True, tip=True):
        """
        T = arm.fk(q, index=-1, base=True, tip=True)
        Description: 
        Returns the transform from one link to another given a set of joint inputs q

        Parameters:
        q - list or iterables of sympy symbols or floats which represent the joint actuator inputs to the arm
        index - integer or list of two integers. If a list of two integers, the first integer represents the starting JOINT 
        (with 0 as the first joint and n as the last joint) and the second integer represents the ending FRAME
        If one integer is given only, then the integer represents the ending Frame and the FK is calculated as starting from 
        the first joint
        base - bool, if True then if index starts from 0 the base transform will also be included
        tool - bool, if true and if the index ends at the nth frame then the tool transform will be included
        """

        if isinstance(index, (list, tuple)):
            start_link = index[0]
            end_link = index[1]
        else:
            start_link = 0
            end_link = index
        
        if end_link == -1:
            end_link = self.n
        elif end_link > self.n:
            print("WARNING: Ending index greater than number of joints")
            return None

        if start_link < 0:
            print("WARNING: Starting index less than zero")
            return None
        
        if base and start_link == 0:
            T = self.base
            start_link = 0
        else:
            T = sp.eye(4)
        
        # For each transform, get the transform by substituting q[i] into the transforms list, then post multiply
        for i in range(start_link, end_link):
            T = T * self.transforms[i].subs({"q"+str(i+1):q[i]})
        
        if tip and end_link == self.n:
            T = T * self.tip
        
        return T
    
    def jacob(self, q, index=-1, tip=True, base=True):
        """
        J = arm.jacob(q, index=-1)
        Description: 
        Returns the geometric jacobian of the arm in a given configuration

        Parameters:
        q - list of sympy symbols or floats, joint actuator inputs
        index - integer, which joint frame to give the jacobian at

        Returns:
        J - sympy matrix 6xN, jacobian of the robot arm
        """

        if index == -1:
            index = self.n
        elif index > self.n:
            print("WARNING: Index greater than number of joints")
            return None

        J = sp.zeros(6, self.n)
        Te = self.fk(q, index, base=base, tip=tip)
        Pe = Te[0:3, 3]
        for i in range(index):
            if self.joint_type[i] == 'r':
                T = self.fk(q, i, base=base, tip=tip)
                z_axis = T[0:3, 2]
                P = T[0:3, 3]
                J[0:3, i] = z_axis.cross(Pe - P)
                J[3:6, i] = z_axis
            else:
                T = self.fk(q, i, base=base, tip=tip)
                z_axis = T[0:3, 2]
                P = T[0:3, 3]
                J[0:3, i] = z_axis
                J[3:6, i] = sp.Matrix([[0.0], [0.0], [0.0]])
        return J

    def jacob_shift(self, q, R, p, index=None):

        """"
        J_shifted = jacob_shift(q, R, p)
        Description: 
        Shifts (rotates and translates) a Jacobian from one point to a new point defined by the relative transform R and the translation p

        Parameters:
            q - Nx1 sympy matrix, represents the current joint variables
            R - 3x3 sympy matrix, the transform from the initial frame to the final frame, expressed in the initial frame (e.g. R^1_2)
            p - 3x1 sympy matrix length 3 iterable, the translation from the initial Jacobian point to the final point, expressed in the initial jacobian frame
        
        Returns:
            J_shifted - 6xN sympy matrix, the new shifted jacobian
        """

        if index==None:
            index = self.n

        # make a helper function to generate as skew symmetric matrix (this could be a class-level function instead)
        def skew(p):
            return sp.Matrix([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])

        # generate our skew matrix
        S = skew(p)

        # generate the shifting matrix assuming that:
        #     -p vector is in the final or next frame
        #     -R takes us from initial frame to final frame 
        Z = sp.Matrix([[sp.eye(3), -S], 
                        [sp.zeros(3), sp.eye(3)]]) @ sp.Matrix([[R, sp.zeros(3)],[sp.zeros(3), R]])

        # this assumes that the R and p we got were relative to the tip and zero frame. 
        J = self.jacob(q, index)

        J_shifted = Z @ J

        return J_shifted

   
if __name__ == "__main__":

    # Defining a table of DH parameters where each row corresponds to another joint.
    # The symbolic joint variables "q" do not need to be explicitly defined here. 
    # This is a two link, planar robot arm with two revolute joints. 
    dh = [[0, 0, 0.3, 0],
          [0, 0, 0.3, 0]]

    # make robot arm (assuming all joints are revolute)
    arm = SerialArm(dh)

    # defining joint configuration
    q = [3.14/4.0, 0.0]  # 45 degrees and 0 degrees

    # show an example of calculating the entire forward kinematics
    Tn_to_0 = arm.fk(q)
    sp.pprint(Tn_to_0)

    # show an example of calculating the kinematics between frames 0 and 1
    T1_to_0 = arm.fk(q, index=[0,1])
    sp.pprint(T1_to_0)
    
