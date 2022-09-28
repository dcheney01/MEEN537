"""
Transforms Module - Contains code for to learn about rotations
and homogeneous transforms. 

Key for HW 03
"""

import numpy as np
from numpy import sin, cos, sqrt
from numpy.linalg import norm

## 2D Rotations
def rot2(th):
    """
    R = rot2(theta)
    Parameters
        theta: float or int, angle of rotation
    Returns
        R: 2 x 2 numpy array representing rotation in 2D by theta
    """
    R = np.array([[cos(th), -sin(th)],
                  [sin(th), cos(th)]])
    return clean_rotation_matrix(R)

## 3D Rotations
def rotx(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about x-axis by amount theta
    """
    R = np.array([[1, 0, 0],
                  [0, cos(th), -sin(th)],
                  [0, sin(th), cos(th)]])
    return clean_rotation_matrix(R)


def roty(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about y-axis by amount theta
    """
    R = np.array([[cos(th), 0, sin(th)],
                  [0, 1, 0],
                  [-sin(th), 0, cos(th)]])
    return clean_rotation_matrix(R)


def rotz(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about z-axis by amount theta
    """
    R = np.array([[cos(th), -sin(th), 0],
                  [sin(th), cos(th), 0],
                  [0, 0, 1]])
    return clean_rotation_matrix(R)


# inverse of rotation matrix 
def rot_inv(R):
    '''
    R = rot_inv(R)
    Parameters
        R: 2x2 or 3x3 numpy array representing a proper rotation matrix
    Returns
        R: 2x2 or 3x3 inverse of the input rotation matrix
    '''

    return R.T


def clean_rotation_matrix(R, eps=1e-12):
    '''
    This function is not required, but helps to make sure that all
    matrices we return are proper rotation matrices
    '''

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.abs(R[i, j]) < eps:
                R[i, j] = 0.
            elif np.abs(R[i, j] - 1) < eps:
                R[i, j] = 1.
    return R
 
def se3(R=np.eye(3), p=np.array([0, 0, 0])):
    """
        T = se3(R, p)
        Description:
            Given a numpy 3x3 array for R, and a 1x3 or 3x1 array for p, 
            this function constructs a 4x4 homogeneous transformation 
            matrix "T". 

        Parameters:
        R - 3x3 numpy array representing orientation, defaults to identity
        p = 3x1 numpy array representing position, defaults to [0, 0, 0]

        Returns:
        T - 4x4 numpy array
    """

    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3, 3] = p

    return T

def inv(T):
    """
        Tinv = inv(T)
        Description:
        Returns the inverse transform to T

        Parameters:
        T

        Returns:
        Tinv - 4x4 numpy array that is the inverse to T so that T @ Tinv = I
    """
    
    T_inv = np.eye(4)

    R = T[0:3,0:3]
    p = T[0:3, 3]
    R_inv = R.T
    p_inv = -R.T @ p
    T_inv[0:3, 0:3] = R_inv
    T_inv[0:3, 3] = p_inv

    return T_inv

