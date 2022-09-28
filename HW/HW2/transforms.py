"""
Transforms Module - Contains code for to learn about rotations
and eventually homogenous transforms. 

Empty outline derived from code written by John Morrell. 
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
                  [sin(th),  cos(th)]])
    return R

## 3D Transformations
def rotx(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about x-axis by amount theta
    """
    R = np.array([[1, 0,       0],
                  [0, cos(th), -sin(th)],
                  [0, sin(th), cos(th)]])

    return R

def roty(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about y-axis by amount theta
    """
    R = np.array([[cos(th), 0,       sin(th)],
                  [0,       1,       0],
                  [-sin(th),0,       cos(th)]])

    return R

def rotz(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about z-axis by amount theta
    """
    R = np.array([[cos(th), -sin(th), 0],
                  [sin(th), cos(th),  0],
                  [0,       0,        1]])

    return R

# inverse of rotation matrix 
def rot_inv(R):
    '''
    R = rot_inv(R)
    Parameters
        R: 2x2 or 3x3 numpy array representing a proper rotation matrix
    Returns
        R: 2x2 or 3x3 inverse of the input rotation matrix
    '''
    return np.transpose(R)


def se3(R=np.eye(3), p=np.array([0, 0, 0])):
    '''
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
    '''
    p = np.reshape(p, (3,1))
    T = np.vstack([np.hstack([R,p]), [0, 0, 0, 1]])
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
    R = T[:3, :3]
    p = T[:3, 3]

    R_inv = np.transpose(R)
    p_inv = -R_inv @ p

    T_inv = se3(R_inv, p_inv)

    return T_inv