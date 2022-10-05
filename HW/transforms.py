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

def dh_transform(th, d, a, alpha):
    return np.array([[cos(th), -sin(th)*cos(alpha), sin(th)*sin(alpha),  a*cos(th)],
                     [sin(th), cos(th)*cos(alpha),  -cos(th)*sin(alpha), a*sin(th)], 
                     [0,       sin(alpha),          cos(alpha),          d], 
                     [0,       0,                   0,                   1]])


def R2rpy(R):
    """
    rpy = R2rpy(R)
    Description:
    Returns the roll-pitch-yaw representation of the SO3 rotation matrix

    Parameters:
    R - 3 x 3 Numpy array for any rotation

    Returns:
    rpy - 1 x 3 Numpy Matrix, containing <roll pitch yaw> coordinates (in radians)
    """

    roll = np.arctan2(R[1,0], R[0,0])
    pitch = np.arctan2(-R[2,0], sqrt(R[2,1]**2 + R[2,2]**2))
    yaw = np.arctan2(R[2,1], R[2,2])

    return np.array([roll, pitch, yaw])


def R2axis(R):
    """
    axis_angle = R2axis(R)
    Description:
    Returns an axis angle representation of a SO(3) rotation matrix

    Parameters:
    R

    Returns:
    axis_angle - 1 x 4 numpy matrix, containing  the axis angle representation
    in the form: <angle, rx, ry, rz>
    """

    ang = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1) / 2)
    axis_angle = np.array([ang,
                            (R[2,1] - R[1,2]) / (2*sin(ang)),
                            (R[0,2] - R[2,0]) / (2*sin(ang)),
                            (R[1,0] - R[0,1]) / (2*sin(ang))])

    return axis_angle

def axis2R(ang, v):
    """
    R = axis2R(angle, rx, ry, rz, radians=True)
    Description:
    Returns an SO3 object of the rotation specified by the axis-angle

    Parameters:
    angle - float, the angle to rotate about the axis in radians
    v = [rx, ry, rz] - components of the unit axis about which to rotate as 3x1 numpy array
    
    Returns:
    R - 3x3 numpy array
    """
    cth_1 = 1 - cos(ang)
    rx = v[0]
    ry = v[1]
    rz = v[2]
    cth = cos(ang)
    sth = sin(ang)

    R11 = rx**2 * cth_1 + cth
    R12 = rx * ry * cth_1 - rz * sth
    R13 = rx * rz * cth_1 + ry * sth

    R21 = rx * ry * cth_1 + rz * sth
    R22 = ry**2 * cth_1 + cth
    R23 = ry*rz * cth_1 - rx*sth

    R31 = rx*rz * cth_1 -ry*sth
    R32 = ry*rz*cth_1 + rx*sth
    R33 = rz**2*cth_1 + cth

    R = np.array([[R11, R12, R13],
                  [R21, R22, R23], 
                  [R31, R32, R33]])
    return R

def R2q(R):
    """
    quaternion = R2q(R)
    Description:
    Returns a quaternion representation of pose

    Parameters:
    R

    Returns:
    quaternion - 1 x 4 numpy matrix, quaternion representation of pose in the 
    format [nu, ex, ey, ez]
    """
    nu = 0.5 * sqrt(R[0,0] + R[1,1] + R[2,2] + 1)

    ex = 0.5 * np.sign(R[2,1]-R[1,2])*sqrt(R[0,0] - R[1,1] - R[2,2] + 1)
    ey = 0.5 * np.sign(R[0,2]-R[2,0])*sqrt(R[1,1] - R[2,2] - R[0,0] + 1)
    ez = 0.5 * np.sign(R[1,0]-R[0,1])*sqrt(R[2,2] - R[0,0] - R[1,1] + 1)

    return np.array([nu,
                     ex,
                     ey,
                     ez])
                    
def q2R(q):
    """
    R = q2R(q)
    Description:
    Returns a 3x3 rotation matrix

    Parameters:
    q - 4x1 numpy array, [nu, ex, ey, ez ] - defining the quaternion
    
    Returns:
    R - a 3x3 numpy array 
    """
    nu = q[0]
    ex = q[1]
    ey = q[2]
    ez = q[3]

    R11 = 2 * (nu**2 + ex**2) - 1
    R12 = 2 * (ex*ey - nu*ez)
    R13 = 2 * (ex*ez + nu*ey)

    R21 = 2 * (ex*ey + nu*ez)
    R22 = 2 * (nu**2 + ey**2) - 1
    R23 = 2 * (ey*ez - nu*ex)

    R31 = 2 * (ex*ez - nu*ey) 
    R32 = 2 * (ey*ez + nu*ex)
    R33 = 2 * (nu**2 + ez**2) - 1

    R = np.array([[R11, R12, R13],
                  [R21, R22, R23], 
                  [R31, R32, R33]])
    return R


def euler2R(th1, th2, th3, order='xyz'):
    """
    R = euler2R(th1, th2, th3, order='xyz')
    Description:
    Returns a 3x3 rotation matrix as specified by the euler angles, we assume in all cases
    that these are defined about the "current axis," which is why there are only 12 versions 
    (instead of the 24 possiblities noted in the course slides). 

    Parameters:
    th1, th2, th3 - float, angles of rotation
    order - string, specifies the euler rotation to use, for example 'xyx', 'zyz', etc.
    
    Returns:
    R - 3x3 numpy matrix
    """

    if order == 'xyx':
        R = rotx(th1) @ roty(th2) @ rotx(th3)
    elif order == 'xyz':
        R = rotx(th1) @ roty(th2) @ rotz(th3)
    elif order == 'xzx':
        R = rotx(th1) @ rotz(th2) @ rotx(th3)
    elif order == 'xzy':
        R = rotx(th1) @ rotz(th2) @ roty(th3)
    elif order == 'yxy':
        R = roty(th1) @ rotx(th2) @ roty(th3)
    elif order == 'yxz':
        R = roty(th1) @ rotx(th2) @ rotz(th3)
    elif order == 'yzx':
        R = roty(th1) @ rotz(th2) @ rotx(th3)
    elif order == 'yzy':
        R = roty(th1) @ rotz(th2) @ roty(th3)
    elif order == 'zxy':
        R = rotz(th1) @ rotx(th2) @ roty(th3)
    elif order == 'zxz':
        R = rotz(th1) @ rotx(th2) @ rotz(th3)
    elif order == 'zyx':
        R = rotz(th1) @ roty(th2) @ rotx(th3)
    elif order == 'zyz':
        R = rotz(th1) @ roty(th2) @ rotz(th3)
    else:
        print("Invalid Order!")
        return

    return R
