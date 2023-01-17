import numpy as np


J02 = np.array([[0,2,0,0,0,1], [0,0,-1,0,1,0]]).T
R20 = np.array([[1,0,0], [0,0,-1], [0,1,0]])

print(R20.shape)

Z = np.block([
            [R20, np.zeros((3,3))],
            [np.zeros((3,3)), R20]
        ])

print(J02)

print(Z @ J02)