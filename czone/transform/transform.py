import numpy as np

#Rotation
## About an arbitrary axis, through an arbitrary point
def rot_vtv(v, vt):
    """
    Vector-to-vector rotation
    Inputs:
        v: original vector
        vt: target vector into which v is rotated
    
    Outputs:
        R: 3x3 rotation matrix

    Use cross product to get axis of rotation, then develop matrix form of
    Rodrigues rotation of formula

    sources:
    https://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    """
    eps = np.finfo(float).eps #get machine epsilon for collinearity check
    theta = np.acos(np.dot(v,vt)/(np.norm(v)*np.norm(vt)))
    if(theta < eps):
        return np.identity(3)
    elif(np.pi - theta < eps):
        e_i = np.identity(3)[np.argmin(v)] #grab axis along minimum component of v
        rot_axis = np.cross(v,e_i)
    else:
        rot_axis = np.cross(v,vt)

    rot_axis /= np.norm(rot_axis)
    A = np.array([[0, -rot_axis[2], rot_axis[1]]
                  [rot_axis[2], 0, -rot_axis[0]]
                  [-rot_axis[1], rot_axis[0], 0]])

    return np.identity(3) + np.sin(theta)*A + (1.0-np.cos(theta))*A @ A



## Align axis to axis

#Translation

#Deformation