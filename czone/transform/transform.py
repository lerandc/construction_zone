import numpy as np
from scipy.spatial.transform import Rotation

#Rotation
def rot_v(v, theta):
    """
    Rotate about an arbitary vector.
    Inputs:
        v: rotation axis
        theta: radians
    Outputs:
        R: 3x3 rotation matrix
    """
    v *= (theta/np.linalg.norm(v))

    return Rotation.from_rotvec(v).as_matrix()

def rot_vtv(v, vt):
    """
    Vector-to-vector rotation.
    Inputs:
        v: 1x3 original vector
        vt: 1x3 target vector into which v is rotated
    
    Outputs:
        R: 3x3 rotation matrix

    Use cross product to get axis of rotation, then develop matrix form of
    Rodrigues rotation of formula

    sources:
    https://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    """
    eps = np.finfo(float).eps #get machine epsilon for collinearity check
    theta = np.acos(np.dot(v,vt)/(np.linalg.norm(v)*np.linalg.norm(vt)))
    if(theta < eps):
        return np.identity(3)
    elif(np.pi - theta < eps):
        e_i = np.identity(3)[np.argmin(v)] #grab axis along minimum component of v
        rot_axis = np.cross(v,e_i)
    else:
        rot_axis = np.cross(v,vt)

    rot_axis /= np.linalg.norm(rot_axis)
    A = np.array([[0, -rot_axis[2], rot_axis[1]]
                  [rot_axis[2], 0, -rot_axis[0]]
                  [-rot_axis[1], rot_axis[0], 0]])

    return np.identity(3) + np.sin(theta)*A + (1.0-np.cos(theta))*A @ A

def rot_align(v, vt):
    """
    Align two sets of vectors
    Used for matching orientations to eachother
    Inputs:
        v: Nx3 set of vectors
        vt: Nx3 set of corresponding target vectors

    Outputs:
        R: 3x3 rotation matrix
    """


    return Rotation.align_vectors(vt, v).as_matrix()

def rot_zxz(alpha, beta, gamma, convention="intrinsic"):
    """
    Rotate to orientation described by zxz euler angles
    Inputs:
        alpha: rotation around z in radians
        beta: rotation around x' in radians
        gammma: rotation around z'' in radians
        convention: "intrinsic" or "extrinsic" (default: "intrinsic")

    Outputs:
        R: 3x3 rotation matrix
    """

    if convention=="intrinsic":
        return Rotation.from_euler("ZXZ", [alpha, beta, gamma], degrees=False).as_matrix()
    elif convention=="extrinsic":
        return Rotation.from_euler("zxz", [gamma, alpha, beta], degrees=False).as_matrix()
    else:
        raise(ValueError("Invalid argument for convention."))


#Translation

#Deformation