import copy
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as scRotation #avoid namespace conflicts
from ..volume.algebraic import Plane, Sphere

#####################################
########### Base Classes ############
#####################################

class BaseTransformation(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def applyTransformation(self, points):
        pass
    
    @abstractmethod
    def applyTransformation_bases(self, points):
        pass

    @abstractmethod
    def applyTransformation_alg(self, alg_object):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @property
    def locked(self):
        return self._locked

    @locked.setter
    def locked(self, locked):
        assert(isinstance(locked, bool)), "Must supply bool to locked parameter."
        self._locked = locked

    @property
    def basis_only(self):
        return self._basis_only
    
    @basis_only.setter
    def basis_only(self, basis_only):
        assert(isinstance(basis_only, bool)), "Must supply bool to basis_only parameter"
        self._basis_only = basis_only

class Translation(BaseTransformation):

    def __init__(self, shift=None, locked=True, basis_only=False):
        self._shift = None
        self._locked = None

        if not (shift is None):
            self.shift = shift

        self.locked = locked
        self.basis_only = basis_only

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, shift):
        shift = np.array(shift)
        assert(shift.size == 3), "Shift must be a vector in 3D space"
        assert(shift.shape[0] == 3), "Shift must be a vector in 3D space"
        self._shift = shift

    @property
    def params(self):
        return self.shift

    def applyTransformation(self, points):
        return points + self.shift

    def applyTransformation_bases(self, points):
        return points

    def applyTransformation_alg(self, alg_object):
        if isinstance(alg_object, Sphere):
            alg_object.center = alg_object.center + self.shift

        if isinstance(alg_object, Plane):
            alg_object.point = alg_object.point + self.shift.reshape(3,1)

        return alg_object

class Rotation(BaseTransformation):

    def __init__(self, matrix=None, origin=None, locked=True,  basis_only=False):
        self._matrix = None
        self._origin = None
        self._locked = None

        if not (matrix is None):
            self.matrix = matrix

        if not (center is None):
            self.origin = origin

        self.locked = locked
        self.basis_only = basis_only

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        assert(matrix.shape == (3,3)), "Input matrix must be square 3x3 numpy array"
        assert(np.abs(np.linalg.det(matrix)-1.0) < 1e-6), "Input matrix not a valid rotation matrix. Fails determinant."
        assert(np.sum(np.abs(matrix * matrix.T - np.eye(3))) < 1e-6), "Input matrix not a valid rotation matrix. Fails orthogonality." 
        self._matrix = matrix

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):
        origin = np.array(origin)
        assert(origin.size == 3), "Origin must be a point in 3D space"
        assert(origin.shape[0] == 3), "Origin must be a point in 3D space"
        self._origin = origin

    @property
    def params(self):
        return self.matrix, self.origin

    def applyTransformation(self, points):
        if not (self.origin is None):
            points = np.dot(self.matrix, (points-self.origin).T).T+self.origin
        else:
            points = np.dot(self.matrix, (points).T).T

        return points

    def applyTransformation_bases(self, points):
        return np.dot(self.matrix, (points).T).T

    def applyTransformation_alg(self, alg_object):

        if isinstance(alg_object, Sphere):
            """
            Should only rotate the Sphere in space if rotation origin and sphere origin differ
            """
            if (self.origin is None):
                alg_object.center = np.dot(self.matrix, (alg_object.center-self.origin).T).T+self.origin
            else:
                alg_object.center = np.dot(self.matrix, (alg_object.center).T).T
            
        if isinstance(alg_object, Plane):
            if (self.origin is None):
                alg_object.point = np.dot(self.matrix, (alg_object.point-self.origin).T).T+self.origin
            else:
                alg_object.point = np.dot(self.matrix, (alg_object.point).T).T
            
            alg_object.normal = np.dot(self.matrix, (alg_object.normal).T).T

        return alg_object

class Reflection(BaseTransformation):

    def __init__(self, plane=None, locked=True, basis_only=False):
        self._plane = None
        self._locked = None
        self._matrix = None

        if not (plane is None):
            self.plane = plane

        self.locked = locked
        self.basis_only = basis_only

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, normal):
        self._matrix = np.eye(3) - (2.0/(normal.T @ normal))*(normal @ normal.T)

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, shift):
        self._shift = np.reshape(shift, (1,3))

    @property
    def plane(self):
        return self._plane
    
    @plane.setter
    def plane(self, plane):
        assert(isinstance(plane, Plane)), "Input plane must be Plane object from algebraic module"
        self._plane = plane
        self.matrix = plane.normal
        self.shift = plane.point

    @property
    def params(self):
        return self.plane

    def applyTransformation(self, points):
        """
        Reflect with shifted Householder transformation
        """
        return np.dot(self.matrix, (points-self.shift).T).T+self.shift

    def applyTransformation_bases(self, points):
        """
        Reflect with Householder transformation
        """
        return np.dot(self.matrix, (points).T).T

    def applyTransformation_alg(self, alg_object):

        if(isinstance(alg_object), Sphere):
            alg_object.center = np.dot(self.matrix, (alg_object.center-self.shift).T).T+self.shift

        if(isinstance(alg_object), Plane):
            alg_object.point = np.dot(self.matrix, (alg_object.point-self.shift).T).T+self.shift

        return alg_object

class MultiTransform(BaseTransformation):

    def __init__(self, transforms=None):
        self._transforms = []

        if not transforms is None:
            self.add_transform(transforms)

    """
    Properties
    """
    @property
    def transforms(self):
        return self._transforms

    @property
    def params(self):
        return [x.params for x in self.transforms]

    """
    Methods
    """
    def add_transform(self, transform):
        if hasattr(transform, '__iter__'):
            for t in transform:
                assert(isinstance(t, BaseTransformation)), "transforms must be BaseTransformation objects"
            self._transforms.extend(transform)
        else:
            assert(isinstance(transform, BaseTransformation)), "transforms must be BaseTransformation objects"
            self._transforms.append(transform)

    def applyTransformation(self, points):
        for t in self.transforms:
            points = t.applyTransformation(points)
        return points

    def applyTransformation_bases(self, points):
        for t in self.transforms:
            points = t.applyTransformation_bases(points)
        return points

    def applyTransformation_alg(self, alg_object):
        for t in self.transforms:
            alg_object = t.applyTransformation(alg_object)
        return alg_object


#####################################
############# Utilities #############
#####################################

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

    return scRotation.from_rotvec(v).as_matrix()

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

    return scRotation.align_vectors(vt, v).as_matrix()

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
        return scRotation.from_euler("ZXZ", [alpha, beta, gamma], degrees=False).as_matrix()
    elif convention=="extrinsic":
        return scRotation.from_euler("zxz", [gamma, alpha, beta], degrees=False).as_matrix()
    else:
        raise(ValueError("Invalid argument for convention."))

def s2s_alignment(M_plane: Plane, T_plane: Plane, M_point, T_point):
    R = rot_vtv(M_plane.normal, -1.0*T_plane.normal)
    R_t = Rotation(matrix=R, origin=M_plane.point)
    M_plane_r = R_t.applyTransformation_alg(M_plane)
    M_point_r = R_t.applyTransformation(M_point)

    proj_point = T_plane.project_point(M_point)

    T_t = Translation(shift=T_point-proj_point)

    return MultiTransform(transforms=[R_t, T_t])
