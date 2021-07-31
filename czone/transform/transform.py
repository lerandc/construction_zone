import copy
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import \
    Rotation as scRotation  # avoid namespace conflicts

from ..volume.algebraic import BaseAlgebraic, Plane, Sphere

#####################################
########### Base Classes ############
#####################################


class BaseTransform(ABC):
    """Base class for transformation objects which manipulate Generators and Volumes.

    Transformation objects contain logic and parameters for manipulating the 
    different types of objects used in Construction Zone, namely, Generators and
    Volumes. BaseTransform is typically not created directly. Use MatrixTransform for 
    generalized matrix transformations.

    Attributes:
        locked (bool): whether or not transformation applies jointly to volumes containing generators
        basis_only (bool): whether or not transformation applies only to basis of generators
        params (tuple): parameters describing transformation

    """

    def __init__(self, locked: bool = True, basis_only: bool = False):
        self.locked = locked
        self.basis_only = basis_only

    @abstractmethod
    def applyTransformation(self, points: np.ndarray) -> np.ndarray:
        """Apply transformation to a collection of points in space.
     
        Args:
            points (np.ndarray): Nx3 array of points to transform.

        Returns:
            np.ndarray: Nx3 array of transformed points.
        """
        pass

    @abstractmethod
    def applyTransformation_bases(self, points: np.ndarray) -> np.ndarray:
        """Apply transformation to bases of a generator.

        Args:
            points (np.ndarray): 3x3 array of bases from generator voxel.

        Returns:
            np.ndarray: transformed 3x3 array of bases.
        """
        pass

    @abstractmethod
    def applyTransformation_alg(self,
                                alg_object: BaseAlgebraic) -> BaseAlgebraic:
        """Apply transformation to algebraic object.

        Args:
            alg_object (BaseAlgebraic): Algebraic object to transform.

        Returns:
            BaseAlgebraic: Transformed object.
        """
        pass

    @property
    @abstractmethod
    def params(self) -> tuple:
        """Return parameters describing transformation. """
        pass

    @property
    def locked(self) -> bool:
        """Boolean value indicating whether or not transformation jointly applied to Volumes and Generators."""
        return self._locked

    @locked.setter
    def locked(self, locked):
        assert (isinstance(locked,
                           bool)), "Must supply bool to locked parameter."
        self._locked = locked

    @property
    def basis_only(self) -> bool:
        """Boolean value indicating whether or not transformation applied only to basis of Generators."""
        return self._basis_only

    @basis_only.setter
    def basis_only(self, basis_only):
        assert (isinstance(basis_only,
                           bool)), "Must supply bool to basis_only parameter"
        self._basis_only = basis_only


class Translation(BaseTransform):
    """Transformation object that applies translations to Generators and Volumes.

    Attributes:
        shift (np.ndarray): Translation vector in 3D space.
    """

    def __init__(self,
                 shift=None,
                 locked: bool = True,
                 basis_only: bool = False):
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
        assert (shift.size == 3), "Shift must be a vector in 3D space"
        assert (shift.shape[0] == 3), "Shift must be a vector in 3D space"
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
            alg_object.point = alg_object.point + self.shift

        return alg_object


class MatrixTransform(BaseTransform):
    """Transformation object that applies arbitrary matrix transformations.

    Inversion, Rotation, and Reflection classes are derived classes of MatrixTransform
    that have special setters which ensure validity of their respective transformations.

    Attributes:
        matrix (np.ndarray): 3x3 matrix representing transformation
        origin (np.ndarray): point in R3 representing relative origin in which 
                             transformation is applied
    """

    def __init__(self,
                 matrix=None,
                 origin=None,
                 locked: bool = True,
                 basis_only: bool = False):
        self._matrix = None
        self._origin = None

        if not (matrix is None):
            self.matrix = matrix

        if not (origin is None):
            self.origin = origin

        super().__init__(locked=locked, basis_only=basis_only)

    @property
    def matrix(self) -> np.ndarray:
        """3x3 array representing matrix transformation """
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: np.ndarray):
        assert (matrix.shape == (
            3, 3)), "Input matrix must be square 3x3 numpy array"
        self._matrix = matrix

    @property
    def origin(self) -> np.ndarray:
        """Point in R3 representing relative origin of transformation"""
        return self._origin

    @origin.setter
    def origin(self, origin):
        origin = np.array(origin)
        assert (origin.size == 3), "Origin must be a point in 3D space"
        assert (origin.shape[0] == 3), "Origin must be a point in 3D space"
        self._origin = origin

    @property
    def params(self):
        return self.matrix, self.origin

    def applyTransformation(self, points):
        if not (self.origin is None):
            points = np.dot(self.matrix,
                            (points - self.origin).T).T + self.origin
        else:
            points = np.dot(self.matrix, (points).T).T

        return points

    def applyTransformation_bases(self, points):
        return np.dot(self.matrix, points)

    def applyTransformation_alg(self, alg_object):

        if isinstance(alg_object, Sphere):
            if not (self.origin is None):
                alg_object.center = np.dot(
                    self.matrix,
                    (alg_object.center - self.origin).T).T + self.origin
            else:
                alg_object.center = np.dot(self.matrix, (alg_object.center).T).T

        if isinstance(alg_object, Plane):
            if not (self.origin is None):
                alg_object.point = np.dot(
                    self.matrix,
                    (alg_object.point - self.origin).T).T + self.origin
            else:
                alg_object.point = np.dot(self.matrix, (alg_object.point).T).T

            alg_object.normal = np.dot(self.matrix, (alg_object.normal).T).T

        return alg_object


class Inversion(MatrixTransform):
    """ Transformation which applies inversion about an origin.

    Inversion, Rotation, and Reflection classes are derived classes of MatrixTransform
    that have special setters which ensure validity of their respective transformations.
    """

    @property
    def matrix(self):
        """Inversion matrix. Cannot be altered."""
        return -1.0 * np.eye(3)


class Rotation(MatrixTransform):
    """Transformation which applies a rotation about an origin.

    Inversion, Rotation, and Reflection classes are derived classes of MatrixTransform
    that have special setters which ensure validity of their respective transformations.
    """

    def __init__(self,
                 matrix=None,
                 origin=None,
                 locked: bool = True,
                 basis_only: bool = False):
        super().__init__(matrix=matrix,
                         origin=origin,
                         locked=locked,
                         basis_only=basis_only)

    @MatrixTransform.matrix.setter
    def matrix(self, val):
        assert (val.shape == (3,
                              3)), "Input matrix must be square 3x3 numpy array"
        assert (np.abs(np.linalg.det(val) - 1.0) < 1e-6
               ), "Input matrix not a valid rotation matrix. Fails determinant."
        assert (
            np.sum(np.abs(val @ val.T - np.eye(3))) < 1e-6
        ), "Input matrix not a valid rotation matrix. Fails orthogonality."
        self._matrix = val


class Reflection(MatrixTransform):
    """Transformation which applies a relfection about an arbitrary plane.

    Inversion, Rotation, and Reflection classes are derived classes of MatrixTransform
    that have special setters which ensure validity of their respective transformations.

    Attributes:
        plane (Plane): plane about which reflection is performed
    """

    def __init__(self,
                 plane=None,
                 locked: bool = True,
                 basis_only: bool = False):
        super().__init__(matrix=None,
                         origin=None,
                         locked=locked,
                         basis_only=basis_only)

        if not (plane is None):
            self.plane = plane

    @MatrixTransform.matrix.setter
    def matrix(self, normal):
        self._matrix = np.eye(3) - (2.0 /
                                    (normal.T @ normal)) * (normal @ normal.T)

    @property
    def plane(self):
        return self._plane

    @plane.setter
    def plane(self, plane):
        assert (isinstance(
            plane,
            Plane)), "Input plane must be Plane object from algebraic module"
        self._plane = plane
        self.matrix = plane.normal.reshape(3, 1)
        self.origin = plane.point

    @property
    def params(self):
        return self.plane

    def applyTransformation_alg(self, alg_object):
        if (isinstance(alg_object), Sphere):
            alg_object.center = np.dot(
                self.matrix,
                (alg_object.center - self.origin).T).T + self.origin

        if (isinstance(alg_object), Plane):
            alg_object.point = np.dot(
                self.matrix, (alg_object.point - self.origin).T).T + self.origin

        return alg_object


class MultiTransform(BaseTransform):
    """Transformation sequence which applies a numeruous transformations in succession.

    Attributes:
        transforms (list[BaseTransform]): list of individual transformations applied
        locked (bool): whether any of transformations applied is a joint transformation
        basic_only (bool): whether any of transformations applied transforms 
                           only the generator basis

    """

    def __init__(self, transforms=None):
        self._transforms = []

        if not transforms is None:
            self.add_transform(transforms)

    @property
    def transforms(self):
        """ Sequence of individual transformations to apply"""
        return self._transforms

    @property
    def params(self):
        return [x.params for x in self.transforms]

    @property
    def locked(self):
        """Boolean value indicating whether or not any of transformations applies jointly."""
        locked_list = [x.locked for x in self.transforms]
        return np.any(locked_list)

    @property
    def basis_only(self):
        """Boolean value indicating whether or not any of transformations applies only to basis of generator"""
        """
        Essetially signals to generator that at least one of its transforms will transform the origin
        Should instead pass the voxel to the transformation objects and handle it internally per transform
        because if not all of them affect the origin, the origin will get wrongly manipulated.
        """
        basis_only_list = [x.basis_only for x in self.transforms]
        return np.any(basis_only_list)

    """
    Methods
    """

    def add_transform(self, transform):
        """Add a transformation to sequence of transformations"""
        if hasattr(transform, '__iter__'):
            for t in transform:
                assert (isinstance(t, BaseTransform)
                       ), "transforms must be BaseTransformation objects"
            self._transforms.extend(transform)
        else:
            assert (isinstance(
                transform,
                BaseTransform)), "transforms must be BaseTransformation objects"
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
            alg_object = t.applyTransformation_alg(alg_object)
        return alg_object


#####################################
############# Utilities #############
#####################################


def rot_v(v, theta):
    """Calculate rotation about an arbitary vector.

    Args:
        v (np.ndarray): axis of rotation
        theta (float): angle of rotation in radians

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    v = np.array(v)
    v = v * (theta / np.linalg.norm(v))

    return scRotation.from_rotvec(v).as_matrix()


def rot_vtv(v: np.ndarray, vt: np.ndarray) -> np.ndarray:
    """Calculate rotation to align one vector to another.

    Args:
        v (np.ndarray): 1x3 original vector
        vt (np.ndarray): 1x3 target vector into which v is rotated
    
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    """
    implementation based on following:
    https://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    In short, use cross product to get axis of rotation, then develop matrix form of
    Rodrigues rotation of formula and return rotation matrix.
    """
    v = np.array(v).reshape(3,)
    vt = np.array(vt).reshape(3,)
    eps = np.finfo(float).eps  #get machine epsilon for collinearity check
    theta = np.arccos(np.dot(v, vt) / (np.linalg.norm(v) * np.linalg.norm(vt)))
    if (theta < eps):
        return np.eye(3)
    elif (np.pi - theta < eps):
        e_i = np.eye(3)[np.argmin(v)]  #grab axis along minimum component of v
        rot_axis = np.cross(v, e_i).astype(float)
    else:
        rot_axis = np.cross(v, vt).astype(float)

    rot_axis /= np.linalg.norm(rot_axis)
    A = np.array([[0, -rot_axis[2],
                   rot_axis[1]], [rot_axis[2], 0, -rot_axis[0]],
                  [-rot_axis[1], rot_axis[0], 0]])

    return np.eye(3) + np.sin(theta) * A + (1.0 - np.cos(theta)) * A @ A


def rot_align(v: np.ndarray, vt: np.ndarray) -> np.ndarray:
    """Calculate rotation aligning two sets of ordered vectors.

    Args:
        v  (np.ndarray): Nx3 set of vectors
        vt (np.ndarray): Nx3 set of corresponding target vectors

    Returns:
        np.ndarray: 3x3 rotation matrix
    """

    return scRotation.align_vectors(vt, v).as_matrix()


def rot_zxz(alpha: float,
            beta: float,
            gamma: float,
            convention="intrinsic") -> np.ndarray:
    """Rotate to orientation described by zxz euler angles.

    Calculate rotation matrix as determined by zxz Euler angles. Convention can
    be either intrinsic, in which rotations are performed on the moving coordinate
    system XYZ in reference to a fixed coordinate system xyz, or extrinsic, 
    in which the rotations are performed about the fixed coordinate system xyz.
    Intrisic rotations are performed alpha-beta-gamma, while extrinsic rotations
    are perfomed gamma-beta-alpha.

    Args:
        alpha (float): rotation around Z / z in radians
        beta (float): rotation around X'/ x in radians
        gammma (float): rotation around Z''/ z in radians
        convention (float): "intrinsic" or "extrinsic" (default: "intrinsic")

    Returns:
        np.ndarray : 3x3 rotation matrix
    """

    if convention == "intrinsic":
        return scRotation.from_euler("ZXZ", [alpha, beta, gamma],
                                     degrees=False).as_matrix()
    elif convention == "extrinsic":
        return scRotation.from_euler("zxz", [gamma, alpha, beta],
                                     degrees=False).as_matrix()
    else:
        raise (ValueError("Invalid argument for convention."))


def s2s_alignment(M_plane: Plane, T_plane: Plane, M_point: np.ndarray,
                  T_point: np.ndarray) -> MultiTransform:
    """Calculates a transformation aligning two surfaces to eachother.

    Aligns two surfaces, represented by algebraic plane objects, such that 
    their normal vectors are anti-parallel. The translational degree of freedom
    is resolved by minimizing the distance of a point in the target alignment to
    a transformed point from the moving frame. Returned Transformation sequence
    is a rotation followed by translation.

    Specifically, given a point in the moving frame M_p with plane M and a 
    point in the target frame T_p with plane T, calculate rotation R such that 
    (R @ M.normal ) .* T.normal = -1 and translation vector T_t under the 
    constraint T_t = min T_t ||R @ M_p - T_p||

    Args:
        M_plane (Plane): Moving plane to align to another surface
        T_plane (Plane): Target plane to which M_plane is aligned 
        M_point (np.ndarray): Point in moving plane reference frame
        T_point (np.ndarray): Point in target plane reference frame

    Returns:
        MultiTransform: Transformation routine represented by surface alignment.
    """
    R = rot_vtv(M_plane.normal, -1.0 * T_plane.normal)

    R_t = Rotation(matrix=R, origin=M_plane.point)
    M_plane_r = R_t.applyTransformation_alg(M_plane)
    M_point_r = R_t.applyTransformation(M_point)

    proj_point = T_plane.project_point(M_point)
    plane_shift = T_plane.project_point(T_point) - proj_point

    ortho_shift = 1.0 * T_plane.dist_from_plane(M_plane.point) * T_plane.normal
    T_t = Translation(shift=ortho_shift + plane_shift)

    return MultiTransform(transforms=[R_t, T_t])
