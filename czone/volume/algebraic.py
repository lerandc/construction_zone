from abc import ABC, abstractmethod
from typing import Generator, List, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

from ..util.misc import round_away

#####################################
##### Geometric Surface Classes #####
#####################################


class BaseAlgebraic(ABC):
    """Base class for algebraic surfaces.


    Attributes:
        params (Tuple): parameters describing algebraic object
        tol (float): numerical tolerance used to pad interiority checks.
                    Default is 1e-5.
    
    """

    def __init__(self, tol: float = 1e-5):
        self.tol = tol

    @abstractmethod
    def checkIfInterior(self, testPoints: np.ndarray):
        """Check if points lie on interior side of geometric surface.

        Args:
            testPoints (np.ndarray): Nx3 array of points to check.
        
        Returns:
            Nx1 logical array indicating whether or not point is on interior 
            of surface.
        """
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, val):
        assert (float(val) > 0.0)
        self._tol = float(val)


class Sphere(BaseAlgebraic):
    """Algebraic surface for spheres. 
    
    Interior points are points with distance from center smaller than the radius.
    
    Attributes:
        radius (float): Radius of sphere.
        center (np.ndarray): 3x1 array representing center of sphere in space.
        tol (float): Tolerance value for interiority check. Default is 1e-5.
    """

    def __init__(self,
                 radius: float = None,
                 center: np.ndarray = None,
                 tol=1e-5):
        self._radius = None
        self._center = np.array([0, 0, 0])

        if not (radius is None):
            self.radius = radius

        if not (center is None):
            self.center = center

        super().__init__(tol=tol)

    def checkIfInterior(self, testPoints: np.ndarray) -> np.ndarray:
        return np.sum((testPoints - self.center)**2.0,
                      axis=1) < (self.radius + self.tol)**2.0

    @property
    def params(self):
        """Return radius, center of Sphere."""
        return self.radius, self.center

    @property
    def radius(self):
        """Radius of sphere."""
        return self._radius

    @radius.setter
    def radius(self, radius: float):
        if radius > 0.0:
            self._radius = radius
        else:
            raise ValueError("Radius needs to be positive valued.")

    @property
    def center(self):
        """Center of sphere in space."""
        return self._center

    @center.setter
    def center(self, center: np.ndarray):
        center = np.array(center)  #cast to np array if not already
        assert (center.size == 3), "Center must be a point in 3D space"
        assert (center.shape[0] == 3), "Center must be a point in 3D space"
        self._center = center


class Plane(BaseAlgebraic):
    """Algebraic surface for planes in R3.

    Interior points lie opposite in direction of plane normal.

    Attributes:
        point (np.ndarray): point lying on plane.
        normal (np.ndarray): normal vector describing orientation of plane.
        tol (float): Tolerance value for interiority check. Default is 1e-5.
    
    """

    def __init__(self,
                 normal: np.ndarray = None,
                 point: np.ndarray = None,
                 tol: float = 1e-5):
        self._normal = None
        self._point = None

        if not (normal is None):
            self.normal = normal

        if not (point is None):
            self.point = point

        super().__init__(tol=tol)

    @property
    def params(self):
        """Return normal vector, point on plane of Plane."""
        return self.normal, self.point

    @property
    def point(self):
        """Point lying on surface of Plane."""
        return self._point

    @point.setter
    def point(self, point: np.ndarray):
        point = np.squeeze(np.array(point))  #cast to np array if not already
        assert (point.shape[0] == 3), "Point must be a point in 3D space"
        self._point = point

    @property
    def normal(self):
        """Normal vector defining orientation of Plane in space."""
        return self._normal

    @normal.setter
    def normal(self, normal: np.ndarray):
        normal = np.array(normal)  #cast to np array if not already
        assert (normal.size == 3), "normal must be a vector in 3D space"
        # normal = np.reshape(normal, (3,1)) #make a consistent shape
        if (np.linalg.norm(normal) > np.finfo(float).eps):
            self._normal = normal / np.linalg.norm(normal)
        else:
            raise ValueError("Normal vector must have some length")

    def checkIfInterior(self, testPoints: np.ndarray):
        return np.sum((testPoints * self.normal), axis=1) - np.squeeze(
            np.dot(self.normal, self.point + self.normal * self.tol)) < 0

    def flip_orientation(self):
        """Flip the orientation of the plane."""
        self.normal = -1 * self.normal

    def dist_from_plane(self, point: np.ndarray):
        """Calculate the distance from a point or series of points to the Plane.
        
        Arg:
            point (np.ndarray): Point in space to calculate distance.

        Returns:
            Array of distances to plane.

        """
        return np.abs(np.dot(point - self.point, self.normal))

    def project_point(self, point: np.ndarray):
        """Project a point in space onto Plane.
        
        Arg:
            point (np.ndarray): Point in space to project onto Plane.

        Returns:
            Projected point lying on surface of Plane.
        """
        return point - self.dist_from_plane(point) * self.normal.T


class Cylinder(BaseAlgebraic):
    """Algebraic surface for circular cylinders in R3.

    Cylinders are defined with vectors, pointing parallel to central axis;
    points, lying along central axis; and radii, defining size of cylinder.

    Attributes:
        axis (np.ndarray): vector parallel to central axis of cylinder.
        point (np.ndarray): point which lies along central axis of cylinder.
        radius (float): radius of cylinder.
        tol (float): Tolerance value for interiority check. Default is 1e-5.
    """

    #TODO: Write transformations for handling cylinder parameters.

    def __init__(self,
                 axis: np.ndarray = [0, 0, 1],
                 point: np.ndarray = [0, 0, 0],
                 radius: float = 1.0,
                 tol: float = 1e-5):
        self.axis = axis
        self.point = point
        self.radius = radius
        super().__init__(tol=tol)

    def params(self):
        """Return axis, point, and radius of cylinder."""
        return self.axis, self.point, self.radius

    @property
    def axis(self):
        """Vector lying parallel to central axis."""
        return self._axis

    @axis.setter
    def axis(self, arr):
        arr = np.squeeze(np.array(arr))
        assert (arr.shape == (3,))
        self._axis = arr / np.linalg.norm(arr)

    @property
    def point(self):
        """Point lying along central axis."""
        return self._point

    @point.setter
    def point(self, arr):
        arr = np.squeeze(np.array(arr))
        assert (arr.shape == (3,))
        self._point = arr

    @property
    def radius(self):
        """Radius of cylinder."""
        return self._radius

    @radius.setter
    def radius(self, val):
        try:
            val = float(val)
            self._radius = val
        except TypeError:
            raise TypeError("Supplied value must be castable to float")

    def checkIfInterior(self, testPoints):
        dists = np.linalg.norm(np.cross(testPoints - self.point,
                                        self.axis[None, :]),
                               axis=1)
        return dists < self.radius + self.tol


#####################################
######### Utility routines ##########
#####################################


def get_bounding_box(planes: List[Plane]):
    """Get convex region interior to set of Planes, if one exists.

    Determines if set of planes forms a valid interior convex region. If so, 
    returns vertices of convex region. Uses scipy half space intersection and 
    linear progamming routines to determine boundaries of convex region and 
    valid interior points.

    Args:
        planes (List[Plane]): set of planes to check mutual intersection of.

    Returns:
        np.ndarray: Nx3 array of vertices of convex region.
        2: no valid intersection.
        3: if intersection is unbounded.
    """

    # some issues arose when all planes were in negative coordinate space
    # so shift planes so that all points line positive coordinates
    shift = np.zeros_like(planes[0].point)
    for plane in planes:
        shift = np.min([shift, plane.point], axis=0)

    shift = -1.5 * shift

    # convert plane set to matrix form of half spaces
    A = np.zeros((len(planes), 3))
    d = np.zeros((len(planes), 1))
    norms = np.zeros((len(planes), 1))
    for i, plane in enumerate(planes):
        n, p = plane.params
        A[i, :] = n.squeeze()
        d[i, 0] = -1.0 * np.dot(n.squeeze(), (p + shift).squeeze())
        norms[i, 0] = np.linalg.norm(n)

    #check feasiblity of region and get interior point
    c = np.zeros(4)
    c[-1] = -1
    res = linprog(c, A_ub=np.hstack([A, norms]), b_ub=-1.0 * d)

    if (res.status == 0):
        hs = HalfspaceIntersection(np.hstack([A, d]), res.x[:-1])
        return hs.intersections - shift
    else:
        return res.status


def snap_plane_near_point(point: np.ndarray,
                          generator: Generator,
                          miller_indices: Tuple[int],
                          mode: str = "nearest"):
    """Determine nearest crystallographic nearest to point in space for given crystal coordinate system.
    
    Args:
        point (np.ndarray): Point in space.
        generator (Generator): Generator describing crystal coordinat system.
        miller_indices (Tuple[int]): miller indices of desired plane.
        mode (str): "nearest" for absolute closest plane to point; "floor" for
                    next nearest valid plane towards generator origin; "ceil"
                    for next furthest valid plane from generator origin.

    Returns:
        Plane in space with orientation given by Miller indices snapped to 
        nearest valid location.

    """

    miller_indices = np.array(miller_indices)

    # get point coordinates in generator coordinate system
    point_fcoord = np.array(np.linalg.solve(generator.voxel.sbases, point))

    # get lattice points that are intersected by miller plane
    with np.errstate(divide="ignore"):  #check for infs directly
        target_fcoord = 1 / miller_indices

    new_point = np.zeros((3, 1))

    # TODO: if bases are not orthonormal, this procedure is not correct
    # since the following rounds towards the nearest lattice points, with equal
    # weights given to all lattice vectors
    if mode == "nearest":
        for i in range(3):
            new_point[i,0] = np.round(point_fcoord[i]/target_fcoord[i])*target_fcoord[i] \
                                if not np.isinf(target_fcoord[i]) else point_fcoord[i]
    elif mode == "ceil":
        for i in range(3):
            new_point[i,0] = round_away(point_fcoord[i]/target_fcoord[i])*target_fcoord[i] \
                                if not np.isinf(target_fcoord[i]) else point_fcoord[i]
    elif mode == "floor":
        for i in range(3):
            new_point[i,0] = np.fix(point_fcoord[i]/target_fcoord[i])*target_fcoord[i] \
                                if not np.isinf(target_fcoord[i]) else point_fcoord[i]

    # scale back to real space
    new_point = generator.voxel.sbases @ new_point

    # get perpendicular vector
    normal = generator.voxel.reciprocal_bases.T @ miller_indices
    return Plane(normal=normal, point=new_point)
