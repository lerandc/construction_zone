import copy
from abc import ABC, abstractmethod
from typing import Generator, List, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull, Delaunay, QhullError

from ..util.misc import round_away
from ..transform.strain import HStrain

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

    def __init__(self, tol: float = 1e-10):
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

    def from_alg_object(self, **kwargs):
        """Constructor for new algebraic objects based on existing Algebraic object.
        
        Args:
            **kwargs: "transformation"=List[BaseTransformation] to apply a 
                        series of transformations to the copied generator.
        """
        new_alg_object = copy.deepcopy(self)

        if "transformation" in kwargs.keys():
            for t in kwargs["transformation"]:
                new_alg_object = t.applyTransformation_alg(new_alg_object)

        return new_alg_object

class Sphere(BaseAlgebraic):
    """Algebraic surface for spheres. 
    
    Interior points are points with distance from center smaller than the radius.
    
    Attributes:
        radius (float): Radius of sphere.
        center (np.ndarray): 3x1 array representing center of sphere in space.
        tol (float): Tolerance value for interiority check. Default is 1e-5.
    """

    def __init__(self,
                 radius: float,
                 center: np.ndarray,
                 tol=1e-5):
                 
        self.radius = radius
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
        if center.size != 3 or center.shape[0] != 3:
            raise ValueError("Center must be an array with 3 elements")
        self._center = center


class Plane(BaseAlgebraic):
    """Algebraic surface for planes in R3.

    Interior points lie opposite in direction of plane normal,
    e.g., the point (0, 0, -1) is interior to Plane((0,0,1), (0,0,0))

    Attributes:
        normal (np.ndarray): normal vector describing orientation of plane.
        point (np.ndarray): point lying on plane.
        tol (float): Tolerance value for interiority check. Default is 1e-5.
    
    """

    def __init__(self,
                 normal: np.ndarray,
                 point: np.ndarray,
                 tol: float = 1e-5):
        self.normal = normal
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
        normal = np.reshape(normal, (3,)) #make a consistent shape
        if (np.linalg.norm(normal) > np.finfo(float).eps):
            self._normal = normal / np.linalg.norm(normal)
        else:
            raise ValueError(f"Input normal vector length {np.linalg.norm(normal)} is below machine precision.")

    def checkIfInterior(self, testPoints: np.ndarray):
        return self.sdist_from_plane(testPoints) < self.tol

    def flip_orientation(self):
        """Flip the orientation of the plane."""
        self.normal = - self.normal
        return self
    
    def sdist_from_plane(self, point: np.ndarray):
        """Calculate the signed distance from a point or series of points to the Plane.
        
        Arg:
            point (np.ndarray): Point in space to calculate distance.

        Returns:
            Array of distances to plane.

        """
        # separate into two dot products to avoid an an array subtraction against testPoints
        return np.dot(point, self.normal) - np.dot(self.point, self.normal)

    def dist_from_plane(self, point: np.ndarray):
        """Calculate the distance from a point or series of points to the Plane.
        
        Arg:
            point (np.ndarray): Point in space to calculate distance.

        Returns:
            Array of distances to plane.

        """
        return np.abs(self.sdist_from_plane(point))

    def project_point(self, point: np.ndarray):
        """Project a point in space onto Plane.
        
        Arg:
            point (np.ndarray): Point in space to project onto Plane.

        Returns:
            Projected point lying on surface of Plane.
        """
        return point - self.sdist_from_plane(point)[:,None] * self.normal[None,:]


class Cylinder(BaseAlgebraic):
    """Algebraic surface for circular cylinders in R3.

    Cylinders are defined with vectors, pointing parallel to central axis;
    points, lying along central axis; and radii, defining size of cylinder.

    Attributes:
        axis (np.ndarray): vector parallel to central axis of cylinder.
        point (np.ndarray): point which lies at the center of the cylinder
        radius (float): radius of cylinder.
        length (float): length of cylinder
        tol (float): Tolerance value for interiority check. Default is 1e-5.
    """

    def __init__(self,
                 axis: np.ndarray,
                 point: np.ndarray,
                 radius: float,
                 length: float,
                 tol: float = 1e-5):
        self.axis = axis
        self.point = point
        self.radius = radius
        self.length = length
        super().__init__(tol=tol)

    def params(self):
        """Return axis, point, radius, and length of cylinder."""
        return self.axis, self.point, self.radius, self.length

    @property
    def axis(self): 
        """Vector lying parallel to central axis."""
        return self._axis

    @axis.setter
    def axis(self, arr):
        arr = np.array(arr).reshape((3,))
        norm = np.linalg.norm(arr)
        if norm <= np.finfo(float).eps:
            raise ValueError(f"Input axis {arr} has norm {norm}, which is below machine precision.")
        self._axis = arr / norm

    @property
    def point(self):
        """Point lying along central axis."""
        return self._point

    @point.setter
    def point(self, arr):
        self._point = np.array(arr).reshape((3,))

    @property
    def radius(self):
        """Radius of cylinder."""
        return self._radius

    @radius.setter
    def radius(self, val):
        val = float(val)
        if val < np.finfo(float).eps: # negative or subnormal
            raise ValueError(f"Radius must be positive but is close to zero or negative.")
        self._radius = val

    @property
    def length(self):
        """Length of cylinder."""
        return self._length

    @length.setter
    def length(self, val):
        val = float(val)
        if val < np.finfo(float).eps: # negative or subnormal
            raise ValueError(f"Length must be positive but is close to zero or negative.")
        self._length = val

    def checkIfInterior(self, testPoints):
        rad_dists = np.linalg.norm(np.cross(testPoints - self.point,
                                        self.axis[None, :]),
                               axis=1)

        rad_check = rad_dists < self.radius + self.tol

        length_dists = np.abs(np.dot(testPoints-self.point, self.axis))
        length_check = length_dists < self.length/2.0 + self.tol
        return np.logical_and(rad_check,length_check)

    def get_bounding_box(self):
        # make a square inscribing cylinder at center disk
        # any rotation is valid

        # need vz to be normalized to project out vs_0
        vz = np.copy(self.axis.T)[:,None]

        # get vectors perpendicular to axis
        vs_0 = np.array([[1,1,1]]).T # any vector works; fix to make generation stable
        vs_0 = vs_0/np.linalg.norm(vs_0)
        vs_0 = vs_0 - (vs_0.T @ vz) * vz
        vs_0 = vs_0/np.linalg.norm(vs_0)

        vs_1 = np.cross(vs_0, vz, axis=0)
        vs_1 = vs_1/np.linalg.norm(vs_1)

        vs_0 = np.squeeze(vs_0)
        vs_1 = np.squeeze(vs_1)

        # factors of two cancel
        square = self.point + (self.radius)*np.array([vs_0+vs_1,
                                        vs_0-vs_1,
                                        -vs_0+vs_1,
                                        -vs_0-vs_1,
                                            ])
    
        # extend square halfway in both directions into rectangular prism
        vz = self.length*vz / 2.0
        return np.vstack([square+vz.T, square-vz.T])



#####################################
######### Utility routines ##########
#####################################


def convex_hull_to_planes(points, **kwargs):
    """Convert the convex hull of a set of points into a set of Planes."""

    tri = Delaunay(points)
    facets = tri.convex_hull

    def facet_to_plane(facet):
        v0 = points[facet[1],:] - points[facet[0],:]
        v1 = points[facet[2],:] - points[facet[0],:]
        n = np.cross(v0,v1)
        return Plane(n, points[facet[0]])

    planes = [facet_to_plane(f) for f in facets]

    ipoint = np.mean(points, axis=0)
    if tri.find_simplex(ipoint) < 0:
        raise AssertionError

    for i, plane in enumerate(planes):
        if not plane.checkIfInterior(ipoint):
            plane = plane.flip_orientation()

    return planes

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
        status: 0 if successful, 2 if no valid intersection, 3 if intersection is unbounded.
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
        try:
            hs = HalfspaceIntersection(np.hstack([A, d]), res.x[:-1])
        except QhullError:
            return np.empty((0,3)), 2
            
        return hs.intersections - shift, res.status
    else:
        return np.empty((0,3)), res.status

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


    # check if generator has a strain field
    if generator.strain_field is None:
        # get point coordinates in generator coordinate system
        point_fcoord = np.array(np.linalg.solve(generator.voxel.sbases, point))
    else:
        assert ( isinstance(
            generator.strain_field, HStrain)), "Finding Miller planes with inhomogenous strain fields is not supported."

        if generator.strain_field.mode == "crystal":
            H = generator.strain_field.matrix
            point_fcoord = np.array(np.linalg.solve(H @ generator.voxel.sbases, point))

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

    if generator.strain_field is None:
        # scale back to real space
        new_point = generator.voxel.sbases @ new_point

        # get perpendicular vector
        normal = generator.voxel.reciprocal_bases.T @ miller_indices
    else:
        H = generator.voxel.sbases
        G = generator.strain_field.matrix
        new_point = G @ H @ new_point

        # get perpendicular vector
        normal = np.linalg.inv(H @ G).T @ miller_indices

    return Plane(normal=normal, point=new_point)