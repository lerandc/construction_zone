import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

#####################################
##### Geometric Surface Classes #####
#####################################

class BaseAlgebraic(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def checkIfInterior(self, testPoints):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

class Sphere(BaseAlgebraic):

    def __init__(self, radius=None, center=None):
        self._radius = None
        self._center = None

        if not (radius is None):
            self.radius = radius
            
        if not (center is None):
            self.center = center

    def checkIfInterior(self, testPoints):
        return np.sum((testPoints-self.center)**2.0, axis=1) < self.radius**2.0

    @property
    def params(self):
        return self.radius, self.center

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        if radius > 0.0:
            self._radius = radius
        else:
            raise ValueError("Radius needs to be positive valued.")

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center):
        center = np.array(center) #cast to np array if not already
        assert(center.size == 3), "Center must be a point in 3D space"
        assert(center.shape[0] == 3), "Center must be a point in 3D space"
        self._center = center

class Plane(BaseAlgebraic):
    """
    Normal vector points to excluded half-space
    """
    def __init__(self, normal=None, point=None):
        self._normal = None
        self._point = None

            
        if not (normal is None):
            self.normal = normal

        if not (point is None):
            self.point = point

    @property
    def params(self):
        return self.normal, self.point

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, point):
        point = np.array(point) #cast to np array if not already
        assert(point.shape[0] == 3), "Point must be a point in 3D space"

        self._point = point

    @property
    def normal(self):
        return self._normal

    @normal.setter
    def normal(self, normal):
        normal = np.array(normal) #cast to np array if not already
        assert(normal.size == 3), "normal must be a vector in 3D space"
        normal = np.reshape(normal, (3,1)) #make a consistent shape
        if(np.linalg.norm(normal) > np.finfo(float).eps):
            self._normal = normal/np.linalg(normal)
        else:
            raise ValueError("Normal vector must have some length")
    
    def checkIfInterior(self, testPoints):
        return np.sum((testPoints*self.normal.T), axis=1) - np.squeeze(np.dot(self.normal.T, self.point)) < 0

    def flip_orientation(self):
        self.normal = -1*self.normal

    def dist_from_plane(self, point):
        return np.dot(point-self.point, self.normal)

    def project_point(self, point):
        return point - self.dist_from_plane(point)*self.normal

            

#####################################
######### Utility routines ##########
#####################################

def get_bounding_box(planes):
    """
    Input is iterable of Plane objects.

    Determines if set of planes forms a valid interior convex region and 
    returns vertices representing convex region.


    If fails,
    returns 2 if no valid intersection
    returns 3 if intersection is unbounded 
    """

    A = np.zeros((len(planes),3))
    d = np.zeros((len(planes),1))
    norms = np.zeros((len(planes),1))
    for i, plane in enumerate(planes):
        n, p = plane.params
        A[i,:] = n.squeeze()
        d[i,0] = -1.0*np.dot(n.squeeze(),p.squeeze())
        norms[i,0] = np.linalg.norm(n)

    #check feasiblity of region and get interior point
    c = np.zeros(4)
    c[-1] = -1
    res = linprog(c, A_ub=np.hstack([A, norms]), b_ub=-1.0*d)

    if(res.status == 0):
        hs = HalfspaceIntersection(np.hstack([A,d]),res.x[:-1])
        return hs.intersections
    else:
        return res.status

def snap_plane_near_point(point, generator, miller_indices, mode="nearest"):
    """
    Returns nearest plane defined by miller indices in current generator coordinate system
    to the supplied point.

    Mode:
    nearest- closest plane to point
    floor  - next nearest valid plane towards generator origin 
    ceil   - next furthest valid plane from generator origin
    """

    miller_indices = np.array(miller_indices)

    # get point coordinates in generator coordinate system
    point_fcoord = np.array(np.linalg.solve(generator.voxel.sbases, point))

    # get lattice points that are intersected by miller plane
    with np.errstate(divide="ignore"): #check for infs directly
        target_fcoord = 1/miller_indices

    # TODO: if bases are not orthonormal, this procedure is not correct
    new_point = np.zeros((3,1))
    if mode=="nearest":
        for i in range(3):
            new_point[i,0] = np.round(point_fcoord[i]/target_fcoord[i])*target_fcoord[i] \
                                if not np.isinf(target_fcoord[i]) else point_fcoord[i]
    elif mode=="ceil":
        for i in range(3):
            new_point[i,0] = np.ceil(point_fcoord[i]/target_fcoord[i])*target_fcoord[i] \
                                if not np.isinf(target_fcoord[i]) else point_fcoord[i]
    elif mode == "floor":
        for i in range(3):
            new_point[i,0] = np.floor(point_fcoord[i]/target_fcoord[i])*target_fcoord[i] \
                                if not np.isinf(target_fcoord[i]) else point_fcoord[i]
    
    # scale back to real space
    new_point = generator.voxel.sbases @ new_point
    
    # get perpendicular vector
    normal = generator.lattice.reciprocal_lattice_crystallographic.matrix @ miller_indices
    return Plane(normal=normal, point=new_point)
