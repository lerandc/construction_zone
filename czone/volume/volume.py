"""
Volume Class
Luis Rangel DaCosta
"""
from .algebraic import BaseAlgebraic, Plane, Sphere
from .algebraic import get_bounding_box as get_bounding_box_planes
from ..generator import BaseGenerator, AmorphousGenerator
from ..transform import BaseTransformation
from abc import ABC, abstractmethod
from scipy.spatial import ConvexHull, Delaunay

import numpy as np
import copy

############################
###### Volume Classes ######
############################

class BaseVolume(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @property
    def atoms(self):
        return self._atoms

    @property
    def species(self):
        return self._species

    @property
    def priority(self):
        return self._priority
    
    @priority.setter
    def priority(self, priority):
        if not isinstance(priority, int):
            raise TypeError("Priority needs to be integer valued")

        self._priority = priority

    @abstractmethod
    def transform(self, transformation):
        pass

    @abstractmethod
    def populate_atoms(self):
        pass

    @abstractmethod
    def checkIfInterior(self):
        pass


class Volume(BaseVolume):

    def __init__(self, points=None, alg_objects=None, generator=None, priority=None, **kwargs):
        """
        points is N x 3 numpy array of coordinates (x,y,z)
        Default orientation of a volume is aligned with global orthonormal system
        """
        self._points = None
        self._hull = None
        self._orientation = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        self._generator = None
        self._atoms = None
        self._tri = None
        self._alg_objects = []
        self._priority = 0

        if not (points is None):
            #expect 2D array with Nx3 points
            assert(len(points.shape) == 2), "points must be N x 3 numpy array (x,y,z)"
            assert(points.shape[1] == 3), "points must be N x 3 numpy array (x,y,z)"
            self.addPoints(points)

        if not (generator is None):
            if 'gen_origin' in kwargs:
                self.add_generator(generator, kwargs["gen_origin"])
            else:
                self.add_generator(generator)

        if not (priority is None):
            self.priority = priority

        if not (alg_objects is None):
            for obj in alg_objects:
                self.add_alg_object(obj)

    """
    Properties
    """
    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        try:
            self._points = np.array([]) #clear points
            self.addPoints(points)
        except AssertionError:
            raise ValueError("Check shape of input array.")

    @property
    def alg_objects(self):
        return self._alg_objects

    def add_alg_object(self, obj):
        assert(isinstance(obj, (BaseAlgebraic))), "Must be adding algebraic objects from derived BaseAlgebraic class"
        self._alg_objects.append(obj)
        
    @property
    def hull(self):
        return self._hull
    
    @property
    def tri(self):
        return self._tri

    @property
    def centroid(self):
        """
        Return heuristic centroid-- works okay if convex hull has points well distributed on surface.
        Used to track translations and apply rotations if no center is specified.
        """
        try:
            return np.mean(self.hull.points,axis=0)
        except AttributeError:
            raise AttributeError("No hull has been created.")
    
    @property
    def volume(self):
        """
        Return volume of convex hull defined by volume points
        """
        try:
            return (self.hull.volume)
        except AttributeError:
            raise AttributeError("No hull has been created.")

    @property
    def area(self):
        """
        Return surface area of convex hull defined by volume points
        """
        try:
            return (self.hull.area)
        except AttributeError:
            raise AttributeError("No hull has been created.")

    @property
    def orientation(self):
        """
        Return orientation of volume as orthonormal 3x3 matrix, representing
        current orientations of local X/Y/Z vectors.
        Each row is direction the original X/Y/Z now points, in terms of a 
        global othornomal system.
        """
        return self._orientation

    @property
    def generator(self):
        return self._generator

    def add_generator(self, generator, origin=None):
        if not isinstance(generator, BaseGenerator):
            raise TypeError("Supplied generator is not of Generator() class")
        
        new_generator = copy.deepcopy(generator)

        if not isinstance(generator, AmorphousGenerator):
            if not origin is None:
                new_generator.voxel.origin = origin
        
        self._generator = new_generator

    """
    Methods
    """
    def createHull(self):
        """
        Create convex hull of volume boundaries
        """
        #check to make sure there are N>3 points in point list
        assert(self.points.shape[0] > 3), "must have more than 3 points to create hull"
        self._hull = ConvexHull(self.points,incremental=True)
        self._tri = Delaunay(self.hull.points[self.hull.vertices])

    def addPoints(self, points):
        """
        expects points as an Nx3 numpy array 
        """
        assert(points.shape[-1]==3), "points must be N x 3 numpy array (x,y,z)"
        assert(len(points.shape)<3), "points must be N x 3 numpy array (x,y,z)"

        if(self._points is None):
            self._points = np.copy(points)
            if len(points.shape) == 1: #add dim if only single new point
                self._points = np.expand_dims(points)
        else:    
            if len(points.shape) == 1: #check for single point
                points = np.expand_dims(points,axis=0)
            
            self._points = np.append(self._points,points,axis=0)

        #if hull created, update points; else, create hull
        try:
            self._hull.add_points(points)
        except AttributeError:
            self.createHull()

    def transform(self, transformation):
        assert(isinstance(transformation, BaseTransformation)), "Supplied transformation not transformation object."

        if not(self.points is None):
            self.points = transformation.applyTransformation(self.points)
            self.createHull()

        if len(self.alg_objects) > 0:
            for i, obj in enumerate(self.alg_objects):
                self.alg_objects[i] = transformation.applyTransformation_alg(obj)

        if transformation.locked and (not (self.generator is None)):
            self.generator.transform(transformation)


    def checkIfInterior(self, testPoints):
        """
        Checks if points in testPoints lie within convex hull
        testPoints should be Nx3 numpy array
        """
        assert(testPoints.shape[-1]==3), "testPoints must be N x 3 numpy array (x,y,z)"
        assert(len(testPoints.shape)<3), "testPoints must be N x 3 numpy array (x,y,z)"
        if(len(testPoints.shape)==1):
            testPoints = np.expand_dims(testPoints,axis=0)

        check = np.ones(testPoints.shape[0]).astype(bool)

        if not self.tri is None:
            check = np.logical_and(check, self.tri.find_simplex(testPoints, tol=2.5e-1) >= 0)

        if len(self.alg_objects) > 0:
            for obj in self.alg_objects:
                check = np.logical_and(check, obj.checkIfInterior(testPoints))

        return check

    def get_bounding_box(self):
        if not(self.points is None):
            return self.points
        else:
            # As heuristic, look for any sphere first
            # Then, gather planes and check if valid intersection exists
            spheres = [obj for obj in self.alg_objects if isinstance(obj, Sphere)]
            if len(spheres) > 0:
                d = 2*spheres[0].radius
                bbox = makeRectPrism(d,d,d) 
                shift = spheres[0].center-(d/2)*np.ones(3)
                return  bbox + shift
            
            planes = [obj for obj in self.alg_objects if isinstance(obj, Plane)]
            if len(planes) > 3:
                return get_bounding_box_planes(planes)

    def populate_atoms(self):
        """
        Fill bounding space with atoms then remove atoms falling outside 
        """
        bbox = self.get_bounding_box()
        coords, species = self.generator.supply_atoms(bbox)
        check = self.checkIfInterior(coords)
    
        self._atoms = coords[check,:]
        self._species = species[check]

class MultiVolume(BaseVolume):

    def __init__(self, volumes=None):
        self._volumes = []
        if not (volumes is None):
            self.add_volume(volumes)

    """
    Propeties
    """
    @property
    def volumes(self):
        return self._volumes

    @property
    def atoms(self):
        return self._atoms

    """
    Methods
    """

    def add_volume(self, volume):
        if hasattr(volume, '__iter__'):
            for v in volume:
                assert(isinstance(v, BaseVolume)), "volumes must be volume objects"
            self._volumes.extend(volume)
        else:
            assert(isinstance(volume, BaseVolume)), "volumes must be volume objects"
            self._volumes.append(volume)

    def get_priorities(self):
        # get all priority levels active first
        self.volumes.sort(key=lambda ob: ob.priority)
        plevels = np.array([x.priority for x in self.volumes]) 

        # get unique levels and create relative priority array
        __, idx = np.unique(plevels, return_index=True)
        rel_plevels = np.zeros(len(self.volumes)).astype(int)
        for i in idx[1:]:
            rel_plevels[i:] += 1

        offsets = np.append(idx, len(self.volumes))

        return rel_plevels, offsets

    def transform(self, transformation):
        assert(isinstance(transformation, BaseTransformation)), "Supplied transformation not transformation object."

        for vol in self.volumes:
            vol.transform(transformation)

    def checkIfInterior(self, testPoints):
        assert(testPoints.shape[-1]==3), "testPoints must be N x 3 numpy array (x,y,z)"
        assert(len(testPoints.shape)<3), "testPoints must be N x 3 numpy array (x,y,z)"
        if(len(testPoints.shape)==1):
            testPoints = np.expand_dims(testPoints,axis=0)

        check = np.zeros(testPoints.shape[0]).astype(bool)

        for vol in self.volumes:
            check = np.logical_or(check, vol.checkIfInterior(testPoints))

        return check

    def populate_atoms(self):
        """
        routine is modified form of scene atom population
        """
        for vol in self.volumes:
            vol.populate_atoms()

        rel_plevels, offsets = self.get_priorities()

        checks = []

        for i, ob in enumerate(self.objects):
            check = np.ones(vol.atoms.shape[0]).astype(bool)
            eidx = offsets[rel_plevels[i]+1]

            for j in range(eidx):
                if(i != j):
                    check_against = np.logical_not(self.volumes[j].checkIfInterior(vol.atoms))
                    check = np.logical_and(check, check_against)

            checks.append(check)

        self._atoms = np.vstack([vol.atoms[checks[i],:] for i, vol in enumerate(self.volumes)])
        self._species = np.hstack([vol.species[checks[i]] for i, vol in enumerate(self.volumes)])



############################
#### Utility functions #####
############################


def checkCollisionHulls(volumeA, volumeB):
    """
    Expects two volume objects.
    Checks to see if any region is interior to two separate convex hulls.
    Uses interior point check with vertice list of Volume B
    to check for intersection of hulls. Checks both lists for 
    the case where one volume is entirely interior to the other.

    TODO: add tolerance, i.e., check shell around a volume
    """
    #check point interiority, i.e., if addition of point defines new vertices
    if volumeA.checkIfInterior(volumeB.points) or volumeB.checkIfInterior(volumeA.points):
        return True
    
    return False

def makeRectPrism(a,b,c,center=None):
    """
    Returns 8x3 numpy array of 8 points defining a rectangular prism in space.
    a,b,c are floats defining sidelength.
    center is midpoint of prism
    """
    points = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                        [1,1,0],[1,0,1],[0,1,1],[1,1,1]],dtype=np.float64)
    #stretch unit cube
    points *= np.array([a,b,c])

    if center is None:
        return points
    else:
        #translate prism to desired center if specified
        cur_center = np.mean(points,axis=0)
        return points + (center-cur_center)

def from_volume(orig: Volume, **kwargs) -> Volume:
    """
    Construct a volume from another volume
    **kwargs encodes relationship
    """
    if not isinstance(orig, Volume):
            raise TypeError("Supplied volume is not of Volume() class")

    new_volume = Volume(points=orig.points, alg_objects=orig.alg_objects)
    if "generator" in kwargs.keys():
        new_volume.generator = kwargs["generator"]
    else:
        new_volume.generator = copy.deepcopy(orig.generator)
        
    if "transformation" in kwargs.keys():
        for t in kwargs["transformation"]:
            new_volume.transform(t)

    return new_volume