"""
Volume Class
Luis Rangel DaCosta
"""
from scipy.spatial import ConvexHull
from ..generator import Generator
import numpy as np
import copy

class Volume():

    def __init__(self, points=None, generator=None):
        """
        points is N x 3 numpy array of coordinates (x,y,z)
        Default orientation of a volume is aligned with global orthonormal system
        """
        self._points = None
        self._hull = None
        self._orientation = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        self._generator = generator
        self._atoms = None

        if points is None:
            return
        else:
            #expect 2D array with Nx3 points
            assert(len(points.shape) == 2), "points must be N x 3 numpy array (x,y,z)"
            assert(points.shape[1] == 3), "points must be N x 3 numpy array (x,y,z)"
            self.addPoints(points)

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
    def hull(self):
        return self._hull

    @property
    def centroid(self):
        """
        Return heuristic centroid-- works okay if convex hull has points well distributed on surface.
        Used to track translations and apply rotations if no center is specified.
        """
        try:
            return np.mean(self.hull.points)
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

    @generator.setter
    def generator(self, generator):
        if not isinstance(generator, Generator):
            raise TypeError("Supplied generator is not of Generator() class")
        
        self._generator = generator
    
    @property
    def atoms(self):
        return self._atoms

    @property
    def species(self):
        return self._species

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

    def translate(self, vec):
        """
        expects translation vector as 1x3 numpy array
        """
        assert(self._points.size > 0), "No points to translate"
        self._points += vec
        self.createHull() #implcitly update hull, since can't transform points directly
    
    def rotate(self, R, center=None, locked=False):
        if center is None:
            self._points = np.dot(R, (self._points-self.centroid).T).T+self.centroid
        else:
            self._points = np.dot(R, (self._points-center).T).T+center

        if locked:
            self.generator.rotate(R)

        self.createHull()

    def checkIfInterior(self,testPoints):
        """
        TODO: adopt interiority test used in manipulatt
        Checks if any point in testPoints lies within convex hull
        testPoints should be Nx3 numpy array
        """
        assert(testPoints.shape[-1]==3), "testPoints must be N x 3 numpy array (x,y,z)"
        assert(len(testPoints.shape)<3), "testPoints must be N x 3 numpy array (x,y,z)"
        if(len(testPoints.shape)==1):
            testPoints = np.expand_dims(testPoints,axis=0)

        for point in testPoints:
            point = np.expand_dims(point,axis=0)
            testPoints = np.append(self._hull.points,point,axis=0)
            testHull = ConvexHull(testPoints)
            #if vertice list is exactly the same, then the new point should be interior to hull
            #or on one of its simplices
            if np.array_equal(testHull.vertices,self.hull.vertices):
                return True
            
        return False

    def get_bounding_box(self):
        return self.points

    def populate_atoms(self):
        """
        Fill bounding space with atoms then remove atoms falling outside 
        """
        bbox = self.get_bounding_box()
        coords, species = self.generator.supply_atoms(bbox)
        check = np.zeros(coords.shape[0], dtype=np.bool_)
        for i in range(coords.shape[0]):
            check[i] = self.checkIfInterior(coords[i])
    
        self._atoms = coords[check,:]
        self._species = species[check]





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

def from_volume(orig, **kwargs):
    """
    Construct a volume from another volume
    **kwargs encodes relationship
    """
    if not isinstance(orig, Volume):
            raise TypeError("Supplied volume is not of Volume() class")

    new_volume = Volume(points=orig.points)
    if "generator" in kwargs.keys():
        new_volume.generator = kwargs["generator"]
    else:
        new_volume.generator = copy.deepcopy(orig.generator)
        
    if "translate" in kwargs.keys():
        new_volume.translate(kwargs["translate"])

    if "rotate" in kwargs.keys():
        new_volume.rotate(kwargs["rotate"], locked=True)

    return new_volume