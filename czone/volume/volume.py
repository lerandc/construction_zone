"""
Volume Class
Luis Rangel DaCosta May 6, 2020
"""
from scipy.spatial import ConvexHull
import numpy as np
class Volume():

    def __init__(self, points=None):
        if points is None:
            self.points = np.array([])
        else:
            #expect 2D array with Nx3 points
            assert(len(points.shape) == 2), "points must be N x 3 numpy array (x,y,z)"
            assert(points.shape[1] == 3), "points must be N x 3 numpy array (x,y,z)"
            self.points = points

    def createHull(self):
        """
        Create convex hull of volume boundaries
        """
        #check to make sure there are N>3 points in point list
        assert(self.points.shape[0] > 3), "must have at least 3 points to create hull"
        self.hull = ConvexHull(self.points,incremental=True)

    def addPoints(self, points):
        """
        expects points as an Nx3 numpy array 
        """
        assert(points.shape[-1]==3), "points must be N x 3 numpy array (x,y,z)"
        assert(len(points.shape)<3), "points must be N x 3 numpy array (x,y,z)"

        if(self.points.size == 0):
            self.points = points
            if len(points.shape) == 1: #add dim if only single new point
                self.points = np.expand_dims(points)
        else:    
            if len(points.shape) == 1: #check for single point
                points = np.expand_dims(points,axis=0)
            
            self.points = np.append(self.points,points,axis=0)

        #if hull created, update points
        try:
            self.hull.add_points(points)
        except AttributeError:
            return

    def translate(self, vec):
        """
        expects translation vector as 1x3 numpy array
        """
        assert(self.points.size > 0), "No points to translate"
        self.points = self.points + vec
        
        try:
            self.hull.points += vec #translate hull if it exists
        except ValueError:
            print("entered value error")
            return
        except AttributeError:
            print("entered attribute error")
            return

    def checkIfInterior(self,testPoints):
        """
        Checks if any point in testPoints lies within convex hull
        testPoints should be Nx3 numpy array
        """
        assert(testPoints.shape[-1]==3), "testPoints must be N x 3 numpy array (x,y,z)"
        assert(len(testPoints.shape)<3), "testPoints must be N x 3 numpy array (x,y,z)"
        if(len(testPoints.shape)==1):
            testPoints = np.expand_dims(testPoints,axis=0)

        for point in testPoints:
            point = np.expand_dims(point,axis=0)
            testPoints = np.append(self.hull.points,point,axis=0)
            testHull = ConvexHull(testPoints)
            #if vertice list is exactly the same, then the new point should be interior to hull
            #or on one of it's simplices
            if np.array_equal(testHull.vertices,self.hull.vertices):
                return True
            
        
        return False

    def checkIfCoplanar(self,testPoints):
        """
        Checks if any point in testPoints lies on a face 
        """

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