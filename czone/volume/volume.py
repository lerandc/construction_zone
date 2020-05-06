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
