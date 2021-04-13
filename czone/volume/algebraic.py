from abc import ABC, abstractmethod
import numpy as np

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
    def getParams(self):
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
        return (testPoints-self.center)**2.0 < self.radius**2.0

    def getParams(self):
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
        assert(center.shape[0] == 3), "Center must be a point in 3D space"
        self._center = center

class Plane(BaseAlgebraic):

    def __init__(self, normal=None, point=None):
        self._normal = None
        self._point = None

            
        if not (normal is None):
            self.normal = normal

        if not (point is None):
            self.point = point

    def checkIfInterior(self, testPoints):
        return (testPoints*self.normal) + self.point > 0

    def getParams(self):
        return self.normal, self.point

    @property
    def point(self):
        return self._point

    @radius.setter
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
        assert(normal.shape[0] == 3), "normal must be a vector in 3D space"
        if(np.linalg.norm(normal) > np.finfo().eps):
            self._normal = normal
        else:
            raise ValueError("Normal vector must have some length")
            

#####################################
######### Utility routines ##########
#####################################