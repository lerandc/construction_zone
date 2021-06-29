import numpy as np
from abc import ABC, abstractmethod

# basically just want a short calss that gets attached to generators

def BaseStrain(ABC):

    def __init__(self):
        self.origin = np.array([0,0,0])

    @abstractmethod
    def apply_strain(self, points):
        """
        F: R^3 -> R^3
        """
        pass

    @abstractmethod
    def scrape_params(self, obj):
        """
        scrape necessary params from object applying strain to
        """
        pass

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, val):
        assert(origin.shape == (3,)), "Origin must have shape (3,)"
        self._origin = np.array(origin)

def HStrain(BaseStrain):
    """
    Homogenous strain field
    Applies strain in crystal coordinates by default with respect to generator origin
    No species selectivity
    """

    def __init__(self, matrix=None, origin="generator", mode="crystal"):
        if not matrix is None:
            self.matrix = matrix

        self.mode = mode

        if origin != "generator":
            self.origin = origin
        else:
            super.__init__()

        self._bases = None

    ##############
    # Properties #
    ##############

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, vals):
        vals = np.squeeze(np.array(vals))
        if vals.shape == (3,):
            self._matrix = np.eye(3)*vals
        elif vals.shape == (3,3):
            self._matrix = vals
        elif vals.shape == (9,):
            self._matrix = np.reshape(vals, (3,3))
        else:
            raise ValueError("Input shape must be either 3 or 9 elements")

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val):
        if mode == "crystal" or "standard":
            self._mode = val
        else:
            raise ValueError("Mode must be either crystal or standard")

    @property
    def bases(self):
        return self._bases

    ##############
    ### Methods ##
    ##############

    def apply_strain(self, points):

        # get points relative to origin
        sp = np.copy(points)-self.origin

        if self.mode == "crystal":
            # project onto crystal coordinates, strain, project back into real space
            sp = sp @ np.linalg.inv(self.bases) @ self.matrix @ self.bases
        else:
            # strain
            sp = sp @ self.matrix
        
        # shift back w.r.t. origin
        s_points += self.origin

        return s_points

    def scrape_params(self, obj):
        """
        grab origin and bases from generator if needed
        """
        if self.mode == "crystal":
            self._bases = np.copy(obj.voxel.sbases)

        if self.origin == "generator":
            self.origin = np.copy(obj.origin)



