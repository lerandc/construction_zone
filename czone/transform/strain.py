import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Callable
import copy

class BaseStrain(ABC):

    def __init__(self):
        self.origin = np.array([0,0,0])

    @abstractmethod
    def apply_strain(self, points):
        """
        F: R^3 -> R^3
        """
        pass

    def scrape_params(self, obj):
        """
        grab origin and bases from generator if needed
        """
        if self.mode == "crystal":
            self._bases = np.copy(obj.voxel.sbases)

        if self.origin == "generator":
            self.origin = np.copy(obj.origin)

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, val):
        assert(origin.shape == (3,)), "Origin must have shape (3,)"
        self._origin = np.array(origin)

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

class HStrain(BaseStrain):
    """
    Homogenous strain field
    Applies strain in crystal coordinates by default with respect to generator origin
    TODO: add species selectivity filter
    """

    def __init__(self, matrix=None, origin="generator", mode="crystal"):
        if not matrix is None:
            self.matrix = matrix
        else:
            # apply no strain
            self._matrix = np.eye(3)

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
        elif vals.shape == (6,):
            # voigt notation
            v = vals
            self._matrix = np.array([[v[0], v[5], v[4]], \
                                     [v[5], v[1], v[3]], \
                                     [v[4], v[3], v[2]]])
        else:
            raise ValueError("Input shape must be either 3,6, or 9 elements")

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
        sp += self.origin

        return sp


class IStrain(BaseStrain):
    """
    Inhomogenous strain field
    Applies strain in crystal coordinates by default with respect to generator origin
    User must input a custom strain function F: R^3 -> R^3 for np.arrays of shape (N,3)->(N,3)
    """
    def __init__(self, fun=None, origin="generator", mode="crystal", **kwargs):
        if not fun is None:
            self.strain_fun = fun
        else:
            # apply no strain
            self.strain_fun = lambda x: x 

        self.mode = mode

        if origin != "generator":
            self.origin = origin
        else:
            super.__init__()

        self._bases = None
        self.fun_kwargs = kwargs

    ##############
    # Properties #
    ##############
    
    @property
    def fun_kwargs(self):
        return self._fun_kwargs

    @fun_kwargs.setter
    def fun_kwargs(self, kwargs_dict):
        assert(isinstance(kwargs_dict, dict)), "Must supply dictionary for arbirtrary extra kwargs"
        self._fun_kwargs = kwargs_dict

    @property
    def strain_fun(self):
        return self._strain_fun

    @strain_fun.setter
    def strain_fun(self, fun) -> np.ndarray:
        """
        Strain function should take in nparray of shape (N,3) and arbitrary kwargs (including basis)
        """
        # def strain_fun(self, fun: Callable [[np.ndarray], np.ndarray]) -> np.ndarray:
        try:
            ref_arr = np.random.rand((100,3))
            test_arr = fun(ref_arr, **self.fun_kwargs)
            assert(test_arr.shape == (100,3))
        except AssertionError:
            raise ValueError("Strain function must return numpy arrays with shape (N,3) for input arrays of shape (N,3)")
        
        self._strain_fun = copy.deepcopy(fun)
        
    ##############
    ### Methods ##
    ##############

    def apply_strain(self, points):

        # get points relative to origin
        sp = np.copy(points)-self.origin

        if self.mode == "crystal":
            # project onto crystal coordinates 
            sp = sp @ np.linalg.inv(self.bases)

            # strain
            sp = self.strain_fun(sp, basis=self.bases, **self.fun_kwargs)

            # project back into real space
            sp = sp @ self.bases
        else:
            # strain
            sp = self.strain_fun(sp, **self.fun_kwargs)
        
        # shift back w.r.t. origin
        s_points += self.origin

        return sp