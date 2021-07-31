from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class BaseStrain(ABC):
    """Base class for strain fields that act on Generators.

    Strain objects can be attached to generators, and transform the coordinates
    of the atoms post-generation of the supercell. Strain fields apply strain
    in crystal coordinate system by default.

    Attributes:
        origin (np.ndarray): origin with which respect coordinates are strained
        mode (str): "crystal" or "standard", for straining in crystal coordinates
                    or for straining coordinates with respect to standard R3
                    orthonromal basis and orientation, respectively
        bases (np.ndarray): 3x3 array representing generator basis vectors
    """

    def __init__(self):
        self.origin = np.array([0, 0, 0])

    @abstractmethod
    def apply_strain(self, points: np.ndarray) -> np.ndarray:
        """Apply strain to a collection of points.

        Args:
            points (np.ndarray): Nx3 array of points in space
        
        Returns:
            np.ndarray: Nx3 array of strained points in space
        """
        pass

    def scrape_params(self, obj: BaseGenerator):
        """Helper method to grab origin and bases from host generator.

        Args:
            obj (BaseGenerator): generator to grab parameters from
        """
        if self.mode == "crystal":
            self._bases = np.copy(obj.voxel.sbases)

        if self.origin == "generator":
            self.origin = np.copy(obj.origin)

    @property
    def origin(self):
        """Origin with respect to which strain is applied."""
        return self._origin

    @origin.setter
    def origin(self, val):
        assert (val.shape == (3,)), "Origin must have shape (3,)"
        self._origin = np.array(val)

    @property
    def mode(self):
        """Coordinate system for strain application, either 'crystal' or 'standard'."""
        return self._mode

    @mode.setter
    def mode(self, val):
        if val == "crystal" or "standard":
            self._mode = val
        else:
            raise ValueError("Mode must be either crystal or standard")

    @property
    def bases(self):
        """"Basis vectors of crystal coordinate system."""
        return self._bases


class HStrain(BaseStrain):
    """Strain class for applying homogeneous strain fields to generators.

    HStrain objects can be attached to generators, and transform the coordinates
    of the atoms post-generation of the supercell via simple strain tensor. 
    HStrain fields apply strain in crystal coordinate system by default.

    Attributes:
        matrix (np.ndarray): Matrix representing homogeneous strain tensor.
                            Can be set with 3 (x,y,z), 6 (Voigt notation), or
                            9 values (as list or 3x3 array).
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
        """Homogeneous strain tensor."""
        return self._matrix

    @matrix.setter
    def matrix(self, vals):
        vals = np.squeeze(np.array(vals))
        if vals.shape == (3,):
            self._matrix = np.eye(3) * vals
        elif vals.shape == (3, 3):
            self._matrix = vals
        elif vals.shape == (9,):
            self._matrix = np.reshape(vals, (3, 3))
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
    def apply_strain(self, points: np.ndarray) -> np.ndarray:

        # get points relative to origin
        sp = np.copy(points) - self.origin

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
    """Strain class for applying inhomogenous strain fields to generators.

    IStrain objects can be attached to generators, and transform the coordinates
    of the atoms post-generation of the supercell via arbitrary strain functions. 
    IStrain fields apply strain in crystal coordinate system by default.

    User must input a custom strain function; strain functions by default should
    accept only points as positional arguments and can take any kwargs.

    Attributes:
        fun_kwargs (dict): kwargs to pass to custom strain function
        strain_fun (Callable): strain function F: R3 -> R3 for 
                                np.arrays of shape (N,3)->(N,3)
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
        """kwargs passed to custom strain function upon application of strain."""
        return self._fun_kwargs

    @fun_kwargs.setter
    def fun_kwargs(self, kwargs_dict: dict):
        assert (isinstance(
            kwargs_dict,
            dict)), "Must supply dictionary for arbirtrary extra kwargs"
        self._fun_kwargs = kwargs_dict

    @property
    def strain_fun(self):
        """Inhomogenous strain function to apply to coordinates."""
        return self._strain_fun

    @strain_fun.setter
    def strain_fun(self, fun: Callable[[np.ndarray], np.ndarray]):
        try:
            ref_arr = np.random.rand((100, 3))
            test_arr = fun(ref_arr, **self.fun_kwargs)
            assert (test_arr.shape == (100, 3))
        except AssertionError:
            raise ValueError(
                "Strain function must return numpy arrays with shape (N,3) for input arrays of shape (N,3)"
            )

        self._strain_fun = copy.deepcopy(fun)

    ##############
    ### Methods ##
    ##############
    def apply_strain(self, points: np.ndarray) -> np.ndarray:

        # get points relative to origin
        sp = np.copy(points) - self.origin

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
        sp += self.origin

        return sp
