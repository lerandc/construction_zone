import copy
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from scipy.spatial import ConvexHull, Delaunay

from ..generator import AmorphousGenerator, BaseGenerator
from ..transform import BaseTransform
from .algebraic import BaseAlgebraic, Plane, Sphere
from .algebraic import get_bounding_box as get_bounding_box_planes

############################
###### Volume Classes ######
############################


class BaseVolume(ABC):
    """Base abstract class for Volume objects.
    
    Volume objects are subtractive components in Construction Zone. When designing
    nanostructures, Volumes contain information about where atoms should and
    should not be placed. Semantically, volumes can be thought of as singular
    objects in space.

    BaseVolumes are typically not created directly. Use the Volume class for
    generalized convex objects, and the MultiVolume class for unions of convex
    objects.

    Attributes:
        atoms (np.ndarray): Nx3 array of atom positions of atoms lying within volume.
        species (np.ndarray): Nx1 array of atomic numbers of atoms lying within volume.
        ase_atoms (Atoms): Collection of atoms in volume as ASE Atoms object.
        priority (int): Relative generation precedence of volume.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @property
    def atoms(self):
        """Array of atomic positions of atoms lying within volume."""
        return self._atoms

    @property
    def species(self):
        """Array of atomic numbers of atoms lying within volume."""
        return self._species

    @property
    def ase_atoms(self):
        """Collection of atoms in volume as ASE Atoms object."""
        return Atoms(symbols=self.species, positions=self.atoms)

    @property
    def priority(self):
        """Relative generation precedence of volume."""
        return self._priority

    @priority.setter
    def priority(self, priority):
        if not isinstance(priority, int):
            raise TypeError("Priority needs to be integer valued")

        self._priority = priority

    @abstractmethod
    def transform(self, transformation):
        """Transform volume with given transformation.

        Args:
            transformation (BaseTransform): transformation to apply to volume.
        """
        pass

    @abstractmethod
    def populate_atoms(self):
        """Fill volume with atoms. """
        pass

    @abstractmethod
    def checkIfInterior(self, testPoints: np.ndarray):
        """Check points to see if they lie in interior of volume.
        
        Returns:
            Logical array indicating which points lie inside the volume.
        """
        pass

    def to_file(self, fname, **kwargs):
        """Write object to an output file, using ASE write utilities.

        Args:
            fname (str): output file name.
            **kwargs: any key word arguments otherwise accepted by ASE write.
        """
        ase_write(filename=fname, images=self.ase_atoms, **kwargs)


class Volume(BaseVolume):
    """Volume object for representing convex spaces.
    
    Volume objects are subtractive components in Construction Zone. When designing
    nanostructures, Volumes contain information about where atoms should and
    should not be placed. Semantically, volumes can be thought of as singular
    objects in space. In order to supply atoms, Volumes must be given a Generator.

    Volumes can be created with a series of points in space, in which the 
    interior of the volume is taken as the convex hull of the points in space.
    They can also be created with a series of algebraic surfaces, such as planes
    and spheres. Both points and algebraic objects can be used to define a Volume,
    in which the interior of the Volume is taken as the intersection of the 
    interior region defined by the convex hull of the points and the interior
    regions of the algebraic objects.

    Attributes:
        points (np.ndarray): Nx3 array of points used to defined convex hull.
        alg_objects (List[BaseAlgebraic]): Algebraic objects used to define convex region.
        hull (ConvexHull): Convex hull of points defining volume.
        tri (Delaunay): Delaunay triangulation of facets of convex hull.
        generator (Generator): Generator object associated with volume that supplies atoms.
        atoms (np.ndarray): Nx3 array of atom positions of atoms lying within volume.
        species (np.ndarray): Nx1 array of atomic numbers of atoms lying within volume.
        ase_atoms (Atoms): Collection of atoms in volume as ASE Atoms object.
        priority (int): Relative generation precedence of volume.
    """

    def __init__(self,
                 points: np.ndarray = None,
                 alg_objects: np.ndarray = None,
                 generator: BaseGenerator = None,
                 priority: int = 0,
                 **kwargs):
        self._points = None
        self._hull = None
        self._generator = None
        self._atoms = None
        self._tri = None
        self._alg_objects = []
        self._priority = 0

        if not (points is None):
            #expect 2D array with Nx3 points
            assert (len(
                points.shape) == 2), "points must be N x 3 numpy array (x,y,z)"
            assert (points.shape[1] == 3
                   ), "points must be N x 3 numpy array (x,y,z)"
            self.addPoints(points)

        if not (generator is None):
            if 'gen_origin' in kwargs:
                self.add_generator(generator, kwargs["gen_origin"])
            else:
                self.add_generator(generator)

        self.priority = priority

        if not (alg_objects is None):
            for obj in alg_objects:
                self.add_alg_object(obj)

    """
    Properties
    """

    @property
    def points(self):
        """Nx3 array of points used to defined convex hull."""
        return self._points

    @points.setter
    def points(self, points):
        try:
            self._points = np.array([])  #clear points
            self.addPoints(points)
        except AssertionError:
            raise ValueError("Check shape of input array.")

    @property
    def alg_objects(self):
        """Algebraic objects used to define convex region."""
        return self._alg_objects

    def add_alg_object(self, obj: BaseAlgebraic):
        """Add an algebraic surface to the volume.
        
        Args:
            obj (BaseAlgebraic): Algebraic surface to add to volume.
        """
        assert (
            isinstance(obj, (BaseAlgebraic))
        ), "Must be adding algebraic objects from derived BaseAlgebraic class"
        self._alg_objects.append(copy.deepcopy(obj))

    @property
    def hull(self):
        """Convex hull of points defining volume."""
        return self._hull

    @property
    def tri(self):
        """Delaunay triangulation of facets of convex hull."""
        return self._tri

    @property
    def generator(self):
        """Generator object associated with volume that supplies atoms."""
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
        """Create convex hull from points defining volume boundaries."""
        #check to make sure there are N>3 points in point list
        assert (self.points.shape[0] >
                3), "must have more than 3 points to create hull"
        self._hull = ConvexHull(self.points, incremental=True)
        self._tri = Delaunay(self.hull.points[self.hull.vertices])

    def addPoints(self, points: np.ndarray):
        """Add points to list of points defining convex hull and update hull.

        Args:
            points (np.ndarray): Nx3 array of points to add to hull.
        """
        assert (
            points.shape[-1] == 3), "points must be N x 3 numpy array (x,y,z)"
        assert (len(points.shape) <
                3), "points must be N x 3 numpy array (x,y,z)"

        if (self._points is None):
            self._points = np.copy(points)
            if len(points.shape) == 1:  #add dim if only single new point
                self._points = np.expand_dims(points)
        else:
            if len(points.shape) == 1:  #check for single point
                points = np.expand_dims(points, axis=0)

            self._points = np.append(self._points, points, axis=0)

        #if hull created, update points; else, create hull
        try:
            self._hull.add_points(points)
        except AttributeError:
            self.createHull()

    def transform(self, transformation: BaseTransform):

        assert (isinstance(transformation, BaseTransform)
               ), "Supplied transformation not transformation object."

        if not (self.points is None):
            self.points = transformation.applyTransformation(self.points)
            self.createHull()

        if len(self.alg_objects) > 0:
            for i, obj in enumerate(self.alg_objects):
                self.alg_objects[i] = transformation.applyTransformation_alg(
                    obj)

        if transformation.locked and (not (self.generator is None)):
            self.generator.transform(transformation)

    def checkIfInterior(self, testPoints: np.ndarray):
        assert (testPoints.shape[-1] == 3
               ), "testPoints must be N x 3 numpy array (x,y,z)"
        assert (len(testPoints.shape) <
                3), "testPoints must be N x 3 numpy array (x,y,z)"
        if (len(testPoints.shape) == 1):
            testPoints = np.expand_dims(testPoints, axis=0)

        check = np.ones(testPoints.shape[0]).astype(bool)

        if not self.tri is None:
            check = np.logical_and(
                check,
                self.tri.find_simplex(testPoints, tol=2.5e-1) >= 0)

        if len(self.alg_objects) > 0:
            for obj in self.alg_objects:
                check = np.logical_and(check, obj.checkIfInterior(testPoints))

        return check

    def get_bounding_box(self):
        """Get some minimal bounding box defining extremities of regions.
        
        Returns:
            Nx3 array of points defining extremities of region enclosed by volume.
        """
        if not (self.points is None):
            return self.points
        else:
            # As heuristic, look for any sphere first
            # Then, gather planes and check if valid intersection exists
            spheres = [
                obj for obj in self.alg_objects if isinstance(obj, Sphere)
            ]
            if len(spheres) > 0:
                d = 2 * spheres[0].radius
                bbox = makeRectPrism(d, d, d)
                shift = spheres[0].center - (d / 2) * np.ones(3)
                return bbox + shift

            planes = [obj for obj in self.alg_objects if isinstance(obj, Plane)]
            if len(planes) > 3:
                return get_bounding_box_planes(planes)

    def populate_atoms(self):
        bbox = self.get_bounding_box()
        coords, species = self.generator.supply_atoms(bbox)
        check = self.checkIfInterior(coords)

        self._atoms = coords[check, :]
        self._species = species[check]

    def from_volume(self, **kwargs):
        """Constructor for new Volumes based on existing Volume object.

        Args:
            **kwargs: 
                    - transformation=List[BaseTransformation] to apply a series
                     of transfomrations to copied Volume.
                    - generator=BaseGenerator to replace generator associated with volume.
                    - Any kwargs accepted in creation of Volume object.
        """
        new_volume = Volume(points=self.points,
                            alg_objects=self.alg_objects,
                            priority=self.priority)
        if "generator" in kwargs.keys():
            new_volume.add_generator(kwargs["generator"])
        else:
            new_volume.add_generator(self.generator)

        if "transformation" in kwargs.keys():
            for t in kwargs["transformation"]:
                new_volume.transform(t)

        return new_volume


class MultiVolume(BaseVolume):
    """Volume object for representing arbitrary union of convex spaces.

    Volume objects are subtractive components in Construction Zone. When designing
    nanostructures, Volumes contain information about where atoms should and
    should not be placed. Semantically, volumes can be thought of as singular
    objects in space. In order to supply atoms, Volumes must be given a Generator.

    MultiVolumes group multiple Volume objects together into a single semantic object.
    Within the MultiVolume, Volume intersection is handled with relative precedence levels, 
    analagous to the precedence relationships that are used to handle conflict
    resolution between Volumes in scenes. Transformations applied to a MultiVolume
    are applied to every owned volume. MultiVolumes can be nested.

    Attributes:
        volumes (np.ndarray): Nx3 array of points used to defined convex hull.
        atoms (np.ndarray): Nx3 array of atom positions of atoms lying within volume.
        species (np.ndarray): Nx1 array of atomic numbers of atoms lying within volume.
        ase_atoms (Atoms): Collection of atoms in volume as ASE Atoms object.
        priority (int): Relative generation precedence of volume.
    """

    def __init__(self, volumes: List[BaseVolume] = None, priority: int = None):
        self._priority = 0
        self._volumes = []
        if not (volumes is None):
            self.add_volume(volumes)

        if not (priority is None):
            self.priority = priority

    @property
    def volumes(self):
        """Collection of volumes grouped in MultiVolume."""
        return self._volumes

    def add_volume(self, volume: BaseVolume):
        """Add volume to MultiVolume.
        
        Args:
            volume (BaseVolume): Volume object to add to MultiVolume.
        """
        if hasattr(volume, '__iter__'):
            for v in volume:
                assert (isinstance(
                    v, BaseVolume)), "volumes must be volume objects"
            self._volumes.extend(volume)
        else:
            assert (isinstance(volume,
                               BaseVolume)), "volumes must be volume objects"
            self._volumes.append(volume)

    def _get_priorities(self):
        """Grab priority levels of all volumes in MultiVolume to determine precedence relationship.

        Returns:
            List of relative priority levels and offsets. Relative priority levels
            and offsets are used to determine which objects whill be checked
            for the inclusion of atoms in the scene of the atoms contributed by
            another object.
        """

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

    def transform(self, transformation: BaseTransform):
        assert (isinstance(transformation, BaseTransform)
               ), "Supplied transformation not transformation object."

        for vol in self.volumes:
            vol.transform(transformation)

    def checkIfInterior(self, testPoints: np.ndarray):
        assert (testPoints.shape[-1] == 3
               ), "testPoints must be N x 3 numpy array (x,y,z)"
        assert (len(testPoints.shape) <
                3), "testPoints must be N x 3 numpy array (x,y,z)"
        if (len(testPoints.shape) == 1):
            testPoints = np.expand_dims(testPoints, axis=0)

        check = np.zeros(testPoints.shape[0]).astype(bool)

        for vol in self.volumes:
            check = np.logical_or(check, vol.checkIfInterior(testPoints))

        return check

    def populate_atoms(self):
        #routine is modified form of scene atom population
        for vol in self.volumes:
            vol.populate_atoms()

        rel_plevels, offsets = self._get_priorities()

        checks = []

        for i, vol in enumerate(self.volumes):
            check = np.ones(vol.atoms.shape[0]).astype(bool)
            eidx = offsets[rel_plevels[i] + 1]

            for j in range(eidx):
                if (i != j):
                    check_against = np.logical_not(
                        self.volumes[j].checkIfInterior(vol.atoms))
                    check = np.logical_and(check, check_against)

            checks.append(check)

        self._atoms = np.vstack(
            [vol.atoms[checks[i], :] for i, vol in enumerate(self.volumes)])
        self._species = np.hstack(
            [vol.species[checks[i]] for i, vol in enumerate(self.volumes)])

    def from_volume(self, **kwargs):
        """Constructor for new MultiVolume based on existing MultiVolume object.

        **kwargs passed to volume are applied to every owned Volume individually.

        Args:
            **kwargs: 
                    - transformation=List[BaseTransformation] to apply a series
                     of transfomrations to copied Volume.
                    - generator=BaseGenerator to replace generator associated with volume.
                    - Any kwargs accepted in creation of Volume object.
        """
        new_vols = []
        for vol in self.volumes:
            new_vols.append(vol.from_volume(**kwargs))

        return MultiVolume(volumes=new_vols, priority=self.priority)


############################
#### Utility functions #####
############################


def makeRectPrism(a, b, c, center=None):
    """Create rectangular prism.

    Args:
        a (float): dimension of prism along x
        b (float): dimension of prism along y
        c (float): dimension of prism along z
        center (np.ndarray): center of prism, default None. If None, corner of
                            prism is at origin. Else, prism is translated to 
                            have midpoint at center.

    Returns:
        8x3 numpy array of 8 points defining a rectangular prism in space.
    """
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                       [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                      dtype=np.float64)
    #stretch unit cube
    points *= np.array([a, b, c])

    if center is None:
        return points
    else:
        #translate prism to desired center if specified
        cur_center = np.mean(points, axis=0)
        return points + (center - cur_center)
