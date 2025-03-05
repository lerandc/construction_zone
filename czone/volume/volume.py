from __future__ import annotations

import copy
from functools import reduce
from typing import TYPE_CHECKING, List

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from czone.types import BaseAlgebraic, BaseGenerator, BaseVolume
from czone.util.eset import EqualSet, array_set_equal

from .algebraic import Cylinder, Plane, Sphere
from .algebraic import get_bounding_box as get_bounding_box_planes

# if TYPE_CHECKING:
from czone.transform.transform import BaseTransform
############################
###### Volume Classes ######
############################


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

    def __init__(
        self,
        points: np.ndarray = None,
        alg_objects: np.ndarray = None,
        generator: BaseGenerator = None,
        priority: int = 0,
        tolerance: float = 1e-10,
        **kwargs,
    ):
        self._points = None
        self._hull = None
        self._generator = None
        self._atoms = None
        self._tri = None
        self._alg_objects = []
        self._priority = 0
        self._tolerance = tolerance

        if points is not None:
            # expect 2D array with Nx3 points
            assert len(points.shape) == 2, "points must be N x 3 numpy array (x,y,z)"
            assert points.shape[1] == 3, "points must be N x 3 numpy array (x,y,z)"
            self.addPoints(points)

        if generator is not None:
            if "gen_origin" in kwargs:
                self.add_generator(generator, kwargs["gen_origin"])
            else:
                self.add_generator(generator)

        self.priority = priority

        if alg_objects is not None:
            self.add_alg_object(alg_objects)

    def __repr__(self):
        args = (
            f"points={repr(self.points)}, ",
            f"alg_objects={repr(self.alg_objects)}, ",
            f"generator={repr(self.generator)}, ",
            f"priority={repr(self.priority)}, ",
            f"tolerance={repr(self.tolerance)}",
        )

        return f"Volume({reduce(lambda x, y: x+y, args)})"

    def __eq__(self, other):
        # TODO: For now, this only checks set equivalance of the properties
        # In the future, should also reduce to a minimum convex set
        # e.g., if V1 has Sphere(5, np.zeros(3)) and Sphere(2, np.zeros(3))
        # and V2 2 has only Sphere(2, np.zeros(3)), then V1 == V2 -> they define the same space

        if isinstance(other, Volume):
            # TODO: check against the convex hull instead
            ## use hard ands instead of reduce over properties to short circuit

            points_check = (self.points is None and other.points is None) or array_set_equal(
                self.points, other.points
            )
            check = (
                self.generator == other.generator
                and points_check
                and EqualSet(self.alg_objects) == (EqualSet(other.alg_objects))
                and self.priority == other.priority
                and np.isclose(self.tolerance, other.tolerance)
            )

            return check
        else:
            return False

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
            self._points = None  # clear points
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
        try:
            ob_iter = iter(obj)
        except TypeError:
            ob_iter = iter([obj])

        ob_copies = []
        for ob in ob_iter:
            if not isinstance(ob, BaseAlgebraic):
                raise TypeError("Must be adding algebraic objects from derived BaseAlgebraic class")
            ob_copies.append(copy.deepcopy(ob))

        self._alg_objects.extend(ob_copies)

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

    @property
    def tolerance(self):
        """Numerical tolerance for simplex checking with convex hulls. Defaults to 1e-10"""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, val):
        assert isinstance(val, float)
        self._tolerance = val

    def add_generator(self, generator: BaseGenerator, origin=None):
        try:
            new_generator = generator.from_generator()
        except AttributeError:
            raise TypeError("Supplied generator is not of Generator() class")

        self._generator = new_generator

    """
    Methods
    """

    def createHull(self):
        """Create convex hull from points defining volume boundaries."""
        # check to make sure there are N>3 points in point list
        assert self.points.shape[0] > 3, "must have more than 3 points to create hull"
        self._hull = ConvexHull(self.points, incremental=True)
        self._tri = Delaunay(self.hull.points[self.hull.vertices])

    def addPoints(self, points: np.ndarray):
        """Add points to list of points defining convex hull and update hull.

        Args:
            points (np.ndarray): Nx3 array of points to add to hull.
        """
        assert points.shape[-1] == 3, "points must be N x 3 numpy array (x,y,z)"
        assert len(points.shape) < 3, "points must be N x 3 numpy array (x,y,z)"

        if self._points is None:
            self._points = np.copy(points)
            if len(points.shape) == 1:  # add dim if only single new point
                self._points = np.expand_dims(points)
        else:
            if len(points.shape) == 1:  # check for single point
                points = np.expand_dims(points, axis=0)

            self._points = np.append(self._points, points, axis=0)

        # if hull created, update points; else, create hull
        try:
            self._hull.add_points(points)
        except AttributeError:
            self.createHull()

    def transform(self, transformation: BaseTransform):
        if self.points is not None:
            try:
                self.points = transformation.applyTransformation(self.points)
            except AttributeError:
                raise TypeError("Supplied transformation not transformation object.")

            self.createHull()

        if len(self.alg_objects) > 0:
            try:
                for i, obj in enumerate(self.alg_objects):
                    self.alg_objects[i] = transformation.applyTransformation_alg(obj)
            except AttributeError:
                raise TypeError("Supplied transformation not transformation object.")

        if transformation.locked and (self.generator is not None):
            self.generator.transform(transformation)

    def checkIfInterior(self, testPoints: np.ndarray):
        assert testPoints.shape[-1] == 3, "testPoints must be N x 3 numpy array (x,y,z)"
        assert len(testPoints.shape) < 3, "testPoints must be N x 3 numpy array (x,y,z)"
        if len(testPoints.shape) == 1:
            testPoints = np.expand_dims(testPoints, axis=0)

        check = np.ones(testPoints.shape[0]).astype(bool)

        if self.tri is not None:
            check = np.logical_and(
                check, self.tri.find_simplex(testPoints, tol=self.tolerance) >= 0
            )

        if len(self.alg_objects) > 0:
            for obj in self.alg_objects:
                check = np.logical_and(check, obj.checkIfInterior(testPoints))

        return check

    def get_bounding_box(self):
        """Get some minimal bounding box defining extremities of regions.

        Returns:
            Nx3 array of points defining extremities of region enclosed by volume.
        """
        if self.points is not None:
            return self.points
        else:
            # As heuristic, look for any sphere first
            # Then, gather planes and check if valid intersection exists
            # Then, look for spheres
            # To improve, check all objects and select smalllest bounding box
            spheres = [obj for obj in self.alg_objects if isinstance(obj, Sphere)]
            if len(spheres) > 0:
                s_idx = sorted(range(len(spheres)), key=lambda i: spheres[i].radius)[0]
                d = 2 * spheres[s_idx].radius
                bbox = makeRectPrism(d, d, d)
                shift = spheres[s_idx].center - (d / 2) * np.ones(3)
                return bbox + shift

            planes = [obj for obj in self.alg_objects if isinstance(obj, Plane)]
            if len(planes) > 3:
                bbox, status = get_bounding_box_planes(planes)
                if status == 0:
                    return bbox

            cylinders = [obj for obj in self.alg_objects if isinstance(obj, Cylinder)]
            if len(cylinders) > 0:
                bbox = cylinders[0].get_bounding_box()
                return bbox

    def populate_atoms(self, **kwargs):
        bbox = self.get_bounding_box()
        coords, species = self.generator.supply_atoms(bbox, **kwargs)
        check = self.checkIfInterior(coords)

        self._atoms = coords[check, :]
        self._species = species[check]

    # TODO: let user raise warnings if te object is the same
    def from_volume(self, **kwargs):
        """Constructor for new Volumes based on existing Volume object.

        Args:
            **kwargs:
                    - transformation=List[BaseTransformation] to apply a series
                     of transfomrations to copied Volume.
                    - generator=BaseGenerator to replace generator associated with volume.
                    - Any kwargs accepted in creation of Volume object.
        """
        new_volume = Volume(
            points=self.points, alg_objects=self.alg_objects, priority=self.priority
        )
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
        if volumes is not None:
            self.add_volume(volumes)

        if priority is not None:
            self.priority = priority

    def __repr__(self):
        volume_substr = reduce(lambda x, y: x + y, [f"{repr(v)}, " for v in self.volumes])
        return f"MultiVolume(volumes=[{volume_substr}], priority={repr(self.priority)})"

    def __eq__(self, other):
        if isinstance(other, MultiVolume):
            return self.priority == other.priority and (
                EqualSet(self.volumes) == EqualSet(other.volumes)
            )
        else:
            return False

    @property
    def volumes(self):
        """Collection of volumes grouped in MultiVolume."""
        return self._volumes

    def add_volume(self, volume: BaseVolume):
        """Add volume to MultiVolume.

        Args:
            volume (BaseVolume): Volume object to add to MultiVolume.
        """
        if hasattr(volume, "__iter__"):
            for v in volume:
                assert isinstance(v, BaseVolume), "volumes must be volume objects"
            self._volumes.extend(volume)
        else:
            assert isinstance(volume, BaseVolume), "volumes must be volume objects"
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
        assert isinstance(
            transformation, BaseTransform
        ), "Supplied transformation not transformation object."

        for vol in self.volumes:
            vol.transform(transformation)

    def checkIfInterior(self, testPoints: np.ndarray):
        assert testPoints.shape[-1] == 3, "testPoints must be N x 3 numpy array (x,y,z)"
        assert len(testPoints.shape) < 3, "testPoints must be N x 3 numpy array (x,y,z)"
        if len(testPoints.shape) == 1:
            testPoints = np.expand_dims(testPoints, axis=0)

        check = np.zeros(testPoints.shape[0]).astype(bool)

        for vol in self.volumes:
            check = np.logical_or(check, vol.checkIfInterior(testPoints))

        return check

    def populate_atoms(self):
        # routine is modified form of scene atom population
        for vol in self.volumes:
            vol.populate_atoms()

        rel_plevels, offsets = self._get_priorities()

        checks = []

        for i, vol in enumerate(self.volumes):
            check = np.ones(vol.atoms.shape[0]).astype(bool)
            eidx = offsets[rel_plevels[i] + 1]

            for j in range(eidx):
                if i != j:
                    check_against = np.logical_not(self.volumes[j].checkIfInterior(vol.atoms))
                    check = np.logical_and(check, check_against)

            checks.append(check)

        self._atoms = np.vstack([vol.atoms[checks[i], :] for i, vol in enumerate(self.volumes)])
        self._species = np.hstack([vol.species[checks[i]] for i, vol in enumerate(self.volumes)])

    def get_bounding_box(self):
        """Return union of bounding boxes. TODO: Update to convex hull"""

        bboxes = np.concatenate([vol.get_bounding_box() for vol in self.volumes])
        return bboxes

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
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    # stretch unit cube
    points *= np.squeeze(np.array([a, b, c]))

    if center is None:
        return points
    else:
        # translate prism to desired center if specified
        cur_center = np.mean(points, axis=0)
        return points + (center - cur_center)
