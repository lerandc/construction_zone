import copy
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from ..generator import Generator
from ..transform import *
from ..util.misc import get_N_splits
from ..volume import BaseVolume, MultiVolume, Volume
from ..volume.algebraic import Plane, snap_plane_near_point


class BasePrefab(ABC):
    """Base abstract class for Prefab objects.

    Prefab objects are objects and classes that can run predesigned algorithms
    for generating certain classes of regular objects, typically with 
    sampleable features or properties. For example, planar defects in FCC 
    systems are easily described in algorithmic form-- a series of {111} planes 
    can be chosen to put a defect on.

    Prefab objects will generally take in at least a base Generator object defining
    the system of interest and potentially take in Volume objects. They will return
    Volumes, or, more likely, MultiVolume objects which contains the resultant
    structure defined by the prefab routine.
    """

    @abstractmethod
    def build_object(self) -> BaseVolume:
        """Construct and return a prefabicated structure."""
        pass


class fccMixedTwinSF(BasePrefab):
    """Prefab routine for returning FCC volume with mixed stacking fault and twin defects.

    Given a volume with an attached generator (assumed to be FCC symmetry), 
    return a MultiVolume with a series of twin defects and stacking faults. 
    Defects can be placed on any plane in the {111} family, and are uniformly
    randomly distributed on the available (111) planes with a minimum separation.
    Types of defects are uniformly randomly sampled, as well, according to a ratio
    of twin defects to stacking faults.

    Attributes:
        N (int): Number of defects to attempt to place into volume
        plane (Tuple[int]): Length 3 tuple of Miller indices representing set of 
                            planes to put defects on
        ratio (float): Ratio of stacking faults:total defects. A ratio of 1.0 
                        will produce only stacking faults, while a ratio of 0.0
                        will produce only twin defects. 
        generator (Generator): Generator object representing FCC crystal system.
        volume (Volume): Volume object representing bounds of object in which
                        defects will be placed.

    """

    def __init__(self,
                 generator: Generator = None,
                 volume: BaseVolume = None,
                 ratio: float = 0.5,
                 N: int = 1,
                 min_sep: int = 3,
                 plane: Tuple[int] = (1, 1, 1)):
        self._N = None
        self._plane = None
        self._ratio = None
        self._min_sep = None
        self._generator = None
        self._volume = None

        self.N = N
        self.plane = plane
        self.ratio = ratio
        self.min_sep = min_sep

        if not generator is None:
            self.generator = generator

        if not volume is None:
            self.volume = volume

    @property
    def N(self):
        """Number of defects to place in volume."""
        return self._N

    @N.setter
    def N(self, val):
        self._N = int(val)

    @property
    def min_sep(self):
        """Minimum seperation between defects in numbers of planes."""
        return self._min_sep

    @min_sep.setter
    def min_sep(self, val: int):
        assert (
            isinstance(val, int)
        ), "Must supply integer number of planes for minimum seperation of defects."
        self._min_sep = val

    @property
    def plane(self):
        """Miller indices of defect planes."""
        return self._plane

    @plane.setter
    def plane(self, val: Tuple[int]):
        self._plane = val

    @property
    def ratio(self):
        """Ratio of stacking faults:total defects placed into object."""
        return self._ratio

    @ratio.setter
    def ratio(self, val: float):
        assert (0.0 <= val and
                val <= 1.0), "Ratio must be value in interval [0,1)."
        self._ratio = val

    @property
    def generator(self):
        """Crystalline geneartor used to sample planes for defects."""
        return self._generator

    @generator.setter
    def generator(self, val: Generator):
        assert (isinstance(
            val, Generator)), "Must supply crystalline Generator object."
        self._generator = val

    @property
    def volume(self):
        """Volume used to defined outer bounds of defected object."""
        return self._volume

    @volume.setter
    def volume(self, val: BaseVolume):
        assert (isinstance(
            val,
            BaseVolume)), "Must supply either Volume or MultiVolume object."
        self._volume = val

    def build_object(self):
        # get list of all planes in bounding box
        # TODO: the bounding box isn't necessarily tangent to the valid volume (e.g., spheres)
        # perhaps refine the end points until planes intersect
        norm_vec = self.generator.voxel.bases @ np.array(
            self.plane)  # fine to use real bases since cubic
        norm_vec /= np.linalg.norm(norm_vec)
        bbox = self.volume.get_bounding_box()
        ci = np.dot(norm_vec.reshape(1, 3), bbox.T).T
        start = bbox[np.argmin(ci), :]
        finish = bbox[np.argmax(ci), :]

        if isinstance(self.volume.alg_objects[0], Sphere):
            # override bbox in this special case
            start = -1.0 * self.volume.alg_objects[0].radius * norm_vec
            finish = self.volume.alg_objects[0].radius * norm_vec

        sp = snap_plane_near_point(start,
                                   self.generator,
                                   self.plane,
                                   mode="floor")
        ep = snap_plane_near_point(finish,
                                   self.generator,
                                   self.plane,
                                   mode="ceil")
        d_tot = ep.dist_from_plane(sp.point)
        d_hkl = self.generator.lattice.d_hkl(self.plane)
        N_planes = np.round(d_tot / d_hkl).astype(int)
        planes = [sp]
        new_point = np.copy(sp.point)
        for i in range(N_planes):
            new_point += sp.normal * d_hkl
            planes.append(Plane(normal=sp.normal, point=new_point))

        # select N planes with min separation apart
        splits = get_N_splits(self.N, self.min_sep, len(planes))
        splits.reverse()

        # create sub volumes for final multivolume
        # origins should be successively shifted
        gen_tmp = self.generator.from_generator()
        vols = [self.volume.from_volume(generator=gen_tmp)]
        vols[0].priority = self.N
        twin_last = 1
        for i in range(self.N):
            if (np.random.rand() < self.ratio):
                # add stacking fault
                burger = twin_last * gen_tmp.voxel.sbases @ (
                    (1 / 3) * np.array([1, 1, -2]) * np.sign(self.plane))
                t = Translation(shift=burger)
            else:
                # add twin defect
                t = Reflection(planes[splits[i]])
                twin_last *= -1
            gen_tmp = gen_tmp.from_generator(transformation=[t])
            new_vol = self.volume.from_volume(generator=gen_tmp)
            plane_tmp = Plane(normal=1.0 * planes[splits[i]].normal,
                              point=planes[splits[i]].point)
            new_vol.add_alg_object(plane_tmp)
            new_vol.priority = self.N - (i + 1)
            vols.append(new_vol)

        return MultiVolume(volumes=vols)


class fccStackingFault(fccMixedTwinSF):
    """Prefab routine for returning FCC volume with stacking fault defects.

    Given a volume with an attached generator (assumed to be FCC symmetry), 
    return a MultiVolume with a series of. stacking faults. Defects can be placed
    on any plane in the {111} family, and are uniformly randomly distributed on 
    the available (111) planes with a minimum separation.

    Attributes:
        N (int): Number of defects to attempt to place into volume
        plane (Tuple[int]): Length 3 tuple of Miller indices representing set of 
                            planes to put defects on/
        generator (Generator): Generator object representing FCC crystal system.
        volume (Volume): Volume object representing bounds of object in which
                        defects will be placed.
    """

    def __init__(self,
                 generator: Generator = None,
                 volume: BaseVolume = None,
                 N: int = 1,
                 min_sep: int = 3,
                 plane: Tuple[int] = (1, 1, 1)):
        super().__init__(generator=generator,
                         volume=volume,
                         N=N,
                         min_sep=min_sep,
                         plane=plane)

    @property
    def ratio(self):
        return 1.0


class fccTwin(fccMixedTwinSF):
    """Prefab routine for returning FCC volume with twin defects.

    Given a volume with an attached generator (assumed to be FCC symmetry), 
    return a MultiVolume with a series of. stacking faults. Defects can be placed
    on any plane in the {111} family, and are uniformly randomly distributed on 
    the available (111) planes with a minimum separation.

    Attributes:
        N (int): Number of defects to attempt to place into volume
        plane (Tuple[int]): Length 3 tuple of Miller indices representing set of 
                            planes to put defects on/
        generator (Generator): Generator object representing FCC crystal system.
        volume (Volume): Volume object representing bounds of object in which
                        defects will be placed.
    """

    def __init__(self,
                 generator: Generator = None,
                 volume: BaseVolume = None,
                 N: int = 1,
                 min_sep: int = 3,
                 plane: Tuple[int] = (1, 1, 1)):
        super().__init__(generator=generator,
                         volume=volume,
                         N=N,
                         min_sep=min_sep,
                         plane=plane)

    @property
    def ratio(self):
        return 0.0


class SimpleGrainBoundary(BasePrefab):
    """Prefab routine for crystalline grain boundaries.

    Under development.
    """

    def __init__(self,
                 z1,
                 r1,
                 z2=None,
                 r2=None,
                 plane=None,
                 point=None,
                 volume=None,
                 generator=None):
        """
        z1, z2 are zone axes for grains 1, 2
        r1 and r2 are phase about the zone axes, respectively
        plane is either a Plane object defining a split plane or miller indices of split plane, nearest center of volume
        """
        self.z1 = z1
        self.z2 = z2
        self.r1 = r1
        self.r2 = r2
        self.plane = plane
        self.volume = volume
        self.generator = generator

    @property
    def z1(self):
        return self._z1

    @z1.setter
    def z1(self, z1):
        z1 = np.array(z1)
        assert (
            z1.size == 3
        ), "Zone axes inputs must be 3-element list, tuple, or numpy array"
        self._z1 = z1

    @property
    def z2(self):
        return self._z2

    @z2.setter
    def z2(self, z2):
        if z2 is None:
            self._z2 = self.z1
        else:
            z2 = np.array(z2)
            assert (
                z2.size == 3
            ), "Zone axes inputs must be 3-element list, tuple, or numpy array"
            self._z2 = z2

    def build_object(self):

        zv = np.array([0, 0, 1])
        z1v = self.generator.voxel.sbases @ self.z1
        z2v = self.generator.voxel.sbases @ self.z2

        rot_1 = Rotation(rot_vtv(z1v, zv))
        rot_2 = Rotation(rot_vtv(z2v, zv))
        gen_1 = self.generator.from_generator(transformation=[rot_1])
        gen_2 = self.generator.from_generator(transformation=[rot_2])

        vol_1 = self.volume.from_volume(generator=gen_1)
        vol_1.add_alg_object(self.plane)
        vol_2 = self.volume.from_volume(generator=gen_2)
        vol_2.priority += 1

        return MultiVolume(volumes=[vol_1, vol_2])
