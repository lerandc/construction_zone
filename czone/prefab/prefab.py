import copy
import numpy as np
from abc import ABC, abstractmethod
from ..volume import Volume, MultiVolume
from ..volume.algebraic import Plane, snap_plane_near_point, from_volume
from ..generator import Generator, from_generator
from ..util.misc import get_N_splits
from ..transform import *

class BasePrefab(ABC):

    @abstractmethod
    def build_object(self):
        pass

class fccStackingFault(BasePrefab):
    
    def __init__(self, basis=None, species=None, a=1.0, N=1, plane=(1,1,1), volume=None, generator=None):

        self.basis = basis
        self.species = species
        self.a = a
        self.N = N
        self.plane = plane
        self.generator = generator
        self.volume = volume

    @property
    def basis(self):
        return self._basis

    @property
    def species(self):
        return self.species

    @property
    def a(self):
        return self._a
    
    @property
    def N(self):
        return self._N

    @property
    def plane(self):
        return self._plane
    
    """
    Methods
    """

    def _get_end_points(self, points, norm_vec):
        start = 0
        finish = 0

        return start, finish


    def build_object(self):
        """
        general algorithm:

        """

        # get list of all planes in bounding box
        # TODO: the bounding box isn't necessarily tangent to the valid volume (e.g., spheres)
        # perhaps refine the end points until planes intersect
        norm_vec = self.generator.voxel.bases @ np.array(self.plane)
        bbox = self.volume.get_bounding_box()
        ci = np.dot(norm_vec, bbox)
        start = bbox[np.argmin(ci),:]
        finish = bbox[np.argmax(ci),:]

        sp = snap_plane_near_point(start, self.generator, self.plane, mode="floor")
        ep = snap_plane_near_point(finish, self.generator, self.plane, mode="ceil")
        d_tot = ep.dist_from_plane(sp.point)
        d_hkl = self.generator.lattice.d_hkl(self.plane)
        N_planes = np.round(d_tot/d_hkl)
        planes = [sp]
        new_point = np.copy(sp.point)
        for i in range(N_planes):
            new_point += sp.normal * d_hkl
            planes.append(Plane(normal=sp.normal, point=new_point))

        # select N planes with min separation apart
        min_sep = 3
        splits = get_N_splits(self.N, min_sep, len(planes))

        burger = self.generator.voxel.sbases @ ((1/3)*np.array([1,1,-2])*np.sign(self.plane))

        # create sub volumes for final multivolume
        # origins should be successively shifted
        gen_tmp = from_generator(self.generator) 
        vols = [from_volume(self.volume, generator=gen_tmp)]
        for i in range(self.N):
            t = Translation(shift=(i+1)*burger)
            gen_tmp = from_generator(self.generator, transformation=t)
            new_vol = from_volume(self.volume, generator=gen_tmp)
            new_vol.add_alg_object(planes[splits[i]])
            new_vol.priority = i
            vols.append(new_vol)

        return  MultiVolume(volumes=vols)

class fccTwin(BasePrefab):
    
    def __init__(self, basis=None, species=None, a=1.0, N=1, plane=(1,1,1), volume=None, generator=None):

        self.basis = basis
        self.species = species
        self.a = a
        self.N = N
        self.plane = plane
        self.generator = generator
        self.volume = volume

    @property
    def basis(self):
        return self._basis

    @property
    def species(self):
        return self.species

    @property
    def a(self):
        return self._a
    
    @property
    def N(self):
        return self._N

    @property
    def plane(self):
        return self._plane
    
    """
    Methods
    """

    def build_object(self):
        """
        general algorithm:

        """

        # get list of all planes in bounding box
        # TODO: the bounding box isn't necessarily tangent to the valid volume (e.g., spheres)
        # perhaps refine the end points until planes intersect
        norm_vec = self.generator.voxel.bases @ np.array(self.plane)
        bbox = self.volume.get_bounding_box()
        ci = np.dot(norm_vec, bbox)
        start = bbox[np.argmin(ci),:]
        finish = bbox[np.argmax(ci),:]

        sp = snap_plane_near_point(start, self.generator, self.plane, mode="floor")
        ep = snap_plane_near_point(finish, self.generator, self.plane, mode="ceil")
        d_tot = ep.dist_from_plane(sp.point)
        d_hkl = self.generator.lattice.d_hkl(self.plane)
        N_planes = np.round(d_tot/d_hkl)
        planes = [sp]
        new_point = np.copy(sp.point)
        for i in range(N_planes):
            new_point += sp.normal * d_hkl
            planes.append(Plane(normal=sp.normal, point=new_point))

        # select N planes with min separation apart
        min_sep = 3
        splits = get_N_splits(self.N, min_sep, len(planes))

        # create sub volumes for final multivolume
        # origins should be successively shifted
        gen_tmp = from_generator(self.generator) 
        vols = [from_volume(self.volume, generator=gen_tmp)]
        for i in range(self.N):
            r = Reflection(planes[splits[i]])
            gen_tmp = from_generator(gen_tmp, transformation=r)
            new_vol = from_volume(self.volume, generator=gen_tmp)
            new_vol.add_alg_object(planes[splits[i]])
            new_vol.priority = i
            vols.append(new_vol)

        return  MultiVolume(volumes=vols)

class fccMixedTwinSF(BasePrefab):
    
    def __init__(self, basis=None, species=None, ratio=0.5, a=1.0, N=1, plane=(1,1,1), volume=None, generator=None):

        self.basis = basis
        self.species = species
        self.a = a
        self.N = N
        self.plane = plane
        self.generator = generator
        self.volume = volume

    @property
    def basis(self):
        return self._basis

    @property
    def species(self):
        return self.species

    @property
    def a(self):
        return self._a
    
    @property
    def N(self):
        return self._N

    @property
    def plane(self):
        return self._plane
    
    """
    Methods
    """

    def build_object(self):
        """
        general algorithm:

        """

        # get list of all planes in bounding box
        # TODO: the bounding box isn't necessarily tangent to the valid volume (e.g., spheres)
        # perhaps refine the end points until planes intersect
        norm_vec = self.generator.voxel.bases @ np.array(self.plane)
        bbox = self.volume.get_bounding_box()
        ci = np.dot(norm_vec, bbox)
        start = bbox[np.argmin(ci),:]
        finish = bbox[np.argmax(ci),:]

        sp = snap_plane_near_point(start, self.generator, self.plane, mode="floor")
        ep = snap_plane_near_point(finish, self.generator, self.plane, mode="ceil")
        d_tot = ep.dist_from_plane(sp.point)
        d_hkl = self.generator.lattice.d_hkl(self.plane)
        N_planes = np.round(d_tot/d_hkl)
        planes = [sp]
        new_point = np.copy(sp.point)
        for i in range(N_planes):
            new_point += sp.normal * d_hkl
            planes.append(Plane(normal=sp.normal, point=new_point))

        # select N planes with min separation apart
        min_sep = 3
        splits = get_N_splits(self.N, min_sep, len(planes))

        # create sub volumes for final multivolume
        # origins should be successively shifted
        gen_tmp = from_generator(self.generator) 
        vols = [from_volume(self.volume, generator=gen_tmp)]
        for i in range(self.N):
            if (np.random.rand() < self.ratio):
                burger = gen_tmp.voxel.sbases @ ((1/3)*np.array([1,1,-2])*np.sign(self.plane))
                t = Translation(shift=(i+1)*burger)
            else:
                t = Reflection(planes[splits[i]])
            gen_tmp = from_generator(gen_tmp, transformation=t)
            new_vol = from_volume(self.volume, generator=gen_tmp)
            new_vol.add_alg_object(planes[splits[i]])
            new_vol.priority = i
            vols.append(new_vol)

        return  MultiVolume(volumes=vols)

class SimpleGrainBoundary(BassePrefab):

    def __init__(self, z1, r1, z2=None, r2=None, plane=None, point=None, volume=None, generator=None):
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
        assert(z1.size==3), "Zone axes inputs must be 3-element list, tuple, or numpy array"
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
            assert(z2.size==3), "Zone axes inputs must be 3-element list, tuple, or numpy array"
            self._z2 = z2

    @property
    def r1()
    

    def build_object(self):
        """
        make two generator copies
        rotate to respective zone axes
        add split plane
        return multi volume
        """
        
        zv = np.array([0,0,1])
        z1v = self.generator.voxel.sbases @ self.z1
        z2v = self.generator.voxel.sbases @ self.z2

        rot_1 = Rotation(rot_vtv(z1v, zv))
        rot_2 = Rotation(rot_vtv(z2v, zv))
        gen_1 = from_generator(self.generator, transformation=rot_1)
        gen_2 = from_generator(self.generator, transformation=rot_2)

        if()

        vol_1 = from_volume(self.volume, generator=gen_1)
        vol_1.add_alg_object(self.plane)
        vol_2 = from_volume(self.volume, generator=gen_2)
        vol_2.priority += 1

        return MultiVolume(volumes=[vol_1, vol_2])