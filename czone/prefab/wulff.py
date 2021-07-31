from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms
from wulffpack import Decahedron, Icosahedron, SingleCrystal, Winterbottom
from wulffpack.core import BaseParticle

from ..generator.generator import Generator
from ..transform.strain import HStrain, IStrain
from ..transform.transform import MatrixTransform, Rotation, rot_v
from ..volume.algebraic import Cylinder, Plane, Sphere
from ..volume.volume import MultiVolume, Volume

"""
Submodule provides convenience constructors for returning Wulff constructions
as calculated and implemented by Wulffpack in their respective forms as 
Construction Zone volumes and generators
"""


class WulffBase(ABC):
    """Base class for Wulff constructions, as adapted from WulffPack.
    
    Wulff constructions are energetic constructions of single crystals. From a 
    list of surfaces and corresponding surface energies, the equilibirium shape
    minimizing the total surface energy for a given volume is determined. The
    equilbirium shape is looseley the inner surface of the intersection of the 
    supplied surfaces, where the supplied surfaces are translated from the origin
    along their normal directions by their relative surface energies.

    Attributes:
        generator (Generator): Crystalline generator.
        p_atoms (Atoms): Primitive structure of generator, as ASE Atoms object.
        natoms (int): Approximate size of particle in number of atoms.
        surface_energies (Dict[Tuple, float]): dictionary of surfaces and corresponding
                                            surface energies, where keys are
                                            tuples of miller indices and values
                                            are surfaces energies as floats
    """

    @abstractmethod
    def __init__(self):
        pass

    ####################
    #### Properties ####
    ####################

    @property
    def generator(self):
        """Generator object representing crystal system."""
        return self._generator

    @generator.setter
    def generator(self, val):
        assert (isinstance(val, Generator)
               ), "Supplied generator must be crystalline Generator class"
        self._generator = val
        p_struct = val.structure.get_primitive_structure()
        self._p_atoms = Atoms(positions=p_struct.cart_coords)

    @property
    def p_atoms(self):
        """Primitive structure of generator as ASE Atoms object."""
        return self._p_atoms

    @property
    def natoms(self):
        """Size of particle in approximate number of atoms."""
        return self._natoms

    @natoms.setter
    def natoms(self, val):
        try:
            self._natoms = int(val)
        except:
            raise TypeError("natoms must be convertible to integer")

    @property
    def surface_energies(self):
        """Dictionary of surface: surface energy pairs."""
        return self._surface_energies

    @surface_energies.setter
    def surface_energies(self, vals):
        for key, val in vals.items():
            assert(isinstance(key, tuple) and isinstance(val, float)), \
                "surface energies must be passed as tuple pairs of miller indices and floats"

        self._surface_energies = vals

    ####################
    ###### Methods #####
    ####################

    @staticmethod
    def planes_from_wulff(wulff):
        """Return facets of a full Wulff construction as list of Planes.

        Args:
            wulff (BaseParticle): WulffPack particle.

        Returns:
            List of Planes representing facets of Wulff constructions.
        """
        return [
            Plane(f.normal, f.normal * f.distance_from_origin)
            for f in wulff._yield_facets()
        ]

    @staticmethod
    def planes_from_form(wulff_form):
        """Return facets of a full Wulff construction as list of Planes.

        Args:
            wulff (BaseParticle): WulffPack particle.

        Returns:
            List of Planes representing facets of Wulff constructions.
        """
        return [
            Plane(f.normal, f.normal * f.distance_from_origin)
            for f in wulff_form.facets
        ]

    @abstractmethod
    def get_construction(self):
        """For a given crystal and set of surface energies, return a Wulff construction.

        Returns:
            Volume or MultiVolume with planes and generator with object prescribed 
            by given Wulff Construction.
        """
        pass


class WulffSingle(WulffBase):

    def __init__(self,
                 generator: Generator,
                 surface_energies: dict,
                 natoms: int = 1000):
        self.generator = generator
        self.natoms = natoms
        self.surface_energies = surface_energies

    ####################
    #### Properties ####
    ####################

    ####################
    ###### Methods #####
    ####################

    def get_construction(self):
        wulff = SingleCrystal(self.surface_energies, self.p_atoms, self.natoms)
        planes = self.planes_from_wulff(wulff)
        return Volume(alg_objects=planes, generator=self.generator)

    def winterbottom(self, interface: Tuple[int], interface_energy: float):
        """Construct Winterbottom construction for single crystals.
        
        A Winterbottom construction is analagous to a Wulff construction, in that
        it is the shape for a single crystal particle that minimizes the surface
        energy when the particle is in contact with another surface.

        Args:
            interface (Tuple[int]): Miller indices of interfacial surface
            interface_energy: Relative surface energy of interface. Must be 
                            less than or equal to half of the smallest surface
                            energy for the other surfaces.

        Returns:
            Volume with shape described by Winterbottom construction for given 
            surfce energies.
        """
        wulff = Winterbottom(self.surface_energies, interface, interface_energy,
                             self.p_atoms, self.natoms)
        planes = self.planes_from_wulff(wulff)

        return Volume(alg_objects=planes, generator=self.generator)

    @classmethod
    def cube(cls, generator: Generator, natoms: int = 1000):
        """Convenience constructor for cubic nanoparticles with Wulff construction.
        
        Args:
            generator (Generator): crystalline system for Wulff construction.
            natoms (int): approximate size of particle in atoms.

        Returns:
            Wulff construction for a cubic nanoparticle.
        """
        energies = {(1, 0, 0): 1.0}
        wulff = cls(generator, natoms)
        return wulff.get_construction(energies)

    @classmethod
    def cuboctohedron(cls, generator: Generator, natoms: int = 1000):
        """Convenience constructor for cuboctohedra; nanoparticles with Wulff construction.

        Args:
            generator (Generator): crystalline system for Wulff construction.
            natoms (int): approximate size of particle in atoms.

        Returns:
            Wulff construction for a cuboctohedral nanoparticle.
        """
        energies = {(1, 0, 0): 1.0, (1, 1, 1): 1.15}
        wulff = cls(generator, natoms)
        return wulff.get_construction(energies)

    @classmethod
    def octohedron(cls, generator: Generator, natoms: int = 1000):
        """Convenience constructor for octohedral nanoparticles with Wulff construction.
        
        Args:
            generator (Generator): crystalline system for Wulff construction.
            natoms (int): approximate size of particle in atoms.

        Returns:
            Wulff construction for a octohedral nanoparticle.
        """
        energies = {(1, 1, 1): 1.0}
        wulff = cls(generator, natoms)
        return wulff.get_construction(energies)

    @classmethod
    def truncated_octohedron(cls, generator: Generator, natoms: int = 1000):
        """Convenience constructor for truncted octohedral nanoparticles with Wulff construction.
        
        Args:
            generator (Generator): crystalline system for Wulff construction.
            natoms (int): approximate size of particle in atoms.

        Returns:
            Wulff construction for a truncated octohedral nanoparticle.
        """
        energies = {(1, 0, 0): 1.1, (1, 1, 0): 1.15, (1, 1, 1): 1.0}
        wulff = cls(generator, natoms)
        return wulff.get_construction(energies)


"""
WulffPack's dechaedron and icosahedron class is not as clean to grab.
Generally, they form the nanoparticle twice, in independent steps.
For facet visualization, they generate the first grain's surfaces as facets,
then rotate the group of facets about the central axes. 

Since the number of rotations applied is known, and the grains are stored in order,
we can back out which facets (and which rotations) are applied to each grain.
This will yield us all (most) of the subvolumes we need for the particle,
minus the internal zone of the rotation axis.

For atomic generation, they work again with a single grain. They first remove
atoms on the twin boundaries, and store atoms within the grain/inside the fivefold axis separately.
That is, twin_indices contains twin boundaries, which contains five_old axis atoms, too.
They then strain the in grain atoms out, make rotated duplicates, and return the atoms only.

The five fold axis is just a stack of atoms along the center. We should be able to replicate this in CZ with a 
cylinder object, placed at the origin and with diameter equal to the atom spacing.
####################################################

Strategy for replication of Decahedron:
1) Get list of facets.
2) Prune list into the 5 separate grains, create subvolumes
4) Create strained generator, rotate it to create grain specific generators, add to subvolumes
5) Create subvolume for 5 fold axis based on Cylinder, add unstrained generator
6) Return MultiVolume object with mutually exclusive priorities

"""


class WulffDecahedron(WulffBase):
    """Prefab routine for constructing strained Decahedra for FCC systems.
    
    Decahedral nanoparticles cannot be created in a defect-free, strain-free manner;
    however, relatively simple, physically close models can be generated through
    simple strain and twin defect models with Wulff constructions defining 
    tetrahedral units. Construction algorithm is an adapatation of that found
    in WulffPack. 
    
    See L. D. Marks,  Modified Wulff constructions for twinned particles,
    J. Cryst. Growth 61, 556 (1983), doi: 10.1016/0022-0248(83)90184-7 for 
    background.

    Attributes:
        generator (Generator): Crystalline generator.
        p_atoms (Atoms): Primitive structure of generator, as ASE Atoms object.
        natoms (int): Approximate size of particle in number of atoms.
        surface_energies (Dict[Tuple, float]): dictionary of surfaces and corresponding
                                            surface energies, where keys are
                                            tuples of miller indices and values
                                            are surfaces energies as floats.
        twin_energy (float): surface energy of twin boundaries in system.
    """

    def __init__(self,
                 generator: Generator,
                 surface_energies: dict,
                 twin_energy: float,
                 natoms: int = 1000):
        self.generator = generator
        self.natoms = natoms
        self.surface_energies = surface_energies
        self.twin_energy = twin_energy

    @property
    def twin_energy(self):
        """Surface energy of twin boundary defects."""
        return self._twin_energy

    @twin_energy.setter
    def twin_energy(self, val):
        assert (isinstance(val, float)), "twin energy must be float"
        self._twin_energy = val

    def get_construction(self):
        wulff = Decahedron(self.surface_energies, self.twin_energy,
                           self.p_atoms, self.natoms)
        sf = wulff._get_dechedral_scale_factor()
        planes_lists = [[], [], [], [], []]
        for form in wulff.forms:
            planes = self.planes_from_form(form)
            len_p = len(planes)
            for i, p in enumerate(planes):
                # there will always be a multiple 5 planes in planes
                idx = i // (len_p // 5)
                planes_lists[idx].append(p)

        vols = []
        for plist in planes_lists:
            vols.append(Volume(alg_objects=plist))

        # create strained and rotated generators
        strain_field = HStrain(matrix=[sf, 1, 1, 0, 0, 0])

        for i in range(0, 5):
            rot_mat = rot_v([0, 0, 1], i * 2 * np.pi / 5)
            rot = Rotation(rot_mat)
            gen = self.generator.from_generator(transformation=[rot])
            gen.strain_field = strain_field
            vols[i].add_generator(gen)

        z_extremes = np.array([
            np.max(np.vstack(f.vertices)[:, 2]) for f in wulff._yield_facets()
        ])
        ff_axis_b = Plane(normal=[0, 0, -1], point=[0, 0, np.min(z_extremes)])
        ff_axis_t = Plane(normal=[0, 0, 1], point=[0, 0, np.max(z_extremes)])
        ff_axis_cyl = Cylinder(radius=1e-3)
        vols.append(
            Volume(alg_objects=[ff_axis_b, ff_axis_t, ff_axis_cyl],
                   generator=self.generator))

        return MultiVolume(volumes=vols)


class WulffIcosahedron(WulffDecahedron):
    """Prefab routine for constructing strained Decahedra for FCC systems.
    
    Icosahedral nanoparticles cannot be created in a defect-free, strain-free manner;
    however, relatively simple, physically close models can be generated through
    simple strain and twin defect models with Wulff constructions defining 
    tetrahedral units. Construction algorithm is an adapatation of that found
    in WulffPack. 
    
    See L. D. Marks,  Modified Wulff constructions for twinned particles,
    J. Cryst. Growth 61, 556 (1983), doi: 10.1016/0022-0248(83)90184-7 for 
    background.

    Attributes:
        generator (Generator): Crystalline generator.
        p_atoms (Atoms): Primitive structure of generator, as ASE Atoms object.
        natoms (int): Approximate size of particle in number of atoms.
        surface_energies (Dict[Tuple, float]): dictionary of surfaces and corresponding
                                            surface energies, where keys are
                                            tuples of miller indices and values
                                            are surfaces energies as floats
        twin_energy (float): surface energy of twin boundaries in system.
    """

    @property
    def icosahedral_scale_factor(self):
        k = (5 + np.sqrt(5)) / 8
        return np.sqrt(2 / (3 * k - 1))

    @staticmethod
    def _strain_from_111(points, basis=np.eye(3).astype(float), sf=1):
        """Inhomogenous strain function to strain atoms away from [111] axis.
        
        Args:
            points (np.ndarray): Nx3 array of points
            basis (np.ndarray): 3x3 array representing crystal basis vectors
            sf (float): scale factor for icosahedral particle.

        Returns:
            np.ndarray: Nx3 set of strained points
        """
        vec_111 = basis @ np.array([[1.0, 1.0, 1.0]]).T
        vec_111 /= np.linalg.norm(vec_111)

        disp_vec = points - np.dot(points, vec_111) * vec_111.T
        return points + (sf - 1.0) * disp_vec

    def get_construction(self):
        wulff = Icosahedron(self.surface_energies, self.twin_energy,
                            self.p_atoms, self.natoms)

        sf = wulff._get_icosahedral_scale_factor()
        planes_lists = [[] for x in range(20)]

        for form in wulff.forms:
            planes = self.planes_from_form(form)
            len_p = len(planes)
            for i, p in enumerate(planes):
                # there will always be a multiple 20 planes in planes
                idx = i // (len_p // 20)
                planes_lists[idx].append(p)

        vols = []
        for plist in planes_lists:
            vols.append(Volume(alg_objects=plist))

        # get strained and rotated generators for the 20 tetrahedral grains
        strain_field = IStrain(fun=self._strain_from_111(),
                               sf=self.icosahedral_scale_factor())

        sym_mats = [MatrixTransform(np.eye(3))] + \
                    [MatrixTransform(m) for m in wulff._get_all_symmetry_operations()]

        for i, m in enumerate(sym_mats):
            gen = self.generator.from_generator(transformation=m)
            gen.strain_field = strain_field

            # setting an ordered priority might ensure that the fivefold axes, twin boundaries aren't duplicated
            # TODO: test this against wulffpack implementation
            vols[i].add_generator(gen)
            vols[i].priority = i

        central_sphere = Volume(alg_objects=Sphere(radius=1.0),
                                generator=self.generator,
                                priority=-1)
        vols.append(central_sphere)

        return MultiVolume(volumes=vols)
