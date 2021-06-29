from wulffpack.core import wp_BaseParticle
from wulffpack import wp_SingleCrystal wp_Winterbottom
from ..volume.volume import Volume, MultiVolume
from ..generator.generator import Generator
from ..volume.algebraic import Plane
from ase import Atoms
from abc import ABC, abstractmethod

"""
Submodule provides convenience constructors for returning Wulff constructions
as calculated and implemented by Wulffpack in their respective forms as 
Construction Zone volumes and generators
"""

class WulffBase(ABC):

    @abstractmethod
    def __init__(self):
        pass

    ####################
    #### Properties ####
    ####################

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, val):
        assert(isinstance(val, Generator)), "Supplied generator must be crystalline Generator class"
        self._generator = val
        p_struct = val.structure.get_primitive_structure()
        self._p_atoms = Atoms(positions=p_struct.cart_coords)

    @property
    def p_atoms(self):
        return self._p_atoms

    @property
    def natoms(self):
        return self._natoms

    @natoms.setter
    def natoms(self, val):
        try:
            self._natoms = int(val)
        except:
            raise TypeError("natoms must be convertible to integer")
    
    @property
    def surface_energies(self):
        return self._surface_energies
    
    @surface_energies.setter
    def surface_energies(self, vals):
        for key, val in vals.items():
            assert(isintance(key, tuple) and isinstance(val, float)), \
                "surface energies must be passed as tuple pairs of miller indices and floats"
        
        self._surface_energies = vals

    ####################
    ###### Methods #####
    ####################

    @staticmethod
    def planes_from_facets(wulff):
        return [Plane(f.normal, f.normal*f.distance_from_origin) for f in wulff._yield_facets()]

    @abstractmethod
    def get_construction(self):
        pass

class WulffSingle(WulffBase):

    def __init__(self, generator: Generator, surface_energies: dict[tuple, float], natoms: int = 1000):
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
        """
        for single crystals, base particles
        -provide generator
        -get primitive structure as Atoms object
        -use wulffpack constructors
        -convert facets to CZ planes-> place w.r.t. generator origin and orientation
        -return volume with original generator attached
        """
        wulff = wp_SingleCrystal(self.surface_energies, self.p_atoms, self.natoms)
        planes = planes_from_facets(wulff)

        return Volume(alg_objects=planes, generator=self.generator)

    def winterbottom(self, interface: tuple, interface_energy: float):
        wulff = wp_Winterbottom(self.surface_energies, interface, interface_energy, self.p_atoms, self.natoms)
        planes = planes_from_facets(wulff)

        return Volume(alg_objects=planes, generator=self.generator)

    @classmethod
    def cube(generator: Generator, natoms: int = 1000):
        energies = {(1,0,0):1.0}
        wulff = cls(generator, natoms)
        return wulff.get_construction(energies)

    @classmethod
    def cuboctohedron(generator: Generator, natoms: int = 1000):
        energies = {(1,0,0):1.0, (1,1,1):1.15}
        wulff = cls(generator, natoms)
        return wulff.get_construction(energies)

    @classmethod
    def octohedron(generator: Generator, natoms: int = 1000):
        energies = {(1,1,1):1.0}
        wulff = cls(generator, natoms)
        return wulff.get_construction(energies)

    @classmethod
    def truncated_octohedron(generator: Generator, natoms: int = 1000):
        energies = {(1,0,0):1.1, (1,1,0):1.15, (1,1,1):1.0}
        wulff = cls(generator, natoms)
        return wulff.get_construction(energies)

# class WulffDecahedron(WulffBase):