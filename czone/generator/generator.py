"""
Generator Class
Luis Rangel DaCosta
"""
from .amorphous_algorithms import *
from ..transform import BaseTransformation, Translation
from ..volume.voxel import Voxel
import copy
import numpy as np
from abc import ABC, abstractmethod
from pymatgen import Structure, Lattice
from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number

#####################################
########## Generator Classes ########
#####################################

class BaseGenerator():

    @abstractmethod
    def supply_atoms(self):
        pass

class Generator(BaseGenerator):
    """
    Generator class for crystal systems
    Utilizes pymatgen Lattice and Structure to supply relevant methods
    """
    def __init__(self, origin=None, structure=None):
        self._structure = None
        self._lattice = None
        self._species = None
        self._coords = None
        self._orientation = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self._voxel = None

    """
    Properties
    """
    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        assert(isinstance(structure, Structure)), "Structure must be pymatgen Structure object"
        self._structure = structure

    @property
    def lattice(self):
        return self.structure.lattice

    
    @property
    def species(self):
        return self.structure.atomic_numbers

    @property
    def coords(self):
        return self.structure.frac_coords

    @property
    def orientation(self, type):
        return self._orientation

    @property
    def voxel(self):
        return self._voxel

    @voxel.setter
    def voxel(self, voxel):
        if not isinstance(voxel, Voxel):
            raise TypeError("Supplied voxel is not of Voxel() class")

        self._voxel = voxel

    """ 
    Methods
    """
    def supply_atoms(self, bbox):
        """
        Add atoms to bounding box region
        """
        #generation method for pymatgen lattice structure
        min_extent, max_extent = self.voxel.get_extents(bbox)
        fcoords = np.copy(self.coords) #grab fractional in case bases are rotated

        coords = []
        species = []
        # TODO: remove loops for speed
        for i in range(min_extent[0], max_extent[0]+1):
            for j in range(min_extent[1], max_extent[1]+1):
                for k in range(min_extent[2], max_extent[2]+1):
                    new_coords = np.matmul(self.voxel.sbases, np.array([i,j,k])) \
                                + np.dot(self.voxel.sbases, fcoords.T).T \
                                + self.voxel.origin
                    coords.append(new_coords)
                    species.append(self.species)

        coords = np.squeeze(np.array(coords))
        species = np.squeeze(np.array(species))

        if len(coords.shape) > 2:
            coords = np.reshape(coords, (np.prod(coords.shape[0:-1]), 3))
            species = species.ravel()
        return coords, species

    def rotate(self,R):
        #rotate basis of voxel for equivalence
        self.voxel.rotate(R)

    def transform(self, transformation):
        assert(isinstance(transformation, BaseTransformation)), "Supplied transformation not transformation object."
        self.voxel.bases = transformation.applyTransformation_bases(self.voxel.bases)

        if not (transformation.basis_only):
            new_origin = transformation.applyTransformation(np.reshape(self.voxel.origin,(1,3)))
            self.voxel.origin = np.squeeze(new_origin)


class AmorphousGenerator(BaseGenerator):
    """
    Currently supports only monatomic, periodically uniformly disrtributed blocks

    Defaults to carbon properties
    """
    def __init__(self, origin=None, min_dist=1.4, density=.1103075, species=6):
        self._origin = None
        self._species = None
        self._density = None
        self._min_dist = None

        if not (origin is None):
            self.origin = origin

        self.species = species
        self.min_dist = min_dist
        self.density = density

    """
    Properties
    """
    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):
        origin = np.array(origin)
        assert(origin.size==3), "Origin must be a point in 3D space"
        self._origin = origin

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, species):
        self._species = species

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        assert(density>0.0)
        self._density = density

    @property
    def min_dist(self):
        return self._min_dist

    @min_dist.setter
    def min_dist(self, min_dist):
        assert(min_dist > 0)
        self._min_dist = min_dist
    """
    Methods
    """
    def supply_atoms(self, bbox):

        coords = gen_p_substrate(np.max(bbox,axis=0)-np.min(bbox,axis=0), self.min_dist)
        return coords, np.ones(coords.shape[0])*self.species

#####################################
######### Utility routines ##########
#####################################

def BasicStructure(Z=[1], coords=[[0.0, 0.0, 0.0]], cellDims=[2.5, 2.5, 2.5], cellAngs=[90, 90, 90]):
    """
    Define simple translating unit
    """
    tmp = Generator()
    tmp_lattice = Lattice.from_parameters(a=cellDims[0], b=cellDims[1], c=cellDims[2], \
                                            alpha=cellAngs[0], beta=cellAngs[1], gamma=cellAngs[2])

    tmp.structure = Structure(tmp_lattice, Z, coords)
    tmp.voxel = Voxel(scale=cellDims[0])

    return tmp

def from_spacegroup(Z, coords, cellDims, cellAngs, sgn=None, sym=None):
    if sym is None:
        sg = SpaceGroup(int_symbol=sg_symbol_from_int_number(sgn))
    else:
        sg = SpaceGroup(int_symbol=sym)

    test_lattice = Lattice.from_parameters(a=cellDims[0], b=cellDims[1], c=cellDims[2],
                                        alpha=cellAngs[0], beta=cellAngs[1], gamma=cellAngs[2])

    tmp = Generator()
    tmp.structure = Structure.from_spacegroup(sg.int_number, lattice=test_lattice, species=Z, coords=coords)
    tmp.voxel = Voxel(scale=cellDims[0])

    return tmp

def from_generator(orig, **kwargs):
    """
    Construct a generator from another generator
    **kwargs encodes relationship
    """

    if not isinstance(orig, Generator):
        raise TypeError("Supplied generator is not of Generator() class")

    new_generator = copy.deepcopy(orig)

    if "transformation" in kwargs.keys():
        for t in kwargs["transformation"]:
            new_generator.transform(t)

    return new_generator
