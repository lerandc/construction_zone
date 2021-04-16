from ase import Atoms
from ase.io import write as ase_write
from ..scene import Scene

def write_scene(fname, scene, format=None):
    assert(isinstance(scene, Scene)), "Input scene not a Scene object"
    tmp = Atoms(symbols=scene.all_species, positions=scene.all_atoms)
    ase_write(filename=fname, images=tmp, format=format)