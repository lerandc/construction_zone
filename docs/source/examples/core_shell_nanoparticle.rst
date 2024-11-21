Core-Shell Mn3O4/Co3O4 Nanoparticle
=========================================

In this example, we'll be making a core-shell oxide nanoparticle similar to those 
studied in :footcite:t:`oh_design_2020`. This nanoparticle has a cubic Co3O4 
grain in the center with strained Mn3O4 grains on all faces of the inner grain.

First, let's import the routines we need.

.. code-block::

    import numpy as np
    import czone as cz
    from pymatgen.core import Structure
    from cz.volume import Volume, MultiVolume, Plane, snap_plane_near_point
    from cz.generator import Generator, AmorphousGenerator
    from cz.transform import Rotation, Reflection, Translation, rot_vtv, rot_v
    from cz.scene import Scene

We downloaded `cif` files from the [Materials Project](https://materialsproject.org/) database 
for the unit cell data for the two grain types. Here, we load the the files into 
[pymatgen](https://pymatgen.org/) Structure objects which serve as the core drivers of our crystalline Generators.

.. code-block::

    mn_crystal = Structure.from_file("Mn3O4_mp-18759_conventional_standard.cif")
    co_crystal = Structure.from_file("Co3O4_mp-18748_conventional_standard.cif")

In this nanoparticle, the Mn grains grow with their basal planes on the faces of (100)
planes of the Co core. The Mn lattice is rotated and strained such that Mn sites
along the <110> direction are coincident with the Co sites. Here, we calculate 
the appropriate strain for the Mn unit cell and apply it to the structure.

.. code-block::

    # correct lattice mismatch
    co_100 = np.linalg.norm(co_crystal.lattice.matrix @ np.array([1,0,0]))
    mn_110 = np.linalg.norm(mn_crystal.lattice.matrix @ np.array([1,1,0]))
    mn_crystal.apply_strain([co_100/mn_110-1, co_100/mn_110-1, 0])

We start the nanoparticle by making the Co core. We create a generator using the
Co structure above, and then find the 6 planes in the (100) family that are  
6 unit cells away from the lattice origin to use as boundaries for the cube.
We add the planes and the generator to a Volume object and are done with the core.

.. code-block::

    # Make Co3O4 core
    co_gen = Generator(structure=co_crystal)

    N_uc = 6 # stretch N unit cells from center
    co_a = co_crystal.lattice.a
    vecs_100 = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    planes_100 = []
    for v in vecs_100:
        planes_100.append(snap_plane_near_point(N_uc*co_a*v, co_gen, tuple(v)))

    co_core = Volume(alg_objects=planes_100, generator=co_gen)


For the Mn grains, we will start with a single unstrained grain and then replicate
that grain about the Co core. We create the Generator from the Structure object, 
and then rotate it about its c-axis 45 degrees and then translate it to the top of the
Co core. Since by default Generators have a local origin of (0,0,0), we do not need
to do anything more to properly align the Mn and Co lattices.

.. code-block::

    # Make unstrained Mn3O4 grain
    mn_gen = Generator(structure=mn_crystal)

    # rotate 45 degrees and place on top of Co core
    r = Rotation(matrix=rot_v(np.array([0,0,1]), np.pi/4))
    t = Translation(np.array([0,0,N_uc*co_a]))
    mn_gen.transform(r)
    mn_gen.transform(t)

The Mn grain has many facets. Keeping in mind that we are working in the grain 
on the +Z side of the, we have (100) facets on the bottom and top of the grain;
(112) facets at Mn-Mn interfaces; and (101) facets facing free space. 

For the bottom facet, we use grab a point at the Co surface, and for the top facet,
we grab a point 6 unit cells above said point and snap (001) planes in the Mn
crystal coordinate system to these points. For the (112) facets, which meet the
edges of the Co core, we grab (112) planes running through the edge by choosing 
points that lie on the edge and snapping to them. For the (101) facets, we 
just choose points near the top surface and translate outwards--- this was chosen
heuristically to look nice.

.. code-block::

    # define top and bottom 100 surfaces
    surface_point = np.array([0,0,N_uc*co_a])
    mn_c = mn_crystal.lattice.c
    mn_bot = snap_plane_near_point(surface_point, mn_gen, (0,0,-1))
    mn_top = snap_plane_near_point(surface_point+6*mn_c*np.array([0,0,1]), mn_gen, (0,0,1))

    # define 112 facets
    side_vs_112 = [(1,1,-2),(1,-1,-2), (-1,1,-2), (-1,-1,-2)]
    co_vector = [(0,1,0), (1,0,0), (-1,0,0), (0,-1,0)]
    sides_112 = []
    for s, c in zip(side_vs_112, co_vector):
        tmp_point = N_uc * np.array(c) * co_a + surface_point
        tmp_plane = snap_plane_near_point(tmp_point, mn_gen, s) 
        sides_112.append(tmp_plane)

    # define 101 facets
    mn_a = mn_crystal.lattice.a
    side_vs_101 = [(1,0,1),(0,1,1),(-1,0,1),(0,-1,1)]
    sides_101 = []
    for s in side_vs_101:
        tmp_point = np.array([1,1,0]) * np.sign(s) * 12*mn_a + mn_top.point
        tmp_plane = snap_plane_near_point(tmp_point, mn_gen, s) 
        sides_101.append(tmp_plane)

Now that we have the facets of our grain defiend, we create a Volume for the grain
and with all of the planes we defined along with the Mn lattice generator that 
we have previously rotated and translated.

.. code-block::

    # create volume representing grain
        mn_vols = [mn_bot, mn_top] + sides_112+sides_101
        mn_grain = Volume(alg_objects=mn_vols, generator=mn_gen)
        mn_grain.priority=1

For the five other grains, we can rotate the original grain. We first find rotate about 
the global y-axis and then the global x-axis to flip the grain around appropriately.
By default, the origin of a rotation is set to the global origin, but any origin can be chosen.

.. code-block::

    # rotate to make 5 other grains from +z shell grain
    mn_grains = [mn_grain]

    # get +x, -z, -x
    for theta in [np.pi/2, np.pi, -np.pi/2]:
        rot = Rotation(rot_v(np.array([0,1,0]),theta))
        tmp_grain = mn_grain.from_volume(transformation=[rot])
        mn_grains.append(tmp_grain)

    # get +-y
    for theta in [np.pi/2, -np.pi/2]:
        rot = Rotation(rot_v(np.array([1,0,0]),theta))
        tmp_grain = mn_grain.from_volume(transformation=[rot])
        mn_grains.append(tmp_grain)

We finally add all the volumes together to a MultiVolume and write out the nanoparticle 
to a structure file for visualization.

.. code-block::

    # make final core-shell NP as multivolume and save to file
    core_shell_NP = MultiVolume([co_core] + mn_grains)
    core_shell_NP.populate_atoms()
    core_shell_NP.to_file("core_shell_NP.xyz")

We now have this complex oxide nanoparticle structure! However, there is one glaringly un-physical
feature-- between the Mn grains, there is a gap between the (112) facets. As grown,
these nanoparticles are continuous in the Mn grains. The exact mechanism by which the nanoparticles 
accomodate for this gap is still an object of research. However, for now, we can concieve of
accomodating this gap by a simple homogeneous strain that compresses the Mn grain along their c-axes 
until the gap is closed. In Construction Zone, this is also easy to accomplish.

TO COME: Applying a geometrically necessary strain field to the particle.
