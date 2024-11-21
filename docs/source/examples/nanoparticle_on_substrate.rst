Defected FCC Nanoparticle on Carbon Substrate
=======================================================

In this example, we'll be creating a spherical gold nanoparticle with a couple of
planar defects, which will be then be placed onto an amorphous carbon substrate. 
In this example, we'll be going through most of the core functionality of the package.

First, let's import the classes we'll need. 

.. code-block::

    import czone as cz
    import numpy as np
    from cz.volume import MultiVolume, Volume, Sphere, Plane, snap_plane_near_point, makeRectPrism
    from cz.generator import Generator, AmorphousGenerator
    from cz.transform import Rotation, Reflection, Translation, rot_vtv, s2s_alignment
    from cz.scene import Scene

Let's start with making the substrate. We first will create an AmorphousGenerator object,
which by default generates blocks of amorphous carbon which are periodic in x and y.
We want a substrate that is 12 nm x 12 nm x 5 nm thick, so we use a utility to create
the points of a rectangular prism with those dimensions. We then create a Volume object,
to which we attach our amorphous carbon generator and the points defining the boundaries of the substrate.

.. code-block::

    c_generator = AmorphousGenerator()
    substrate_prism = makeRectPrism(120,120,50)
    substrate = Volume(points=substrate_prism, generator=c_generator)

For the gold nanoparticle, we we will be working with a crystalline generator. 
Gold is an FCC metal with a lattice parameter about 4 angstroms. Here, we can 
request a Generator with the correct unit cell and symmetry by providing the space group
information. Only symmetric sites in the unit cell need to be passed in.

.. code-block::

    base_Au_gen = Generator.from_spacegroup(Z=[79],
                                            coords=np.array([[0,0,0]]), 
                                            cellDims=4.07825*np.ones(3),
                                            cellAngs=[90,90,90],
                                            sgn=225)

Now, we can work on making the spherical nanoparticle itself. We first create
a sphere to represent the outer boundary. The sphere is right now centered at the global origin.
For the defects, we'll put a twin defect in the center of the nanoparticle and two
stacking faults further toward the side. We use the snap_plane_near_point utility 
to grab several (111) planes for the defect placement.

.. code-block::

    d_111 = 4.07825/np.sqrt(3)

    sphere = Sphere(center=np.array([0,0,0]), radius=32.5)
    refl_111_plane = snap_plane_near_point(sphere.center, base_Au_gen, (1,1,1))

    b_111 =  base_Au_gen.voxel.sbases @ np.array([1,1,1])
    b_111 *= d_111 / np.linalg.norm(b_111)
    cutoff_111_a = snap_plane_near_point(-3*b_111, base_Au_gen, (1,1,1))
    cutoff_111_b = snap_plane_near_point(-8*b_111, base_Au_gen, (1,1,1))

In order to have regions of the nanoparticle have defects, we essentially need to 
have new rules for how atoms are supplied to those regions. We can use a series
of derived generators for that very purpose. For the twin grain, we reflect the 
original generator over the twin plane. For the two stacking faults, we create
generators with shifted local origins. 

.. code-block::

    b_112 = base_Au_gen.voxel.sbases @ np.array([1,1,-2]).T / 3
    b_121 = base_Au_gen.voxel.sbases @ np.array([-1,2,-1]).T / 3

    refl_111 = Reflection(refl_111_plane)
    translate_112 = Translation(shift=b_112)
    translate_121 = Translation(shift=b_121)

    twin_gen = base_Au_gen.from_generator(transformation=[refl_111])
    shift_a_gen = base_Au_gen.from_generator(transformation=[translate_112])
    shift_b_gen = base_Au_gen.from_generator(transformation=[translate_121])

Now that we have all the generators for the sub-grains of the nanoparticle, we
can make the nanoparticle by combining all of the volumes together. Each grain
will get its own volume, which is represented by the intersection of the interiors
of the sphere and their respective defect planes. To make sure the grains don't
generate atoms on top of eachother where their volumes intersect, we assign the grains
different priorities. A lower priority means that volume has precedence over other 
volumes with higher priority levels. Two volumes with the same precedence will
remove atoms in their interesecting region. The volumes are added to a MultiVolume object,
which let's us manipulate all the grains simultaneously as one large semantic object.

.. code-block::

    Au_main_grain = Volume(alg_objects=[sphere, refl_111_plane], generator=base_Au_gen)
    Au_sf_a_grain = Volume(alg_objects=[sphere, cutoff_111_a], generator=shift_a_gen)
    Au_sf_b_grain = Volume(alg_objects=[sphere, cutoff_111_b], generator=shift_b_gen)
    Au_twin_grain = Volume(alg_objects=[sphere], generator=twin_gen)

    Au_twin_grain.priority = 4
    Au_main_grain.priority = 3
    Au_sf_a_grain.priority = 2
    Au_sf_b_grain.priority = 1

    defected_NP = MultiVolume(volumes=[Au_twin_grain, Au_main_grain, Au_sf_a_grain, Au_sf_b_grain])

We now rotate the nanoparticle to a random zone axis with a rotation transformation.

.. code-block::

    # rotate to a random zone axis
    zone_axis = np.random.randn(3)
    zone_axis /= np.linalg.norm(zone_axis)
    rot = Rotation(matrix=rot_vtv(zone_axis, [0,0,1]))

    defected_NP.transform(rot)


We use a surface alignment routine to help place the particle in the desired location
on the substrate. We take one plane, near the bottom of the nanoparticle, and align it
to the surface of the substrate. We also want to align the center of the sphere (which is 
currently the origin) with the center of substrate in X and Y. The surface alignment routine
returns a MultiTransform object, which contains a sequence of transformations (in this case, 
a rotation followed by a translation).

.. code-block::

    moving_plane = Plane(point=[0,0,-0.8*sphere.radius], normal=[0,0,-1]) # not quite the bottom of the NP
    target_plane = Plane(point=[0,0,50], normal=[0,0,1]) # the surface of the substrate
    alignment_transform = s2s_alignment(moving_plane,
                                            target_plane,
                                            np.array([0,0,0]),
                                            np.array([60,60,0]))

    defected_NP.transform(alignment_transform)

Finally, we add the substrate and the nanoparticle to a scene. We use the populate method of the 
scene to actually generate the atoms, and once that is done (it may take 10-30 seconds for the
carbon generation), write the structure an output file for visualization with our favorite 
visualization software.

.. code-block::

    # remove substrate where NP exists
    defected_NP.priority = 0
    substrate.priority = 1
    defected_NP_scene = cz.scene.Scene(bounds=np.array([[0,0,0],[120,120,120]]),
                                    objects=[defected_NP, substrate])
    defected_NP_scene.populate()
    defected_NP_scene.to_file("defected_NP.xyz")


While the code above is pretty compact, and hopefully, straightforward and readable, it can still be a little cumbersome.
Imagine that we want to not sample a specific planar defect location, but many nanoparticles with random
placement of defects. The above procedure has a couple of key steps where we define lattice relationships
that make up our planar defects---this can certainly be reduced to a generalized algorithm.
Construction Zone is designed to take algorithms and make repeatable programmatic workflows
that can be sampled many times for large scale structure generation. Some such routines are 
already developed in the Prefab module. FCC planar defects is one such prefab routine
currently available. 

In the following code, we create the defected nanoparticle itself in all of four lines.
We then rotate and place the code onto the substrate as before, and create two structure files---
one without the substrate, and one with the substrate. 

.. code-block::
    from cz.prefab import fccMixedTwinSF

    sphere = Sphere(center=np.array([0,0,0]), radius=radius)
    vol = Volume(alg_objects=[small_sphere])
    sf_object_prefab = fccMixedTwinSF(generator=base_Au_gen, volume=vol, ratio=0.75, N=3)
    current_vol = sf_object_prefab.build_object() #sample a defected nanoparticle

    # apply random rotation
    zone_axis = np.random.randn(3)
    zone_axis /= np.linalg.norm(zone_axis)
    rot = Rotation(matrix=rot_vtv(zone_axis, [0,0,1]))
    current_vol.transform(rot)

    # put on substrate and apply random shift about center of FOV
    moving_plane = Plane(point=[0,0,-0.8*small_sphere.radius], normal=[0,0,-1]) # not quite the bottom of the NP
    target_plane = Plane(point=[0,0,50], normal=[0,0,1]) # the surface of the substrate
    final_center = np.array([60,60,0]) + 10*np.random.randn(3)*np.array([1,1,0])
    alignment_transform = s2s_alignment(moving_plane,
                                        target_plane,
                                        small_sphere.center,
                                        final_center)

    current_vol.transform(alignment_transform)

    scene = cz.scene.Scene(bounds=np.array([[0,0,0],[120,120,125]]), objects=[current_vol])
    scene.populate()
    scene.to_file("particle.xyz")
    scene.add_object(substrate)
    scene.populate()
    scene.to_file("particle_on_substrate.xyz")
