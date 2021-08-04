Defected FCC Nanoparticle on Carbon Substrate
=======================================================

.. code-block::

    import czone as cz
    import numpy as np
    from cz.volume import MultiVolume, Volume, Sphere, Plane, snap_plane_near_point
    from cz.generator import Generator, AmorphousGenerator
    from cz.transform import Rotation, Reflection, Translation, rot_vtv
    from cz.scene import Scene


.. code-block::

    c_generator = AmorphousGenerator()
    substrate_prism = makeRectPrism(120,120,50)
    substrate = Volume(points=substrate_prism, generator=c_generator)


.. code-block::

    base_Au_gen = Generator.from_spacegroup(Z=[79],
                                            coords=np.array([[0,0,0]]), 
                                            cellDims=4.07825*np.ones(3),
                                            cellAngs=[90,90,90],
                                            sgn=225)


.. code-block::

    d_111 = 4.07825/np.sqrt(3)

    sphere = Sphere(center=np.array([0,0,0]), radius=32.5)
    refl_111_plane = snap_plane_near_point(sphere.center, base_Au_gen, (1,1,1))

    b_111 =  base_Au_gen.voxel.sbases @ np.array([1,1,1])
    b_111 *= d_111 / np.linalg.norm(b_111)
    cutoff_111_a = snap_plane_near_point(-3*b_111, base_Au_gen, (1,1,1))
    cutoff_111_b = snap_plane_near_point(-8*b_111, base_Au_gen, (1,1,1))



.. code-block::

    b_112 = base_Au_gen.voxel.sbases @ np.array([1,1,-2]).T / 3
    b_121 = base_Au_gen.voxel.sbases @ np.array([-1,2,-1]).T / 3

    refl_111 = Reflection(refl_111_plane)
    translate_112 = Translation(shift=b_112)
    translate_121 = Translation(shift=b_121)

    twin_gen = base_Au_gen.from_generator(transformation=[refl_111])
    shift_a_gen = base_Au_gen.from_generator(transformation=[translate_112])
    shift_b_gen = base_Au_gen.from_generator(transformation=[translate_121])



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


.. code-block::

    # rotate to a random zone axis
    zone_axis = np.random.randn(3)
    zone_axis /= np.linalg.norm(zone_axis)
    rot = Rotation(matrix=rot_vtv(zone_axis, [0,0,1]))

    defected_NP.transform(rot)


.. code-block::

    moving_plane = Plane(point=[0,0,-0.8*sphere.radius], normal=[0,0,-1]) # not quite the bottom of the NP
    target_plane = Plane(point=[0,0,50], normal=[0,0,1]) # the surface of the substrate
    alignment_transform = s2s_alignment(moving_plane,
                                            target_plane,
                                            np.array([0,0,0]),
                                            np.array([60,60,0]))

    defected_NP.transform(alignment_transform)


.. code-block::

    # remove substrate where NP exists
    defected_NP.priority = 0
    substrate.priority = 1
    defected_NP_scene = cz.scene.Scene(bounds=np.array([[0,0,0],[120,120,120]]),
                                    objects=[defected_NP, substrate])
    defected_NP_scene.populate()
    defected_NP_scene.to_file("defected_NP.xyz")


.. code-block::

    sphere = Sphere(center=np.array([0,0,0]), radius=radius)
    vol = Volume(alg_objects=[small_sphere])
    sf_object_prefab = fccMixedTwinSF(generator=base_Au_gen, volume=vol, ratio=0.75, N=3)
    current_vol = sf_object_prefab.build_object()

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
