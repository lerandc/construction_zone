import matplotlib.pyplot as plt
import numpy as np
try:
    import ipywidgets as widgets
    from ipywidgets import (FloatSlider, Layout, fixed, interact, interact_manual,
                            interactive)
except:
    print("Could not import ipywidgets.")

from ..scene import Scene


def simple_view_with_rotate(data, azim=-60, elev=30):
    fig = plt.figure(figsize=(12, 3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05,
                                    h_pad=0.05,
                                    hspace=0.1,
                                    wspace=0.2)
    ax_x = fig.add_subplot(1, 4, 3, projection='3d')
    ax_y = fig.add_subplot(1, 4, 2, projection='3d')
    ax_z = fig.add_subplot(1, 4, 1, projection='3d')
    ax_r = fig.add_subplot(1, 4, 4, projection='3d')

    ax_r.scatter(data[:, 0], data[:, 1], data[:, 2], s=30, alpha=1)
    ax_x.scatter(data[:, 0], data[:, 1], data[:, 2], s=30, alpha=1)
    ax_y.scatter(data[:, 0], data[:, 1], data[:, 2], s=30, alpha=1)
    ax_z.scatter(data[:, 0], data[:, 1], data[:, 2], s=30, alpha=1)

    ax_z.azim = -90.0
    ax_z.elev = 90.0

    ax_y.azim = -90.0
    ax_y.elev = 0.0

    ax_x.azim = 0.0
    ax_x.elev = 0.0

    ax_r.azim = azim
    ax_r.elev = elev
    ax_r.set_xticks([])
    ax_r.set_yticks([])
    ax_r.set_zticks([])

    ax_x.set_xticks([])
    ax_y.set_yticks([])
    ax_z.set_zticks([])

    ax_x.grid(True)
    ax_y.grid(True)
    ax_z.grid(True)

    ax_x.set_ylabel("Y")
    ax_x.set_zlabel("Z")

    ax_y.set_xlabel("X")
    ax_y.set_zlabel("Z")

    ax_z.set_xlabel("X")
    ax_z.set_ylabel("Y")

    ax_x.set_title("YZ projection")
    ax_y.set_title("XZ projection")
    ax_z.set_title("XY projection")
    ax_r.set_title("Rotatable view")

    plt.draw()


def simple_scene_view(scene, azim=-60, elev=30):
    if not isinstance(scene, Scene):
        raise TypeError("Supplied scene not a Scene() object")

    data = scene.all_atoms
    species = scene.all_species
    bbox = scene.bounds

    fig = plt.figure(figsize=(12, 3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05,
                                    h_pad=0.05,
                                    hspace=0.1,
                                    wspace=0.2)

    ax_x = fig.add_subplot(1, 4, 3, projection='3d')
    ax_y = fig.add_subplot(1, 4, 2, projection='3d')
    ax_z = fig.add_subplot(1, 4, 1, projection='3d')
    ax_r = fig.add_subplot(1, 4, 4, projection='3d')

    for s in np.unique(species):
        mask = species == s
        ax_r.scatter(data[mask, 0], data[mask, 1], data[mask, 2], s=30, alpha=1)
        ax_x.scatter(data[mask, 0], data[mask, 1], data[mask, 2], s=30, alpha=1)
        ax_y.scatter(data[mask, 0], data[mask, 1], data[mask, 2], s=30, alpha=1)
        ax_z.scatter(data[mask, 0], data[mask, 1], data[mask, 2], s=30, alpha=1)

    ax_z.azim = -90.0
    ax_z.elev = 90.0

    ax_y.azim = -90.0
    ax_y.elev = 0.0

    ax_x.azim = 0.0
    ax_x.elev = 0.0

    ax_r.azim = azim
    ax_r.elev = elev

    ax_r.set_xlim([bbox[0, 0], bbox[1, 0]])
    ax_r.set_ylim([bbox[0, 1], bbox[1, 1]])
    ax_r.set_zlim([bbox[0, 2], bbox[1, 2]])

    ax_x.set_xlim([bbox[0, 0], bbox[1, 0]])
    ax_x.set_ylim([bbox[0, 1], bbox[1, 1]])
    ax_x.set_zlim([bbox[0, 2], bbox[1, 2]])

    ax_y.set_xlim([bbox[0, 0], bbox[1, 0]])
    ax_y.set_ylim([bbox[0, 1], bbox[1, 1]])
    ax_y.set_zlim([bbox[0, 2], bbox[1, 2]])

    ax_z.set_xlim([bbox[0, 0], bbox[1, 0]])
    ax_z.set_ylim([bbox[0, 1], bbox[1, 1]])
    ax_z.set_zlim([bbox[0, 2], bbox[1, 2]])

    ax_r.set_xticks([])
    ax_r.set_yticks([])
    ax_r.set_zticks([])

    ax_x.set_xticks([])
    ax_y.set_yticks([])
    ax_z.set_zticks([])

    ax_x.grid(True)
    ax_y.grid(True)
    ax_z.grid(True)

    ax_x.set_ylabel("Y")
    ax_x.set_zlabel("Z")

    ax_y.set_xlabel("X")
    ax_y.set_zlabel("Z")

    ax_z.set_xlabel("X")
    ax_z.set_ylabel("Y")

    ax_x.set_title("YZ projection")
    ax_y.set_title("XZ projection")
    ax_z.set_title("XY projection")
    ax_r.set_title("Rotatable view")

    plt.draw()


def simple_scene_view_with_rotate(scene, azim=-60, elev=30):
    if not isinstance(scene, Scene):
        raise TypeError("Supplied scene not a Scene() object")

    data = scene.all_atoms
    species = scene.all_species
    bbox = scene.bounds

    fig = plt.figure(figsize=(12, 3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05,
                                    h_pad=0.05,
                                    hspace=0.1,
                                    wspace=0.2)

    ax_x = fig.add_subplot(1, 4, 3, projection='3d')
    ax_y = fig.add_subplot(1, 4, 2, projection='3d')
    ax_z = fig.add_subplot(1, 4, 1, projection='3d')
    ax_r = fig.add_subplot(1, 4, 4, projection='3d')

    for s in np.unique(species):
        mask = species == s
        ax_r.scatter(data[mask, 0], data[mask, 1], data[mask, 2], s=30, alpha=1)
        ax_x.scatter(data[mask, 0], data[mask, 1], data[mask, 2], s=30, alpha=1)
        ax_y.scatter(data[mask, 0], data[mask, 1], data[mask, 2], s=30, alpha=1)
        ax_z.scatter(data[mask, 0], data[mask, 1], data[mask, 2], s=30, alpha=1)

    ax_z.azim = -90.0
    ax_z.elev = 90.0

    ax_y.azim = -90.0
    ax_y.elev = 0.0

    ax_x.azim = 0.0
    ax_x.elev = 0.0

    ax_r.azim = azim
    ax_r.elev = elev

    ax_r.set_xlim([bbox[0, 0], bbox[1, 0]])
    ax_r.set_ylim([bbox[0, 1], bbox[1, 1]])
    ax_r.set_zlim([bbox[0, 2], bbox[1, 2]])

    ax_x.set_xlim([bbox[0, 0], bbox[1, 0]])
    ax_x.set_ylim([bbox[0, 1], bbox[1, 1]])
    ax_x.set_zlim([bbox[0, 2], bbox[1, 2]])

    ax_y.set_xlim([bbox[0, 0], bbox[1, 0]])
    ax_y.set_ylim([bbox[0, 1], bbox[1, 1]])
    ax_y.set_zlim([bbox[0, 2], bbox[1, 2]])

    ax_z.set_xlim([bbox[0, 0], bbox[1, 0]])
    ax_z.set_ylim([bbox[0, 1], bbox[1, 1]])
    ax_z.set_zlim([bbox[0, 2], bbox[1, 2]])

    ax_r.set_xticks([])
    ax_r.set_yticks([])
    ax_r.set_zticks([])

    ax_x.set_xticks([])
    ax_y.set_yticks([])
    ax_z.set_zticks([])

    ax_x.grid(True)
    ax_y.grid(True)
    ax_z.grid(True)

    ax_x.set_ylabel("Y")
    ax_x.set_zlabel("Z")

    ax_y.set_xlabel("X")
    ax_y.set_zlabel("Z")

    ax_z.set_xlabel("X")
    ax_z.set_ylabel("Y")

    ax_x.set_title("YZ projection")
    ax_y.set_title("XZ projection")
    ax_z.set_title("XY projection")
    ax_r.set_title("Rotatable view")

    plt.draw()


def simple_view_widget(data):
    azim_widget = FloatSlider(value=-60,
                              min=-180,
                              max=180,
                              step=0.1,
                              description='Azimuth',
                              readout_format='.1f',
                              style={'description_width': '150px'},
                              layout=Layout(width='400px', height='30px'))

    elev_widget = FloatSlider(value=30,
                              min=-90,
                              max=90,
                              step=0.1,
                              description='Elevation',
                              readout_format='.1f',
                              style={'description_width': '150px'},
                              layout=Layout(width='400px', height='30px'))

    return interact(simple_view_with_rotate,
                    data=fixed(data),
                    azim=azim_widget,
                    elev=elev_widget)


def simple_scene_widget(scene):
    azim_widget = FloatSlider(value=-60,
                              min=-180,
                              max=180,
                              step=0.1,
                              description='Azimuth',
                              readout_format='.1f',
                              style={'description_width': '150px'},
                              layout=Layout(width='400px', height='30px'))

    elev_widget = FloatSlider(value=30,
                              min=-90,
                              max=90,
                              step=0.1,
                              description='Elevation',
                              readout_format='.1f',
                              style={'description_width': '150px'},
                              layout=Layout(width='400px', height='30px'))

    return interact(simple_scene_view_with_rotate,
                    scene=fixed(scene),
                    azim=azim_widget,
                    elev=elev_widget)
