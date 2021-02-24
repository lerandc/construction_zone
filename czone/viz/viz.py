import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider, Layout
import ipywidgets as widgets

def simple_view_with_rotate(data, azim, elev):
    fig = plt.figure(figsize=(12,3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.1, wspace=0.2)
    ax_x = fig.add_subplot(1,4,3, projection='3d')
    ax_y = fig.add_subplot(1,4,2, projection='3d')
    ax_z = fig.add_subplot(1,4,1, projection='3d')
    ax_r = fig.add_subplot(1,4,4, projection='3d')
    
    ax_r.scatter(data[:,0], data[:,1], data[:,2], s=30, alpha=1)
    ax_x.scatter(data[:,0], data[:,1], data[:,2], s=30, alpha=1)
    ax_y.scatter(data[:,0], data[:,1], data[:,2], s=30, alpha=1)
    ax_z.scatter(data[:,0], data[:,1], data[:,2], s=30, alpha=1)
    
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

def simple_view_widget(data):
    azim_widget = FloatSlider(value=60, min=-180, max=180, step=0.1, 
                       description='Azimuth', readout_format='.1f', 
                       style={'description_width':'150px'}, layout=Layout(width='400px', height='30px'))

    elev_widget = FloatSlider(value=30, min=-90, max=90, step=0.1, 
                           description='Elevation', readout_format='.1f', 
                           style={'description_width':'150px'}, layout=Layout(width='400px', height='30px'))
    
    return interact(rotate_plot, data=fixed(data), azim=azim_widget, elev=elev_widget)