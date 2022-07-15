#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import path

from SimPEG import Utils
from SimPEG import Mesh

import PVGeo
import pyvista as pv


"""
Allow user to interactively select points within polygon.

allow user to visualize 3d model in pyvista.

core idea of this script is borrowed from Jiajia Sun Matlab code.

Author: Xiaolong Wei, updated on Oct 28th, 2020


"""




# # Input number of padding cells
nPx, nPy, nPz = 5, 5, 0 # number of padding cells for each side
sizeM = [110, 45, 29]


mesh = Mesh.TensorMesh._readUBC_3DMesh('mesh_p4.txt')
mesh_rm = Mesh.TensorMesh._readUBC_3DMesh('mesh_p4_rm.txt')

model_dens = mesh.readModelUBC("joint_dens_model_UBC.txt")
model_susc = mesh.readModelUBC("joint_susc_model_UBC.txt")

model_dens_3d = np.reshape(
    model_dens, (sizeM[0],sizeM[1],sizeM[2]), 
    order="F"
    )
model_susc_3d = np.reshape(
    model_susc, (sizeM[0],sizeM[1],sizeM[2]), 
    order="F"
    )

# remove padding cells
model_dens_rm = model_dens_3d[
    nPx:sizeM[0]-nPx, 
    nPy:sizeM[1]-nPy-8, 
    nPz:sizeM[2] # 0:28, 0 is deep part, 28 is near surface layer
    ]

model_dens_rm = Utils.mkvc(model_dens_rm)

model_susc_rm = model_susc_3d[
    nPx:sizeM[0]-nPx, 
    nPy:sizeM[1]-nPy-8, 
    nPz:sizeM[2]
    ]

model_susc_rm = Utils.mkvc(model_susc_rm)


# scatter plot
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot()
ax.scatter(model_dens_rm, model_susc_rm, s=10, c="black")
ax.set_xlabel("Density ($g/cm^3$)", size=18)
ax.set_ylabel("Susceptibility (SI)", size=18)
ax.tick_params(labelsize=18)
ax.set_xlim([model_dens_rm.min(), model_dens_rm.max()])
ax.set_ylim([model_susc_rm.min(), model_susc_rm.max()])
ax.locator_params(nbins=8, axis="both")
ax.set_title("", size=18)
ax.grid()



"""
acquire the pts in the picked polygon area.

step 1: input the number of geological units (polygons).

step 2: left click to select the polygon area. Display a red cross after each click.

step 3: right click, cancel the last selected location.

step 4: press "enter" or "return" on the keyboard to show colorful dots in the selected polygon area.

step 5: repeat step 2, left click to select the second polygon.

step 6: continue until finish the selection for all target polygon areas.


"""


n_unit = input("Number of geological units: ")
model_unit = np.zeros(len(model_dens_rm)) * np.nan

# generating different colors
colors = cm.RdYlBu_r(np.linspace(0, 1, eval(n_unit)))

for i, c in zip(range(eval(n_unit)), colors):
    getpts = plt.ginput(1000) # pick pts
    polygon = path.Path(np.asarray(getpts)) 
    model_array = np.asarray([model_dens_rm, model_susc_rm]).transpose()
    inpolygon = polygon.contains_points(model_array)
    np.savetxt("inpolygon_{}.txt".format(i+1), inpolygon)
    ax.scatter(model_dens_rm[inpolygon], model_susc_rm[inpolygon], s=10, color=c, label="Unit {}".format(i+1))
    # ax.grid()
    ax.legend(fontsize=12)
    model_unit[inpolygon] = i

fig.savefig("fig_scatter.png", bbox_inches="tight", dpi=300)



"""

Visulize the geological differential models in depth-slice and cross-section.



"""
# plot inverted model
vmax, vmin = np.nanmax(model_unit), np.nanmin(model_unit)

# density model
fig = plt.figure(figsize=(20,20))
# Plot horizontal section of inverted grav model
ax1 = plt.subplot(211)
slicePosition = -2000 #meter in z direction
sliceInd = int(round(np.searchsorted(mesh_rm.vectorCCz, slicePosition)))
mesh_rm.plotSlice(model_unit, ax=ax1, normal='Z', ind=sliceInd,
           grid=False, clim=(vmin, vmax), pcolorOpts={"cmap":"RdYlBu_r"})
# ax1.set_xlabel('Easting (m)', size=22)
ax1.set_xlabel('', size=22)
ax1.set_ylabel('Northing (m)', size=20)
ax1.set_title('')
ax1.set_title('(a), z={} m'.format(slicePosition), loc='left', size = 20)
ax1.set_xticks([])
ax1.tick_params(labelsize=20)
ax1.locator_params(nbins=6, axis='x')
ax1.ticklabel_format(useOffset=False, style='plain')
ax1.set_aspect('equal', adjustable='box')
pos1 = ax1.get_position()

# Vertical section
ax2 = plt.subplot(212)
slicePosition = 4820000#4792000# #meter in y direction
sliceInd = int(round(np.searchsorted(mesh_rm.vectorCCy, slicePosition)))
im = mesh_rm.plotSlice(model_unit, ax=ax2, normal='Y', ind=sliceInd,
           grid=False, clim=(vmin, vmax), pcolorOpts={"cmap":"RdYlBu_r"})
ax2.set_xlabel('Easting (m)', size=20)
ax2.set_ylabel('Depth (m)', size=20)
ax2.set_title('')
ax2.set_title('(b), y={} m'.format(slicePosition), loc='left', size = 20)
ax2.tick_params(labelsize=20)
ax2.locator_params(nbins=6, axis='x')
ax2.ticklabel_format(useOffset=False, style='plain')
ax2.set_aspect('equal', adjustable='box')   
pos2 = ax2.get_position() 
pos2_new = [pos1.x0, pos1.y0-0.35, pos1.width, pos2.height] 
ax2.set_position(pos2_new)

cb_ax = fig.add_axes([pos1.x0+0.07, pos2.y0-0.02, pos1.width *0.8, pos2.height*0.04])
kwargs = {'format': '%.3f'}
cb = plt.colorbar(im[0], cax=cb_ax, orientation="horizontal",
              ticks=np.linspace(vmin, vmax, 4), **kwargs)
cb.ax.tick_params(labelsize=20) 
cb.set_label('Geologic units', size=20)
plt.savefig('fig_quasi_geology_model.png', bbox_inches="tight", dpi=300)  




"""
output the index of pts in the selected polygon area

using padding removed mesh file

"""

model_inpolygon_bool = mesh_rm.writeModelUBC("model_inpolygon.txt", model_unit)



"""

3D visualization using Pyvista

"""
model = model_unit
models= {'model in polygon':model}
mesh_rm = mesh_rm.to_vtk(models)

cmap = plt.cm.get_cmap('RdYlBu_r')
# set color
colr = dict(
        show_scalar_bar=False, 
        cmap=cmap,
        clim=[0, 1]
        )

# set bounds
bounds = dict(
        grid = False,
        color = 'black',
        location = 'outer', # 'all', 'back' 'front'
        ylabel = 'Northing(m)', 
        xlabel = 'Easting(m)', 
        zlabel = 'Depth(m)',
        )


threshed = mesh_rm.threshold([0, 9], invert=False)


p = pv.Plotter()

p.add_mesh(threshed, show_edges=False, **colr) # anomaly body

p.add_mesh(
    mesh_rm, color = True, show_edges=True, 
    show_scalar_bar=False, opacity = 0.1
    ) 

p.add_mesh(mesh_rm.outline(), color='black')

cpos = [(596421.0, 4770075.0, 69626.71950281913), (596421.0, 4820075.0, -4645.0), (0.0, 1.0, 0.0)]

p.camera_position = cpos 
print(p.camera_position)

p.show_bounds(**bounds)
p.set_background('white')
p.show(window_size=[1600, 1400])
p.screenshot('fig_quasi_geology_model_3d.png')

