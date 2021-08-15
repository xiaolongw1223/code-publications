#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 20:10:37 2020

@author: wxl
"""

import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import numpy as np

from discretize import TensorMesh

from SimPEG import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)

from time import time
from datetime import timedelta

init_t = time()
np.random.seed(0)

# Read in mesh
mesh = TensorMesh._readUBC_3DMesh('mesh.txt')

'''
Load data

'''
# Create a MAGsurvey
data_object = utils.io_utils.readUBCgravityObservations('grav.obs')
xyzLoc = data_object.survey.receiver_locations
dobs = data_object.dobs
dpred = np.loadtxt("dpred.txt")
nor_residual = np.loadtxt("dnres.txt")

'''
Plot data

'''
fig = plt.figure(figsize=(14,14))
ax = plt.subplot()
im = utils.plot2Ddata(xyzLoc, dobs, ax=ax, contourOpts={"cmap": "viridis"})
 
ax.set_xlabel('Easting (m)', size=22)
ax.set_ylabel('Northing (m)', size=22)
ax.set_title('')
ax.tick_params(labelsize=16)
ax.locator_params(nbins=10, axis='x')
ax.ticklabel_format(useOffset=False, style="plain")
#ax.set_title('L02_Observed gravity anomaly')
plt.gca().set_aspect('equal', adjustable='box')
pos = ax.get_position()
cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.06, pos.width *0.02, pos.height*0.8])
cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical")
cb.ax.tick_params(labelsize=16) 
cb.ax.set_title('E', size=22)
plt.savefig('fig_obs_data.png', bbox_inches="tight", dpi=300)


# preditcted data
fig = plt.figure(figsize=(14,14))
ax = plt.subplot()
im = utils.plot2Ddata(xyzLoc, dpred, ax=ax, contourOpts={"cmap": "viridis"})
ax.set_xlabel('Easting (m)', size=22)
ax.set_ylabel('Northing (m)', size=22)
ax.set_title('')
ax.tick_params(labelsize=16)
ax.locator_params(nbins=10, axis='x')
ax.ticklabel_format(useOffset=False, style="plain")
#ax.set_title('L02_Observed gravity anomaly')
plt.gca().set_aspect('equal', adjustable='box')
pos = ax.get_position()
cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.06, pos.width *0.02, pos.height*0.8])
cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical")
cb.ax.tick_params(labelsize=16) 
cb.ax.set_title('E', size=22)
plt.savefig('fig_pred_data.png', bbox_inches="tight", dpi=300)


# data residual
fig = plt.figure(figsize=(14,14))
ax = plt.subplot()
im = utils.plot2Ddata(xyzLoc, nor_residual, ax=ax, contourOpts={"cmap": "viridis"})
ax.set_xlabel('Easting (m)', size=22)
ax.set_ylabel('Northing (m)', size=22)
ax.set_title('')
ax.tick_params(labelsize=16)
ax.locator_params(nbins=10, axis='x')
ax.ticklabel_format(useOffset=False, style="plain")
#ax.set_title('L02_Observed gravity anomaly')
plt.gca().set_aspect('equal', adjustable='box')
pos = ax.get_position()
cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.06, pos.width *0.02, pos.height*0.8])
cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical")
cb.ax.tick_params(labelsize=16) 
cb.ax.set_title('E', size=22)
plt.savefig('fig_nor_residual.png', bbox_inches="tight", dpi=300)



'''
Load recovered model

'''
model_rec = -1*mesh.readModelUBC("model_rec.txt")
vmax0, vmin0 = np.nanmax(model_rec), np.nanmin(model_rec)


# density model
fig = plt.figure(figsize=(18,6))
# Plot horizontal section of inverted grav model
ax1 = plt.subplot(121)
slicePosition = -2000 #meter in z direction
sliceInd = int(round(np.searchsorted(mesh.vectorCCz, slicePosition)))
mesh.plotSlice(model_rec, ax=ax1, normal='Z', ind=sliceInd,
           grid=False, clim=(vmin0, vmax0), pcolorOpts={"cmap":"RdYlBu_r"})
ax1.plot(mesh.vectorCCx, np.ones(90)*4790800, c="darkred", linewidth=2)
ax1.set_xlabel('Easting (m)', size=24)
ax1.set_ylabel('Northing (m)', size=24)
ax1.set_title('')
ax1.set_title('(a)', loc='left', size = 24)
ax1.tick_params(labelsize=24)
ax1.ticklabel_format(useOffset=False, style="plain")
ax1.locator_params(nbins=8, axis='x')
ax1.set_aspect('equal', adjustable='box')
pos1 = ax1.get_position()

# Vertical section
ax2 = plt.subplot(122)
slicePosition = 4790800 #meter in y direction
sliceInd = int(round(np.searchsorted(mesh.vectorCCy, slicePosition)))
im = mesh.plotSlice(model_rec, ax=ax2, normal='Y', ind=sliceInd,
           grid=False, clim=(vmin0, vmax0), pcolorOpts={"cmap":"RdYlBu_r"})
ax2.plot(mesh.vectorCCx, np.ones(90)*(-2000), c="darkred", linewidth=2)
ax2.set_xlabel('Easting (m)', size=24)
ax2.set_ylabel('Depth (m)', size=24)
ax2.set_title('')
ax2.set_title('(b)', loc='left', size = 24)
ax2.tick_params(labelsize=24)
ax2.locator_params(nbins=8, axis='x')
ax2.set_aspect('equal', adjustable='box')   
pos2 = ax2.get_position() 
pos2_new = [pos2.x0+0.02, pos1.y0, pos2.width, pos2.height] 
ax2.set_position(pos2_new)

cb_ax = fig.add_axes([pos2.x0+0.09, pos2.y0+0.35, pos2.width *0.6, pos2.height*0.1])
kwargs = {'format': '%.3f'}
cb = plt.colorbar(im[0], cax=cb_ax, orientation="horizontal",
              ticks=np.linspace(vmin0, vmax0, 4), **kwargs)
cb.ax.tick_params(labelsize=24) 
cb.set_label('Density (g/cc)', size=24)
plt.savefig('fig_dens_model.png', bbox_inches="tight", dpi=300)  


elapse = time() - init_t
print("elapsed time: ", str(timedelta(seconds=elapse)))