#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Created on Mon Sep 21 13:02:25 2020

@author: Xiaolong Wei & Jiajia Sun

email: xwei7@uh.edu; jsun20@uh.edu
    
University of Houston


This script can be used to perform:
    
    classic L2 norm inversion, 
    sparse inversion,
    mixed Lp norm inversion, 
    total variation inversion,
    focusing inversion.


"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.interpolate import NearestNDInterpolator

from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG.Optimization import IterationPrinters, StoppingCriteria
from SimPEG import InvProblem
from SimPEG import Directives
from SimPEG import Inversion
from SimPEG import PF

import sys
import os
import json

from time import time
from datetime import timedelta, datetime



def run(plotIt=True):

    init_t = time()
    np.random.seed(0)
    
    input_file = sys.argv[1]
    
    # read json file
    with open(input_file, "r") as json_file:
        input_dict = json.load(json_file)
    
    work_dir = os.path.abspath(os.path.dirname(input_file))
    path_sep = os.path.sep
    
    
    if "example" in list(input_dict.keys()):
        if input_dict["example"] == "horse":
            work_dir = os.path.sep.join(
                [work_dir, "Example2_horseShoe"]
                )
            work_dir += path_sep
            
        elif input_dict["example"] == "spheric":
            work_dir = os.path.sep.join(
                [work_dir, "Example1_spheric"]
                )
            work_dir += path_sep
    else:
        raise Exception("Missing the selection of examples")


    # read mesh file
    if "mesh_file" in list(input_dict.keys()):
        mesh = Mesh.TensorMesh._readUBC_3DMesh(
            work_dir+input_dict["mesh_file"]
            )
    else:
        raise Exception("Missing mesh file")
    
   
    # read data file
    if "data_file" in list(input_dict.keys()): 
        # Create a MAGsurvey
        survey = Utils.io_utils.readUBCgravityObservations(
            work_dir+input_dict["data_file"]
            )
        xyzLoc = survey.rxLoc
        dobs = survey.dobs
        print("Number of data points: ", dobs.size)
        std = survey.std      
    else:
        raise Exception("Missing observed data file")
    
    
    # load the topo
    if "topography" in list(input_dict.keys()):
        # Load topo
        topo = np.genfromtxt(
            work_dir+input_dict["topography"], 
            skip_header=1
            )
        F = NearestNDInterpolator(topo[:, :2], topo[:, 2])
        newTopo = np.c_[xyzLoc[:, :2], F(xyzLoc[:, :2])]
    # without topo
    else:
        newTopo = np.c_[
            xyzLoc[:, :2], 
            np.zeros(xyzLoc.shape[0])
            ]
    
    actv = Utils.surface2ind_topo(mesh, newTopo, gridLoc='CC')
    actv = np.where(actv)[0]
    nC = len(actv)
    print("Number of active cells: ", nC)
    
    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, np.nan)    
    
    # Create reduced identity map
    idenMap = Maps.IdentityMap(nP=nC)
    
    # Create the forward model operator
    prob = PF.Gravity.GravityGradient(
        mesh, 
        components=['gzz'], 
        rhoMap=idenMap, 
        actInd=actv
        )
    
    # Pair the survey and problem
    survey.pair(prob)
    
    # Create depth weighting
    z0 = 0
    v = 3 # exponent parameter
    temp1 = np.abs(np.max(topo[:,2]) - mesh.vectorNz)
    index = temp1.argmin()
    sum_factor = mesh.vectorNz[index]
    wz = np.power(
        -mesh.vectorCCz+sum_factor+z0,
        -v
        )
    wz = wz/np.nanmax(wz)
    wz = wz**0.5

    # allocate depth decay to all the cells
    tmp1 = (np.outer(np.ones(mesh.nCx),np.ones(mesh.nCy))).flatten('F')
    wz_depth = (np.outer(tmp1,wz)).flatten('F')
    wz_depth = actvMap.P.T * wz_depth
    
    # Create a regularization
    reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg.cell_weights = wz_depth
    
    if "norm_p" and "norm_q" in list(input_dict.keys()):
        p, q = input_dict["norm_p"], input_dict["norm_q"]
    else:
        raise Exception("Missing norm values")
    
    norms = [p, q, q, q]
    reg.norms = [norms]
    
    if "alpha_s" in list(input_dict.keys()):
        alpha_s = input_dict["alpha_s"]
    else:
        raise Exception("Missing alpha_s")
        
    if "alpha_x" in list(input_dict.keys()):
        alpha_x = input_dict["alpha_x"]
    else:
        raise Exception("Missing alpha_x")
        
    if "alpha_y" in list(input_dict.keys()):
        alpha_y = input_dict["alpha_y"]
    else:
        raise Exception("Missing alpha_y")

    if "alpha_z" in list(input_dict.keys()):
        alpha_z = input_dict["alpha_z"]
    else:
        raise Exception("Missing alpha_z")        
    
    reg.alpha_s = alpha_s
    reg.alpha_x = alpha_x
    reg.alpha_y = alpha_y
    reg.alpha_z = alpha_z
    
    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = Utils.sdiag(1/std)
    
    printers = [
        IterationPrinters.iteration, IterationPrinters.beta,
        IterationPrinters.phi_d, IterationPrinters.phi_m,
        IterationPrinters.f, IterationPrinters.iterationCG,
        IterationPrinters.totalLS
        ]
    
    stoppers = [StoppingCriteria.iteration]
    
    if "lower_bound" and "upper_bound" in list(input_dict.keys()):
        lb = input_dict["lower_bound"]
        ub = input_dict["upper_bound"]
    else:
        raise Exception("Missing lower, upper bounds")
    
    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(
        maxIter=1000, lower=lb, upper=ub,
        maxIterLS=100, maxIterCG=2000, tolCG=1e-3,
        printers = printers, stoppers = stoppers
        )
    
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e-1)
    
    IRLS = Directives.Update_IRLS(f_min_change=1e-3, maxIRLSiter=100)
    
    update_Jacobi = Directives.UpdatePreconditioner()
    
    save_output = Directives.SaveOutputEveryIteration_Single()
    
    inv = Inversion.BaseInversion(
        invProb, 
        directiveList=[
            IRLS, update_Jacobi, 
            betaest, save_output
            ]
        )
    
    # Run the inversion
    m0 = np.ones(nC)*1e-8  # Starting model
    mrec = inv.run(m0)
    
    
    # make directories (path)
    data_dir = os.path.join(
    work_dir, 
    "Data-{}".format(
        datetime.now().strftime("%Y-%m-%d-%H-%M")
        )
    )
    os.mkdir(data_dir)
    data_dir += path_sep
    
    fig_dir = os.path.join(
        work_dir, 
        "Figs-{}".format(
            datetime.now().strftime("%Y-%m-%d-%H-%M")
            )
        )
    os.mkdir(fig_dir)
    fig_dir += path_sep
    
    
    # save data files 
    dpred = prob.fields(mrec)
    np.savetxt(data_dir+"data_predicted.txt", dpred)
    
    dnres = (dobs - dpred)/std
    np.savetxt(data_dir+"data_residual_nor.txt", dnres)
    
    model_dens = actvMap*mrec
    mesh.writeModelUBC(data_dir+"model_rec_dens.txt", model_dens)

    if plotIt:

        # dobs
        fig = plt.figure(figsize=(14,14))
        ax = plt.subplot()
        im = Utils.PlotUtils.plot2Ddata(xyzLoc, dobs, ax=ax, contourOpts={"cmap": "jet"})
        ax.set_xlabel("Easting (m)", size=22)
        ax.set_ylabel("Northing (m)", size=22)
        ax.set_title("")
        ax.tick_params(labelsize=16)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.locator_params(nbins=10, axis="x")
        ax.set_title("Observed gravity gradient anomaly", size=22)
        plt.gca().set_aspect("equal", adjustable="box")
        pos = ax.get_position()
        cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.06, pos.width *0.02, pos.height*0.8])
        cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical")
        cb.ax.tick_params(labelsize=16) 
        cb.ax.set_title("E", size=22)
        plt.savefig(fig_dir+"plot_obs_grav.png", bbox_inches="tight", dpi=300)
        
        
        # dpred
        fig = plt.figure(figsize=(14,14))
        ax = plt.subplot()
        im = Utils.PlotUtils.plot2Ddata(xyzLoc, dpred, ax=ax, contourOpts={"cmap": "jet"})
        ax.set_xlabel("Easting (m)", size=22)
        ax.set_ylabel("Northing (m)", size=22)
        ax.set_title("")
        ax.tick_params(labelsize=16)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.locator_params(nbins=10, axis="x")
        ax.set_title("Predicted gravity gradient anomaly", size=22)
        plt.gca().set_aspect("equal", adjustable='box')
        pos = ax.get_position()
        cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.06, pos.width *0.02, pos.height*0.8])
        cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical")
        cb.ax.tick_params(labelsize=16) 
        cb.ax.set_title("E", size=22)
        plt.savefig(fig_dir+"plot_pred_grav.png", bbox_inches="tight", dpi=300)
        
        
        # nor residual
        fig = plt.figure(figsize=(14,14))
        ax = plt.subplot()
        im = Utils.PlotUtils.plot2Ddata(xyzLoc, dnres, ax=ax, contourOpts={"cmap": "jet"})
        ax.set_xlabel("Easting (m)", size=22)
        ax.set_ylabel("Northing (m)", size=22)
        ax.set_title("")
        ax.tick_params(labelsize=16)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.locator_params(nbins=10, axis="x")
        ax.set_title("Normalized residual", size=22)
        plt.gca().set_aspect("equal", adjustable="box")
        pos = ax.get_position()
        cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.06, pos.width *0.02, pos.height*0.8])
        cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical")
        cb.ax.tick_params(labelsize=16) 
        cb.ax.set_title("E", size=22)
        plt.savefig(fig_dir+"plot_residual_grav.png", bbox_inches="tight", dpi=300)
            
        if input_dict["example"] == "horse":           
            # plot inverted model
            vmax, vmin = np.nanmax(model_dens), np.nanmin(model_dens)
            fig = plt.figure(figsize=(18,6))
            # Plot horizontal section of inverted grav model
            ax1 = plt.subplot(121)
            slicePosition = -2000 #meter in z direction
            sliceInd = int(round(np.searchsorted(mesh.vectorCCz, slicePosition)))
            mesh.plotSlice(model_dens, ax=ax1, normal="Z", ind=sliceInd,
                        grid=False, clim=(vmin, vmax), pcolorOpts={"cmap":"jet"})
            ax1.set_xlabel("Easting (m)", size=22)
            ax1.set_ylabel("Northing (m)", size=22)
            ax1.set_title("")
            ax1.set_title("(a), z={}m".format(slicePosition+430), loc="left", size = 22)
            ax1.tick_params(labelsize=16)
            ax1.ticklabel_format(useOffset=False, style="plain")
            ax1.locator_params(nbins=8, axis="x")
            ax1.set_aspect("equal", adjustable="box")
            pos1 = ax1.get_position()
            
            # Vertical section
            ax2 = plt.subplot(122)
            slicePosition = 4790800 #meter in y direction
            sliceInd = int(round(np.searchsorted(mesh.vectorCCy, slicePosition)))
            im = mesh.plotSlice(model_dens, ax=ax2, normal="Y", ind=sliceInd,
                        grid=False, clim=(vmin, vmax), pcolorOpts={"cmap":"jet"})
            ax2.set_xlabel("Easting (m)", size=22)
            ax2.set_ylabel("Depth (m)", size=22)
            ax2.set_title("")
            ax2.set_title("(b), y={}".format(slicePosition), loc="left", size = 22)
            ax2.tick_params(labelsize=16)
            ax2.ticklabel_format(useOffset=False, style="plain")
            ax2.locator_params(nbins=8, axis="x")
            ax2.set_aspect("equal", adjustable="box")   
            pos2 = ax2.get_position() 
            pos2_new = [pos2.x0+0.02, pos1.y0, pos2.width, pos2.height] 
            ax2.set_position(pos2_new)
            
            cb_ax = fig.add_axes([pos2.x0+0.09, pos2.y0+0.35, pos2.width *0.6, pos2.height*0.1])
            kwargs = {'format': '%.3f'}
            cb = plt.colorbar(im[0], cax=cb_ax, orientation="horizontal",
                          ticks=np.linspace(vmin, vmax, 4), **kwargs)
            cb.ax.tick_params(labelsize=16) 
            cb.set_label("Density (g/cc)", size=22)
            plt.savefig(fig_dir+"plot_model_dens.png", bbox_inches="tight", dpi=300)  

        
        else:
            # plot inverted model
            vmax, vmin = np.nanmax(model_dens), 0
            fig = plt.figure(figsize=(20,8))
            # Plot horizontal section of inverted grav model
            ax1 = plt.subplot(121)
            slicePosition = -375 #meter in z direction
            sliceInd = int(round(np.searchsorted(mesh.vectorCCz, slicePosition)))
            mesh.plotSlice(model_dens, ax=ax1, normal="Z", ind=sliceInd,
                        grid=False, clim=(vmin, vmax), pcolorOpts={"cmap":"jet"})
            circle = plt.Circle((1525, 1525), 275, color='white', linewidth=3, fill=False)
            ax1.add_patch(circle)
            ax1.set_xlabel("Easting (m)", size=24)
            ax1.set_ylabel("Northing (m)", size=24)
            ax1.set_title("")
            ax1.set_title("(a), z={}m".format(slicePosition), loc="left", size = 24)
            ax1.tick_params(labelsize=24)
            ax1.ticklabel_format(useOffset=False, style="plain")
            ax1.locator_params(nbins=8, axis="x")
            ax1.set_aspect("equal", adjustable="box")
            pos1 = ax1.get_position()
            
            # Vertical section
            ax2 = plt.subplot(122)
            slicePosition = 1525 #meter in y direction
            sliceInd = int(round(np.searchsorted(mesh.vectorCCy, slicePosition)))
            im = mesh.plotSlice(model_dens, ax=ax2, normal="Y", ind=sliceInd,
                        grid=False, clim=(vmin, vmax), pcolorOpts={"cmap":"jet"})
            circle = plt.Circle((1525, -375), 275, color='white', linewidth=3, fill=False)
            ax2.add_patch(circle)            
            ax2.set_xlabel("Easting (m)", size=24)
            ax2.set_ylabel("Depth (m)", size=24)
            ax2.set_title("")
            ax2.set_title("(b), y={}".format(slicePosition), loc="left", size = 24)
            ax2.tick_params(labelsize=24)
            ax2.ticklabel_format(useOffset=False, style="plain")
            ax2.locator_params(nbins=8, axis="x")
            ax2.set_aspect("equal", adjustable="box")   
            pos2 = ax2.get_position() 
            pos2_new = [pos2.x0+0.02, pos1.y0, pos2.width, pos2.height] 
            ax2.set_position(pos2_new)
            
            cb_ax = fig.add_axes([pos2.x0+0.09, pos2.y0+0.35, pos2.width *0.6, pos2.height*0.1])
            kwargs = {'format': '%.3f'}
            cb = plt.colorbar(im[0], cax=cb_ax, orientation="horizontal",
                          ticks=np.linspace(vmin, vmax, 4), **kwargs)
            cb.ax.tick_params(labelsize=24) 
            cb.set_label("Density (g/cc)", size=24)
            plt.savefig(fig_dir+"plot_model_dens.png", bbox_inches="tight", dpi=300)             
     
            
    elapse = time() - init_t
    print("elapsed time: ", str(timedelta(seconds=elapse)))


if __name__ == '__main__':
    run(plotIt=True)
