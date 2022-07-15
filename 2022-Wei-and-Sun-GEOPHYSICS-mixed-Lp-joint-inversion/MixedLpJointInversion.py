# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 19:11:07 2019

@author: wxl
"""


import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import PF

from SimPEG import Directives
from SimPEG import Optimization
from SimPEG import Inversion
from SimPEG import InvProblem
from SimPEG.regularization import coupling as Coupling

from scipy.interpolate import NearestNDInterpolator

import sys
import os
import json

from time import time
from datetime import timedelta



def run(plotIt=True):

    np.random.seed(0)

    t = time()
    
    input_file = sys.argv[1]
    
    # read json file
    with open(input_file, "r") as json_file:
        input_dict = json.load(json_file)
    
    work_dir = os.path.abspath(os.path.dirname(input_file))
    path_sep = os.path.sep    
    work_dir += path_sep

    # read mesh file
    if "mesh_file" in list(input_dict.keys()):
        mesh = Mesh.TensorMesh._readUBC_3DMesh(
            work_dir+input_dict["mesh_file"]
            )
    else:
        raise Exception("Missing mesh file")

    # # Read in mesh
    # mesh = Mesh.TensorMesh._readUBC_3DMesh('mesh_p4.txt')
    # read gravity data file
    if "gravity_data_file" in list(input_dict.keys()): 
        # Create a MAGsurvey
        survey_grav = Utils.io_utils.readUBCgravityObservations(
            work_dir+input_dict["gravity_data_file"]
            )  
    else:
        raise Exception("Missing observed gravity data file")
   
    # Create a MAGsurvey
    # survey_grav = Utils.io_utils.readUBCgravityObservations('gzz_p4.txt')
    if "magnetic_data_file" in list(input_dict.keys()): 
        # Create a MAGsurvey
        survey_mag, H0 = Utils.io_utils.readUBCmagneticsObservations(
            work_dir+input_dict["magnetic_data_file"]
            )  
    else:
        raise Exception("Missing observed gravity data file")
    # survey_mag, H0 = Utils.io_utils.readUBCmagneticsObservations('tmi_p4.txt')
    survey_mag.components = ['tmi']

    # Load topo
    if "topography" in list(input_dict.keys()): 
        topo = np.genfromtxt(
            work_dir+input_dict["topography"], 
            skip_header=1
            )
    else:
        raise Exception("Missing topo file")
        
    # topo = np.genfromtxt('topo_goodformat.txt', skip_header=1)
    F = NearestNDInterpolator(topo[:, :2], topo[:, 2])
    xyzLoc = survey_grav.rxLoc # same with survey_mag.rxLoc
    newTopo = np.c_[xyzLoc[:, :2], F(xyzLoc[:, :2])]
    actv = Utils.surface2ind_topo(mesh, newTopo, gridLoc='CC')
    actv = np.where(actv)[0]
    nC = len(actv)

    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, np.nan)

    # Create Wires Map that maps from stacked models to individual model components
    wires = Maps.Wires(('m1', nC), ('m2', nC))


# =============================================================================
#
#   Gravity
#
# =============================================================================

    prob_grav = PF.Gravity.GravityGradient(
        mesh,
        components=['gzz'],
        rhoMap=wires.m1, actInd=actv
        )

    survey_grav.pair(prob_grav)
    data_grav = survey_grav.dobs
    print("number of data points: ", data_grav.size)
    wd_grav = survey_grav.std

    # Data misfit function
    dmis_grav = DataMisfit.l2_DataMisfit(survey_grav)
    dmis_grav.W = Utils.sdiag(1/wd_grav)

    # Create depth weighting
    z0 = 2
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
    wz_depth_grav = (np.outer(tmp1,wz)).flatten('F')
    wz_depth_grav = actvMap.P.T * wz_depth_grav

    # Reg
    if "norm_p" and "norm_q" in list(input_dict.keys()):
        p, q = input_dict["norm_p"], input_dict["norm_q"]
    else:
        raise Exception("Missing norm values")
    
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

    reg_grav = Regularization.Sparse(mesh, indActive=actv, mapping=wires.m1)
    reg_grav.cell_weights = wz_depth_grav
    norms = [p, q, q, q]
    reg_grav.norms = [norms]
    reg_grav.scaledIRLS = True

    reg_grav.alpha_s = alpha_s
    reg_grav.alpha_x = alpha_x
    reg_grav.alpha_y = alpha_y
    reg_grav.alpha_z = alpha_z

# =============================================================================
#
#   Magnetic
#
# =============================================================================

    # Create the forward model operator
    prob_mag = PF.Magnetics.MagneticIntegral(
        mesh,
        chiMap=wires.m2,
        actInd=actv
        )

    survey_mag.pair(prob_mag)
    data_mag = survey_mag.dobs
    wd_mag = survey_mag.std

    # Data misfit function
    dmis_mag = DataMisfit.l2_DataMisfit(survey_mag)
    dmis_mag.W = Utils.sdiag(1/wd_mag)

    # Create depth weighting
    z0 = 2
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
    wz_depth_mag = (np.outer(tmp1,wz)).flatten('F')
    wz_depth_mag = actvMap.P.T * wz_depth_mag

     # Create a regularization
    reg_mag = Regularization.Sparse(mesh, indActive=actv, mapping=wires.m2)
    reg_mag.cell_weights = wz_depth_mag
    reg_mag.norms = [norms]
    reg_mag.scaledIRLS = True

    reg_mag.alpha_s = alpha_s
    reg_mag.alpha_x = alpha_x
    reg_mag.alpha_y = alpha_y
    reg_mag.alpha_z = alpha_z


# =============================================================================
#
#   Joint Inversion
#
# =============================================================================

    if "lambda" in list(input_dict.keys()):
        l = input_dict["lambda"]
    else:
        raise Exception("Missing lambda")     
    lambd = l
    cross_grad = Coupling.CrossGradient(
        mesh,
        indActive=actvMap.indActive,
        mapping=(wires.m1+wires.m2)
        )

    dmis = dmis_grav + dmis_mag
    reg = reg_grav + reg_mag + lambd*cross_grad
    reg.multipliers[:-1] = [1e+3, 2e+6]

    m0 = np.ones(2*nC)*1e-6

# =============================================================================
#
# Optimization
#
# =============================================================================

    lower = np.ones_like(m0)*-10.
    upper = np.ones_like(m0)*10.

    opt = Optimization.ProjectedGNCG(
        maxIter=5,
        maxIterLS=100,
        lower=lower,
        upper=upper,
        maxIterCG=5, tolCG=1e-2
        )

    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

    IRLS = Directives.Update_IRLS_Joint(
        f_min_change=1e-1,
        maxIRLSiter=120,
        beta_tol = 1e-1,
        beta_initial_estimate = False
    )
    IRLS.target = [2.5e+4, 3e+4]
    IRLS.start = [2.5e+4, 3e+4]

    update_Jacobi = Directives.UpdatePreconditioner()

    save_output = Directives.SaveOutputEveryIteration_JointInversion(name='JointInversion')
    save_model = Directives.SaveModelEveryIteration_JointInversion(name="JointModel")

    joint_inv_dir = Directives.JointInversion_Directive()

    inv = Inversion.BaseInversion(
        invProb,
        directiveList=[
            joint_inv_dir,
            IRLS,
            update_Jacobi,
            save_output,
            save_model
            ]
        )

    mrec = inv.run(m0)

    # save model
    m_grav_joint, m_mag_joint = wires.m1*mrec, wires.m2*mrec
    mesh.writeModelUBC('joint_dens_model_UBC.txt', actvMap * m_grav_joint)
    mesh.writeModelUBC('joint_susc_model_UBC.txt', actvMap * m_mag_joint)

    # save data
    d_pred_grav = dmis_grav.prob.fields(mrec)
    np.savetxt('d_pre_grav.txt', d_pred_grav, fmt='%1.5f')
    d_pred_mag = dmis_mag.prob.fields(mrec)
    np.savetxt('d_pre_mag.txt', d_pred_mag, fmt='%1.5f')
    residual_grav = data_grav - d_pred_grav
    np.savetxt('residual_grav.txt', residual_grav, fmt='%1.5f')
    residual_mag = data_mag - d_pred_mag
    np.savetxt('residual_mag.txt', residual_mag, fmt='%1.5f')


    if plotIt:
        
        # read mesh file remove padding cells
        if "mesh_file_rm" in list(input_dict.keys()):
            mesh_padding_rm = Mesh.TensorMesh._readUBC_3DMesh(
                work_dir+input_dict["mesh_file_rm"]
                )
        else:
            raise Exception("Missing mesh file remove")
        
        
        '''
        Load recovered density model
        
        '''
        nPx, nPy, nPz = 5, 5, 0 # number of padding cells for each side
        sizeM = [110, 45, 29]
        
        model_rec = actvMap * m_grav_joint
        model_rec_3d = np.reshape(
            model_rec, 
            (sizeM[0],sizeM[1],sizeM[2]), 
            order="F"
            )
        
        model_rec_3d_dens_rm = model_rec_3d[
            nPx:sizeM[0]-nPx, 
            nPy:sizeM[1]-nPy-8, 
            nPz:sizeM[2] # 0:30, 0 is deep part, 30 is near surface layer
            ]
        
        vmax0, vmin0 = np.nanmax(model_rec_3d_dens_rm), np.nanmin(model_rec_3d_dens_rm)
        
        '''
        Plot recovered model
        
        '''
        # density model
        fig = plt.figure(figsize=(20,20))
        # Plot horizontal section of inverted grav model
        ax1 = plt.subplot(211)
        slicePosition = -2000 #meter in z direction
        sliceInd = int(round(np.searchsorted(mesh_padding_rm.vectorCCz, slicePosition)))
        mesh_padding_rm.plotSlice(model_rec_3d_dens_rm, ax=ax1, normal='Z', ind=sliceInd,
                   grid=False, clim=(vmin0, vmax0), pcolorOpts={"cmap":"RdYlBu_r"})
        ax1.plot(mesh_padding_rm.vectorCCx, 4820000*np.ones_like(mesh_padding_rm.vectorCCx), "-.k", linewidth=3)
        # ax1.set_xlabel('Easting (m)', size=22)
        ax1.set_xlabel('', size=36)
        ax1.set_ylabel('Northing (m)', size=36)
        ax1.set_title('')
        ax1.set_xticks([])
        ax1.tick_params(labelsize=36)
        ax1.locator_params(nbins=6, axis='x')
        ax1.ticklabel_format(useOffset=False, style='plain')
        ax1.set_aspect('equal', adjustable='box')
        pos1 = ax1.get_position()
        
        # Vertical section
        ax2 = plt.subplot(212)
        slicePosition = 4820000#4792000# #meter in y direction
        sliceInd = int(round(np.searchsorted(mesh_padding_rm.vectorCCy, slicePosition)))
        im = mesh_padding_rm.plotSlice(model_rec_3d_dens_rm, ax=ax2, normal='Y', ind=sliceInd,
                   grid=False, clim=(vmin0, vmax0), pcolorOpts={"cmap":"RdYlBu_r"})
        ax2.plot(mesh_padding_rm.vectorCCx, -2000*np.ones_like(mesh_padding_rm.vectorCCx), "-.k", linewidth=3)
        ax2.set_xlabel('Easting (m)', size=36)
        ax2.set_ylabel('Depth (m)', size=36)
        ax2.set_title('')
        # ax2.set_title('y={} m'.format(slicePosition), loc='left', size = 36)
        ax2.tick_params(labelsize=36)
        ax2.locator_params(nbins=6, axis='x')
        ax2.ticklabel_format(useOffset=False, style='plain')
        ax2.set_aspect('equal', adjustable='box')   
        pos2 = ax2.get_position() 
        pos2_new = [pos1.x0, pos1.y0-0.25, pos1.width, pos2.height] 
        ax2.set_position(pos2_new)
        
        cb_ax = fig.add_axes([pos1.x0+0.07, pos2.y0+0.08, pos1.width *0.8, pos2.height*0.04])
        kwargs = {'format': '%.3f'}
        cb = plt.colorbar(im[0], cax=cb_ax, orientation="horizontal",
                      ticks=np.linspace(vmin0, vmax0, 4), **kwargs)
        cb.ax.tick_params(labelsize=36) 
        cb.set_label('Density ($g/cm^3$)', size=36)
        plt.savefig('fig_model_dens.png', bbox_inches="tight", dpi=300)  
        
        # =============================================================================
        # 
        # 
        #  mag
        # 
        # =============================================================================
        
        '''
        Load recovered susceptibility model
        
        '''
        model_rec = actvMap * m_mag_joint
        
        model_rec_3d = np.reshape(
            model_rec, 
            (sizeM[0],sizeM[1],sizeM[2]), 
            order="F"
            )
        
        model_rec_3d_susc_rm = model_rec_3d[
            nPx:sizeM[0]-nPx, 
            nPy:sizeM[1]-nPy-8, 
            nPz:sizeM[2] # 0:30, 0 is deep part, 30 is near surface layer
            ]
        
        vmax0, vmin0 = np.nanmax(model_rec_3d_susc_rm), np.nanmin(model_rec_3d_susc_rm)
        
        
        '''
        Plot recovered model
        
        '''
        # susceptibility model
        fig = plt.figure(figsize=(20,20))
        # Plot horizontal section of inverted grav model
        ax1 = plt.subplot(211)
        slicePosition = -2000 #meter in z direction
        sliceInd = int(round(np.searchsorted(mesh_padding_rm.vectorCCz, slicePosition)))
        mesh_padding_rm.plotSlice(model_rec_3d_susc_rm, ax=ax1, normal='Z', ind=sliceInd,
                   grid=False, clim=(vmin0, vmax0), pcolorOpts={"cmap":"RdYlBu_r"})
        ax1.plot(mesh_padding_rm.vectorCCx, 4820000*np.ones_like(mesh_padding_rm.vectorCCx), "-.k", linewidth=3)
        ax1.set_xlabel('Easting (m)', size=22)
        ax1.set_xlabel('', size=36)
        ax1.set_ylabel('Northing (m)', size=36)
        ax1.set_title('')
        ax1.set_xticks([])
        ax1.tick_params(labelsize=36)
        ax1.locator_params(nbins=6, axis='x')
        ax1.ticklabel_format(useOffset=False, style='plain')
        ax1.set_aspect('equal', adjustable='box')
        pos1 = ax1.get_position()
        
        # Vertical section
        ax2 = plt.subplot(212)
        slicePosition = 4820000#4792000# #meter in y direction
        sliceInd = int(round(np.searchsorted(mesh_padding_rm.vectorCCy, slicePosition)))
        im = mesh_padding_rm.plotSlice(model_rec_3d_susc_rm, ax=ax2, normal='Y', ind=sliceInd,
                   grid=False, clim=(vmin0, vmax0), pcolorOpts={"cmap":"RdYlBu_r"})
        ax2.plot(mesh_padding_rm.vectorCCx, -2000*np.ones_like(mesh_padding_rm.vectorCCx), "-.k", linewidth=3)
        ax2.set_xlabel('Easting (m)', size=36)
        ax2.set_ylabel('Depth (m)', size=36)
        ax2.set_title('')
        ax2.tick_params(labelsize=36)
        ax2.locator_params(nbins=6, axis='x')
        ax2.ticklabel_format(useOffset=False, style='plain')
        ax2.set_aspect('equal', adjustable='box')   
        pos2 = ax2.get_position() 
        pos2_new = [pos1.x0, pos1.y0-0.25, pos1.width, pos2.height] 
        ax2.set_position(pos2_new)
        
        cb_ax = fig.add_axes([pos1.x0+0.07, pos2.y0+0.08, pos1.width *0.8, pos2.height*0.04])
        kwargs = {'format': '%.3f'}
        cb = plt.colorbar(im[0], cax=cb_ax, orientation="horizontal",
                      ticks=np.linspace(vmin0, vmax0, 4), **kwargs)
        cb.ax.tick_params(labelsize=36) 
        cb.set_label('Susceptibility (SI)', size=36)
        plt.savefig('fig_model_susc.png', bbox_inches="tight", dpi=300)  
        
        
        
        # =============================================================================
        # Scatter
        # =============================================================================
        
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot()
        ax.scatter(model_rec_3d_dens_rm, model_rec_3d_susc_rm, s=10, c="black")
        ax.set_xlabel("Density (g/cc)", size=18)
        ax.set_ylabel("Susceptibility (SI)", size=18)
        ax.tick_params(labelsize=18)
        ax.locator_params(nbins=8, axis="both")
        ax.set_title("Cross plot for joint inversion", size=18)
        fig.savefig("fig_scatter_joint.png", bbox_inches="tight", dpi=300)
        
        
        
        
        # =============================================================================
        # 
        # Gravity data
        # 
        # =============================================================================
        
        
        '''
        Plot data
        
        '''
        fig = plt.figure(figsize=(10,5))
        ax = plt.subplot()
        im = Utils.PlotUtils.plot2Ddata(xyzLoc, data_grav, ax=ax, contourOpts={"cmap": "RdYlBu_r"})
        ax.set_xlabel('Easting (m)', size=22)
        ax.set_ylabel('Northing (m)', size=22)
        ax.set_title('Observed gravity anomaly', size=32)
        ax.tick_params(labelsize=22)
        ax.locator_params(nbins=4, axis='x')
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.gca().set_aspect('equal', adjustable='box')
        pos = ax.get_position()
        cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.02, pos.width *0.02, pos.height*0.8])
        kwargs = {'format': '%.1f'}
        cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical", **kwargs)
        cb.ax.tick_params(labelsize=22) 
        cb.ax.set_title('E', size=22)
        plt.savefig('fig_dobs_grav.png', bbox_inches="tight", dpi=300)
        
        # preditcted data
        fig = plt.figure(figsize=(14,14))
        ax = plt.subplot()
        im = Utils.PlotUtils.plot2Ddata(xyzLoc, d_pred_grav, ax=ax, contourOpts={"cmap": "RdYlBu_r"})
        ax.set_xlabel('Easting (m)', size=22)
        ax.set_ylabel('Northing (m)', size=22)
        ax.tick_params(labelsize=22)
        ax.locator_params(nbins=10, axis='x')
        ax.set_title('Predicted gravity anomaly', size=32 )
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.gca().set_aspect('equal', adjustable='box')
        pos = ax.get_position()
        cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.02, pos.width *0.02, pos.height*0.8])
        kwargs = {'format': '%.3f'}
        cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical", **kwargs)
        cb.ax.tick_params(labelsize=22) 
        cb.ax.set_title('E', size=22)
        plt.savefig('fig_dpred_grav.png', bbox_inches="tight", dpi=300)
        
        
        # data residual
        fig = plt.figure(figsize=(14,14))
        ax = plt.subplot()
        im = Utils.PlotUtils.plot2Ddata(xyzLoc, residual_grav, ax=ax, contourOpts={"cmap": "RdYlBu_r"})
        ax.set_xlabel('Easting (m)', size=22)
        ax.set_ylabel('Northing (m)', size=22)
        ax.tick_params(labelsize=22)
        ax.locator_params(nbins=10, axis='x')
        ax.set_title('Gravity residual', size=32 )
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.gca().set_aspect('equal', adjustable='box')
        pos = ax.get_position()
        cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.02, pos.width *0.02, pos.height*0.8])
        kwargs = {'format': '%.3f'}
        cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical", **kwargs)
        cb.ax.tick_params(labelsize=22) 
        cb.ax.set_title('E', size=22)
        plt.savefig('fig_dres_grav.png', bbox_inches="tight", dpi=300)
        
        
        '''
        Plot data
        
        '''
        fig = plt.figure(figsize=(10,5))
        ax = plt.subplot()
        im = Utils.PlotUtils.plot2Ddata(xyzLoc, data_mag, ax=ax, contourOpts={"cmap": "RdYlBu_r"})
        ax.set_xlabel('Easting (m)', size=22)
        ax.set_ylabel('Northing (m)', size=22)
        ax.tick_params(labelsize=22)
        ax.locator_params(nbins=4, axis='x')
        ax.set_title('Observed magnetic anomaly', size=32 )
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.gca().set_aspect('equal', adjustable='box')
        pos = ax.get_position()
        cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.02, pos.width *0.02, pos.height*0.8])
        kwargs = {'format': '%.1f'}
        cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical", **kwargs)
        cb.ax.tick_params(labelsize=22) 
        cb.ax.set_title('nT', size=22)
        plt.savefig('fig_dobs_mag.png', bbox_inches="tight", dpi=300)
        
        # preditcted data
        fig = plt.figure(figsize=(14,14))
        ax = plt.subplot()
        im = Utils.PlotUtils.plot2Ddata(xyzLoc, d_pred_mag, ax=ax, contourOpts={"cmap": "RdYlBu_r"})
        ax.set_xlabel('Easting (m)', size=22)
        ax.set_ylabel('Northing (m)', size=22)
        ax.tick_params(labelsize=22)
        ax.locator_params(nbins=10, axis='x')
        ax.set_title('Predicted magnetic anomaly', size=32 )
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.gca().set_aspect('equal', adjustable='box')
        pos = ax.get_position()
        cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.02, pos.width *0.02, pos.height*0.8])
        kwargs = {'format': '%.3f'}
        cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical", **kwargs)
        cb.ax.tick_params(labelsize=22) 
        cb.ax.set_title('nT', size=22)
        plt.savefig('fig_dpred_mag.png', bbox_inches="tight", dpi=300)
        
        
        # data residual
        fig = plt.figure(figsize=(14,14))
        ax = plt.subplot()
        im = Utils.PlotUtils.plot2Ddata(xyzLoc, residual_mag, ax=ax, contourOpts={"cmap": "RdYlBu_r"})
        ax.set_xlabel('Easting (m)', size=22)
        ax.set_ylabel('Northing (m)', size=22)
        ax.tick_params(labelsize=22)
        ax.locator_params(nbins=10, axis='x')
        ax.set_title('Magnetic residual', size=32 )
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.gca().set_aspect('equal', adjustable='box')
        pos = ax.get_position()
        cb_ax = fig.add_axes([pos.x0+0.8, pos.y0+0.02, pos.width *0.02, pos.height*0.8])
        kwargs = {'format': '%.3f'}
        cb = plt.colorbar(im[0], cax=cb_ax, orientation="vertical", **kwargs)
        cb.ax.tick_params(labelsize=22) 
        cb.ax.set_title('nT', size=22)
        plt.savefig('fig_dres_mag.png', bbox_inches="tight", dpi=300)


    elapse = time() - t
    print("elapsed time: ", str(timedelta(seconds=elapse)))


if __name__ == '__main__':
    run()
    plt.show()
