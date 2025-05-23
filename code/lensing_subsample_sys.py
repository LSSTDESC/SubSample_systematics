import pyccl as ccl
import healpy as hp
import numpy as np
import astropy.io.fits as fits
import multiprocessing as mp
from scipy.interpolate import CubicSpline
import os

N_res = 100
delta_ebv_low = -0.05
delta_ebv_high = 0.05
dz_debv = 5
zs = np.linspace(0.1,2,1000)
sigma = 0.15

def compute_clij_GGL():
    
    bins = np.linspace(delta_ebv_low, delta_ebv_high, N_res+1)
    ells = np.arange(800)
    for i in range(N_res):
        for j in range(N_res):
            fn = "/global/u2/h/huikong/desc_work/wtheta/cl_ggl/" + "wtheta_%d_%d.txt"%(i,j)
            if os.path.isfile(fn):
                continue
            else:
                print("%d %d"%(i,j))
            delta_z_i = dz_debv*(bins[i]+bins[i+1])/2.
            dndz_i = np.exp(-(zs-0.8+delta_z_i)**2/sigma**2)
            
            delta_z_j = dz_debv*(bins[j]+bins[j+1])/2.
            dndz_j = np.exp(-(zs-0.8-0.5+delta_z_j)**2/sigma**2)
            
            cosmo = ccl.Cosmology(Omega_c=0.26377065934278865, Omega_b=0.0482754208891869, h=0.67556, n_s=0.9667, sigma8=0.8225, transfer_function = "boltzmann_camb")
            bias = 1.4/cosmo.growth_factor(1/(1+zs))
            gals_i = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(zs, dndz_i), bias=(zs, bias))
            #gals_j = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(zs, dndz_j), bias=(zs, bias))
            source_j = ccl.tracers.WeakLensingTracer(cosmo, dndz=(zs, dndz_j), has_shear = True,  ia_bias=None)
            
            clij = ccl.angular_cl(cosmo, gals_i, source_j, ells)
            wtheta = ccl.correlations.correlation(cosmo, np.arange(len(clij)), clij, np.logspace(-1,1,1000), type='NG')
            np.savetxt("/global/u2/h/huikong/desc_work/wtheta/cl_ggl/" + "wtheta_%d_%d.txt"%(i,j), np.array([np.logspace(-1,1,1000), wtheta]).transpose())

def compute_clij_CS():
    bins = np.linspace(delta_ebv_low, delta_ebv_high, N_res+1)
    ells = np.arange(800)
    for i in range(N_res):
        for j in range(i, N_res):
            delta_z_i = dz_debv*(bins[i]+bins[i+1])/2.
            dndz_i = np.exp(-(zs-0.8-0.5+delta_z_i)**2/sigma**2)
            
            delta_z_j = dz_debv*(bins[j]+bins[j+1])/2.
            dndz_j = np.exp(-(zs-0.8-0.5+delta_z_j)**2/sigma**2)
            
            cosmo = ccl.Cosmology(Omega_c=0.26377065934278865, Omega_b=0.0482754208891869, h=0.67556, n_s=0.9667, sigma8=0.8225, transfer_function = "boltzmann_camb")
            bias = 1.4/cosmo.growth_factor(1/(1+zs))
            source_i = ccl.tracers.WeakLensingTracer(cosmo, dndz=(zs, dndz_i), has_shear = True,  ia_bias=None)
            #gals_j = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(zs, dndz_j), bias=(zs, bias))
            source_j = ccl.tracers.WeakLensingTracer(cosmo, dndz=(zs, dndz_j), has_shear = True,  ia_bias=None)
            
            clij = ccl.angular_cl(cosmo, source_i, source_j, ells)
            wtheta = ccl.correlations.correlation(cosmo, np.arange(len(clij)), clij, np.logspace(-1,1,1000), type='GG+')
            np.savetxt("/global/u2/h/huikong/desc_work/wtheta/cl_cs/" + "wtheta_%d_%d.txt"%(i,j), np.array([np.logspace(-1,1,1000), wtheta]).transpose())
            
            wtheta = ccl.correlations.correlation(cosmo, np.arange(len(clij)), clij, np.logspace(-1,1,1000), type='GG-')
            np.savetxt("/global/u2/h/huikong/desc_work/wtheta/cl_cs_minus/" + "wtheta_%d_%d.txt"%(i,j), np.array([np.logspace(-1,1,1000), wtheta]).transpose())
            

def compute_wtheta(i,j, type = 'NG'):
    if i>j:
        window = np.loadtxt("/global/u2/h/huikong/desc_work/wtheta/winodw_res2048/"+'window_%d_%d.txt'%(j,i))
    else:
        window = np.loadtxt("/global/u2/h/huikong/desc_work/wtheta/winodw_res2048/"+'window_%d_%d.txt'%(i,j))
    if type == 'NG':
        if i>j:
            dat_w_theta = np.loadtxt("/global/u2/h/huikong/desc_work/wtheta/cl_ggl/" + "wtheta_%d_%d.txt"%(i,j))
        else:
            dat_w_theta = np.loadtxt("/global/u2/h/huikong/desc_work/wtheta/cl_ggl/" + "wtheta_%d_%d.txt"%(i,j))
    elif type == 'GG+':
        dat_w_theta = np.loadtxt("/global/u2/h/huikong/desc_work/wtheta/cl_cs/" + "wtheta_%d_%d.txt"%(i,j))
    elif type == 'GG-':
        dat_w_theta = np.loadtxt("/global/u2/h/huikong/desc_work/wtheta/cl_cs_minus/" + "wtheta_%d_%d.txt"%(i,j))
    else:
        raise ValueError("invalid type")
    dat_w_theta1 = dat_w_theta[:,0]
    dat_w_theta2 = dat_w_theta[:,1]
    f_w_theta = CubicSpline(dat_w_theta1, dat_w_theta2)
    
    theta_min = 0.1
    theta_max = 10.0
    theta_res = 200
    theta_bins = np.logspace(np.log10(theta_min), np.log10(theta_max), theta_res+1)
    
    theta_bin_i_cen = (theta_bins[1:]+theta_bins[:-1])/2.
    wtheta = f_w_theta(theta_bin_i_cen)
    RiRj = window[:,5]
    
    return wtheta*RiRj, RiRj

def compute_wtheta_obs(type = 'NG'):
    theta_res = 200
    wtheta = np.zeros(theta_res)
    rr = np.zeros(theta_res)
    if type != 'NG':
        for i in range(N_res):
            print(i)
            for j in range(i,N_res):
                wtheta_ij, rr_ij = compute_wtheta(i,j, type = type)
                if i == j: 
                    wtheta += wtheta_ij
                    rr += rr_ij
                else:
                    wtheta += 2*wtheta_ij
                    rr += 2*rr_ij
        wtheta = wtheta/rr
    else:
        for i in range(N_res):
            print(i)
            for j in range(N_res):
                wtheta_ij, rr_ij = compute_wtheta(i,j, type = type)
                wtheta += wtheta_ij
                rr += rr_ij
        wtheta = wtheta/rr
    
    theta_min = 0.1
    theta_max = 10.0
    theta_res = 200
    theta_bins = np.logspace(np.log10(theta_min), np.log10(theta_max), theta_res+1)
    theta_bin_i_cen = (theta_bins[1:]+theta_bins[:-1])/2.
    
    np.savetxt("wtheta_measured_res2048_%s.txt"%type, np.array([theta_bin_i_cen, wtheta]))

def run_wtheta_obs():    
    compute_wtheta_obs(type = 'NG')
    compute_wtheta_obs(type = 'GG+')
    compute_wtheta_obs(type = 'GG-')
    
def get_real_wtheta():
    z_source,nz_source = np.loadtxt("source_nz.txt")
    z_lens,nz_lens = np.loadtxt("lens_nz.txt")
    cosmo = ccl.Cosmology(Omega_c=0.26377065934278865, Omega_b=0.0482754208891869, h=0.67556, n_s=0.9667, sigma8=0.8225, transfer_function = "boltzmann_camb")
    bias = 1.4/cosmo.growth_factor(1/(1+z_lens))
    
    gals = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(z_lens, nz_lens), bias=(z_lens, bias))
    source = ccl.tracers.WeakLensingTracer(cosmo, dndz=(z_source, nz_source), has_shear = True,  ia_bias=None)
    
    ells = np.arange(800)
    cl_ng = ccl.angular_cl(cosmo, gals, source, ells)
    wtheta = ccl.correlations.correlation(cosmo, np.arange(len(cl_ng)), cl_ng, np.logspace(-1,1,1000), type='NG')
    
    np.savetxt("/global/u2/h/huikong/desc_work/wtheta/wthta_ng.txt", np.array([np.logspace(-1,1,1000), wtheta]).transpose())
    
    cl_gg = ccl.angular_cl(cosmo, source, source, ells)
    wtheta_pl = ccl.correlations.correlation(cosmo, np.arange(len(cl_gg)), cl_gg, np.logspace(-1,1,1000), type='GG+')
    np.savetxt("/global/u2/h/huikong/desc_work/wtheta/wthta_gg+.txt", np.array([np.logspace(-1,1,1000), wtheta_pl]).transpose())
    
    
    wtheta_mn = ccl.correlations.correlation(cosmo, np.arange(len(cl_gg)), cl_gg, np.logspace(-1,1,1000), type='GG-')
    np.savetxt("/global/u2/h/huikong/desc_work/wtheta/wthta_gg-.txt", np.array([np.logspace(-1,1,1000), wtheta_mn]).transpose())
    
compute_wtheta_obs(type = 'NG')
    
    
    
    
    
    
    