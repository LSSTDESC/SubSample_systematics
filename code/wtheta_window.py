import pyccl as ccl
import healpy as hp
import numpy as np
import astropy.io.fits as fits
#import matplotlib.pyplot as plt
#import treecorr
import multiprocessing as mp
from scipy.interpolate import CubicSpline
import os

#IDX = int(os.environ['SLURM_NODEID'])
#TOT = 2
#delta ebv res
N_res = 100
delta_ebv_low = -0.05
delta_ebv_high = 0.05
#nside for desi ebv map
nside = 256 
dz_debv = 0.5
fn = "/global/cfs/cdirs/desi/users/huikong/truth_input/dust_maps/desi/desi_dust_gr_%d.fits"%nside
DESI = fits.getdata(fn)
bins = np.linspace(delta_ebv_low, delta_ebv_high, N_res+1)
sel = (DESI['EBV_GR']-DESI['EBV_SFD']>delta_ebv_low)&(DESI['EBV_GR']-DESI['EBV_SFD']<delta_ebv_high)
DESI_trimmed = DESI[sel]
delta_bin = (delta_ebv_high - delta_ebv_low)/N_res
idx = ((DESI_trimmed['EBV_GR'] - DESI_trimmed['EBV_SFD'])/delta_bin+N_res/2).astype(int)
zs = np.linspace(0.1,1.5,500)
sigma = 0.15

def produce_randoms(nside_random):
    #fake "random": same nside_randoms will give you the same answer 
    print('random res:%d'%nside_random)
    hpix_r = np.arange(12*nside_random**2)
    ra,dec = hp.pixelfunc.pix2ang(nside_random, hpix_r, lonlat=True, nest = False)
    
    hpix = hp.pixelfunc.ang2pix(nside, ra,dec, lonlat=True, nest = False)
    return ra,dec,hpix


def compute_window_ij(X):
    bin_i,bin_j, ra, dec, hpix = X
    import os
    if os.path.isfile('./wtheta/winodw_res2048/window_%d_%d.txt'%(bin_i,bin_j)):
        return None
    print("running %d %d"%(bin_i, bin_j))
    
    
    sel_bin_num_i = (idx == bin_i)
    sel_bin_num_j = (idx == bin_j)
    
    sel_ran_i = np.zeros_like(hpix).astype(bool)
    for ipix in DESI_trimmed['HPXPIXEL'][sel_bin_num_i]:
        sel_ran_ipix = (hpix == ipix)
        sel_ran_i = sel_ran_i|sel_ran_ipix
    
    sel_ran_j = np.zeros_like(hpix).astype(bool)
    for ipix in DESI_trimmed['HPXPIXEL'][sel_bin_num_j]:
        sel_ran_ipix = (hpix == ipix)
        sel_ran_j = sel_ran_j|sel_ran_ipix
        
    ra_i = ra[sel_ran_i]
    dec_i = dec[sel_ran_i]
    
    ra_j = ra[sel_ran_j]
    dec_j = dec[sel_ran_j]
    
    cat_i = treecorr.Catalog(ra=ra_i, dec=dec_i, ra_units='degrees', dec_units='degrees', w=np.ones_like(ra_i))
    cat_j = treecorr.Catalog(ra=ra_j, dec=dec_j, ra_units='degrees', dec_units='degrees', w=np.ones_like(ra_j))
    
    cat_all = treecorr.Catalog(ra=ra, dec=dec, ra_units='degrees', dec_units='degrees', w=np.ones_like(ra))
    
    nn = treecorr.NNCorrelation(min_sep=0.1, max_sep=10.0, nbins = 200, sep_units='degrees')
    nn.process(cat_i,cat_j) 

    rr = treecorr.NNCorrelation(min_sep=0.1, max_sep=10.0, nbins = 200, sep_units='degrees')
    #rr.process(cat_all,cat_all)
    #rr is not used anyway
    rr.process(cat_i,cat_j)
    
    nn.write('./wtheta/winodw_res2048/window_%d_%d.txt'%(bin_i,bin_j),rr=rr)
    print('written window_%d_%d.txt'%(bin_i,bin_j))
    
def run_all_window():
    p = mp.Pool(128)
    ra,dec,hpix = produce_randoms(2048)
    X = []
    for i in range(N_res):
        for j in range(i,N_res):
            X.append((i,j, ra, dec, hpix))
    #import pdb;pdb.set_trace()
    #X = (50,59, ra, dec, hpix)
    #compute_window_ij(X)
    #p.map(compute_window_ij,X[IDX::TOT][::-1])
    p.map(compute_window_ij,X)
            
def compute_mask_window():
    ra_i = DESI_trimmed['RA']
    dec_i = DESI_trimmed['DEC']
    cat_m = treecorr.Catalog(ra=ra_i, dec=dec_i, ra_units='degrees', dec_units='degrees', w=np.ones_like(ra_i))

    ra,dec,hpix = produce_randoms(nside)
    cat_random = treecorr.Catalog(ra=ra, dec=dec, ra_units='degrees', dec_units='degrees', w=np.ones_like(ra))
    
    nn = treecorr.NNCorrelation(min_sep=0.5, max_sep=10.0, nbins = 100, sep_units='degrees')
    nn.process(cat_m,cat_m) 
    
    rr = treecorr.NNCorrelation(min_sep=0.5, max_sep=10.0, nbins = 100, sep_units='degrees')
    rr.process(cat_random, cat_random) 
    
    nn.write('./wtheta/winodw/mask.txt', rr=rr)
    

def compute_clij():
    bins = np.linspace(delta_ebv_low, delta_ebv_high, N_res+1)
    ells = np.arange(800)
    for i in range(N_res):
        for j in range(i, N_res):
            delta_z_i = dz_debv*(bins[i]+bins[i+1])/2.
            dndz_i = np.exp(-(zs-0.8+delta_z_i)**2/sigma**2)
            
            delta_z_j = dz_debv*(bins[j]+bins[j+1])/2.
            dndz_j = np.exp(-(zs-0.8+delta_z_j)**2/sigma**2)
            
            cosmo = ccl.Cosmology(Omega_c=0.26377065934278865, Omega_b=0.0482754208891869, h=0.67556, n_s=0.9667, sigma8=0.8225, transfer_function = "boltzmann_camb")
            bias = 1.4/cosmo.growth_factor(1/(1+zs))
            gals_i = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(zs, dndz_i), bias=(zs, bias))
            gals_j = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(zs, dndz_j), bias=(zs, bias))
            clij = ccl.angular_cl(cosmo, gals_i, gals_j, ells)
            wtheta = ccl.correlations.correlation(cosmo, np.arange(len(clij)), clij, np.logspace(-1,1,1000), type='NN')
            np.savetxt("/global/u2/h/huikong/desc_work/wtheta/cl_small/" + "wtheta_%d_%d.txt"%(i,j), np.array([np.logspace(-1,1,1000), wtheta]).transpose())
    
    
def compute_wtheta(i,j):
    window = np.loadtxt("/global/u2/h/huikong/desc_work/wtheta/winodw_res2048/"+'window_%d_%d.txt'%(i,j))
    dat_w_theta = np.loadtxt("/global/u2/h/huikong/desc_work/wtheta/cl_small/" + "wtheta_%d_%d.txt"%(i,j))
    dat_w_theta1 = dat_w_theta[:,0]
    dat_w_theta2 = dat_w_theta[:,1]
    f_w_theta = CubicSpline(dat_w_theta1, dat_w_theta2)
    
    theta_min = 0.1
    theta_max = 10.0
    theta_res = 200
    theta_bins = np.logspace(np.log10(theta_min), np.log10(theta_max), theta_res+1)
    
    theta_bin_i_cen = (theta_bins[1:]+theta_bins[:-1])/2.
    wtheta = f_w_theta(theta_bin_i_cen)
    try:
        RiRj = window[:,5]
    except:
        print(i,j)
        import pdb;pdb.set_trace()
    
    return wtheta*RiRj, RiRj
    
def get_real_wtheta():
    ells = np.arange(800)
    #real_redshift = np.loadtxt("/global/cfs/cdirs/desicollab/users/huikong/truth_input/dust_maps/real_redshift_gr256.txt").transpose()
    real_redshift = np.loadtxt("./lens_nz_small.txt")
    
    #zs = np.linspace(0.1,1.5,500)
    #dndz = np.exp(-(zs-0.8)**2/sigma**2)
    
    zs = real_redshift[0]
    dndz = real_redshift[1]
    print(len(zs))
    cosmo = ccl.Cosmology(Omega_c=0.26377065934278865, Omega_b=0.0482754208891869, h=0.67556, n_s=0.9667, sigma8=0.8225, transfer_function = "boltzmann_camb")
    bias = 1.4/cosmo.growth_factor(1/(1+zs))
    gals = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(zs, dndz), bias=(zs, bias))
    clij = ccl.angular_cl(cosmo, gals, gals, ells)
    wtheta = ccl.correlations.correlation(cosmo, np.arange(len(clij)), clij, np.logspace(-1,1,200), type='NN')
    np.savetxt("/global/u2/h/huikong/desc_work/wtheta/cl/" + "wtheta_real_bias.txt", np.array([np.logspace(-1,1,200), wtheta]).transpose())

    
if __name__ == "__main__":
    
    #run_all_window()
    #compute_mask_window()
    #compute_clij()
    #get_real_wtheta()
    
    #wtheta_mm = np.loadtxt("/global/u2/h/huikong/desc_work/wtheta/cl/wtheta_real_bias.txt").transpose()[1]
    #assert(len(wtheta_mm)>5)
    
    #b_real = 1.5339410147401307
    
    theta_res = 200
    wtheta = np.zeros(theta_res)
    rr = np.zeros(theta_res)
    for i in range(N_res):
        print(i)
        for j in range(i,N_res):
            wtheta_ij, rr_ij = compute_wtheta(i,j)
            bias_i = 0.5+i/100
            bias_j = 0.5+j/100
            #print(bias_i, bias_j)
            
            #wtheta_ij, rr_ij = compute_wtheta(i,j)
            
            if i == j: 
                wtheta += bias_i*bias_j*wtheta_ij
                #wtheta += bias_i*bias_j*rr_ij*wtheta_mm
                rr += rr_ij
            else:
                wtheta += 2*bias_i*bias_j*wtheta_ij
                #wtheta += 2*bias_i*bias_j*rr_ij*wtheta_mm
                rr += 2*rr_ij
    wtheta = wtheta/rr
    
    theta_min = 0.1
    theta_max = 10.0
    theta_res = 200
    theta_bins = np.logspace(np.log10(theta_min), np.log10(theta_max), theta_res+1)
    theta_bin_i_cen = (theta_bins[1:]+theta_bins[:-1])/2.
    
    #np.savetxt("wtheta_measured_res2048_bias.txt", np.array([theta_bin_i_cen, wtheta]))
    np.savetxt("wthta_nz_bz_sys.txt", np.array([theta_bin_i_cen, wtheta]))
    
    
    
    
    
            
