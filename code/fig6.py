import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import numpy as np
import healpy as hp
import astropy.io.fits as fits
from scipy.interpolate import InterpolatedUnivariateSpline as IUS


#-------------
#input parameters
h = 0.67556
Omega_c=0.26377065934278865
Omega_b=0.0482754208891869
ns = 0.9667
redshift = 0.9
sigma8=0.8225
default_As = 2e-9
#-------------
H0 = h*100
ombh2 = Omega_b*h*h
omch2 = Omega_c*h*h

zs = np.linspace(0.1,1.5,500)
sigma = 0.15
dndz = np.exp(-(zs-0.8)**2/sigma**2)
f_dndz = IUS(zs, dndz/dndz.sum(), ext = 1)
zs_spline = np.linspace(zs.min(),zs.max(),800)
dndz_spline = f_dndz(zs_spline)


nside = 256
fn = "/global/cfs/cdirs/desi/users/huikong/truth_input/dust_maps/desi/desi_dust_gr_%d.fits"%nside
dat = fits.getdata(fn)
dat = dat[(dat['EBV_SFD']<0.1)&(dat['EBV_SFD']>0.0)&(dat['EBV_SFD']-dat['EBV_GR']>-0.05)&(dat['EBV_SFD']-dat['EBV_GR']<0.05)]
mean = (dat['EBV_SFD']-dat['EBV_GR']).mean()
debvs = np.zeros(12*256**2)
for i in range(len(dat)):
    iid = dat['HPXPIXEL'][i]
    debvs[iid] = dat['EBV_SFD'][i]-dat['EBV_GR'][i]-mean

def get_cell(mnu = 0):
    cosmo_A = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=ns,\
            A_s=default_As, transfer_function = "boltzmann_camb", m_nu=mnu)
    omega_m_A = cosmo_A['Omega_m']
    cosmo_B = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=ns,\
                          A_s=default_As, transfer_function = "boltzmann_camb", m_nu=0)
    omega_m_B = cosmo_B['Omega_m']
    dm = omega_m_A-omega_m_B
    
    cosmo = ccl.Cosmology(Omega_c=Omega_c-dm, Omega_b=Omega_b, h=h, n_s=ns,\
                          A_s=default_As, transfer_function = "boltzmann_camb", m_nu=mnu)
    

    bias = 1.0/cosmo.growth_factor(1/(1+zs_spline))
    gals = ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(zs_spline, dndz_spline),
                                  bias=(zs_spline, bias))
    ells = np.arange(1000)
    c_ells = ccl.angular_cl(cosmo, gals, gals, ells)
    return c_ells 

def get_cell_camb(mnu = 0):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, neutrino_hierarchy='normal')
    pars.InitPower.set_params(ns=ns, As  = default_As)
    pars.set_matter_power(redshifts=[redshift], kmax=2.0)
    pars.NonLinear = model.NonLinear_both
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    omch2_no_nu = (Omega_c-results.get_Omega('nu'))*h*h  
    
    As = default_As
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2_no_nu, mnu=mnu, neutrino_hierarchy='normal')
    pars.InitPower.set_params(ns=ns, As  = As)
    pars.set_matter_power(redshifts=[redshift], kmax=2.0)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    results = camb.get_results(pars)
    
    kh, z, pk_camb = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2, npoints = 200)
    
    k_ccl = kh*h
    pl_ccl = pk_camb/(h**3)
    
    

def generate_density_field(cl , nside = 256, lmax=800):
    """
    Generate a HEALPix map from a given C_l spectrum.
    
    Parameters:
    cl (array): Angular power spectrum C_l
    nside (int): HEALPix nside parameter
    lmax (int, optional): Maximum l to use from the provided cl. If None, use all provided.
    
    Returns:
    array: HEALPix map of the density field
    
    no Poission noise
    """
    if lmax is None:
        lmax = len(cl) - 1
    
    # Generate complex a_lm with proper normalization
    alm = hp.synalm(cl, lmax=lmax, new=True)
    
    # Convert a_lm to map
    density_map = hp.alm2map(alm, nside, lmax=lmax)
    
    return density_map


def make_cell_ratios(cells = None, ebv_coeff = 15, mode = 'bias', cells2 = None):
    '''
    b = 1.5+ebv_coeff*debvs
    '''
    if mode == 'bias':
        density_field = generate_density_field(cells, nside=256, lmax=800)
        density_field[debvs==0]=0
        b = 1.5+ebv_coeff*debvs
        CELL = hp.anafast((1.5+ebv_coeff*debvs)*density_field, (1.5+ebv_coeff*debvs)*density_field)
        CELL2 = hp.anafast(1.5*density_field, 1.5*density_field)
        print("16,84, 84-16: %f, %f, %f"%( np.percentile(b[debvs!=0], 16), np.percentile(b[debvs!=0], 84), np.percentile(b[debvs!=0], 84)-np.percentile(b[debvs!=0], 16) ))
    else:
        density_field1 = generate_density_field(cells, nside=256, lmax=800)
        density_field1[debvs==0] = 0
        density_field2 = generate_density_field(cells2, nside=256, lmax=800)
        density_field2[debvs==0] = 0
        
        CELL = hp.anafast(1.5*density_field1, 1.5*density_field1)
        CELL2 = hp.anafast(1.5*density_field2, 1.5*density_field2)
    return CELL/CELL2, CELL, CELL2

Ntot = 500
mode = 'bias'
mnu = 0.1
ebv_coeff = 15
c_ells = get_cell(mnu = mnu)
c_ells2 = get_cell(mnu = 0)

ratiosl = []
cell_1 = []
cell_2 = []
for i in range(Ntot):
    print(i)
    ratios, cell, cell2 = make_cell_ratios(cells = c_ells, mode = mode, cells2 = c_ells2, ebv_coeff = ebv_coeff)
    ratiosl.append(ratios)
    cell_1.append(cell)
    cell_2.append(cell2)
    
ratios_mean = np.zeros(len(ratios))
cell_1_mean = np.zeros(len(cell))
cell_2_mean = np.zeros(len(cell2))

for i in range(Ntot):
    ratios_mean+=ratiosl[i]
    cell_1_mean += cell_1[i]
    cell_2_mean += cell_2[i]
ratios_mean /= Ntot
cell_1_mean /= Ntot
cell_2_mean /= Ntot

std = np.zeros(len(ratios))
for i in range(Ntot):
    std += (ratiosl[i] - ratios_mean)**2
std /= Ntot

np.savetxt("bias_ratios_%s.txt"%mode,np.array([np.arange(len(ratios)), ratios_mean, np.sqrt(std), cell_1_mean, cell_2_mean]))


