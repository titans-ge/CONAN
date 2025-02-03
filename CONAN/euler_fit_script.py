#!/usr/bin/env python
# coding: utf-8

# # Euler transit observation fit

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os, sys
from astropy.io import fits

import CONAN
from CONAN.get_files import get_parameters
print(f"CONAN version: {CONAN.__version__}")

# In[29]:

plt.rc('ytick', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('font', size=14)

# In[2]:



#inputs
planet_name = sys.argv[1]  #"WASP-127 b"
filepath    = sys.argv[2]  #"data/lc6bjd.dat"
filt        = sys.argv[3]  #"gg"
path    = os.path.dirname(filepath) + "/"

if os.path.splitext(filepath)[1] == ".fits":
    lc = fits.open(filepath)
    
    ti        = lc[1].data['time (BJD-TDB)'] -2450000
    fl        = lc[1].data['flux']
    fl_err    = lc[1].data['sflux']

    fwhm      = lc[1].data['fwhm']
    peak_flux = lc[1].data['peak']
    air_mass  = lc[1].data['airmass']
    sky       = lc[1].data['bkg']
    xshift    = lc[1].data['dx']
    yshift    = lc[1].data['dy']
    exptime   = lc[1].data['exptime']
    
    np.savetxt(f'{path}{planet_name}_lc.dat', np.transpose([ti, fl, fl_err, xshift, yshift, air_mass, fwhm, sky, exptime]), fmt= '%3.5f')
    lc_list = os.path.basename(f'{path}{planet_name}_lc.dat')

else:
    lc_list = os.path.basename(filepath)


# In[4]:


lc_obj = CONAN.load_lightcurves(file_list        = lc_list, 
                                    data_filepath = path, 
                                    filters       = [filt], 
                                    wl        = [0.6],
                                    nplanet=1)
print(lc_obj)


print("\n")
params = get_parameters(planet_name)
print(params)


# In[6]:

print(f"\nGetting limb darkening parameters for filter {filt}")
q1,q2 = lc_obj.get_LDs(Teff=params['star']['Teff'],
                        logg=params['star']['logg'],
                        Z   =params['star']['FeH'],
                        filter_names=[filt],use_result=False)


# In[7]:
t0  = params["planet"]["T0"][0] - 2450000
# In[8]:
traocc_pars =dict(  T_0           = (t0-0.1,t0,t0+0.1),
                    Period        = params["planet"]["period"][0],
                    Impact_para   = (0,params["planet"]["b"][0],1),
                    RpRs          = (0.001,params["planet"]["rprs"][0],0.2),
                    rho_star      = params["star"]["density"]
                    )
# In[9]:
decorr_res = lc_obj.get_decorr( **traocc_pars, cheops=False, show_steps=False, q1=q1,q2=q2,
                                plot_model=False, setup_baseline=False)

# In[12]:
res = decorr_res[0]
res.params.pretty_print()

# In[35]:
fig, ax = plt.subplots(1,2, figsize=(15,6),sharey=True)
plt.suptitle(planet_name)
ax[0].set_title("undetrended")
ax[0].errorbar(res.time, res.flux, res.flux_err, fmt="b.", ecolor="gray")
ax[0].plot(res.time, res.transit,"c",lw=3,zorder=3, label="planet")
ax[0].plot(res.time, res.trend,"r",zorder=3, label="trend")
ax[0].legend()
ax[0].set_ylabel("Relative Flux")
ax[0].set_xlabel("Time")

ax[1].set_title("detrended")
ax[1].errorbar(res.time, res.flux/res.trend, res.flux_err, fmt="b.", ecolor="gray")
ax[1].plot(res.time, res.transit,"c", lw=3,zorder=3)
ax[1].set_xlabel("Time")


plt.subplots_adjust(wspace=0.01)
fig.savefig(f"{planet_name}_{filt}_fit.png", bbox_inches="tight")
