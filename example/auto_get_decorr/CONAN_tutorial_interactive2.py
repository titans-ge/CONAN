#!/usr/bin/env python
# coding: utf-8

# # CONAN in ~5 minutes

# In[1]:


from glob import glob
from os.path import basename

import numpy as np
import CONAN3
import matplotlib.pyplot as plt
import pandas as pd


path = "data/"
lc_list  = ["lc6bjd.dat"]


# In[4]:


df = pd.read_fwf(path+lc_list[0], names=[f"cols{i}" for i in range(9)])
df.head(5)


# ### load light curve into CONAN

# In[5]:


lc_data = CONAN3.load_lightcurves(file_list     = lc_list, 
                                  data_filepath = path, 
                                  filters       = ["R"], 
                                  lamdas        = [600])
lc_data


# - The lc_data object holds information now about the light curves. The light curves can be plotted using the `plot` method of the object.
# 
# By default this plots column 0 (time) against column 1 (flux) with column 3(flux err) as uncertainties. 

# In[6]:


# lc_data.plot()


# In[7]:


lc_data.clip_outliers(clip=4, width=15)


# - correlations between the flux and other columns in the lightcurve file can be visualized by specifying the columns to plot. e.g. to plot column5 (roll angle) against column 1 (flux)

# In[8]:


# lc_data.split_transits("lc8det_clp.dat", t_ref =6776.624, baseline_amount=0.3, P = 4.178062, show_plot=True,
#                        save_separate=True, same_filter=True)


# In[9]:


# lc_data.plot()


# ### baseline and decorrelation parameters

# - the baseline model for each lightcurve in the lc_data object can be defined using the `lc_baseline` method
# - However the `get_decorr` method can be used to automatically determine the best decorrelation parameters to use. (Based on least-squares fit to the data and bayes factor comparison)
# 

# First, we would like take into account the transit when peforming the least-squares fit. so define the transit parameters to fit alongside

# In[10]:


# lc_data._filters


# In[13]:


tra_occ_pars =dict( T_0           = (6776.614, 6776.624, 6776.634),
                    Period        = 4.178062,
                    Duration      = (0.15, 0.19, 0.22),
                    Impact_para   = (0, 0.35, 0.6),
                    RpRs          = (0.05, 0.108,0.15), 
                    u1            = [(0.4, 0.01)],  #one LD prior per filter
                    u2            = [(0.25, 0.01)],
                    L             = 0 )


# In[14]:


decorr_res = lc_data.get_decorr( **tra_occ_pars, cheops=False, show_steps=False)


# In[22]:
# lc_data.update_setup_transit_rv(K=(0,20,50), Eccentricity=0.001)

# lc_data.plot(plot_cols=(5,1), show_decorr_model=True)
lc_data.lc_baseline( gp = "ce")


lc_data.add_GP(["lc6bjd_clp.dat"],   pars = ["col0"], 
                kernels = ["mat32"], WN = "y",
                log_scale  = [(-20, -16.5, -1)],
                log_metric = [(-12, -7.76, 0)]
                )


# rv_list = ["rv1.dat","rv2.dat"]
# rv_data = CONAN3.load_rvs(file_list = rv_list, data_filepath=path)
# rv_data.rv_baseline(gammas_kms = [(-9.235,0.01), (-9.24,0.01)])


# lc_data.print()



# In[14]:

mcmc = CONAN3.mcmc_setup(n_chains = 40, n_burn   = 1000,
                         n_cpus   = 6, n_steps  = 1000, 
                         leastsq_for_basepar="n", apply_jitter="y")



result = CONAN3.fit_data(lc_data,None, mcmc, out_folder="testing_gp_ce", rerun_result=True);
result
