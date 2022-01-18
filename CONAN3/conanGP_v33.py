

# define: 
#       -  # of lightcurves  (2 to start)
#       -  model          
#       - jump parameters
#                     - dF
#                     - b
#                     - P
#                     - T0
#                     - a/Rs


# procedure:
#       - set up all modules
#       - read the data
#       - read the input jump parameters and set their initial values
#       - call the mcmc
#       - look at the results

# model:
#       - call a transit lightcurve for each data set
#       - multiply this lightcurve with the baseline
#       - output it as one large array

# VERSION 27 of CONAN uses emcee to minimize the log likelihood instead of the chi2
#                     uses george for GPs

import sys

# sys.path.append("/home/lendl/software/MC3/MCcubed/") 
# sys.path.append("/home/lendl/Work/OurCode/CONAN/")
# sys.path.append("/home/lendl/Work/OurCode/mandagol/")
# #sys.path.append("/home/lendl/software/corner/corner.py/corner/")
# sys.path.append("/home/lendl/software/emcee/emcee-master/emcee/")
# sys.path.append("/home/lendl/Work/OurCode/GPcode/Functions/")

import numpy as np
import matplotlib

#matplotlib.use('TKagg')
#matplotlib.use('Agg')

import matplotlib.pyplot as plt

import mc3
import emcee

from occultquad import *
from occultnl import *
from .basecoeff_v14_LM import *
from . import fitfunc_v25 as ff
from .model_GP_v3 import *
from .logprob_multi_sin_v4 import *
from .plots_v12 import *
from .corfac_v8 import *
from .groupweights_v2 import *
from .ecc_om_par import *
from .outputs_v6_GP import *
from .jitter_v1 import *
from .GRtest_v1 import *

from george.modeling import Model
from george import kernels
import corner
from .gpnew import *
from .get_files import *

#plt.ion()     

# read the input file
#file=open('input_v31GP.dat')

dump=file.readline()
dump=file.readline()
fpath=file.readline()         # the path where the files are
fpath=fpath.rstrip()
dump=file.readline()
dump=file.readline()
# now read in the lightcurves one after the other
names=[]                    # array where the LC filenames are supposed to go
filters=[]                  # array where the filter names are supposed to go
lamdas=[]
bases=[]                    # array where the baseline exponents are supposed to go
groups=[]                   # array where the group indices are supposed to go
useGPphot=[]         # array where the GP switch goes

grbases=[]
dump=file.readline()        # read the next line
dump=file.readline()        # read the next line
while dump[0] != '#':           # if it is not starting with # then
    adump=dump.split()          # split it
    names.append(adump[0])      # append the first field to the name array
    filters.append(adump[1])    # append the second field to the filters array
    lamdas.append(adump[2])     # append the second field to the filters array
    strbase=adump[3:11]         # string array of the baseline function exponents
    base = [int(i) for i in strbase]
    bases.append(base)
    group = int(adump[11])
    groups.append(group)
    grbase=int(adump[10])
    grbases.append(grbase)
    useGPphot.append(adump[12])
    dump=file.readline()    # and read the following line - either a lc or something starting with "#"

nphot=len(names)             # the number of photometry input files
njumpphot=np.zeros(nphot)
filnames=np.array(list(sorted(set(filters),key=filters.index))) 
ulamdas=np.array(list(sorted(set(lamdas),key=lamdas.index))) 
grnames=np.array(list(sorted(set(groups))))
nfilt=len(filnames)
ngroup=len(grnames)
dump=file.readline()

GPphotlc = []
GPphotvars = []
GPphotkerns = []
GPphotWN = []
GPphotWNstartppm = 50      # start at 50 ppm 
GPphotWNstart = []
GPphotWNstep = []
GPphotWNprior = []
GPphotWNpriorwid = []
GPphotWNlimup = []
GPphotWNlimlo = []
GPphotpars1 = []
GPphotstep1 = []
GPphotprior1 = []
GPphotpriorwid1 = []
GPphotlim1up = []
GPphotlim1lo = []
GPphotpars2 = []
GPphotstep2 = []
GPphotprior2 = []
GPphotpriorwid2 = []
GPphotlim2up = []
GPphotlim2lo = []
GPncomponent = []           # number of components in the kernel
GPjumping = []

prev_lcname = 'none'
GPchoices = ["time", "xshift", "yshift", "air", "fwhm", "sky", "eti"]
ndimGP = len(GPchoices)

dump=file.readline()        # read the next line
while dump[0] != '#':           # if it is not starting with # then

    adump=dump.split()          # split it

    if (adump[0]!= prev_lcname):  # a new light curve that gets WN values 
        GPphotWNstart.append(np.log((GPphotWNstartppm/1e6)**2)) # in absolute
        GPphotWNstep.append(0.1)
        GPphotWNprior.append(0.)
        GPphotWNpriorwid.append(0.)
        GPphotWNlimup.append(-5.2)
        GPphotWNlimlo.append(-21)
        GPphotWN.append(adump[3])   # fit for the white noise ?

   # GPphotlc.append(adump[0])      # the LC name this GP setting refers to
    
    if (adump[0]!=prev_lcname):     # a new light curve
        d_jumpingGP = ['n']*ndimGP            # y/n of all of the GPs dimensions
        d_GPkernel = ['NaN']*ndimGP
        d_GPphotpars1 = [0.]*ndimGP
        d_GPphotstep1 = [0.]*ndimGP
        d_GPphotprior1 = [0.]*ndimGP
        d_GPphotpriorwid1 = [0.]*ndimGP
        d_GPphotlim1up = [0.]*ndimGP
        d_GPphotlim1lo = [0.]*ndimGP
        d_GPphotpars2 = [0.]*ndimGP
        d_GPphotstep2 = [0.]*ndimGP
        d_GPphotprior2 = [0.]*ndimGP
        d_GPphotpriorwid2 = [0.]*ndimGP
        d_GPphotlim2up = [0.]*ndimGP
        d_GPphotlim2lo = [0.]*ndimGP
                
    # GPphotvars.append(adump[1])     # the variable name this GP will use
    # GPphotkerns.append(adump[2])    # the kernel name
   
    # now add these into the lists: 0 if not in GP, value otherwise   
    # gamprihia = (0. if (adump[7] == 'n' or adump[6] == 0.) else adump[10])
    
    for i in range(ndimGP):
        if (adump[1]==GPchoices[i]): 
            d_jumpingGP[i]='y'
            d_GPkernel[i]=adump[2]
            d_GPphotpars1[i]=float(adump[4])
            d_GPphotstep1[i]=float(adump[5])
            d_GPphotprior1[i]=float(adump[6])
            d_GPphotpriorwid1[i]=float(adump[7])
            d_GPphotlim1up[i]=float(adump[8])
            d_GPphotlim1lo[i]=float(adump[9])
            d_GPphotpars2[i]=float(adump[10])
            d_GPphotstep2[i]=float(adump[11])
            d_GPphotprior2[i]=float(adump[12])
            d_GPphotpriorwid2[i]=float(adump[13])
            d_GPphotlim2up[i]=float(adump[14])
            d_GPphotlim2lo[i]=float(adump[15])
#        else:
#           d_GPphotpars1[i]=0.
#           d_GPphotstep1[i]=0.
#           d_GPphotprior1[i]=0.
#           d_GPphotpriorwid1[i]=0.
#           d_GPphotlim1up[i]=0.
#           d_GPphotlim1lo[i]=0.
#           d_GPphotpars2[i]=0.
#           d_GPphotstep2[i]=0.
#           d_GPphotprior2[i]=0.
#           d_GPphotpriorwid2[i]=0.
#           d_GPphotlim2up[i]=0.
#           d_GPphotlim2lo[i]=0.

    dump=file.readline()           # read the next line
    if (adump[0]!=prev_lcname):     # a new light curve  
    # if the line is the last one of this lc
        GPjumping.append(d_jumpingGP)
        GPphotkerns.append(d_GPkernel)
        GPphotpars1.append(d_GPphotpars1)
        GPphotstep1.append(d_GPphotstep1)
        GPphotprior1.append(d_GPphotprior1)
        GPphotpriorwid1.append(d_GPphotpriorwid1)
        GPphotlim1up.append(d_GPphotlim1up)
        GPphotlim1lo.append(d_GPphotlim1lo)
        GPphotpars2.append(d_GPphotpars2)
        GPphotstep2.append(d_GPphotstep2)
        GPphotprior2.append(d_GPphotprior2)
        GPphotpriorwid2.append(d_GPphotpriorwid2)
        GPphotlim2up.append(d_GPphotlim2up)
        GPphotlim2lo.append(d_GPphotlim2lo)
    
    prev_lcname = adump[0]

dump=file.readline()               # read the next line

# ========== RV input ====================

RVnames=[]
RVbases=[]
gammas=[]
gamsteps=[]
gampri=[]
gamprilo=[]
gamprihi=[]
sinPs=[]

dump=file.readline()
dump=file.readline()

while dump[0] != '#':           # if it is not starting with # then
    adump=dump.split()
    RVnames.append(adump[0])      # append the first field to the RVname array
    strbase=adump[1:6]         # string array of the baseline function exponents 
    base = [int(i) for i in strbase]
    RVbases.append(base)
    gammas.append(adump[6])
    gamsteps.append(adump[7])
    gampri.append(adump[9])
    gampriloa = (0. if (adump[8] == 'n' or adump[7] == 0.) else adump[9])
    gamprilo.append(gampriloa)
    gamprihia = (0. if (adump[8] == 'n' or adump[7] == 0.) else adump[10])
    gamprihi.append(gamprihia)
    sinPs.append(adump[12])
    dump=file.readline()    # and read the following line - either a lc or something starting with "#"
    
nRV=len(RVnames)             # the number of RV input files
njumpRV=np.zeros(nRV)

# --> adding RV time series' to groups -->> probably not the best way of doing it
#grnames_add = np.max(grnames)+np.zeros(nRV)+1
#grnames = np.append(grnames, grnames_add)
#groups = np.append(groups,grnames_add)
#ngroup=ngroup+nRV

gamma_in = np.zeros((nRV,7)) # set up array to contain the gamma values
extinpars = []

dump=file.readline()

dump=file.readline()
adump=dump.split()
bdump=np.copy(adump)
adump[3] = (0. if adump[1] == 'n' else adump[3])
adump[7] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[7])
adump[8] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[8])
adump[9] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[9])
rprs_in=[float(adump[2]),float(adump[3]),float(adump[4]),float(adump[5]),float(adump[7]),float(adump[8]),float(adump[9])] 
rprs0=np.copy(rprs_in[0])
rprs_ext=[float(bdump[7]),float(bdump[8]),float(bdump[9])]
if rprs_in[1] != 0.:
    njumpphot=njumpphot+1
if (adump[1] == 'n' and adump[6] == 'p'):
    extinpars.append('RpRs')


erprs0=np.copy(0)     # note: erprs0 is set to zero here! because we don't actually need it any more
dump=file.readline()
adump=dump.split()
bdump=np.copy(adump)
adump[3] = (0. if adump[1] == 'n' else adump[3])
adump[7] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[7])
adump[8] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[8])
adump[9] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[9])
b_in=[float(adump[2]),float(adump[3]),float(adump[4]),float(adump[5]),float(adump[7]),float(adump[8]),float(adump[9])]
b_ext=[float(bdump[7]),float(bdump[8]),float(bdump[9])]
if b_in[1] != 0.:
    njumpphot=njumpphot+1
if (adump[1] == 'n' and adump[6] == 'p'):
    extinpars.append('b')

    
dump=file.readline()
adump=dump.split()
bdump=np.copy(adump)
adump[3] = (0. if adump[1] == 'n' else adump[3])
adump[7] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[7])
adump[8] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[8])
adump[9] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[9])
dur_in=[float(adump[2]),float(adump[3]),float(adump[4]),float(adump[5]),float(adump[7]),float(adump[8]),float(adump[9])]
dur_ext=[float(bdump[7]),float(bdump[8]),float(bdump[9])]
if dur_in[1] != 0.:
    njumpphot=njumpphot+1
if (adump[1] == 'n' and adump[6] == 'p'):
    extinpars.append('dur_[d]')


dump=file.readline()
adump=dump.split()
bdump=np.copy(adump)
adump[3] = (0. if adump[1] == 'n' else adump[3])
adump[7] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[7])
adump[8] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[8])
adump[9] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[9])
T0_in=[float(adump[2]),float(adump[3]),float(adump[4]),float(adump[5]),float(adump[7]),float(adump[8]),float(adump[9])]
T0_ext=[float(bdump[7]),float(bdump[8]),float(bdump[9])]

if T0_in[1] != 0.:
    njumpRV=njumpRV+1
    njumpphot=njumpphot+1
if (adump[1] == 'n' and adump[6] == 'p'):
    extinpars.append('T_0')

    
dump=file.readline()
adump=dump.split()
bdump=np.copy(adump)
adump[3] = (0. if adump[1] == 'n' else adump[3])
adump[7] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[7])
adump[8] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[8])
adump[9] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[9])
per_in=[float(adump[2]),float(adump[3]),float(adump[4]),float(adump[5]),float(adump[7]),float(adump[8]),float(adump[9])]
per_ext=[float(bdump[7]),float(bdump[8]),float(bdump[9])]
if per_in[1] != 0.:
    njumpRV=njumpRV+1
    njumpphot=njumpphot+1
if (adump[1] == 'n' and adump[6] == 'p'):
    extinpars.append('Period')
    
dump=file.readline()
adump=dump.split()
adump[3] = (0. if adump[1] == 'n' else adump[3])
adump[7] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[7])
adump[8] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[8])
adump[9] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[9])
eccpri=np.copy(adump[6])
ecc_in=[float(adump[2]),float(adump[3]),float(adump[4]),float(adump[5]),float(adump[7]),float(adump[8]),float(adump[9])]
if ecc_in[1] != 0.:
    njumpRV=njumpRV+1
if (adump[1] == 'n' and adump[6] == 'p'):
    print('cant externally input eccentricity at this time!')
    print(noth)
    
dump=file.readline()
adump=dump.split()
adump[3] = (0. if adump[1] == 'n' else adump[3])
adump[7] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[7])
adump[8] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[8])
adump[9] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[9])
opri=np.copy(adump[6])
omega_in=[float(adump[2]),float(adump[3]),float(adump[4]),float(adump[5]),float(adump[7]),float(adump[8]),float(adump[9])]
omega_in=np.multiply(omega_in,np.pi)/180.
if omega_in[1] != 0.:
    njumpRV=njumpRV+1
if (adump[1] == 'n' and adump[6] == 'p'):
    print('cant externally input omega at this time!')
    print(noth)
    
dump=file.readline()
adump=dump.split()
bdump=np.copy(adump)
adump[3] = (0. if adump[1] == 'n' else adump[3])
adump[7] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[7])
adump[8] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[8])
adump[9] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[9])
Kpri=np.copy(adump[6])
K_in=[float(adump[2]),float(adump[3]),float(adump[4]),float(adump[5]),float(adump[7]),float(adump[8]),float(adump[9])]
K_in=np.divide(K_in,1000.)  # convert to km/s
K_ext=[float(bdump[7]),float(bdump[8]),float(bdump[9])]
K_ext=np.divide(K_ext,1000.)  # convert to km/s
if K_in[1] != 0.:
    njumpRV=njumpRV+1
if (adump[1] == 'n' and adump[6] == 'p'):
    extinpars.append('K')


for i in range(nRV):
    gamma_in[i,:]=[float(gammas[i]),float(gamsteps[i]),-1000,1000,float(gampri[i]),float(gamprilo[i]),float(gamprihi[i])]
    if (gamma_in[i,1] != 0.) :
        njumpRV[i]=njumpRV[i]+1
      
# adapt the eccentricity and omega jump parameters sqrt(e)*sin(o), sqrt(e)*cos(o)

if ((eccpri == 'y' and opri == 'n') or (eccpri == 'n' and opri == 'y')):
    print('priors on eccentricity and omega: either both on or both off')
    print(nothing)
    
eos_in,eoc_in = ecc_om_par(ecc_in, omega_in)

dump=file.readline()
dump=file.readline()
dump=file.readline()
adump=dump.split()
ddfYN = adump[0]      # (y/n) fit ddFs?

drprs_op=[0.,float(adump[1]),float(adump[2]),float(adump[3]), 0., float(adump[5]),float(adump[6])]  # the dRpRs options
drprs_op[5] = (0 if (adump[4] == 'n' or ddfYN == 'n' or float(adump[2]) == 0.) else adump[5])
drprs_op[6] = (0 if (adump[4] == 'n' or ddfYN == 'n' or float(adump[2]) == 0.) else adump[6])
divwhite = adump[7]      # (y/n) divide-white?
dump=file.readline()

grprs=np.zeros(ngroup)   # the group rprs values
egrprs=np.zeros(ngroup)  # the uncertainties of the group rprs values
dwfiles = []             # the filenames of the white residuals

for i in range(ngroup):
    dump=file.readline()
    adump=dump.split()
    grprs[i]=np.copy(float(adump[1]))
    egrprs[i]=np.copy(float(adump[2]))
    dwfiles.append(adump[3])

dwCNMarr=np.array([])      # initializing array with all the dwCNM values
dwCNMind=[]                # initializing array with the indices of each group's dwCNM values
dwind=np.array([])
if (divwhite=='y'):           # do we do a divide-white?    
    for i in range(ngroup):   # read fixed dwCNMs for each group
        tdwCNM, dwCNM = np.loadtxt(fpath+dwfiles[i], usecols=(0,1), unpack = True)
        dwCNMarr=np.concatenate((dwCNMarr,dwCNM), axis=0)
        dwind=np.concatenate((dwind,np.zeros(len(dwCNM),dtype=np.int)+i), axis=0)
        indices=np.where(dwind==i)
        dwCNMind.append(indices)        
    if (ddfYN=='n'):
        print('you can not do divide-white and not fit ddfs!')
        print(nothing)
    
    for i in range(nphot):
        if (bases[i][6]>0):
            print('you can not have CNMs active and do divide-white')
            print(nothing)
    

if (ddfYN=='n' and np.max(grbases)>0):
    print('no dDFs but groups? Not a good idea!')
    print(base)
    print(nothing)

dump=file.readline()
dump=file.readline()
dump=file.readline()

occ_in = np.zeros((nfilt,7))

for i in range(nfilt):
    adump=dump.split()
    j=np.where(filnames == adump[0])               # make sure the sequence in this array is the same as in the "filnames" array
    k=np.where(np.array(filters) == adump[0])

    adump[3] = (0. if adump[1] == 'n' else adump[3])
    adump[7] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[7])
    adump[8] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[8])
    adump[9] = (0. if (adump[6] == 'n' or adump[1] == 'n' or float(adump[3]) == 0.) else adump[9])
    occ_in[j,:]=[float(adump[2]),float(adump[3]),float(adump[4]),float(adump[5]),float(adump[7]),float(adump[8]),float(adump[9])]
    
    if occ_in[j,1] != 0.:
        njumpphot[k]=njumpphot[k]+1

#BUG: nocc could be less than nfilt if we don't have occultations for all filters! But make it work like this first!
nocc=np.copy(nfilt)

dump=file.readline()
dump=file.readline()
# setup the arrays for the LD coefficients
c1_in=np.zeros((nfilt,7))
c2_in=np.zeros((nfilt,7))
c3_in=np.zeros((nfilt,7))
c4_in=np.zeros((nfilt,7))

for i in range(nfilt):
    dump=file.readline()
    adump=dump.split()
    j=np.where(filnames == adump[0])               # make sure the sequence in this array is the same as in the "filnames" array
    k=np.where(np.array(filters) == adump[0])

    c1_in[j,:]=[float(adump[2]),float(adump[3]),-3.,3.,float(adump[2]),float(adump[4]),float(adump[5])]  # the limits are -3 and 3 => very safe
    c1_in[j,5] = (0. if (adump[1] == 'n' or float(adump[3]) == 0.) else c1_in[j,5])
    c1_in[j,6] = (0. if (adump[1] == 'n' or float(adump[3]) == 0.) else c1_in[j,6])
    if c1_in[j,1] != 0.:
        njumpphot[k]=njumpphot[k]+1
        
    c2_in[j,:]=[float(adump[6]),float(adump[7]),-3.,3.,float(adump[6]),float(adump[8]),float(adump[9])]
    c2_in[j,5] = (0. if (adump[1] == 'n' or float(adump[7]) == 0.) else c2_in[j,5])
    c2_in[j,6] = (0. if (adump[1] == 'n' or float(adump[7]) == 0.) else c2_in[j,6])
    if c2_in[j,1] != 0.:
        njumpphot[k]=njumpphot[k]+1

    c3_in[j,:]=[float(adump[10]),float(adump[11]),-3.,3.,float(adump[10]),float(adump[12]),float(adump[13])]
    c3_in[j,5] = (0. if (adump[1] == 'n' or float(adump[11]) == 0.) else c3_in[j,5])
    c3_in[j,6] = (0. if (adump[1] == 'n' or float(adump[11]) == 0.) else c3_in[j,6])
    if c3_in[j,1] != 0.:
        njumpphot[k]=njumpphot[k]+1

    c4_in[j,:]=[float(adump[14]),float(adump[15]),-3.,3.,float(adump[14]),float(adump[16]),float(adump[17])]
    c4_in[j,5] = (0. if (adump[1] == 'n' or float(adump[15]) == 0.) else c4_in[j,5])
    c4_in[j,6] = (0. if (adump[1] == 'n' or float(adump[15]) == 0.) else c4_in[j,6])
    if c4_in[j,1] != 0.:
        njumpphot[k]=njumpphot[k]+1
        
    # convert the input u1, u2 to c1, c2 if the limb-darkening law is quadratic
    if (c3_in[j,0] == 0. and c4_in[j,0]==0 and c3_in[j,1] == 0. and c4_in[j,1] == 0.):
        print('Limb-darkening law: quadratic')
        v1=2.*c1_in[j,0]+c2_in[j,0]
        v2=c1_in[j,0]-c2_in[j,0]
        ev1=np.sqrt(4.*c1_in[j,1]**2+c2_in[j,1]**2)
        ev2=np.sqrt(c1_in[j,1]**2+c2_in[j,1]**2)
        lov1=np.sqrt(4.*c1_in[j,5]**2+c2_in[j,5]**2)
        lov2=np.sqrt(c1_in[j,5]**2+c2_in[j,5]**2)
        hiv1=np.sqrt(4.*c1_in[j,6]**2+c2_in[j,6]**2)
        hiv2=np.sqrt(c1_in[j,6]**2+c2_in[j,6]**2) 
        c1_in[j,0]=np.copy(v1)
        c2_in[j,0]=np.copy(v2)
        c1_in[j,4]=np.copy(v1)
        c2_in[j,4]=np.copy(v2)
        c1_in[j,1]=np.copy(ev1)
        c2_in[j,1]=np.copy(ev2)
        if (adump[1] == 'y'):    # prior on LDs
            c1_in[j,5]=np.copy(lov1)
            c1_in[j,6]=np.copy(hiv1)
            c2_in[j,5]=np.copy(lov2)
            c2_in[j,6]=np.copy(hiv2)

dump=file.readline()
dump=file.readline()

# setup the arrays for the contamination factors
cont=np.zeros((nfilt,2))   # contains for each filter [value, error]

# read the contamination factors
for i in range(nfilt):
    dump=file.readline()
    adump=dump.split()
    j=np.where(filnames == adump[0])               # make sure the sequence in this array is the same as in the "filnames" array
    cont[j,:]=[float(adump[1]),float(adump[2])]

# read the stellar input properties
dump=file.readline()
dump=file.readline()
dump=file.readline()
adump=dump.split()
Rs_in = float(adump[1])
sRs_lo = float(adump[2])
sRs_hi = float(adump[3])
dump=file.readline()
adump=dump.split()
Ms_in = float(adump[1])
sMs_lo = float(adump[2])
sMs_hi = float(adump[3])
dump=file.readline()
adump=dump.split()
howstellar = adump[1]

# read the MCMC setup 
dump=file.readline()
dump=file.readline()
adump=dump.split() 
nsamples=int(adump[1])   # total number of integrations
dump=file.readline()
adump=dump.split()
nchains=int(adump[1])  #  number of chains
ppchain = nsamples/nchains  # number of points per chain
dump=file.readline()
adump=dump.split()
nproc=int(adump[1])   #  number of processes
dump=file.readline()
adump=dump.split()
burnin=int(adump[1])    # Length of bun-in
dump=file.readline()
adump=dump.split()
walk=adump[1]            # Differential Evolution?          
dump=file.readline()
adump=dump.split()
grtest = True if adump[1] == 'y' else False  # GRtest done?
dump=file.readline()
adump=dump.split()
plots = True if adump[1] == 'y' else False  # Make plots done
dump=file.readline()
adump=dump.split()
leastsq = True if adump[1] == 'y' else False  # Do least-square?
dump=file.readline()
adump=dump.split()
savefile = adump[1]   # Filename of save file
dump=file.readline()
adump=dump.split()
savemodel = adump[1]   # Filename of model save file
dump=file.readline()
adump=dump.split()
adaptBL = adump[1]   # Adapt baseline coefficent
dump=file.readline()
adump=dump.split()
paraCNM = adump[1]   # remove parametric model for CNM computation
dump=file.readline()
adump=dump.split()
baseLSQ = adump[1]   # do a leas-square minimization for the baseline (not jump parameters)
dump=file.readline()
adump=dump.split()
lm = True if adump[1] =='y' else False  # use Levenberg-Marquardt algorithm for minimizer?
dump=file.readline()
adump=dump.split()
cf_apply = adump[1]  # which CF to apply
dump=file.readline()
adump=dump.split()
jit_apply = adump[1] # apply jitter

### BUG: this should be read in! And should contain RV inputs ###
# all of these should be lists with nphot bzw nRV items

useGPrv=['n']

GPrvpars1=np.array([0.])
GPrvpars2=np.array([0.])
GPrvstep1=np.array([0.001])
GPrvstep2=np.array([0.001])

GPrvWN=['y']    #fitWN
GPrvwnstartms = np.array([10])
GPrvwnstart = np.log((GPrvwnstartms/1e3)**2)  # in km/s
GPrvWNstep = np.array([0.1])

#inc_in=np.multiply(inc_in,np.pi)/180.
#cos_in=[np.cos(inc_in[0]),np.multiply(np.sin(inc_in[0]),inc_in[1]),np.cos(inc_in[3]),np.cos(inc_in[2])]
                                   
tarr=np.array([]) # initializing array with all timestamps
farr=np.array([]) # initializing array with all flux values
earr=np.array([]) # initializing array with all error values
xarr=np.array([]) # initializing array with all x_shift values
yarr=np.array([]) # initializing array with all y_shift values
aarr=np.array([]) # initializing array with all airmass values
warr=np.array([]) # initializing array with all fwhm values
sarr=np.array([]) # initializing array with all sky values
lind=np.array([]) # initializing array with the lightcurve indices
barr=np.array([]) # initializing array with all bisector values
carr=np.array([]) # initializing array with all contrast values

indlist = []    # the list of the array indices
bvars    = []   # a list that will contain lists of [0, 1] for each of the baseline parameters, for each of the LCs. 0 means it's fixed. 1 means it's variable
bvarsRV    = []   # a list that will contain lists of [0, 1] for each of the baseline parameters, for each of the RV curves. 0 means it's fixed. 1 means it's variable


if ddfYN == 'y':   # if ddFs are fit: set the Rp/Rs to the value specified at the jump parameters, and fix it.
    rprs_in=[rprs_in[0],0,0,1,0,0,0]
    nddf=nfilt
else:
    nddf=0

# set up the parameters
params   = np.array([T0_in[0], rprs_in[0], b_in[0], dur_in[0], per_in[0], eos_in[0], eoc_in[0], K_in[0]])  # initial guess params
stepsize = np.array([T0_in[1], rprs_in[1], b_in[1], dur_in[1], per_in[1], eos_in[1], eoc_in[1], K_in[1]])  # stepsizes
pmin     = np.array([T0_in[2], rprs_in[2], b_in[2], dur_in[2], per_in[2], eos_in[2], eoc_in[2], K_in[2]])  # Boundaries (min)
pmax     = np.array([T0_in[3], rprs_in[3], b_in[3], dur_in[3], per_in[3], eos_in[3], eoc_in[3], K_in[3]])  # Boundaries (max)
prior    = np.array([T0_in[4], rprs_in[4], b_in[4], dur_in[4], per_in[4], eos_in[4], eoc_in[4], K_in[4]])  # Prior centers
priorlow = np.array([T0_in[5], rprs_in[5], b_in[5], dur_in[5], per_in[5], eos_in[5], eoc_in[5], K_in[5]])  # Prior sigma low side
priorup  = np.array([T0_in[6], rprs_in[6], b_in[6], dur_in[6], per_in[6], eos_in[6], eoc_in[6], K_in[6]])  # Prior sigma high side
pnames   = np.array(['T_0', 'RpRs', 'b', 'dur_[d]', 'Period_[d]', 'esin(w)', 'ecos(w)', 'K']) # Parameter names

extcens = np.array([T0_ext[0], rprs_ext[0], b_ext[0], dur_ext[0], per_ext[0], 0., 0., K_ext[0]])
extup = np.array([T0_ext[1], rprs_ext[1], b_ext[1], dur_ext[1], per_ext[1], 0., 0., K_ext[1]])
extlow = np.array([T0_ext[2], rprs_ext[2], b_ext[2], dur_ext[2], per_ext[2], 0., 0., K_ext[2]])

if (divwhite=='y'):           # do we do a divide-white? If yes, then fix all the transit shape parameters
    stepsize[0:6] = 0
    prior[0:6] = 0

if ddfYN == 'y':   # if ddFs are fit: set the Rp/Rs to the specified value, and fix it.
    drprs_in=np.zeros((nfilt,7))
    njumpphot=njumpphot+1   # each LC has another jump pm

    for i in range(nfilt):  # and make an array with the drprs inputs |  drprs_op=[0.,float(adump[3]),float(adump[4]),float(adump[5])]  # the dRpRs options
        drprs_in[i,:]=drprs_op
        params=np.concatenate((params, [drprs_in[i,0]]))     # add them to the parameter arrays    
        stepsize=np.concatenate((stepsize, [drprs_in[i,1]]))
        pmin=np.concatenate((pmin, [drprs_in[i,2]]))
        pmax=np.concatenate((pmax, [drprs_in[i,3]]))
        prior=np.concatenate((prior, [drprs_in[i,4]]))
        priorlow=np.concatenate((priorlow, [drprs_in[i,5]]))
        priorup=np.concatenate((priorup, [drprs_in[i,6]]))
        pnames=np.concatenate((pnames, [filnames[i]+'_dRpRs']))
        

for i in range(nfilt):  # add the occultation depths
    params=np.concatenate((params, [occ_in[i,0]]))
    stepsize=np.concatenate((stepsize, [occ_in[i,1]]))
    pmin=np.concatenate((pmin, [occ_in[i,2]]))
    pmax=np.concatenate((pmax, [occ_in[i,3]]))
    prior=np.concatenate((prior, [occ_in[i,4]]))
    priorlow=np.concatenate((priorlow, [occ_in[i,5]]))
    priorup=np.concatenate((priorup, [occ_in[i,6]]))
    pnames=np.concatenate((pnames, [filnames[i]+'_DFocc']))

for i in range(nfilt):  # add the LD coefficients for the filters to the parameters
    params=np.concatenate((params, [c1_in[i,0], c2_in[i,0], c3_in[i,0], c4_in[i,0]]))
    stepsize=np.concatenate((stepsize, [c1_in[i,1], c2_in[i,1], c3_in[i,1], c4_in[i,1]]))
    pmin=np.concatenate((pmin, [c1_in[i,2], c2_in[i,2], c3_in[i,2], c4_in[i,2]]))
    pmax=np.concatenate((pmax, [c1_in[i,3], c2_in[i,3], c3_in[i,3], c4_in[i,3]]))
    prior=np.concatenate((prior, [c1_in[i,4], c2_in[i,4], c3_in[i,4], c4_in[i,4]]))
    priorlow=np.concatenate((priorlow, [c1_in[i,5], c2_in[i,5], c3_in[i,5], c4_in[i,5]]))
    priorup=np.concatenate((priorup, [c1_in[i,6], c2_in[i,6], c3_in[i,6], c4_in[i,6]]))
    pnames=np.concatenate((pnames, [filnames[i]+'_c1',filnames[i]+'_c2',filnames[i]+'_c3',filnames[i]+'_c4']))

for i in range(nRV):
    params=np.concatenate((params,[gamma_in[i,0]]), axis=0)
    stepsize=np.concatenate((stepsize,[gamma_in[i,1]]), axis=0)
    pmin=np.concatenate((pmin,[gamma_in[i,2]]), axis=0)
    pmax=np.concatenate((pmax,[gamma_in[i,3]]), axis=0)
    prior=np.concatenate((prior,[gamma_in[i,4]]), axis=0)
    priorlow=np.concatenate((priorlow,[gamma_in[i,5]]), axis=0)
    priorup=np.concatenate((priorup,[gamma_in[i,6]]), axis=0)
    pnames=np.concatenate((pnames,[RVnames[i]+'_gamma']), axis=0)
    
    if (jit_apply=='y'):
        print('does jitter work?')
        print(nothing)
        params=np.concatenate((params,[0.]), axis=0)
        stepsize=np.concatenate((stepsize,[0.001]), axis=0)
        pmin=np.concatenate((pmin,[0.]), axis=0)
        pmax=np.concatenate((pmax,[100]), axis=0)
        prior=np.concatenate((prior,[0.]), axis=0)
        priorlow=np.concatenate((priorlow,[0.]), axis=0)
        priorup=np.concatenate((priorup,[0.]), axis=0)
        pnames=np.concatenate((pnames,[RVnames[i]+'_jitter']), axis=0)        
    
nbc_tot = np.copy(0)  # total number of baseline coefficients let to vary (leastsq OR jumping)

GPobjects = []
GPparams = []
GPstepsizes = []
GPindex = []  # this array contains the lightcurve index of the lc it applies to
GPprior = []
GPpriwid = []
GPlimup = []
GPlimlo =[]
GPnames = []
pargps = []

for i in range(nphot):
    t, flux, err, xshift, yshift, airm, fwhm, sky, eti = np.loadtxt(fpath+names[i], usecols=(0,1,2,3,4,5,6,7,8), unpack = True)  # reading in the data
    if (divwhite=='y'): # if the divide - white is activated, divide the lcs by the white noise model before proceeding
        dwCNM = np.copy(dwCNMarr[dwCNMind[groups[i]-1]])
        flux=np.copy(flux/dwCNM)
    
    sky=sky-np.mean(sky)
    tarr=np.concatenate((tarr,t), axis=0)
    farr=np.concatenate((farr,flux), axis=0)
    earr=np.concatenate((earr,err), axis=0)
    xarr=np.concatenate((xarr,xshift), axis=0)
    yarr=np.concatenate((yarr,yshift), axis=0)
    aarr=np.concatenate((aarr,airm), axis=0)
    warr=np.concatenate((warr,fwhm), axis=0)
    sarr=np.concatenate((sarr,sky), axis=0)
    barr=np.concatenate((barr,np.zeros(len(t),dtype=np.int)), axis=0)   # bisector array: filled with 0s
    carr=np.concatenate((carr,np.zeros(len(t),dtype=np.int)), axis=0)   # contrast array: filled with 0s
    lind=np.concatenate((lind,np.zeros(len(t),dtype=np.int)+i), axis=0)
    indices=np.where(lind==i)
    indlist.append(indices)
    
    pargp = np.vstack((t, xshift, yshift, airm, fwhm, sky, eti)).T  # the matrix with all the possible inputs to the GPs
    
    if (useGPphot[i]=='n'):
        A_in,B_in,C1_in,C2_in,D_in,E_in,G_in,H_in,nbc = basecoeff(bases[i])  # the baseline coefficients for this lightcurve; each is a 2D array
        nbc_tot = nbc_tot+nbc # add up the number of jumping baseline coeff
        njumpphot[i]=njumpphot[i]+nbc   # each LC has another jump pm

        # if the least-square fitting for the baseline is turned on (baseLSQ = 'y'), then set the stepsize of the jump parameter to 0
        if (baseLSQ == "y"):
            abvar=np.concatenate(([A_in[1,:],B_in[1,:],C1_in[1,:],C2_in[1,:],D_in[1,:],E_in[1,:],G_in[1,:],H_in[1,:]]))
            abind=np.where(abvar!=0.)
            bvars.append(abind)
            A_in[1,:]=B_in[1,:]=C1_in[1,:]=C2_in[1,:]=D_in[1,:]=E_in[1,:]=G_in[1,:]=H_in[1,:]=0                             # the step sizes are set to 0 so that they are not interpreted as MCMC JUMP parameters

        # append these to the respective mcmc input arrays
        params=np.concatenate((params,A_in[0,:],B_in[0,:],C1_in[0,:],C2_in[0,:],D_in[0,:],E_in[0,:],G_in[0,:],H_in[0,:]))
        stepsize=np.concatenate((stepsize,A_in[1,:],B_in[1,:],C1_in[1,:],C2_in[1,:],D_in[1,:],E_in[1,:],G_in[1,:],H_in[1,:]))
        pmin=np.concatenate((pmin,A_in[2,:],B_in[2,:],C1_in[2,:],C2_in[2,:],D_in[2,:],E_in[2,:],G_in[2,:],H_in[2,:]))
        pmax=np.concatenate((pmax,A_in[3,:],B_in[3,:],C1_in[3,:],C2_in[3,:],D_in[3,:],E_in[3,:],G_in[3,:],H_in[3,:]))
        prior=np.concatenate((prior, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
        priorlow=np.concatenate((priorlow, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
        priorup=np.concatenate((priorup, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
        pnames=np.concatenate((pnames, [names[i]+'_A0', names[i]+'_A1',names[i]+'_A2',names[i]+'_A3',names[i]+'_A4',names[i]+'_B1',names[i]+'_B2',names[i]+'_C11', names[i]+'_C12',names[i]+'_C21', names[i]+'_C22',names[i]+'_D1',names[i]+'_D2',names[i]+'_E1',names[i]+'_E2',names[i]+'_G1',names[i]+'_G2',names[i]+'_G3',names[i]+'_H1',names[i]+'_H2']))
        # note currently we have the following parameters in these arrays:
        #   [T0,RpRs,b,dur,per,eos, eoc, ddf_1, ..., ddf_n, occ_1, ... , occ_n, c1_f1,c2_f1,c3_f1,c4_f1, c1_f2, .... , c4fn, A0_lc1,A1_lc1,A2_lc0,A3_lc0,A4_lc0, B0_lc1,B1_lc1,C0_lc1,C1_lc1,C2_lc1,C3_lc1,C4_lc1,D0_lc1,D1_lc1,E0_lc1,E1_lc1,G0_lc1,G1_lc1,H0_lc1,H1_lc1,H2_lc1,A0_lc2, ...]
        #    0  1    2  3   4  5   6    |  7 - 4+nddf     |          [7+nddf -- 6+nddf+4*n_filt]       |                           7+nddf+4*n_filt -- 7+nddf+4*n_filt + 15                                                |    7+nddf+4*n_filt + 16
        #    p a r a m e t e r s   |  d d F s        | Limb Darkening                             | const.  time  (5)     |      AM (2)  |     coordinate shifts   (5)      |     FWHM  (2) |   sky  (2)   | SIN (3)  | CNM (2) |
        #    each lightcurve has 21 baseline jump parameters, starting with index  8+nddf+4*n_filt+nRV


    elif (useGPphot[i]=='y'):
        # first, also allocate spots in the params array for the BL coefficients, but set them all to 0/1 and the stepsize to 0
        A_in,B_in,C1_in,C2_in,D_in,E_in,G_in,H_in,nbc = basecoeff(bases[i])  # the baseline coefficients for this lightcurve; each is a 2D array
        nbc_tot = nbc_tot+nbc # add up the number of jumping baseline coeff
        #if (baseLSQ == "y"):
        #    abvar=np.concatenate(([A_in[1,:],B_in[1,:],C_in[1,:],D_in[1,:],E_in[1,:],G_in[1,:],H_in[1,:]]))
        #    abind=np.where(abvar!=0.)
        #    bvars.append(abind)
        #    A_in[1,:]=B_in[1,:]=C_in[1,:]=D_in[1,:]=E_in[1,:]=G_in[1,:]=H_in[1,:]=0                             # the step sizes are set to 0 so that they are not interpreted as MCMC JUMP parameters
        
        params=np.concatenate((params,A_in[0,:],B_in[0,:],C1_in[0,:],C2_in[0,:],D_in[0,:],E_in[0,:],G_in[0,:],H_in[0,:]))
        stepsize=np.concatenate((stepsize,A_in[1,:],B_in[1,:],C1_in[1,:],C2_in[1,:],D_in[1,:],E_in[1,:],G_in[1,:],H_in[1,:]))
        pmin=np.concatenate((pmin,A_in[2,:],B_in[2,:],C1_in[2,:],C2_in[2,:],D_in[2,:],E_in[2,:],G_in[2,:],H_in[2,:]))
        pmax=np.concatenate((pmax,A_in[3,:],B_in[3,:],C1_in[3,:],C2_in[3,:],D_in[3,:],E_in[3,:],G_in[3,:],H_in[3,:]))
        prior=np.concatenate((prior, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
        priorlow=np.concatenate((priorlow, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
        priorup=np.concatenate((priorup, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
        pnames=np.concatenate((pnames, [names[i]+'_A0', names[i]+'_A1',names[i]+'_A2',names[i]+'_A3',names[i]+'_A4',names[i]+'_B1',names[i]+'_B2',names[i]+'_C11', names[i]+'_C12',names[i]+'_C21', names[i]+'_C22',names[i]+'_D1',names[i]+'_D2',names[i]+'_E1',names[i]+'_E2',names[i]+'_G1',names[i]+'_G2',names[i]+'_G3',names[i]+'_H1',names[i]+'_H2']))

        # define the index in the set of filters that this LC has:
        k = np.where(filnames == filters[i])  # k is the index of the LC in the filnames array
        k = np.asscalar(k[0])
        # ddf of this LC:
        if (ddfYN=='y'):           
            ddfhere=params[8+k]
        else:
            ddfhere=0.
        
        # define the correct LDCs c1, c2:
        c1here=params[8+nddf+nocc+k*4]
        c2here=params[8+nddf+nocc+k*4+1]
        #bfstart=8+nddf+nfilt*4 + nRV  # the first index in the param array that refers to a baseline function    
        #blind = np.asarray(range(bfstart+i*20,bfstart+i*20+20))  # the indices for the coefficients for the base function    
        #basehere = params[blind]
        
        mean_model=Transit_Model(T0=params[0],RpRs=params[1],b=params[2],dur=params[3],per=params[4],eos=params[5],eoc=params[6],ddf=ddfhere,occ=params[8+nddf+k],c1=c1here,c2=c2here)
        #specify the GP objects
        
        
        if (GPphotWN[i]=='y'):
            # check the number of kernel components for this light curve
           # print GPphotkerns
           # print noth
            # go through them and build up the kernel 
            # create the lists as needed, containing [white_noise, gp1_param1, gp1_param2, gp2_param1, gp2_param2, ...]
            # once the kernel is built, define the GP and specify that WN is fitted
            GPparams=np.concatenate((GPparams,[GPphotWNstart[i]]), axis=0)            
            GPstepsizes=np.concatenate((GPstepsizes,[GPphotWNstep[i]]),axis=0)
            GPindex=np.concatenate((GPindex,[i]),axis=0)
            GPprior=np.concatenate((GPprior,[GPphotWNprior[i]]),axis=0)
            GPpriwid=np.concatenate((GPpriwid,[GPphotWNpriorwid[i]]),axis=0)
            GPlimup=np.concatenate((GPlimup,[GPphotWNlimup[i]]),axis=0)
            GPlimlo=np.concatenate((GPlimlo,[GPphotWNlimlo[i]]),axis=0)
            GPnames=np.concatenate((GPnames,['GPphotWN_lc'+str(i)]),axis=0)

            iii=0

            for ii in range(ndimGP):
                #set up the individual kernel
                #if there is a GP to jump in this dimension                
                if GPjumping[i][ii]=='y':
                    if iii>0:
                        k2 = k3
                    print(GPphotkerns[i][ii])

                    if (GPphotkerns[i][ii]=='sqexp'):
                        k1 = GPphotpars1[i][ii] * kernels.ExpSquaredKernel(GPphotpars2[i][ii], ndim=ndimGP, axes=ii)  
                    elif (GPphotkerns[i][ii]=='mat32'):
                        k1 = GPphotpars1[i][ii] * kernels.Matern32Kernel(GPphotpars2[i][ii], ndim=ndimGP, axes=ii)  
                    else:
                        print('kernel not recognized!')
                        print(noth)
                    
                    if iii==0:
                        k3 = k1
                    else:
                        k3 = k2 + k1
                    
                    GPparams=np.concatenate((GPparams,(np.log(GPphotpars1[i][ii]),np.log(GPphotpars2[i][ii]))), axis=0)            
                    GPstepsizes=np.concatenate((GPstepsizes,(GPphotstep1[i][ii],GPphotstep2[i][ii])),axis=0)
                    GPindex=np.concatenate((GPindex,(np.zeros(2)+i)),axis=0)
                    GPprior=np.concatenate((GPprior,(GPphotprior1[i][ii],GPphotprior2[i][ii])),axis=0)
                    GPpriwid=np.concatenate((GPpriwid,(GPphotpriorwid1[i][ii],GPphotpriorwid2[i][ii])),axis=0)
                    GPlimup=np.concatenate((GPlimup,(GPphotlim1up[i][ii],GPphotlim2up[i][ii])),axis=0)
                    GPlimlo=np.concatenate((GPlimlo,(GPphotlim1lo[i][ii],GPphotlim2lo[i][ii])),axis=0)
                    GPnames=np.concatenate((GPnames,(['GPphotscale_lc'+str(i)+'dim'+str(ii),"GPphotmetric_lc"+str(i)+'dim'+str(ii)])),axis=0)

                    iii=iii+1
                
            gp = GPnew(k3, mean=mean_model,white_noise=GPphotWNstart[i],fit_white_noise=True)

        else:

            GPparams=np.concatenate((GPparams,[0.]), axis=0)            
            GPstepsizes=np.concatenate((GPstepsizes,[0.]),axis=0)
            GPindex=np.concatenate((GPindex,[i]),axis=0)
            GPprior=np.concatenate((GPprior,[0.]),axis=0)
            GPpriwid=np.concatenate((GPpriwid,[0.]),axis=0)
            GPlimup=np.concatenate((GPlimup,[0.]),axis=0)
            GPlimlo=np.concatenate((GPlimlo,[0.]),axis=0)
            GPnames=np.concatenate((GPnames,['GPphotWN_lc'+str(i)]),axis=0)
            
            iii=0

            for ii in range(ndimGP):
                #set up the individual kernel

                if GPjumping[i][ii]=='y':

                    if ii>0:
                        k2 = k3
                
                    if (GPphotkerns[i][ii]=='sqexp'):
                        k1 = GPphotpars1[i][ii] * kernels.ExpSquaredKernel(GPphotpars2[i][ii], ndim=ndimGP, axes=ii)  
                    elif (GPphotkerns[i][ii]=='mat32'):
                        k1 = GPphotpars1[i][ii] * kernels.Matern32Kernel(GPphotpars2[i][ii], ndim=ndimGP, axes=ii)  
                    else:
                        print('kernel not recognized!')
                        print(noth)
                    
                    if iii==0:
                        k3 = k1
                    else:
                        k3 = k2 + k1
                    
                    GPparams=np.concatenate((GPparams,(np.log(GPphotpars1[i][ii]),np.log(GPphotpars2[i][ii]))), axis=0)            
                    GPstepsizes=np.concatenate((GPstepsizes,(GPphotstep1[i][ii],GPphotstep2[i][ii])),axis=0)
                    GPindex=np.concatenate((GPindex,(np.zeros(2)+i)),axis=0)
                    GPprior=np.concatenate((GPprior,(GPphotprior1[i][ii],GPphotprior2[i][ii])),axis=0)
                    GPpriwid=np.concatenate((GPpriwid,(GPphotpriorwid1[i][ii],GPphotpriorwid2[i][ii])),axis=0)
                    GPlimup=np.concatenate((GPlimup,(GPphotlim1up[i][ii],GPphotlim2up[i][ii])),axis=0)
                    GPlimlo=np.concatenate((GPlimlo,(GPphotlim1lo[i][ii],GPphotlim2lo[i][ii])),axis=0)
                    GPnames=np.concatenate((GPnames,(['GPphotscale_lc'+str(i),"GPphotmetric_lc"+str(i)])),axis=0)

                    iii=iii+1
    
            gp = GPnew(k3, mean=mean_model,white_noise=0.,fit_white_noise=False)  
            
        # freeze the parameters that are not jumping!
        # indices of the GP mean model parameters in the params model
        pindices = [0,1,2,3,4,5,6,8+k,8+nddf+k,8+nddf+nocc+k*4,8+nddf+nocc+k*4+1]
                
        GPparnames=gp.get_parameter_names(include_frozen=True)
        for ii in range(len(pindices)):
            if (stepsize[pindices[ii]]==0.):
                gp.freeze_parameter(GPparnames[ii])
                print((GPparnames[ii]))

        gp.compute(pargp, err)
        GPobjects.append(gp)
        pargps.append(pargp) 
 
for i in range(nRV):
    t, rv, err, bis, fwhm, contrast = np.loadtxt(fpath+RVnames[i], usecols=(0,1,2,3,4,5), unpack = True)  # reading in the data
    
    tarr = np.concatenate((tarr,t), axis=0)
    farr = np.concatenate((farr,rv), axis=0)    # ! add the RVs to the "flux" array !
    earr = np.concatenate((earr,err), axis=0)   # ! add the RV errors to the "earr" array !
    xarr=np.concatenate((xarr,np.zeros(len(t),dtype=np.int)), axis=0)  # xshift array: filled with 0s
    yarr=np.concatenate((yarr,np.zeros(len(t),dtype=np.int)), axis=0)  # yshift array: filled with 0s
    aarr=np.concatenate((aarr,np.zeros(len(t),dtype=np.int)), axis=0)  # airmass array: filled with 0s
    warr=np.concatenate((warr,fwhm), axis=0)
    sarr=np.concatenate((sarr,np.zeros(len(t),dtype=np.int)), axis=0)  # sky array: filled with 0s
    barr=np.concatenate((barr,bis), axis=0)  # bisector array
    carr=np.concatenate((carr,contrast), axis=0)  # contrast array
    lind=np.concatenate((lind,np.zeros(len(t),dtype=np.int)+i+nphot), axis=0)   # indices
    indices=np.where(lind==i+nphot)
    indlist.append(indices)
    Pin = sinPs[i]

    
    if (useGPrv[i]=='n'):
        W_in,V_in,U_in,S_in,P_in,nbcRV = basecoeffRV(RVbases[i],Pin)  # the baseline coefficients for this lightcurve; each is a 2D array
        nbc_tot = nbc_tot+nbcRV # add up the number of jumping baseline coeff
        abvar=np.concatenate(([W_in[1,:],V_in[1,:],U_in[1,:],S_in[1,:],P_in[1,:]]))
        abind=np.where(abvar!=0.)
        njumpRV[i] = njumpRV[i]+len(abind)
    
        if (baseLSQ == "y"):
            bvarsRV.append(abind)
            W_in[1,:]=V_in[1,:]=U_in[1,:]=S_in[1,:]=P_in[1,:]=0        # the step sizes are set to 0 so that they are not interpreted as MCMC JUMP parameters
        # append these to the respective mcmc input arrays
        params=np.concatenate((params,W_in[0,:],V_in[0,:],U_in[0,:],S_in[0,:],P_in[0,:]))
        stepsize=np.concatenate((stepsize,W_in[1,:],V_in[1,:],U_in[1,:],S_in[1,:],P_in[1,:]))
        pmin=np.concatenate((pmin,W_in[2,:],V_in[2,:],U_in[2,:],S_in[2,:],P_in[2,:]))
        pmax=np.concatenate((pmax,W_in[3,:],V_in[3,:],U_in[3,:],S_in[3,:],P_in[3,:]))
        prior=np.concatenate((prior, np.zeros(len(W_in[0,:])+len(V_in[0,:])+len(U_in[0,:])+len(S_in[0,:])+len(P_in[0,:]))))
        priorlow=np.concatenate((priorlow, np.zeros(len(W_in[0,:])+len(V_in[0,:])+len(U_in[0,:])+len(S_in[0,:])+len(P_in[0,:]))))
        priorup=np.concatenate((priorup, np.zeros(len(W_in[0,:])+len(V_in[0,:])+len(U_in[0,:])+len(S_in[0,:])+len(P_in[0,:]))))
        pnames=np.concatenate((pnames, [RVnames[i]+'_W1',RVnames[i]+'_W2',RVnames[i]+'_V1',RVnames[i]+'_V2',RVnames[i]+'_U1', RVnames[i]+'_U2',RVnames[i]+'_S1',RVnames[i]+'_S2',RVnames[i]+'_P1',RVnames[i]+'_P2',RVnames[i]+'_P3',RVnames[i]+'_P4']))
        # note currently we have the following parameters in these arrays:
        #   [T0,RpRs,b,dur,per,eos, eoc,K,ddf_1, ..., ddf_n, c1_f1,c2_f1,c3_f1,c4_f1, c1_f2, .... , c4fn, A0_lc1,A1_lc1,A2_lc0,A3_lc0,A4_lc0, B0_lc1,B1_lc1,C0_lc1,C1_lc1,C2_lc1,C3_lc1,C4_lc1,D0_lc1,D1_lc1,E0_lc1,E1_lc1,G0_lc1,G1_lc1,H0_lc1,H1_lc1,H2_lc1,A0_lc2, ...]
        #    0  1    2  3   4  5   6    |  7 - 4+nddf     |          [7+nddf -- 6+nddf+4*n_filt]       |                           7+nddf+4*n_filt -- 7+nddf+4*n_filt + 15                                                |    7+nddf+4*n_filt + 16
        #    p a r a m e t e r s   |  d d F s        | Limb Darkening                             | const.  time  (5)     |      AM (2)  |     coordinate shifts   (5)      |     FWHM  (2) |   sky  (2)   | SIN (3)  | CNM (2) | time_rv (2) | bisector (2) | fwhm_rv (2) | contrast (2)   | sinus (3)
        # each rv curve has 8 baseline jump parameters, starting with index  8+nddf+4*n_filt+nRV + nphot* 21


# calculate the weights for the lightcurves to be used for the CNM calculation later: do this in a function!
#ewarr=grweights(earr,indlist,grnames,groups,ngroup)

for i in range(len(params)):
    print(pnames[i], params[i], stepsize[i], pmin[i], pmax[i], priorup[i], priorlow[i])
    
inmcmc='n'

#print nothing
LCjump = [] # a list where each item contain a list of the indices of params that jump and refer to this specific lc

#ATTENTION: pass to the lnprob function the individual subscript (of variable p) that are its jump parameters for each LC
# which indices of p0 are referring to lc n

for i in range(nphot):
    
    temp=np.ndarray([])
    
    lcstep1 = np.where(stepsize[0:7]!=0.)  # the common transit jump parameters
    lcstep = lcstep1[0]
    
    if (len(lcstep) > 0): 
        temp=np.copy(lcstep)
    
    # define the index in the set of filters that this LC has:
    k = np.where(filnames == filters[i])  # k is the index of the LC in the filnames array
    k = np.asscalar(k[0])

    if (ddfYN=='y'):    
        if temp.shape:    
            temp=np.concatenate((np.asarray(temp),np.asarray([8+k])),axis=0)
        else:
            temp=np.asarray([8+k])

    occind=8+nddf+k 
    if (stepsize[occind]!=0.):
        temp=np.concatenate((np.asarray(temp),[occind]),axis=0)
    
    print(temp)

    c1ind=8+nddf+nocc+k*4
    if (stepsize[c1ind]!=0.):
        temp=np.concatenate((np.asarray(temp),[c1ind]),axis=0)
    
    c2ind=8+nddf+nocc+k*4+1
    if (stepsize[c2ind]!=0.):
        temp=np.concatenate((np.asarray(temp),[c2ind]),axis=0)
 
    bfstart= 8+nddf+nocc+nfilt*4 + nRV  # the first index in the param array that refers to a baseline function    
    blind = np.asarray(list(range(bfstart+i*20,bfstart+i*20+20)))  # the indices for the coefficients for the base function    

    #BUG: this here is not set to the correct indices
    lcstep1 = np.where(stepsize[blind]!=0.)
    
    if (len(lcstep1) > 0): 
        lcstep = lcstep1[0]
        temp=np.concatenate((np.asarray(temp),blind[lcstep]),axis=0)

    #and also add the GPparams
    gind = np.where(GPindex==i)
    gindl = list(gind[0]+len(params))
    gind = gind[0]+len(params)

    if gindl:
        temp = np.concatenate((temp,gind),axis=0)
    
    LCjump.append(temp)

RVjump = [] # a list where each item contain a list of the indices of params that jump and refer to this specific RV dataset

for i in range(nRV):
    
    temp=np.ndarray([])
    
    rvstep1 = np.where(stepsize[0:8]!=0.)  # the common RV jump parameters: transit + K
    rvstep = rvstep1[0]
    
    if (len(rvstep) > 0): 
        temp=np.copy(rvstep)

    # identify the gamma index of this RV
    gammaind = 8+nddf+nocc+nfilt*4+i
    
    if (stepsize[gammaind]!=0.):           
        temp=np.concatenate((temp,[gammaind]),axis=0)

    bfstart= 8+nddf+nocc+nfilt*4 + nRV + nphot*20  # the first index in the param array that refers to an RV baseline function    
    blind = list(range(bfstart+i*8,bfstart+i*8+8))  # the indices for the coefficients for the base function    

    rvstep = np.where(stepsize[blind]!=0.)
    if rvstep[0]: 
        temp.append(rvstep)
        temp=np.concatenate(([temp],[rvstep]),axis=0)

    #and also add the GPparams
    #gind = np.where(GPindex==i)
    #gindl = list(gind[0]+len(params))
    #gind = gind[0]+len(params)

    #if gindl:
    #    temp = np.concatenate((temp,gind),axis=0)
    
    RVjump.append(temp)

func=ff.fitfunc
#ln_func=ln.lnprob

# ==============================================================================

# here we start: set up the problem. our initial parameters are the parameters as the input p in logprob_multi


pnames_all = np.concatenate((pnames,GPnames))
initial = np.concatenate((params,GPparams))
steps = np.concatenate((stepsize,GPstepsizes))
priors = np.concatenate((prior,GPprior))
priwid = (priorup+priorlow)/2.
priorwids = np.concatenate((priwid,GPpriwid))
lim_low = np.concatenate((pmin,GPlimlo))
lim_up = np.concatenate((pmax,GPlimup))

#ATTENTION: we now want to isolate the parameters that jump. Only those should form p0
# so, initial contains params AND GPparams. 

# how many dimensions does the problem have?
ndim = np.count_nonzero(steps)
jumping=np.where(steps!=0.)
jumping_noGP = np.where(stepsize!=0.)
jumping_GP = np.where(GPstepsizes!=0.)

pindices = []
for i in range(nphot):
    fullist=list(jumping[0])
    lclist=list(LCjump[i])
    both = list(set(fullist).intersection(lclist))  # attention: set() makes it unordered. we'll need to reorder it
    both.sort()
    indices_A = [fullist.index(x) for x in both]
    pindices.append(indices_A)

for i in range(nRV):
    fullist=list(jumping[0])
    rvlist=list(RVjump[i])
    both = list(set(fullist).intersection(rvlist))  # attention: set() makes it unordered. we'll need to reorder it
    both.sort()
    indices_A = [fullist.index(x) for x in both]
    pindices.append(indices_A)

ewarr=grweights(earr,indlist,grnames,groups,ngroup,nphot)


inmcmc = 'n'
indparams = [tarr,farr,xarr,yarr,warr,aarr,sarr,barr,carr, nphot, nRV, indlist, filters, nfilt, filnames,nddf,nocc,rprs0,erprs0,grprs,egrprs,grnames,groups,ngroup,ewarr, inmcmc, paraCNM, baseLSQ, bvars, bvarsRV, cont,names,RVnames,earr,divwhite,dwCNMarr,dwCNMind,params,useGPphot,useGPrv,GPobjects,GPparams,GPindex,pindices,jumping,pnames,LCjump,priors[jumping],priorwids[jumping],lim_low[jumping],lim_up[jumping],pargps,jumping_noGP,GPphotWN,jit_apply,jumping_GP,GPstepsizes]

mval, merr = logprob_multi(initial[jumping],*indparams)
mcmc_plots(mval,tarr,farr,earr,xarr,yarr,warr,aarr,sarr,barr,carr,lind, nphot, nRV, indlist, filters, names, RVnames, 'init_',initial)


inmcmc = 'y'
indparams = [tarr,farr,xarr,yarr,warr,aarr,sarr,barr,carr, nphot, nRV, indlist, filters, nfilt, filnames,nddf,nocc,rprs0,erprs0,grprs,egrprs,grnames,groups,ngroup,ewarr, inmcmc, paraCNM, baseLSQ, bvars, bvarsRV, cont,names,RVnames,earr,divwhite,dwCNMarr,dwCNMind,params,useGPphot,useGPrv,GPobjects,GPparams,GPindex,pindices,jumping,pnames,LCjump,priors[jumping],priorwids[jumping],lim_low[jumping],lim_up[jumping],pargps,jumping_noGP,GPphotWN,jit_apply,jumping_GP,GPstepsizes]

print('No of dimensions: ', ndim)
print('No of chains: ', nchains)
print(pnames_all[jumping])


# put starting points for all walkers, i.e. chains
p0 = np.random.rand(ndim * nchains).reshape((nchains, ndim))*np.asarray(steps[jumping])*2 + (np.asarray(initial[jumping])-np.asarray(steps[jumping]))

sampler = emcee.EnsembleSampler(nchains, ndim, logprob_multi, args=(indparams))

print("Running first burn-in...")
p0, lp, _ = sampler.run_mcmc(p0, 20)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + steps[jumping] * np.random.randn(nchains, ndim) # this can create problems!
sampler.reset()
pos, prob, state = sampler.run_mcmc(p0, burnin)
sampler.reset()

print("Running production...")
pos, prob, state = sampler.run_mcmc(pos, ppchain)
bp = pos[np.argmax(prob)]

posterior = sampler.flatchain
chains = sampler.chain

print(("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))))
GRvals = grtest_emcee(chains)

ijnames = np.where(steps != 0.)
jnames = pnames_all[[ijnames][0]]  # jnames are the names of the jump parameters
gr_print(jnames,GRvals)

nijnames = np.where(steps == 0.)
njnames = pnames_all[[nijnames][0]]  # njnames are the names of the fixed parameters
exti = np.intersect1d(pnames_all,extinpars, return_indices=True)
exti[1].sort()
extins=np.copy(exti[1])

dim=posterior.shape
# calculate PDFs of the stellar parameters given 
Rs_PDF = get_PDF_Gauss(Rs_in,sRs_lo,sRs_hi,dim)
Ms_PDF = get_PDF_Gauss(Ms_in,sMs_lo,sMs_hi,dim)
extind_PDF = np.zeros((dim[0],len(extins)))
for i in range(len(extinpars)):
    ind = extins[i]
    par_PDF = get_PDF_Gauss(extcens[ind],extup[ind],extlow[ind],dim)
    extind_PDF[:,i] = par_PDF

# newparams == initial here are the parameter values as they come in. they get used only for the fixed values
bpfull = np.copy(initial)
bpfull[[ijnames][0]] = bp

medvals,maxvals=mcmc_outputs(posterior,jnames, ijnames, njnames, nijnames, bpfull, ulamdas, Rs_in, Ms_in, Rs_PDF, Ms_PDF, nfilt, filnames, howstellar, extinpars, extins, extind_PDF)

npar=len(jnames)
if (baseLSQ == "y"):
    print("BUG here if you are doing GPs")
    npar = npar + nbc_tot   # add the baseline coefficients if they are done by leastsq

medp=np.copy(initial)
medp[[ijnames][0]]=medvals
maxp=initial
maxp[[ijnames][0]]=maxvals

# now do some plotting --- this still needs some modifications I think

inmcmc='n'
indparams = [tarr,farr,xarr,yarr,warr,aarr,sarr,barr,carr, nphot, nRV, indlist, filters, nfilt, filnames,nddf,nocc,rprs0,erprs0,grprs,egrprs,grnames,groups,ngroup,ewarr, inmcmc, paraCNM, baseLSQ, bvars, bvarsRV, cont,names,RVnames,earr,divwhite,dwCNMarr,dwCNMind,params,useGPphot,useGPrv,GPobjects,GPparams,GPindex,pindices,jumping,pnames,LCjump,priors[jumping],priorwids[jumping],lim_low[jumping],lim_up[jumping],pargps,jumping_noGP,GPphotWN,jumping_GP,jit_apply,GPstepsizes]

mval, merr = logprob_multi(medp[jumping],*indparams)

mcmc_plots(mval,tarr,farr,earr,xarr,yarr,warr,aarr,sarr,barr,carr,lind, nphot, nRV, indlist, filters, names, RVnames, 'med_',medp)

mval2, merr2 = logprob_multi(maxp[jumping],*indparams)
mcmc_plots(mval2,tarr,farr,earr,xarr,yarr,warr,aarr,sarr,barr,carr,lind, nphot, nRV, indlist, filters, names, RVnames, 'max_', maxp)


maxresiduals = farr - mval2
chisq = np.sum(maxresiduals**2/earr**2)

ndat = len(tarr)
bic=get_BIC_emcee(npar,ndat,chisq)
aic=get_AIC_emcee(npar,ndat,chisq)


rarr=farr-mval2  # the full residuals
bw, br, brt, cf, cfn = corfac(rarr, tarr, earr, indlist, nphot, njumpphot) # get the beta_w, beta_r and CF and the factor to get redchi2=1

outfile='CF.dat'

of=open(outfile,'w')
for i in range(nphot):   #adapt the error values
   # print(earr[indlist[i][0]])
    of.write('%8.3f %8.3f %8.3f %8.3f %10.6f \n' % (bw[i], br[i], brt[i],cf[i],cfn[i]))
    if (cf_apply == 'cf'):
        earr[indlist[i][0]] = np.multiply(earr[indlist[i][0]],cf[i])
        print((earr[indlist[i][0]]))        
    if (cf_apply == 'rchisq'):
        earr[indlist[i][0]] = np.sqrt((earr[indlist[i][0]])**2 + (cfn[i])**2)
        print((earr[indlist[i][0]]))

of.close()

# ==== corner plot ================

c=corner.corner(sampler.flatchain, labels=jnames)
c.savefig("corner_GP.pdf", bbox_inches='tight')
plt.close()

#matplotlib.use('TKagg')
