# this is a routine to calculate the red noise and white noise amplitudes and the respective noise CFs
#
# takes as input: 
#     the complete residuals, time and error arrays
#     indlist: the index array - specifying which pieces of data belong together
#     the number of LCs
#     
# produces outputs:
#     the beta_w, beta_r, and CF values for each lightcurve
#     the beta_r timescales.

import sys
import numpy as np
from scipy.stats import binned_statistic
import matplotlib
import matplotlib.pyplot as plt
import scipy 

def corfac(rarr, tarr, earr, indlist, nphot, njumpphot):
    bw=np.ones(nphot)
    br=np.ones(nphot)
    cf=np.ones(nphot)
    brt=np.zeros(nphot)
    tscales=np.arange(16)+5  # time bin sizes between 5 and 20 minutes
    for j in range(nphot):
        res=np.copy(rarr[indlist[j][0]])
        tt=np.copy(tarr[indlist[j][0]])
        err=np.copy(earr[indlist[j][0]])
        
        # calculate the red noise factor
        brs=np.zeros(len(tscales))
        for l in range(len(tscales)):
            tscale=tscales[l]/60./24.  # timescales 
            min1=1./60./24.  # 1 minute in days
            
            # calculate the shifting bins: shift by 1 minute each
            nshifts=np.copy(tscales[l])   # number of minutes of bin == number of shifts
            brs2=np.zeros(nshifts)  # the br amplitudes for all shifts
            for ll in range(nshifts):
                # calculate the bin edges
                TT0=tt[0]+ll*min1 # starting point is start + ll * 1 minute
                nbin=int((np.max(tt)-TT0)/tscale)
                binlims=np.zeros(nbin+1)
                bincens=np.zeros(nbin)
                binnps=np.zeros(nbin)  #number of points per bin
                binlims[0]=np.copy(TT0)
                binind=[]
                for k in range(1,nbin+1):
                    binlims[k]=binlims[k-1]+tscale
                for k in range(nbin):
                    bincens[k]=binlims[k]+0.5*tscale
                    binnps[k]=len(tt[(tt>binlims[k]) & (tt<binlims[k+1])])
                
                mbin = np.nanmean(binnps)  #mean number of points per bin
                resbin, dump, dump2 = binned_statistic(tt,res,statistic='mean',bins=binlims)
                sigNred = np.nanstd(res)/np.sqrt(mbin) * np.sqrt(nbin/(nbin-1))
                sigNreal = np.nanstd(resbin)
                brs2[ll]=sigNreal/sigNred
                
            brs[l]=np.nanmedian(brs2)
            
        br[j] = np.max(brs)

        if (br[j]<1.):
            br[j]=1.

        brt[j]= tscales[brs.argmax()]
        
        
        # calculate the white noise factor
        #   - apply b_red to the data errors
        #   - calculate the b_white values for the adapted errors
        
        err2=err*br[j]
        tscale=15./60./24.  # timescale is 20 minutes
        min1=1./60./24.  # 1 minute in days
            
        # calculate the shifting bins: shift by 1 minute each
        nshifts=np.copy(tscales[l])   # number of minutes of bin == number of shifts
        bws=np.zeros(nshifts)  # the bw amplitudes for all shifts
        for ll in range(nshifts):
            # calculate the bin edges
            TT0=tt[0]+ll*min1 # starting point is start + ll * 1 minute
            nbin=int((np.max(tt)-TT0)/tscale)
            binlims=np.zeros(nbin+1)
            bincens=np.zeros(nbin)
            binnps=np.zeros(nbin)  #number of points per bin
            binlims[0]=np.copy(TT0)
            binind=[]
            bws2=np.zeros(nbin)
            for k in range(1,nbin+1):
                binlims[k]=binlims[k-1]+tscale
            for k in range(nbin):
                bsig=np.nanstd(res[(tt>binlims[k]) & (tt<binlims[k+1])])
                beme=np.nanmean(err[(tt>binlims[k]) & (tt<binlims[k+1])]) # NOTE: either this or err2, i.e. the red-noise adapted errors
                bws2[k]=bsig/beme
         
            bws[ll]=np.nanmean(bws2)
            
        bw[j]=np.nanmean(bws)
        
        cf[j]= br[j]*bw[j]
        # print br[j], bw[j], brt[j], cf[j]
        
        
    # now compute CF for chi2red==1
        cfn = np.array([])
    
        for j in range(nphot):
            res=rarr[indlist[j][0]]
            err=earr[indlist[j][0]]
            nfree = len(res)-njumpphot
            pjit=0.0002
            pjit, dump = scipy.optimize.leastsq(redchisqmin, pjit, args=(nfree, res, err))
            pjit = np.abs(pjit)
            cfn = np.append(cfn,pjit)
         
    return bw, br, brt, cf, cfn



def redchisqmin(jit,nfree, res, err):
    
    redchisq = np.sum(res**2/(err**2 + jit**2)) / nfree
    # print(redchisq, np.std(res), np.mean(err), nfree, jit)
    
    return 1-redchisq
