import numpy as np
from scipy.stats import binned_statistic
import scipy 
import scipy.stats as stats
import scipy.interpolate as si

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

def jitter(rarr, earr, indlist, nphot, nRV, njumpRV):
    
    jitters = np.array([])
    
    for j in range(nRV):
        res=rarr[indlist[nphot+j][0]]
        err=earr[indlist[nphot+j][0]]
        nfree = len(res)-njumpRV[j]
        jit=1
        jit, dump = scipy.optimize.leastsq(redchisqmin, jit, args=(nfree, res, err))
        jit = np.abs(jit)
        jitters = np.append(jitters,jit)
            
    return jitters


def credregionML(posterior=None, percentile=0.6827, pdf=None, xpdf=None):
    """
    Compute a smoothed posterior density distribution and the minimum
    density for a given percentile of the highest posterior density.
    These outputs can be used to easily compute the HPD credible regions.

    Parameters:
    ----------
    posterior: 1D float ndarray
        A posterior distribution.
    percentile: Float
        The percentile (actually the fraction) of the credible region.
        A value in the range: (0, 1).
    pdf: 1D float ndarray
        A smoothed-interpolated PDF of the posterior distribution.
    xpdf: 1D float ndarray
        The X location of the pdf values.
    Returns
    -------
    pdf: 1D float ndarray
        A smoothed-interpolated PDF of the posterior distribution.
    xpdf: 1D float ndarray
        The X location of the pdf values.
    HPDmin: Float
        The minimum density in the percentile-HPD region.

    Example
    -------
    >>> import numpy as np
    >>> npoints = 100000
    >>> posterior = np.random.normal(0, 1.0, npoints)
    >>> pdf, xpdf, HPDmin = credregion(posterior)
    >>> # 68% HPD credible-region boundaries (somewhere close to +/-1.0):
    >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))
    >>> # Re-compute HPD for the 95% (withour recomputing the PDF):
    >>> pdf, xpdf, HPDmin = credregion(pdf=pdf, xpdf=xpdf, percentile=0.9545)
    >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))
    """

    if pdf is None and xpdf is None:
        # Thin if posterior has too many samples (> 120k):
        thinning = np.amax([1, int(np.size(posterior)/120000)])
        # Compute the posterior's PDF:
        kernel = stats.gaussian_kde(posterior[::thinning])
        # Remove outliers:
        mean = np.mean(posterior)
        std  = np.std(posterior)
        k = 6
        lo = np.amax([mean-k*std, np.amin(posterior)])
        hi = np.amin([mean+k*std, np.amax(posterior)])
        # Use a Gaussian kernel density estimate to trace the PDF:
        x  = np.linspace(lo, hi, 100)
        # Interpolate-resample over finer grid (because kernel.evaluate
        #  is expensive):
        f    = si.interp1d(x, kernel.evaluate(x))
        xpdf = np.linspace(lo, hi, 3000)
        pdf  = f(xpdf)

    # Sort the PDF in descending order:
    ip = np.argsort(pdf)[::-1]
    # Sorted CDF:
    cdf = np.cumsum(pdf[ip])
    # Indices of the highest posterior density:
    iHPD = np.where(cdf >= percentile*cdf[-1])[0][0]
    mHPD = np.argmax(pdf)
    # Minimum density in the HPD region:
    HPDmin = np.amin(pdf[ip][0:iHPD])
    return pdf, xpdf, HPDmin, mHPD


def grweights(earr,indlist,grnames,groups,ngroup,nphot):
    # calculate the total of the error in each groups
    
    #ewarr = np.zeros(len(indlist[nphot-1][0]))
    #for j in range(len(ewarr)):
    #    ewarr[j]=earr[j]
        
    ewarr=np.copy(earr)  # error weight array
    
    for j in range(ngroup):
        jj=j+1
        # select LCs belonging to this group
        ind = np.where(np.array(groups) == jj)[0]
        nlc=len(ind)            # number of lcs in this group
        nplc=len(indlist[ind[0]][0])      # number of lc points
        errsum=np.zeros(nplc)   # this will contain the error sum for the group
        # sum the errors up at each time step
        for k in range(nlc):
            if (len(indlist[ind[k]][0]) != nplc):
                print('group LCs dont have the same number of points')
            errsum=errsum+np.divide(1,np.power(earr[indlist[ind[k]][0]],2))

    # calculate the weights for each lightcurve

        for k in range(nlc):
            ewarr[indlist[ind[k]][0]]=np.power(ewarr[indlist[ind[k]][0]],2)*errsum
     
 #   print np.divide(1.,ewarr[indlist[0]]), np.divide(1.,ewarr[indlist[1]]), np.divide(1.,ewarr[indlist[2]])
 #   print nothing
    return(ewarr)

def grtest_emcee(chains_out):
    # note: chains_out has the dimensions (nchains,nsteps,ndim)
    nc,ns,nd = np.shape(chains_out)
    mcs = np.mean(chains_out, axis=1)      # mcs should be an (nchains, ndim) array
    mc = np.mean(mcs, axis=0)              # mc should be an (ndim) array
    BV = ns/(nc-1.) * np.sum((mcs-mc)**2, axis=0)  # BV should be an (ndim) array
    mvs = np.var(chains_out, axis=1)       # mvs should be an (nchains, ndim) array
    WV = np.mean(mvs, axis=0)              # WV should be an (ndim) array
    VV = (ns-1.)/ns * WV + (nc + 1.)/(ns*nc) * BV # VV should be an (ndim) array
    GR = VV/WV                             # GR should be an (ndim) array
    
    return GR