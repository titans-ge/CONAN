def mcmc_plots(yval,tarr,farr,earr,xarr,yarr,warr,aarr,sarr,barr,carr, lind, nphot, nRV, indlist, filters,names,RVnames,prefix,params,T0,period):
    
    import matplotlib
    import os
    matplotlib.use('Agg')
    from ._classes import __default_backend__
    import matplotlib.pyplot as plt    
    import numpy as np
    from matplotlib.ticker import FormatStrFormatter    
    from scipy.stats import binned_statistic
    from astropy.stats import sigma_clip

    phase_phasefold = np.empty(0) #array containing phase of each data point
    flux_phasefold = np.empty(0) #array containig baseline corrected flux of each data point
    model_phasefold = np.empty(0) #array containg baseline corrected model of each data point

    for j in range(nphot):
        mod=yval[indlist[j][0]]
        time=tarr[indlist[j][0]]
        flux=farr[indlist[j][0]]
        err=earr[indlist[j][0]]
        am=aarr[indlist[j][0]]
        cx=xarr[indlist[j][0]]
        cy=yarr[indlist[j][0]]
        fwhm=warr[indlist[j][0]]
        sky=sarr[indlist[j][0]]
        
        infile=names[j][:-4]+'_out_full.dat'
        tt, ft, et, mt, bfunc, mm, fco = np.loadtxt(infile, usecols=(0,1,2,3,4,5,6), unpack = True)  # reading in the lightcurve data
        
        #CLIP
        # fluxclip = sigma_clip(fco, sigma=5.0)
        # clipin = np.argwhere(fluxclip>0)

        # tt = tt[clipin.flatten()]
        # fco = fco[clipin.flatten()]
        # ft = ft[clipin.flatten()]
        # mt = mt[clipin.flatten()]
        # mm = mm[clipin.flatten()]
        # bfunc = bfunc[clipin.flatten()]
        # et = et[clipin.flatten()]

        ph = np.modf((np.modf((tt-T0)/period)[0])+1.0)[0] #calculate phase

        #Add data to array to phasefold them later
        phase_phasefold = np.append(phase_phasefold,ph)
        flux_phasefold = np.append(flux_phasefold,fco)
        model_phasefold = np.append(model_phasefold,mm)


        # bin the lightcurve data
        binsize=10./(24.*60.)
        nbin = int((np.max(tt)-np.min(tt))/binsize)  # number of bins
        binlims=np.zeros(nbin+1)
        tbin=np.zeros(nbin)
        binnps=np.zeros(nbin)  #number of points per bin
        binlims[0]=min(tt)
        binind=[]
        for k in range(1,nbin+1):
            binlims[k]=binlims[k-1]+binsize
            for k in range(nbin):
                tbin[k]=binlims[k]+0.5*binsize
                binnps[k]=len(tt[(tt>binlims[k]) & (tt<binlims[k+1])])
        
        ftbin, dump, dump2 = binned_statistic(tt,ft,statistic='mean',bins=binlims)
        mtbin, dump, dump2 = binned_statistic(tt,mt,statistic='mean',bins=binlims)
        fcobin, dump, dump2 = binned_statistic(tt,fco,statistic='mean',bins=binlims)
        mmbin, dump, dump2 = binned_statistic(tt,mm,statistic='mean',bins=binlims)
        resbin, dump, dump2 = binned_statistic(tt,ft - mt,statistic='mean',bins=binlims)
        etbin, dump, dump2 = binned_statistic(tt,et,statistic='mean',bins=binlims)        
        etbin = etbin/np.sqrt(binnps)
        
        ########## Plot and save lightcurve with fit ########
        tit='Fit for lightcurve '+names[j][:-4]
        outname=prefix+names[j][:-4]+'_fit.png'
        plt.figure(10)
        plt.clf()
        plt.plot(tt, ft,'o',c='skyblue', ms=2, zorder=1, label='data')
        plt.plot(tt, mt, "-r", zorder=5,label='MCMC best fit')
        plt.errorbar(tbin, ftbin, yerr=etbin, fmt='o', c='midnightblue', ms=5, capsize=2, zorder=3, label='binned data')
        plt.title(tit)
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%5.2f'))
        plt.legend(loc="best")
        plt.xlabel("BJD")
        plt.ylabel("Relative Flux")
        plt.ylim([min(ft)-0.1*(1-min(ft)), max(ft)+0.1*(max(ft)-1)])
        plt.savefig(outname)

        
        ########## Plot and save detrended lightcurve, model and residuals #############

        outname=prefix+names[j][:-4]+'_model.png'
        tit='Model for lightcurve '+names[j][:-4]
        fig,ax=plt.subplots(nrows=2, gridspec_kw={'height_ratios':[2,1]},figsize=(12,9))
        ax[0].plot(tt, fco,'o',c='skyblue',ms=2, zorder=1)
        ax[1].plot(tt, ft-mt,'o',c='skyblue',ms=2, zorder=1)
        ax[0].plot(tt,mm,'r-',zorder=5)
        ax[1].plot(tt,np.zeros(np.size(tt)),'r-')
        ax[0].errorbar(tbin, fcobin, yerr=etbin, fmt='o', c='midnightblue', ms=5, capsize=2, zorder=3)
        ax[1].plot(tbin, resbin,'o', c='midnightblue', ms=5, zorder=3)

        ax[0].set_ylim([np.min(fco) - 0.1 *np.abs((1.0 - np.min(fco))),np.max(fco) + 0.1 * np.abs((np.max(fco)-1.0))])
        ax[1].set_ylim([np.min(ft-mt) - 0.1 * np.min(ft-mt),np.max(ft-mt) + 0.1 * np.max(ft-mt)])

        ax[0].set_title(tit)
        ax[1].set_xlabel('BJD')
        ax[0].set_ylabel('Flux - Baseline')
        ax[1].set_ylabel('Residuals')

        ax[0].legend(['Flux - Baseline','Model','Binned data'])

        fig.savefig(outname,bbox_inches='tight')
        


    ######### PLOT phasecurve ##############
    ### Clip outliers
    # fluxclip = sigma_clip(flux_phasefold, sigma=5.0)
    # clipin = np.argwhere(fluxclip>0)

    # phase_phasefold = phase_phasefold[clipin.flatten()]
    # flux_phasefold = flux_phasefold[clipin.flatten()]
    # model_phasefold = model_phasefold[clipin.flatten()]

    #Adapt phase for transit to be in the middle of the plot
    if model_phasefold[np.argmin(phase_phasefold)] < 1.0:
        phase_phasefold[phase_phasefold > 0.5] = phase_phasefold[phase_phasefold >= 0.5] - 1.0

    ### Order by phase and calculate residuals
    phorder = phase_phasefold.argsort()
    phase_phasefold = phase_phasefold[phorder[::1]]
    flux_phasefold = flux_phasefold[phorder[::1]]
    model_phasefold = model_phasefold[phorder[::1]]

    res_phasefold = flux_phasefold - model_phasefold

    #Bin the data
    binsize = 10./(24.*60.) / period #10 minute bins in units of one phase
    nbin = int((np.max(phase_phasefold)-np.min(phase_phasefold))/binsize)
    phase_bins = np.linspace(np.min(phase_phasefold),np.max(phase_phasefold),nbin+1)
    pbin = phase_bins+0.5*binsize # MONIKA: defining the bin centers for plotting
    flux_bins = np.zeros(np.size(phase_bins))
    res_bins = np.zeros(np.size(phase_bins))
    error_bins = np.zeros(np.size(phase_bins))
    res_err_bins = np.zeros(np.size(phase_bins))

    for i in range(np.size(phase_bins)-1):
        fluxes_in_bin = flux_phasefold[(phase_phasefold >= phase_bins[i]) & (phase_phasefold < phase_bins[i+1])]
        if fluxes_in_bin.size == 0: #Check whether bin is empty
            flux_bins[i] = np.nan
            error_bins[i] = np.nan
        else:
            flux_bins[i] = np.mean(fluxes_in_bin)
            error_bins[i] = np.std(fluxes_in_bin)/np.sqrt(np.size(fluxes_in_bin))

        res_in_bin = res_phasefold[(phase_phasefold >= phase_bins[i]) & (phase_phasefold < phase_bins[i+1])]
        if res_in_bin.size == 0: #Check whether bin is empty
            res_bins[i] = np.nan
            res_err_bins[i] = np.nan
        else:
            res_bins[i] = np.mean(res_in_bin)
            res_err_bins[i] = np.std(res_in_bin)/np.sqrt(np.size(res_in_bin))

    ### Plot Phasecurve
    fig,ax=plt.subplots(nrows=2, gridspec_kw={'height_ratios':[2,1]},figsize=(12,9))

    ax[0].set_title('Phasefolded, fitted lightcurve')

    ax[0].plot(phase_phasefold, flux_phasefold,'o',c='skyblue',ms=2, zorder=1)
    ax[1].plot(phase_phasefold, res_phasefold,'o',c='skyblue',ms=2, zorder=1)
    ax[0].plot(phase_phasefold,model_phasefold,'r-',zorder=5)
    ax[1].plot(phase_phasefold,np.zeros(np.size(phase_phasefold)),'r-')
    ax[0].errorbar(pbin, flux_bins, yerr=error_bins, fmt='o', c='midnightblue', ms=5, capsize=2, zorder=3)
    ax[1].errorbar(pbin, res_bins, yerr=res_err_bins, fmt='o', c='midnightblue', ms=5, capsize=2)

    ax[0].set_ylim([np.min(flux_phasefold) - 0.1 *np.abs((1.0 - np.min(flux_phasefold))),np.max(flux_phasefold) + 0.1 * np.abs((np.max(flux_phasefold)-1.0))])
    ax[1].set_ylim([np.min(res_phasefold) + 0.1 * np.min(res_phasefold),np.max(res_phasefold) + 0.1 * np.max(res_phasefold)])

    ax[1].set_xlabel('Phase')

    ax[0].set_ylabel('Flux - Baseline')
    ax[1].set_ylabel('Residuals')

    ax[0].legend(['Flux - Baseline','Model','Binned data'])

    fig.savefig(prefix+'Phasefolded LC.png')


    ############ RVs#####################
        
    for j in range(nRV):
        mod=yval[indlist[nphot+j][0]]
        time=tarr[indlist[nphot+j][0]]
        flux=farr[indlist[nphot+j][0]]
        err=earr[indlist[nphot+j][0]]
        fwhm=warr[indlist[nphot+j][0]]
        sky=sarr[indlist[nphot+j][0]]
        bid=barr[indlist[nphot+j][0]]
        cid=carr[indlist[nphot+j][0]]
        
        # normalize the timestamps to the center of the transit
        timenorm = np.divide((time-params[0]),params[4]) -  np.round(np.divide((time-params[0]),params[4]))
        
        indsort = np.unravel_index(np.argsort(timenorm, axis=None), timenorm.shape)
         
        outname=prefix+RVnames[j][:-4]+'_fit.png'
        plt.figure(10)
        plt.clf()
        plt.errorbar(timenorm[indsort], flux[indsort], yerr=err[indsort], fmt=".g", label='data')
        plt.plot(timenorm[indsort], mod[indsort], "-r", label='MCMC best fit')
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%5.2f'))
        plt.xlabel("Orbital phase")
        plt.ylabel("RV [km/s]")
        plt.savefig(outname)     
        
        infile=RVnames[j][:-4]+'_out.dat'
        tt, ft, et, mt, bfunc, mm, fco = np.loadtxt(infile, usecols=(0,1,2,3,4,5,6), unpack = True)  # reading in the rvcurve data
        
        outname=prefix+RVnames[j][:-4]+'_cor.png'
        plt.figure(10)
        plt.clf()
        plt.errorbar(timenorm[indsort], fco[indsort], yerr=err[indsort], fmt=".g", label='data')
        plt.plot(timenorm[indsort], mm[indsort], "-r", label='MCMC best fit')
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%5.2f'))
        plt.xlabel("Orbital phase")
        plt.ylabel("RV [km/s]")
        plt.savefig(outname)  

    matplotlib.use(__default_backend__)

def param_hist(vals,pname,mv,s1v,s3v,mav,s1m,s3m):
    
    # this needs to be written. Just make a histogram plot of the parameter (values vals), label it (pname), and indicate the 1- and 3- sigma limits (s1, s3)

    import numpy as np
    import matplotlib
    from ._classes import __default_backend__
    matplotlib.use('Agg')
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    import os

    fig = plt.figure()

    matplotlib.rcParams.update({'font.size': 10})
    
    num_bins = 50
    # the histogram of the data
    n, bins, patches = plt.hist(vals, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel(pname)
    plt.ylabel('N samples')
    l=plt.axvline(x=mv, color='r')    
    l=plt.axvline(x=mv+s1v[0], color='b')
    l=plt.axvline(x=mv+s1v[1], color='b')
    l=plt.axvline(x=mv+s3v[0], color='y')
    l=plt.axvline(x=mv+s3v[1], color='y')

    l=plt.axvline(x=mav, color='r', linestyle='dashed')    
    l=plt.axvline(x=mav+s1m[0], color='b', linestyle='dashed')
    l=plt.axvline(x=mav+s1m[1], color='b', linestyle='dashed')
    l=plt.axvline(x=mav+s3m[0], color='y', linestyle='dashed')
    l=plt.axvline(x=mav+s3m[1], color='y', linestyle='dashed')
    
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    if not os.path.exists("histograms"): os.mkdir("histograms")
    outname="histograms/hist_"+pname+".png"
    plt.savefig(outname)

    matplotlib.use(__default_backend__)

#    if not os.path.exists("posteriors"): os.mkdir("posteriors")
#    outfile="posteriors/posterior_"+pname+".dat"
#    of=open(outfile,'w')
#    for ii in range(len(vals)):
#        of.write('%14.8f\n' % (vals[ii]) ) 
#
#    of.close()


def param_histbp(vals,pname,mv,s1v,s3v,mav,s1m,s3m,bpm,s1bpm):
    
    # this needs to be written. Just make a histogram plot of the parameter (values vals), label it (pname), and indicate the 1- and 3- sigma limits (s1, s3)

    import numpy as np
    import matplotlib
    from ._classes import __default_backend__ 
    matplotlib.use('Agg')
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    import os

    fig = plt.figure()

    matplotlib.rcParams.update({'font.size': 10})
    
    num_bins = 50
    # the histogram of the data
    n, bins, patches = plt.hist(vals, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel(pname)
    plt.ylabel('N samples')
    l=plt.axvline(x=mv, color='r', linestyle='dotted')    
    l=plt.axvline(x=mv+s1v[0], color='b', linestyle='dotted')
    l=plt.axvline(x=mv+s1v[1], color='b', linestyle='dotted')
    l=plt.axvline(x=mv+s3v[0], color='y', linestyle='dotted')
    l=plt.axvline(x=mv+s3v[1], color='y', linestyle='dotted')

    l=plt.axvline(x=mav, color='r', linestyle='dashed')    
    l=plt.axvline(x=mav+s1m[0], color='b', linestyle='dashed')
    l=plt.axvline(x=mav+s1m[1], color='b', linestyle='dashed')
    l=plt.axvline(x=mav+s3m[0], color='y', linestyle='dashed')
    l=plt.axvline(x=mav+s3m[1], color='y', linestyle='dashed')

    l=plt.axvline(x=bpm, color='r')    
 #   l=plt.axvline(x=bpm+s1bpm[0], color='g', linestyle='dashed')
 #   l=plt.axvline(x=bpm+s1bpm[1], color='g', linestyle='dashed')
    
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    if not os.path.exists("histograms"): os.mkdir("histograms")
    outname="histograms/hist_"+pname+".png"
    plt.savefig(outname)

    matplotlib.use(__default_backend__)


#    if not os.path.exists("posteriors"): os.mkdir("posteriors")
#    outfile="posteriors/posterior_"+pname+".dat"
#    of=open(outfile,'w')
#    for ii in range(len(vals)):
#        of.write('%14.8f\n' % (vals[ii]) ) 

#    of.close()

def plot_traspec(dRpRsres, edRpRsres, ulamdas):
    
    import numpy as np
    import matplotlib
    from ._classes import __default_backend__
    matplotlib.use('Agg')
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
#    import plotly.plotly as py  # tools to communicate with Plotly's server
    
    outname='transpec.png'
    fig = plt.figure()
    plt.errorbar(ulamdas, dRpRsres, yerr=edRpRsres, fmt=".b")
    plt.xlabel("Wavelength [A]")
    plt.ylabel("Rp/Rs")
    plt.savefig(outname)

    matplotlib.use(__default_backend__)

    
def plot_phasecurve(params, filename):

    import lightkurve as lk
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    
    phaseCenter = 0.5
    #OK: tData, yData, dyData = np.loadtxt(os.path.join(outdir, filename)).T
    tt, ft, et, mt, bfunc, mm, fco = np.loadtxt(filename).T
    #not needed: yData -= fpMean
    data = lk.LightCurve(time=tt, flux=ft, flux_err=et).fold(params[4], params[0] - phaseCenter*params[4])#.bin(75)

    lc = lk.LightCurve(time=tt, flux=mt).fold(params[4], params[0] - phaseCenter*params[4])

    #t = np.linspace(data.time.min(), data.time.max(), 1000)
    #lc = lk.LightCurve(time=t, flux=modelLC(t * batman_params.per - phaseCenter * batman_params.per + batman_params.t0))
    
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1.5, 1, 1]}, figsize=(12, 10.5))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.98], 'pad': 1., 'h_pad': 0})

    dataLine = axes[0].plot(data.time, data.flux, '.', label="Data", alpha=0.5, c='lightgrey', zorder=0, rasterized=True)
    dataBinned = data.bin(80, method="median")
    modelBinned = lc.bin(80, method="median")
    binLine = axes[0].errorbar(dataBinned.time, dataBinned.flux, yerr=dataBinned.flux_err, label="Binned data", fmt='.', c='r', ms=3, zorder=1)
    modelLine = axes[0].plot(lc.time ,lc.flux, label="Model", c='k', zorder=2)
    axes[0].set_ylabel("Normalized flux")
    axes[0].legend(loc=0)

    axes[0].get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[0].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[0].grid(b=True, which='minor', linewidth=.2)
    axes[0].grid(b=True, which='major', linewidth=1)

    dataLine = axes[1].plot(data.time, data.flux, '.', label="Data", alpha=0.5, c='lightgrey', zorder=0, rasterized=True)
    axes[1].axhline(1., ls='--', color='grey', lw=2)
    modelLine = axes[1].plot(lc.time ,lc.flux, label="Model", c='k', zorder=2, lw=2)
    binLine = axes[1].errorbar(dataBinned.time, dataBinned.flux, yerr=dataBinned.flux_err, label="Binned data", fmt='.', c='r', ms=5, zorder=1)


    axes[1].set_ylabel("Normalized flux")

    axes[1].get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[1].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[1].grid(b=True, which='minor', linewidth=.2)
    axes[1].grid(b=True, which='major', linewidth=1)
    axes[1].set_ylim([0.9996,1.0003])

    #lc = lk.LightCurve(time=data.time, flux=data.flux)
    axes[2].plot(data.time, 1e6 * (data.flux-lc.flux), '.', alpha=0.5, c='lightgrey', zorder=0, rasterized=True)

    #lc = lk.LightCurve(time=dataBinned.time, flux=dataBinned.flux)
    axes[2].errorbar(dataBinned.time, 1e6 * (dataBinned.flux-modelBinned.flux), yerr=dataBinned.flux_err*1e6, fmt='.', c='r', ms=3, zorder=1)

    axes[2].get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[2].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[2].grid(b=True, which='major', linewidth=1)

    axes[2].set_ylabel("Residuals [ppm]")

    t = dataBinned.time
    lc = lk.LightCurve(time=t, flux=dataBinned.flux)
    axes[3].errorbar(dataBinned.time, 1e6 * (dataBinned.flux-modelBinned.flux), yerr=dataBinned.flux_err*1e6, fmt='.', c='r', ms=3, zorder=1)

    axes[3].get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[3].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axes[3].grid(b=True, which='minor', linewidth=.2, linestyle=":")
    axes[3].grid(b=True, which='major', linewidth=1)

    axes[3].set_ylabel("Binned residuals [ppm]")
    axes[3].set_xlabel("Phase")
    axes[2].set_xlim([-0.51, 0.51])

    fig.savefig(filename+"_PC.pdf")
    matplotlib.use(__default_backend__)
