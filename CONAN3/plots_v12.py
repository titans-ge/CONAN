from .utils import bin_data, phase_fold

def mcmc_plots(yval,tarr,farr,earr, nphot, nRV, indlist, filters,names,RVnames,prefix,params,T0,period,Dur):
    
    import matplotlib
    import os
    matplotlib.use('Agg')
    from ._classes import __default_backend__
    import matplotlib.pyplot as plt    
    import numpy as np
    from matplotlib.ticker import FormatStrFormatter    


    if np.iterable(T0): npl = len(T0)
    else: npl = 1; T0 = [T0]
    
    phase_phasefold = {}    # dictionary containing phase of each planet across several datasets
    flux_phasefold  = {}
    model_phasefold = {}
    for n in range(npl):
        phase_phasefold[n] = np.empty(0) #array containing phase of each data point
        flux_phasefold[n]  = np.empty(0) #array containig baseline corrected flux of each data point
        model_phasefold[n] = np.empty(0) #array containg baseline corrected model of each data point


        
    for j in range(nphot):
        mod  = yval[indlist[j][0]]
        time = tarr[indlist[j][0]]
        flux = farr[indlist[j][0]]
        err  = earr[indlist[j][0]]

        infile=prefix.split("/")[0] + "/" + names[j][:-4]+'_lcout.dat'
        tt, ft, et, mt, bfunc, mm, fco = np.loadtxt(infile, usecols=(0,1,2,3,4,5,6), unpack = True)  # reading in the lightcurve data

        # bin the lightcurve data
        binsize_min=15.
        binsize    = binsize_min/(24.*60.)
        nbin = int((np.max(tt)-np.min(tt))/binsize)  # number of bins
        # bins_edges = np.linspace(np.min(tt),np.max(tt),nbin+1)

        tbin, ftbin, etbin = bin_data(tt, ft,et,statistic='mean',bins=nbin)
        _,    fcobin       = bin_data(tt, fco,  statistic='mean',bins=nbin)
        _,    resbin       = bin_data(tt, ft-mt,statistic='mean',bins=nbin)

        ########## Plot and save lightcurve with fit ########
        titl='Fit for lightcurve '+names[j][:-4]
        outname=prefix+names[j][:-4]+'_fit.png'

        fig,ax = plt.subplots(3,1, figsize=(12,12), sharex=True,gridspec_kw={"height_ratios":(3,3,1)})
        ax[0].set_title(titl)
        ax[0].set_ylabel("Flux")
        ax[0].plot(tt, ft,'.',c='skyblue', ms=2, zorder=1, label='Data')
        ax[0].plot(tt, mt, "-r", lw=2,zorder=5,label='Full Model fit')
        ax[0].plot(tt, bfunc, "g--",  zorder=5,label='Baseline')
        ax[0].errorbar(tbin, ftbin, yerr=etbin, fmt='o', c='midnightblue', ms=3, capsize=2, zorder=3, label=f'{int(binsize_min)} min bin')
        ax[0].set_ylim([min(ft)-0.1*(1-min(ft)), max(ft)+0.1*(max(ft)-1)])
        ax[0].legend()

        ax[1].set_ylabel("Flux - baseline")
        ax[1].plot(tt, fco,'.',c='skyblue',ms=2, zorder=1, label="Detrended data")
        ax[1].plot(tt, mm,'r-',lw=2,zorder=5, label="Model fit")
        ax[1].errorbar(tbin, fcobin, yerr=etbin, fmt='o', c='midnightblue', ms=3, capsize=2, zorder=3)
        ax[1].set_ylim([min(fco)-0.1*(1-min(fco)), max(fco)+0.1*(max(fco)-1)])
        ax[1].legend()

        ax[2].set_ylabel("O – C [ppm]")
        ax[2].plot(tt, 1e6*(ft-mt),'.',c='skyblue',ms=2, zorder=1)
        ax[2].plot(tbin, 1e6*resbin,'o', c='midnightblue', ms=3, zorder=3)
        ax[2].axhline(0,ls="--", color="k", alpha=0.3)

        ax[2].set_xlabel("Time")
        plt.subplots_adjust(hspace=0.02)
        fig.savefig(outname,bbox_inches='tight')



        for n in range(npl):
        #phase folded plot
            ph = np.modf((np.modf((tt-T0[n])/period[n])[0])+1.0)[0] #calculate phase   #TODO: modify for multiplanetary

            #Add data to array to phasefold them later
            phase_phasefold[n] = np.append(phase_phasefold[n], ph)
            flux_phasefold[n]  = np.append(flux_phasefold[n],  fco)
            model_phasefold[n] = np.append(model_phasefold[n], mm)




    ######### PLOT phasecurve ##############

    for n in range(npl):

        #Adapt phase for transit to be in the middle of the plot
        if model_phasefold[n][np.argmin(phase_phasefold[n])] < 1.0:
            phase_phasefold[n][phase_phasefold[n] > 0.5] = phase_phasefold[n][phase_phasefold[n] >= 0.5] - 1.0

        ### Order by phase and calculate residuals
        phorder = phase_phasefold[n].argsort()
        phase_phasefold[n] = phase_phasefold[n][phorder[::1]]
        flux_phasefold[n]  = flux_phasefold[n][phorder[::1]]
        model_phasefold[n] = model_phasefold[n][phorder[::1]]

        res_phasefold      = flux_phasefold[n] - model_phasefold[n]

        #Bin the data
        binsize      = 15./(24.*60.) / period[n] #10 minute bins in units of one phase
        Tdur_phase   = Dur[n]/period[n]
        nbin         = int((np.max(phase_phasefold[n])-np.min(phase_phasefold[n]))/binsize)
        phase_bins   = np.linspace(np.min(phase_phasefold[n]),np.max(phase_phasefold[n]),nbin+1)
        pbin         = phase_bins+0.5*binsize # MONIKA: defining the bin centers for plotting
        flux_bins    = np.zeros(np.size(phase_bins))
        res_bins     = np.zeros(np.size(phase_bins))
        error_bins   = np.zeros(np.size(phase_bins))
        res_err_bins = np.zeros(np.size(phase_bins))

        for i in range(np.size(phase_bins)-1):
            fluxes_in_bin = flux_phasefold[n][(phase_phasefold[n] >= phase_bins[i]) & (phase_phasefold[n] < phase_bins[i+1])]
            if fluxes_in_bin.size == 0: #Check whether bin is empty
                flux_bins[i]  = np.nan
                error_bins[i] = np.nan
            else:
                flux_bins[i]  = np.mean(fluxes_in_bin)
                error_bins[i] = np.std(fluxes_in_bin)/np.sqrt(np.size(fluxes_in_bin))

            res_in_bin = res_phasefold[(phase_phasefold[n] >= phase_bins[i]) & (phase_phasefold[n] < phase_bins[i+1])]
            if res_in_bin.size == 0: #Check whether bin is empty
                res_bins[i]     = np.nan
                res_err_bins[i] = np.nan
            else:
                res_bins[i]     = np.mean(res_in_bin)
                res_err_bins[i] = np.std(res_in_bin)/np.sqrt(np.size(res_in_bin))

        ### Plot Phasecurve
        fig,ax=plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios':[2,1]},figsize=(12,9))
        #TODO imporve ylim ranges for better model visualization
        ax[0].set_title(f'Phasefolded, fitted lightcurve - planet{n+1} - P={period[n]:.2f}d')
        ax[0].plot(phase_phasefold[n], flux_phasefold[n],'o',c='skyblue',ms=2, zorder=1)
        ax[0].plot(phase_phasefold[n], model_phasefold[n],'r-',zorder=5)
        ax[0].errorbar(pbin, flux_bins, yerr=error_bins, fmt='o', c='midnightblue', ms=5, capsize=2, zorder=3)
        ax[0].set_ylim([np.min(flux_phasefold[n]) - 0.1 *np.abs((1.0 - np.min(flux_phasefold[n]))),np.max(flux_phasefold[n]) + 0.1 * np.abs((np.max(flux_phasefold[n])-1.0))])
        ax[0].set_ylabel('Flux - Baseline')
        ax[0].set_xlim([-Tdur_phase,Tdur_phase])
        ax[0].legend(['Flux - Baseline','Model','Binned data'])

        ax[1].plot(phase_phasefold[n], 1e6*res_phasefold,'o',c='skyblue',ms=2, zorder=1)
        ax[1].errorbar(pbin, 1e6*res_bins, yerr=res_err_bins*1e6, fmt='o', c='midnightblue', ms=5, capsize=2)
        ax[1].axhline(0, ls="--",color="k", alpha=0.3)
        ax[1].set_ylim([np.min(res_phasefold*1e6) + 0.1 * np.min(res_phasefold*1e6),np.max(res_phasefold*1e6) + 0.1 * np.max(res_phasefold)])
        ax[1].set_xlabel('Phase')
        ax[1].set_ylabel('Residuals [ppm]')
        plt.subplots_adjust(hspace=0.02)

        fig.savefig(prefix+f'Phasefolded_LC{n}.png')


    ############ RVs#####################
        
    for j in range(nRV):
        mod  = yval[indlist[nphot+j][0]]
        time = tarr[indlist[nphot+j][0]]
        flux = farr[indlist[nphot+j][0]]
        err  = earr[indlist[nphot+j][0]]
        
        # normalize the timestamps to the center of the transit
        timenorm = np.divide((time-params[0]),params[4]) -  np.round(np.divide((time-params[0]),params[4]))
        indsort = np.unravel_index(np.argsort(timenorm, axis=None), timenorm.shape)

        
        infile=prefix.split("/")[0] + "/" + RVnames[j][:-4]+'_rvout.dat'
        tt, ft, et, mt, bfunc, mm, fco = np.loadtxt(infile, usecols=(0,1,2,3,4,5,6), unpack = True)  # reading in the rvcurve data
        
        outname=prefix+RVnames[j][:-4]+'_fit.png'
        fig,ax = plt.subplots(3,1, figsize=(12,12), sharex=True,gridspec_kw={"height_ratios":(3,3,1)})
        ax[0].set_title('Fit for RV curve '+RVnames[j][:-4])
        ax[0].set_ylabel("RV [km/s]")
        ax[0].errorbar(timenorm[indsort], flux[indsort], yerr=err[indsort], fmt=".",c='skyblue', label='Data')
        ax[0].plot(timenorm[indsort], mod[indsort], "-r", label='Full Model fit')
        ax[0].set_ylabel("RV [km/s]")

        ax[1].errorbar(timenorm[indsort], fco[indsort], yerr=err[indsort], fmt=".g", label='Data')
        ax[1].plot(timenorm[indsort], mm[indsort], "-r", label='MCMC best fit')
        ax[1].set_ylabel("RV [km/s]")

        ax[2].errorbar(timenorm[indsort], (flux[indsort]-mod[indsort]), yerr=err[indsort], fmt=".",c='skyblue', label='Data')
        ax[2].axhline(0,ls="--", color="k", alpha=0.3)
        ax[2].set_xlabel("Orbital phase")
        ax[2].set_ylabel("O – C [km/s]")
        fig.savefig(outname, bbox_inches='tight')  

    matplotlib.use(__default_backend__)

def param_hist(vals,pname,mv,s1v,s3v,mav,s1m,s3m,out_folder):
    
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
    l=plt.axvline(x=mv, color='r', label="med")    
    l=plt.axvline(x=mv+s1v[0], color='b')
    l=plt.axvline(x=mv+s1v[1], color='b')
    l=plt.axvline(x=mv+s3v[0], color='y')
    l=plt.axvline(x=mv+s3v[1], color='y')

    l=plt.axvline(x=mav, color='r', linestyle='dashed',label="max")    
    l=plt.axvline(x=mav+s1m[0], color='b', linestyle='dashed')
    l=plt.axvline(x=mav+s1m[1], color='b', linestyle='dashed')
    l=plt.axvline(x=mav+s3m[0], color='y', linestyle='dashed')
    l=plt.axvline(x=mav+s3m[1], color='y', linestyle='dashed')
    plt.legend()
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    if not os.path.exists(out_folder+"/histograms"): os.mkdir(out_folder+"/histograms")
    outname=out_folder+"/histograms/hist_"+pname+".png"
    plt.savefig(outname)

    matplotlib.use(__default_backend__)

#    if not os.path.exists("posteriors"): os.mkdir("posteriors")
#    outfile="posteriors/posterior_"+pname+".dat"
#    of=open(outfile,'w')
#    for ii in range(len(vals)):
#        of.write('%14.8f\n' % (vals[ii]) ) 
#
#    of.close()


def param_histbp(vals,pname,mv,s1v,s3v,mav,s1m,s3m,bpm,s1bpm,out_folder):
    
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
    l=plt.axvline(x=mv, color='r', linestyle='dotted',label="med")    
    l=plt.axvline(x=mv+s1v[0], color='b', linestyle='dotted')
    l=plt.axvline(x=mv+s1v[1], color='b', linestyle='dotted')
    l=plt.axvline(x=mv+s3v[0], color='y', linestyle='dotted')
    l=plt.axvline(x=mv+s3v[1], color='y', linestyle='dotted')

    l=plt.axvline(x=mav, color='r', linestyle='dashed',label="max")    
    l=plt.axvline(x=mav+s1m[0], color='b', linestyle='dashed')
    l=plt.axvline(x=mav+s1m[1], color='b', linestyle='dashed')
    l=plt.axvline(x=mav+s3m[0], color='y', linestyle='dashed')
    l=plt.axvline(x=mav+s3m[1], color='y', linestyle='dashed')

    l=plt.axvline(x=bpm, color='r',label="bf")    
 #   l=plt.axvline(x=bpm+s1bpm[0], color='g', linestyle='dashed')
 #   l=plt.axvline(x=bpm+s1bpm[1], color='g', linestyle='dashed')
    plt.legend()
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    if not os.path.exists(out_folder+"/histograms"): os.mkdir(out_folder+"/histograms")
    outname=out_folder+"/histograms/hist_"+pname+".png"
    plt.savefig(outname)

    matplotlib.use(__default_backend__)


#    if not os.path.exists("posteriors"): os.mkdir("posteriors")
#    outfile="posteriors/posterior_"+pname+".dat"
#    of=open(outfile,'w')
#    for ii in range(len(vals)):
#        of.write('%14.8f\n' % (vals[ii]) ) 

#    of.close()

def plot_traspec(dRpRsres, edRpRsres, ulamdas,out_folder):
    
    import numpy as np
    import matplotlib
    from ._classes import __default_backend__
    matplotlib.use('Agg')
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
#    import plotly.plotly as py  # tools to communicate with Plotly's server
    
    outname=out_folder+'/transpec.png'
    fig = plt.figure()
    plt.errorbar(ulamdas, dRpRsres, yerr=edRpRsres, fmt=".b")
    plt.xlabel("Wavelength [A]")
    plt.ylabel("Rp/Rs")
    plt.savefig(outname)

    matplotlib.use(__default_backend__)

    
def plot_phasecurve(params, filename):
    #TODO: look at phase curve plot

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
#TODO backend issue causes timeout 