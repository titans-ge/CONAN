from .utils import bin_data, phase_fold
from ._classes import __default_backend__
import numpy as np
import matplotlib, os
from matplotlib.ticker import FormatStrFormatter    
import matplotlib.pyplot as plt    
from CONAN3.logprob_multi import logprob_multi
import pickle


def mcmc_plots(yval,tarr,farr,earr, nphot, nRV, indlist, filters,names,RVnames,prefix,RVunit,params,T0,period,Dur):

    _ind_para   = pickle.load(open(prefix.split("/")[0]+"/.par_config.pkl","rb"))
    all_models  = logprob_multi(params,*_ind_para,get_model=True)

    matplotlib.use('Agg')

    if np.iterable(T0): npl = len(T0)
    else: npl = 1; T0 = [T0]
    
    #model plot for each LC
    for j in range(nphot):
        infile=prefix.split("/")[0] + "/" + names[j][:-4]+'_lcout.dat'
        tt, flux, err, full_mod, bfunc, mm, det_flux = np.loadtxt(infile, usecols=(0,1,2,3,4,5,6), unpack = True)  # reading in the lightcurve data
        
        #evaluate rv model on smooth time grid
        t_sm  = np.linspace(tt.min(),tt.max(), max(2000,len(tt)))
        lc_sm = logprob_multi(params,*_ind_para,t=t_sm,get_model=True).lc[names[j]][0]

        # bin the lightcurve data
        binsize_min=15.
        binsize    = binsize_min/(24.*60.)
        nbin = int(np.ptp(tt)/binsize)  # number of bins

        t_bin, f_bin, err_bin = bin_data(tt, flux,err,     statistic='mean',bins=nbin)    #original data
        _,    det_fbin        = bin_data(tt, det_flux,     statistic='mean',bins=nbin)    #detrended data
        _,    resbin          = bin_data(tt, flux-full_mod,statistic='mean',bins=nbin)    #residuals

        ########## Plot and save lightcurve with fit ########
        outname=prefix+names[j][:-4]+'_fit.png'

        fig,ax = plt.subplots(3,1, figsize=(12,12), sharex=True,gridspec_kw={"height_ratios":(3,3,1)})
        ax[0].set_title('Fit for lightcurve '+names[j][:-4])
        ax[0].set_ylabel("Flux")
        ax[0].plot(tt, flux,'.',c='skyblue', ms=2, zorder=1, label='Data')
        ax[0].plot(tt, full_mod, "-r", lw=2,zorder=5,label='Full Model fit')
        ax[0].plot(tt, bfunc, "g--",  zorder=5,label='Baseline')
        ax[0].errorbar(t_bin, f_bin, yerr=err_bin, fmt='o', c='midnightblue', ms=3, capsize=2, zorder=3, label=f'{int(binsize_min)} min bin')
        ax[0].set_ylim([min([min(flux),min(full_mod)])-0.1*np.ptp(min(flux)), 
                        max([max(flux),max(full_mod)])+0.1*np.ptp(max(flux))])
        ax[0].legend()

        ax[1].set_ylabel("Flux - baseline")
        ax[1].plot(tt, det_flux,'.',c='skyblue',ms=2, zorder=1, label="Detrended data")
        # ax[1].plot(tt, mm,'r-',lw=2,zorder=5, label="Model fit")
        ax[1].plot(t_sm, lc_sm,'r-',lw=2,zorder=5, label="Best fit")
        ax[1].errorbar(t_bin, det_fbin, yerr=err_bin, fmt='o', c='midnightblue', ms=3, capsize=2, zorder=3)
        ax[1].set_ylim([min(det_flux)-0.1*(1-min(det_flux)), max(det_flux)+0.1*(max(det_flux)-1)])
        ax[1].legend()

        ax[2].set_ylabel("O – C [ppm]")
        ax[2].plot(tt, 1e6*(flux-full_mod),'.',c='skyblue',ms=2, zorder=1)
        ax[2].plot(t_bin, 1e6*resbin,'o', c='midnightblue', ms=3, zorder=3)
        ax[2].axhline(0,ls="--", color="k", alpha=0.3)

        ax[2].set_xlabel("Time")
        plt.subplots_adjust(hspace=0.02)
        fig.savefig(outname,bbox_inches='tight')


    #### phase plot for each planet in system across multiple LCs ####
    if nphot > 0:
        for n in range(npl):
            fig,ax = plt.subplots(2,1, figsize=(12,12), sharex=True,gridspec_kw={"height_ratios":(3,1)})
            ax[0].set_title(f'Phasefolded LC - planet{n+1}: P={period[n]:.2f} d')
            ax[0].set_ylabel(f"Flux – baseline")
            ax[0].axhline(1,ls="--", color="k", alpha=0.3)
            ax[1].axhline(0,ls="--", color="k", alpha=0.3)
            ax[1].set_xlabel("Orbital phase")
            ax[1].set_ylabel(f"O – C [ppm]")
            
            flux_all,phase_all,err_all,res_all  = [],[],[],[]

            for j in range(nphot):
                infile=prefix.split("/")[0] + "/" + names[j][:-4]+'_lcout.dat'
                tt, flux, err, full_mod, bfunc, mm, det_flux = np.loadtxt(infile, usecols=(0,1,2,3,4,5,6), unpack = True)
                flux_resid = flux - full_mod

                #calculations for each planet (n) in the system
                phase    = ((tt-T0[n])/period[n]) - np.round( ((tt-T0[n])/period[n]))
                lc_comps = all_models.lc[names[j]][1]    #lc components for each planet in the system

                #evaluate lc model on smooth time grid
                t_sm       = np.linspace(tt.min(),tt.max(), max(2000,len(tt)))
                ph_sm      = ((t_sm-T0[n])/period[n]) - np.round( ((t_sm-T0[n])/period[n]))
                lc_sm_comp = logprob_multi(params,*_ind_para,t=t_sm,get_model=True).lc[names[j]][1]

                #remove other planet's LC signal from det_flux
                for i in range(npl):
                    if i != n: det_flux -= lc_comps[f"pl_{i+1}"]-1

                ax[0].plot(phase, det_flux, "o", c='skyblue',ms=2, zorder=1)
                ax[1].plot(phase, 1e6*flux_resid, "o", c='skyblue',ms=2, zorder=1)

                flux_all.append(det_flux)
                phase_all.append(phase)
                err_all.append(err)
                res_all.append(flux_resid)


            #Bin the data
            binsize      = 15./(24.*60.) / period[n]  #15 minute bins in phase units
            # Tdur_phase   = Dur[n]/period[n]
            nbin         = int(np.ptp(np.concatenate(phase_all))/binsize)

            srt = np.argsort(np.concatenate(phase_all)) if nphot>1 else np.argsort(phase_all[0])
            pbin, flux_bins, error_bins = bin_data(np.concatenate(phase_all)[srt], np.concatenate(flux_all)[srt], 
                                                    np.concatenate(err_all)[srt], statistic='mean',bins=nbin)
            _,    res_bins              = bin_data(np.concatenate(phase_all)[srt], np.concatenate(res_all)[srt], 
                                                    statistic='mean',bins=nbin)
            srt_sm = np.argsort(ph_sm)
            ax[0].errorbar(pbin, flux_bins, yerr=error_bins, fmt='o', c='midnightblue', ms=5, capsize=2, zorder=3)
            ax[0].plot(ph_sm[srt_sm], lc_sm_comp[f"pl_{n+1}"][srt_sm], "-r", zorder=5, lw=3, label='Best-fit')
            ax[0].legend()
            ax[1].errorbar(pbin, 1e6*res_bins, yerr=error_bins*1e6, fmt='o', c='midnightblue', ms=5, capsize=2)
            
            plt.subplots_adjust(hspace=0.04,wspace=0.04)
            fig.savefig(prefix+f'Phasefolded_LC_[planet{n+1}].png',bbox_inches="tight")
        


    ############ RVs#####################
    for j in range(nRV):
        infile  = prefix.split("/")[0] + "/" + RVnames[j][:-4]+'_rvout.dat'
        outname = prefix+RVnames[j][:-4]+'_fit.png'
        tt, y_rv , e_rv, full_mod, base, rv_mod, det_RV = np.loadtxt(infile, usecols=(0,1,2,3,4,5,6),unpack = True)  # reading in the rvcurve data
        rv_resid = y_rv-full_mod

        #evaluate rv model on smooth time grid
        t_sm  = np.linspace(tt.min(),tt.max(), max(2000,len(tt)))
        rv_sm = logprob_multi(params,*_ind_para,t=t_sm,get_model=True).rv[RVnames[j]][0]

        fig,ax = plt.subplots(3,1, figsize=(10,15),sharex=True,gridspec_kw={"height_ratios":(3,3,1)})
        ax[0].set_title('Fit for RV curve '+RVnames[j][:-4])
        ax[0].errorbar(tt, y_rv, yerr=e_rv, fmt="o",capsize=2, label=RVnames[j])
        ax[0].plot(tt, full_mod, "-r", label='Full Model fit')
        ax[0].plot(tt, base, "--g", label='Baseline')
        ax[0].set_ylabel(f"RV [{RVunit}]")
        ax[0].legend()

        if RVunit == 'km/s':   #conv to m/s
            rv_mod, det_RV, e_rv, rv_resid = rv_mod*1e3, det_RV*1e3, e_rv*1e3, rv_resid*1e3
            rv_sm = rv_sm*1e3

        ax[1].axhline(0,ls="--", color="k", alpha=0.3)
        ax[1].errorbar(tt, det_RV, e_rv, fmt="o",capsize=2) 
        
        ax[1].plot(t_sm, rv_sm, "-r", alpha=0.4,label='best fit planet RV')
        # ax[1].plot(tt, rv_mod, "-r", label='MCMC best fit') 
        ax[1].set_ylabel(f"RV [m/s]")
        ax[1].legend()

        ax[2].errorbar(tt, rv_resid, yerr=e_rv, fmt="o",capsize=2)
        ax[2].axhline(0,ls="--", color="k", alpha=0.3)
        ax[2].set_xlabel("Time")
        ax[2].set_ylabel(f"O – C [m/s]")

        plt.subplots_adjust(hspace=0.02)
        fig.savefig(outname, bbox_inches='tight')  


    #joint plot
    if nRV > 0:
        for n in range(npl):
            fig,ax = plt.subplots(2,1, figsize=(12,12), sharex=True,gridspec_kw={"height_ratios":(3,1)})
            ax[0].set_title(f'Fit for RV curve - planet{n+1}: P={period[n]:.2f} d')
            ax[0].set_ylabel(f"RV [m/s]")
            ax[0].axhline(0,ls="--", color="k", alpha=0.3)
            ax[1].axhline(0,ls="--", color="k", alpha=0.3)
            ax[1].set_xlabel("Orbital phase")
            ax[1].set_ylabel(f"O – C [m/s]")
            rv_all = []
            phase_all = []

            for j in range(nRV):
                infile  = prefix.split("/")[0] + "/" + RVnames[j][:-4]+'_rvout.dat'
                tt, y_rv , e_rv, full_mod, base, rv_mod, det_RV = np.loadtxt(infile, usecols=(0,1,2,3,4,5,6), unpack = True)
                rv_resid = y_rv-full_mod

                #calculations for each planet (n) in the system
                phase    = ((tt-T0[n])/period[n]) - np.round( ((tt-T0[n])/period[n]))
                rv_comps = all_models.rv[RVnames[j]][1]    #rv components for each planet in the system
                
                #evaluate rv model on smooth time grid
                t_sm       = np.linspace(tt.min(),tt.max(), max(2000,len(tt)))
                ph_sm      = ((t_sm-T0[n])/period[n]) - np.round( ((t_sm-T0[n])/period[n]))
                rv_sm_comp = logprob_multi(params,*_ind_para,t=t_sm,get_model=True).rv[RVnames[j]][1]

                #remove other planet's RV signal from det_RV
                for i in range(npl):
                    if i != n: det_RV -= rv_comps[f"pl_{i+1}"]

                if RVunit == 'km/s': #conv to m/s
                    rv_mod, det_RV, e_rv, rv_resid = rv_mod*1e3, det_RV*1e3, e_rv*1e3, rv_resid*1e3
                    rv_sm_comp[f"pl_{n+1}"] *=1e3

                ax[0].errorbar(phase, det_RV, yerr=e_rv, fmt="o",capsize=2, label=RVnames[j])
                ax[1].errorbar(phase, rv_resid, yerr=e_rv, fmt="o",capsize=2)
                
                rv_all.append(rv_sm_comp[f"pl_{n+1}"])
                phase_all.append(ph_sm)
        
            srt = np.argsort(np.concatenate(phase_all)) if nRV>1 else np.argsort(phase_all[0])
            ax[0].plot(np.concatenate(phase_all)[srt], np.concatenate(rv_all)[srt], "-k", lw=3, label='Best-fit')
            ax[0].legend()
            plt.subplots_adjust(hspace=0.04,wspace=0.04)
            fig.savefig(prefix+f'Phasefolded_RV_[planet{n+1}].png',bbox_inches="tight") 

    matplotlib.use(__default_backend__)

def param_hist(vals,pname,mv,s1v,s3v,mav,s1m,s3m,out_folder):
    
    # this needs to be written. Just make a histogram plot of the parameter (values vals), label it (pname), and indicate the 1- and 3- sigma limits (s1, s3)
    matplotlib.use('Agg')

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


def param_histbp(vals,pname,mv,s1v,s3v,mav,s1m,s3m,bpm,s1bpm,out_folder):
    
    # this needs to be written. Just make a histogram plot of the parameter (values vals), label it (pname), and indicate the 1- and 3- sigma limits (s1, s3)

    matplotlib.use('Agg')

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




def plot_traspec(dRpRsres, edRpRsres, ulamdas,out_folder):
    
    matplotlib.use('Agg')
    
    outname=out_folder+'/transpec.png'
    fig = plt.figure()
    plt.errorbar(ulamdas, dRpRsres, yerr=edRpRsres, fmt=".b")
    plt.xlabel("Wavelength [A]")
    plt.ylabel("Rp/Rs")
    plt.savefig(outname)

    matplotlib.use(__default_backend__)