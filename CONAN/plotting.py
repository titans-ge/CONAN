from .utils import bin_data_with_gaps, phase_fold
from ._classes import __default_backend__
import numpy as np
import matplotlib, os
from matplotlib.ticker import FormatStrFormatter    
import matplotlib.pyplot as plt    
from CONAN.logprob_multi import logprob_multi
import dill as pickle
from os.path import splitext,dirname
import traceback
import concurrent.futures


def fit_plots(nttv, nphot, nRV, filters,names,RVnames,out_folder,prefix="/",RVunit="km/s",params=None,T0=None,period=None,Dur=None):

    _ind_para   = pickle.load(open(out_folder+"/.par_config.pkl","rb"))
    params_all  = np.concatenate((_ind_para["params"], _ind_para["GPparams"],_ind_para["rvGPparams"]))
    params_all[_ind_para["jumping"]] = params 
    filnames    = list(_ind_para["filnames"])
    model_PC    = _ind_para["model_phasevar"]
    all_models  = logprob_multi(params,_ind_para,get_planet_model=True)

    matplotlib.use('Agg')

    if np.iterable(T0): npl = len(T0)
    else: npl = 1; T0 = [T0]

    # extract the directory names of "\init\init_", "\max\max_", "\med\med_"
    if not os.path.exists(dirname(out_folder+prefix)): os.mkdir(dirname(out_folder+prefix))
    plot_folder = out_folder+prefix

    outdata_folder = out_folder+"/out_data/" if os.path.exists(out_folder+"/out_data/") else out_folder+"/"

    #model plot for each LC
    def plot_lightcurve(j):
        infile = outdata_folder + splitext(names[j])[0]+'_lcout.dat'
        fdata  = np.loadtxt(infile)
        n_nan  = np.sum(np.isnan(fdata).any(axis=1))
        n_inf  = np.sum(np.isinf(fdata).any(axis=1))  # find rows with inf in output
        if n_inf>0 or n_nan>0:
            print(f"\n       Warning: {names[j]} – ",end="")
            if n_inf > 0: print(f"infs on {n_inf} row(s). ",end="")
            if n_nan > 0: print(f"NaNs on {n_nan} row(s).")
        fin    = ~np.isinf(fdata).any(axis=1)  # find rows with inf in output
        tt, flux, err, full_mod, bfunc, mm, det_flux = np.loadtxt(infile, usecols=(0,1,2,3,8,9,10), unpack = True)  # reading in the lightcurve data
        tt, flux, err, full_mod, bfunc, mm, det_flux = tt[fin], flux[fin], err[fin], full_mod[fin], bfunc[fin], mm[fin], det_flux[fin]
        
        #evaluate lc model on smooth time grid
        t_sm  = np.linspace(tt.min(),tt.max(), int(np.ptp(tt)*24*60/2))
        lc_sm = logprob_multi(params,_ind_para,t=t_sm,get_planet_model=True).lc[names[j]][0]
        t_sm, lc_sm = t_sm[~np.isinf(lc_sm)], lc_sm[~np.isinf(lc_sm)]

        # bin the lightcurve data
        binsize     = min(Dur)/10 #binsize_min/(24.*60.)

        t_bin,    f_bin,  err_bin       = bin_data_with_gaps(tt, flux, err, binsize=binsize)    #original data
        det_tbin, det_fbin, det_errbin  = bin_data_with_gaps(tt, det_flux, err, binsize=binsize)    #detrended data
        res_tbin, resbin                = bin_data_with_gaps(tt, flux-full_mod,binsize=binsize)    #residuals

        ########## Plot and save lightcurve with fit ########
        outname=plot_folder+splitext(names[j])[0]+'_fit.png'

        fig,ax = plt.subplots(3,1, figsize=(12,12), sharex=True,gridspec_kw={"height_ratios":(3,3,1)})
        ax[0].set_title('Fit for lightcurve '+names[j][:-4])
        ax[0].set_ylabel("Flux")
        ax[0].plot(tt, flux,'.',c='skyblue', ms=3, zorder=1, label='Data')
        ax[0].plot(tt, full_mod, "-r", lw=2,zorder=5,label='Full Model fit')
        ax[0].plot(tt, bfunc, "g--",  zorder=5,label='Baseline')
        ax[0].errorbar(t_bin, f_bin, yerr=err_bin, fmt='o', c='midnightblue', ms=3, capsize=2, zorder=3, 
                        label=f'{int(binsize*24*60)} min bin')
        ax[0].set_ylim([np.nanmin([np.nanmin(flux),np.nanmin(full_mod)])-0.1*(np.nanmax(flux)-np.nanmin(flux)), 
                        np.nanmax([np.nanmax(flux),np.nanmax(full_mod)])+0.1*(np.nanmax(flux)-np.nanmin(flux))])
        ax[0].legend()

        ax[1].set_ylabel("Detrended Flux")
        ax[1].plot(tt, det_flux,'.',c='skyblue',ms=3, zorder=1, label="Detrended data")
        # ax[1].plot(tt, mm,'r-',lw=2,zorder=5, label="Model fit")
        ax[1].plot(t_sm, lc_sm,'r-',lw=2,zorder=5, label="Best fit")
        ax[1].errorbar(det_tbin, det_fbin, yerr=det_errbin, fmt='o', c='midnightblue', ms=3, capsize=2, zorder=3)
        ax[1].set_ylim([np.nanmin([np.nanmin(det_flux),min(lc_sm)])-0.1*(np.nanmax(det_flux)-np.nanmin(det_flux)), 
                        np.nanmax([np.nanmax(det_flux),np.nanmax(lc_sm)])+0.1*(np.nanmax(det_flux)-np.nanmin(det_flux))])
        ax[1].legend()

        ax[2].set_ylabel("O – C [ppm]")
        ax[2].plot(tt, 1e6*(flux-full_mod),'.',c='skyblue',ms=2, zorder=1,label=f"rms:{np.std(1e6*(flux-full_mod)):.2f} ppm")
        ax[2].plot(res_tbin, 1e6*resbin,'o', c='midnightblue', ms=3, zorder=3)
        ax[2].axhline(0,ls="--", color="k", alpha=0.3)
        ax[2].legend()

        ax[2].set_xlabel("Time")
        plt.subplots_adjust(hspace=0.02)
        fig.savefig(outname,bbox_inches='tight',dpi=150)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(plot_lightcurve, range(nphot))


    #### phase plot for each planet in system across multiple LCs ####
    # check that filter list contains same string
        
    def plot_phase_folded_lightcurve(n, filt):
        flux_filter, phase_filter, err_filter, res_filter = [], [], [], []
        phsm_filter, lcsm_filter = [], []
        filt_index = filnames.index(filt)

        for j in range(nphot):
            infile = outdata_folder + names[j][:-4] + '_lcout.dat'
            fin    = ~np.isinf(np.loadtxt(infile)).any(axis=1)  # get rows with finite values
            tt, flux, err, full_mod, bfunc, mm, det_flux = np.loadtxt(infile, usecols=(0, 1, 2, 3, 8, 9, 10), unpack=True)
            tt, flux, err, full_mod, bfunc, mm, det_flux = tt[fin], flux[fin], err[fin], full_mod[fin], bfunc[fin], mm[fin], det_flux[fin]
            flux_resid = flux - full_mod

            if filters[j] == filt:
                # calculations for each planet (n) in the system
                phase    = phase_fold(tt, period[n], T0[n], -0.25)
                lc_comps = all_models.lc[names[j]][1]  # lc components for each planet in the system

                # evaluate lc model on smooth time grid
                t_sm       = np.linspace(tt.min(), tt.max(), int(np.ptp(tt)*24*60/2))
                ph_sm      = phase_fold(t_sm, period[n], T0[n], -0.25)
                lc_sm_comp = logprob_multi(params, _ind_para, t=t_sm, get_planet_model=True).lc[names[j]][1]

                # remove other planet's LC signal from det_flux
                for i in range(npl):
                    if i != n: det_flux -= lc_comps[f"pl_{i+1}"] - 1

                flux_filter.append(det_flux)
                phase_filter.append(phase)
                err_filter.append(err)
                res_filter.append(flux_resid)
                phsm_filter.append(ph_sm)
                lcsm_filter.append(lc_sm_comp[f"pl_{n + 1}"])

        # Bin the data
        binsize = Dur[n] / 10   # 10 bins in transit

        srt = np.argsort(np.concatenate(phase_filter)) if len(flux_filter) > 1 else np.argsort(phase_filter[0])
        pbin, flux_bins, error_bins = bin_data_with_gaps(np.concatenate(phase_filter)[srt], np.concatenate(flux_filter)[srt],
                                                np.concatenate(err_filter)[srt], binsize=binsize/ period[n])
        _, res_bins = bin_data_with_gaps(np.concatenate(phase_filter)[srt], np.concatenate(res_filter)[srt], binsize=binsize/ period[n])
        srt_sm = np.argsort(np.concatenate(phsm_filter))

        
        # if (only one planet) & (non-zero occ_deppt) & (modeling phase curve or there's transit & eclipse phases) in this filter, then create new panel to show zoom
        occ_ind     = 1+7*npl+_ind_para["nttv"]+_ind_para["nddf"]+filt_index
        plot_PCzoom = (npl==1) and (params_all[occ_ind]>0) and (model_PC[filt_index] or (min(pbin) < 0 < max(pbin) and min(pbin) < 0.5 < max(pbin)))
        if plot_PCzoom:  
            fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={"height_ratios": (1.5, 2, 1)})
        else:
            fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={"height_ratios": (3, 1)})
        
        ax[0].set_title(f'Phasefolded LC {filt} - planet{n + 1}: P={period[n]:.2f} d ({filt})')
        ax[0].set_ylabel(f"Detrended Flux [ppm]")
        ax[0].axhline(1, ls="--", color="k", alpha=0.3)
        ax[0].plot(np.concatenate(phase_filter), np.concatenate(flux_filter), '.', c='skyblue', ms=3, zorder=1, label='Data')
        ax[0].errorbar(pbin, flux_bins, yerr=error_bins, fmt='o', c='midnightblue', ms=5, capsize=2, zorder=3, label=f"{int(binsize*24*60)}-min bins")
        ax[0].plot(np.concatenate(phsm_filter)[srt_sm], np.concatenate(lcsm_filter)[srt_sm], "-r", zorder=5, lw=3,
                    label='Best-fit')
        ax[0].set_yticklabels([round(v) for v in (ax[0].get_yticks()-1)*1e6])
        ax[0].legend()


        if plot_PCzoom:  # if modeling a full phase curve (or there's transit & eclipse) in this filter create new panel to show zoom
            occ_ind = 1+7*npl+_ind_para["nttv"]+_ind_para["nddf"]+filt_index
            aev_ind = 1+7*npl+_ind_para["nttv"]+_ind_para["nddf"]+len(filnames)*3 + filt_index
            adb_ind = 1+7*npl+_ind_para["nttv"]+_ind_para["nddf"]+len(filnames)*4 + filt_index
            tot_pc_amp = params_all[occ_ind] + 2*params_all[aev_ind] + 2*params_all[adb_ind]

            # print(f"Occultation depth_{filt}: {_ind_para['pnames_all'][occ_ind]} - {DF_occ:.2f}ppm")
            ax[1].set_ylabel(f"Detrended Flux [ppm]")
            ax[1].axhline(1, ls="--", color="k", alpha=0.3)
            ax[1].plot(np.concatenate(phase_filter), np.concatenate(flux_filter), '.', c='skyblue', alpha=0.4,ms=3, zorder=1, label='Data')
            ax[1].errorbar(pbin, flux_bins, yerr=error_bins, fmt='o', c='midnightblue', ms=5, capsize=2, zorder=3, label=f"{int(binsize*24*60)}-min bins")
            ax[1].plot(np.concatenate(phsm_filter)[srt_sm], np.concatenate(lcsm_filter)[srt_sm], "-r", zorder=5, lw=3,
                        label='Best-fit')
            ax[1].set_ylim([1-1.5*tot_pc_amp*1e-6, max(flux_bins.max()+error_bins.max(), 1+1.5*tot_pc_amp*1e-6)])
            ax[1].set_yticklabels([round(v) for v in (ax[1].get_yticks()-1)*1e6])
            ax[1].legend()

        ax[-1].axhline(0, ls="--", color="k", alpha=0.3)
        ax[-1].plot(np.concatenate(phase_filter), np.concatenate(res_filter)*1e6, '.', c='skyblue', ms=2, zorder=1)
        ax[-1].errorbar(pbin, 1e6 * res_bins, yerr=error_bins * 1e6, fmt='o', c='midnightblue', ms=5, capsize=2)
        ax[-1].set_xlabel("Orbital phase")
        ax[-1].set_ylim([-np.nanstd(np.concatenate(res_filter)*1e6), np.nanstd(np.concatenate(res_filter)*1e6)])
        ax[-1].set_ylabel(f"O – C [ppm]")

        plt.subplots_adjust(hspace=0.04, wspace=0.04)
        fig.savefig(out_folder + prefix + f'Phasefolded_LC_[planet{n + 1}]_{filt}.png', bbox_inches="tight",dpi=150)


    if nphot > 0 and nttv == 0:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for n in range(npl):
                for filt in np.unique(filters):
                    try:
                        executor.submit(plot_phase_folded_lightcurve, n, filt)
                        # plot_phase_folded_lightcurve(n, filt)
                    except Exception as e:
                        print(f"Error occurred in plotting phase folded lightcurve ({n=},{filt=}): {e}")
                        print(traceback.format_exc())

    ############ RVs#####################
    def plot_rv(j):
        infile  = outdata_folder + splitext(RVnames[j])[0]+'_rvout.dat'
        outname = plot_folder+splitext(RVnames[j])[0]+'_fit.png'
        tt, y_rv , e_rv, full_mod, base, rv_mod, det_RV = np.loadtxt(infile, usecols=(0,1,2,3,7,8,9),unpack = True)  # reading in the rvcurve data
        rv_resid = y_rv-full_mod

        #evaluate rv model on smooth time grid
        t_sm  = np.linspace(tt.min(),tt.max(), int(np.ptp(tt)/min(period)*10))
        rv_sm = logprob_multi(params,_ind_para,t=t_sm,get_planet_model=True).rv[RVnames[j]][0]

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
        fig.savefig(outname, bbox_inches='tight',dpi=100)  


    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(plot_rv, range(nRV))

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
                infile  = outdata_folder + splitext(RVnames[j])[0]+'_rvout.dat'
                tt, y_rv , e_rv, full_mod, base, rv_mod, det_RV = np.loadtxt(infile, usecols=(0,1,2,3,7,8,9), unpack = True)
                rv_resid = y_rv-full_mod

                #calculations for each planet (n) in the system
                phase    = phase_fold(tt, period[n], T0[n],-0.5)
                rv_comps = all_models.rv[RVnames[j]][1]    #rv components for each planet in the system
                
                #evaluate rv model on smooth time grid
                t_sm       = np.linspace(tt.min(),tt.max(), int(np.ptp(tt)/min(period)*10))
                ph_sm      = phase_fold(t_sm, period[n], T0[n],-0.5)
                # ph_sm      = np.where(ph_sm<0, ph_sm+1, ph_sm)
                rv_sm_comp = logprob_multi(params,_ind_para,t=t_sm,get_planet_model=True).rv[RVnames[j]][1]

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
            fig.savefig(plot_folder+f'Phasefolded_RV_[planet{n+1}].png',bbox_inches="tight",dpi=100) 
    
    plt.close('all')
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
    plt.close()
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




def plot_traspec(dRpRsres, edRpRsres, uwl,out_folder):
    
    matplotlib.use('Agg')
    
    outname=out_folder+'/transpec.png'
    fig = plt.figure()
    plt.errorbar(uwl, dRpRsres, yerr=edRpRsres, fmt="ob", ecolor="gray")
    plt.xlabel("Wavelength [microns]")
    plt.ylabel("Rp/Rs")
    plt.savefig(outname)

    matplotlib.use(__default_backend__)