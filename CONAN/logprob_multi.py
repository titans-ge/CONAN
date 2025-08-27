import numpy as np
import time
from .models import *
from types import SimpleNamespace
from .utils import rho_to_tdur, rho_to_aR, Tdur_to_aR, sesinw_secosw_to_ecc_omega,get_Tconjunctions
from .geepee import gp_params_convert, GPSaveObj
from spleaf import cov, term
import matplotlib, os
from ._classes import __default_backend__
from os.path import splitext, exists
from .utils import light_travel_time_correction,sinusoid
from copy import deepcopy


def logprob_multi(p, args,t=None,make_outfile=False,verbose=False,debug=False,
                    get_planet_model=False, get_model=None,get_base_model=False,out_folder="",get_GPobj=False):
    """
    calculate log probability and create output file of full model calculated using posterior parameters

    Parameters
    ----------
    p : array
        model parameters 
    args : dict 
        contains various config parameters to use in mode calculation
    t : array, optional
        user-defined time array at which to evaluate the model, by default None
    make_outfile : bool, optional
        whether to make the output  model file "*out.dat". by default False
    verbose : bool, optional
        whether to print out information during function call, by default False
    debug : bool, optional
        see debug statements, by default False
    get_planet_model: bool, optional
        flag to output dictionary of planet model results (phot and RV) for specific input parameters.
    get_base_model: bool, optional
        flag to output dictionary of baseline model results (phot and RV) for specific input parameters.
    out_folder : str, optional
        folder to save output files, by default ""
    get_GPobj : bool
        whether to return GPobj for LCs and RVs. Default is False.
    
    Returns
    -------
    model : array
        model computed with parameter input `p`
    model_err : array
        model uncertainties
    T0 : float
        mid transit time 
    P0 : float
        period
    """
    if make_outfile and not exists(out_folder+"/out_data"): os.mkdir(out_folder+"/out_data")    #folder to put output data from the initial or fit run
    # if not exists(out_folder+"/out_data"): os.mkdir(out_folder+"/out_data")    #folder to put output data from the initial or fit run   

    if get_model is not None:
        get_planet_model = get_model
        print("Warning: get_model is deprecated. Use get_planet_model instead.\n")

    # distribute out all the input arguments
    argvals = args.values()
    if len(argvals)==72: argvals = list(argvals)[7:]  #backwards compatibility: skip the first 7 arguments
    
    (   custom_RVfunc, custom_LCfunc, nphot, nRV, sine_conf, filters, nfilt, filnames,\
        nddf, nocc, nttv, col8_arr, grprs, ttv_conf, grnames, groups, pc_model, \
        ewarr, inmcmc, paraCNM, baseLSQ, bvars, bvarsRV, model_phasevar, LCnames, RVnames, \
        lc_Qsho, rv_Qsho, dwCNMarr, dwCNMind, params, useGPphot, useGPrv, GPobjects, \
        GPparams, GPindex, pindices, jumping, jnames, prior_distr, pnames_all, norm_sigma, \
        uni_low, uni_up,rv_gp_colnames, gp_colnames, gpkerns, LTT, conditionals, GPstepsizes, sameLCgp, \
        npl, useSpline_lc, useSpline_rv, s_samp, rvGPobjects, rvGPparams, rvGPindex, input_lcs, input_rvs, \
        rv_gp_colerr_names, rv_pargps, rv_gpkerns, sameRVgp, fit_sampler, shared_params) = argvals if 'shared_params' in args.keys() else argvals+[{}]
    #TODO pass wavelength into args
    
    Rstar = LTT.Rstar if type(LTT) == SimpleNamespace else None   #get Rstar value to use for LTT correction
    if type(custom_LCfunc) != SimpleNamespace: custom_LCfunc = None
    if type(sine_conf)     != SimpleNamespace: sine_conf = SimpleNamespace(flag=None)
    if type(custom_RVfunc) != SimpleNamespace: custom_RVfunc = None
    if "model_phasevar" not in args.keys():
        model_phasevar = [False]*nfilt
    if isinstance(pc_model, int):   #backwards compatibility
        pc_model = ["cosine"]*nfilt

    params_all  = np.concatenate((params, GPparams,rvGPparams))

    lnprob     = 0.
    lnprior    = 0.

    #compute log prior if using emcee else skip
    if fit_sampler == "emcee":
        for jj in range(len(p)):
            lnprior += prior_distr[jj].logpdf(p[jj])  # add the log prior for each jumping parameter
        
    #if in mcmc check that lnprior is finite
    if inmcmc == 'y':
        if np.isfinite(lnprior):
            lnprob += lnprior      # add to logprob if finite. later steps calculate log_likelihood to be added 
        else:
            return -np.inf         # no point computing log_likelihood if log prior is not finite
    
    
    mod, emod = [], [] # output arrays in case we're not in the mcmc
    if get_planet_model: 
        model_outputs = SimpleNamespace(lc={},rv={}) 
    if get_base_model:
        basemodel_outputs = SimpleNamespace(lc={},rv={}) 

    params_all[jumping] = p   # set the jumping parameters to the values in p which are varied in mcmc 
    for sp in shared_params:
        for s_recip in shared_params[sp]:
            params_all[pnames_all == s_recip] = params_all[pnames_all == sp]

    # conditionals
    if isinstance(conditionals, dict) and len(conditionals)>0:
        for cond in conditionals.keys():
            if eval(conditionals[cond], globals={k:v for k,v in zip(pnames_all, params_all)}) == False:
                if inmcmc=='n': 
                    print(f"logprob_multi(): {conditionals[cond]=} is not satisfied")
                if inmcmc=='y': 
                    return -np.inf
    if debug:
        _ = [print(f"{k}:{v}") for k,v in zip(pnames_all[jumping], params_all[jumping])]

    ncustom = custom_LCfunc.npars if custom_LCfunc!=None else 0# number of custom function parameters
    nsin    = sum([v for v in sine_conf.npars.values()]) if sine_conf.flag else 0
    # sin_st  = 1+7*npl +nttv+ nddf+nocc*7 + nfilt*2 + nphot + ncustom    #starting index of sinuoid parameters

    time_all, flux_all, err_all, full_mod_all, base_para_all, base_sine_all, base_spl_all, base_gp_all, base_total_all, transit_all, det_flux_all, residual_all = [],[],[],[],[],[],[],[],[],[],[],[]
    lcGPsave = dict()
    # restrict the parameters to those of the light curve
    for j in range(nphot):
        
        name       = LCnames[j]
        thisLCdata = input_lcs[name]
        t_in       = thisLCdata["col0"] if t is None else t
        f_in       = thisLCdata["col1"]
        e_in       = thisLCdata["col2"]
        
        if custom_LCfunc.func!=None and custom_LCfunc.replace_LCmodel==False:
            if custom_LCfunc.x.startswith('col'):
                custom_LCfunc.xarr = thisLCdata[custom_LCfunc.x]
                custom_LCfunc.t    = thisLCdata['col0']   #needed to create interpolate func for other columns 

        s_samp[j]  = s_samp[j] if t is None else None  # if t is provided, don't use s_samp

        if baseLSQ == "y": bvar = bvars[j][0]
        else: bvar=[]

        pp      = p[pindices[j]]                    # the elements of the p array jumping in this LC, pp is the array of non-GP jumping parameters for this LC
        ppnames = pnames_all[jumping][pindices[j]]  # the names of the parameters jumping in this LC
        
        # extract the parameters input to the modeling function from the input array
        # specify the LD and ddf correctly
        # identify the filter index of this LC
        k = np.where(filnames == filters[j])  # k is the index of the LC in the filnames array
        k = k[0].item()  
        # vcont = cont[k,0]

        occind    = 1+7*npl+nttv+nddf+k             # index in params of the occultation depth value
        Fn_ind    = 1+7*npl+nttv+nddf+nocc+k        # index in params of the atm value
        phoff_ind = 1+7*npl+nttv+nddf+nocc*2+k      # index in params of the phoff value
        Aev_ind   = 1+7*npl+nttv+nddf+nocc*3+k      # index in params of the Aev value
        Adb_ind   = 1+7*npl+nttv+nddf+nocc*4+k      # index in params of the Adb value
        f1ev_ind  = 1+7*npl+nttv+nddf+nocc*5+k      # index in params of the Adb value
        cont_ind  = 1+7*npl+nttv+nddf+nocc*6+k      # index in params of the cont value

        q1ind     = 1+7*npl+nttv+nddf+nocc*7+2*k    # index in params of the first LD coeff of this filter
        q2ind     = 1+7*npl+nttv+nddf+nocc*7+2*k+1  # index in params of the second LD coeff of this filter
        gg        = int(groups[j]-1)

        LCjitterind = 1+7*npl + nttv+nddf+nocc*7 + nfilt*2 + j

        # get the index of pp that is 
        # adapt the RpRs value used in the LC creation to any ddfs
        ppcount = 0   # the index of the jumping parameter in the pp array
        T0in, RpRsin, bbin, perin, sesinwin, secoswin, Kin = [], [], [], [], [], [], []
        rhoin, durin = None, None
        #if parameter is jumping, get value from pp array, otherwise use the value from the params array
        if "rho_star" in pnames_all:    #using rho_star for transit model
            if 0 in jumping[0]:
                rhoin = pp[ppcount]
                ppcount = ppcount+1
            else:
                rhoin = params_all[0]
        else:  #using duration for transit model
            if 0 in jumping[0]:
                durin = pp[ppcount]
                ppcount = ppcount+1
            else:
                durin = params_all[0]

        for n in range(npl):
            if (1+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                T0in.append(pp[ppcount])        # the first element of pp is the T0 value
                ppcount = ppcount+1 
            else:
                T0in.append(params_all[1+7*n])
                
            if (2+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                RpRsin.append(pp[ppcount])
                ppcount = ppcount+1   
            else:
                RpRsin.append(params_all[2+7*n])

            if (3+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                bbin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                bbin.append(params_all[3+7*n])
                    
            if (4+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                perin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                perin.append(params_all[4+7*n])

            if (5+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                sesinwin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                sesinwin.append(params_all[5+7*n])

            if (6+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                secoswin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                secoswin.append(params_all[6+7*n])
        
        #ttv
        if nttv>0:
            for i,t_0 in enumerate(ttv_conf[j].t0_list):
                ttv_conf[j].t0_list[i] = pp[ppcount:ppcount+len(t_0)]
                ppcount = ppcount+len(t_0)

        if nddf>0:   
            ddf0 = pp[ppcount]
            isddf = 'y'
            grprs_here = grprs[gg]
            ppcount = ppcount+1
        else:
            ddf0 = 0.
            isddf = 'n'
            grprs_here = 0.
        
        if (occind in jumping[0]):   
            occin = pp[ppcount]
            ppcount = ppcount+1
        else:
            occin = params_all[occind]

        if (Fn_ind in jumping[0]):   
            Fn_in = pp[ppcount]
            ppcount = ppcount+1
        else:
            Fn_in = params_all[Fn_ind]

        if (phoff_ind in jumping[0]):   
            phoff_in = pp[ppcount]
            ppcount = ppcount+1
        else:
            phoff_in = params_all[phoff_ind]

        if (Aev_ind in jumping[0]):   
            Aev_in = pp[ppcount]
            ppcount = ppcount+1
        else:
            Aev_in = params_all[Aev_ind]

        if (Adb_ind in jumping[0]):   
            Adb_in = pp[ppcount]
            ppcount = ppcount+1
        else:
            Adb_in = params_all[Adb_ind]

        if (f1ev_ind in jumping[0]):   
            f1ev_in = pp[ppcount]
            ppcount = ppcount+1
        else:
            f1ev_in = params_all[f1ev_ind]

        if (cont_ind in jumping[0]):   
            cont_in = pp[ppcount]
            ppcount = ppcount+1
        else:
            cont_in = params_all[cont_ind]
            
        #########
        #now check the correct LD coeffs
        if (q1ind in jumping[0]):    # index of specific LC LD in jumping array -> check in jumping array
            q1in    = pp[ppcount]
            ppcount = ppcount + 1
        else:
            q1in = params_all[q1ind]

        if (q2ind in jumping[0]):   # index of specific LC LD in jumping array -> check in jumping array
            q2in    = pp[ppcount]
            ppcount = ppcount + 1
        else:
            q2in = params_all[q2ind]

        if (LCjitterind in jumping[0]):   # index of specific LC jitter in jumping array -> check in jumping array
            ppcount = ppcount + 1
        
        cst_pars= {}
        if ncustom>0:
            customind = 1+7*npl + nttv+nddf+nocc*7 + nfilt*2 + nphot + np.arange(ncustom)
            for kk,cst_ind in enumerate(customind):
                if cst_ind in jumping[0]:
                    cst_pars[list(custom_LCfunc.par_dict.keys())[kk]] = pp[ppcount]
                    ppcount = ppcount + 1
                else:
                    cst_pars[list(custom_LCfunc.par_dict.keys())[kk]] = params[cst_ind]

        if sine_conf.flag:
            file_slct = filnames[k] if sine_conf.fit=="filt" else "same" if sine_conf.fit=="same" else LCnames[j]
            sin_ind   = 1+7*npl +nttv+ nddf+nocc*7 + nfilt*2 + nphot + ncustom + sine_conf.index[file_slct]  #the indices of this lc/filt sinuoid parameters
            # sine_conf.pars[file_slct] = params_all[sin_ind]     #TODO all parameters should be updated like this as opppsed to using ppcount
            for kk,s_ind in enumerate(sin_ind):
                if s_ind in jumping[0]:
                    sine_conf.pars[file_slct][kk]  = pp[ppcount]
                    ppcount = ppcount + 1
                else:
                    sine_conf.pars[file_slct][kk] = params_all[s_ind]
            #compute sinusoidal model
            x      = thisLCdata[sine_conf.x[file_slct]]
            amp    = sine_conf.pars[file_slct][:-2]
            Per,x0 = sine_conf.pars[file_slct][-2:]
            base_sine = sinusoid(x,amp,x0,Per,sine_conf.n[file_slct],sine_conf.trig[file_slct])
        else:
            base_sine = np.zeros(len(t_in))

        if get_planet_model:
            if nttv>0:
                LCmod,compo = TTV_Model(tarr=t_in, rho_star=rhoin, dur=durin, T0_list=ttv_conf[j].t0_list, RpRs=RpRsin, b=bbin, per=perin, sesinw=sesinwin, secosw=secoswin,
                                        ddf=ddf0,occ=occin, Fn=Fn_in, delta=phoff_in, A_ev=Aev_in, f1_ev=f1ev_in, A_db=Adb_in,
                                        q1=q1in, q2=q2in,split_conf=ttv_conf[j],ss=s_samp[j],vcont=cont_in,Rstar=Rstar,grprs=grprs_here,
                                        custom_LCfunc=custom_LCfunc if ncustom>0 else None, cst_pars=cst_pars)
            else:
                TM = Planet_LC_Model(rho_star=rhoin, dur=durin, T0=T0in, RpRs=RpRsin, b=bbin, per=perin, sesinw=sesinwin, secosw=secoswin, ddf=ddf0, 
                                    occ=occin, Fn=Fn_in, delta=phoff_in, A_ev=Aev_in, f1_ev=f1ev_in, A_db=Adb_in, q1=q1in, q2=q2in,cst_pars=cst_pars,npl=npl)
                LCmod,compo = TM.get_value(t_in, ss=s_samp[j],grprs=grprs_here,vcont=cont_in,Rstar=Rstar,model_phasevar=model_phasevar[k],
                                            custom_LCfunc=custom_LCfunc if ncustom>0 else None, pc_model=pc_model[k])

            model_outputs.lc[name] = LCmod, compo
            continue

        #compute transit model
        if nttv>0:
            mt0, _ = TTV_Model(tarr=t_in, rho_star=rhoin, dur=durin, T0_list=ttv_conf[j].t0_list, RpRs=RpRsin, b=bbin, per=perin, sesinw=sesinwin, secosw=secoswin,
                                    ddf=ddf0,occ=occin, Fn=Fn_in, delta=phoff_in, A_ev=Aev_in, f1_ev=f1ev_in, A_db=Adb_in,q1=q1in, q2=q2in,split_conf=ttv_conf[j],ss=s_samp[j],vcont=cont_in,Rstar=Rstar,grprs=grprs_here,
                                    custom_LCfunc=custom_LCfunc if ncustom>0 else None,cst_pars=cst_pars)
        else:
            TM = Planet_LC_Model(rho_star=rhoin, dur=durin, T0=T0in, RpRs=RpRsin, b=bbin, per=perin, sesinw=sesinwin, secosw=secoswin, ddf=ddf0, 
                                        occ=occin, Fn=Fn_in, delta=phoff_in, A_ev=Aev_in, f1_ev=f1ev_in, A_db=Adb_in,q1=q1in, q2=q2in,cst_pars=cst_pars,npl=npl)
            mt0, _ = TM.get_value(t_in,ss=s_samp[j], grprs=grprs_here, vcont=cont_in,Rstar=Rstar,model_phasevar=model_phasevar[k],
                                    custom_LCfunc=custom_LCfunc if ncustom>0 else None, pc_model=pc_model[k])   
        
        
        if inmcmc=="n" and Rstar!=None and j==0:   #calculate light travel time correction for each planet and plot
            matplotlib.use('Agg')
            fig,ax = plt.subplots(npl,1, figsize=(12,3*npl),sharex=True)
            if npl==1: ax = [ax]
            ax[0].set_title("Light Travel Time Correction")
            for i in range(npl):
                _b,_D,_rho,_P,_rp = bbin[i],durin,rhoin,perin[i],RpRsin[i]+ddf0
                _e, _w = sesinw_secosw_to_ecc_omega(sesinwin[i],secoswin[i])
                _aR  = rho_to_aR(_rho,_P,_e,np.rad2deg(_w)) if _rho != None else Tdur_to_aR(_D,_b,_rp,_P,_e,np.rad2deg(_w))
                _inc = np.arccos(_b/(_aR*(1-_e**2)/(1+_e*np.sin(_w))))
                
                tsmooth = np.linspace(-0.25,1.25,300)
                t_ltt = light_travel_time_correction(t=tsmooth,t0=0,aR=_aR,P=1,inc=_inc, Rstar=Rstar,ecc=_e,w=_w)
                ax[i].plot(tsmooth,24*3600*(tsmooth-t_ltt), label=f"Planet {i+1}")
                ax[i].set_ylabel("LTT [s]")
                tconj = get_Tconjunctions(t=tsmooth,t0=0,per=1,ecc=_e,omega=_w,Rstar=Rstar,aR=_aR,inc=_inc,verbose=False)
                
                ax[i].axvline(tsmooth[np.argmax(tsmooth-t_ltt)], color="g",ls="--", label="maximum LTT delay")
                ax[i].axvline(tconj.transit, color="k",ls=":",label="mid-transit")
                ax[i].axvline(tconj.eclipse, color="r",ls=":",label="mid-eclipse")
                if i==0: ax[i].legend()
            
            if out_folder!="": fig.savefig(f"{out_folder}/LTT.png",bbox_inches="tight")
            matplotlib.use(__default_backend__)

        # compute baseline model (w/ or w/o spline)
        bfstart = 1 + 7*npl + nttv + nddf +nocc*7 +2*nfilt + nphot + ncustom + nsin + nRV*2+ j*22  # index in params of the first baseline param of this light curve
        blind   = np.asarray(list(range(bfstart,bfstart+22))) # the indices of the baseline params of this light curve
        basesin = np.zeros(22)
        
        for jj in range(len(blind)):
            basein = blind[jj]
            if (basein in jumping[0]):
                basesin[jj] = pp[ppcount]
                ppcount = ppcount + 1
            else:
                basesin[jj]=params_all[basein]

        coeff = np.copy(basesin)
        if (baseLSQ == 'y'):
            coeffstart  = basesin[bvar]
            icoeff, _   = scipy.optimize.leastsq(para_minfunc, coeffstart, args=(bvar, mt0, thisLCdata))
            coeff[bvar] = icoeff 
        
        base_para, base_spl = basefuncLC(coeff, thisLCdata, f_in/mt0,useSpline_lc[LCnames[j]])
        base_total = (base_para+base_sine)*base_spl
        trans_base = mt0  * base_total    #transit*baseline(para*spl)
        det_LC     = f_in / base_total    #detrended data   

        # lc jitter 
        LCjitterind = 1+7*npl + nttv + nddf + nocc*7 + nfilt*2 + j
        LCjit       = np.exp(params_all[LCjitterind])
        err_in      = (e_in**2 + LCjit**2)**0.5

        time_all.append(t_in)
        flux_all.append(f_in)
        err_all.append(err_in)
        full_mod_all.append(trans_base)
        base_para_all.append(base_para)
        base_sine_all.append(base_sine)
        base_spl_all.append(base_spl)
        base_gp_all.append(np.zeros(len(t_in)))
        base_total_all.append(base_total)
        transit_all.append(mt0)
        det_flux_all.append(det_LC)
        residual_all.append(f_in-trans_base)
    
    if not get_planet_model and nphot>0:
        for j in range(nphot):
            if useGPphot[j]=='n':
                if inmcmc == 'y':
                    lnprob_thislc = -1./2. * np.sum( (flux_all[j] - full_mod_all[j])**2/err_all[j]**2 + np.log(2*np.pi*err_all[j]**2))
                    lnprob = lnprob + lnprob_thislc

                if inmcmc == 'n':
                    mod = np.concatenate((mod,full_mod_all[j])) #append the model to the output array   
                    emod = np.concatenate((emod,err_all[j])) #error array including jitter

                    # write the lightcurve and the model to file or return output if we're not inside the MCMC
                    out_data   = np.stack((time_all[j],flux_all[j],err_all[j],full_mod_all[j],base_para_all[j],base_sine_all[j],base_spl_all[j],base_gp_all[j],base_total_all[j],transit_all[j],det_flux_all[j],residual_all[j]),axis=1)
                    header     = ["time","flux","error","full_mod","base_para","base_sine","base_spl","base_gp","base_total","transit","det_flux","residual"]
                    header_fmt = "{:<16s}\t"*len(header)
                    phases     = np.zeros((len(time_all[j]),npl))

                    for n in range(npl):
                        phases[:,n] = np.modf(np.modf( (time_all[j]-T0in[n])/perin[n])[0]+1)[0]
                        if transit_all[j][np.argmin(phases[:,n])] < 1:
                            phases[:,n][phases[:,n]>0.5] = phases[:,n][phases[:,n]>0.5]-1
                        header_fmt += "{:<16s}\t"
                        header     += [f"phase_{n+1}"] if npl>1 else ["phase"]

                    out_data = np.hstack((out_data,phases))
                    if make_outfile:
                        outfile=out_folder+"/out_data/"+splitext(LCnames[j])[0]+'_lcout.dat' 
                        if verbose: print(f"    - Writing LC output to file: {outfile}")
                        np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%-16.6f',delimiter="\t")
            
            elif useGPphot[j] in ['ge','ce','sp'] and sameLCgp.flag==False and sameLCgp.filtflag==False: 
                thisLCdata = input_lcs[LCnames[j]]

                gppars        = params_all[len(params):len(params)+len(GPparams)]   # the GP parameters for all lcs
                gppars_names  = pnames_all[len(params):len(params)+len(GPparams)]   # the GP parameter names for all lcs
                gpcol         = gp_colnames[j]
                pargp_arr     = thisLCdata[gpcol] if isinstance(gpcol, str) else np.column_stack( [thisLCdata[c] for c in gpcol]) 
                ndim_gp       = 1 if isinstance(gpcol, str) else len(set(gpcol)) #number of independent axis columns for the GP

                srt_gp        = np.argsort(pargp_arr) if pargp_arr.ndim==1 else np.argsort(pargp_arr[:,0])  #indices to sort the gp axis
                unsrt_gp      = np.argsort(srt_gp)  #indices to unsort the gp axis
                
                #indices of the gp params for this lc file
                thislc_gpind        = GPindex == j         #individual gp params         
                thislc_gppars       = gppars[thislc_gpind] #the gp params for this lc
                thislc_gppars_names = gppars_names[thislc_gpind] #the gp params for this rv

                #need conversion of gp pars before setting them
                gp_conv            = gp_params_convert()
                kernels            = [f"{useGPphot[j]}_{gpk}" for gpk in gpkerns[j]]  #prepend correct GP package symbol for the kernel
                thislc_gppars_orig = gp_conv.get_values(kernels=kernels, data="lc",pars=thislc_gppars)
                
                if useGPphot[j] in ['ge','ce']:
                    gp = GPobjects[j]      #gp for this lc
                    gp.set_parameter_vector(thislc_gppars_orig)
                    gp.compute(pargp_arr[srt_gp], yerr = err_all[j][srt_gp]) #compute gp with jitter included 
                else:
                    gp = cov.Cov(t=pargp_arr[srt_gp],err=term.Error(err_all[j][srt_gp]),kterm=deepcopy(GPobjects[j])) #create the spleaf gp error term and kernel term. use a copy of kernel as spleaf doesnt allow to reuse the kernel this way
                    gp.set_param(thislc_gppars_orig)


                if inmcmc == 'y':
                    if useGPphot[j] in ['ge','ce']: 
                        lnprob_thislc = gp.log_likelihood((residual_all[j])[srt_gp], quiet=True)  
                    else:                           
                        lnprob_thislc = gp.loglike((residual_all[j])[srt_gp])
                        del gp #delete the kernel term to avoid memory issues
                    lnprob = lnprob + lnprob_thislc
                
                # if not in MCMC, get a prediction and append it to the output array
                if inmcmc == 'n':
                    which_GP = "George" if useGPphot[j]=='ge' else "Celerite" if useGPphot[j]=='ce' else "Spleaf"

                    if useGPphot[j] in ['ge','ce']: 
                        base_gp_all[j] = gp.predict((residual_all[j])[srt_gp], t=pargp_arr[srt_gp], return_cov=False, return_var=False)[unsrt_gp] #gp_fit to residual
                    else:                           
                        base_gp_all[j] = gp.conditional((residual_all[j])[srt_gp], pargp_arr[srt_gp], calc_cov=False)[unsrt_gp] #gp_fit to residual


                    lcGPsave[LCnames[j]] = GPSaveObj(fname=LCnames[j], gp_pck=useGPphot[j], gp=gp, kernel=kernels,
                                                        ndim_gp=ndim_gp, gp_cols=gpcol, gp_cols_err=None,
                                                        gp_pars = {k:v for k,v in zip(thislc_gppars_names,thislc_gppars) },
                                                        x=pargp_arr, residuals=(residual_all[j])[srt_gp])

                    full_mod_all[j]   = full_mod_all[j] + base_gp_all[j]    #update trans_base with base_gp [transit*baseline(para*spl) + gp]
                    mod               = np.concatenate((mod,full_mod_all[j])) #append the model to the output array   
                    emod              = np.concatenate((emod,err_all[j])) #error array including jitter

                    # base_total = base_total/(1-base_gp/f_in)    #base_total + base_gp         #update base_total with base_gp
                    det_flux_all[j]   = (flux_all[j] - base_gp_all[j])/( (base_para_all[j]+base_sine_all[j])*base_spl_all[j])
                    base_total_all[j] = flux_all[j]/det_flux_all[j]
                    residual_all[j]   = flux_all[j] - full_mod_all[j]
                    
                    # write the lightcurve and the model to file or return output if we're not inside the MCMC
                    out_data     = np.stack((time_all[j],flux_all[j],err_all[j],full_mod_all[j],base_para_all[j],base_sine_all[j],base_spl_all[j],base_gp_all[j],base_total_all[j],transit_all[j],det_flux_all[j],residual_all[j]),axis=1)
                    header       = ["time","flux","error","full_mod","base_para","base_sine","base_spl","base_gp","base_total","transit","det_flux","residual"]
                    header_fmt   = "{:<16s} "*len(header)
                    phases       = np.zeros((len(time_all[j]),npl))

                    for n in range(npl):
                        phases[:,n] = np.modf(np.modf( (time_all[j]-T0in[n])/perin[n])[0]+1)[0]
                        if transit_all[j][np.argmin(phases[:,n])] < 1: phases[:,n][phases[:,n]>0.5] = phases[:,n][phases[:,n]>0.5]-1
                        header_fmt += "{:<16s} "
                        header     += [f"phase_{n+1}"] if npl>1 else ["phase"]

                    out_data = np.hstack((out_data,phases))
                    if make_outfile:
                        outfile=out_folder+"/out_data/"+splitext(LCnames[j])[0]+'_lcout.dat'
                        if verbose: print(f"    - Writing LC output with GP({which_GP}) to file: {outfile}")
                        np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%-16.6f',delimiter=" ")

            else:
                pass

        if sameLCgp.flag==True: #if using the same GP for multiple LCs
            gppars       = params_all[len(params):len(params)+len(GPparams)]   # the GP parameters for all lcs
            gppars_names = pnames_all[len(params):len(params)+len(GPparams)]   # the GP parameter names for all lcs
            gpcol        = gp_colnames[sameLCgp.first_index]
            ndim_gp      = 1 if isinstance(gpcol, str) else len(set(gpcol)) #number of independent axis columns for the GP

            if isinstance(gpcol, str):
                pargp_arr = np.concatenate([input_lcs[nm][gpcol] for nm in sameLCgp.LCs])   #join column data for all datasets with useGPphot !="n"
            else:
                pargp_arr = np.column_stack( [np.concatenate([input_lcs[nm][c] for nm in sameLCgp.LCs]) for c in gpcol ])
                                        
            srt_gp   = np.argsort(pargp_arr) if pargp_arr.ndim==1 else np.argsort(pargp_arr[:,0])  #indices to sort the gp axis
            unsrt_gp = np.argsort(srt_gp)  #indices to unsort the gp axis

            
            same_lc_gpind        = GPindex == sameLCgp.first_index          
            same_lc_gppars       = gppars[same_lc_gpind] #the gp params for the lcs in sameLCgp
            same_lc_gppars_names = gppars_names[same_lc_gpind] #the gp params for the lcs in sameLCgp 

            #need conversion of gp pars before setting them
            gp_conv             = gp_params_convert()
            kernels             = [f"{useGPphot[sameLCgp.first_index]}_{gpk}" for gpk in gpkerns[sameLCgp.first_index ]]#prepend correct GP package symbol for the kernel--> ce_sho,sp_mat32...
            same_lc_gppars_orig = gp_conv.get_values(kernels=kernels, data="lc",pars=same_lc_gppars)
            
            if useGPphot[sameLCgp.first_index] in ['ge','ce']:
                gp  = GPobjects[sameLCgp.first_index]
                gp.set_parameter_vector(same_lc_gppars_orig)
                gp.compute(pargp_arr[srt_gp], yerr = np.concatenate([err_all[i] for i in sameLCgp.indices])[srt_gp])
            else: #spleaf
                gp = cov.Cov(t=pargp_arr[srt_gp], err=term.Error(np.concatenate([err_all[i] for i in sameLCgp.indices])[srt_gp]),
                                kterm=deepcopy(GPobjects[sameLCgp.first_index]))
                gp.set_param(same_lc_gppars_orig)


            if inmcmc == 'y':
                if useGPphot[sameLCgp.first_index] in ['ge','ce']: 
                    lnprob_allLC = gp.log_likelihood((np.concatenate([residual_all[i] for i in sameLCgp.indices]))[srt_gp], quiet=True)
                else:
                    lnprob_allLC = gp.loglike((np.concatenate([residual_all[i] for i in sameLCgp.indices]))[srt_gp])
                    del gp
                lnprob  = lnprob + lnprob_allLC

            # if not in MCMC, get a prediction and append it to the output array
            if inmcmc == 'n':
                resid_cctn = np.concatenate([residual_all[i] for i in sameLCgp.indices])
                for j in sameLCgp.indices:
                    if isinstance(gpcol, str):
                        thispargp_arr = input_lcs[LCnames[j]][gpcol]  
                    else:
                        thispargp_arr = np.column_stack( [input_lcs[LCnames[j]][c] for c in gpcol] )
                    
                    thissrt_gp   = np.argsort(thispargp_arr) if thispargp_arr.ndim==1 else np.argsort(thispargp_arr[:,0])  #indices to sort the gp axis
                    thisunsrt_gp = np.argsort(thissrt_gp)  #indices to unsort the gp axis
                    which_GP     = "George" if useGPphot[j]=='ge' else "Celerite" if useGPphot[j]=='ce' else "Spleaf"
                    
                    if useGPphot[j] in ['ge','ce']:
                        base_gp_all[j]  = gp.predict(resid_cctn[srt_gp], t=thispargp_arr[thissrt_gp], 
                                                        return_cov=False, return_var=False)[thisunsrt_gp] #gp_fit to residual
                    else: #spleaf
                        base_gp_all[j]  = gp.conditional(resid_cctn[srt_gp], thispargp_arr[thissrt_gp], calc_cov=False)[thisunsrt_gp]

                    
                    lcGPsave[LCnames[j]] = GPSaveObj(fname=LCnames[j], gp_pck=useGPphot[j], gp=gp, kernel=kernels,
                                                        ndim_gp=ndim_gp, gp_cols=gpcol, gp_cols_err=None,
                                                        gp_pars = {k:v for k,v in zip(same_lc_gppars_names,same_lc_gppars) },
                                                        x=thispargp_arr, residuals=resid_cctn[srt_gp])

                    full_mod_all[j] = full_mod_all[j] + base_gp_all[j]    #update trans_base with base_gp [gp + transit*baseline(para*spl)]
                    mod             = np.concatenate((mod,full_mod_all[j])) #append the model to the output array   
                    emod            = np.concatenate((emod,err_all[j])) #error array including jitter

                    # base_total = base_total/(1-base_gp/f_in)    #base_total + base_gp         #update base_total with base_gp
                    det_flux_all[j]   = (flux_all[j] - base_gp_all[j])/( (base_para_all[j]+base_sine_all[j])*base_spl_all[j])
                    base_total_all[j] = flux_all[j]/det_flux_all[j]
                    residual_all[j]   = flux_all[j] - full_mod_all[j]
                    
                    # write the lightcurve and the model to file or return output if we're not inside the MCMC
                    out_data     = np.stack((time_all[j],flux_all[j],err_all[j],full_mod_all[j],base_para_all[j],base_sine_all[j],base_spl_all[j],base_gp_all[j],base_total_all[j],transit_all[j],det_flux_all[j],residual_all[j]),axis=1)
                    header       = ["time","flux","error","full_mod","base_para","base_sine","base_spl","base_gp","base_total","transit","det_flux","residual"]
                    header_fmt   = "{:<16s} "*len(header)
                    phases       = np.zeros((len(time_all[j]),npl))

                    for n in range(npl):
                        phases[:,n] = np.modf(np.modf( (time_all[j]-T0in[n])/perin[n])[0]+1)[0]
                        if transit_all[j][np.argmin(phases[:,n])] < 1: phases[:,n][phases[:,n]>0.5] = phases[:,n][phases[:,n]>0.5]-1
                        header_fmt += "{:<16s} "
                        header     += [f"phase_{n+1}"] if npl>1 else ["phase"]

                    out_data = np.hstack((out_data,phases))
                    if make_outfile:
                        outfile=out_folder+"/out_data/"+splitext(LCnames[j])[0]+'_lcout.dat'
                        if verbose: print(f"    - Writing LC output with GP({which_GP}) to file: {outfile}")
                        np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%-16.6f',delimiter=" ")

        if sameLCgp.filtflag == True: #if using the same GP for each filter
            gppars       = params_all[len(params):len(params)+len(GPparams)]   # the GP parameters for all lcs
            gppars_names = pnames_all[len(params):len(params)+len(GPparams)]   # the GP parameter names for all lcs

            for filt in filnames:
                resid_cctn_filt = np.concatenate([residual_all[i] for i in sameLCgp.indices[filt]])
                gpcol           = gp_colnames[sameLCgp.first_index[filt]]
                ndim_gp         = 1 if isinstance(gpcol, str) else len(set(gpcol)) #number of independent axis columns for the GP

                if isinstance(gpcol, str):
                    pargp_arr = np.concatenate([input_lcs[nm][gpcol] for nm in sameLCgp.LCs[filt]])   #join column data for all datasets of this filter with useGPphot !="n"
                else:
                    pargp_arr = np.column_stack( [np.concatenate([input_lcs[nm][c] for nm in sameLCgp.LCs[filt]]) for c in gpcol ])
                
                srt_gp   = np.argsort(pargp_arr) if pargp_arr.ndim==1 else np.argsort(pargp_arr[:,0])  #indices to sort the gp axis
                unsrt_gp = np.argsort(srt_gp)  #indices to unsort the gp axis


                same_lc_gpind        = GPindex == sameLCgp.first_index[filt]          
                same_lc_gppars       = gppars[same_lc_gpind] #the gp params 
                same_lc_gppars_names = gppars_names[same_lc_gpind] #the gp params for the lcs in sameLCgp

                #need conversion of gp pars before setting them
                gp_conv             = gp_params_convert()
                kernels             = [f"{useGPphot[sameLCgp.first_index[filt]]}_{gpk}" for gpk in gpkerns[sameLCgp.first_index[filt] ]]#prepend correct GP package symbol for the kernel--> ce_sho,sp_mat32...
                same_lc_gppars_orig = gp_conv.get_values(kernels=kernels, data="lc",pars=same_lc_gppars)
                
                if useGPphot[sameLCgp.first_index[filt]] in ['ge','ce']:
                    gp  = GPobjects[sameLCgp.first_index[filt]]
                    gp.set_parameter_vector(same_lc_gppars_orig)
                    gp.compute(pargp_arr[srt_gp], yerr = np.concatenate([err_all[i] for i in sameLCgp.indices[filt]])[srt_gp])
                else: #spleaf
                    gp = cov.Cov(t=pargp_arr[srt_gp], err=term.Error(np.concatenate([err_all[i] for i in sameLCgp.indices[filt]])[srt_gp]),
                                    kterm=deepcopy(GPobjects[sameLCgp.first_index[filt]]))
                    gp.set_param(same_lc_gppars_orig)


                if inmcmc == 'y':
                    if useGPphot[sameLCgp.first_index[filt]] in ['ge','ce']: 
                        lnprob_filtLC = gp.log_likelihood(resid_cctn_filt[srt_gp], quiet=True)
                    else:
                        lnprob_filtLC = gp.loglike(resid_cctn_filt[srt_gp])
                        del gp
                    lnprob  = lnprob + lnprob_filtLC

                # if not in MCMC, get a prediction and append it to the output array
                if inmcmc == 'n':      
                    for j in sameLCgp.indices[filt]: #loop over the LCs in this filter
                        if isinstance(gpcol, str) :
                            thispargp_arr   = input_lcs[LCnames[j]][gpcol] 
                        else: 
                            thispargp_arr   = np.column_stack( [input_lcs[LCnames[j]][c] for c in gpcol] ) #get the gp column data for this lc

                        thissrt_gp      = np.argsort(thispargp_arr) if thispargp_arr.ndim==1 else np.argsort(thispargp_arr[:,0])  #indices to sort the gp axis
                        thisunsrt_gp    = np.argsort(thissrt_gp)  #indices to unsort the gp axis
                        which_GP        = "George" if useGPphot[j]=='ge' else "Celerite" if useGPphot[j]=='ce' else "Spleaf"
                        
                        if useGPphot[j] in ['ge','ce']:
                            base_gp_all[j]  = gp.predict(resid_cctn_filt[srt_gp], t=thispargp_arr[thissrt_gp], 
                                                            return_cov=False, return_var=False)[thisunsrt_gp] #gp_fit to residual
                        else:
                            base_gp_all[j]  = gp.conditional(resid_cctn_filt[srt_gp], thispargp_arr[thissrt_gp], calc_cov=False)[thisunsrt_gp]


                        lcGPsave[LCnames[j]] = GPSaveObj(fname=LCnames[j], gp_pck=useGPphot[j], gp=gp, kernel=kernels,
                                    ndim_gp=ndim_gp, gp_cols=gpcol, gp_cols_err=None,
                                    gp_pars = {k:v for k,v in zip(same_lc_gppars_names,same_lc_gppars) },
                                    x= thispargp_arr, residuals=resid_cctn_filt[srt_gp])
                        
                        full_mod_all[j] = full_mod_all[j] + base_gp_all[j]    #update trans_base with base_gp [gp + transit*baseline(para*spl)]
                        mod             = np.concatenate((mod,full_mod_all[j])) #append the model to the output array   
                        emod            = np.concatenate((emod,err_all[j])) #error array including jitter

                        # base_total = base_total/(1-base_gp/f_in)    #base_total + base_gp         #update base_total with base_gp
                        det_flux_all[j]   = (flux_all[j] - base_gp_all[j])/( (base_para_all[j]+base_sine_all[j])*base_spl_all[j])
                        base_total_all[j] = flux_all[j]/det_flux_all[j]
                        residual_all[j]   = flux_all[j] - full_mod_all[j]
                        
                        # write the lightcurve and the model to file or return output if we're not inside the MCMC
                        out_data     = np.stack((time_all[j],flux_all[j],err_all[j],full_mod_all[j],base_para_all[j],base_sine_all[j],base_spl_all[j],base_gp_all[j],base_total_all[j],transit_all[j],det_flux_all[j],residual_all[j]),axis=1)
                        header       = ["time","flux","error","full_mod","base_para","base_sine","base_spl","base_gp","base_total","transit","det_flux","residual"]
                        header_fmt   = "{:<16s} "*len(header)
                        phases       = np.zeros((len(time_all[j]),npl))

                        for n in range(npl):
                            phases[:,n] = np.modf(np.modf( (time_all[j]-T0in[n])/perin[n])[0]+1)[0]
                            if transit_all[j][np.argmin(phases[:,n])] < 1: phases[:,n][phases[:,n]>0.5] = phases[:,n][phases[:,n]>0.5]-1
                            header_fmt += "{:<16s} "
                            header     += [f"phase_{n+1}"] if npl>1 else ["phase"]

                        out_data = np.hstack((out_data,phases))
                        if make_outfile:
                            outfile=out_folder+"/out_data/"+splitext(LCnames[j])[0]+'_lcout.dat'
                            if verbose: print(f"    - Writing LC output with GP({which_GP}) to file: {outfile}")
                            np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%-16.6f',delimiter=" ")
    
        if get_base_model:
            for j in range(nphot):
                basemodel_outputs.lc[LCnames[j]] = base_total_all[j]


    # now do the RVs and add their probabilities to the model
    time_all, RV_all, err_all, full_mod_all, base_para_all, base_spl_all, base_gp_all, base_total_all, rvmod_all, det_rv_all, gamma_all, residual_all = [],[],[],[],[],[],[],[],[],[],[],[]
    rvGPsave = dict()

    for j in range(nRV):
        # if verbose: print(f'RV{j+1}', end= " ...")

        name       = RVnames[j]
        thisRVdata = input_rvs[name]

        t_in  = thisRVdata["col0"] if t is None else t
        RV_in = thisRVdata["col1"]
        e_in  = thisRVdata["col2"]

        if custom_RVfunc.func!=None and custom_RVfunc.replace_RVmodel==False:
            if custom_RVfunc.x.startswith('col'):
                custom_RVfunc.xarr = thisRVdata[custom_RVfunc.x]
                custom_RVfunc.t    = thisRVdata['col0']   #needed to create interpolate func for other columns 

        if baseLSQ == "y": bvarRV = bvarsRV[j][0]
        else : bvarRV = []

        # get the current parameters from the pp array.
        pp      = p[pindices[j+nphot]]  # the elements of the p (noGP) array jumping in this RV curve
        ppnames = jnames[pindices[j+nphot]]

        ppcount = 0 
        T0in, RpRsin, bbin, perin, sesinwin, secoswin, Kin = [], [], [], [], [], [],[]
        rhoin, durin = None, None

        if "rho_star" in pnames_all:    #using rho_star  for transit model
            if 0 in jumping[0]:
                rhoin = pp[ppcount]
                ppcount = ppcount+1
            else:
                rhoin = params_all[0]
        else:  #using duration for transit model
            if 0 in jumping[0]:
                durin = pp[ppcount]
                ppcount = ppcount+1
            else:
                durin = params_all[0]

        for n in range(npl):
            if (1+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                T0in.append(pp[ppcount])        # the first element of pp is the T0 value
                ppcount = ppcount+1 
            else:
                T0in.append(params_all[1+7*n])
                
            if (2+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                RpRsin.append(pp[ppcount])
                ppcount = ppcount+1   
            else:
                RpRsin.append(params_all[2+7*n])

            if (3+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                bbin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                bbin.append(params_all[3+7*n])
                    
            if (4+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                perin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                perin.append(params_all[4+7*n])

            if (5+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                sesinwin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                sesinwin.append(params_all[5+7*n])

            if (6+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                secoswin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                secoswin.append(params_all[6+7*n])

            if (7+7*n in jumping[0]):   # same for all data -> check in jumping array
                Kin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                Kin.append(params_all[7+7*n])
        

        gammaind = 1+7*npl + nttv+nddf + nocc*7 + nfilt*2 + nphot + ncustom + nsin + j*2   #pass the right gamma index for each file (Akin)
        Gamma_in = params_all[gammaind]

        # rv jitter 
        jitterind = 1+7*npl + nttv+ nddf + nocc*7 + nfilt*2 + nphot + ncustom + nsin + j*2 + 1
        jit       = params_all[jitterind]
        err_in    =  (e_in**2 + jit**2)**0.5

        ncustomRV  = custom_RVfunc.npars if custom_RVfunc!=None else 0 # number of custom RV function parameters
        cstRV_pars = {}
        if ncustomRV>0:
            customRVind = 1+7*npl + nttv+nddf+nocc*7 + nfilt*2 + nphot + ncustom + nsin + 2*nRV + np.arange(ncustomRV)
            cstRV_pars
            for kk,cst_ind in enumerate(customRVind):
                    cstRV_pars[list(custom_RVfunc.par_dict.keys())[kk]] = params_all[cst_ind]
        
        if get_planet_model:   #skip other calculations and return the RV model for this dataset
            RVmodel,compo = Planet_RV_Model(t_in,T0in,perin,Kin,sesinwin,secoswin,Gamma=0,cst_pars=cstRV_pars,npl=npl,
                                                    custom_RVfunc=custom_RVfunc if ncustomRV>0 else None)
            model_outputs.rv[name] = RVmodel,compo
            continue     # skip the rest of the loop, go to next rv file

        #compute planet RV model
        mod_RV, _    = Planet_RV_Model(t_in,T0in,perin,Kin,sesinwin,secoswin,Gamma=0,cst_pars=cstRV_pars,npl=npl,
                                            custom_RVfunc=custom_RVfunc if ncustomRV>0 else None)
        mod_RV_gamma = mod_RV + Gamma_in

        #compute baseline model (w/ or w/o spline)
        bfstartRV = 1 + 7*npl + nttv + nddf + nocc*7 + 2*nfilt + nphot+ ncustom+ nsin + 2*nRV + ncustomRV + nphot*22 +j*12  #the first index in the param array that refers to a baseline function
        incoeff   = np.array(list(range(bfstartRV,bfstartRV+12)))  # the indices for the coefficients for the baseline function        
        rvbasesin = params_all[incoeff]

        coeff = np.copy(rvbasesin)   # the full coefficients -- set to those defined in params (in case any are fixed non-zero)
        if (baseLSQ == 'y'):
            coeffstart = rvbasesin[bvarRV]  # the jumping coefficients
            if len(bvarRV) > 0:
                icoeff, _ = scipy.optimize.leastsq(para_minfuncRV, coeffstart, args=(bvarRV, mod_RV_gamma, thisRVdata))
                coeff[bvarRV] = np.copy(icoeff)     # and the variable ones are set to the result from the minimization

        base_paraRV, base_splRV = basefuncRV(coeff, thisRVdata, res=RV_in-mod_RV_gamma,useSpline=useSpline_rv[RVnames[j]])
        base_totalRV = base_paraRV + base_splRV + Gamma_in
        mod_RV_base  = mod_RV + base_totalRV   #full RVmodel+baseline(para+spl+gamma)
        det_RV       = RV_in  - base_totalRV    #detrended data

        time_all.append(t_in)
        RV_all.append(RV_in)
        err_all.append(err_in)
        full_mod_all.append(mod_RV_base)
        base_para_all.append(base_paraRV)
        base_spl_all.append(base_splRV)
        base_gp_all.append(np.zeros(len(t_in)))
        base_total_all.append(base_totalRV)
        rvmod_all.append(mod_RV)
        det_rv_all.append(det_RV)
        gamma_all.append(np.ones_like(t_in)*Gamma_in)
        residual_all.append(det_RV-mod_RV)      

    if not get_planet_model and nRV>0:
        for j in range(nRV):
            if useGPrv[j] == "n":
                if inmcmc == 'y':
                    lnprob_thisRV = -1./2. * np.sum( (RV_all[j] - full_mod_all[j])**2/err_all[j]**2 + np.log(2*np.pi*err_all[j]**2))
                    lnprob = lnprob + lnprob_thisRV

                if inmcmc == 'n':
                    mod  = np.concatenate((mod,full_mod_all[j]))
                    emod = np.concatenate((emod,err_all[j]))
            
                    # write the RVcurve and the model to file if we're not inside the MCMC
                    out_data   = np.stack((time_all[j],RV_all[j],err_all[j],full_mod_all[j],base_para_all[j],base_spl_all[j],base_gp_all[j],base_total_all[j],rvmod_all[j],det_rv_all[j],gamma_all[j],residual_all[j]),axis=1)
                    header     = ["time","RV","error","full_mod","base_para","base_spl","base_gp","base_total","Rvmodel","det_RV","gamma","residual"]
                    header_fmt = "{:<16s}\t"*len(header)
                    phases     = np.zeros((len(time_all[j]),npl))

                    for n in range(npl):
                        phases[:,n] = ((time_all[j]-T0in[n])/perin[n]) - np.round( ((time_all[j]-T0in[n])/perin[n]))
                        header_fmt += "{:<16s}\t"
                        header     += [f"phase_{n+1}"] if npl>1 else ["phase"]
                    if make_outfile:   
                        outfile  = out_folder+"/out_data/"+splitext(RVnames[j])[0]+'_rvout.dat'
                        out_data = np.hstack((out_data,phases))
                        if verbose: print(f"    - Writing RV output to file: {outfile}")
                        np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%-16.6f',delimiter="\t")
                        # pd.DataFrame(out_data,columns=header).to_csv(outfile,sep="\t",index=False, float_format='%-16.6f')

            elif useGPrv[j] in ['ge','ce','sp'] and sameRVgp.flag==False:
                thisRVdata = input_rvs[RVnames[j]]

                rv_gppars       = params_all[len(params)+len(GPparams):]   #all rv gp parameters
                rv_gppars_names = pnames_all[len(params)+len(GPparams):] 
                gpcol           = rv_gp_colnames[j] 
                gpcol_err       = rv_gp_colerr_names[j]
                ndim_gp         = 1 if isinstance(gpcol, str) else len(set(gpcol)) #number of independent axis columns for the GP
                rvpargp_arr     = rv_pargps[j]            #independent axis columns
                # rvpargp    = thisRVdata[gpcol] if isinstance(gpcol, str) else np.column_stack( [thisRVdata[c] for c in gpcol]) 

                #indices of the gp params for this rv file
                thisrv_gpind        = rvGPindex == j          #individual gp params 
                thisrv_gppars       = rv_gppars[thisrv_gpind] #the gp params for this RV
                thisrv_gppars_names = rv_gppars_names[thisrv_gpind]

                if useGPrv[j]=='sp' and ndim_gp>1:
                    thisrv_gppars, gp_off, gp_jitt = thisrv_gppars[:-2*(ndim_gp-1)], thisrv_gppars[-2*(ndim_gp-1):-(ndim_gp-1)], thisrv_gppars[-(ndim_gp-1):] #get the gp parameters for this rv file, offset and jitter
                    thisrv_gppars_names = thisrv_gppars_names[:-2*(ndim_gp-1)]

                #need conversion of gp pars before setting them
                gp_conv             = gp_params_convert()
                kernels             = [f"{useGPrv[j]}_{gpk}" for gpk in rv_gpkerns[j]] #prepend correct GP package symbol for the kernel
                thisrv_gppars_orig  = gp_conv.get_values(kernels=kernels, data="rv",pars=thisrv_gppars, gp_dim=ndim_gp)
                
                if useGPrv[j] in ['ge','ce']:
                    srt_rvgp   = np.argsort(rvpargp_arr) if rvpargp_arr.ndim==1 else np.argsort(rvpargp_arr[:,0])  #indices to sort the gp axis
                    unsrt_rvgp = np.argsort(srt_rvgp)  #indices to unsort the gp axis

                    gp  = rvGPobjects[j]      #gp for this rv
                    gp.set_parameter_vector(thisrv_gppars_orig)
                    gp.compute(rvpargp_arr[srt_rvgp], yerr = err_in[srt_rvgp])
                else:
                    if ndim_gp == 1:
                        srt_rvgp   = np.argsort(rvpargp_arr) if rvpargp_arr.ndim==1 else np.argsort(rvpargp_arr[:,0])  #indices to sort the gp axis
                        unsrt_rvgp = np.argsort(srt_rvgp)  #indices to unsort the gp axis

                        gp = cov.Cov(t=rvpargp_arr[srt_rvgp],err=term.Error(err_in[srt_rvgp]),kterm=deepcopy(rvGPobjects[j])) #create the spleaf gp from the dictionary of kernels
                        gp.set_param(thisrv_gppars_orig)
                    else:
                        # Merge all time series
                        Yerr     = [thisRVdata[c] for c in gpcol_err]
                        Yerr[0]  = err_all[j]  #replace RV error with the jitter already added
                        for ee in range(1,ndim_gp):       #add jitter for error of each timeseries
                            Yerr[ee] = np.sqrt(Yerr[ee]**2 + gp_jitt[ee-1]**2) 
                        
                        Y    = [thisRVdata[c] for c in gpcol]
                        Y[0] = residual_all[j]     #replace RV timeseries with residual after removing planet
                        for ee in range(1, ndim_gp):
                            Y[ee] = Y[ee] - gp_off[ee-1]

                        T    = [thisRVdata["col0"]]*ndim_gp #time array is same for all timeseries

                        t_full, y_full, yerr_full, series_index = cov.merge_series(T, Y, Yerr)
                        gp = cov.Cov(   t   = t_full, 
                                        err = term.Error(yerr_full),
                                        GP  = term.MultiSeriesKernel(deepcopy(rvGPobjects[j]), series_index,
                                                                        np.ones(ndim_gp), np.ones(ndim_gp)),
                                    )
                        gp.set_param(thisrv_gppars_orig)


                if inmcmc == 'y':
                    if useGPrv[j] in ['ge','ce']:
                        lnprob_thisRV = gp.log_likelihood((residual_all[j])[srt_rvgp], quiet=True)
                    else:
                        if ndim_gp == 1:
                            lnprob_thisRV = gp.loglike((residual_all[j])[srt_rvgp])
                        else:
                            lnprob_thisRV = gp.loglike(y_full)

                    lnprob = lnprob + lnprob_thisRV

                # if not in MCMC, get a prediction and append it to the output array
                if inmcmc == 'n':
                    which_GP = "George" if useGPrv[j]=='ge' else "Celerite" if useGPrv[j]=='ce' else "Spleaf"

                    if useGPrv[j] in ['ge','ce']:
                        base_gp_all[j]  = gp.predict((residual_all[j])[srt_rvgp], t=rvpargp_arr[srt_rvgp], return_cov=False, return_var=False)[unsrt_rvgp]
                    else:
                        if ndim_gp == 1:
                            base_gp_all[j]  = gp.conditional((residual_all[j])[srt_rvgp], rvpargp_arr[srt_rvgp], calc_cov=False)[unsrt_rvgp]
                        else:
                            gp.kernel["GP"].set_conditional_coef(series_id=0)
                            base_gp_all[j]  = gp.conditional(y_full, T[0], calc_cov=False)

                    rvGPsave[RVnames[j]] = GPSaveObj(fname=RVnames[j], gp_pck=useGPrv[j], gp=gp, kernel=kernels,
                                                        ndim_gp=ndim_gp, gp_cols=gpcol, gp_cols_err=gpcol_err,
                                                        gp_pars = {k:v for k,v in zip(thisrv_gppars_names,thisrv_gppars) },
                                                        x= T[0] if (useGPrv[j] == 'sp' and ndim_gp > 1) else rvpargp_arr, 
                                                        residuals=y_full if (useGPrv[j] == 'sp' and ndim_gp > 1) else (residual_all[j])[srt_rvgp])
                    
                    full_mod_all[j] = full_mod_all[j] + base_gp_all[j] #update mod_RV_base with base_gpRV [gp + planet rv + baseline(para+spl+gamma)]
                    mod             = np.concatenate((mod,full_mod_all[j]))
                    emod            = np.concatenate((emod,err_all[j]))

                    base_total_all[j] = base_total_all[j] + base_gp_all[j]  #update base_total with base_gp
                    det_rv_all[j]     = RV_all[j] - base_total_all[j]
                    residual_all[j]   = RV_all[j] - full_mod_all[j]

                    out_data   = np.stack((time_all[j],RV_all[j],err_all[j],full_mod_all[j],base_para_all[j],base_spl_all[j],base_gp_all[j],base_total_all[j],rvmod_all[j],det_rv_all[j],gamma_all[j],residual_all[j]),axis=1)
                    header     = ["time","RV","error","full_mod","base_para","base_spl","base_gp","base_total","Rvmodel","det_RV","gamma","residual"]
                    header_fmt = "{:<16s}\t"*len(header)
                    phases     = np.zeros((len(time_all[j]),npl))

                    for n in range(npl):
                        phases[:,n] = ((time_all[j]-T0in[n])/perin[n]) - np.round( ((time_all[j]-T0in[n])/perin[n]))
                        header_fmt += "{:<16s}\t"
                        header     += [f"phase_{n+1}"] if npl>1 else ["phase"]
                    
                    if make_outfile:   
                        out_data = np.hstack((out_data,phases))
                        outfile  = out_folder+"/out_data/"+splitext(RVnames[j])[0]+'_rvout.dat'
                        if verbose: print(f"    - Writing RV output with GP({which_GP}) to file: {outfile}")
                        np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%-16.6f',delimiter="\t")
            else:
                pass
        
        if sameRVgp.flag==True: 
            rv_gppars        = params_all[len(params)+len(GPparams):]   #all rv gp parameters
            rv_gppars_names  = pnames_all[len(params)+len(GPparams):]   #all rv gp parameters names
            gpcol            = rv_gp_colnames[sameRVgp.first_index] 
            gpcol_err        = rv_gp_colerr_names[sameRVgp.first_index]
            ndim_gp          = 1 if isinstance(gpcol, str) else len(set(gpcol)) #number of independent axis columns for the GP

            if ndim_gp == 1:
                rvpargp_arr = np.concatenate([input_rvs[nm][gpcol] for nm in sameRVgp.RVs])   #join column data for all datasets
            else:
                if useGPrv[sameRVgp.first_index]=='sp': #spleaf with multi series gp
                    rvpargp_arr = [np.concatenate([input_rvs[nm][c] for nm in sameRVgp.RVs]) for c in gpcol ] 
                else:
                    rvpargp_arr = np.column_stack( [np.concatenate([input_rvs[nm][c] for nm in sameRVgp.RVs]) for c in gpcol ])

            #indices of the gp params for this rv file
            same_rv_gpind        = rvGPindex == sameRVgp.first_index 
            same_rv_gppars       = rv_gppars[same_rv_gpind] #the gp params
            same_rv_gppars_names = rv_gppars_names[same_rv_gpind] #the gp params names

            if useGPrv[j]=='sp' and ndim_gp>1:
                same_rv_gppars, gp_off, gp_jitt = same_rv_gppars[:-2*(ndim_gp-1)], same_rv_gppars[-2*(ndim_gp-1):-(ndim_gp-1)], same_rv_gppars[-(ndim_gp-1):] #get the gp parameters for this rv file, offset and jitter
                same_rv_gppars_names = same_rv_gppars_names[:-2*(ndim_gp-1)] #get the gp parameters for this rv file

            #need conversion of gp pars to original package values before setting them
            gp_conv             = gp_params_convert()
            kernels             = [f"{useGPrv[sameRVgp.first_index]}_{gpk}" for gpk in rv_gpkerns[j]] #prepend correct GP package symbol for the kernel--> ce_sho,sp_mat32...
            same_rv_gppars_orig = gp_conv.get_values(kernels=kernels, data="rv",pars=same_rv_gppars)

            
            if useGPrv[sameRVgp.first_index] in ['ge','ce']:
                srt_rvgp   = np.argsort(rvpargp_arr) if ndim_gp == 1 else np.argsort(rvpargp_arr[:,0])  #indices to sort the gp axis
                unsrt_rvgp = np.argsort(srt_rvgp)  #indices to unsort the gp axis

                gp = rvGPobjects[sameRVgp.first_index]
                gp.set_parameter_vector(same_rv_gppars_orig)
                gp.compute(rvpargp_arr[srt_rvgp], yerr = np.concatenate([err_all[i] for i in sameRVgp.indices])[srt_rvgp])
            else:
                if ndim_gp == 1:
                    srt_rvgp   = np.argsort(rvpargp_arr) if ndim_gp == 1 else np.argsort(rvpargp_arr[:,0])
                    unsrt_rvgp = np.argsort(srt_rvgp)  #indices to unsort the gp axis

                    gp = cov.Cov(t=rvpargp_arr[srt_rvgp],err=term.Error(np.concatenate([err_all[i] for i in sameRVgp.indices])[srt_rvgp]),
                                    kterm=deepcopy(rvGPobjects[sameRVgp.first_index]))
                    gp.set_param(same_rv_gppars_orig)
                else:
                    #first merge time series across all rvs in sameRV then append gp columns in a list
                    Yerr    = [np.concatenate([input_rvs[nm][c] for nm in sameRVgp.RVs]) for c in gpcol_err]
                    Yerr[0] = np.concatenate([err_all[i] for i in sameRVgp.indices])
                    for ee in range(1,ndim_gp):       #add jitter for error of each timeseries
                        Yerr[ee] = np.sqrt(Yerr[ee]**2 + gp_jitt[ee-1]**2)

                    Y    = [np.concatenate([input_rvs[nm][c] for nm in sameRVgp.RVs]) for c in gpcol]
                    Y[0] = np.concatenate([residual_all[i] for i in sameRVgp.indices])
                    for ee in range(1, ndim_gp):
                        Y[ee] = Y[ee] - gp_off[ee-1]

                    T    = [np.concatenate([input_rvs[nm]["col0"] for nm in sameRVgp.RVs])]*ndim_gp #time array is same for all timeseries

                    t_full, y_full, yerr_full, series_index = cov.merge_series(T, Y, Yerr)
                    gp = cov.Cov(   t   = t_full, 
                                    err = term.Error(yerr_full),
                                    GP  = term.MultiSeriesKernel(deepcopy(rvGPobjects[sameRVgp.first_index]), series_index,
                                                                np.ones(ndim_gp), np.ones(ndim_gp)),
                                )
                    gp.set_param(same_rv_gppars_orig)


            if inmcmc == 'y':
                resid_cctn = np.concatenate([residual_all[i] for i in sameRVgp.indices])
                if useGPrv[sameRVgp.first_index] in ['ge','ce']:
                    lnprob_allRV = gp.log_likelihood(resid_cctn[srt_rvgp], quiet=True)
                else:
                    if ndim_gp == 1:
                        lnprob_allRV = gp.loglike(resid_cctn[srt_rvgp])
                    else:
                        lnprob_allRV = gp.loglike(y_full)
                
                lnprob       = lnprob + lnprob_allRV

            if inmcmc == 'n':
                resid_cctn = np.concatenate([residual_all[i] for i in sameRVgp.indices])
                for j in sameRVgp.indices:
                    which_GP       = "George" if useGPrv[j]=='ge' else "Celerite" if useGPrv[j]=='ce' else "Spleaf"

                    if ndim_gp == 1:
                        thisrvpargp_arr = input_rvs[RVnames[j]][gpcol] 
                    else:
                        thisrvpargp_arr = np.column_stack( [input_rvs[RVnames[j]][c] for c in gpcol] ) 

                    if useGPrv[j] in ['ge','ce']:
                        thissrt_rvgp   = np.argsort(thisrvpargp_arr) if thisrvpargp_arr.ndim==1 else np.argsort(thisrvpargp_arr[:,0])  #indices to sort the gp axis
                        thisunsrt_rvgp = np.argsort(thissrt_rvgp)  #indices to unsort the gp axis
                        
                        base_gp_all[j] = gp.predict(resid_cctn[srt_rvgp],t=thisrvpargp_arr[thissrt_rvgp], return_cov=False, return_var=False)[thisunsrt_rvgp]
                    else:
                        if ndim_gp == 1:
                            base_gp_all[j]  = gp.conditional(resid_cctn[srt_rvgp], thisrvpargp_arr[thissrt_rvgp], calc_cov=False)[thisunsrt_rvgp]
                        else:
                            gp.kernel["GP"].set_conditional_coef(series_id=0) #predict only RV time series
                            base_gp_all[j]  = gp.conditional(y_full, input_rvs[RVnames[j]]["col0"], calc_cov=False)


                    rvGPsave[RVnames[j]] = GPSaveObj(fname=RVnames[j], gp_pck=useGPrv[j], gp=gp, kernel=kernels,
                                                        ndim_gp=ndim_gp, gp_cols=gpcol, gp_cols_err=gpcol_err,
                                                        gp_pars = {k:v for k,v in zip(same_rv_gppars_names,same_rv_gppars) },
                                                        x=T[0] if (useGPrv[j] == 'sp' and ndim_gp > 1) else thisrvpargp_arr, 
                                                        residuals=y_full if (useGPrv[j] == 'sp' and ndim_gp > 1) else resid_cctn[srt_rvgp])

                    full_mod_all[j] = full_mod_all[j] + base_gp_all[j] #update mod_RV_base with base_gpRV [gp + planet rv + baseline(para+spl+gamma)]
                    mod             = np.concatenate((mod,full_mod_all[j]))
                    emod            = np.concatenate((emod,err_all[j]))

                    base_total_all[j] = base_total_all[j] + base_gp_all[j]  #update base_totalRV with base_gp
                    det_rv_all[j]     = RV_all[j] - base_total_all[j]
                    residual_all[j]   = RV_all[j] - full_mod_all[j]
                    
                    out_data   = np.stack((time_all[j],RV_all[j],err_all[j],full_mod_all[j],base_para_all[j],base_spl_all[j],base_gp_all[j],base_total_all[j],rvmod_all[j],det_rv_all[j],gamma_all[j],residual_all[j]),axis=1)
                    header     = ["time","RV","error","full_mod","base_para","base_spl","base_gp","base_total","Rvmodel","det_RV","gamma","residual"]
                    header_fmt = "{:<16s}\t"*len(header)
                    phases     = np.zeros((len(time_all[j]),npl))

                    for n in range(npl):
                        phases[:,n] = ((time_all[j]-T0in[n])/perin[n]) - np.round( ((time_all[j]-T0in[n])/perin[n]))
                        header_fmt += "{:<16s}\t"
                        header     += [f"phase_{n+1}"] if npl>1 else ["phase"]

                    if make_outfile:   
                        out_data = np.hstack((out_data,phases))
                        outfile  = out_folder+"/out_data/"+splitext(RVnames[j])[0]+'_rvout.dat'
                        if verbose: print(f"    - Writing RV output with GP({which_GP}) to file: {outfile}")
                        np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%-16.6f',delimiter="\t")

        if get_base_model:
            for j in range(nRV):
                basemodel_outputs.rv[RVnames[j]] = base_total_all[j]

    if get_planet_model:
        return model_outputs

    if get_base_model:
        return basemodel_outputs
    
    # ====== return total outputs ======
    if inmcmc == 'y':
        if np.isnan(lnprob) == True:
            lnprob = -np.inf
        return lnprob
    else:   
        #calculate transit duration for each planet
        if "rho_star" in pnames_all:
            _e,_w = sesinw_secosw_to_ecc_omega(np.array(sesinwin),np.array(secoswin))
            durin = list( rho_to_tdur(rhoin, np.array(bbin), np.array(RpRsin), np.array(perin),_e,np.degrees(_w)) ) if nphot>0 else 0
                            # e=np.array(sesinwin)**2+np.array(secoswin)**2, w=np.degrees(np.arctan2(sesinwin,secoswin)) ))

        if isinstance(durin,(int,float)): durin = [durin]    
        if get_GPobj:
            GPobj = SimpleNamespace(lc=lcGPsave,rv=rvGPsave)
            return (mod, emod, T0in, perin, durin,GPobj)
        else:
            return (mod, emod, T0in, perin, durin)

