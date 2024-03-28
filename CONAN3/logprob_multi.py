import numpy as np
import time
from .models import *
from types import SimpleNamespace
from .utils import rho_to_tdur, gp_params_convert
import matplotlib
from ._classes import __default_backend__


def logprob_multi(p, *args,t=None,make_outfile=False,verbose=False,debug=False,get_model=False,out_folder=""):
    """
    calculate log probability and create output file of full model calculated using posterior parameters

    Parameters
    ----------
    p : array
        model parameters

    *args : list 
        contains various config parameters

    make_outfile : bool, optional
        whether to make the output  model file "*out.dat". by default False

    verbose : bool, optional
        whether to print out information during function call, by default False

    debug : bool, optional
        see debug statements, by default False

    get_model: bool, optional
        flag to output dictionary of model results (phot and RV) for specific input parameters.
    
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
    
    # distribute out all the input arguments
    t_arr       = args[0]
    f_arr       = args[1]
    col3_arr    = args[2]
    col4_arr    = args[3]
    col6_arr    = args[4]
    col5_arr    = args[5]
    col7_arr    = args[6]
    bis_arr     = args[7]
    contr_arr   = args[8]
    nphot       = args[9]
    nRV         = args[10]
    indlist     = args[11]
    filters     = args[12]
    nfilt       = args[13]
    filnames    = args[14]
    nddf        = args[15]
    nocc        = args[16]
    nttv        = args[17]
    col8_arr    = args[18]
    grprs       = args[19]
    ttv_conf    = args[20] if not isinstance(args[20],float) else []   #backwards compatibility
    grnames     = args[21]
    groups      = args[22]
    ngroup      = args[23]
    ewarr       = args[24]
    inmcmc      = args[25]
    paraCNM     = args[26]
    baseLSQ     = args[27]
    bvars       = args[28]
    bvarsRV     = args[29]
    cont        = args[30]
    names       = args[31]
    RVnames     = args[32]
    e_arr       = args[33]
    divwhite    = args[34]
    dwCNMarr    = args[35]
    dwCNMind    = args[36]
    params      = args[37]  # all the input params no GP. the function input, p, are the ones that jumping in the mcmc
    useGPphot   = args[38]
    useGPrv     = args[39]
    GPobjects   = args[40]
    GPparams    = args[41]
    GPindex     = args[42]
    pindices    = args[43]  # pindices - list of list -->list of indices of the p array jumping in each LC and then each RV
    jumping     = args[44]  # the indices of the params array that are jumping
    pnames      = args[45]  # the names of the params array that are jumping
    prior_distr = args[46]  # the indices of the params array that are jumping for each LC
    pnames_all  = args[47]  # the names of all the parameters
    priorwid    = args[48]  # the priors, as many as jumping elements (len(p) 
    lim_low     = args[49]  # the lower limits, as many as jumping elements (len(p) 
    lim_up      = args[50]  # the upper limits, as many as jumping elements (len(p) 
    pargps      = args[51]
    jumping_noGP= args[52]
    gpkerns     = args[53]
    jit_apply   = args[54]
    jumping_GP  = args[55]
    GPstepsizes = args[56]
    sameLCgp    = args[57]
    npl         = args[58]
    useSpline_lc= args[59]
    useSpline_rv= args[60]
    s_samp      = args[61]
    rvGPobjects = args[62]
    rvGPparams  = args[63]
    rvGPindex   = args[64]
    input_lcs   = args[65]   #used in result object to access the input data
    input_rvs   = args[66]   #used to result object to access the input data
    RVunit      = args[67]
    rv_pargps   = args[68]
    rv_gpkerns  = args[69]
    sameRVgp    = args[70]
    fit_sampler = args[71]
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
    if get_model: 
        model_outputs = SimpleNamespace(lc={},rv={})   

    params_all[jumping] = p   # set the jumping parameters to the values in p which are varied in mcmc 

    # restrict the parameters to those of the light curve
    for j in range(nphot):
        if inmcmc == 'n':
            if verbose: print(f'LC{j+1}', end=" ...")

        t_in      = np.copy(t_arr[indlist[j][0]]) if t is None else t # time values of lightcurve j
        f_in      = np.copy(f_arr[indlist[j][0]]) # flux values of lightcurve j
        e_in      = np.copy(e_arr[indlist[j][0]]) # error values of lightcurve j
        col3_in   = np.copy(col3_arr[indlist[j][0]])    
        col4_in   = np.copy(col4_arr[indlist[j][0]]) # y values of lightcurve j
        col6_in   = np.copy(col6_arr[indlist[j][0]])
        col5_in   = np.copy(col5_arr[indlist[j][0]])    
        col7_in   = np.copy(col7_arr[indlist[j][0]])
        if np.all(col8_arr == 0): #backwards compatibility with old input where col8 was not included
            col8_arr = [] 
            if nphot >0: _ = [col8_arr.append(input_lcs[nm]["col8"]) for nm in names]
            if nRV > 0:  _ = [col8_arr.append(np.zeros_like(input_rvs[nm]["col0"])) for nm in RVnames]
            col8_arr = np.concatenate(col8_arr)
        col8_in   = np.copy(col8_arr[indlist[j][0]])
        bis_in    = np.copy(bis_arr[indlist[j][0]])
        contra_in = np.copy(contr_arr[indlist[j][0]])
        name = names[j]
        if baseLSQ == "y":
            bvar = bvars[j][0]
        else:
            bvar=[]

        pp=p[pindices[j]]  # the elements of the p array jumping in this LC, pp is the array of non-GP jumping parameters for this LC

        # extract the parameters input to the modeling function from the input array
        # specify the LD and ddf correctly
            
        # identify the filter index of this LC
        k = np.where(filnames == filters[j])  # k is the index of the LC in the filnames array
        k = k[0]  
        vcont = cont[k,0]

        occind    = 1+7*npl+nttv+nddf+k           # index in params of the occultation depth value
        Aatm_ind  = 1+7*npl+nttv+nddf+nocc+k           # index in params of the atm value
        phoff_ind = 1+7*npl+nttv+nddf+nocc*2+k           # index in params of the phoff value
        Aev_ind   = 1+7*npl+nttv+nddf+nocc*3+k           # index in params of the Aev value
        Adb_ind   = 1+7*npl+nttv+nddf+nocc*4+k           # index in params of the Adb value

        q1ind  = 1+7*npl+nttv+nddf+nocc*5+2*k    # index in params of the first LD coeff of this filter
        q2ind  = 1+7*npl+nttv+nddf+nocc*5+2*k+1  # index in params of the second LD coeff of this filter
        gg     = int(groups[j]-1)

        # lc jitter 
        LCjitterind = 1+7*npl + nttv+nddf+nocc*5 + nfilt*2 + j

        # get the index of pp that is 
        # adapt the RpRs value used in the LC creation to any ddfs

        ppcount = 0   # the index of the jumping parameter in the pp array
        T0in, RpRsin, bbin, perin, eosin, eocin, Kin = [], [], [], [], [], [], []
        rhoin, durin = None, None
        #if parameter is jumping, get value from pp array, otherwise use the value from the params array
        
        if "rho_star" in pnames_all:    #using rho_star for transit model
            if 0 in jumping[0]:
                rhoin = pp[ppcount]
                ppcount = ppcount+1
            else:
                rhoin = params[0]
        else:  #using duration for transit model
            if 0 in jumping[0]:
                durin = pp[ppcount]
                ppcount = ppcount+1
            else:
                durin = params[0]

        for n in range(npl):
            if (1+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                T0in.append(pp[ppcount])        # the first element of pp is the T0 value
                ppcount = ppcount+1 
            else:
                T0in.append(params[1+7*n])
                
            if (2+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                RpRsin.append(pp[ppcount])
                ppcount = ppcount+1   
            else:
                RpRsin.append(params[2+7*n])

            if (3+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                bbin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                bbin.append(params[3+7*n])
                    
            if (4+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                perin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                perin.append(params[4+7*n])

            if (5+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                eosin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                eosin.append(params[5+7*n])

            if (6+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                eocin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                eocin.append(params[6+7*n])
        
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
            occin = params[occind]

        if (Aatm_ind in jumping[0]):   
            Aatm_in = pp[ppcount]
            ppcount = ppcount+1
        else:
            Aatm_in = params[Aatm_ind]

        if (phoff_ind in jumping[0]):   
            phoff_in = pp[ppcount]
            ppcount = ppcount+1
        else:
            phoff_in = params[phoff_ind]

        if (Aev_ind in jumping[0]):   
            Aev_in = pp[ppcount]
            ppcount = ppcount+1
        else:
            Aev_in = params[Aev_ind]

        if (Adb_ind in jumping[0]):   
            Adb_in = pp[ppcount]
            ppcount = ppcount+1
        else:
            Adb_in = params[Adb_ind]

        ##########
        # if occin < 0.0:   occin = 0.0
        # if Aatm_in < 0.0: Aatm_in = 0.0 
        # if Aev_in < 0.0:  Aev_in = 0.0 
        # if Adb_in < 0.0:  Adb_in = 0.0 
            
        #########
        #now check the correct LD coeffs
        if (q1ind in jumping[0]):    # index of specific LC LD in jumping array -> check in jumping array
            q1in    = pp[ppcount]
            ppcount = ppcount + 1
        else:
            q1in = params[q1ind]

        if (q2ind in jumping[0]):   # index of specific LC LD in jumping array -> check in jumping array
            q2in    = pp[ppcount]
            ppcount = ppcount + 1
        else:
            q2in = params[q2ind]

        if (LCjitterind in jumping[0]):   # index of specific LC LD in jumping array -> check in jumping array
            ppcount = ppcount + 1

        if get_model:
            if nttv>0:
                LCmod,compo = TTV_Model(tarr=t_in, rho_star=rhoin, dur=durin, T0_list=ttv_conf[j].t0_list, RpRs=RpRsin, b=bbin, per=perin, eos=eosin, eoc=eocin,
                                        q1=q1in, q2=q2in,split_conf=ttv_conf[j],planet_only=True,ss=s_samp[j])
            else:
                TM = Transit_Model(rho_star=rhoin, dur=durin, T0=T0in, RpRs=RpRsin, b=bbin, per=perin, eos=eosin, eoc=eocin, ddf=ddf0, 
                                    occ=occin, A_atm=Aatm_in, delta=phoff_in, A_ev=Aev_in, A_db=Adb_in, q1=q1in, q2=q2in,npl=npl)
                LCmod,compo = TM.get_value(t_in, planet_only=True,ss=s_samp[j])

            model_outputs.lc[name] = LCmod, compo
            continue
            

        bfstart = 1 +7*npl + nttv + nddf +nocc*5 +2*nfilt + nphot + nRV*2+ j*22  # index in params of the first baseline param of this light curve
        blind = np.asarray(list(range(bfstart,bfstart+22))) # the indices of the baseline params of this light curve
        basesin = np.zeros(22)
        
        for jj in range(len(blind)):
            basein = blind[jj]
                
            if (basein in jumping[0]):
                basesin[jj] = pp[ppcount]
                ppcount = ppcount + 1
            else:
                basesin[jj]=params[basein]

        #compute transit and baseline model
        argu = [t_in,f_in,col3_in,col4_in,col6_in,col5_in,col7_in,col8_in,contra_in,isddf,0,grprs_here,inmcmc,baseLSQ,basesin,vcont,name,e_in,bvar,useSpline_lc[j]]     

        if nttv>0:
            lc_result = TTV_Model(tarr=t_in, rho_star=rhoin, dur=durin, T0_list=ttv_conf[j].t0_list, RpRs=RpRsin, b=bbin, per=perin, eos=eosin, eoc=eocin,
                                    q1=q1in, q2=q2in,split_conf=ttv_conf[j],args=argu,ss=s_samp[j])
        else:
            TM = Transit_Model(rho_star=rhoin, dur=durin, T0=T0in, RpRs=RpRsin, b=bbin, per=perin, eos=eosin, eoc=eocin, ddf=ddf0, 
                                        occ=occin, A_atm=Aatm_in, delta=phoff_in, A_ev=Aev_in, A_db=Adb_in,q1=q1in, q2=q2in,npl=npl)
            lc_result=TM.get_value(t_in, argu,ss=s_samp[j])   
        
        trans_base  = lc_result.full_LCmod   #transit*baseline
        mt0         = lc_result.planet_LC    #transit only
        bfunc       = lc_result.LC_bl        #baseline*spline
        det_LC      = f_in / bfunc           #detrended data     


        LCjit = np.exp(params_all[LCjitterind])

        err_in  = (e_in**2 + LCjit**2)**0.5
        
        if useGPphot[j]=='n':
            if inmcmc == 'y':
                lnprob_thislc = -1./2. * np.sum( (trans_base-f_in)**2/err_in**2 + np.log(2*np.pi*err_in**2))
                lnprob = lnprob + lnprob_thislc

            if inmcmc == 'n':
                mod = np.concatenate((mod,trans_base))
                emod = np.concatenate((emod,np.zeros(len(trans_base)))) 

                # write the lightcurve and the model to file or return output if we're not inside the MCMC
                out_data   = np.stack((t_in,f_in,err_in,trans_base,bfunc,mt0,det_LC,lc_result.spline),axis=1)
                header     = ["time","flux","error","full_mod","base","transit","det_flux","spl_fit"]
                header_fmt = "{:<16s}\t"*len(header)
                phases     = np.zeros((len(t_in),npl))

                for n in range(npl):
                    phases[:,n] = np.modf(np.modf( (t_in-T0in[n])/perin[n])[0]+1)[0]
                    if mt0[np.argmin(phases[:,n])] < 1:
                        phases[:,n][phases[:,n]>0.5] = phases[:,n][phases[:,n]>0.5]-1
                    header_fmt += "{:<16s}\t"
                    header     += [f"phase_{n+1}"] if npl>1 else ["phase"]

                out_data = np.hstack((out_data,phases))
                if make_outfile:
                    outfile=out_folder+"/"+name[:-4]+'_lcout.dat' 
                    if verbose: print(f"Writing LC output to file: {outfile}")
                    np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%-16.6f')
        
        elif useGPphot[j] in ['y','ce']: 
            gppars   = params_all[len(params):len(params)+len(GPparams)]   # the GP parameters for all lcs
            pargp    = pargps[j]
            srt_gp   = np.argsort(pargp) if pargp.ndim==1 else np.argsort(pargp[:,0])  #indices to sort the gp axis
            unsrt_gp = np.argsort(srt_gp)  #indices to unsort the gp axis

            #indices of the gp params for this rv file
            if not sameLCgp.flag:
                gp            = GPobjects[j]      #gp for this rv
                thislc_gpind  = GPindex == j      #individual gp params 
            else:
                gp            = GPobjects[sameLCgp.first_index]
                thislc_gpind  = GPindex == sameLCgp.first_index          
            
            thislc_gppars = gppars[thislc_gpind] #the gp params for this lc

            #need conversion of gp pars before setting them
            gp_conv       = gp_params_convert()
            kernels       = gpkerns[j] if useGPphot[j] == "ce" else ["g_"+ gpk for gpk in gpkerns[j]]# ["any_george"]*len(gpkerns[j])
            thislc_gppars = gp_conv.get_values(kernels=kernels, data="lc",pars=thislc_gppars)
            gp.set_parameter_vector(thislc_gppars)
            #recompute gp with jitter included 
            gp.compute(pargp[srt_gp], yerr = err_in[srt_gp])


            if inmcmc == 'y':
                lnprob_thislc = gp.log_likelihood((f_in-trans_base)[srt_gp], quiet=True)
                lnprob = lnprob + lnprob_thislc
            
            # if not in MCMC, get a prediction and append it to the output array
            if inmcmc == 'n':
                which_GP = "George" if useGPphot[j]=='y' else "Celerite"

                bfunc_gp = gp.predict((f_in-trans_base)[srt_gp], t=pargp[srt_gp], return_cov=False, return_var=False)[unsrt_gp] #gp_fit to residual
                pred     = bfunc_gp+trans_base    #gp + transit*baseline(w/wo spl)
                
                mod  = np.concatenate((mod,pred))        #append the model to the output array
                emod = np.concatenate((emod,np.zeros(len(pred)))) #append the model error to the output array

                bfunc_full = bfunc + bfunc_gp
                det_LC     = f_in/bfunc - bfunc_gp    #detrended_data
                
                # write the lightcurve and the model to file or return output if we're not inside the MCMC
                out_data     = np.stack((t_in,f_in,err_in,pred,bfunc_full,mt0,det_LC,lc_result.spline),axis=1)
                header       = ["time","flux","error","full_mod","gp+base","transit","det_flux","spl_fit"]
                header_fmt   = "{:<16s}\t"*len(header)
                phases       = np.zeros((len(t_in),npl))

                for n in range(npl):
                    phases[:,n] = np.modf(np.modf( (t_in-T0in[n])/perin[n])[0]+1)[0]
                    if mt0[np.argmin(phases[:,n])] < 1: phases[:,n][phases[:,n]>0.5] = phases[:,n][phases[:,n]>0.5]-1
                    header_fmt += "{:<16s}\t"
                    header     += [f"phase_{n+1}"] if npl>1 else ["phase"]

                out_data = np.hstack((out_data,phases))
                if make_outfile:
                    outfile=out_folder+"/"+name[:-4]+'_lcout.dat'
                    if verbose: print(f"Writing LC output with GP({which_GP}) to file: {outfile}")
                    np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%-16.6f')


    # now do the RVs and add their probabilities to the model
    for j in range(nRV):
        if verbose: print(f'RV{j+1}', end= " ...")
        t_in      = np.copy(t_arr[indlist[j+nphot][0]]) if t is None else t # time values of lightcurve j
        f_in      = np.copy(f_arr[indlist[j+nphot][0]]) # flux values of lightcurve j
        e_in      = np.copy(e_arr[indlist[j+nphot][0]]) # error values of lightcurve j
        col3_in   = np.copy(col3_arr[indlist[j+nphot][0]])    
        col4_in   = np.copy(col4_arr[indlist[j+nphot][0]]) # y values of lightcurve j
        col6_in   = np.copy(col6_arr[indlist[j+nphot][0]])
        col5_in   = np.copy(col5_arr[indlist[j+nphot][0]])    
        col7_in   = np.copy(col7_arr[indlist[j+nphot][0]])
        bis_in    = np.copy(bis_arr[indlist[j+nphot][0]])
        contra_in = np.copy(contr_arr[indlist[j+nphot][0]])
        name      = RVnames[j]

        # get the current parameters from the pp array.
        pp      = p[pindices[j+nphot]]  # the elements of the p (noGP) array jumping in this RV curve
        ppnames = pnames[pindices[j+nphot]]

        ppcount = 0 
        T0in, RpRsin, bbin, perin, eosin, eocin, Kin = [], [], [], [], [], [],[]
        rhoin, durin = None, None

        if "rho_star" in pnames_all:    #using rho_star  for transit model
            if 0 in jumping[0]:
                rhoin = pp[ppcount]
                ppcount = ppcount+1
            else:
                rhoin = params[0]
        else:  #using duration for transit model
            if 0 in jumping[0]:
                durin = pp[ppcount]
                ppcount = ppcount+1
            else:
                durin = params[0]

        for n in range(npl):
            if (1+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                T0in.append(pp[ppcount])        # the first element of pp is the T0 value
                ppcount = ppcount+1 
            else:
                T0in.append(params[1+7*n])
                
            if (2+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                RpRsin.append(pp[ppcount])
                ppcount = ppcount+1   
            else:
                RpRsin.append(params[2+7*n])

            if (3+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                bbin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                bbin.append(params[3+7*n])
                    
            if (4+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                perin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                perin.append(params[4+7*n])

            if (5+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                eosin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                eosin.append(params[5+7*n])

            if (6+7*n in jumping[0]):   # same for all LCs -> check in jumping array
                eocin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                eocin.append(params[6+7*n])

            if (7+7*n in jumping[0]):   # #TODO check if this can be in lc jumping part. same for all data -> check in jumping array
                Kin.append(pp[ppcount])
                ppcount = ppcount+1
            else:
                Kin.append(params[7+7*n])
        
        # jupind = jumping_noGP[0]
        
        # nGPjump = len(p) - len(jupind)
        # paraminRV[jumping] = p #p[0:-nGPjump] if nGPjump > 0 else p
        gammaind = 1+7*npl + nttv+nddf + nocc*5 + nfilt*2 + nphot + j*2   #pass the right gamma index for each file (Akin)
        Gamma_in = params_all[gammaind]
        RVargs   = [params_all,f_in,e_in,bis_in,col6_in,contra_in,nfilt,baseLSQ,inmcmc,
                    nddf,nocc,nRV,nphot,j,RVnames,bvarsRV,gammaind,nttv]
        
        if get_model:   #skip other calculations and return the RV model for this dataset
            RVmodel,compo = RadialVelocity_Model(t_in,T0in,perin,Kin,eosin,eocin,Gamma_in,*RVargs,
                                        useSpline_rv[j],npl,planet_only=True)
            model_outputs.rv[name] = RVmodel,compo
            continue

        rv_result = RadialVelocity_Model(t_in,T0in,perin,Kin,eosin,eocin,Gamma_in,*RVargs,
                                useSpline_rv[j],npl,planet_only=False) #planet rv + baseline
        
        mod_RVbl  = rv_result.full_RVmod            #planet rv + baseline
        mod_RV    = rv_result.planet_RV             #planet rv
        bfuncRV   = rv_result.RV_bl + Gamma_in      #baseline*spline
        det_RV    = f_in - bfuncRV                  #detrended data
        
        # rv jitter 
        if (jit_apply == 'y'):
            jitterind = 1+7*npl + nttv+ nddf+nocc*5 + nfilt*2 + nphot + j*2 + 1
            jit = params_all[jitterind]
        else:
            jit = 0.
            
        err_in =  (e_in**2 + jit**2)**0.5
        
        if useGPrv[j] == "n":
            
            if inmcmc == 'y':
                lnprob_thisRV = -1./2. * np.sum( (mod_RVbl-f_in)**2/err_in**2 + np.log(2. * np.pi * err_in**2) )
                lnprob = lnprob + lnprob_thisRV

            if inmcmc == 'n':
                # write the RVcurve and the model to file if we're not inside the MCMC
                mod = np.concatenate((mod,mod_RVbl))
                emod = np.concatenate((emod,e_in-e_in))

                out_data   = np.stack((t_in,f_in,err_in,mod_RVbl,bfuncRV,mod_RV,det_RV,rv_result.spline),axis=1)
                header     = ["time","RV","error","full_mod","base","Rvmodel","det_RV","spl_fit"]
                header_fmt = "{:<16s}\t"*len(header)
                phases     = np.zeros((len(t_in),npl))

                for n in range(npl):
                    phases[:,n] = ((t_in-T0in[n])/perin[n]) - np.round( ((t_in-T0in[n])/perin[n]))
                    header_fmt += "{:<16s}\t"
                    header     += [f"phase_{n+1}"] if npl>1 else ["phase"]
                if make_outfile:   
                    outfile  = out_folder+"/"+RVnames[j][:-4]+'_rvout.dat'
                    out_data = np.hstack((out_data,phases))
                    if verbose: print(f"Writing RV output to file: {outfile}")
                    np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%-16.6f')
                    # pd.DataFrame(out_data,columns=header).to_csv(outfile,sep="\t",index=False, float_format='%-16.6f')


        elif useGPrv[j] in ['y','ce']:
            rv_gppars = params_all[len(params)+len(GPparams):]   #all rv gp parameters 
            rvpargp       = rv_pargps[j]            #independent axis columns
            
            #indices of the gp params for this rv file
            if not sameRVgp.flag:
                gp            = rvGPobjects[j]      #gp for this rv
                thisrv_gpind  = rvGPindex == j      #individual gp params 
            else:
                gp            = rvGPobjects[sameRVgp.first_index]
                thisrv_gpind  = rvGPindex == sameRVgp.first_index          
            
            thisrv_gppars = rv_gppars[thisrv_gpind] #the gp params for this RV

            #need conversion of gp pars before setting them
            gp_conv       = gp_params_convert()
            kernels       = rv_gpkerns[j] if useGPrv[j] == "ce" else ["g_"+ gpk for gpk in rv_gpkerns[j]]#["any_george"]*len(rv_gpkerns[j])
            thisrv_gppars = gp_conv.get_values(kernels=kernels, data="rv",pars=thisrv_gppars)
            gp.set_parameter_vector(thisrv_gppars)
            #recompute gp with jitter included 
            gp.compute(rvpargp, yerr = err_in)

            if inmcmc == 'y':
                lnprob_thisRV = gp.log_likelihood(f_in-mod_RVbl, quiet=True)
                lnprob = lnprob + lnprob_thisRV

            # if not in MCMC, get a prediction and append it to the output array
            if inmcmc == 'n':
                which_GP = "George" if useGPrv[j]=='y' else "Celerite"

                bfunc_gp= gp.predict(f_in-mod_RVbl, t=rvpargp, return_cov=False, return_var=False)
                pred = bfunc_gp+mod_RVbl

                mod = np.concatenate((mod,pred))
                emod = np.concatenate((emod,e_in-e_in))

                bfunc_full = bfunc_gp+bfuncRV
                det_RV     = f_in - bfunc_full

                out_data   = np.stack((t_in,f_in,err_in,pred,bfunc_full,mod_RV,det_RV,rv_result.spline),axis=1)
                header     = ["time","RV","error","full_mod","gp+base","Rvmodel","det_RV","spl_fit"]
                header_fmt = "{:<16s}\t"*len(header)
                phases     = np.zeros((len(t_in),npl))

                for n in range(npl):
                    phases[:,n] = ((t_in-T0in[n])/perin[n]) - np.round( ((t_in-T0in[n])/perin[n]))
                    header_fmt += "{:<16s}\t"
                    header     += [f"phase_{n+1}"] if npl>1 else ["phase"]
                
                if make_outfile:   
                    out_data = np.hstack((out_data,phases))
                    outfile  = out_folder+"/"+RVnames[j][:-4]+'_rvout.dat'
                    if verbose: print(f"Writing RV output with GP({which_GP}) to file: {outfile}")
                    np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%-16.6f')


    if get_model:
        return model_outputs
    
    # ====== return total outputs ======
    if inmcmc == 'y':
        if np.isnan(lnprob) == True:
            lnprob = -np.inf
        return lnprob
    else:   
        #calculate transit duration for each planet
        if "rho_star" in pnames_all:
            durin = list( rho_to_tdur(rhoin, np.array(bbin), np.array(RpRsin), np.array(perin),
                            e=np.array(eosin)**2+np.array(eocin)**2, w=np.degrees(np.arctan2(eosin,eocin)) ))

        if isinstance(durin,(int,float)): durin = [durin]    
        return (mod, emod, T0in, perin, durin if nphot>0 else 0)

