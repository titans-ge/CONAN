
import george
import sys
import numpy as np
import time

# from george.modeling import Model
# from george import kernels
# from .gpnew import *
from .model_GP_v3 import *
from celerite.modeling import Model 
from celerite import terms
import celerite
from CONAN3.celeritenew import *
from types import SimpleNamespace
from .utils import rho_to_tdur

from .RVmodel_v3 import *


def logprob_multi(p, *args,t=None,make_out_file=False,verbose=False,debug=False,get_model=False,out_folder=""):
    """
    calculate log probability and create output file of full model calculated using posterior parameters

    Parameters
    ----------
    p : array
        model parameters

    *args : list 
        contains various config parameters

    make_out_file : bool, optional
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
    
    # print(f"t={t}")
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
    rprs0       = args[17]
    erprs0      = args[18]
    grprs       = args[19]
    egrprs      = args[20]
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
    params      = args[37]  #BUG: actually the params are input as p! p jumps in the mcmc
    useGPphot   = args[38]
    useGPrv     = args[39]
    GPobjects   = args[40]
    GPparams    = args[41]
    GPindex     = args[42]
    pindices    = args[43]  # pindices - list of list -->list of indices of the p array jumping in each LC and then each RV
    jumping     = args[44]  # the indices of the params array that are jumping
    pnames      = args[45]
    LCjump      = args[46]  # the indices of the params array that are jumping for each LC
    prior       = args[47]  # the priors, as many as jumping elements (len(p) 
    priorwid    = args[48]  # the priors, as many as jumping elements (len(p) 
    lim_low     = args[49]  # the lower limits, as many as jumping elements (len(p) 
    lim_up      = args[50]  # the upper limits, as many as jumping elements (len(p) 
    pargps      = args[51]
    jumping_noGP= args[52]
    GPphotWN    = args[53]
    jit_apply   = args[54]
    jumping_GP  = args[55]
    GPstepsizes = args[56]
    GPcombined  = args[57]
    npl         = args[58]
    useSpline_lc= args[59]
    useSpline_rv= args[60]
    
    lnprob      = 0.

    #if in mcmc check that prior is finite
    if inmcmc == 'y':
        lnprior = 0.
        for jj in range(len(p)):
            if (priorwid[jj]>0.):
                lpri = norm_prior(p[jj],prior[jj],priorwid[jj])
                lnprior += lpri
                
            llim = limits(p[jj],lim_low[jj], lim_up[jj])
            lnprior += llim
        
        if np.isfinite(lnprior):
            lnprob += lnprior      # add to logprob if finite. later steps calculate log_likelihood to be added 
        else:
            return -np.inf 

    
    lc0_combinedGPs = np.where(GPcombined == 1.0)
    
    mod, emod = [], [] # output arrays in case we're not in the mcmc
    if get_model: 
        model_outputs = SimpleNamespace(lc={},rv={})   

    # restrict the parameters to those of the light curve
    for j in range(nphot):
        if inmcmc == 'n':
            if verbose: print('\nLightcurve ',j+1)

        t_in      = np.copy(t_arr[indlist[j][0]]) if t is None else t # time values of lightcurve j
        f_in      = np.copy(f_arr[indlist[j][0]]) # flux values of lightcurve j
        e_in      = np.copy(e_arr[indlist[j][0]]) # error values of lightcurve j
        col3_in   = np.copy(col3_arr[indlist[j][0]])    
        col4_in   = np.copy(col4_arr[indlist[j][0]]) # y values of lightcurve j
        col6_in   = np.copy(col6_arr[indlist[j][0]])
        col5_in   = np.copy(col5_arr[indlist[j][0]])    
        col7_in   = np.copy(col7_arr[indlist[j][0]])
        bis_in    = np.copy(bis_arr[indlist[j][0]])
        contra_in = np.copy(contr_arr[indlist[j][0]])
        name = names[j]
        if baseLSQ == "y":
            bvar = bvars[j][0]
        else:
            bvar=[]

        pp=p[pindices[j]]  # the elements of the p array jumping in this LC, pp is the array of jumping parameters for this LC
        
        # extract the parameters input to the modeling function from the input array
        # specify the LD and ddf correctly
            
        # identify the filter index of this LC
        k = np.where(filnames == filters[j])  # k is the index of the LC in the filnames array
        k = k[0]  
        vcont = cont[k,0]

        occind = 1+7*npl+nddf+k           # index in params of the occultation depth value
        u1ind  = 1+7*npl+nddf+nocc+4*k    # index in params of the first LD coeff of this filter
        u2ind  = 1+7*npl+nddf+nocc+4*k+1  # index in params of the second LD coeff of this filter
        gg     = int(groups[j]-1)

        # get the index of pp that is 
        # adapt the RpRs value used in the LC creation to any ddfs

        ppcount = 0   # the index of the jumping parameter in the pp array
        T0in, RpRsin, bbin, perin, eosin, eocin, Kin = [], [], [], [], [], [], []

        if 0 in jumping[0]:
            rhoin = pp[ppcount]
            ppcount = ppcount+1
        else:
            rhoin = params[0]

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
        
        #calculate transit duration for each planet
        durin = list( rho_to_tdur(rhoin, np.array(bbin), np.array(RpRsin), np.array(perin),
                     e=np.array(eosin)**2+np.array(eocin)**2, w=np.degrees(np.arctan2(eosin,eocin)) ))

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
            

        ##########
        if occin < 0.0:
            occin = 0.0

        #########


        #now check the correct LD coeffs
        if (u1ind in jumping[0]):    # index of specific LC LD in jumping array -> check in jumping array
            c1in    = pp[ppcount]
            ppcount = ppcount + 1
        else:
            c1in = params[u1ind]

        if (u2ind in jumping[0]):   # index of specific LC LD in jumping array -> check in jumping array
            c2in    = pp[ppcount]
            ppcount = ppcount + 1
        else:
            c2in = params[u2ind]

        if get_model:
            TM = Transit_Model(rho_star=rhoin, T0=T0in, RpRs=RpRsin, b=bbin, per=perin, eos=eosin, eoc=eocin, ddf=ddf0, occ=occin, c1=c1in, c2=c2in,npl=npl)
            LCmod,compo = TM.get_value(t_in, transit_only=True)
            model_outputs.lc[name] = LCmod, compo
            continue
            

        bfstart = 1+7*npl+nddf+nocc+4*nfilt+nRV*2+ j*20  # index in params of the first baseline param of this light curve
        blind = np.asarray(list(range(bfstart,bfstart+20))) # the indices of the baseline params of this light curve
        basesin = np.zeros(20)
        
        for jj in range(len(blind)):
            basein = blind[jj]
                
            if (basein in jumping[0]):
                basesin[jj] = pp[ppcount]
                ppcount = ppcount + 1
            else:
                basesin[jj]=params[basein]

        #A test to find out what kind of model this is
        if (useGPphot[j]=='y' or useGPphot[j]=='ce'):   
            pargp = pargps[j]
            # in this case, this is just the transit+GP model
            # Parest = dict(T0=T0in,RpRs=RpRsin, b=bbin, dur=durin, per=perin, eos=eosin, eoc=eocin, ddf=ddf0, occ=occin, c1=c1in, c2=c2in,npl=npl)
            
            # kwargs = dict(**Parest)
            

            # parameters to go into the GP: transit followed by the GP parameters and the WN            
            # which GP parameters are those of this light curve? 
            GPthisLC = np.where(GPindex == j)[0]   # the indices of the GP parameters that are jumping in this LC
            GPuse = [] # the GP parameters that are used in this LC

            ppcount_lc0 = ppcount 

            l = 0
            
            for jj in range(len(GPthisLC)):
                
                if (GPstepsizes[GPthisLC[jj]]!=0):   # if this GP parameter is jumping, then set it to the next one in pp
                
                    GPuse = np.concatenate((GPuse,[pp[ppcount]]),axis=0)
                    ppcount = ppcount + 1

                elif GPcombined[GPthisLC[jj]] == 1.0:
                    if GPphotWN[j] == 'n' and jj == 0:
                        GPuse = np.concatenate((GPuse,[GPparams[GPthisLC[jj]]]),axis=0)
                    else:
                        GPuse = np.concatenate((GPuse,[p[pindices[0]][ppcount_lc0+lc0_combinedGPs[0][l]]]),axis=0)
                        l = l+1

                else:
                    GPuse = np.concatenate((GPuse,[GPparams[GPthisLC[jj]]]),axis=0)   # otherwise, set it to the value in GPparams

           # if (GPphotWN[j] == 'n'):
           #     GPuse = np.concatenate(([-50.],GPuse),axis=0)   # a crude solution: set WN to very very low
            # here: define the correct para array
            # para=[T0in,RpRsin, bbin, durin, perin, eosin, eocin, ddf0, occin, c1in, c2in,npl]
            # para=np.concatenate((para,(GPuse)),axis=0) if useGPphot[j]=='y' else np.concatenate(((GPuse),para),axis=0)
                        
            # here we need to call the correct GP objects
            gp = GPobjects[j]

            # gp.set_parameter_vector(para, include_frozen=True)
            gp.set_parameter_vector(GPuse)

            mean_model = Transit_Model(rho_star=rhoin, T0=T0in, RpRs=RpRsin, b=bbin, per=perin, eos=eosin, eoc=eocin, ddf=ddf0, occ=occin, c1=c1in, c2=c2in,npl=npl)
            argu = [t_in,f_in,col3_in,col4_in,col6_in,col5_in,col7_in,bis_in,contra_in,isddf,rprs0,grprs_here,inmcmc,baseLSQ,basesin,vcont,name,e_in,bvar,useSpline_lc[j]]
            trans_base = mean_model.get_value(t_in, argu)  #transit* baseline(w/wo spl)

            if inmcmc == 'y':
            # trans_base = trans_base if np.all(np.isfinite(trans_base)) else np.ones_like(t_in)
                lnprob_thislc = gp.log_likelihood(f_in/trans_base, quiet=True)
                lnprob = lnprob + lnprob_thislc
            
            # if not in MCMC, get a prediction and append it to the output array
            if inmcmc == 'n':
                if verbose: 
                    print("Using George GP") if useGPphot[j]=='y' else print("Using Celerite GP")
                    print('GP values used:',GPuse)
                if debug:
                    print("\nDEBUG: In logprob_multi_sinv4")
                    print(f"GP terms: {gp.get_parameter_names()}")
                    print(f"GP vector: {gp.get_parameter_vector()}")
                

                bfunc_gp= gp.predict(f_in/trans_base, t=pargp, return_cov=False, return_var=False) #gp_fit to residual
                pred = bfunc_gp*trans_base    #gp*transit*baseline(w/wo spl)
                
                mod = np.concatenate((mod,pred))        #append the model to the output array
                emod = np.concatenate((emod,np.zeros(len(pred)))) #append the model error to the output array
                
                # return the transit model
                # mo = gp.mean.get_value(t_in, args=argu)  #transit*baseline
                #print mo

                # get the transit-only model with no parametric baselines
                basesin_non = np.zeros(20)
                basesin_non[0] = 1.
                argu2 = [t_in,f_in,col3_in,col4_in,col6_in,col5_in,col7_in,bis_in,contra_in,isddf,rprs0,grprs_here,inmcmc,'n',basesin_non,vcont,name,e_in,bvar,useSpline_lc[j]]
                mt0,_ = mean_model.get_value(t_in, argu2,transit_only=True)  #transit only

                #get parametric baseline and spline
                ts=t_in-t_in[0]
                bfunc_para,spl_comp,spl_x = basefunc_noCNM(basesin, ts, col5_in, col3_in, col4_in, col6_in, col7_in,f_in/mt0,useSpline_lc[j])#mo/mt0#
                
 
                bfunc_full = bfunc_para * bfunc_gp
                model_transit = mt0 #mo/bfunc_para
                fco_full = f_in/bfunc_full    #detrended_data
                
                # write the lightcurve and the model to file or return output if we're not inside the MCMC
                out_data     = np.stack((t_in,f_in,e_in,pred,bfunc_full,model_transit,fco_full,spl_x,spl_comp),axis=1)
                header       = ["time","flux","error","full_mod","gp*base","transit","det_flux","spl_x","spl_fit"]
                header_fmt   = "{:14s}\t"*len(header)
                phases       = np.zeros((len(t_in),npl))

                for n in range(npl):
                    phases[:,n] = np.modf(np.modf( (t_in-T0in[n])/perin[n])[0]+1)[0]
                    if model_transit[np.argmin(phases[:,n])] < 1: phases[:,n][phases[:,n]>0.5] = phases[:,n][phases[:,n]>0.5]-1
                    header_fmt += "{:14s}\t"
                    header     += [f"phase_{n+1}"] if npl>1 else ["phase"]

                out_data = np.hstack((out_data,phases))
                if make_out_file:
                    outfile=out_folder+"/"+name[:-4]+'_lcout.dat'
                    if verbose: print(f"Writing output with gp to file: {outfile}")
                    np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%14.8f')


        else:

            mean_model = Transit_Model(rho_star=rhoin, T0=T0in, RpRs=RpRsin, b=bbin, per=perin, eos=eosin, eoc=eocin, ddf=ddf0, occ=occin, c1=c1in, c2=c2in,npl=npl)
            argu = [t_in,f_in,col3_in,col4_in,col6_in,col5_in,col7_in,bis_in,contra_in,isddf,rprs0,grprs_here,inmcmc,baseLSQ,basesin,vcont,name,e_in,bvar,useSpline_lc[j]]     
            mt=mean_model.get_value(t_in, argu)   #transit*base

            if inmcmc == 'y':
                lnprob_thislc = -1./2. * np.sum( (mt-f_in)**2/e_in**2 + np.log(e_in**2))
                lnprob = lnprob + lnprob_thislc
                # chisq = np.sum((mt-f_in)**2/e_in**2)

            if inmcmc == 'n':
                mod = np.concatenate((mod,mt))
                emod = np.concatenate((emod,np.zeros(len(mt)))) 

                # get the transit-only model with no parametric baselines
                basesin_non = np.zeros(20)
                basesin_non[0] = 1.
                argu2 = [t_in,f_in,col3_in,col4_in,col6_in,col5_in,col7_in,bis_in,contra_in,isddf,rprs0,grprs_here,inmcmc,'n',basesin_non,vcont,name,e_in,bvar,useSpline_lc[j]]
                mt0,_=mean_model.get_value(t_in, argu2,transit_only=True)  #transit only/ baseline set to 1
                
                # #### Monika modificatons for outputs without GPs #####
                #
                # write out an output file in the same format as the GP output files. 
                #   But set the GP prediciton ("pred") to the full model
                ts=t_in-t_in[0]
                if (baseLSQ == 'y'):
                    # print("Running LSQ on baseline model")
                    mres=f_in/mt0
                    #bvar contains the indices of the non-fixed baseline variables
                    coeffstart = np.copy(basesin[bvar])   
                    icoeff,dump = scipy.optimize.leastsq(para_minfunc, coeffstart, args=(bvar, mt0, f_in, ts, col5_in, col3_in, col4_in, col6_in, col7_in))
                    coeff = np.copy(basesin)
                    coeff[bvar] = np.copy(icoeff)
                    bfunc_para,spl_comp,spl_x = basefunc_noCNM(coeff, ts, col5_in, col3_in, col4_in, col6_in, col7_in,f_in/mt0,useSpline_lc[j])
                else:
                    # print("Taking default straight-line baseline")
                    bfunc_para,spl_comp,spl_x = basefunc_noCNM(basesin, ts, col5_in, col3_in, col4_in, col6_in, col7_in,f_in/mt0,useSpline_lc[j])

                pred=mt0*bfunc_para
                fco_full = f_in/bfunc_para

                # write the lightcurve and the model to file or return output if we're not inside the MCMC
                out_data   = np.stack((t_in,f_in,e_in,pred,bfunc_para,mt0,fco_full,spl_x,spl_comp),axis=1)
                header     = ["time","flux","error","full_mod","base","transit","det_flux","spl_x","spl_fit"]
                header_fmt = "{:14s}\t"*len(header)
                phases   = np.zeros((len(t_in),npl))

                for n in range(npl):
                    phases[:,n] = np.modf(np.modf( (t_in-T0in[n])/perin[n])[0]+1)[0]
                    if mt0[np.argmin(phases[:,n])] < 1:
                        phases[:,n][phases[:,n]>0.5] = phases[:,n][phases[:,n]>0.5]-1
                    header_fmt += "{:14s}\t"
                    header     += [f"phase_{n+1}"] if npl>1 else ["phase"]

                out_data = np.hstack((out_data,phases))
                if make_out_file:
                    outfile=out_folder+"/"+name[:-4]+'_lcout.dat' 
                    if verbose: print(f"Writing output without gp to file: {outfile}")
                    np.savetxt(outfile,out_data,header=header_fmt.format(*header),fmt='%14.8f')

    # now do the RVs and add their proba to the model
    for j in range(nRV):
        if verbose: print('\nRV',j+1, " ...\n")
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
        
        argu = [t_in,f_in,col3_in,col4_in,col6_in,col5_in,col7_in,bis_in,contra_in,isddf,
                rprs0,grprs_here,inmcmc,baseLSQ,bvars,vcont,name,e_in]

            # get the current parameters from the pp array.
        pp=p[pindices[j+nphot]]  # the elements of the p array jumping in this RV curve

        ppcount = 0 
        T0in, RpRsin, bbin, perin, eosin, eocin, Kin = [], [], [], [], [], [],[]
        if 0 in jumping[0]:
            rhoin = pp[ppcount]
            ppcount = ppcount+1
        else:
            rhoin = params[0]

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
        
        paraminRV = params
        jupind = jumping_noGP[0]
        
        nGPjump = len(p) - len(jupind)
        paraminRV[jupind] = p[0:-nGPjump] if nGPjump > 0 else p
        gammaind = 1+7*npl + nddf + nocc+ nfilt*4 + j*2   #pass the right gamma index for each file (Akin)
        Gamma_in = paraminRV[gammaind]
        RVargs   = [paraminRV,f_in,e_in,bis_in,col6_in,contra_in,nfilt,baseLSQ,inmcmc,
                    nddf,nocc,nRV,nphot,j,RVnames,bvarsRV,gammaind]
        
        if get_model:   #skip other calculations and return the RV model for this dataset
            RVmodel,compo = get_RVmod(t_in,T0in,perin,Kin,eosin,eocin,Gamma_in,*RVargs,useSpline_rv[j],
                                      npl,make_out_file=False,get_model=True,out_folder=out_folder)
            model_outputs.rv[name] = RVmodel,compo
            continue
        #RV_Model(T0=T0in, RpRs=RpRsin, b=bbin, dur=durin, per=perin, eos=eosin, eoc=eocin, K=Kin, gamma=gammain)
        #RVmod.get_value(tt,args=argu)
        RVmod = get_RVmod(t_in,T0in,perin,Kin,eosin,eocin,Gamma_in,*RVargs,useSpline_rv[j],npl,
                          make_out_file=make_out_file,get_model=False,out_folder=out_folder)

        if (jit_apply == 'y'):
            jitterind = 1+7*npl + nddf+nocc + nfilt*4 + j*2 + 1
            jit = paraminRV[jitterind]
        else:
            jit = 0.
        
        chisq = np.sum((RVmod-f_in)**2/(e_in**2 + jit**2))
    
        
        mod = np.concatenate((mod,RVmod))
        emod = np.concatenate((emod,e_in-e_in))

        if inmcmc == 'y':
            lnprob_thisRV = -1./2. * np.sum( (RVmod-f_in)**2 / (e_in**2 + jit**2) + np.log(2. * np.pi * e_in**2 + jit**2) )
            lnprob = lnprob + lnprob_thisRV


    if get_model:
        return model_outputs
    
# ====== evaluate limits and priors ======
    # for jj in range(len(p)):
    #     if (priorwid[jj]>0.):
    #         lpri = norm_prior(p[jj],prior[jj],priorwid[jj])
    #         lnprob = lnprob + lpri
    #       #  print p[jj], lim_low[jj], lim_up[jj], prior[jj], priorwid[jj]
            
    #     llim = limits(p[jj],lim_low[jj], lim_up[jj])
    #     lnprob = lnprob + llim

# ====== return outputs ======
    
    # MONIKA: adding a check against NaN log probs
    if inmcmc == 'y':
        if np.isnan(lnprob) == True:
            lnprob = -np.inf
        return lnprob
    else:      
        return (mod, emod, T0in, perin, durin)

def norm_prior(value,center,sigma):
    lpri = np.log(1./(2. * np.pi * sigma**2)) - ((value-center)**2/(2. * sigma**2))
    return lpri

def limits(value,lim_low,lim_up):
    if value < lim_low or value > lim_up:  # gp scale
        return -np.inf  
    return 0.
