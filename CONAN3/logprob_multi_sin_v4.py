
import george
import sys
import numpy as np
import time

from george.modeling import Model
from george import kernels
from .gpnew import *
from .model_GP_v3 import *

from .RVmodel_v3 import *


def logprob_multi(p, *args):
    
    # distribute out all the input arguments
    tarr = args[0]
    farr = args[1]
    xarr = args[2]
    yarr = args[3]
    warr = args[4]
    aarr = args[5]
    sarr = args[6]
    barr = args[7]
    carr = args[8]
    nphot = args[9]
    nRV = args[10]
    indlist = args[11]
    filters = args[12]
    nfilt = args[13]
    filnames = args[14]
    nddf = args[15]
    nocc = args[16]
    rprs0 = args[17]
    erprs0 = args[18]
    grprs = args[19]
    egrprs = args[20]
    grnames = args[21]
    groups = args[22]
    ngroup = args[23]
    ewarr = args[24]
    inmcmc = args[25]
    paraCNM = args[26]
    baseLSQ = args[27]
    bvars = args[28]
    bvarsRV = args[29]
    cont = args[30]
    names = args[31]
    RVnames = args[32]
    earr = args[33]
    divwhite = args[34]
    dwCNMarr = args[35]
    dwCNMind = args[36]
    params = args[37]  #BUG: actually the params are input as p! p jumps in the mcmc
    useGPphot = args[38]
    useGPrv = args[39]
    GPobjects = args[40]
    GPparams = args[41]
    GPindex = args[42]
    pindices = args[43] # pindices are the indices of the p array jumping in each LC
    jumping =  args[44] # the indices of the params array that are jumping
    pnames =  args[45]
    LCjump =  args[46] # the indices of the params array that are jumping for each LC
    prior =   args[47] # the priors, as many as jumping elements (len(p) 
    priorwid = args[48] # the priors, as many as jumping elements (len(p) 
    lim_low =  args[49] # the lower limits, as many as jumping elements (len(p) 
    lim_up =  args[50] # the upper limits, as many as jumping elements (len(p) 
    pargps = args[51]
    jumping_noGP = args[52]
    GPphotWN = args[53]
    jit_apply = args[54]
    jumping_GP = args[55]
    GPstepsizes = args[56]
    GPcombined = args[57]
    lnprob = 0.
    
    lc0_combinedGPs = np.where(GPcombined == 1.0)
    
    mod, emod = [], [] # output arrays in case we're not in the mcmc
    
    # restrict the parameters to those of the light curve
    for j in range(nphot):
        tt = np.copy(tarr[indlist[j][0]]) # time values of lightcurve j
        ft = np.copy(farr[indlist[j][0]]) # flux values of lightcurve j
        ee = np.copy(earr[indlist[j][0]]) # error values of lightcurve j
        xt = np.copy(xarr[indlist[j][0]])    
        yt = np.copy(yarr[indlist[j][0]]) # y values of lightcurve j
        wt = np.copy(warr[indlist[j][0]])
        at = np.copy(aarr[indlist[j][0]])    
        st = np.copy(sarr[indlist[j][0]])
        bt = np.copy(barr[indlist[j][0]])
        ct = np.copy(carr[indlist[j][0]])
        name = names[j]
        if baseLSQ == "y":
            bvar = bvars[j][0]
        else:
            bvar=[]

        pp=p[pindices[j]]  # the elements of the p array jumping in this LC
        
        # extract the parameters input to the modeling function from the input array
        
        #specify the LD and ddf correctly
            
        # identify the filter index of this LC
        k = np.where(filnames == filters[j])  # k is the index of the LC in the filnames array
        k = np.asscalar(k[0])
        vcont = cont[k,0]

        occind = 8+nddf+k   # index in params of the occultation depth value
        u1ind = 8+nddf+nocc+4*k  # index in params of the first LD coeff of this filter
        u2ind = 9+nddf+nocc+4*k  # index in params of the second LD coeff of this filter
        gg=int(groups[j]-1)

        # get the index of pp that is 
        # adapt the RpRs value used in the LC creation to any ddfs

        ppcount = 0 
        if (0 in jumping[0]):  # same for all LCs -> check in jumping array
            T0in = pp[0] 
            ppcount = ppcount+1
        else:
            T0in = params[0]
            
        if (1 in jumping[0]):   # same for all LCs -> check in jumping array
            RpRsin = pp[ppcount]
            ppcount = ppcount+1   
        else:
            RpRsin = params[1]

        if (2 in jumping[0]):   # same for all LCs -> check in jumping array
            bbin = pp[ppcount]
            ppcount = ppcount+1
        else:
            bbin = params[2]
                 
        if (3 in jumping[0]):   # same for all LCs -> check in jumping array
            durin=pp[ppcount]
            ppcount = ppcount+1
        else:
            durin = params[3]
                 
        if (4 in jumping[0]):   # same for all LCs -> check in jumping array
            perin=pp[ppcount]
            ppcount = ppcount+1
        else:
            perin = params[4]

        if (5 in jumping[0]):   # same for all LCs -> check in jumping array
            eosin=pp[ppcount]
            ppcount = ppcount+1
        else:
            eosin = params[5]

        if (6 in jumping[0]):   # same for all LCs -> check in jumping array
            eocin=pp[ppcount]
            ppcount = ppcount+1
        else:
            eocin = params[6]
            
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
            c1in = pp[ppcount]
            ppcount = ppcount + 1
        else:
            c1in = params[u1ind]

        if (u2ind in jumping[0]):   # index of specific LC LD in jumping array -> check in jumping array
            c2in = pp[ppcount]
            ppcount = ppcount + 1
        else:
            c2in = params[u2ind]
            
        bfstart = 8+nddf+nocc+4*nfilt+nRV+ j*20  # index in params of the first baseline param of this light curve
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
        if (useGPphot[j]=='y'):   
            pargp = pargps[j]
            # in this case, this is just the transit+GP model
            Parest = dict(T0=T0in,RpRs=RpRsin, b=bbin, dur=durin, per=perin, eos=eosin, eoc=eocin, ddf=ddf0, occ=occin, c1=c1in, c2=c2in)
            
            kwargs = dict(**Parest)
            
            # and specify the correct arguments
            argu = [tt,ft,xt,yt,wt,at,st,bt,ct,isddf,rprs0,grprs_here,inmcmc,baseLSQ,basesin,vcont,name,ee,bvar]
       
            mean_model = Transit_Model(T0=T0in, RpRs=RpRsin, b=bbin, dur=durin, per=perin, eos=eosin, eoc=eocin, ddf=ddf0, occ=occin, c1=c1in, c2=c2in)
            
            # parameters to go into the GP: transit followed by the GP parameters and the WN
            
            # which GP parameters are those of this light curve? 
            GPthisLC = np.where(GPindex == j)[0]
            GPuse = []

            ppcount_lc0 = ppcount

            l = 0
            
            for jj in range(len(GPthisLC)):
                
                if (GPstepsizes[GPthisLC[jj]]!=0):   # if this GP parameter is jumping, then set it to the next one in pp
                
                    GPuse = np.concatenate((GPuse,[pp[ppcount]]),axis=0)
                    ppcount = ppcount + 1
 
                elif GPcombined[GPthisLC[jj]] == 1.0:
                    GPuse = np.concatenate((GPuse,[p[pindices[0]][ppcount_lc0+lc0_combinedGPs[0][l]]]),axis=0)
                    l = l+1

                else:
                    GPuse = np.concatenate((GPuse,[GPparams[GPthisLC[jj]]]),axis=0)   # otherwise, set it to the value in GPparams

           # if (GPphotWN[j] == 'n'):
           #     GPuse = np.concatenate(([-50.],GPuse),axis=0)   # a crude solution: set WN to very very low
            # here: define the correct para array
            para=[T0in,RpRsin, bbin, durin, perin, eosin, eocin, ddf0, occin, c1in, c2in]
            para=np.concatenate((para,(GPuse)),axis=0)
                        
            # here we need to call the correct GP objects
            gp = GPobjects[j]

            gp.set_parameter_vector(para, include_frozen=True)
            
            # if not in MCMC, get a prediction and append it to the output array
            if inmcmc == 'n':
                print('\nLightcurve number:',j)
                print('GP values used:',GPuse)

                pred, pred_var = gp.predict(ft, pargp, return_var=True, args=argu)
                
                mod = np.concatenate((mod,pred))
                emod = np.concatenate((emod,pred_var))
                    # write the lightcurve and the model to file if we're not inside the MCMC
                
                # return the transit model
                mo = gp.mean.get_value(ft, args=argu)
                #print mo
                bfunc = pred/mo
                fco = ft/bfunc
                
                # compute the parametric part of the baseline model  tt,ft,xt,yt,wt,at,st,bt,ct
                
                delta = np.round(np.divide((tt[0]-params[0]),params[4]))
                T0_lc=params[0]+delta*params[4]
                #ts=tt-T0_lc  
                ts=tt-tt[0]
                
                bfunc_para = basefunc_noCNM(basesin, ts, at, xt, yt, wt, st)
                
                #####ANDREAS: This is redunant... This file is only written when GPs are used but prints values without GPs
                # outfile=name[:-4]+'_out.dat'
                # of=open(outfile,'w')
                # for k in range(len(tt)):
                #     of.write('%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n' % (tt[k], ft[k], ee[k], pred[k],bfunc[k],mo[k],fco[k])) 
                
                # of.close() 
 
                bfunc_gp = np.copy(bfunc)
                bfunc_full = bfunc_para * bfunc_gp
                model_transit = mo/bfunc_para
                fco_full = ft/bfunc_full
 
                outfile=name[:-4]+'_out_full.dat'
                of=open(outfile,'w')
                for k in range(len(tt)):
                    of.write('%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n' % (tt[k], ft[k], ee[k], pred[k],bfunc_full[k],model_transit[k],fco_full[k])) 
                
                of.close() 
 
 
                # ===== create and write out the DW model ===============
                
                ####ANDREAS: There's no use for this, just increases the number of files produced
                # outfile2=name[:-4]+'_dw.dat'                
                # of2=open(outfile2,'w')
                # for k in range(len(tt)):
                #     of2.write('%15.6f %10.6f\n' % (tt[k], ft[k]/mo[k]))
    
                # of2.close()   

            lnprob_thislc = gp.log_likelihood(ft, argu, quiet=True)
#            print(lnprob_thislc)
#            time.sleep(0.1)
            lnprob = lnprob + lnprob_thislc
            
        else:

            argu = [tt,ft,xt,yt,wt,at,st,bt,ct,isddf,rprs0,grprs_here,inmcmc,baseLSQ,basesin,vcont,name,ee,bvar]     


            # #### MONIKA: let's call the transit model from "model_GP_v3.pro" to avoid inconsitencies #####
            #      get_Tramod shoud now be obsolete
            #
            #  mt = get_Tramod(T0in, RpRsin, bbin, durin, perin, eosin, eocin, ddf0, occin, c1in, c2in, argu)

            tramod_call = Transit_Model(T0=T0in, RpRs=RpRsin, b=bbin, dur=durin, per=perin, eos=eosin, eoc=eocin, ddf=ddf0, occ=occin, c1=c1in, c2=c2in)
            mt=tramod_call.get_value(tt, argu)

            #lnprob_thislc = -1./2 * (len(tt) * np.log(2*np.pi) + np.sum(np.log(ee**2)) + chisq ) 
            lnprob_thislc = -1./2. * np.sum( (mt-ft)**2/ee**2 + np.log(ee**2))
            lnprob = lnprob + lnprob_thislc
            chisq = np.sum((mt-ft)**2/ee**2)
            
            # get the transit-only model with no parametric baselines
            basesin_non = np.zeros(20)
            basesin_non[0] = 1.
            argu2 = [tt,ft,xt,yt,wt,at,st,bt,ct,isddf,rprs0,grprs_here,inmcmc,'n',basesin_non,vcont,name,ee,bvar]
            mt0=tramod_call.get_value(tt, argu2)
            
            if inmcmc == 'n':
                mod = np.concatenate((mod,mt))
                emod = np.concatenate((emod,np.zeros(len(mt)))) 

                # #### Monika modificatons for outputs without GPs #####
                #
                # write out an output file in the same format as the GP output files. 
                #   But set the GP prediciton ("pred") to the full model
                ts=tt-tt[0]
                if (baseLSQ == 'y'):
                    mres=ft/mt0
                    #bvar contains the indices of the non-fixed baseline variables
                    coeffstart = np.copy(basesin[bvar])   
                    icoeff,dump = scipy.optimize.leastsq(para_minfunc, coeffstart, args=(bvar, mt0, ft, ts, at, xt, yt, wt, st))
                    coeff = np.copy(basesin)
                    coeff[bvar] = np.copy(icoeff)
                    bfunc_para = basefunc_noCNM(coeff, ts, at, xt, yt, wt, st)
                else:
                    bfunc_para = basefunc_noCNM(basesin, ts, at, xt, yt, wt, st)

                pred=mt0*bfunc_para
                fco_full = ft/bfunc_para
                outfile=name[:-4]+'_out_full.dat'
                of=open(outfile,'w')
                for k in range(len(tt)):
                    of.write('%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n' % (tt[k], ft[k], ee[k], pred[k], bfunc_para[k], mt0[k],fco_full[k])) 
                
                of.close()      
    
    # now do the RVs and add their proba to the model
    for j in range(nRV):
    
        tt = np.copy(tarr[indlist[j+nphot][0]]) # time values of lightcurve j
        ft = np.copy(farr[indlist[j+nphot][0]]) # flux values of lightcurve j
        ee = np.copy(earr[indlist[j+nphot][0]]) # error values of lightcurve j
        xt = np.copy(xarr[indlist[j+nphot][0]])    
        yt = np.copy(yarr[indlist[j+nphot][0]]) # y values of lightcurve j
        wt = np.copy(warr[indlist[j+nphot][0]])
        at = np.copy(aarr[indlist[j+nphot][0]])    
        st = np.copy(sarr[indlist[j+nphot][0]])
        bt = np.copy(barr[indlist[j+nphot][0]])
        ct = np.copy(carr[indlist[j+nphot][0]])
        name = names[j]
        
        argu = [tt,ft,xt,yt,wt,at,st,bt,ct,isddf,rprs0,grprs_here,inmcmc,baseLSQ,bvars,vcont,name,ee]

            # get the current parameters from the pp array.
        pp=p[pindices[j+nphot]]  # the elements of the p array jumping in this RV curve

        ppcount = 0 
        if (0 in jumping[0]):  # same for all data -> check in jumping array
            T0in = pp[0] 
            ppcount = ppcount+1
        else:
            T0in = params[0]
        
        if (1 in jumping[0]):   # same for all data -> check in jumping array
            RpRsin = pp[ppcount]
            ppcount = ppcount+1   
        else:
            RpRsin = params[1]

        if (2 in jumping[0]):   # same for all data -> check in jumping array
            bbin = pp[ppcount]
            ppcount = ppcount+1
        else:
            bbin = params[2]
             
        if (3 in jumping[0]):   # same for all data -> check in jumping array
            durin=pp[ppcount]
            ppcount = ppcount+1
        else:
            durin = params[3]
             
        if (4 in jumping[0]):   # same for all data -> check in jumping array
            perin=pp[ppcount]
            ppcount = ppcount+1
        else:
            perin = params[4]

        if (5 in jumping[0]):   # same for all data -> check in jumping array
            eosin=pp[ppcount]
            ppcount = ppcount+1
        else:
            eosin = params[5]

        if (6 in jumping[0]):   # same for all data -> check in jumping array
            eocin=pp[ppcount]
            ppcount = ppcount+1
        else:
            eocin = params[6]
        
        if (7 in jumping[0]):   # same for all data -> check in jumping array
            Kin=pp[ppcount]
            ppcount = ppcount+1
        else:
            Kin = params[7]
        
        paraminRV = params
        jupind = jumping_noGP[0]
        
        nGPjump = len(p) - len(jupind)
        paraminRV[jupind] = p[0:-nGPjump]
        RVmod = get_RVmod(paraminRV,tt,ft,ee,bt,wt,ct,nfilt,baseLSQ,inmcmc,nddf,nocc,nRV,nphot,j,RVnames,bvarsRV)
        
        #RV_Model(T0=T0in, RpRs=RpRsin, b=bbin, dur=durin, per=perin, eos=eosin, eoc=eocin, K=Kin, gamma=gammain)
        #RVmod.get_value(tt,args=argu)
        
        if (jit_apply == 'y'):
            jitterind = 8 + nddf+nocc + nfilt*4 + 1
            jit = paraminRV[jitterind]
        
        else:
            jit = 0.
        
        chisq = np.sum((RVmod-ft)**2/(ee**2 + jit**2))
  
#        print chisq
  
        lnprob_thisRV = -1./2. * np.sum( (RVmod-ft)**2 / (ee**2 + jit**2) + np.log(2. * np.pi * ee**2 + jit**2) )
        
        mod = np.concatenate((mod,RVmod))
        emod = np.concatenate((emod,ee-ee))


        lnprob = lnprob + lnprob_thisRV

# ====== evaluate limits and priors ======

    for jj in range(len(p)):
        if (priorwid[jj]>0.):
            lpri = norm_prior(p[jj],prior[jj],priorwid[jj])
            lnprob = lnprob + lpri
          #  print p[jj], lim_low[jj], lim_up[jj], prior[jj], priorwid[jj]
            
        llim = limits(p[jj],lim_low[jj], lim_up[jj])
        lnprob = lnprob + llim

# ====== return outputs ======
    
    # MONIKA: adding a check against NaN log probs
    if inmcmc == 'y':
        
        if np.isnan(lnprob) == True:
            lnprob = -np.inf

        return lnprob
    else:      
        return mod, emod, T0in, perin

def norm_prior(value,center,sigma):
    lpri = np.log(1./(2. * np.pi * sigma**2)) - ((value-center)**2/(2. * sigma**2))

    return lpri

def limits(value,lim_low,lim_up):
    if value < lim_low or value > lim_up:  # gp scale
        return -np.inf  
    else:
        return 0.
