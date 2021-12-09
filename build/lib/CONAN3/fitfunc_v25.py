# this is the fitting function for the MCMC.
# 
# takes as input: 
#       photometry and external parameters
#       starting values for all parameters
#       limits for all parameters
#       initial step sizes 

# actually, we will need to input a period, and a midpoint and then recalculate the second midpoint

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy

# sys.path.append("/home/lendl/software/MC3/MCcubed/") 
# sys.path.append("/home/lendl/Work/OurCode/CONAN/")
# sys.path.append("/home/lendl/Work/OurCode/mandagol/")
import mc3
from occultquad import *
from occultnl import *

#plt.ion()

def fitfunc(params, tarr, farr,xarr,yarr,warr,aarr,sarr,barr,carr, nphot, nRV, indlist, filters, nfilt, filnames,nddf,rprs0,erprs0,grprs,egrprs,grnames,groups,ngroup,ewarr,inmcmc, paraCNM, baseLSQ, bvars, bvarsRV, cont, names, RVnames, earr,divwhite,dwCNMarr,dwCNMind):
    # note: params = [T0, rprs, b, dur, per, eos, eoc, K, c1, c2, c3, c4]    
    zarr=np.array([])
    yarr=np.array([])
   
    # perturb values that have a given error that needs to be included each MCMC step
    # vcont is an array containing the (perturebed) contamination value for each filter
    # but attention: the perturbations must be correlated - the flux varies the same way in each filter
    
    #print 'hello!'
    
#    rand_conta = np.random.normal(0,1)  # a random variable from a gaussian centered at 0 with stdev=1

#    vcont=np.zeros([nfilt])
#    for j in range(nfilt):
#        if cont[j,1] != 0: 
#            vcont[j] = cont[j,0] + rand_conta*cont[j,1]
#        else:

    vcont = np.copy(cont[:,0])
        
    # check the impact parameter: if below 0, put it to 0
    # params[2] = (0. if params[2]<0 else params[2])
    # print params[2]

    nantest=np.isnan(np.min(params))
    # if (nantest==True):
    #     print params
    #     print params[5], params[6]
    #     print nanstop1

    # ==============================================================
    # calculate the z values for each lightcurve and put them into a z array. Then below just extract them from that array
    # --------
    # calculate eccentricity and omega
    ecc = params[5]**2+params[6]**2
    #print params
    ##print params[5], params[6]
    # BUG: the problem is that if sqrt(e)sin(omega) and sqrt(e)cos(omega) jump independently, that 
    #      this can create a situation where (sqrt(e)sin(omega))**2 + (sqrt(e)cos(omega))**2 = ecc > 1 !!!
    #      this then makes the procedure meaningless!!!
    #      these parameters are connected, they can only occupy a certain parameter space of values
    #      how can they be independent jump parameters?!? 
    #      !!!!! ====== They have to satisfy the condition that: ====== !!!!!
    #           (par[5]/sqrt(e))**2+(par[6]/sqrt(e))**2 = 1
    #      A quick fix is to just set e to 0.99 if it would be > 1
    #      BUT I DON'T THINK THIS IS A VERY GOOD FIX 
    #        - also we would need to use either par[5] or par[6] and recalculate the other one
    #        - so what's the point in jumping...
    if (ecc >= 0.99):
        ecc = 0.99
        if (params[6]/np.sqrt(ecc) < 1.):
            ome2 = np.arccos(params[6]/np.sqrt(ecc))
            print(ome2)
        else:
            ome2 = 0   
            print('ome2 000')
        params[5] = np.sqrt(ecc)*np.sin(ome2)
        print('here')
    
    if (ecc>0.00001):
        ome1 = np.abs(np.arcsin(params[5]/np.sqrt(ecc)))
        ome2 = np.abs(np.arccos(params[6]/np.sqrt(ecc)))
        if (abs(params[5]/np.sqrt(ecc))<abs(params[6]/np.sqrt(ecc))):
            ome=np.copy(ome1)
        else:
            ome=np.copy(ome2)
        if (abs(ome1 - ome2) > 0.11):
            print('problem in calculation of omega!!! ABORT!!!')
            print(ome1, ome2, ome, ecc, params[5]/np.sqrt(ecc), params[6]/np.sqrt(ecc))
            print(nothing)
    else:
        ome=0.
        ecc=0.
    
    # calculate the ars 
    efac1=np.sqrt(1.-ecc**2)/(1.+ecc*np.sin(ome))
    efac2=params[2]*(1.-ecc**2)/(1.+ecc*np.sin(ome))
    ars = np.sqrt(((1.+params[1])**2 - efac2**2 * (1.-(np.sin(params[3]*np.pi/params[4]))**2))/(np.sin(params[3]*np.pi/params[4]))**2) * efac1
    print(ars)
    #print ars, params[1], params[2], params[3], params[4], params[11], params[27]#, params[14], params[15], params[16], params[17], params[18], params[19]
    #time.sleep(0.05) # delays for 5 seconds
   
    # calculate the true -> eccentric -> mean anomaly at transit -> perihelion time
    TA_tra = np.pi/2. - ome
    TA_tra = np.mod(TA_tra,2.*np.pi)
    EA_tra = 2.*np.arctan( np.tan(TA_tra/2.) * np.sqrt((1.-ecc)/(1.+ecc)) )
    EA_tra = np.mod(EA_tra,2.*np.pi)
    MA_tra = EA_tra - ecc * np.sin(EA_tra)
    MA_tra = np.mod(MA_tra,2.*np.pi)
    mmotio = 2.*np.pi/params[4]   # the mean motion, i.e. angular velocity [rad/day] if we had a circular orbit
    T_peri = params[0] - MA_tra/mmotio
    
    # =========== Radial velocity model calculation ===============
    RVmod = [] # array containing the RV model for all RV data points

    for j in range(nRV):
        # get the true anomaly at the RV timestamps
        tt = np.copy(tarr[indlist[nphot+j][0]]) # time values of RV curve j
        RVmes = np.copy(farr[indlist[nphot+j][0]])
        RVerr = np.copy(earr[indlist[nphot+j][0]])
        bis=np.copy(barr[indlist[nphot+j][0]])
        fwhm=np.copy(farr[indlist[nphot+j][0]])
        contra=np.copy(carr[indlist[nphot+j][0]])
        
        MA_rv = (tt-T_peri)*mmotio
        MA_rv = np.mod(MA_rv,2*np.pi)
        EA_rv = MA_rv + np.sin(MA_rv)*ecc + 1./2.*np.sin(2.*MA_rv)*ecc**2 + \
            (3./8.*np.sin(3.*MA_rv) - 1./8.*np.sin(MA_rv))*ecc**3 + \
                (1./3.*np.sin(4.*MA_rv) - 1./6.*np.sin(2*MA_rv))*ecc**4 + \
                    (1./192*np.sin(MA_rv)-27./128.*np.sin(3.*MA_rv)+125./384.*np.sin(5*MA_rv))*ecc**5 + \
                        (1./48.*np.sin(2.*MA_rv)+27./80.*np.sin(6.*MA_rv)-4./15.*np.sin(4.*MA_rv))*ecc**6    
        EA_rv = np.mod(EA_rv,2*np.pi)
        TA_rv = 2.*np.arctan(np.tan(EA_rv/2.) * np.sqrt((1.+ecc)/(1.-ecc)) )
        TA_rv = np.mod(TA_rv,2*np.pi)  # that's the true anomaly!
        
        # get the model RV at each time stamp
        gammaind = 8 + nddf + nfilt*4 + j
        mod_RV = params[7] * (np.cos(TA_rv + ome) + ecc * np.sin(ome)) + params[gammaind]
        
        bfstartRV= 8 + nddf + nfilt*4 + nRV + nphot*21 # the first index in the param array that refers to a baseline function
        incoeff = list(range(bfstartRV+j*8,bfstartRV+j*8+8))  # the indices for the coefficients for the base function        
        
        ts = tt-np.mean(tt)
        
        if (baseLSQ == 'y'):
            RVmres=RVmes/mod_RV
            #get the indices of the variable baseline parameters via bvar (0 for fixed, 1 for variable)
            ivars = np.copy(bvarsRV[j][0])
            incoeff=np.array(incoeff)
            coeffstart = np.copy(params[incoeff[ivars]])   # RANDOM NOTE: you can use lists as indices to np.arrays but not np.arrays to lists or lists to lists
            if len(ivars) > 0:
                icoeff,dump = scipy.optimize.leastsq(para_minfuncRV, coeffstart, args=(ivars, mod_RV, RVmes, ts, bis, fwhm, contra))
                coeff = np.copy(params[incoeff])   # the full coefficients -- set to those defined in params (in case any are fixed non-zero)
                coeff[ivars] = np.copy(icoeff)     # and the variable ones are set to the result from the minimization
            else:
                coeff = np.copy(params[incoeff])

        else:        
            coeff = np.copy(params[incoeff])   # the coefficients for the base function
              
        bfuncRV=basefuncRV(coeff, ts, bis, fwhm, contra)
        
        mod_RVbl = mod_RV * bfuncRV
        
        # write the RVcurve and the model to file if we're not inside the MCMC
        if (inmcmc == 'n'):
            outfile=RVnames[j][:-4]+'_out.dat'
            of=open(outfile,'w')
        
            for k in range(len(tt)):
                of.write('%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n' % (tt[k], RVmes[k], RVerr[k], mod_RVbl[k],bfuncRV[k],mod_RV[k],RVmes[k]/bfuncRV[k])) 
        
            of.close()
        
      #  indsort = np.unravel_index(np.argsort(TA_rv, axis=None), TA_rv.shape) 
      #  plt.clf()
      #  plt.errorbar(ts,RVmes/mod_RV,yerr=RVerr,fmt='g*',ecolor='g')
      #  plt.plot(ts,mod_RVbl,'r-')
      #  plt.plot(ts,bfuncRV,'y-')
      #  plt.errorbar(EA_rv[indsort],RVmes[indsort],yerr=RVerr[indsort],fmt='g*',ecolor='g')
      #  plt.plot(EA_rv[indsort],mod_RV[indsort],'r-')
      #  plt.show()
      #  plt.pause(0.05)
      #  print 'ping'
        
        if nRV>0:
            RVmod = np.append(RVmod,mod_RVbl)
            
    
    # =========== Transit model calculation =======================
    # now, for all lightcurves, calculate the z values for the timestamps
    for j in range(nphot):
        tt = np.copy(tarr[indlist[j][0]]) # time values of lightcurve j
        MA_lc = (tt-T_peri)*mmotio
        MA_lc = np.mod(MA_lc,2*np.pi)
        # source of the below equation: http://alpheratz.net/Maple/KeplerSolve/KeplerSolve.pdf
        EA_lc = MA_lc + np.sin(MA_lc)*ecc + 1./2.*np.sin(2.*MA_lc)*ecc**2 + \
            (3./8.*np.sin(3.*MA_lc) - 1./8.*np.sin(MA_lc))*ecc**3 + \
                (1./3.*np.sin(4.*MA_lc) - 1./6.*np.sin(2*MA_lc))*ecc**4 + \
                    (1./192*np.sin(MA_lc)-27./128.*np.sin(3.*MA_lc)+125./384.*np.sin(5*MA_lc))*ecc**5 + \
                        (1./48.*np.sin(2.*MA_lc)+27./80.*np.sin(6.*MA_lc)-4./15.*np.sin(4.*MA_lc))*ecc**6
        EA_lc = np.mod(EA_lc,2*np.pi)
        TA_lc = 2.*np.arctan(np.tan(EA_lc/2.) * np.sqrt((1.+ecc)/(1.-ecc)) )
        TA_lc = np.mod(TA_lc,2*np.pi)
        R_lc = ars*(1.-ecc*np.cos(EA_lc))  #normalized (to Rs) planet-star separation
        b_lc = params[2]*(1.-ecc*np.cos(EA_lc))
        x_lc = R_lc * np.sin(TA_lc + ome - np.pi/2.)
        y_lc = np.sqrt(R_lc**2 - b_lc**2)*np.cos(TA_lc + ome - np.pi/2.)
        z_lc = (-1)*b_lc*np.cos(TA_lc + ome - np.pi/2.)
        z_ma = np.sqrt(x_lc**2 + z_lc**2)
        zarr = np.concatenate((zarr,z_ma), axis=0)
        yarr = np.concatenate((yarr,y_lc), axis=0)

        #plt.clf()
        #plt.plot(tt,EA_lc,'g-')
        #plt.plot(tt,TA_lc,'r-')
        #plt.plot(tt,MA_lc,'b-')
        #plt.draw()
        #plt.pause(0.05)

        
    # ========  DEFINE THE CNM for each LC groups!!! ======================
    cnmarr=[]  # in cnmarr, we have the CNMs, ordered by group index [[CNM_g1],[CNM_g2],[CNM_g3],...]. The group indices start with 1
    for j in range(ngroup):
        jj=j+1
        # select LCs belonging to this group
        ind = np.where(np.array(groups) == jj)[0]  #indices in the indlist array that refer to lightcurves in this group
        nlc=len(ind)                      # number of lcs in this group
        nplc=len(indlist[ind[0]][0])      # number of lc points
        gcnm=np.zeros(nplc)               # this will contain the CNM for the group
        
        if (inmcmc == 'n' or egrprs[j] == 0.):
            rprs1 = np.copy(grprs[j])
        else:
            rprs1 = np.random.normal(grprs[j],egrprs[j],1)  # if we're in the mcmc, and we let the CNM vary, adapt the model transit depth
        
        for k in range(nlc):
            tt = np.copy(tarr[indlist[ind[k]][0]])  # time values of lightcurve ind[k]
            z  = np.copy(zarr[indlist[ind[k]][0]])  # z values of lightcurve ind[k]
            y  = np.copy(yarr[indlist[ind[k]][0]])  # z values of lightcurve ind[k]
            ft = np.copy(farr[indlist[ind[k]][0]])  # flux values of lightcurve ind[k]
            fw = np.copy(ewarr[indlist[ind[k]][0]]) # weight factors of this lightcurve
            kk = np.where(filnames == filters[ind[k]])  #ind[k] is the index of the LC in filters or indlist
            kk = np.asscalar(kk[0])        # kk is the index of the LC in the filnames array
            u1ind = 8+nddf+4*kk
            u2ind = 9+nddf+4*kk 
            u1=(params[u1ind] + params[u2ind])/3.
            u2=(params[u1ind] - 2.*params[u2ind])/3.
            
            cmm0,cm0 = occultquad(z,u1,u2,rprs1,nplc)   # cmm is the transit model using the group dF (rprs0)
            cmm0[y<0]=1.
            cm0[y<0]=1.
            ### correct here for contamination: vcont[kk] is the contamination value in the filter of the LC
            cmm = (cmm0-1)/(vcont[kk]+1) + 1

            # normalize the timestamps to the center of the transit
            if paraCNM == 'y':
            # if "paraCNM" = y then calculate the parametric model (excluding the CNM)
                delta = np.round(np.divide((tt[0]-params[0]),params[4]))
                T0_lc=params[0]+delta*params[4]  
                ts=tt-T0_lc
                am=np.copy(aarr[indlist[ind[k]][0]])
                cx=np.copy(xarr[indlist[ind[k]][0]])
                cy=np.copy(yarr[indlist[ind[k]][0]])
                fwhm=np.copy(warr[indlist[ind[k]][0]])
                sky=np.copy(sarr[indlist[ind[k]][0]])
                bfstart= 8+nddf+nfilt*4 + nRV  # the first index in the param array that refers to a baseline function
                incoeff1 = list(range(bfstart+ind[k]*21,bfstart+ind[k]*21+19))  # the indices for the coefficients for the base (no CNM); 
                if (baseLSQ == 'y'):
                    print("LSQ base square parametric models can't be calculated independently of CNM")
                    print(nothing)
                else:
                    coeff1 = np.copy(params[incoeff1]) # the coefficients for the base function
               
                coeff1[0] = coeff1[0]+1. 
                bfunc1 = basefunc_noCNM(coeff1, ts, am, cx, cy, fwhm, sky)            
                cmm = cmm*bfunc1
                #print coeff1
            

            fmm=np.divide(ft,cmm)
            fmm=np.divide(fmm,fw)            
            gcnm = gcnm + fmm
        
        # gcnm=gcnm-np.mean(gcnm)   # this is commented out to go to old CNM version
        cnmarr.append(gcnm)
    
    # calculate the mid-transit points for each lightcurve
    marr=np.array([])
    for j in range(nphot):
        # number of periods elapsed between given T0 and lightcurve start
        tt = np.copy(tarr[indlist[j][0]]) # time values of lightcurve j
        ft = np.copy(farr[indlist[j][0]]) # flux values of lightcurve j
        et = np.copy(earr[indlist[j][0]]) # error values of lightcurve j
        z  = np.copy(zarr[indlist[j][0]]) # z values of lightcurve j
        y  = np.copy(yarr[indlist[j][0]]) # y values of lightcurve j
        
        #  the supplementary parameter: am, cx, cy, fwhm, sky
        am=np.copy(aarr[indlist[j][0]])
        cx=np.copy(xarr[indlist[j][0]])
        cy=np.copy(yarr[indlist[j][0]])
        fwhm=np.copy(warr[indlist[j][0]])
        sky=np.copy(sarr[indlist[j][0]])
        # assign the correct CNM
        gg=int(groups[j]-1)
        cnm=np.array(cnmarr[gg])  # groups[j] gives the group "name", which starts at 1, but the cnmarr starts at 0 
        npo=len(z)                # number of lc points
        
        # normalize the timestamps to the center of the transit
        delta = np.round(np.divide((tt[0]-params[0]),params[4]))
        T0_lc=params[0]+delta*params[4]  
        ts=tt-T0_lc
        
        # identify the filter index of this LC
        k = np.where(filnames == filters[j])  # k is the index of the LC in the filnames array
        k = np.asscalar(k[0])
        u1ind = 8+nddf+4*k  # index in params of the first LD coeff of this filter
        u2ind = 9+nddf+4*k  # index in params of the second LD coeff of this filter
       
        # adapt the RpRs value used in the LC creation to any ddfs
        if nddf>0:
            ddfind= 8+k
            RR=grprs[gg]+params[ddfind]    # the specified GROUP rprs + the respective ddf
        else:
            RR=np.copy(params[1])
        
        # convert the LD coefficients to u1 and u2
        u1=(params[u1ind] + params[u2ind])/3.
        u2=(params[u1ind] - 2.*params[u2ind])/3.
        
        #print params
        #print(params[u1ind],params[u2ind])
        #print(params[0:9],u1,u2)
        #print nothing
        
        mm0,m0 = occultquad(z,u1,u2,RR,npo)   # mm is the transit model
        mm0[y<0]=1.
        m0[y<0]=1.
        # correct here for the contamination
        mm = (mm0-1)/(vcont[k]+1) + 1.
        # print filnames[k], vcont[k], cont[k,0],cont[k,1]
  
        if (params[2] > params[1]+1):
            mm = mm-mm + 1.

       # print params 
        bfstart= 8+nddf+nfilt*4 + nRV  # the first index in the param array that refers to a baseline function
        incoeff = list(range(bfstart+j*21,bfstart+j*21+21))  # the indices for the coefficients for the base function        
        
        if (baseLSQ == 'y'):
            mres=ft/mm
            #get the indices of the variable baseline parameters via bvar (0 for fixed, 1 for variable)
            ivars = np.copy(bvars[j][0])
            incoeff=np.array(incoeff)
            coeffstart = np.copy(params[incoeff[ivars]])   # RANDOM NOTE: you can use lists as indices to np.arrays but not np.arrays to lists or lists to lists
            icoeff,dump = scipy.optimize.leastsq(para_minfunc, coeffstart, args=(ivars, mm, ft, ts, am, cx, cy, fwhm, sky, cnm))
            coeff = np.copy(params[incoeff])   # the full coefficients -- set to those defined in params (in case any are fixed non-zero)
            coeff[ivars] = np.copy(icoeff)     # and the variable ones are set to the result from the minimization
            
        else:        
            coeff = np.copy(params[incoeff])   # the coefficients for the base function
       
        #if (inmcmc == 'y'):
        #    print coeff
        # print cnm
        #    print nothing
    
        bfunc=basefunc(coeff, ts, am, cx, cy, fwhm, sky, cnm)
        mod=mm*bfunc
       # print mod

        fco=ft/bfunc
        
        # write the lightcurve and the model to file if we're not inside the MCMC
        if (inmcmc == 'n'):
            outfile=names[j][:-4]+'_out.dat'
            of=open(outfile,'w')
        
            for k in range(len(tt)):
                of.write('%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n' % (tt[k], ft[k], et[k], mod[k],bfunc[k],mm[k],fco[k])) 
        
            of.close()
        
       # print cnm 
       # print mm
       # print params
        
       # nantest=np.isnan(np.min(mod))
       # if (nantest==True):
       #     print 'nans in model'
       #     print params
        #    print params[5], params[6]
        #    print ecc
        #    print ome
        #    print nanstop
       
       # print mod
       # if (j<1 and inmcmc == 'y'):
       #     print mod
       #     print bfunc
       #     print params[2]
       #     print coeff[12], coeff[13]
       #     plt.clf()
       #     plt.axis([-0.2,0.2,0.98,1.01])
       #     plt.plot(ts,ft, 'b+')
       #     plt.plot(ts,mod, 'r-')
       #     plt.plot(ts,mm0, 'r-')
       #     plt.plot(ts,mm, 'g-')
       #     plt.plot(ts,cnm, 'r-')
       #     plt.plot(ts,(coeff[0]+cnm*coeff[14])*mm, 'b-')
       #     plt.draw()
       #     plt.pause(0.01)
       #     print 'ping'
                      
                      
       # if (j<1 and inmcmc == 'y'):
       #     plt.clf()
       #     plt.axis([-0.2,0.2,0.9,1.1])
       #     plt.plot(ts,ft,'b+')
       #     plt.plot(ts,mod,'g-')
       #     plt.draw()
       #     plt.pause(0.05)
       #     print mod

      #  if (j<1 and inmcmc == 'y'):
      #      plt.clf()
      #      plt.axis([-0.2,0.2,-0.01,0.01])
      #      plt.plot(ts,cnm,'g-')
      #      plt.draw()
      #      plt.pause(0.05)
        
        marr=np.append(marr,mod)
        
    #print ecc, ome, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20]
    
    marr=np.append(marr,RVmod) # append the RV model to the photometry model
    
    chisq = np.sum((marr-farr)**2/earr**2)
  #  print chisq
    
    
    return marr

def basefunc(coeff, ts, am, cx, cy, fwhm, sky, cnm):
    # the full baseline function calculated with the coefficients given; of which some are not jumping and set to 0
    bfunc=coeff[0]+coeff[1]*ts+coeff[2]*np.power(ts,2)+coeff[3]*np.power(ts,3)+coeff[4]*np.power(ts,4)+coeff[5]*am+coeff[6]*np.power(am,2)+coeff[7]*cx+coeff[8]*np.power(cx,2)+coeff[9]*cy+coeff[10]*np.power(cy,2)+coeff[11]*cy*cx+coeff[12]*fwhm+coeff[13]*np.power(fwhm,2)+coeff[14]*sky+coeff[15]*np.power(sky,2)+coeff[16]*np.sin(ts*coeff[17]+coeff[18])+coeff[19]*cnm+coeff[20]*np.power(cnm,2)
    
    return bfunc

def basefunc_noCNM(coeff, ts, am, cx, cy, fwhm, sky):
    # the full baseline function calculated with the coefficients given; of which some are not jumping and set to 0
    bfunc=coeff[0]+coeff[1]*ts+coeff[2]*np.power(ts,2)+coeff[3]*np.power(ts,3)+coeff[4]*np.power(ts,4)+coeff[5]*am+coeff[6]*np.power(am,2)+coeff[7]*cx+coeff[8]*np.power(cx,2)+coeff[9]*cy+coeff[10]*np.power(cy,2)+coeff[11]*cy*cx+coeff[12]*fwhm+coeff[13]*np.power(fwhm,2)+coeff[14]*sky+coeff[15]*np.power(sky,2)+coeff[16]*np.sin(ts*coeff[17]+coeff[18])
    
    return bfunc

def para_minfunc(icoeff, ivars, mm, ft, ts, am, cx, cy, fwhm, sky, cnm):
    icoeff_full = np.zeros(21)
    icoeff_full[ivars] = np.copy(icoeff)
    return (ft - mm * basefunc(icoeff_full, ts, am, cx, cy, fwhm, sky, cnm))

def para_minfuncRV(icoeff, ivars, mod_RV, RVmes, ts, bis, fwhm, contra):
    icoeff_full = np.zeros(21)
    icoeff_full[ivars] = np.copy(icoeff)
    return (RVmes - mod_RV * basefuncRV(icoeff_full, ts, bis, fwhm, contra))

def basefuncRV(coeff, ts, bis, fwhm, contra):
    # the full baseline function calculated with the coefficients given; of which some are not jumping and set to 0
    bfunc=1. + coeff[0]*ts+coeff[1]*np.power(ts,2)+coeff[2]*bis+coeff[3]*np.power(bis,2)+coeff[4]*fwhm+coeff[5]*np.power(fwhm,2)+coeff[6]*contra+coeff[7]*np.power(contra,2)
    
    return bfunc
