import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy

#plt.ion()

def get_RVmod(params,tt,RVmes,RVerr,bis,fwhm,contra,nfilt,baseLSQ,inmcmc,nddf,nocc,nRV,nphot,j,RVnames,bvarsRV, gammaind):

    #RVmes = np.copy(farr[indlist[nphot+j][0]])
    #RVerr = np.copy(earr[indlist[nphot+j][0]])

    ecc = params[5]**2+params[6]**2


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
        ome = np.arctan(np.abs(params[5]/params[6]))  #DA LIEGT DER HUND BEGRABEN!!! TANGENS IST KEIN ISOMORPHISMUS!!!
        if (ome<0):
            ome = ome + 2.*np.pi
        if (params[5]>0 and params[6]<0):
            ome = np.pi - ome

        if (params[5]<0 and params[6]<0):
            ome = np.pi + ome

        if (params[5]<0 and params[6]>0):
            ome = 2.*np.pi - ome
            
    else:
        ome=0.
        ecc=0.
    
    # calculate the ars 
    efac1=np.sqrt(1.-ecc**2)/(1.+ecc*np.sin(ome))
    efac2=params[2]*(1.-ecc**2)/(1.+ecc*np.sin(ome))
    ars = np.sqrt(((1.+params[1])**2 - efac2**2 * (1.-(np.sin(params[3]*np.pi/params[4]))**2))/(np.sin(params[3]*np.pi/params[4]))**2) * efac1
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
    # gammaind = 8 + nddf + nocc+ nfilt*4
    # print(f"params:{params}\ngamma_ind={gammaind}\ngamma={params[gammaind]}")
    mod_RV = params[7] * (np.cos(TA_rv + ome) + ecc * np.sin(ome)) + params[gammaind]

    bfstartRV= 8 + nddf + nocc + nfilt*4 + nRV + nphot*20 # the first index in the param array that refers to a baseline function
    incoeff = list(range(bfstartRV+j*12,bfstartRV+j*12+12))  # the indices for the coefficients for the base function        

    ts = tt-np.mean(tt)

    if (baseLSQ == 'y'):
        RVmres=RVmes/mod_RV
        #get the indices of the variable baseline parameters via bvar (0 for fixed, 1 for variable)
        ivars = np.copy(bvarsRV[j][0])
        #print ivars
        #print incoeff
        #print params
        #time.sleep(100)
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

    mod_RVbl = mod_RV + bfuncRV

    indsort = np.unravel_index(np.argsort(TA_rv, axis=None), TA_rv.shape) 
    plt.clf()
    #plt.errorbar(ts,RVmes,yerr=RVerr,fmt='g*',ecolor='g')
    #plt.plot(ts,mod_RVbl,'r-')
    #plt.plot(ts,bfuncRV,'y-')
 #   plt.errorbar(EA_rv[indsort],RVmes[indsort],yerr=RVerr[indsort],fmt='g*',ecolor='g')
 #   plt.plot(EA_rv[indsort],mod_RV[indsort],'r-')
 #   plt.show(block=False)
 #   plt.pause(0.001)

# write the RVcurve and the model to file if we're not inside the MCMC
    if (inmcmc == 'n'):
        outfile=RVnames[j][:-4]+'_out.dat'
        of=open(outfile,'w')
        for k in range(len(tt)):
            of.write('%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n' % (tt[k], RVmes[k], RVerr[k], mod_RVbl[k],bfuncRV[k],mod_RV[k],RVmes[k]-bfuncRV[k])) 

        of.close()

    return mod_RVbl



def para_minfuncRV(icoeff, ivars, mod_RV, RVmes, ts, bis, fwhm, contra):
    icoeff_full = np.zeros(21)
    icoeff_full[ivars] = np.copy(icoeff)
    return (RVmes - mod_RV * basefuncRV(icoeff_full, ts, bis, fwhm, contra))       

def basefuncRV(coeff, ts, bis, fwhm, contra):
    # the full baseline function calculated with the coefficients given; of which some are not jumping and set to 0
    bfunc=coeff[0]*ts+coeff[1]*np.power(ts,2)+coeff[2]*bis+coeff[3]*np.power(bis,2)+ +\
            coeff[4]*fwhm+coeff[5]*np.power(fwhm,2)+coeff[6]*contra+coeff[7]*np.power(contra,2)+ +\
                coeff[8]*np.sin(coeff[9]*ts+coeff[10])
    
    return bfunc    
