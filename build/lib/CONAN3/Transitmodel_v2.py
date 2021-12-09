# this is the transit model as used together with GPs. 
#    it returns model values for ONE transit light curve
#      (will need to be called several times for several light curves)
# 
# takes as input: 
#       photometry 
#       starting values for all parameters

# actually, we will need to input a period, and a midpoint and then recalculate the second midpoint

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
import scipy.stats

from numpy import (array, size, argmin, abs, diag, log, median,  where, zeros, exp, pi, double)

# sys.path.append("/home/lendl/software/MC3/MCcubed/") 
# sys.path.append("/home/lendl/Work/OurCode/CONAN/")
# sys.path.append("/home/lendl/Work/OurCode/mandagol/")

from occultquad import *
from occultnl import *


def get_Tramod(T0, RpRs, b, dur, per, eos, eoc, ddf, occ, c1, c2, args):

# Parameters to add

        # Arguments (fixed param)
        i = 0
        tt, ft        =  args[i], args[i+1] ; i+=2
        xt, yt        =  args[i], args[i+1] ; i+=2
        wt, at        =  args[i], args[i+1] ; i+=2
        st, bt        =  args[i], args[i+1] ; i+=2
        ct            =  args[i] ; i+=1
        isddf, rprs0  =  args[i], args[i+1] ; i+=2
        grprs_here    =  args[i] ; i+=1
        inmcmc        =  args[i] ; i+=1
        baseLSQ       =  args[i] ; i+=1
        bvars         =  args[i] ; i+=1
        vcont         =  args[i] ; i+=1
        name          =  args[i] ; i+=1
        ee            =  args[i] ; i+=1
        #pmin, pmax        =  args[i], args[i+1] ; i+=2
        #c1_in, c2_in      = args[i], args[i+1] ; i+=2 
        #c3_in, c4_in      = args[i], args[i+1] ; i+=2 
        #params, Files_path = args[i], args[i+1] ; i+=2     

        #params[jumping]=np.copy(p)
        #pmin_jump=np.copy(pmin[jumping])
        #pmax_jump=np.copy(pmax[jumping])
        
            
        earrmod=np.copy(ee)

        # ==============================================================
        # calculate the z values for the lightcurve and put them into a z array. Then below just extract them from that array
        # --------
        # calculate eccentricity and omega
        ecc = eos**2+eoc**2

        if (ecc >= 0.99):
            ecc = 0.99
            if (eoc/np.sqrt(ecc) < 1.):
                ome2 = np.arccos(eoc/np.sqrt(ecc))
                print(ome2)
            else:
                ome2 = 0   
                print('ome2 000')
            eos = np.sqrt(ecc)*np.sin(ome2)
            print('here')
        
        if (ecc>0.00001):
            if (np.abs(eos<0.00000001)):
                ome = np.arctan(np.abs(eos/eoc))           
            else:
                ome = np.abs(np.arcsin(eos/np.sqrt(ecc)))
            
            if (eos<0):
                if (eoc<0):
                    ome = ome + np.pi
                else:
                    ome = 2.*np.pi - ome
            else:
                if (eoc<0):
                    ome = np.pi - ome           
    
        else:
            ome=0.
            ecc=0.
        
        # calculate the ars 
        efac1=np.sqrt(1.-ecc**2)/(1.+ecc*np.sin(ome))
        efac2=b*(1.-ecc**2)/(1.+ecc*np.sin(ome))
        ars = np.sqrt(((1.+RpRs)**2 - efac2**2 * (1.-(np.sin(dur*np.pi/per))**2))/(np.sin(dur*np.pi/per))**2) * efac1
        #print ars, self.RpRs, self.b, self.dur, self.per, self.c4, params[27]#, params[14], params[15], params[16], params[17], params[18], params[19]
        #time.sleep(0.05) # delays for 5 seconds
        
        # calculate the true -> eccentric -> mean anomaly at transit -> perihelion time
        TA_tra = np.pi/2. - ome
        TA_tra = np.mod(TA_tra,2.*np.pi)
        EA_tra = 2.*np.arctan( np.tan(TA_tra/2.) * np.sqrt((1.-ecc)/(1.+ecc)) )
        EA_tra = np.mod(EA_tra,2.*np.pi)
        MA_tra = EA_tra - ecc * np.sin(EA_tra)
        MA_tra = np.mod(MA_tra,2.*np.pi)
        mmotio = 2.*np.pi/per   # the mean motion, i.e. angular velocity [rad/day] if we had a circular orbit
        T_peri = T0 - MA_tra/mmotio
        
        # =========== Transit model calculation =======================
        # now, for all lightcurves, calculate the z values for the timestamps
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
        b_lc = b*(1.-ecc*np.cos(EA_lc))
        x_lc = R_lc * np.sin(TA_lc + ome - np.pi/2.)
        y_lc = np.sqrt(R_lc**2 - b_lc**2)*np.cos(TA_lc + ome - np.pi/2.)
        z_lc = (-1)*b_lc*np.cos(TA_lc + ome - np.pi/2.)
        z_ma = np.sqrt(x_lc**2 + z_lc**2)
        z = np.copy(z_ma)
        y = np.copy(y_lc)
        
        #==== set up for LC and baseline creation
        am=np.copy(at)
        cx=np.copy(xt)
        cy=np.copy(yt)
        fwhm=np.copy(wt)
        sky=np.copy(st)
        npo=len(z)                # number of lc points
        
        #===== decide if transit or occultation =======
        if np.mean(y)>0:   # TRANSIT
            # number of periods elapsed between given T0 and lightcurve start            
            # normalize the timestamps to the center of the transit
            delta = np.round(np.divide((tt[0]-T0),per))
            T0_lc=T0+delta*per  
            #ts=tt-T0_lc
            ts=tt-tt[0]
            # adapt the RpRs value used in the LC creation to any ddfs
            if isddf=='y':
                RR=grprs_here+ddf    # the specified GROUP rprs + the respective ddf
            else:
                RR=np.copy(RpRs)
            
            # convert the LD coefficients to u1 and u2
            u1=(c1 + c2)/3.
            u2=(c1 - 2.*c2)/3.
        
        else:    # OCCULTATION
            delta = np.round(np.divide((tt[0]-T0),per)) + 0.5   # offset by half a period
            T0_lc=T0+delta*per 
            #ts=tt-T0_lc
            ts=tt-tt[0]
            RR=np.sqrt(occ)    # the occultation depth converted into a radius ratio
            u1, u2 = 0., 0.         # no limb darkening
            #print(RR,self.occ,T0_lc)
            
        
        # ======== THE TRANSIT (OCCULTATION) MODEL ===========
        
        mm0,m0 = occultquad(z,u1,u2,RR,npo)   # mm is the transit model
        # correct here for the contamination
        mm = (mm0-1)/(vcont+1) + 1
        
        # ======= THE BASELINE MODEL =====================
        bfunc=basefunc_noCNM(bvars, ts, am, cx, cy, fwhm-np.median(fwhm), sky)
        mod=mm*bfunc
        fco=ft/bfunc
        # =================================================
        
        #time.sleep(0.1)
        
        marr=np.copy(mod)
        
        # write the lightcurve and the model to file if we're not inside the MCMC
        if (inmcmc == 'n'):
            print('here')
            print(bvars)
            outfile=name[:-4]+'_out.dat'
            of=open(outfile,'w')
            
            for k in range(len(tt)):
                of.write('%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n' % (tt[k], ft[k], ee[k], mod[k],bfunc[k],mm[k],fco[k])) 
        
            of.close()
            
        return marr


def basefunc_noCNM(coeff, ts, am, cx, cy, fwhm, sky):
    # the full baseline function calculated with the coefficients given; of which some are not jumping and set to 0
    bfunc=coeff[0]+coeff[1]*ts+coeff[2]*np.power(ts,2)+coeff[3]*np.power(ts,3)+coeff[4]*np.power(ts,4)+coeff[5]*am+coeff[6]*np.power(am,2)+coeff[7]*cx+coeff[8]*np.power(cx,2)+coeff[9]*cy+coeff[10]*np.power(cy,2)+coeff[11]*fwhm+coeff[12]*np.power(fwhm,2)+coeff[13]*sky+coeff[14]*np.power(sky,2)+coeff[15]*np.sin(ts*coeff[16]+coeff[17])
    
    return bfunc

