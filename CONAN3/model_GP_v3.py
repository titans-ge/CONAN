# this is the transit model as used together with GPs. 
#    it returns model values for ONE transit light curve
#      (will need to be called several times for several light curves)
# 

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
import scipy.stats
from scipy.interpolate import LSQUnivariateSpline

from numpy import (array, size, argmin, abs, diag, log, median,  where, zeros, exp, pi, double)

# GP packages
import george
from george.modeling import Model

import mc3
from occultquad import *
from occultnl import *

#plt.ion()


class Transit_Model(Model):

    parameter_names = ['T0', 'RpRs', 'b', 'dur', 'per', 'eos', 'eoc', 'ddf', 'occ', 'c1', 'c2'] 
    # Parameter names - these are all scalars: this works only for one lc 

    def get_value(self, tarr, args):

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
        bases         =  args[i] ; i+=1   # MONIKA: changed this variable name from "bvar" to "bases" for consistency
        vcont         =  args[i] ; i+=1
        name          =  args[i] ; i+=1
        ee            =  args[i] ; i+=1
        bvar          =  args[i] ; i+=1
        useSpline     =  args[i] ; i+=1

        #pmin, pmax        =  args[i], args[i+1] ; i+=2
        #c1_in, c2_in      = args[i], args[i+1] ; i+=2 
        #c3_in, c4_in      = args[i], args[i+1] ; i+=2 
        #params, Files_path = args[i], args[i+1] ; i+=2     

        earrmod=np.copy(ee)

        # ==============================================================
        # calculate the z values for the lightcurve and put them into a z array. Then below just extract them from that array
        # --------

        ph = np.modf((np.modf((tt-self.T0)/self.per)[0])+1.0)[0] #calculate phase

        # calculate eccentricity and omega
        ecc = self.eos**2+self.eoc**2

        if (ecc >= 0.99):
            ecc = 0.99
            if (self.eoc/np.sqrt(ecc) < 1.):
                ome2 = np.arccos(self.eoc/np.sqrt(ecc))
                # print(ome2)
            else:
                ome2 = 0   
                # print('ome2 000')
            self.eos = np.sqrt(ecc)*np.sin(ome2)
            # print('here')
        
        if (ecc>0.00001):
            if (np.abs(self.eos<0.00000001)):
                ome = np.arctan(np.abs(self.eos/self.eoc))           
            else:
                ome = np.abs(np.arcsin(self.eos/np.sqrt(ecc)))
            
            if (self.eos<0):
                if (self.eoc<0):
                    ome = ome + np.pi
                else:
                    ome = 2.*np.pi - ome
            else:
                if (self.eoc<0):
                    ome = np.pi - ome           
    
        else:
            ome=0.
            ecc=0.
        
        # calculate the ars 
        efac1=np.sqrt(1.-ecc**2)/(1.+ecc*np.sin(ome))
        efac2=self.b*(1.-ecc**2)/(1.+ecc*np.sin(ome))
        ars = np.sqrt(((1.+self.RpRs)**2 - efac2**2 * (1.-(np.sin(self.dur*np.pi/self.per))**2))/(np.sin(self.dur*np.pi/self.per))**2) * efac1
        #print ars, self.RpRs, self.b, self.dur, self.per, self.c4, params[27]#, params[14], params[15], params[16], params[17], params[18], params[19]
        #time.sleep(0.05) # delays for 5 seconds
        
        # calculate the true -> eccentric -> mean anomaly at transit -> perihelion time
        TA_tra = np.pi/2. - ome
        TA_tra = np.mod(TA_tra,2.*np.pi)
        EA_tra = 2.*np.arctan( np.tan(TA_tra/2.) * np.sqrt((1.-ecc)/(1.+ecc)) )
        EA_tra = np.mod(EA_tra,2.*np.pi)
        MA_tra = EA_tra - ecc * np.sin(EA_tra)
        MA_tra = np.mod(MA_tra,2.*np.pi)
        mmotio = 2.*np.pi/self.per   # the mean motion, i.e. angular velocity [rad/day] if we had a circular orbit
        T_peri = self.T0 - MA_tra/mmotio
        
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
        b_lc = self.b*(1.-ecc*np.cos(EA_lc))
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
        
        m0 = np.zeros(npo)
        mm0 = np.zeros(npo)
        mm = np.zeros(npo)
        mod = np.zeros(npo)

        # convert the LD coefficients to u1 and u2
        u1=(self.c1 + self.c2)/3.
        u2=(self.c1 - 2.*self.c2)/3.

        #============= TRANSIT ===========================
        # MONIKA: replaced the y coordinate as defining value for
        #    choice of occultation or transit to be robust for eccentric orbits
        #    old:   ph_transit = np.where((ph <= 0.25) | (ph >= 0.75))

        ph_transit = np.where((y >= 0))
        npo_transit = len(z[ph_transit])

        # delta = np.round(np.divide((tt[0]-self.T0),self.per))
        # T0_lc=self.T0+delta*self.per  
        ts=tt-tt[0]
        
        # adapt the RpRs value used in the LC creation to any ddfs
        if isddf=='y':
            RR=grprs_here+self.ddf    # the specified GROUP rprs + the respective ddf
        else:
            RR=np.copy(self.RpRs)

        mm0[ph_transit],m0[ph_transit] = occultquad(z[ph_transit],u1,u2,RR,npo_transit)   # mm is the transit model
        mm[ph_transit] = (mm0[ph_transit]-1)/(vcont+1) + 1 # correct for the contamination
        
        #============= OCCULTATION ==========================
        ph_occultation = np.where((y < 0))
        npo_occultation = len(z[ph_occultation])

        # delta = np.round(np.divide((tt[0]-self.T0),self.per)) + 0.5   # offset by half a period
        # T0_lc=self.T0+delta*self.per  
        RR=np.sqrt(self.occ)    # the occultation depth converted into a radius ratio
        u1, u2 = 0., 0.         # no limb darkening

        mm0[ph_occultation],m0[ph_occultation] = occultquad(z[ph_occultation],u1,u2,RR,npo_occultation)   # mm is the occultation model
        mm[ph_occultation] = (mm0[ph_occultation]-1)/(vcont+1) + 1 # correct here for the contamination

        # MONIKA: added least square optimisation for baselines
        if (baseLSQ == 'y'):
            #bvar contains the indices of the non-fixed baseline variables        
            coeffstart = np.copy(bases[bvar])   
            icoeff,dump = scipy.optimize.leastsq(para_minfunc, coeffstart, args=(bvar, mm, ft, ts, am, cx, cy, fwhm, sky))
            coeff = np.copy(bases)
            coeff[bvar] = np.copy(icoeff)
            
        else:        
            coeff = np.copy(bases)  # the input coefficients 

        bfunc,_ =basefunc_noCNM(coeff, ts, am, cx, cy, fwhm, sky,ft/mm,useSpline)
        mod=mm*bfunc
        
        marr=np.copy(mod)

        return marr


def basefunc_noCNM(coeff, ts, am, cx, cy, fwhm, sky,res,useSpline):
    # the full baseline function calculated with the coefficients given; of which some are not jumping and set to 0
    bfunc=coeff[0]+coeff[1]*ts+coeff[2]*np.power(ts,2)+ coeff[3]*np.power(ts,3)+ coeff[4]*np.power(ts,4)+ +\
        coeff[5]*am+coeff[6]*np.power(am,2)+coeff[7]*cx+coeff[8]*np.power(cx,2)+coeff[9]*cy+coeff[10]*np.power(cy,2)+ +\
            coeff[11]*fwhm+coeff[12]*np.power(fwhm,2)+coeff[13]*sky+coeff[14]*np.power(sky,2)+coeff[15]*np.sin(ts*coeff[16]+coeff[17])

    if np.all(bfunc==np.ones_like(ts)) or isinstance(res,int) or useSpline.use==False: #if not computing baseline set spline to ones
        spl=np.ones_like(ts)
    else:
        kn, per = useSpline.knots, useSpline.period   #knot spacing and periodicity
        knots = np.arange(min(am)+kn, max(am), kn )
        srt = np.argsort(am)
        x, y = am[srt], (res/bfunc)[srt]
        if per > 0:
            x = np.hstack([x-360,x,x+360])
            y = np.hstack([y,y,y])
            knots = np.hstack([knots-360,knots,knots+360])

        splfunc = LSQUnivariateSpline(x, y, knots, ext="const")
        spl = splfunc(am)
        spl = spl/np.median(spl)  #center spline around 1 so as not to interfere with offset of baseline function
    return bfunc*spl, spl

def para_minfunc(icoeff, ivars, mm, ft, ts, am, cx, cy, fwhm, sky):
    icoeff_full = np.zeros(20)
    icoeff_full[ivars] = np.copy(icoeff)
    bfunc,_ = basefunc_noCNM(icoeff_full, ts, am, cx, cy, fwhm, sky,0,False)
    fullmod = np.multiply(bfunc, mm)

    return (ft - fullmod)
        
#class RV_Model(Model):
    
    #parameter_names = ['T0', 'RpRs', 'b', 'dur', 'per', 'eos', 'eoc', 'K', 'gamma'] 

    #def get_value(self, tarr, args):
    
    #Parameters to REVISE : 

        #Arguments (fixed param)
        #i = 0
        #tt, ft        =  args[i], args[i+1] ; i+=2
        #xt, yt        =  args[i], args[i+1] ; i+=2
        #wt, at        =  args[i], args[i+1] ; i+=2
        #st, bt        =  args[i], args[i+1] ; i+=2
        #ct            =  args[i] ; i+=1
        #isddf, rprs0  =  args[i], args[i+1] ; i+=2
        #grprs_here    =  args[i] ; i+=1
        #inmcmc        =  args[i] ; i+=1
        #baseLSQ       =  args[i] ; i+=1
        #bases         =  args[i] ; i+=1
        #vcont         =  args[i] ; i+=1
        #name          =  args[i] ; i+=1
        #ee            =  args[i] ; i+=1
        #pmin, pmax        =  args[i], args[i+1] ; i+=2
        #c1_in, c2_in      = args[i], args[i+1] ; i+=2 
        #c3_in, c4_in      = args[i], args[i+1] ; i+=2 
        #params, Files_path = args[i], args[i+1] ; i+=2     

        #params[jumping]=np.copy(p)
        #pmin_jump=np.copy(pmin[jumping])
        #pmax_jump=np.copy(pmax[jumping])
        
            
        #earrmod=np.copy(ee)

        #==============================================================
        #calculate the z values for the lightcurve and put them into a z array. Then below just extract them from that array
        #--------
        #calculate eccentricity and omega
        #ecc = self.eos**2+self.eoc**2
        #print params
        #print self.eos, self.eoc
        #BUG: the problem is that if sqrt(e)sin(omega) and sqrt(e)cos(omega) jump independently, that 
             #this can create a situation where (sqrt(e)sin(omega))**2 + (sqrt(e)cos(omega))**2 = ecc > 1 !!!
             #this then makes the procedure meaningless!!!
             #these parameters are connected, they can only occupy a certain parameter space of values
             #how can they be independent jump parameters?!? 
             #!!!!! ====== They have to satisfy the condition that: ====== !!!!!
                  #(par[5]/sqrt(e))**2+(par[6]/sqrt(e))**2 = 1
             #A quick fix is to just set e to 0.99 if it would be > 1
             #BUT I DON'T THINK THIS IS A VERY GOOD FIX 
               #- also we would need to use either par[5] or par[6] and recalculate the other one
               #- so what's the point in jumping...
        #if (ecc >= 0.99):
            #ecc = 0.99
            #if (self.eoc/np.sqrt(ecc) < 1.):
                #ome2 = np.arccos(self.eoc/np.sqrt(ecc))
                #print ome2
            #else:
                #ome2 = 0   
                #print 'ome2 000'
            #self.eos = np.sqrt(ecc)*np.sin(ome2)
            #print 'here'
        
        #if (ecc>0.00001):
            #ome1 = np.arcsin(self.eos/np.sqrt(ecc))
            #ome2 = np.arccos(self.eoc/np.sqrt(ecc))
            #if (abs(self.eos/np.sqrt(ecc))<abs(self.eoc/np.sqrt(ecc))):
                #ome=np.copy(ome1)
            #else:
                #ome=np.copy(ome2)
            #if (abs(ome1 - ome2) > 0.11):
                #print 'problem in calculation of omega!!! ABORT!!!'
                #print ome1, ome2, ome, ecc, self.eos/np.sqrt(ecc), self.eoc/np.sqrt(ecc)
                #print nothing
        #else:
            #ome=0.
            #ecc=0.
        
        #calculate the ars 
        #efac1=np.sqrt(1.-ecc**2)/(1.+ecc*np.sin(ome))
        #efac2=self.b*(1.-ecc**2)/(1.+ecc*np.sin(ome))
        #ars = np.sqrt(((1.+self.RpRs)**2 - efac2**2 * (1.-(np.sin(self.dur*np.pi/self.per))**2))/(np.sin(self.dur*np.pi/self.per))**2) * efac1
        #print ars, self.RpRs, self.b, self.dur, self.per, self.c4, params[27]#, params[14], params[15], params[16], params[17], params[18], params[19]
        #time.sleep(0.05) # delays for 5 seconds
        
        #calculate the true -> eccentric -> mean anomaly at transit -> perihelion time
        #TA_tra = np.pi/2. - ome
        #TA_tra = np.mod(TA_tra,2.*np.pi)
        #EA_tra = 2.*np.arctan( np.tan(TA_tra/2.) * np.sqrt((1.-ecc)/(1.+ecc)) )
        #EA_tra = np.mod(EA_tra,2.*np.pi)
        #MA_tra = EA_tra - ecc * np.sin(EA_tra)
        #MA_tra = np.mod(MA_tra,2.*np.pi)
        #mmotio = 2.*np.pi/self.per   # the mean motion, i.e. angular velocity [rad/day] if we had a circular orbit
        #T_peri = self.T0 - MA_tra/mmotio
    
        #MA_rv = (tt-T_peri)*mmotio
        #MA_rv = np.mod(MA_rv,2*np.pi)
        #EA_rv = MA_rv + np.sin(MA_rv)*ecc + 1./2.*np.sin(2.*MA_rv)*ecc**2 + \
            #(3./8.*np.sin(3.*MA_rv) - 1./8.*np.sin(MA_rv))*ecc**3 + \
            #(1./3.*np.sin(4.*MA_rv) - 1./6.*np.sin(2*MA_rv))*ecc**4 + \
            #(1./192*np.sin(MA_rv)-27./128.*np.sin(3.*MA_rv)+125./384.*np.sin(5*MA_rv))*ecc**5 + \
            #(1./48.*np.sin(2.*MA_rv)+27./80.*np.sin(6.*MA_rv)-4./15.*np.sin(4.*MA_rv))*ecc**6    
        #EA_rv = np.mod(EA_rv,2*np.pi)
        #TA_rv = 2.*np.arctan(np.tan(EA_rv/2.) * np.sqrt((1.+ecc)/(1.-ecc)) )
        #TA_rv = np.mod(TA_rv,2*np.pi)  # that's the true anomaly!

        #get the model RV at each time stamp
        #mod_RV = self.K * (np.cos(TA_rv + ome) + ecc * np.sin(ome)) + self.gamma

        #========= !!! now this is tricky: baselines !!! ===============

        #bfstartRV= 8 + nddf + nfilt*4 + nRV + nphot*21 # the first index in the param array that refers to a baseline function
        #incoeff = range(bfstartRV+j*8,bfstartRV+j*8+8)  # the indices for the coefficients for the base function        

        #ts = tt-np.mean(tt)
    
        #return mod_RV
