# this is the transit model as used together with GPs. 
#    it returns model values for ONE transit light curve
#      (will need to be called several times for several light curves)

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from scipy.interpolate import LSQUnivariateSpline, LSQBivariateSpline

from numpy import (array, size, argmin, abs, diag, log, median,  where, zeros, exp, pi, double)
from george.modeling import Model

from occultquad import *
from occultnl import *
from .utils import rho_to_aR, cosine_atm_variation, reflection_atm_variation, phase_fold,convert_LD
from types import SimpleNamespace


class Transit_Model(Model):
    """
    computes the transit model for a given set of parameters along with the baseline

    Parameters
    ----------
    rho_star : float
        Stellar density [g/cm^3]
    T0 : float
        Mid-transit time [days]
    RpRs : float
        Planet-to-star radius ratio
    b : float
        Impact parameter
    per : float
        Orbital period [days]
    eos : float
        sqrt(ecc)*sin(omega)
    eoc : float
        sqrt(ecc)*cos(omega)
    ddf : float
        if ddf is not None, then depth variation being used and the RpRs is the group rprs.
    c1 : float
        LD coefficient 1
    c2 : float
        LD coefficient 2
    occ : float
        Occultation depth
    A : float
        Amplitude of the atmospheric variation
    delta : float
        hotspot shift of the atmospheric variation in degrees
    npl : int
        number of planets

    Returns
    -------
    marr : array-like
        The lightcurve model for the given parameters
    

    """

    def __init__(self, rho_star, T0, RpRs, b, per, eos, eoc, ddf, q1, q2, occ=0, A=None, delta=None, npl=1):
        self.rho_star = rho_star
        self.T0       = T0
        self.RpRs     = RpRs
        self.b        = b
        # self.dur      = dur
        self.per      = per
        self.eos      = eos
        self.eoc      = eoc
        self.ddf      = ddf
        self.occ      = occ
        self.q1       = q1
        self.q2       = q2
        self.npl      = npl
        self.A        = A
        self.delta    = delta

        self.parameter_names = ['rho_star','T0', 'RpRs', 'b', 'per', 'eos', 'eoc', 'ddf', 'q1', 'q2', 'occ', 'A','delta']

    def get_value(self, tarr, args=None,planet_only=False,ss=None):
        """ 
        computes the transit/occultation/phase curve model for a given set of parameters along with the baseline
        
        Parameters
        ----------
        tarr : array-like
            The timestamps of the lightcurve
        args : array-like
            The arguments of the model
        planet_only : bool
            If True, only return the planet model (no baseline)

        Returns
        -------
        marr : array-like
            The lightcurve model for the given parameters
        model_components : dict
            The components of the model for each planet in a system
        """

# Parameters to add
        if args is not None:
            # Arguments (fixed param)
            i = 0
            tt, ft             =  args[i], args[i+1] ; i+=2
            col3_in, col4_in   =  args[i], args[i+1] ; i+=2
            col6_in, col5_in   =  args[i], args[i+1] ; i+=2
            col7_in, bis_in    =  args[i], args[i+1] ; i+=2
            contra_in          =  args[i] ; i+=1
            isddf, rprs0       =  args[i], args[i+1] ; i+=2
            grprs_here         =  args[i] ; i+=1
            inmcmc             =  args[i] ; i+=1
            baseLSQ            =  args[i] ; i+=1
            bases              =  args[i] ; i+=1   # MONIKA: changed this variable name from "bvar" to "bases" for consistency
            vcont              =  args[i] ; i+=1
            name               =  args[i] ; i+=1
            ee                 =  args[i] ; i+=1
            bvar               =  args[i] ; i+=1
            useSpline          =  args[i] ; i+=1

        else: 
            tt    = tarr
            isddf = "n"
            vcont = 0

        f_trans  = np.ones_like(tt)       #transit
        f_occ    = np.ones_like(tt)       #occultation
        pl_mod   = np.zeros_like(tt)      #total lc model
        model_components = {}           #components of the model lc for each planet
        rescale = lambda x: ((x - np.min(x))/np.ptp(x) ) if np.all(min(x) != max(x)) else x
        
        tt_ss   = ss.supersample(tt) if ss is not None else tt   #supersample the timestamps if ss is not None

        for n in range(self.npl):    #iterate through all planets
            # ==============================================================
            # calculate the z values for the lightcurve and put them into a z array. Then below just extract them from that array
            # --------
            # calculate eccentricity and omega
            ecc = self.eos[n]**2+self.eoc[n]**2

            if (ecc >= 0.99):
                ecc = 0.99
                if (self.eoc[n]/np.sqrt(ecc) < 1.):
                    ome2 = np.arccos(self.eoc[n]/np.sqrt(ecc))
                    # print(ome2)
                else:
                    ome2 = 0   
                    # print('ome2 000')
                self.eos[n] = np.sqrt(ecc)*np.sin(ome2)
                # print('here')
            
            if (ecc>0.00001):
                if (np.abs(self.eos[n]<0.00000001)):
                    ome = np.arctan(np.abs(self.eos[n]/self.eoc[n]))           
                else:
                    ome = np.abs(np.arcsin(self.eos[n]/np.sqrt(ecc)))
                
                if (self.eos[n]<0):
                    if (self.eoc[n]<0):
                        ome = ome + np.pi
                    else:
                        ome = 2.*np.pi - ome
                else:
                    if (self.eoc[n]<0):
                        ome = np.pi - ome           
        
            else:
                ome=0.
                ecc=0.
            
            # calculate the ars 
            efac1 = np.sqrt(1.-ecc**2)/(1.+ecc*np.sin(ome))
            efac2 = self.b[n]*(1.-ecc**2)/(1.+ecc*np.sin(ome))
            # ars   = np.sqrt(((1.+self.RpRs[n])**2 - efac2**2 * (1.-(np.sin(self.dur[n]*np.pi/self.per[n]))**2))/(np.sin(self.dur[n]*np.pi/self.per[n]))**2) * efac1
            ars   = rho_to_aR(self.rho_star,self.per[n])
            
            # calculate the true -> eccentric -> mean anomaly at transit -> perihelion time
            TA_tra = np.pi/2. - ome
            TA_tra = np.mod(TA_tra,2.*np.pi)
            EA_tra = 2.*np.arctan( np.tan(TA_tra/2.) * np.sqrt((1.-ecc)/(1.+ecc)) )
            EA_tra = np.mod(EA_tra,2.*np.pi)
            MA_tra = EA_tra - ecc * np.sin(EA_tra)
            MA_tra = np.mod(MA_tra,2.*np.pi)
            mmotio = 2.*np.pi/self.per[n]   # the mean motion, i.e. angular velocity [rad/day] if we had a circular orbit
            T_peri = self.T0[n] - MA_tra/mmotio
            
            # =========== Transit model calculation =======================
            # now, for all lightcurves, calculate the z values for the timestamps
            MA_lc = (tt_ss - T_peri)*mmotio
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
            R_lc  = ars*(1.-ecc*np.cos(EA_lc))  #normalized (to Rs) planet-star separation
            b_lc  = self.b[n]*(1.-ecc*np.cos(EA_lc))
            x_lc  = R_lc * np.sin(TA_lc + ome - np.pi/2.)
            y_lc  = np.sqrt(R_lc**2 - b_lc**2)*np.cos(TA_lc + ome - np.pi/2.)
            z_lc  = (-1)*b_lc*np.cos(TA_lc + ome - np.pi/2.)
            z_ma  = np.sqrt(x_lc**2 + z_lc**2)
            z     = np.copy(z_ma)
            y     = np.copy(y_lc)
            
            npo=len(z)                # number of lc points
            m0  = np.zeros(npo)
            mm0 = np.zeros(npo)

            # convert the LD coefficients to u1 and u2
            u1,u2 = convert_LD(self.q1,self.q2,conv="q2u")

            #============= TRANSIT ===========================
            # MONIKA: replaced the y coordinate as defining value for
            #    choice of occultation or transit to be robust for eccentric orbits
            #    old:   ph_transit = np.where((ph <= 0.25) | (ph >= 0.75))
            ph_transit  = np.where((y >= 0))
            npo_transit = len(z[ph_transit])

            # adapt the RpRs value used in the LC creation to any ddfs
            if isddf=='y':
                RR=grprs_here+self.ddf    # the specified GROUP rprs + the respective ddf (deviation)
            else:
                RR=np.copy(self.RpRs[n])
            mm0[ph_transit],m0[ph_transit] = occultquad(z[ph_transit],u1,u2,RR,npo_transit)   # mm0 is the transit model
            
            #============= OCCULTATION ==========================
            ph_occultation  = np.where((y < 0))
            npo_occultation = len(z[ph_occultation])

            Fp = self.occ       # the occultation depth or dayside flux
            RR = np.sqrt(Fp)    # the occultation depth converted into a radius ratio
            u1, u2 = 0., 0.     # no limb darkening
            mm0[ph_occultation],m0[ph_occultation] = occultquad(z[ph_occultation],u1,u2,RR,npo_occultation)   # mm0 is the occultation model

            if self.A not in [None,0]:    
                #sepate the transit and occultation models and add the atmospheric variation
                f_trans[ph_transit]   = mm0[ph_transit]
                f_occ[ph_occultation] = mm0[ph_occultation]
                f_occ                 = rescale(f_occ)  #rescale the occultation model to be between 0 and 1
                
                phase  = phase_fold(tt_ss,self.per[n],self.T0[n])
                atm    = cosine_atm_variation(phase, Fp, self.A, self.delta)
                lc_mod = f_trans + f_occ*atm.pc
            else:
                lc_mod = mm0.copy()

            lc_mod = ss.rebin_flux(lc_mod) if ss is not None else lc_mod  #rebin the model to the original cadence
            
            #save the model components
            model_components[f"pl_{n+1}"] = lc_mod.copy()
            lc_mod -= 1      #zero baseline
            pl_mod += lc_mod    #add each planet transit/occ model to total mod

        #correct for the contamination
        mm = pl_mod/(vcont+1) + 1

        if planet_only:
            return mm, model_components


        #==== set up for LC and baseline creation
        col5 = np.copy(col5_in)
        col3 = np.copy(col3_in)
        col4 = np.copy(col4_in)
        col6 = np.copy(col6_in)
        col7 = np.copy(col7_in)
        ts   = tt-tt[0]

        # MONIKA: added least square optimisation for baselines
        if (baseLSQ == 'y'):
            #bvar contains the indices of the non-fixed baseline variables        
            coeffstart  = np.copy(bases[bvar])   
            icoeff,dump = scipy.optimize.leastsq(para_minfunc, coeffstart, args=(bvar, mm, ft, ts, col5, col3, col4, col6, col7))
            coeff = np.copy(bases)
            coeff[bvar] = np.copy(icoeff)   
        else:        
            coeff = np.copy(bases)  # the input coefficients 

        bfunc,spl_comp = basefunc_noCNM(coeff, ts, col5, col3, col4, col6, col7,ft/mm,useSpline)
        mod=mm*bfunc
        
        lc_result = SimpleNamespace(planet_LC=mm, full_LCmod=mod, LC_bl=bfunc, spline=spl_comp )

        return lc_result


def basefunc_noCNM(coeff, ts, col5, col3, col4, col6, col7,res,useSpline):
    # the full baseline function calculated with the coefficients given; of which some are not jumping and set to 0
    DA = locals().copy()
    DA["col0"] = ts
    _ = [DA.pop(item) for item in ["coeff", "ts", "res", "useSpline"]]

    bfunc  = coeff[0] + coeff[1]*ts + coeff[2]*np.power(ts,2) + coeff[3]*np.power(ts,3) + coeff[4]*np.power(ts,4)     #time col0
    bfunc += coeff[5]*col3  + coeff[6]*np.power(col3,2)        #x col3
    bfunc += coeff[7]*col4  + coeff[8]*np.power(col4,2)        #y col4
    bfunc += coeff[9]*col5  + coeff[10]*np.power(col5,2)       #airmass col5
    bfunc += coeff[11]*col6 + coeff[12]*np.power(col6,2)  #fwhm/conta col6
    bfunc += coeff[13]*col7 + coeff[14]*np.power(col7,2)    #sky/bg col7
    bfunc += coeff[15]*np.sin(ts*coeff[16]+coeff[17])   #sinusoidal col8

    if isinstance(res,int) or useSpline.use==False: #if not computing baseline set spline to ones
        spl= x = np.ones_like(ts)
    else:
        kn,s_par,dim = useSpline.knots, useSpline.par,useSpline.dim   #knot_spacing,param
        if dim == 1:
            x      = np.copy(DA[s_par])
            knots  = np.arange(min(x)+kn, max(x), kn )
            srt    = np.argsort(x)
            xs, ys = x[srt], (res/bfunc)[srt]

            splfunc = LSQUnivariateSpline(xs, ys, knots, k=useSpline.deg, ext="const")
            spl = splfunc(x)     #evaluate the spline at the original x values
            # spl = spl/np.median(spl)  #center spline around 1 so as not to interfere with offset of baseline function
        if dim == 2:
            x1      = np.copy(DA[s_par[0]])
            x2      = np.copy(DA[s_par[1]])
            x       = x1 #np.vstack([x1,x2]).T
            knots1  = np.arange(min(x1)+kn, max(x1), kn )
            knots2  = np.arange(min(x2)+kn, max(x2), kn )
            ys      = (res/bfunc)

            splfunc = LSQBivariateSpline(x1, x2, ys, knots1, knots2, kx=useSpline.deg[0], ky=useSpline.deg[1])
            spl = splfunc(x1,x2,grid=False)     #evaluate the spline at the original x values

    return bfunc*spl, spl

def para_minfunc(icoeff, ivars, mm, ft, ts, col5, col3, col4, col6, col7):
    icoeff_full = np.zeros(20)
    icoeff_full[ivars] = np.copy(icoeff)
    bfunc,_,_ = basefunc_noCNM(icoeff_full, ts, col5, col3, col4, col6, col7,0,False)   # fit baseline without spline here
    fullmod = np.multiply(bfunc, mm)

    return (ft - fullmod)