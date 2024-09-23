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

try:
    from occultquad import *
except ImportError:
    print("Could not import occultquad. Using python equivalent (~30X slower)")
    from .occultquad_pya import OccultQuadPy
    OQ = OccultQuadPy()
    occultquad = OQ.occultquad

from .utils import rho_to_aR, Tdur_to_aR, cosine_atm_variation, reflection_atm_variation, phase_fold,convert_LD,rescale0_1,sesinw_secosw_to_ecc_omega
from .utils import light_travel_time_correction
from types import SimpleNamespace

def get_anomaly(t, T0, per, ecc, omega):
    """
    Calculate the eccentric and true anomaly for a given time t, eccentricity ecc, argument of periastron omega, mid-transit time T0, and period per.

    Parameters 
    ----------
    t : array-like
        timestamps
    T0 : float
        mid-transit time
    per : float
        orbital period
    ecc : float
        eccentricity
    omega : float  
        argument of periastron in radians
    """
    # calculate the true -> eccentric -> mean anomaly at transit -> perihelion time
    if ecc==0: omega = np.pi/2.  # if circular orbit, set omega to pi/2
    TA_tra = np.pi/2. - omega
    TA_tra = np.mod(TA_tra,2.*np.pi)
    EA_tra = 2.*np.arctan( np.tan(TA_tra/2.) * np.sqrt((1.-ecc)/(1.+ecc)) )
    EA_tra = np.mod(EA_tra,2.*np.pi)
    MA_tra = EA_tra - ecc * np.sin(EA_tra)
    MA_tra = np.mod(MA_tra,2.*np.pi)
    mmotio = 2.*np.pi/per   # the mean motion, i.e. angular velocity [rad/day] if we had a circular orbit
    T_peri = T0 - MA_tra/mmotio

    MA = (t - T_peri)*mmotio       
    MA = np.mod(MA,2*np.pi)
    # # source of the below equation: http://alpheratz.net/Maple/KeplerSolve/KeplerSolve.pdf
    EA_lc = MA + np.sin(MA)*ecc + 1./2.*np.sin(2.*MA)*ecc**2 + \
                (3./8.*np.sin(3.*MA) - 1./8.*np.sin(MA))*ecc**3 + \
                    (1./3.*np.sin(4.*MA) - 1./6.*np.sin(2*MA))*ecc**4 + \
                        (1./192*np.sin(MA)-27./128.*np.sin(3.*MA)+125./384.*np.sin(5*MA))*ecc**5 + \
                            (1./48.*np.sin(2.*MA)+27./80.*np.sin(6.*MA)-4./15.*np.sin(4.*MA))*ecc**6
    EA_lc = np.mod(EA_lc,2*np.pi)
    TA_lc = 2.*np.arctan(np.tan(EA_lc/2.) * np.sqrt((1.+ecc)/(1.-ecc)) )
    TA_lc = np.mod(TA_lc,2*np.pi)  # that's the true anomaly!

    return EA_lc, TA_lc

def TTV_Model(tarr, rho_star=None, dur=None, T0_list=None, RpRs=None, b=None, per=None, 
                sesinw=[0], secosw=[0], q1=0,q2=0, split_conf=None,ss=None, vcont=0,Rstar=None,
                custom_LCfunc=None, cst_pars={}):
    """ 
    computes the TTV model for a given set of parameters along with the baseline

    Parameters
    ----------
    tarr : array-like
        The timestamps of the lightcurve
    rho_star : float
        Stellar density [g/cm^3]
    dur: float
        Duration of the transit [days]
    T0_list : list
        transit times in each chunk of the data
    RpRs : list
        Planet-to-star radius ratio
    b : list
        Impact parameter
    per : list
        Orbital period [days]
    sesinw : list
        sqrt(ecc)*sin(omega)
    secosw : list
        sqrt(ecc)*cos(omega)
    q1 : float
        LD coefficient 1
    q2 : float
        LD coefficient 2
    split_conf : SimpleNamespace
        The configuration for the split data
    ss : SimpleNamespace
        The configuration for the supersampling
    vcont: float
        contamination factor
    Rstar: float
        Stellar radius in Rsun. used to calculate light travel time correction. Default: None to not perform light travel time correction
    custom_LCfunc : SimpleNamespace
        object containing the custom light curve function to be added to the model. Default: None
    cst_pars : dict
        additional parameters for the custom light curve function. Default: {}

    Returns
    -------
    mm : array-like
        The lightcurve model for the given parameters
    model_components : dict
        The components of the model for each planet in a system    

    Examples
    --------
    >>> from CONAN3.models import TTV_Model
    >>> from CONAN3.utils import split_transits
    >>> import numpy as np
    >>> spt = split_transits(t, P=[0.9414526], t_ref=[1375.1698], flux=f)
    >>> trans_mod,_ = TTV_Model(t, rho_star = rho_star, T0=spt.t0_list, RpRs=[RpRs], b =[b], 
    >>>                         per=[per], sesinw=sesinw, secosw=secosw, q1=q1, q2=q2, split_conf=spt)

    """

    mm = np.zeros_like(tarr)  #np.ones_like(tarr)        #zero baseline of lc model

    #for each chunk of the data, compute transit model with different T0s for the transit of each planet
    for i in range(split_conf.n_chunks):     
        this_t0    = T0_list[i]      #transit times in this chunk of data
        cuts       = split_conf.tr_edges[i]
        ind        = (tarr>=cuts[0]) & (tarr<=cuts[1])
        tarr_split = tarr[ind]

        plnum      = split_conf.plnum_list[i]   #planet number in this chunk of data
        P          = list(np.array(per)[plnum])
        rprs       = list(np.array(RpRs)[plnum])
        sesinw_    = list(np.array(sesinw)[plnum])
        secosw_    = list(np.array(secosw)[plnum])
        imp_par    = list(np.array(b)[plnum])

        TM = Transit_Model(rho_star=rho_star, dur=dur, T0=this_t0, RpRs=rprs, b=imp_par, per=P, sesinw=sesinw_, secosw=secosw_, 
                            ddf=0, occ=0, Fn=0, delta=0, q1=q1, q2=q2,cst_pars=cst_pars, npl=len(this_t0))
        this_trans,_ = TM.get_value(tarr_split,ss=ss,vcont=vcont,Rstar=Rstar,model_phasevar=False,custom_LCfunc=custom_LCfunc)  #compute the transit model for this chunk of data
        mm[ind]     += (this_trans-1)   #add the zero-baseline transit chunk to the zero baseline total model

    mm += 1   #take total baseline to 1
    return mm, {"pl_1": mm}

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
    sesinw : float
        sqrt(ecc)*sin(omega)
    secosw : float
        sqrt(ecc)*cos(omega)
    ddf : float
        if ddf is not 0, then depth variation is being used and this value is added to the base rprs, grprs.
    q1 : float
        LD coefficient 1
    q2 : float
        LD coefficient 2
    occ : float
        Occultation depth
    Fp : float
        nightside flux ratio
    delta : float
        hotspot shift of the atmospheric variation in degrees
    A_ev : float
        semi-Amplitude of the ellipsoidal variation
    A_db : float
        semi-Amplitude of the Doppler boosting
    cst_pars : dict
        additional parameters for the custom light curve function. Default: {}
    npl : int
        number of planets

    Returns
    -------
    marr : array-like
        The lightcurve model for the given parameters

    Examples
    --------
    >>> from CONAN3.models import Transit_Model
    >>> TM  = Transit_Model(rho_star= 0.565, T0=0, RpRs=0.1, b=0.1, per=3, sesinw=0, sesinw=0, q1=0.2, q2=0.3)
    >>> flux,_ = TM.get_value(time)
    """

    def __init__(self, rho_star=None, dur=None, T0=None, RpRs=None, b=None, per=None, sesinw=[0], secosw=[0], 
                    ddf=0, q1=0, q2=0, occ=0, Fn=None, delta=None, A_ev=0, A_db=0, cst_pars={},npl=1):
        self.rho_star = rho_star
        self.dur      = dur
        self.T0       = [T0]     if isinstance(T0,      (int,float)) else T0
        self.RpRs     = [RpRs]   if isinstance(RpRs,    (int,float)) else RpRs
        self.b        = [b]      if isinstance(b,       (int,float)) else b
        self.per      = [per]    if isinstance(per,     (int,float)) else per
        self.sesinw   = [sesinw] if isinstance(sesinw,  (int,float)) else sesinw
        self.secosw   = [secosw] if isinstance(secosw,  (int,float)) else secosw
        self.ddf      = ddf
        self.occ      = occ #*1e-6
        self.q1       = q1
        self.q2       = q2
        self.npl      = npl
        self.Fn       = Fn if Fn!=None else 0
        self.delta    = delta if delta!=None else 0
        self.A_ev     = A_ev #*1e-6
        self.A_db     = A_db #*1e-6
        self.cst_pars = cst_pars

        self.parameter_names = ['rho_star','dur', 'T0', 'RpRs', 'b', 'per', 'sesinw', 'secosw', 'ddf', 'q1', 'q2', 'occ', 'Fn','delta', 'A_ev', 'A_db','cst_pars','npl']

    def get_value(self, tarr, ss=None,grprs=0, vcont=0, Rstar=None, model_phasevar=False, custom_LCfunc=None):
        """ 
        computes the transit/occultation/phase curve model for a given set of parameters along with the baseline
        
        Parameters
        ----------
        tarr : array-like
            The timestamps of the lightcurve
        ss : SimpleNamespace
            The configuration for the supersampling
        grprs: float;
            when using fitting depth variation, the base RpRs value to which deviation of each filter is added
        vcont: float;
            contamination factor
        Rstar: float
            Stellar radius in Rsun. used to calculate light travel time correction. Default: None to not perform light travel time correction
        model_phasevar : bool
            fit the phase variation in the model. Default: False, in which case even if occultation depth is given the transit and occ are joined by a straight line
        custom_LCfunc : SimpleNamespace
            object containing the custom light curve function to be combined to the lightcurve model. Default: None
        Returns
        -------
        marr : array-like
            The lightcurve model for the given parameters
        model_components : dict
            The components of the model for each planet in a system

        """
        tt = tarr
        tt_ss   = ss.supersample(tt) if ss is not None else tt   #supersample the timestamps if ss is not None

        f_trans  = np.ones_like(tt_ss)       #transit
        f_occ    = np.ones_like(tt_ss)       #occultation
        pl_mod   = np.zeros_like(tt_ss)      #total lc model
        model_components = {}           #components of the model lc for each planet

        for n in range(self.npl):    #iterate through all planets
            # ==============================================================
            # calculate the z values for the lightcurve and put them into a z array. Then below just extract them from that array
            # --------
            # calculate eccentricity and omega
            ecc, ome = sesinw_secosw_to_ecc_omega(self.sesinw[n],self.secosw[n])
            
            # adapt the RpRs value used in the LC creation to any ddfs
            if self.ddf == 0:
                RR=np.copy(self.RpRs[n])
            else:
                RR=grprs+self.ddf    # the specified GROUP rprs + the respective ddf (deviation)
            
            # calculate the ars 
            efac1 = np.sqrt(1.-ecc**2)/(1.+ecc*np.sin(ome))
            efac2 = self.b[n]*(1.-ecc**2)/(1.+ecc*np.sin(ome))
            if self.dur is not None: 
                # ars  =  np.sqrt(((1.+RR)**2 - efac2**2 * (1.-(np.sin(self.dur*np.pi/self.per[n]))**2))/(np.sin(self.dur*np.pi/self.per[n]))**2) * efac1
                ars  = Tdur_to_aR(self.dur,self.b[n],RR,self.per[n],ecc,np.degrees(ome))
            else: ars = rho_to_aR(self.rho_star,self.per[n],ecc,np.degrees(ome))

            #light travel time correction
            if Rstar is not None:
                inc = np.arccos(self.b[n]/(ars*(1-ecc**2)/(1+ecc*np.sin(ome))))  #inclination in radians
                tt_ss = light_travel_time_correction(tt_ss, self.T0[n],ars,self.per[n],inc,Rstar,ecc,ome)

            # if replacing the LC model with a custom function
            if custom_LCfunc!=None and custom_LCfunc.replace_LCmodel:
                LC_pars = dict(Duration=self.dur, rho_star=self.rho_star,RpRs=RR,Impact_para=self.b[n],T_0=self.T0[n],Period=self.per[n],Eccentricity=ecc,
                                omega=ome*180/np.pi,q1=self.q1,q2=self.q2,D_occ=self.occ,Fn=self.Fn,ph_off=self.delta,A_ev=self.A_ev,A_db=self.A_db)
                lc_mod = custom_LCfunc.func(tt_ss,**self.cst_pars,extra_args=custom_LCfunc.extra_args,LC_pars=LC_pars)
            
            else:
                EA_lc, TA_lc = get_anomaly(tt_ss, self.T0[n], self.per[n], ecc, ome)
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

                mm0[ph_transit],m0[ph_transit] = occultquad(z[ph_transit],u1,u2,abs(RR),npo_transit)   # mm0 is the transit model
                
                if RR < 0: mm0[ph_transit] = 1-mm0[ph_transit]+1   #allow negative depths
                
                #============= OCCULTATION ==========================
                Fp,Fn,A_db, A_ev = self.occ*1e-6, self.Fn*1e-6, self.A_db*1e-6, self.A_ev*1e-6

                ph_occultation  = np.where((y < 0))
                npo_occultation = len(z[ph_occultation])

                u1, u2 = 0., 0.              # no limb darkening
                mm0[ph_occultation],m0[ph_occultation] = occultquad(z[ph_occultation],u1,u2,abs(RR),npo_occultation)   # mm0 is the occultation model (transit model w/o LD)
                if len(mm0[ph_occultation]) >0: mm0[ph_occultation] = 1 + Fp*(rescale0_1(mm0[ph_occultation])-1)  #rescale the occultation model
                
                #phase angle
                ph_angle = TA_lc + (ome-np.pi/2)   
                # phase  = phase_fold(tt_ss,self.per[n],self.T0[n])
                # phi    = 2*np.pi*phase
                if model_phasevar: 
                    #sepate the transit and occultation models and add the atmospheric variation
                    f_trans[ph_transit]   = mm0[ph_transit]
                    f_occ[ph_occultation] = mm0[ph_occultation]
                    f_occ                 = rescale0_1(f_occ)  #rescale the occultation model to be between 0 and 1
                    
                    atm    = cosine_atm_variation(ph_angle, Fp, Fn, self.delta)
                    ellps  = A_ev * (1 - (np.cos(2*ph_angle)) )
                    dopp   = A_db * np.sin(ph_angle)
                    lc_mod = f_trans*(1+ellps+dopp) + f_occ*atm.pc
                else:
                    ellps  = A_ev * (1 - (np.cos(2*ph_angle)) )
                    dopp   = A_db * np.sin(ph_angle)
                    lc_mod = mm0.copy()* (1+ellps+dopp)  #add the ellipsoidal variation to the model
                
            #save the model components (rebinned to the original cadence)
            model_components[f"pl_{n+1}"] = ss.rebin_flux(lc_mod.copy()) if ss is not None else lc_mod.copy() 
            
            lc_mod -= 1         #zero baseline
            pl_mod += lc_mod    #add each planet's transit/occ model to total mod

        #correct for the contamination
        mm = pl_mod/(vcont+1) + 1
        
        # combine the lc model with custom model
        if custom_LCfunc!=None and custom_LCfunc.replace_LCmodel==False:
            cst_model = custom_LCfunc.func(tt_ss,**self.cst_pars) if custom_LCfunc.x=="time" else custom_LCfunc.func(ph_angle,**self.cst_pars)
            mm = custom_LCfunc.op_func(mm,cst_model)

        mm = ss.rebin_flux(mm) if ss is not None else mm    #rebin the full model to the original cadence
        if self.npl==1: model_components[f"pl_{n+1}"] = mm  #save the model components for the single planet system

        return mm, model_components


def basefunc_noCNM(coeff, LCdata,res,useSpline):
    # the full baseline function calculated with the coefficients given; of which some are not jumping and set to 0
    t, flux, err, col3, col4, col5, col6, col7, col8 = LCdata.values()
    ts = t - np.median(t)

    bfunc  = coeff[0] + coeff[1]*ts + coeff[2]*np.power(ts,2) + coeff[3]*np.power(ts,3) + coeff[4]*np.power(ts,4)     #time col0
    bfunc += coeff[5] *col3 + coeff[6] *np.power(col3,2)        #x col3
    bfunc += coeff[7] *col4 + coeff[8] *np.power(col4,2)        #y col4
    bfunc += coeff[9] *col5 + coeff[10]*np.power(col5,2)       #airmass col5
    bfunc += coeff[11]*col6 + coeff[12]*np.power(col6,2)  #fwhm/conta col6
    bfunc += coeff[13]*col7 + coeff[14]*np.power(col7,2)    #sky/bg col7
    bfunc += coeff[15]*col8 + coeff[16]*np.power(col8,2)    #sky/bg col8
    bfunc += coeff[17]*np.sin(ts*coeff[18]+coeff[19])   #sinusoidal 

    if isinstance(res,int) or useSpline.use==False: #if not computing baseline set spline to ones
        spl= x = np.ones_like(ts)
    else:
        kn,s_par,dim = useSpline.knots, useSpline.par,useSpline.dim   #knot_spacing,param
        if dim == 1:
            x      = np.copy(LCdata[s_par])
            if kn=='r': kn = np.ptp(x)   #range of the array
            knots  = np.arange(min(x)+kn, max(x), kn )
            srt    = np.argsort(x)
            xs, ys = x[srt], (res/bfunc)[srt]

            splfunc = LSQUnivariateSpline(xs, ys, knots, k=useSpline.deg, ext="const")
            spl = splfunc(x)     #evaluate the spline at the original x values

        if dim == 2:
            x1 = np.copy(LCdata[s_par[0]])
            x2 = np.copy(LCdata[s_par[1]])
            kn = list(kn)  
            for ii in range(2): 
                if kn[ii]=='r': kn[ii] = np.ptp(LCdata[s_par[ii]])   #fit one spline over the whole range
            knots1  = np.arange(min(x1)+kn[0], max(x1), kn[0] )
            knots2  = np.arange(min(x2)+kn[1], max(x2), kn[1] )
            ys      = (res/bfunc)

            splfunc = LSQBivariateSpline(x1, x2, ys, knots1, knots2, kx=useSpline.deg[0], ky=useSpline.deg[1])
            spl = splfunc(x1,x2,grid=False)     #evaluate the spline at the original x values

    return bfunc, spl

def para_minfunc(icoeff, ivars, mm, LCdata):
    """
    least square fit to the baseline after subtracting model transit mm

    Parameters
    ----------
    icoeff : iterable
        the coefficients of the parametric polynomial
    ivars : iterable
        the indices of the coefficients to be fitted
    mm : array
        transit model
    LCdata : dict
        data for this lc

    Returns
    -------
    residual
        residual of the fit
    """
    flux = LCdata["col1"]
    icoeff_full = np.zeros(22)
    icoeff_full[ivars] = np.copy(icoeff)
    bfunc,_,_ = basefunc_noCNM(icoeff_full, LCdata, res=0, useSpline=False)   # fit baseline without spline here
    fullmod = np.multiply(bfunc, mm)

    return (flux - fullmod)



####### radial velocity model
def RadialVelocity_Model(tt,T0,per,K,sesinw=0,secosw=0,Gamma=0,cst_pars={},npl=None,custom_RVfunc=None):
    """ 
    Model the radial velocity curve of planet(s). 
    T0, per, K, sesinw, secosw are given as lists of the same length (npl), each element corresponding to a planet.
    
    Parameters
    ----------
    tt : array
        time stamps
    T0 : float, list; 
        transit time of each planet
    per : float, list;
        period of each planet
    K : float, list;
        RV semi-amplitude of each planet
    sesinw : float, list;
        sqrt(ecc) * sin(omega)
    secosw : float, list;
        sqrt(ecc) * cos(omega)
    Gamma : float
        systemic velocity in same units as K
    npl : int
        number of planets. Default: 1
    make_outfile : bool
        write the RV curve to file. Default: False
    get_model : bool
        return the model RV curve only. Default: False

    Returns
    -------
    RV_model, model_components : array, dict respectively
        if get_model is True, the full RV model curve, and the RV model components for each planet

    RV_model+bl : array 
        the model RV curve with the baseline function added if get_model is False.

    Examples
    --------
    >>> from CONAN3.models import RadialVelocity_Model
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np   

    >>> time = np.linspace(-5,5,300)
    >>> T0 = 0
    >>> per = 2
    >>> K   = 3 #m/s

    >>> RV = RadialVelocity_Model(time, [T0],[per],[K],[0],[0])

    >>> plt.plot(time, RV)
    >>> plt.axhline(0,ls="--")
    
    """
    
    if isinstance(T0, (int,float)): T0 = [T0]
    if isinstance(per, (int,float)): per = [per]
    if isinstance(K, (int,float)): K = [K]
    if isinstance(sesinw, (int,float)): sesinw = [sesinw]
    if isinstance(secosw, (int,float)): secosw = [secosw]

    mod_RV = np.zeros(len(tt))
    npl    = len(T0)
    assert npl == len(per) or npl == len(K) or npl == len(sesinw) or npl == len(secosw),f"RadialVelocity_Model(): T0, per, K, sesinw, secosw must be lists of the same length!"

    model_components = {}   #components of the model RV curve for each planet
    for n in range(npl):

        ecc, ome = sesinw_secosw_to_ecc_omega(sesinw[n],secosw[n])
        
        if custom_RVfunc!=None and custom_RVfunc.replace_RVmodel:
            RV_pars = dict(T0=T0[n],Period=per[n],K=K[n],ecc=ecc,omega=ome)
            m_RV = custom_RVfunc.func(tt,**cst_pars,extra_args=custom_RVfunc.extra_args,RV_pars=RV_pars)
        else:
            EA_rv, TA_rv = get_anomaly(tt, T0[n], per[n], ecc, ome)    
            # get the model RV at each time stamp
            m_RV = K[n] * (np.cos(TA_rv + ome) + ecc * np.sin(ome))

            if custom_RVfunc!=None and custom_RVfunc.replace_LCmodel==False:
                cst_model = custom_RVfunc.func(tt,**cst_pars) if custom_RVfunc.x=="time" else custom_RVfunc.func(TA_rv,**cst_pars)
                m_RV      = custom_RVfunc.op_func(m_RV,cst_model)
        
        model_components[f"pl_{n+1}"] = m_RV
        mod_RV += m_RV      #add RV of each planet to the total RV

    mod_RV += Gamma #add gamma to the total RV

    return mod_RV, model_components

def para_minfuncRV(icoeff, ivars, mod_RV, RVdata):
    """
    least square fit to the baseline after subtracting model RV mod_RV

    Parameters
    ----------
    icoeff : iterable
        the coefficients of the parametric polynomial
    ivars : iterable
        the indices of the coefficients to be fitted
    mod_RV : array
        planet RV model
    RVdata : dict
        data for this RV

    Returns
    -------
    residual
        residual of the fit
    """
    
    RV = RVdata["col1"]
    icoeff_full = np.zeros(21)
    icoeff_full[ivars] = np.copy(icoeff)
    bfuncRV,_,_ = basefuncRV(icoeff_full, RVdata, res=0, useSpline=False)
    fullmod = mod_RV + bfuncRV

    return RV - fullmod

def basefuncRV(coeff, RVdata, res, useSpline):
    # the full baseline function calculated with the coefficients given; of which some are not jumping and set to 0
    t, RV, err, col3, col4, col5 = RVdata.values()
    ts = t - np.median(t)
    
    bfunc  = coeff[0]*ts   + coeff[1]*np.power(ts,2)
    bfunc += coeff[2]*col3 + coeff[3]*np.power(col3,2)
    bfunc += coeff[4]*col4 + coeff[5]*np.power(col4,2)
    bfunc += coeff[6]*col5 + coeff[7]*np.power(col5,2) 
    bfunc += coeff[8]*np.sin(coeff[9]*ts+coeff[10])

    if isinstance(res,int) or useSpline.use==False: #if not computing baseline set spline to zeros
        spl= x = np.zeros_like(ts)
    else:
        kn,s_par,dim = useSpline.knots, useSpline.par,useSpline.dim   #knot_spacing,param
        if dim == 1:
            x      = np.copy(RVdata[s_par])
            knots  = np.arange(min(x), max(x), kn )
            srt    = np.argsort(x)
            xs, ys = x[srt], (res-bfunc)[srt]

            splfunc = LSQUnivariateSpline(xs, ys, knots, k=useSpline.deg, ext="const")
            spl = splfunc(x)     #evaluate the spline at the original x values

        if dim == 2:
            x1      = np.copy(RVdata[s_par[0]])
            x2      = np.copy(RVdata[s_par[1]])
            for ii in range(2):
                if kn[ii]=='r': kn[ii] = np.ptp(RVdata[s_par[ii]])  #fit one spline over the whole range
            knots1  = np.arange(min(x1)+kn[0], max(x1), kn[0] )
            knots2  = np.arange(min(x2)+kn[0], max(x2), kn[0] )
            ys      = (res-bfunc)

            splfunc = LSQBivariateSpline(x1, x2, ys, knots1, knots2, kx=useSpline.deg[0], ky=useSpline.deg[1])
            spl = splfunc(x1,x2,grid=False)     #evaluate the spline at the original x values

    return bfunc,spl    


