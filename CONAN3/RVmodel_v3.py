import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
from scipy.interpolate import LSQUnivariateSpline,LSQBivariateSpline
from types import SimpleNamespace

#plt.ion()

def get_RVmod(tt,T0,per,K_in,sesinw=0,secosw=0,Gamma_in=0,params=None,RVmes=None,RVerr=None,bis=None,fwhm=None,contra=None,
                nfilt=None,baseLSQ=None,inmcmc=None,nddf=None,nocc=None,nRV=None,nphot=None,j=None,RVnames=None,bvarsRV=None,gammaind=None,
                useSpline=None,npl=None,planet_only=False):
    """ 
    Model the radial velocity curve of planet(s). 
    T0, per, K_in, sesinw, secosw are given as lists of the same length (npl), each element corresponding to a planet.
    
    Parameters
    ----------
    tt : array
        time stamps
    T0 : float, list; 
        transit time of each planet
    per : float, list;
        period of each planet
    K_in : float, list;
        RV semi-amplitude of each planet
    sesinw : float, list;
        sqrt(ecc) * sin(omega)
    secosw : float, list;
        sqrt(ecc) * cos(omega)
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
    >>> from CONAN3.RVmodel_v3 import get_RVmod
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np   

    >>> time = np.linspace(-5,5,300)
    >>> T0 = 0
    >>> per = 2
    >>> K   = 3 #m/s

    >>> RV = get_RVmod(time, [T0],[per],[K],[0],[0], planet_only=True)

    >>> plt.plot(time, RV)
    >>> plt.axhline(0,ls="--")
    
    """
    
    if isinstance(T0, (int,float)): T0         = [T0]
    if isinstance(per, (int,float)): per       = [per]
    if isinstance(K_in, (int,float)): K_in     = [K_in]
    if isinstance(sesinw, (int,float)): sesinw = [sesinw]
    if isinstance(secosw, (int,float)): secosw = [secosw]

    mod_RV = np.zeros(len(tt))
    npl    = len(T0)
    assert npl == len(per) or npl == len(K_in) or npl == len(sesinw) or npl == len(secosw),f"get_RVmod(): T0, per, K_in, sesinw, secosw must be lists of the same length!"

    model_components = {}   #components of the model RV curve for each planet
    for n in range(npl):

        ecc = sesinw[n]**2 + secosw[n]**2

        if (ecc >= 0.99):
            ecc = 0.99
            if (secosw[n]/np.sqrt(ecc) < 1.):
                ome2 = np.arccos(secosw[n]/np.sqrt(ecc))
                # print(ome2)
            else:
                ome2 = 0   
                # print('ome2 000')
            sesinw[n] = np.sqrt(ecc)*np.sin(ome2)
            # print('here')
        
        if (ecc>0.00001):
            ome = np.arctan(np.abs(sesinw[n]/secosw[n]))  #DA LIEGT DER HUND BEGRABEN!!! TANGENS IST KEIN ISOMORPHISMUS!!!
            if (ome<0):
                ome = ome + 2.*np.pi
            if (sesinw[n]>0 and secosw[n]<0):
                ome = np.pi - ome

            if (sesinw[n]<0 and secosw[n]<0):
                ome = np.pi + ome

            if (sesinw[n]<0 and secosw[n]>0):
                ome = 2.*np.pi - ome
                
        else:
            ome=0.
            ecc=0.
        
        # calculate the ars 
        # efac1 = np.sqrt(1.-ecc**2)/(1.+ecc*np.sin(ome))
        # efac2 = bb[n]*(1.-ecc**2)/(1.+ecc*np.sin(ome))
        # ars   = np.sqrt(((1.+RpRs[n])**2 - efac2**2 * (1.-(np.sin(dur[n]*np.pi/per[n]))**2))/(np.sin(dur[n]*np.pi/per[n]))**2) * efac1

        #print ars, params[1], params[2], params[3], params[4], params[11], params[27]#, params[14], params[15], params[16], params[17], params[18], params[19]
        #time.sleep(0.05) # delays for 5 seconds
    
        # calculate the true -> eccentric -> mean anomaly at transit -> perihelion time
        TA_tra = np.pi/2. - ome
        TA_tra = np.mod(TA_tra,2.*np.pi)
        EA_tra = 2.*np.arctan( np.tan(TA_tra/2.) * np.sqrt((1.-ecc)/(1.+ecc)) )
        EA_tra = np.mod(EA_tra,2.*np.pi)
        MA_tra = EA_tra - ecc * np.sin(EA_tra)
        MA_tra = np.mod(MA_tra,2.*np.pi)
        mmotio = 2.*np.pi/per[n]   # the mean motion, i.e. angular velocity [rad/day] if we had a circular orbit
        T_peri = T0[n] - MA_tra/mmotio
    
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
        m_RV = K_in[n] * (np.cos(TA_rv + ome) + ecc * np.sin(ome))
        model_components[f"pl_{n+1}"] = m_RV
        mod_RV += m_RV      #add RV of each planet to the total RV

    mod_RV += Gamma_in #add gamma to the total RV

    if planet_only:
        return mod_RV-Gamma_in, model_components

    bfstartRV= 1+7*npl + nddf + nocc + nfilt*2 + nphot+ 2*nRV + nphot*20 +j*12  #the first index in the param array that refers to a baseline function
    incoeff = list(range(bfstartRV,bfstartRV+12))  # the indices for the coefficients for the base function        

    ts = tt-np.median(tt)

    if (baseLSQ == 'y'):
        #get the indices of the variable baseline parameters via bvar (0 for fixed, 1 for variable)
        ivars = np.copy(bvarsRV[j][0])

        incoeff=np.array(incoeff)
        coeffstart = np.copy(params[incoeff[ivars]]) 
        if len(ivars) > 0:
            icoeff,dump = scipy.optimize.leastsq(para_minfuncRV, coeffstart, args=(ivars, mod_RV, RVmes, ts, bis, fwhm, contra))
            coeff = np.copy(params[incoeff])   # the full coefficients -- set to those defined in params (in case any are fixed non-zero)
            coeff[ivars] = np.copy(icoeff)     # and the variable ones are set to the result from the minimization
        else:
            coeff = np.copy(params[incoeff])

    else:        
        coeff = np.copy(params[incoeff])   # the coefficients for the base function
    
    bfuncRV,spl_comp=basefuncRV(coeff, ts, bis, fwhm, contra,RVmes-mod_RV ,useSpline)

    mod_RVbl = mod_RV + bfuncRV

    rv_result = SimpleNamespace(planet_RV=mod_RV-Gamma_in, full_RVmod=mod_RVbl, RV_bl=bfuncRV, spline=spl_comp ) 

    return rv_result

    # indsort = np.unravel_index(np.argsort(TA_rv, axis=None), TA_rv.shape) 


def para_minfuncRV(icoeff, ivars, mod_RV, RVmes, ts, bis, fwhm, contra):
    icoeff_full = np.zeros(21)
    icoeff_full[ivars] = np.copy(icoeff)
    return (RVmes - (mod_RV + basefuncRV(icoeff_full, ts, bis, fwhm, contra,0,False)) )      

def basefuncRV(coeff, ts, col3, col4, col5,res, useSpline):
    # the full baseline function calculated with the coefficients given; of which some are not jumping and set to 0
    DA = locals().copy()
    DA["col0"] = ts
    _ = [DA.pop(item) for item in ["coeff", "ts", "res", "useSpline"]]
    
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
            x      = np.copy(DA[s_par])
            knots  = np.arange(min(x), max(x), kn )
            srt    = np.argsort(x)
            xs, ys = x[srt], (res-bfunc)[srt]

            splfunc = LSQUnivariateSpline(xs, ys, knots, k=useSpline.deg, ext="const")
            spl = splfunc(x)     #evaluate the spline at the original x values
            # spl = spl/np.median(spl)  #center spline around 1 so as not to interfere with offset of baseline function
        if dim == 2:
            x1      = np.copy(DA[s_par[0]])
            x2      = np.copy(DA[s_par[1]])
            x       = x1 #np.vstack([x1,x2]).T
            knots1  = np.arange(min(x1)+kn, max(x1), kn )
            knots2  = np.arange(min(x2)+kn, max(x2), kn )
            ys      = (res-bfunc)

            splfunc = LSQBivariateSpline(x1, x2, ys, knots1, knots2, kx=useSpline.deg[0], ky=useSpline.deg[1])
            spl = splfunc(x1,x2,grid=False)     #evaluate the spline at the original x values

    return bfunc+spl,spl    
