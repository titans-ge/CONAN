import numpy as np
import astropy.constants as c
import astropy.units as u
from types import SimpleNamespace
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from uncertainties.core import AffineScalarFunc
from uncertainties.umath import sin,cos, sqrt
from scipy.interpolate import interp1d
from celerite import terms
import spleaf
from numba import vectorize



def light_travel_time_correction(t,t0,aR,P,inc,Rstar,ecc=0,w=1.57079):
    '''
    Corrects the time array for light travel time effects i.e subtracts the light travel time at each time point

    Args:
        t: time array
        t0: time of transit center
        aR: semi-major axis over stellar radius
        P: orbital period
        inc: inclination in radians
        Rstar: stellar radius in solar radii
        ecc: eccentricity. default is 0
        w: argument of periastron in radians. default is pi/2

    Returns:
        tcorr: time array corrected for light travel time effects
    '''
    if ecc==0:
        # circular orbit
        ph_angle = 2*np.pi*(t-t0)/P
        d = aR*np.ones_like(t)    # distance to the planet same at all times
    else:
        orb = get_orbital_elements(t,t0,P,ecc,w)
        ph_angle = orb.phase_angle
        d   = aR * (1-ecc**2)/(1+ecc*np.cos(orb.true_anom))    # distance to the planet at each time/true anomaly

    assert inc<2*np.pi, f"inc should be in radians but {inc} gotten" 
    assert w<2*np.pi, f"w should be in radians but {w} gotten" 

    c    = 299792.458 *(60*60*24)   # speed of light in km/day
    Rsun = 695700.0                 # solar radius in km
    c_R = c/(Rstar*Rsun)
    tcorr = t - d/c_R * np.sin(inc) * (1-np.cos(ph_angle))

    return tcorr


def phase_fold(t, per, t0,phase0=-0.5):
    """Phase fold a light curve.

    Parameters:
    ----------
    t : array-like
        Time stamps.
    per : float
        Period.
    t0 : float
        Time of transit center.
    phase0 : float
        start phase of the folded data

    Returns
    -------
    phase : array-like
        Phases starting from phase0.
    """
    return ( ( ( (t-t0)/per % 1) - phase0) % 1) + phase0



def get_transit_time(t, per, t0):
    """Get the transit time within a light curve.

    Parameters:
    ----------
    t : array-like
        Time stamps.
    per : float
        Period.
    t0 : float
        Time of transit center.

    Returns
    -------
    tt : array-like
        Transit times.
    """
    T01 = t0 + per * np.floor((np.median(t) - t0)/per)
    T02 = t0 + per * np.round((np.median(t) - t0)/per)

    if t.min() <= T01 <= t.max(): # if T01 is within the data time range
        return T01
    elif t.min() <= T02 <= t.max(): # if T02 is within the data time range 
        return T02
    else: # if neither T01 nor T02 is within the data time range, select closest to data start
        T0  = np.array([T01,T02])
        return T0[np.argmin(abs(T0 - min(t)))]
    

class get_orbital_elements:
    def __init__(self, t, T0, per, ecc, omega, aR=None, inc=None, approx=False):
        """
        Calculate the mean,eccentric, and true anomaly for a given time t.

        Parameters: 
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
        aR : optional - float, None;
            semi-major axis over stellar radius
        inc : optional - float, None;
            inclination in radians.
        approx: bool;
            if to use approximate equation to calculate eccentric anomaly. Default is False.
        """
        
        self._time  = t
        self._T0    = T0
        self._per   = per
        self._ecc   = ecc
        self._omega = omega
        self._aR    = aR
        self._inc   = inc
        
        # calculate the true -> eccentric -> mean anomaly at transit -> perihelion time
        if self._ecc==0: 
            self._omega = np.pi/2.  # if circular orbit, set omega to pi/2

        ph_tra = self.get_phase("transit")
        
        self.t_peri      = self._T0  - ph_tra*self._per
        self.mean_anom   = (self._time - self.t_peri)*2*np.pi/self._per
        self.mean_anom   = self.mean_anom % (2*np.pi)
        self.ecc_anom    = self.get_EccentricAnomaly(approx=approx)
        self.true_anom   = 2.*np.arctan(np.tan(self.ecc_anom/2.) * np.sqrt((1.+self._ecc)/(1.-self._ecc)) )
        self.phase_angle = self.true_anom + self._omega-np.pi/2   #phase angle is therefore 0 at transit and pi at eclipse
        self.phase_angle = np.mod(self.phase_angle, 2*np.pi)
        
        if self._aR!=None:
            self._r = self._aR * (1-self._ecc**2)/(1+self._ecc*np.cos(self.true_anom))
    
    def get_phase(self, position):
        """
        Get the phase of the planet at the given position.

        Parameters:
        ----------
        position : str
            Position of the planet. Can be one of "transit","eclipse"
        
        Returns:
        --------
        phase : float
        """
        if position == "transit":   TA = np.pi/2 - self._omega
        elif position == "eclipse": TA = 3*np.pi/2 - self._omega

        TA    = np.mod(TA,2*np.pi)
        EA    = 2.*np.arctan( np.tan(TA/2.) * np.sqrt((1.-self._ecc)/(1.+self._ecc)) )
        EA    = np.mod(EA,2*np.pi)
        MA    = EA - self._ecc * np.sin(EA)     #in radians
        MA    = MA % (2*np.pi)
        phase = MA/(2*np.pi)
        return phase


    def get_EccentricAnomaly(self, approx=False):
        """
        Get the eccentric anomaly from the mean anomaly and eccentricity.
        By solving Kepler's equation M = E - ecc.sin(E) 

        Returns
        -------
        E : array-like
            Eccentric anomaly.
        """
        M = self.mean_anom
        ecc = self._ecc

        M = M % (2*np.pi)
        if ecc == 0:
            return M
        
        if approx:
        # source of the below equation: http://alpheratz.net/Maple/KeplerSolve/KeplerSolve.pdf
        # solution not accurate at high eccentricities (~> 0.5)
            E = M + np.sin(M)*ecc + 1./2.*np.sin(2.*M)*ecc**2 + \
                        (3./8.*np.sin(3.*M) - 1./8.*np.sin(M))*ecc**3 + \
                            (1./3.*np.sin(4.*M) - 1./6.*np.sin(2*M))*ecc**4 + \
                                (1./192*np.sin(M)-27./128.*np.sin(3.*M)+125./384.*np.sin(5*M))*ecc**5 + \
                                    (1./48.*np.sin(2.*M)+27./80.*np.sin(6.*M)-4./15.*np.sin(4.*M))*ecc**6
            # E = np.mod(E,2*np.pi)
        else:
            # from pycheops.funcs import esolve
            E = esolve(M, ecc)

        return E
    
    def get_Rsky(self,aR=None, inc=None):
        """
        Get the projected distance of the planet from the star in the orbital plane
        following the equations in Winn 2010 (https://arxiv.org/abs/1001.2010).
        Also return mask of transit and eclipse phases.

        Parameters:
        ----------
        aR : float, None;
            semi-major axis over stellar radius
        inc : float, None;
            inclination in radians

        Returns:
        --------
        rsky : array-like
            projected distance of the planet from the star in the orbital plane
        y : array-like
            coordinates of the planet in the orbital plane. y>=0 is the transit phase.

        """
        if self._aR==None:
            assert aR is not None, "aR should be provided"
            self._aR = aR
        if self._inc==None:
            assert inc is not None, "inc should be provided"
            self._inc = inc

        if not hasattr(self,"_r") or aR!=None:   # recompute only if not exists or new aR given
            self._r  = self._aR * (1-self._ecc**2)/(1+self._ecc*np.cos(self.true_anom))   #eqn 1
        
        y       = self._r * np.sin(self._omega + self.true_anom)*np.sin(self._inc)                   #eqn 4 transit = y<=0
        rsky    = self._r * np.sqrt(1-np.sin(self.true_anom+self._omega)**2*np.sin(self._inc)**2)

        return rsky, y

    def get_xyz(self,aR=None, inc=None):
        """
        Get the x,y,z coordinates of the planet in the orbital plane
        following the equations in Winn 2010 (https://arxiv.org/abs/1001.2010)
        
        Parameters:
        ----------
        aR : float
            semi-major axis over stellar radius
        inc : float
            inclination in radians

        Returns:
        -------
        x, y, z : array-like
            x, y, z coordinates of the planet in the orbital plane.
        """
        if self._aR==None:
            assert aR is not None, "aR should be provided"
            self._aR = aR
        if self._inc==None:
            assert inc is not None, "inc should be provided"
            self._inc = inc

        if not hasattr(self,"_r") or aR!=None:   # recompute only if not exists or new aR given
            self._r  = self._aR * (1-self._ecc**2)/(1+self._ecc*np.cos(self.true_anom))   #eqn 1
        
        x  = -self._r * np.cos(self._omega + self.true_anom)                          #eqn 2
        y  = -self._r * np.sin(self._omega + self.true_anom)*np.cos(self._inc)        #eqn 3
        z  =  self._r * np.sin(self._omega + self.true_anom)*np.sin(self._inc)        #eqn 4

        return x, y, z

    def get_t_eclipse(self):
        """
        Get the time of eclipse.

        Returns:
        -------
        t_eclipse : float
            Time of eclipse.
        """
        T0     = get_transit_time(self._time, self._per, self._T0)  #get transit time in the data
        ph_tra = self.get_phase("transit")
        ph_ecl = self.get_phase("eclipse")
        
        return T0 + self._per*(ph_ecl - ph_tra)

@vectorize(nopython=True)
def esolve(M, ecc):
    """
    Solve Kepler's equation M = E - ecc.sin(E), returning the eccentric anomaly.
    Algorithm is from Markley 1995, CeMDA, 63, 101 via pyAstronomy class keplerOrbit.py
    function taken 'from pycheops.funcs import esolve'
    """
    M = M % (2*np.pi)
    if ecc == 0:
        return M
    if M > np.pi:
        M = 2*np.pi - M
        flip = True
    else:
        flip = False
    alpha = (3*np.pi + 1.6*(np.pi-abs(M))/(1+ecc) )/(np.pi - 6/np.pi)
    d = 3*(1 - ecc) + alpha*ecc
    r = 3*alpha*d * (d-1+ecc)*M + M**3
    q = 2*alpha*d*(1-ecc) - M**2
    w = (abs(r) + np.sqrt(q**3 + r**2))**(2/3)
    E = (2*r*w/(w**2 + w*q + q**2) + M) / d
    f_0 = E - ecc*np.sin(E) - M
    f_1 = 1 - ecc*np.cos(E)
    f_2 = ecc*np.sin(E)
    f_3 = 1-f_1
    d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1)
    d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3**2)*f_3/6)
    E = E -f_0/(f_1 + 0.5*d_4*f_2 + d_4**2*f_3/6 - d_4**3*f_2/24)
    if flip:
        E =  2*np.pi - E
    return E


def get_Tconjunctions(t, t0, per, ecc=0,omega=1.5707963,Rstar=None,aR=None,inc=None,verbose=True):
    """Get the time of conjunctions (transit and eclipse) for the given time array.

    Parameters:
    ----------
    t : array-like
        Time stamps.
    per : float
        Period.
    t0 : float
        Time of transit center.
    ecc : float
        Eccentricity. Default is 0.
    omega : float
        Argument of periastron in radians. Default is 1.5707963.
    Rstar : float
        Stellar radius in solar radii. used to calculate delay in eclipse time due to light travel travel. Default is None to ignore the delay.
    aR : float
        Semi-major axis over stellar radius. used to calculate delay in eclipse time due to light travel travel. Default is None to ignore the delay.
    inc : float
        Inclination in radians. used to calculate delay in eclipse time due to light travel travel. Default is None to ignore the delay.

    Returns
    -------
    t_conj : SimpleNamespace
        transit : float/numpy.ndarray
            Time of transit. if multiple transits, only the time of first transit is returned
        eclipse : float/numpy.ndarray
            Time of eclipse. if multiple eclipses, only the time of first eclipse is returned
    """
    in_args = locals()
    _ = [in_args.pop(k) for k in ["t","verbose"]]

    #check if any of the keys of in_args is iterable using np.iterable
    if any([np.iterable(in_args[k]) for k in in_args.keys()]):
        len_arr = len([k for k in in_args.keys() if np.iterable(in_args[k])][0])   # len of array
        in_args = {k: np.array(in_args[k]) if np.iterable(in_args[k]) else np.full(len_arr,in_args[k]) for k in in_args.keys()}
        
        tconj = SimpleNamespace(transit=np.zeros(len_arr),eclipse=np.zeros(len_arr))
        for i in range(len_arr):
            t_vals = _get_Tconjunctions(t, in_args['t0'][i], in_args['per'][i], in_args['ecc'][i], 
                                                                    in_args['omega'][i], in_args['Rstar'][i], in_args['aR'][i], 
                                                                    in_args['inc'][i], verbose)
            p_vals = phase_fold(np.array([t_vals[0],t_vals[1]]), in_args['per'][i], in_args['t0'][i])
            tconj.transit[i],tconj.eclipse[i]       = t_vals
            tconj.transit_phase,tconj.eclipse_phase = np.round(p_vals,6)%1

        return tconj
    else: 
        tconj = SimpleNamespace(transit=0,eclipse=0)
        tconj.transit,tconj.eclipse = _get_Tconjunctions(t, t0, per, ecc, omega, Rstar, aR, inc, verbose)
        tconj.transit_phase, tconj.eclipse_phase = np.round(phase_fold(np.array([tconj.transit,tconj.eclipse]),per,t0), 6)%1
        return tconj


def _get_Tconjunctions(t, t0, per, ecc=0,omega=1.5707963,Rstar=None,aR=None,inc=None,verbose=True):
    """Get the time of conjunctions (transit and eclipse) for the given time array.

    Parameters:
    ----------
    t : array-like
        Time stamps.
    per : float
        Period.
    t0 : float
        Time of transit center.
    ecc : float
        Eccentricity. Default is 0.
    omega : float
        Argument of periastron in radians. Default is 1.5707963.
    Rstar : float
        Stellar radius in solar radii. used to calculate delay in eclipse time due to light travel travel. Default is None to ignore the delay.
    aR : float
        Semi-major axis over stellar radius. used to calculate delay in eclipse time due to light travel travel. Default is None to ignore the delay.
    inc : float
        Inclination in radians. used to calculate delay in eclipse time due to light travel travel. Default is None to ignore the delay.

    Returns
    -------
    t_conj : tuple of floats
        (transit_time, eclipse_time)
        if multiple transits/eclipses, only the time of first transit/eclipse is returned
    """

    if ecc==0: omega = np.pi/2.
    tsm      = np.linspace(t.min(),t.min()+per,max(len(t)*10,20000))  # only need to calculate for one period after start of timeseries
    orb_pars = get_orbital_elements(tsm, t0, per, ecc, omega)
    ph_angle = orb_pars.phase_angle

    #mid transit and mid eclipse are at ph_angle of 0 and np.pi respectively
    #create interpolation function to find the time at these angles, extrapolate if needed
    intpd  = interp1d(ph_angle,tsm,fill_value="extrapolate")
    t_conj = intpd([0,np.pi])
    if ecc==0: assert np.isclose(abs(t_conj[1] - t_conj[0]), per/2), "time between transit and Eclipse not equal to per/2, as required for circular orbit!. Check the code/inputs."

    #account for LTT delay in eclipse time
    if None not in [Rstar,aR,inc]:  # if Rstar,aR,inc are provided
        dt = t_conj[1] - light_travel_time_correction(t_conj[1],t0,aR,per,inc,Rstar,ecc,omega)   #time delay of mid-eclipse
        if verbose: print(f"adding light travel time delay of {24*3600*dt:.4f} secs to the eclipse time")
        t_conj[1] += dt
    return t_conj


def bin_data(t,f,err=None,statistic="mean",bins=20):
    """
    Bin data in time.

    Parameters:
    ----------
    t : array-like
        Time stamps.
    f : array-like
        Fluxes.
    err : array-like
        Flux uncertainties.
    statistic : str
        Statistic to compute in each bin. See `scipy.stats.binned_statistic`.
    bins : int or array-like
        Number of bins or bin edges. See `scipy.stats.binned_statistic`.
    
    Returns
    -------
    t_bin : array-like
        Binned time stamps.
    f_bin : array-like
        Binned fluxes.
    err_bin : array-like
        Binned flux uncertainties. Only returned if `err` is not None.
    """
    from scipy.stats import binned_statistic
    try:
        y_bin, y_binedges, _ = binned_statistic(t, f, statistic=statistic, bins=bins)
        bin_width            = y_binedges[1] - y_binedges[0]
        t_bin                = y_binedges[:-1] + bin_width/2.
        nans = np.isnan(y_bin)

        if err is not None:
            err_bin, _, _ = binned_statistic(t, err, statistic = lambda x: 1/np.sqrt(np.sum(1/x**2)), bins=bins)
            return t_bin[~nans], y_bin[~nans], err_bin[~nans]

        return t_bin[~nans], y_bin[~nans]
    except:
        return (t,f,err) if err is not None else (t,f)

def bin_data_with_gaps(t,f,e=None, binsize=0.0104, gap_threshold=2.):
    """
    # split t into chunks with gaps larger than gap_threshold*bin_size
    # then bin each chunk separately
    """
    if binsize==0 or binsize<=np.median(np.diff(t)):  # no binning, if binsize is zero or less than the median time difference
        return (t,f) if e is None else (t,f,e)
    try:
        gap = np.diff(t)
        gap = np.insert(gap,0,0)

        #split t into chunks by the gaps
        t_chunks = np.split(t, np.where(gap>gap_threshold*binsize)[0]) 
        f_chunks = np.split(f, np.where(gap>gap_threshold*binsize)[0])
        e_chunks = np.split(e, np.where(gap>gap_threshold*binsize)[0]) if e is not None else f_chunks

        t_binned, f_binned, e_binned = [],[],[]
        
        for tc,fc,ec in zip(t_chunks,f_chunks,e_chunks):
            if np.ptp(tc) < binsize: nbin = 1 #continue
            else: nbin = int(np.ptp(tc)/binsize)
            if e is not None: t_bin, f_bin, e_bin = bin_data(tc,fc,ec,statistic="mean",bins=nbin)
            else: t_bin, f_bin = bin_data(tc,fc,statistic="mean",bins=nbin)

            t_binned = np.concatenate((t_binned, t_bin))
            f_binned = np.concatenate((f_binned, f_bin))
            if e is not None: e_binned = np.concatenate((e_binned, e_bin))

        return (t_binned, f_binned, e_binned) if e is not None else (t_binned, f_binned)

    except:
        return bin_data(t,f,e,statistic="mean",bins=int(np.ptp(t)/binsize))

def outlier_clipping(x, y, yerr = None, clip=5, width=15, verbose=True, return_clipped_indices = False):

    """
    Remove outliers using a running median method. Points > clip*M.A.D are removed
    where M.A.D is the mean absolute deviation from the median in each window
    
    Parameters:
    ----------
    x: array_like;
        dependent variable.
        
    y: array_like; same shape as x
        Depedent variable. data on which to perform clipping
        
    yerr: array_like(x);
        errors on the dependent variable
        
    clip: float;
       cut off value above the median. Default is 5
    
    width: int;
        Number of points in window to use when computing the running median. Must be odd. Default is 15
        
    Returns:
    --------
    x_new, y_new, yerr_new: Each and array with the remaining points after clipping
    
    """
    from scipy.signal import medfilt

    dd = abs( medfilt(y-1, width)+1 - y)   #medfilt pads with zero, so filtering at edge is better if flux level is taken to zero(y-1)
    mad = dd.mean()
    ok= dd < clip * mad

    if verbose:
        print('\nRejected {} points more than {:0.1f} x MAD from the median'.format(sum(~ok),clip))
    
    if yerr is None:
        if return_clipped_indices:
            return x[ok], y[ok], ~ok
            
        return x[ok], y[ok]
    
    if return_clipped_indices:
        return x[ok], y[ok], yerr[ok], ~ok
    
    return x[ok], y[ok], yerr[ok]

def sesinw_secosw_to_ecc_omega(sesinw, secosw, angle_unit="radians"):
    """
    Convert sesinw and secosw to eccentricity and argument of periastron in angle_unit

    Parameters:
    -----------
    sesinw: array-like
        sqrt(ecc)*sin(omega)
    secosw: array-like
        sqrt(ecc)*cos(omega)
    angle_unit: str
        unit in which to return the argument of periastron. one of ["radians", "degrees"]. 
        Default is "radians"

    Returns:
    --------
    ecc: array-like
        eccentricity
    omega: array-like
        argument of periastron in angle_unit
    """

    assert angle_unit in ["radians","degrees"], "angle_unit should be either 'radians' or 'degrees'"
    ecc = sesinw**2 + secosw**2
    omega = np.where(ecc==0,np.pi/2, np.arctan2(sesinw,secosw))
    omega = np.where(omega<0, omega+2*np.pi, omega) + 0    # add zero to make it scalar if necessary
    if angle_unit=="degrees": omega *= 180/np.pi
    return ecc, omega

def ecc_om_par(ecc, omega, conv_2_obj=False, return_tuple=False):
    """
    This function calculates the prior values and limits for the eccentricity and omega parameters, 
    sesinw and secosw. It also converts the input values given as tuples to a SimpleNamespace object

    Parameters:
    ----------
    ecc: float, tuple, SimpleNamespace;
        eccentricity value or tuple of (mean, width) or SimpleNamespace object with the following Attributes
        to_fit: str; "y" if to be fit, "n" if not to be fit
        start_value: float; starting value
        step_size: float; step size for the MCMC
        prior: str; "p" if normal prior is set, "n" if not
        prior_mean: float; prior mean
        prior_width_lo: float; lower width of the prior
        prior_width_hi: float; upper width of the prior
        bounds_lo: float; lower bound
        bounds_hi: float; upper bound
    
    omega: float, tuple, SimpleNamespace;
        argument of periastron value (in radians) or tuple of (mean, width) or SimpleNamespace object with the  same attributes as ecc
    
    conv_2_obj: bool;
        If True, convert the input values (int/float/tuple) to a SimpleNamespace object with attributes like ecc. Default is False.

    return_tuple: bool;
        If True, return the values as a int/float or tuple of length 2/3 . Default is False.

    """
    if conv_2_obj:
        if isinstance(ecc, (int,float)):
            ecc = SimpleNamespace(to_fit="n",start_value=ecc, step_size=0, prior="n", prior_mean=ecc,
                                        prior_width_lo=0, prior_width_hi=0, bounds_lo=0, bounds_hi=1)
        if isinstance(ecc, tuple):
            if len(ecc)==2:
                ecc = SimpleNamespace(to_fit="y",start_value=ecc[0], step_size=0.1*ecc[1], prior="p", prior_mean=ecc[0],
                                        prior_width_lo=ecc[1], prior_width_hi=ecc[1], bounds_lo=0, bounds_hi=1)
            elif len(ecc)==3:
                ecc = SimpleNamespace(to_fit="y",start_value=ecc[1], step_size=0.01, prior="n", prior_mean=ecc[1],
                                        prior_width_lo=0, prior_width_hi=0, bounds_lo=ecc[0], bounds_hi=ecc[2])

        if isinstance(omega, (int,float)):
            omega = SimpleNamespace(to_fit="n",start_value=omega, step_size=0, prior="n", prior_mean=omega,
                                        prior_width_lo=0, prior_width_hi=0, bounds_lo=0, bounds_hi=2*np.pi)
        if isinstance(omega, tuple):
            if len(omega)==2:
                omega = SimpleNamespace(to_fit="y",start_value=omega[0], step_size=0.1*omega[1], prior="p", prior_mean=omega[0],
                                        prior_width_lo=omega[1], prior_width_hi=omega[1], bounds_lo=0, bounds_hi=2*np.pi)
            elif len(omega)==3:
                omega = SimpleNamespace(to_fit="y",start_value=omega[1], step_size=0.01, prior="n", prior_mean=omega[1],
                                        prior_width_lo=0, prior_width_hi=0, bounds_lo=omega[0], bounds_hi=omega[2])

    sesinw=np.sqrt(ecc.start_value)*np.sin(omega.start_value)     # starting value
    secosw=np.sqrt(ecc.start_value)*np.cos(omega.start_value)     # starting value

    sinw_bounds = np.sin(np.linspace(omega.bounds_lo,omega.bounds_hi,1000))     #limits
    sesinw_bounds_lo, sesinw_bounds_hi = np.sqrt(ecc.bounds_hi)*np.nanmin(sinw_bounds), np.sqrt(ecc.bounds_hi)*np.nanmax(sinw_bounds)

    cosw_bounds = np.cos(np.linspace(omega.bounds_lo,omega.bounds_hi,1000))     #limits
    secosw_bounds_lo, secosw_bounds_hi = np.sqrt(ecc.bounds_hi)*np.nanmin(cosw_bounds), np.sqrt(ecc.bounds_hi)*np.nanmax(cosw_bounds)

    if ecc.prior_width_lo!=0. and omega.prior_width_lo!=0.:   # if an eccentricity and omega prior is set
        sesinw_prior = sqrt(ufloat(ecc.prior_mean,ecc.prior_width_lo)) * sin(ufloat(omega.prior_mean,omega.prior_width_lo))     # the prior value
        secosw_prior = sqrt(ufloat(ecc.prior_mean,ecc.prior_width_lo)) * cos(ufloat(omega.prior_mean,omega.prior_width_lo))     # the prior value
        sesinw_prior_mean, sesinw_prior_width = sesinw_prior.n, sesinw_prior.s
        secosw_prior_mean, secosw_prior_width = secosw_prior.n, secosw_prior.s

    if (ecc.prior_width_lo!=0.) and omega.prior_width_lo==0:   # normal ecc, uniform omega
        ecc_distr = np.random.normal(ecc.prior_mean,ecc.prior_width_lo,10000)            # generate random normal ecc distr
        w_distr   = np.random.uniform(omega.bounds_lo,omega.bounds_hi,10000)             # generate random uniform omega distr
        fc, fs    = np.sqrt(ecc_distr)*np.cos(w_distr), np.sqrt(ecc_distr)*np.sin(w_distr)
        sesinw_prior_mean, sesinw_prior_width = np.nanmean(fs), np.nanstd(fs)
        secosw_prior_mean, secosw_prior_width = np.nanmean(fc), np.nanstd(fc)

    if ecc.prior_width_lo==0 and omega.prior_width_lo!=0:   # uniform ecc, normal omega
        ecc_distr = np.random.uniform(ecc.bounds_lo,ecc.bounds_hi,10000)
        w_distr   = np.random.normal(omega.prior_mean,omega.prior_width_lo,10000)
        fc, fs    = np.sqrt(ecc_distr)*np.cos(w_distr),np.sqrt(ecc_distr)*np.sin(w_distr)
        sesinw_prior_mean, sesinw_prior_width = np.nanmean(fs), np.nanstd(fs)
        secosw_prior_mean, secosw_prior_width = np.nanmean(fc), np.nanstd(fc)

    if ecc.prior_width_lo==0 and omega.prior_width_lo==0:   # uniform ecc, uniform omega
        sesinw_prior_mean, sesinw_prior_width = 0, 0
        secosw_prior_mean, secosw_prior_width = 0, 0


    sesinw_step = 0.1*sesinw_prior_width if sesinw_prior_width>0 else 0.001 if sesinw_bounds_hi>0 else 0
    secosw_step = 0.1*secosw_prior_width if secosw_prior_width>0 else 0.001 if secosw_bounds_hi>0 else 0


    to_fit = "y" if ecc.to_fit=="y" or omega.to_fit=="y" else "n"
    pri = "p" if (ecc.prior_width_lo!=0. or omega.prior_width_lo!=0.) else "n"
    sesinw_in=[to_fit,sesinw,sesinw_step,pri,sesinw_prior_mean,sesinw_prior_width,sesinw_prior_width,sesinw_bounds_lo,sesinw_bounds_hi]
    secosw_in=[to_fit,secosw,secosw_step,pri,secosw_prior_mean,secosw_prior_width,secosw_prior_width,secosw_bounds_lo,secosw_bounds_hi]

    from ._classes import _param_obj
    sesinw_in = _param_obj(*sesinw_in)
    secosw_in = _param_obj(*secosw_in)

    if return_tuple:
        sesinw = sesinw_in.start_value if sesinw_in.to_fit=="n" else (sesinw_in.start_value, sesinw_prior_width) if sesinw_prior_width>0 else (sesinw_in.bounds_lo, sesinw_in.start_value,sesinw_in.bounds_hi)
        secosw = secosw_in.start_value if secosw_in.to_fit=="n" else (secosw_in.start_value, secosw_prior_width) if secosw_prior_width>0 else (secosw_in.bounds_lo, secosw_in.start_value,secosw_in.bounds_hi)
        return sesinw, secosw

    return sesinw_in, secosw_in


def rho_to_aR(rho, P, e=0, w=90, qm=0):
    """
    convert stellar density to semi-major axis of planet with a particular period.
    uses eqn 39 of kipping 2010 https://doi.org/10.1111/j.1365-2966.2010.16894.x to account for eccentricity

    Parameters:
    -----------
    rho: float, ufloat, array-like;
        The density of the star in g/cm^3.
    P: float, ufloat, array-like;
        The period of the planet in days.
    e: float, ufloat, array-like;
        The eccentricity of the orbit. Default is 0
    w: float, ufloat, array-like;
        The argument of periastron in degrees. Default is 90
    qm: float, ufloat, array-like;
        The mass ratio of the planet to the star. Default is 0 (Mp<<Ms)
        
    Returns:
    --------
    aR: array-like;
        The scaled semi-major axis of the planet.
    """

    G   = (c.G.to(u.cm**3/(u.g*u.second**2))).value
    Ps  = P*(u.day.to(u.second))
    w   = unp.radians(w)
    ecc_factor = (1+e*unp.sin(w))**3/(1-e**2)**(3/2)
    rho = rho*ecc_factor  #eccentricity correction  eqn 39 of kipping 2010 https://doi.org/10.1111/j.1365-2966.2010.16894.x
    aR  = ( rho*G*Ps**2 / (3*np.pi) *(1+qm)) **(1/3.)

    if np.iterable(aR):
        return unp.nominal_values(aR) if all(unp.std_devs(aR) == np.zeros_like(aR)) else aR
    else:
        return unp.nominal_values(aR).item() if unp.std_devs(aR) == 0 else aR

def aR_to_rho(P,aR,e=0,w=90,qm=0):
    """
    Compute the transit derived stellar density from the planet period and scaled semi major axis.
    uses eqn 39 of kipping 2010 https://doi.org/10.1111/j.1365-2966.2010.16894.x to account for eccentricity
    
    
    Parameters:
    -----------
    P: float, ufloat, array-like;
        The planet period in days
    aR: float, ufloat, array-like;
        The scaled semi-major axis of the planet orbit
    e: float, ufloat, array-like;
        The eccentricity of the orbit. Default is 0
    w: float, ufloat, array-like;
        The argument of periastron in degrees. Default is 90
    qm: float, ufloat, array-like;
        The mass ratio of the planet to the star. Default is 0 (Mp<<Ms)
    Returns:
    --------
    rho: array-like;
        The stellar density in g/cm^3
    """

    G  = (c.G.to(u.cm**3/(u.g*u.second**2))).value
    Ps = P*(u.day.to(u.second))
    w  = unp.radians(w)
    
    st_rho=3*np.pi*aR**3 / (G*Ps**2) * (1+qm)

    #eccentricity correction  eqn 39 of kipping 2010 https://doi.org/10.1111/j.1365-2966.2010.16894.x
    ecc_factor = (1+e*unp.sin(w))**3/(1-e**2)**(3/2)
    st_rho = st_rho/ecc_factor

    if np.iterable(st_rho):
        return unp.nominal_values(st_rho) if all(unp.std_devs(st_rho) == np.zeros_like(st_rho)) else st_rho
    else:
        return unp.nominal_values(st_rho).item() if unp.std_devs(st_rho) == 0 else st_rho
    
def inclination(b, a, e=0, w=90, tra_occ="tra"):
    """
    Function to convert impact parameter b to inclination in degrees.
        
    Parameters:
    ----------
    b: Impact parameter of the transit.
    a: Scaled semi-major axis i.e. a/R*.
    e: float;
        eccentricity of the orbit.
    w: float;
        longitude of periastron in degrees

    Returns
    --------
    inc: The inclination of the planet orbit in degrees.
    
    """
    w = unp.radians(w)
    esinw = e*unp.sin(w) if tra_occ=="tra" else -e*unp.sin(w)
    ecc_factor=(1-e**2)/(1+esinw)  
    inc = unp.degrees(unp.arccos( b / (a*ecc_factor)) )

    if np.iterable(inc):
        return unp.nominal_values(inc) if all(unp.std_devs(inc) == np.zeros_like(inc)) else inc
    else:
        return unp.nominal_values(inc).item() if unp.std_devs(inc) == 0 else inc

def impact_parameter(inc, a, e=0, w=90, tra_occ="tra"):
    """
    Function to convert inclination in degrees to  impact parameter b.
        
    Parameters:
    ----------
    inc: Inclination in degrees.
    a: Scaled semi-major axis i.e. a/R*.
    e: float;
        eccentricity of the orbit.
    w: float;
        longitude of periastron in degrees
    
    Returns
    --------
    
    b: the impact parameter
    
    """
    inc = unp.radians(inc)
    w   = unp.radians(w)
    
    esinw      = e*unp.sin(w) if tra_occ=="tra" else -e*unp.sin(w)
    ecc_factor = (1-e**2)/(1+esinw)  

    b = a*unp.cos(inc)*ecc_factor

    if np.iterable(b):
        return unp.nominal_values(b) if all(unp.std_devs(b) == np.zeros_like(b)) else b
    else:
        return unp.nominal_values(b).item() if unp.std_devs(b) == 0 else b

def k_to_Mp(k, P, Ms, i, e, Mp_unit = "star"):
    """
    Compute the mass of a planet from the rv semi amplitude following https://iopscience.iop.org/article/10.1086/529429/pdf

    Parameters:
    -----------
    k: float, ufloat, array-like;
        The RV semi-amplitude in m/s.

    P: float, ufloat, array-like;
        The period of the planet in days.
        
    Ms: float, ufloat, array-like;
        The mass of the star in solar masses.

    i: float, ufloat, array-like;
        The inclination of the orbit in degrees.

    e: float, ufloat, array-like;
        The eccentricity of the orbit.

    Mp_unit: str;
        The unit of the mass of the planet ["star","jup"]
        Default is "star" which returns the mass in units of the mass of the star.

    Returns:
    --------
    Mp: array-like;
        The mass of the planet in Jupiter masses.
    """
    G = (c.G.to(u.m**3/(u.kg*u.second**2))).value
    P = P*(u.day.to(u.second))
    Ms = Ms*c.M_sun.value
    i = np.deg2rad(i)
    Mp = k * (1-e**2)**(1/2) * (P/(2*np.pi*G))**(1/3) * (Ms**(2/3)) / np.sin(i)  
    if Mp_unit == "jup":
        return Mp/c.M_jup.value
    if Mp_unit == "star":
        return Mp/Ms

def aR_to_Tdur(aR, b, Rp, P,e=0,w=90, tra_occ="tra",total=True):
    """
    convert scaled semi-major axis to transit duration in days 
    using eqn 30 and 31 of Kipping 2010 https://doi.org/10.1111/j.1365-2966.2010.16894.x
    it is a more precise modification of eq 14,16 of Winn2010 https://arxiv.org/pdf/1001.2010.pdf

    Parameters:
    -----------
    aR: float, ufloat, array-like;
        The scaled semi-major axis of the planet.
    b: float, ufloat, array-like;
        The impact parameter.
    Rp: float, ufloat, array-like;
        planet-to-star radius ratio.
    P: float, ufloat, array-like;
        The period of the planet in days.
    e: float, ufloat, array-like;
        The eccentricity of the orbit.
    w: float, ufloat, array-like;
        The argument of periastron in degrees.
    tra_occ: str;
        select duration of transit (tra) or occultation (occ)
    total: bool;
        select total duration T14 (True) or full duration T23 (False)
    Returns:
    --------
    Tdur: array-like;
        The transit duration in days (same unit as P).
    """
    #return zeros if aR is zero
    if isinstance(aR, AffineScalarFunc):
        if aR.n==0: 
            return 0
    else:
        if any(aR == np.zeros_like(aR if np.iterable(aR) else [aR])):
            return np.zeros_like(aR) if np.iterable(aR) else 0

    w      = unp.radians(w)
    Rp     = abs(Rp)     #allow calculation for inverted transits
    esinw  = e*unp.sin(w) if tra_occ=="tra" else -e*unp.sin(w)
    rp     = Rp if total else -Rp
    
    ecc_fac = (1-e**2)/(1+esinw)
    inc     = unp.arccos(b/(aR*ecc_fac))
    sini    = unp.sin(inc)
    numer   = (1+rp)**2 - b**2
    numer   = np.where(numer < 0,np.nan, numer) + 0

    
    Tdur = P/np.pi * (ecc_fac**2/(unp.sqrt(1-e**2))) * unp.arcsin(unp.sqrt( numer )/(aR*ecc_fac*sini))
    Tdur = unp.uarray(np.round(unp.nominal_values(Tdur),8), np.round(unp.std_devs(Tdur),8))

    if np.iterable(Tdur):
        return unp.nominal_values(Tdur) if all(unp.std_devs(Tdur) == np.zeros_like(Tdur)) else Tdur
    else:
        return unp.nominal_values(Tdur).item() if unp.std_devs(Tdur) == 0 else Tdur

def ingress_duration(aR, b, Rp, P,e=0,w=90, tra_occ="tra"):
    """
    Compute the ingress duration of transit or occultation in units of P

    Parameters:
    -----------
    aR: float, ufloat, array-like;
        The scaled semi-major axis of the planet.
    b: float, ufloat, array-like;
        The impact parameter.
    Rp: float, ufloat, array-like;
        planet-to-star radius ratio.
    P: float, ufloat, array-like;
        The period of the planet in days.
    e: float, ufloat, array-like;
        The eccentricity of the orbit.
    w: float, ufloat, array-like;
        The argument of periastron in degrees.
    tra_occ: str;
        select duration of transit (tra) or occultation (occ)
        
    Returns:
    --------
    Tdur: array-like;
        The ingress duration in days (same unit as P).
    """
    T14 = aR_to_Tdur(aR, b, Rp, P,e,w, tra_occ, total=True)
    T23 = aR_to_Tdur(aR, b, Rp, P,e,w, tra_occ, total=False)
    return (T14 - T23)/2


def Tdur_to_aR(Tdur, b, Rp, P,e=0,w=90, tra_occ = "tra"):
    """
    convert transit duration to scaled semi-major axis
    using eqn 41 of Kipping 2010 https://doi.org/10.1111/j.1365-2966.2010.16894.x
    note (1+p^2) in the equation should instead be (1+p)^2

    Parameters:
    -----------
    Tdur: float, ufloat, array-like;
        The transit duration in days.
    b: float, ufloat, array-like;
        The impact parameter.
    Rp: float, ufloat, array-like;
        planet-to-star radius ratio.
    P: float, ufloat, array-like;
        The period of the planet in days.
    e: float, ufloat, array-like;
        The eccentricity of the orbit.
    w: float, ufloat, array-like;
        The argument of periastron in degrees.
        
    Returns:
    --------
    aR: array-like;
        The scaled semi-major axis of the planet.
    """
    w       = unp.radians(w)
    esinw   = e*unp.sin(w) if tra_occ=="tra" else -e*unp.sin(w)
    ecc_fac = (1-e**2)/(1+esinw)

    numer =  (1+abs(Rp))**2 - b**2
    numer = np.where(numer < 0,np.nan, numer) + 0
    denom = unp.sin( Tdur*np.pi*unp.sqrt(1-e**2)/(P*ecc_fac**2) )**2 *ecc_fac**2

    aR = unp.sqrt(numer/denom + (b/ecc_fac)**2)

    if np.iterable(aR):
        return unp.nominal_values(aR) if all(unp.std_devs(aR) == np.zeros_like(aR)) else aR
    else:
        return unp.nominal_values(aR).item() if unp.std_devs(aR) == 0 else aR

def rho_to_tdur(rho, b, Rp, P,e=0,w=90):
    """
    convert stellar density to transit duration in days https://doi.org/10.1093/mnras/stu318

    Parameters:
    -----------
    rho: float, ufloat, array-like;
        The density of the star in g/cm^3.
    b: float, ufloat, array-like;
        The impact parameter.
    Rp: float, ufloat, array-like;
        planet-to-star radius ratio.
    P: float, ufloat, array-like;
        The period of the planet in days.
    e: float, ufloat, array-like;
        The eccentricity of the orbit.
    w: float, ufloat, array-like;
        The argument of periastron in degrees.

    Returns:
    --------
    Tdur: array-like;
        The transit duration in days.
    """
    aR   = rho_to_aR(rho, P,e,w)
    Tdur = aR_to_Tdur(aR, b, Rp, P,e,w)
    return Tdur

def tdur_to_rho(Tdur, b, Rp, P,e=0,w=90):
    """
    convert transit duration to stellar density in g/cm^3 https://doi.org/10.1093/mnras/stu318

    Parameters:
    -----------
    Tdur: float, ufloat, array-like;
        The transit duration in days.
    b: float, ufloat, array-like;
        The impact parameter.
    Rp: float, ufloat, array-like;
        planet-to-star radius ratio.
    P: float, ufloat, array-like;
        The period of the planet in days.
    e: float, ufloat, array-like;
        The eccentricity of the orbit.
    w: float, ufloat, array-like;
        The argument of periastron in degrees.

    Returns:
    --------
    rho: array-like;
        The stellar density in g/cm^3.
    """
    aR = Tdur_to_aR(Tdur, b, Rp, P,e,w)
    rho = aR_to_rho(P,aR)
    return rho

def convert_rho(rho, ecc, w, conv="true2obs"):
    """
    convert true stellar density to transit derived density or vice-versa 

    Parameters:
    ----------  
    rho : float 
        stellar density, true or observed
    ecc : float
        eccentricity
    w : float
        argumenent of periastron in radians
    conv : str, optional
        whether to convert from true2obs or obs2true, by default "true2obs"

    Returns
    -------
    rho_star: float
        stellar density
    """

    assert conv in ["true2obs", "obs2true"],f'conv must be one of ["true2obs", "obs2true"]'
    
    omega = w * unp.pi/180
    phi = (1 + ecc*unp.sin(omega))**3 / (1-ecc**2)**(3/2)

    res = rho*phi if conv=="true2obs" else rho/phi

    if np.iterable(res):
        return unp.nominal_values(res) if all(unp.std_devs(res) == np.zeros_like(res)) else res
    else:
        return unp.nominal_values(res).item() if unp.std_devs(res) == 0 else res

def sinusoid(x, A=0, x0= None, P=None, n=1,trig="sin"):
    """
    Calculate the sinusoidal function y = A*sin(2*pi*(x-x0)/P) or y = A*cos(2*pi*(x-x0)/P) given the parameters.

    Parameters:
    ----------
    x : array-like
        x values
    A : float
        amplitude of the sinusoid
    x0 : float
        phase offset of the sinusoid
    P : float
        period of the sinusoid
    trig : str
        trigonometric function to use. Default is "sin". Options are "sin", "cos", "sincos"
    
    Returns
    -------
    y : array-like
        sinusoidal function
    """
    if trig is None:
        return np.zeros_like(x)

    assert trig in ["sin","cos","sincos"], "trig must be one of ['sin','cos','sincos']"
    if isinstance(A,(int,float)): A = [A]
    if trig in ["sin","cos"]: assert len(A)==n,f"len(A) must be equal to n but {A=} given while {n=}"
    elif trig=="sincos": assert len(A)==2*n,f"len(A) must be equal to 2*n but {A=} given while {n=}"


    # if x0 is [None,0]: x0 = np.nanmin(x)
    if P in [None,0]: P   = 2*np.pi     # default period is 2pi so the function is a sine function of x
    
    phi = 2*np.pi*(x-x0)/P
    sinus = 0
    trig_func = [np.sin,np.cos] if trig=="sincos" else [np.sin] if trig=="sin" else [np.cos]

    for j,fxn in enumerate(trig_func):
        for i in range(n):
            sinus += A[i+(n*j)]*1e-6 * fxn((i+1)*phi)
    return sinus


def cosine_atm_variation(phi, Fd=0, Fn=0, delta_deg=0,cosine_order=1):
    """
    Calculate the phase curve of a planet approximated by a cosine function from Fmin to Fmax
    The equation is given as F = Fmin + (Fmax-Fmin)/2*(1-cos(phi + delta)). The semi-amplitude of the atmospheric phase variation  is (Fmax-Fmin)/2,
    phi is the phase angle of the planet (true anomaly+omega-pi/2) in radians, and delta is the hotspot offset (in radians).
    Fday and Fnight are obtained as the value of F at phi=pi and 0 respectively.


    Parameters:
    ----------
    phi : array-like
        phase angle (2*pi*phase for circular orbit) or true anomaly+omega-pi/2 in radians.
    Fd : float
        Dayside flux/occultation depth
    Fn : float
        night side flux
    delta_deg : float
        hotspot offset in degrees.
    cosine_order: float/int;
        order of the cosine function. Default is 1
        
    Returns
    -------
    F : array-like
        planetary flux as a function of phase
    """
    res        = SimpleNamespace()
    res.delta  = np.deg2rad(delta_deg)
    
    res.Aatm   = (Fd - Fn)/(2*np.cos(res.delta)**cosine_order)

    res.Fmin   = Fd - res.Aatm*(1-np.cos(np.pi+res.delta)**cosine_order)
    res.pc     = res.Fmin + res.Aatm*(1-np.cos(phi+res.delta)**cosine_order)
    res.Fmax   = 2*res.Aatm + res.Fmin
    return res 

def gauss_atm_variation(phi,A=0,delta_deg=0,width_deg=90):
    width = np.deg2rad(width_deg)
    delta = np.deg2rad(delta_deg)
    
    F = np.exp(- ((phi-np.pi)+delta)**2 / (2*(width)**2))
    # Fmax = np.exp(- ((np.pi-np.pi)+delta)**2 / (2*(width)**2))
    # Fmin = np.exp(- ((0-np.pi)+delta)**2 / (2*(width)**2))
    # return A * (F - Fmin)/(Fmax-Fmin)
    return A * F
    
def reflection_atm_variation(phase, Fd=0, A=0, delta_deg=0):
    """
    Calculate the phase curve of a planet approximated by a cosine function with peak-to-peak amplitude  A=F_max-F_min.
    The equation is given as F = Fmin + A/2(1-cos(phi + delta)) where
    phi is the phase angle in radians = 2pi*phase
    delta is the hotspot offset (in radians)

    Parameters:
    ----------
    phase : array-like
        Phases.
    Fd : float
        Dayside flux/occultation depth
    A : float
        peak-to-peak amplitude
    delta_deg : float
        hotspot offset in degrees.
        
    Returns
    -------
    F : array-like
        planetary flux as a function of phase
    """
    raise NotImplementedError("This function is not yet implemented")

def rescale0_1(x):
    """Rescale an array to the range [0,1]."""
    return ((x - np.min(x))/np.ptp(x) ) if np.all(min(x) != max(x)) else x

def rescale_minus1_1(x):
    """Rescale an array to the range [-1,1]."""
    return ((x - np.min(x))/np.ptp(x) - 0.5)*2 if np.all(min(x) != max(x)) else x

def convert_LD(coeff1, coeff2,conv="q2u"):
    """ 
    convert quadratic limb darkening coefficients between different parameterizations.
    conversion is done as described in https://arxiv.org/pdf/1308.0009.pdf
    """

    if conv=="u2q":
        if coeff1==0 and coeff2==0: 
            return 0, 0
        q1 = (coeff1 + coeff2)**2
        q2 = coeff1/(2*(coeff1 + coeff2))
        return q1,q2
    elif conv=="q2u":
        u1 = 2*np.sqrt(coeff1)*coeff2
        u2 = np.sqrt(coeff1)*(1-2*coeff2)
        return u1,u2
    else:
        raise ValueError("LD conv must be either q2u or u2q")


class supersampling:
    def __init__(self, exp_time=0, supersample_factor=1):
        """
        supersample long integration timestamps and rebin the data after computation with supersampled time 
        
        Parameters:
        -----------
        supersample_factor : int;
            number of points subdividing exposure
        
        exp_time: float;
            exposure time of current data in same units as input time

        Returns:
        --------
        ss : supersampling object with attributes containing supersampled_time (t_ss) and function to rebin the dependent data back to original cadence.
        
        Example:
        --------
        >>> t = np.array([0,30,60,90])

        some function to generate data based on t
        >>> fxn = lambda t: t**2 + 5*t + 10

        #divide each 10min point in t into 30 observations
        >>> ss = supersampling(30, 10 )
        >>> ss.supersample(t)
        >>> t_supersampled = ss.t_ss

        #generate value of function at the supersampled time points
        >>> f_ss = fxn(t_supersampled)
        #then rebin f_ss back to cadence of observation t
        >>> f = ss.rebin_flux(f_ss)

        """
        self.supersample_factor = int(supersample_factor)
        self.exp_time = exp_time
        if self.supersample_factor==1: 
            self.config = "None"
        else: 
            self.config   = f"x{self.supersample_factor}"

    def supersample(self,time):
        assert isinstance(time, np.ndarray), f'time must be a numpy array and not {type(time)}'
        self.t = time
        t_offsets = np.linspace(-self.exp_time/2., self.exp_time/2., self.supersample_factor)
        self.t_ss = (t_offsets + self.t.reshape(self.t.size, 1)).flatten()
        return self.t_ss

    def rebin_flux(self, flux):
        rebinned_flux = np.mean(flux.reshape(-1,self.supersample_factor), axis=1)
        return rebinned_flux

    def __repr__(self):
        if self.exp_time != 0:
            return f"Supersampling: Exp_time:{self.exp_time*24*60:.1f}mins, SS_factor:{self.supersample_factor} --> sampling of {self.exp_time*24*60/self.supersample_factor:.1f}min"
        else:
            return f"No Supersampling. Sampling at original cadence"

def split_transits( t=None, P=None, t_ref=None, baseline_amount=0.25, input_t0s=None, flux =None,show_plot=True):
    
        """
        Function to split the transits in the data into individual transits and save them in separate files or to remove a certain amount of data points around the transits while keeping them in the original file.
        Recommended to set show_plot=True to visually ensure that transits are well separated.

        Parameters:
        -----------
        P : float;
            Orbital period in same unit as t.
        t_ref : float;
            reference time of transit - T0 from literature or visual estimate of a mid-transit time in the data 
            Used to calculate expected time of transits in the data assuming linear ephemerides.
        baseline_amount: float between 0.05 and 0.5 times the period P;
            amount of baseline data to keep before and after each transit. Default is 0.3*P, has to be between 0.05P and 0.5P.   
        input_t0s: array, list, (optional);
            split transit using these mid-transit times
        show_plot: bool;
            set true to plot the data and show split points.
        """        
        assert t is not None, "t must be provided"

        if baseline_amount == None:
            pass
        elif baseline_amount < 0.05 :
            baseline_amount = 0.05
            print("Baseline amount defaulted to minimum 0.05")
        elif baseline_amount > 0.5 :
            baseline_amount = 0.5
            print("Baseline amount defaulted to maximum 0.5")  
        
        t0s, Ps, plnum, trnum       = [], [], [],[]
        tr_times, fluxes, indz      = [], [], []
        t0_list, P_list, plnum_list = [], [], []
        npl = len(P)
        if input_t0s is None: input_t0s = [None]*npl

        for j in range(npl):
            #t0s for each planet
            if input_t0s[j] is not None: t0s.append(list(input_t0s[j]))
            else: t0s.append(get_T0s(t, t_ref[j], P[j]))    #get all transit times of a planet in this data
            if len(t0s[j]) > 0: 
                Ps.append([P[j]]*len(t0s[j]))    #period of each t0
                plnum.append([j]*len(t0s[j]) )         #planet number
                trnum.append(list(np.arange(1,len(t0s[j])+1)))
        
        srt_t0s = np.argsort(np.concatenate(t0s))    #sort t0s
        t0s     = np.concatenate(t0s)[srt_t0s]
        Ps      = np.concatenate(Ps)[srt_t0s]
        plnum   = np.concatenate(plnum)[srt_t0s]
        trnum   = np.concatenate(trnum)[srt_t0s]

        #split data into individual/planet group transits. taking points around each tmid    
        i=0
        while i < len(t0s):
            lo_cut = t0s[i]-baseline_amount*Ps[i] if baseline_amount!=None else min(t)
            hi_cut = t0s[i]+baseline_amount*Ps[i] if baseline_amount!=None else max(t)
            Phere  = [Ps[i]]
            T0here = [t0s[i]]
            plnum_here = [plnum[i]]
            if i != len(t0s)-1 and baseline_amount!=None: #check if next npl transit is close to this one, if so, include it in this chunk
                for _ in range(npl):   
                    if (i+1<=len(t0s)-1) and (Ps[i+1] not in Phere) and (hi_cut > t0s[i+1]-baseline_amount*Ps[i+1]):
                        hi_cut = t0s[i+1]+baseline_amount*Ps[i+1]
                        Phere.append(Ps[i+1])
                        T0here.append(t0s[i+1])
                        plnum_here.append(plnum[i+1])
                        i+=1
                    else:
                        break
            t0_list.append(T0here)
            P_list.append(Phere)
            plnum_list.append(plnum_here)
            tr_times.append(t[(t>=lo_cut) & (t<hi_cut)])
            indz.append( np.argwhere((t>=lo_cut) & (t<hi_cut)).reshape(-1) )
            fluxes.append(flux[indz[-1]])
            i+=1
                
        tr_edges = [(tr_t[0], tr_t[-1]) for tr_t in tr_times]    #store start and end time of each transit

        if show_plot:
            assert fluxes is not None, f"plotting requires input flux"
            plt.figure(figsize=(15,3))
            plt.plot(t,flux,".",c="gray",ms=2)
            [plt.plot(t,f,".",c="C0") for t,f in zip(tr_times,fluxes)]
            [plt.axvspan(edg[0], edg[1], alpha=0.1, color='cyan') for edg in tr_edges]
            [plt.plot(t0s[Ps == P[i]], (0.997*np.min(flux))*np.ones_like(t0s[Ps == P[i]]),"^") for i in range(npl)]
            plt.xlabel("Time (days)", fontsize=14)
            plt.title(f"Using t_ref: shaded regions={len(indz)} transit chunk(s);  triangles=expected linear ephemeris");
        
        return SimpleNamespace(t0s=list(t0s), t0_list=t0_list, plnum=list(plnum), trnum=list(trnum),plnum_list=plnum_list, P_list = P_list,
                                n_chunks=len(tr_times),tr_times=tr_times, fluxes=fluxes, tr_edges=tr_edges, indices=indz)


def get_T0s(t, t_ref, P):
    """
    get the transit times of a light curve
    """
    #if reference time t0 is not within this timeseries, find the transit time that falls around middle of the data
    if t_ref < t.min() or t.max() < t_ref:        
        tref = get_transit_time(t, P, t_ref)
    else: tref = t_ref

    nt       = int( (tref-t.min())/P )                        #how many transits behind tref is the first transit
    tr_first = tref - nt*P                                    #time of first transit in data
    tr_last  = tr_first + int((t.max() - tr_first)/P)*P        #time of last transit in data

    n_tot_tr = round((tr_last - tr_first)/P)                  #total nmumber of transits in data_range
    t0s      = [tr_first + P*n for n in range(n_tot_tr+1) ]        #expected tmid of transits in data (if no TTV)
    #remove tmid without sufficient transit data around it
    t0s      = list(filter( lambda t0: ( t[ (t<t0+0.1*P) & (t>t0-0.1*P)] ).size>5, t0s))  # reserve only expected t0s where there is data around it (0.05P on each side)
    t0s      = [t0 for t0 in t0s if (t.min()-0.5)<t0<(t.max()+0.5)] # only t0s within the data time range +/-0.5d (this allows to take t0s of partial transits)

    return t0s



def rms_estimate_LC(f):
    """
    Estimate the RMS of a light curve
    """
    return np.std(np.diff(f))/np.sqrt(2)

def jitter_estimate(f,e):
    """
    Estimate the jitter of a light curve
    """
    return np.sqrt(rms_estimate_LC(f)**2 - np.mean(e)**2)