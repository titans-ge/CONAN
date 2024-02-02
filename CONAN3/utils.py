import numpy as np
import astropy.constants as c
import astropy.units as u
from types import SimpleNamespace

def phase_fold(t, per, t0,phase0=-0.5):
    """Phase fold a light curve.

    Parameters
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
    """Get the transit time of a light curve.

    Parameters
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
    return t0 + per * np.round((np.median(t) - t0)/per)

def bin_data(t,f,err=None,statistic="mean",bins=20):
    """
    Bin data in time.

    Parameters
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
    y_bin, y_binedges, _ = binned_statistic(t, f, statistic=statistic, bins=bins)
    bin_width            = y_binedges[1] - y_binedges[0]
    t_bin                = y_binedges[:-1] + bin_width/2.
    nans = np.isnan(y_bin)

    if err is not None:
        err_bin, _, _ = binned_statistic(t, err, statistic = lambda x: 1/np.sqrt(np.sum(1/x**2)), bins=bins)
        return t_bin[~nans], y_bin[~nans], err_bin[~nans]

    return t_bin[~nans], y_bin[~nans]


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



def ecc_om_par(ecc, omega, conv_2_obj=False, return_tuple=False):
    # This function calculates the prior values and limits for the eccentricity and omega parameters

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
                                        prior_width_lo=0, prior_width_hi=0, bounds_lo=0, bounds_hi=360)
        if isinstance(omega, tuple):
            if len(omega)==2:
                omega = SimpleNamespace(to_fit="y",start_value=omega[0], step_size=0.1*omega[1], prior="p", prior_mean=omega[0],
                                        prior_width_lo=omega[1], prior_width_hi=omega[1], bounds_lo=0, bounds_hi=360)
            elif len(omega)==3:
                omega = SimpleNamespace(to_fit="y",start_value=omega[1], step_size=0.01, prior="n", prior_mean=omega[1],
                                        prior_width_lo=0, prior_width_hi=0, bounds_lo=omega[0], bounds_hi=omega[2])
        for key,val in omega.__dict__.items():   #convert to radians
            if isinstance(val, (float,int)): omega.__dict__[key] *= np.pi/180
            

    sesino=np.sqrt(ecc.start_value)*np.sin(omega.start_value)     # starting value
    sesinolo = -1.   # lower limit
    sesinoup = 1.   # upper limit
    
    dump1=np.sqrt(ecc.start_value+ecc.step_size)*np.sin(omega.start_value+omega.step_size)-np.sqrt(ecc.start_value)*np.sin(omega.start_value)
    dump2=np.sqrt(ecc.start_value-ecc.step_size)*np.sin(omega.start_value+omega.step_size)-np.sqrt(ecc.start_value)*np.sin(omega.start_value)
    dump3=np.sqrt(ecc.start_value+ecc.step_size)*np.sin(omega.start_value-omega.step_size)-np.sqrt(ecc.start_value)*np.sin(omega.start_value)
    dump4=np.sqrt(ecc.start_value-ecc.step_size)*np.sin(omega.start_value-omega.step_size)-np.sqrt(ecc.start_value)*np.sin(omega.start_value)

    sesinostep=np.nanmax(np.abs([dump1,dump2,dump3,dump4])) # the stepsize

    if (ecc.prior_width_lo!=0.):   # if an eccentricity prior is set
        edump= np.copy(ecc.prior_mean)
        eup  = np.copy(ecc.prior_width_lo)
        elo  = np.copy(ecc.prior_width_hi)
    else:
        edump= np.copy(ecc.start_value)
        eup=0.
        elo=0.

    if (omega.prior_width_lo!=0.):   # if an omega prior is set
        odump= np.copy(omega.prior_mean)
        oup  = np.copy(omega.prior_width_lo)
        olo  = np.copy(omega.prior_width_hi)
    else:
        odump= np.copy(omega.start_value)
        oup=0.
        olo=0.

    sesinop=np.sqrt(edump)*np.sin(odump)     # the prior value

    dump1=np.sqrt(edump+eup)*np.sin(odump+oup)-np.sqrt(edump)*np.sin(odump)
    dump2=np.sqrt(edump-elo)*np.sin(odump+oup)-np.sqrt(edump)*np.sin(odump)
    dump3=np.sqrt(edump+eup)*np.sin(odump-olo)-np.sqrt(edump)*np.sin(odump)
    dump4=np.sqrt(edump-elo)*np.sin(odump-olo)-np.sqrt(edump)*np.sin(odump)

    sesinoplo=np.abs(np.nanmin([dump1,dump2,dump3,dump4]))
    sesinopup=np.abs(np.nanmax([dump1,dump2,dump3,dump4]))
                                    
    secoso=np.sqrt(ecc.start_value)*np.cos(omega.start_value)
    secosolo=-1.   # lower limit
    secosoup=1.   # upper limit

    dump1=np.sqrt(ecc.start_value+ecc.step_size)*np.cos(omega.start_value+omega.step_size)-np.sqrt(ecc.start_value)*np.cos(omega.start_value)
    dump2=np.sqrt(ecc.start_value-ecc.step_size)*np.cos(omega.start_value+omega.step_size)-np.sqrt(ecc.start_value)*np.cos(omega.start_value)
    dump3=np.sqrt(ecc.start_value+ecc.step_size)*np.cos(omega.start_value-omega.step_size)-np.sqrt(ecc.start_value)*np.cos(omega.start_value)
    dump4=np.sqrt(ecc.start_value-ecc.step_size)*np.cos(omega.start_value-omega.step_size)-np.sqrt(ecc.start_value)*np.cos(omega.start_value)

    secosostep=np.nanmax(np.abs([dump1,dump2,dump3,dump4]))

    dump1=np.sqrt(edump+eup)*np.cos(odump+oup)-np.sqrt(edump)*np.cos(odump)
    dump2=np.sqrt(edump-elo)*np.cos(odump+oup)-np.sqrt(edump)*np.cos(odump)
    dump3=np.sqrt(edump+eup)*np.cos(odump-olo)-np.sqrt(edump)*np.cos(odump)
    dump4=np.sqrt(edump-elo)*np.cos(odump-olo)-np.sqrt(edump)*np.cos(odump)
                        
    secosoplo=np.abs(np.nanmin([dump1,dump2,dump3,dump4]))
    secosopup=np.abs(np.nanmax([dump1,dump2,dump3,dump4]))

    secosop=np.sqrt(edump)*np.cos(odump)     # the prior

    to_fit = "y" if ecc.to_fit=="y" or omega.to_fit=="y" else "n"
    pri    =  ecc.prior
    eos_in=[to_fit,sesino,sesinostep,pri,sesinop,sesinoplo,sesinopup,sesinolo,sesinoup]
    eoc_in=[to_fit,secoso,secosostep,pri,secosop,secosoplo,secosopup,secosolo,secosoup]

    from ._classes import _param_obj
    eos_in = _param_obj(*eos_in)
    eoc_in = _param_obj(*eoc_in)

    if return_tuple:
        eos_mean_prior_width = np.mean([eos_in.prior_width_lo,eos_in.prior_width_hi])
        eoc_mean_prior_width = np.mean([eoc_in.prior_width_lo,eoc_in.prior_width_hi])

        eos = eos_in.start_value if eos_in.to_fit=="n" else (eos_in.start_value, eos_mean_prior_width) if eos_mean_prior_width>0 else (eos_in.bounds_lo, eos_in.start_value,eos_in.bounds_hi)
        eoc = eoc_in.start_value if eoc_in.to_fit=="n" else (eoc_in.start_value, eoc_mean_prior_width) if eoc_mean_prior_width>0 else (eoc_in.bounds_lo, eoc_in.start_value,eoc_in.bounds_hi)
        return eos, eoc

    return eos_in, eoc_in



def rho_to_aR(rho, P):
    """
    convert stellar density to semi-major axis of planet with a particular period

    Parameters:
    -----------
    rho: float, ufloat, array-like;
        The density of the star in g/cm^3.
        
    P: float, ufloat, array-like;
        The period of the planet in days.
        
    Returns:
    --------
    aR: array-like;
        The scaled semi-major axis of the planet.
    """

    G = (c.G.to(u.cm**3/(u.g*u.second**2))).value
    Ps = P*(u.day.to(u.second))
    aR = ((rho*G*Ps**2)/(3*np.pi)) **(1/3.)

    return aR

def aR_to_rho(P,aR):
    """
    Compute the transit derived stellar density from the planet period and 
    scaled semi major axis
    
    
    Parameters:
    -----------
    P: float, ufloat, array-like;
        The planet period in days
    
    aR: float, ufloat, array-like;
        The scaled semi-major axis of the planet orbit
        
    Returns:
    --------
    rho: array-like;
        The stellar density in cgs units
    """

    G = (c.G.to(u.cm**3/(u.g*u.second**2))).value
    Ps = P*(u.day.to(u.second))
    
    st_rho=3*np.pi*aR**3 / (G*Ps**2) 
    return st_rho

def aR_to_Tdur(aR, b, Rp, P,e=0,w=90):
    """
    convert scaled semi-major axis to transit duration in days 
    eq 1 of https://doi.org/10.1093/mnras/stu318, eq 14,16 of https://arxiv.org/pdf/1001.2010.pdf

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
        
    Returns:
    --------
    Tdur: array-like;
        The transit duration in days.
    """
    factr =  ((1+Rp)**2 - b**2)/(aR**2-b**2)
    ecc_fac = np.sqrt(1-e**2)/(1+e*np.sin(np.deg2rad(w)))
    Tdur = (P/np.pi)*np.arcsin( np.sqrt(factr) ) * ecc_fac
    return Tdur


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
    aR = rho_to_aR(rho, P)
    Tdur = aR_to_Tdur(aR, b, Rp, P,e,w)
    return Tdur


def cosine_atm_variation(phase, Fd=0, A=0, delta_deg=0):
    """
    Calculate the phase curve of a planet approximated by a cosine function with peak-to-peak amplitude  A=F_max-F_min.
    The equation is given as F = Fmin + A/2(1-cos(phi + delta)) where
    phi is the phase angle in radians = 2pi*phase
    delta is the hotspot offset (in radians)

    Parameters
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
    res        = SimpleNamespace()
    res.delta  = np.deg2rad(delta_deg)
    res.phi    = 2*np.pi*phase

    res.Fmin   = Fd - A/2*(1-np.cos(np.pi+res.delta))
    res.Fnight = Fd - A * np.cos(res.delta)
    res.pc     = res.Fmin + A/2*(1-np.cos(res.phi+res.delta))
    return res    
    
def reflection_atm_variation(phase, Fd=0, A=0, delta_deg=0):
    """
    Calculate the phase curve of a planet approximated by a cosine function with peak-to-peak amplitude  A=F_max-F_min.
    The equation is given as F = Fmin + A/2(1-cos(phi + delta)) where
    phi is the phase angle in radians = 2pi*phase
    delta is the hotspot offset (in radians)

    Parameters
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
    # res        = SimpleNamespace()
    # res.delta  = np.deg2rad(delta_deg)
    # res.phi    = 2*np.pi*phase

    # res.Fmin   = Fd - A/2*(1-np.cos(np.pi+res.delta))
    # res.Fnight = Fd - A * np.cos(res.delta)
    # res.pc     = res.Fmin + A/2*(1-np.cos(res.phi+res.delta))
    # return res   

def rescale0_1(x):
    """Rescale an array to the range [0,1]."""
    return ((x - np.min(x))/np.ptp(x) ) if np.all(min(x) != max(x)) else x

def rescale_minus1_1(x):
    """Rescale an array to the range [-1,1]."""
    return ((x - np.min(x))/np.ptp(x) - 0.5)*2 if np.all(min(x) != max(x)) else x

def convert_LD(coeff1, coeff2,conv="q2u"):
    """ 
    convert 2 parameter limb darkening coefficients between different parameterizations.
    conversion is done as described in https://arxiv.org/pdf/1308.0009.pdf
    """
    assert conv in ["c2u","u2c","q2u","u2q"], "conv must be either c2u or u2c"
    if conv == "c2u":
        u1 = (coeff1 + coeff2)/3
        u2 = (coeff1 - 2.*coeff2)/3.
        return u1,u2
    elif conv=="u2c":
        c1 = 2*coeff1 + coeff2
        c2 = coeff1 - coeff2
        return c1,c2
    elif conv=="u2q":
        q1 = (coeff1 + coeff2)**2
        q2 = coeff1/(2*(coeff1 + coeff2))
        return q1,q2
    elif conv=="q2u":
        u1 = 2*np.sqrt(coeff1)*coeff2
        u2 = np.sqrt(coeff1)*(1-2*coeff2)
        return u1,u2

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
        self.supersample_factor = supersample_factor
        self.exp_time = exp_time
        self.config   = f"x{exp_time*24*60}" if exp_time !=0 else "None"

    def supersample(self,time):
        assert isinstance(time, np.ndarray), f'time must be a numpy array and not {type(time)}'
        self.t = time
        t_offsets = np.linspace(-self.exp_time/2., self.exp_time/2., self.supersample_factor)
        self.t_ss = (t_offsets + self.t.reshape(self.t.size, 1)).flatten()
        return self.t_ss

    def rebin_flux(self, flux):
        rebinned_flux = np.mean(flux.reshape(-1,self.supersample_factor), axis=1)
        return rebinned_flux


class gp_params_convert:
    """
    object to convert gp amplitude and lengthscale to required value for different kernels
    """
        
    def get_values(self, kernels, data, pars):
        """
        transform pars into required values for given kernels.
        
        Parameters:
        -----------
        kernels: list,str
            kernel for which parameter transformation is performed. Must be one of ["any_george","sho","mat32","real"]
        data: str,
            one of ["lc","rv"]
        pars: iterable,
            parameters (amplitude,lengthscale) for each kernel in kernels.
            
        Returns:
        --------
        log_pars: iterable,
            log parameters to be used to set parameter vector of the gp object.
            
        """
        assert data in ["lc","rv"],f'data can only be one of ["lc","rv"]'
        if isinstance(kernels, str): kernels= [kernels]
            
        log_pars = []
        for i,kern in enumerate(kernels):
            assert kern in ["g_mat32","g_mat52","g_expsq","g_exp","g_cos","sho","mat32","real"],  \
                f'gp_params_convert(): kernel to convert must be one of ["any_george","sho","mat32","real"] but "{kern}" given'

            # call class function with the name kern
            p = self.__getattribute__(kern)(data,pars[i*2],pars[i*2+1])
            log_pars.append(p)
            
        return np.concatenate(log_pars)
            
        
    def any_george(self, data, amplitude,lengthscale):
        """
        simple conversion where amplitude corresponds to the standard deviation of the process
        """        
        amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        log_var    = np.log(amplitude**2)
        log_metric = np.log(lengthscale)
        return log_var, log_metric
    
    
    #celerite kernels  
    def sho(self, data, amplitude, lengthscale):
        """
        amplitude: the standard deviation of the process
        lengthscale: the undamped period of the oscillator
        
        see transformation here: https://celerite2.readthedocs.io/en/latest/api/python/#celerite2.terms.SHOTerm
        """
        amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        Q  = 1/np.sqrt(2)
        w0 = 2*np.pi/lengthscale
        S0 = amplitude**2/(w0*Q)
        
        log_S0, log_w0 = np.log(S0), np.log(w0)
        return log_S0, log_w0
    
    def real(self, data, amplitude, lengthscale):
        """
        really an exponential kernel like in George
        """
        amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        c     = 1/lengthscale
        log_c = np.log(c)
        log_a = np.log(amplitude**2)     #log_variance
        return log_a, log_c
    
    def mat32(self, data, amplitude, lengthscale):
        """
        celerite mat32
        """
        amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        log_sigma  = np.log(amplitude)
        rho        = lengthscale
        log_rho    = np.log(rho)
        return log_sigma, log_rho
    

    #george kernels
    def g_mat32(self, data, amplitude, lengthscale):
        """
        George mat32
        """
        amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def g_cos(self, data, amplitude, lengthscale):
        """
        George CosineKernel
        """
        amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        log_var    = np.log(amplitude**2)
        log_period = np.log(lengthscale)
        return log_var, log_period

    def g_mat52(self, data, amplitude, lengthscale):
        """
        George mat52
        """
        amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def g_expsq(self, data, amplitude, lengthscale):
        """
        George expsq
        """
        amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        log_var    = np.log(amplitude**2)
        metric     = lengthscale
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def g_exp(self, data, amplitude, lengthscale):
        """
        George exp
        """
        amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def g_cos(self, data, amplitude, lengthscale):
        """
        George cosine
        """
        amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        log_var    = np.log(amplitude**2)
        period     = lengthscale
        log_period = np.log(period)
        return log_var, log_period
    
    def __repr__(self):
        return 'object to convert gp amplitude and lengthscale to required value for different kernels'
        
    

        