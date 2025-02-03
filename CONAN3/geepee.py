import numpy as np
from celerite import terms
import george, celerite, spleaf
import spleaf.term as term


class celerite_cosine(terms.Term):
    parameter_names = ("log_a", "log_P")

    def get_real_coefficients(self, params):
        log_a, log_nu = params
        return (
            0.0, 0.0,
        )

    def get_complex_coefficients(self, params):
        log_a, log_nu = params
        return (
            np.exp(log_a), 0.0,
            0.0, np.exp(log_nu),
        )
    

def spleaf_cosine(a, b, la, nu):
    """
    Cosine kernel with period P and amplitude sig
    """
    return term.QuasiperiodicKernel(a,0,0,nu)



class gp_params_convert:
    """
    object to convert gp amplitude and lengthscale to required value for different kernels
    """
    def __init__(self):
        self._allowed_kernels = dict(   ge_ = ["ge_mat32","ge_mat52","ge_exp","ge_cos","ge_expsq"],
                                        ce_ = ["ce_mat32","ce_exp","ce_cos","ce_sho"],
                                        sp_ = ["sp_mat32","sp_mat52","sp_exp","sp_cos","sp_expsq","sp_sho"]
                                    )
        
    def get_values(self, kernels, data, pars,fixed_arg=None):
        """
        transform pars into required values for given kernels.
        
        Parameters
        -----------
        kernels: list,str
            kernel for which parameter transformation is performed. Must be one of ["any_george","sho","mat32","real"]
        data: str,
            one of ["lc","rv"]
        pars: iterable,
            parameters (amplitude,lengthscale) for each kernel in kernels.
        fixed_arg: float,
            fixed argument for the kernels with more than 2 pars. e.g Q for SHO kernel
            
        Returns:
        --------
        log_pars: iterable,
            log parameters to be used to set parameter vector of the gp object.
            
        """
        assert data in ["lc","rv"],f'data can only be one of ["lc","rv"]'
        if isinstance(kernels, str): kernels= [kernels]
            
        conv_pars = []
        for i,kern in enumerate(kernels):
            assert kern[:3] in ["ge_","ce_","sp_"], f'gp_params_convert(): kernel must start with "ge_","ce_" or "sp_" but "{kern}" given'
            assert kern in self._allowed_kernels[kern[:3]], f'gp_params_convert(): `{kern[:2]}` kernel to convert must be one of {self._allowed_kernels[kern[:3]]} but "{kern}" given'

            # call class function with the name kern
            if kern=="ce_sho":
                p = self.__getattribute__(kern)(data,pars[i*2],pars[i*2+1], Q=fixed_arg)
            else:
                p = self.__getattribute__(kern)(data,pars[i*2],pars[i*2+1])
            conv_pars.append(p)
            
        return np.concatenate(conv_pars)
            
        
    def any_george(self, data, amplitude,lengthscale):
        """
        simple conversion where amplitude corresponds to the standard deviation of the process
        """        
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        

        log_var    = np.log(amplitude**2)
        log_metric = np.log(lengthscale)
        return log_var, log_metric
    
    #spleaf kernels
    def sp_expsq(self, data, amplitude, lengthscale):
        """
        exponential sine kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        sig        = amplitude
        rho        = lengthscale
        return sig, rho
    
    def sp_exp(self, data, amplitude, lengthscale):
        """
        exponential kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
    
        a          = amplitude**2  #variance
        la         = 1/lengthscale
        return a, la
    
    def sp_sho(self, data, amplitude, lengthscale):
        """
        simple harmonic oscillator kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        sig        = amplitude
        P0         = lengthscale
        return sig, P0
    
    def sp_mat32(self, data, amplitude, lengthscale):
        """
        Matern32 kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        sig        = amplitude
        rho        = lengthscale
        return sig, rho
    
    def sp_mat52(self, data, amplitude, lengthscale):
        """
        Matern52 kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
                
        sig        = amplitude
        rho        = lengthscale
        return sig, rho
    
    def sp_cos(self, data, amplitude, lengthscale):
        """
        Cosine kernel built from the quasiperiodic kernel with b and la set to 0
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
                
        variance   = amplitude**2
        nu         = 2*np.pi/lengthscale
        return variance, 0, 0, nu  
        
    #celerite kernels  
    def ce_sho(self, data, amplitude, lengthscale, Q=1/np.sqrt(2)):
        """
        amplitude: the standard deviation of the process
        lengthscale: the undamped period of the oscillator

        for quality factor Q > 1/2, the characteristic oscillation freq(or period) is not equal to 
        the freq(period) of the undamped oscillator, ω0
        
        see transformation here: https://celerite2.readthedocs.io/en/latest/api/python/#celerite2.terms.SHOTerm
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        w0 = 2*np.pi/lengthscale
        S0 = amplitude**2/(w0*Q)
        log_S0, log_w0 = np.log(S0), np.log(w0)
        return log_S0, log_w0

    def ce_cos(self, data, amplitude, lengthscale):
        """
        CosineKernel implementation in celerite
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var   = np.log(amplitude**2)
        log_nu    = np.log(2*np.pi/lengthscale)
        return log_var, log_nu

    def ce_exp(self, data, amplitude, lengthscale):
        """
        really an exponential kernel like in George
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        c     = 1/lengthscale
        log_c = np.log(c)
        log_a = np.log(amplitude**2)     #log_variance
        return log_a, log_c
    
    def ce_mat32(self, data, amplitude, lengthscale):
        """
        celerite mat32
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        log_sigma  = np.log(amplitude)
        rho        = lengthscale
        log_rho    = np.log(rho)
        return log_sigma, log_rho
    
    #george kernels
    def ge_mat32(self, data, amplitude, lengthscale):
        """
        George mat32
        """
        
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_cos(self, data, amplitude, lengthscale):
        """
        George CosineKernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        log_period = np.log(lengthscale)
        return log_var, log_period

    def ge_mat52(self, data, amplitude, lengthscale):
        """
        George mat52
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_expsq(self, data, amplitude, lengthscale):
        """
        George expsq
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_exp(self, data, amplitude, lengthscale):
        """
        George exp
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_cos(self, data, amplitude, lengthscale):
        """
        George cosine
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        period     = lengthscale
        log_period = np.log(period)
        return log_var, log_period
    
    def __repr__(self):
        return 'object to convert gp amplitude and lengthscale to required value for different kernels'
        

#possible kernels
george_kernels = dict(  mat32 = george.kernels.Matern32Kernel, 
                        mat52 = george.kernels.Matern52Kernel,
                        exp   = george.kernels.ExpKernel,
                        cos   = george.kernels.CosineKernel,
                        expsq = george.kernels.ExpSquaredKernel)

celerite_kernels = dict(mat32 = celerite.terms.Matern32Term,
                        exp   = celerite.terms.RealTerm,
                        cos   = celerite_cosine,
                        sho   = celerite.terms.SHOTerm)
                        

spleaf_kernels = dict(  mat32  = term.Matern32Kernel,
                        mat52  = term.Matern52Kernel,
                        exp    = term.ExponentialKernel,
                        cos    = spleaf_cosine,
                        sho    = term.SHOKernel,
                        expsq  = term.ESKernel,    #exp_sine kernel
                        )
