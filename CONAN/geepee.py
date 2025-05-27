import numpy as np
from celerite import terms
import george, celerite, spleaf
import spleaf.term as term
from types import SimpleNamespace as SN

gp_h3h4names = SN(  h3 = {"sho":"Q", "exps2":"η", "rquad":"α",  "qp":"η",  "qp_mp":"η", "qp_sc":"b", "qp_ce":"C"},
                    h4 = {"qp":"P",  "qp_sc":"P",  "qp_ce":"P",  "qp_mp":"P"})

#to introduce new GP kernel into CONAN, simply add the kernel to the dictionary of possible kernels below.
# with a shortcut name for calling the kernel in the code.
# the kernel should be callable directly from the package, or if not, define a class/function for the kernel
# then, add the conversion function to the gp_params_convert class
# the conversion function should take in the data type (lc or rv), amplitude, lengthscale, h3 and h4
# and return the parameters in the order required by the kernel's set_param function

class celerite_QPTerm(terms.Term):
    # This is a celerite term that implements an approximate quasiperiodic kernel
    # eqn 56 of the celerite paper https://arxiv.org/pdf/1703.09710
    # implemented in the notebook of the celerite paper https://github.com/dfm/celerite/blob/master/paper/figures/rotation/rotation.ipynb
    
    parameter_names = ("log_amp", "log_timescale", "factor", "log_period")

    def get_real_coefficients(self, params):
        log_amp, log_timescale, factor, log_period = params
        f = factor
        return (
            np.exp(log_amp) * (1.0 + f) / (2.0 + f),
            np.exp(-log_timescale),
        )

    def get_complex_coefficients(self, params):
        log_amp, log_timescale, factor, log_period = params
        f = factor
        return (
            np.exp(log_amp) / (2.0 + f),
            0.0,
            np.exp(-log_timescale),
            2 * np.pi * np.exp(-log_period),
        )
        
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
    
    
def translate_param_cos(a,nu):
    return dict(a=a,b=0,la=0,nu=nu)
def spleaf_cosine(a, nu):
    """ Cosine kernel with variance a and amplitude nu """
    return term.TransformKernel(inner_kernel    = term.QuasiperiodicKernel(1,0,0,1),
                                translate_param = translate_param_cos, 
                                a = a, nu = nu)

def translate_param_exps2(sig,P,eta):
    return dict(sig=sig,P=P,rho=1e100,eta=eta)
def spleaf_exp_sine2(sig, P, eta, nharm=4):
    """expsine2 kernel from ESP kernel    """
    return term.TransformKernel(inner_kernel    = term.ESPKernel(1,1,1e100,1,nharm),
                                translate_param = translate_param_exps2, 
                                sig = sig, P = P, eta = eta)


def translate_param_qp(sig,rho,eta,P):
    return dict(sig=sig, P=P, rho=rho,eta=eta)
def spleaf_qp(sig, rho, eta, P, nharm=4):
    """ qp kernel from ESP kernel    """
    return term.TransformKernel(inner_kernel    = term.ESPKernel(1,1,1,1,nharm),
                                translate_param = translate_param_qp, 
                                sig = sig, rho=rho, eta = eta, P = P)

def spleaf_mep(sig, rho, eta, P, nharm=4):
    """ qp kernel from MEP kernel    """
    return term.TransformKernel(inner_kernel    = term.MEPKernel(1,1,1,1),
                                translate_param = translate_param_qp, 
                                sig = sig, rho=rho, eta = eta, P = P)

def translate_param_qpsc(a,la,b,nu):
    return dict(a=a,b=b,la=la,nu=nu)
def spleaf_qpsc(a, la, b, nu):
    """ quasiperiodic kernel from sin and cos    """
    return term.TransformKernel(inner_kernel    = term.QuasiperiodicKernel(1,1,1,1),
                                translate_param = translate_param_qpsc, 
                                a = a, la = la, b = b, nu = nu)


#========= possible kernels=========
george_kernels = dict(  mat32 = george.kernels.Matern32Kernel, 
                        mat52 = george.kernels.Matern52Kernel,
                        exp   = george.kernels.ExpKernel,
                        cos   = george.kernels.CosineKernel,
                        expsq = george.kernels.ExpSquaredKernel,
                        exps2 = george.kernels.ExpSine2Kernel,
                        qp    = (george.kernels.ExpSquaredKernel,george.kernels.ExpSine2Kernel),  # eq 55 (https://arxiv.org/pdf/1703.09710)
                        rquad = george.kernels.RationalQuadraticKernel)

celerite_kernels = dict(mat32 = celerite.terms.Matern32Term,
                        exp   = celerite.terms.RealTerm,
                        cos   = celerite_cosine,
                        sho   = celerite.terms.SHOTerm,
                        qp_ce = celerite_QPTerm)
                        
spleaf_kernels = dict(  mat32  = term.Matern32Kernel,
                        mat52  = term.Matern52Kernel,
                        exp    = term.ExponentialKernel,
                        cos    = spleaf_cosine,
                        sho    = term.SHOKernel,
                        expsq  = term.ESKernel,    #exp_sine kernel
                        exps2  = spleaf_exp_sine2,
                        qp     = spleaf_qp,
                        qp_sc  = term.QuasiperiodicKernel,
                        qp_mp  = term.MEPKernel)

npars_gp = dict(mat32 = 2, mat52 = 2, exp = 2, cos = 2, expsq = 2, exps2 = 3, sho = 3,
                rquad = 3, qp    = 4, qp_sc = 4, qp_mp = 4, qp_ce = 4 )


class gp_params_convert:
    """
    object to convert gp amplitude and lengthscale to required value for different kernels
    """
    def __init__(self):
        self._allowed_kernels = dict(   ge_ = [f"ge_{k}" for k in george_kernels.keys()],
                                        ce_ = [f"ce_{k}" for k in celerite_kernels.keys()],
                                        sp_ = [f"sp_{k}" for k in spleaf_kernels.keys()]
                                    )
        
    def get_values(self, kernels, data, pars):
        """
        transform pars into required values for given kernels.
        
        Parameters
        -----------
        kernels: list,str
            kernel for which parameter transformation is performed. the kernels are prepended with
            "ge_","ce_" or "sp_" to indicate the package used. e.g. ["ge_mat32","ce_exp","sp_expsq"]
        data: str,
            one of ["lc","rv"]
        pars: iterable,
            parameters (amplitude,lengthscale,h3,h4) for each kernel in kernels.

        Returns
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
            p = self.__getattribute__(kern)(data,*pars[i*4:i*4+4])
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
    
    #======spleaf kernels=======
    # the kernels here are a direct function of the distance between points
    def sp_expsq(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        exponential sine kernel. spleaf ESKernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        sig        = amplitude
        rho        = lengthscale
        return sig, rho
    
    def sp_exp(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        spleaf ExponentialKernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
    
        a          = amplitude**2  #variance
        la         = 1/lengthscale
        return a, la
    
    def sp_sho(self, data, amplitude, lengthscale, h3, h4=None):
        """
        simple harmonic oscillator kernel. spleaf SHOKernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        sig        = amplitude
        P0         = lengthscale
        Q          = h3
        return sig, P0, Q
    
    def sp_mat32(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        Matern32 kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        sig        = amplitude
        rho        = lengthscale
        return sig, rho
    
    def sp_mat52(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        Matern52 kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
                
        sig        = amplitude
        rho        = lengthscale
        return sig, rho
    
    def sp_cos(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        Cosine kernel built from spleaf QuasiperiodicKernel, with b and la set to 0
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
                
        variance   = amplitude**2
        nu         = 2*np.pi/lengthscale
        return variance, nu  
        
    def sp_exps2(self, data, amplitude, lengthscale, h3, h4=None):
        """
        expsine2 kernel. derived from spleaf ESPKernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        sig        = amplitude
        P          = lengthscale
        eta        = h3
        return sig, P, eta
    
    def sp_qp(self, data, amplitude, lengthscale, h3, h4):
        """
        Exponential-sine periodic (ESP) kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
                
        sig        = amplitude
        rho        = lengthscale
        eta        = h3
        P          = h4
        return sig, rho, eta, P
    
    def sp_qp_mp(self, data, amplitude, lengthscale, h3, h4):
        """
        MEP kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
                
        sig        = amplitude
        rho        = lengthscale
        eta        = h3
        P          = h4
        return sig, rho, eta, P
    
    def sp_qp_sc(self, data, amplitude, lengthscale, h3, h4):  
        """
        Quasiperiodic kernel with sine and cosine components
        """
        if amplitude==-1: amplitude = 1   #a
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        amplitude2 = h3     #b 
        if amplitude2==-1: amplitude2 = 1
        else: amplitude2  = amplitude2*1e-6 if data == "lc" else amplitude2
                
        a   = amplitude**2
        la  = 1/lengthscale
        b   = amplitude2**2
        nu  = 2*np.pi/h4
        return a, la, b, nu
    
    #=======celerite kernels =======
    # the kernels here are directly a function of the distance between points, 
    # so we do not square the lengthscale
    def ce_sho(self, data, amplitude, lengthscale, h3, h4=None):
        """
        amplitude: the standard deviation of the process
        lengthscale: the undamped period of the oscillator

        for quality factor Q > 1/2, the characteristic oscillation freq(or period) is not equal to 
        the freq(period) of the undamped oscillator, ω0
        
        see transformation here: https://celerite2.readthedocs.io/en/latest/api/python/#celerite2.terms.SHOTerm
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        Q = h3
        
        w0 = 2*np.pi/lengthscale
        S0 = amplitude**2/(w0*Q)
        log_S0, log_w0, log_Q = np.log(S0), np.log(w0), np.log(Q)
        return log_S0, log_Q, log_w0

    def ce_cos(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        CosineKernel implementation in celerite
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var   = np.log(amplitude**2)
        log_nu    = np.log(2*np.pi/lengthscale)
        return log_var, log_nu

    def ce_exp(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        celerite real term: same as an exponential kernel like in George
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        c     = 1/lengthscale
        log_c = np.log(c)
        log_a = np.log(amplitude**2)     #log_variance
        return log_a, log_c
    
    def ce_mat32(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        celerite mat32
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        log_sigma  = np.log(amplitude)
        rho        = lengthscale
        log_rho    = np.log(rho)
        return log_sigma, log_rho
    
    def ce_qp_ce(self, data, amplitude, lengthscale, h3, h4):
        """
        celerite approximation of quasiperiodic kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        log_B  = np.log(amplitude**2)
        log_L  = np.log(1/lengthscale)
        
        # log_C      = np.log(h3)
        C = h3
        log_P      = np.log(h4)
        
        return log_B, log_L, C, log_P,
    
    #=====george kernels=======
    # stationary kernels depend on the square of the distance between points r^2 = (x1-x2)^2
    # in this case, the metric=lengthscale^2, since it scales r^2 
    # Non-stationary kernels depend on the distance between points themselves, r
    # and the amplitude is the standard deviation of the process. but george takes in the variance, so we need to square it
    def ge_mat32(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        George mat32, stationary kernel
        """
        
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_cos(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        George CosineKernel, non-stationary kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        log_period = np.log(lengthscale)
        return log_var, log_period

    def ge_mat52(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        George mat52, stationary kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_expsq(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        George expsq, stationary kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_exp(self, data, amplitude, lengthscale, h3=None, h4=None):
        """
        George exp, stationary kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_exps2(self, data, amplitude, lengthscale, h3, h4=None):
        """
        George expsine2 kernel, non-stationary kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        eta        = h3
        log_var    = np.log(amplitude**2)
        gamma      = 1/(2*eta**2)
        period     = lengthscale
        log_period = np.log(period)
        return log_var, gamma, log_period

    def ge_qp(self, data, amplitude, lengthscale, h3, h4):
        """
        George quasiperiodic. ge_qp = ge_expsq *  ge_exps2
        """
        eta, P = h3, h4 
        log_var, log_metric         = self.ge_expsq(data, amplitude, lengthscale)
        log_var2, gamma, log_period = self.ge_exps2(data, -1, P, eta)

        return log_var, log_metric, log_var2, gamma, log_period


    def ge_rquad(self, data, amplitude, lengthscale, h3, h4=None):
        """
        George rationalquadratic, stationary kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        log_alpha  = np.log(h3)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_alpha, log_metric
    
    def __repr__(self):
        return 'object to convert gp amplitude and lengthscale to required value for different kernels'
        

