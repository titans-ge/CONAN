import numpy as np
from celerite import terms
import george, celerite, spleaf
import spleaf.term as term
from types import SimpleNamespace as SN

gp_h3h4names = SN(  h3 = {"sho":"Q", "exps2":"η", "rquad":"α",  "qp":"η",  "qp_mp":"η", "qp_sc":"b", "qp_ce":"C"},
                    h4 = {"qp":"P",  "qp_sc":"P",  "qp_ce":"P",  "qp_mp":"P"})

#to introduce new GP kernel into CONAN
# - simply add the kernel to the dictionary of possible kernels below.
#   with a shortcut name for calling the kernel in the code.
#   the kernel should be callable directly from the gp package, or if not, define a class/function for the kernel
# - add the conversion function to the gp_params_convert class
#   the conversion function should take in the data type (lc or rv), amplitude, lengthscale, h3 and h4
#   and return the parameters in the order required by the kernel's set_param function


# new celerite kernels
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
    

# spleaf kernel translations    
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
    object to convert gp parameters to required values for different kernels
    """
    def __init__(self):
        self._allowed_kernels = dict(   ge_ = [f"ge_{k}" for k in george_kernels.keys()],
                                        ce_ = [f"ce_{k}" for k in celerite_kernels.keys()],
                                        sp_ = [f"sp_{k}" for k in spleaf_kernels.keys()]
                                    )
        
    def get_values(self, kernels, data, pars, gp_dim=1):
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
            
        conv_pars     = []
        alphas, betas = [], []
        for i,kern in enumerate(kernels):
            assert kern[:3] in ["ge_","ce_","sp_"], f'gp_params_convert(): kernel must start with "ge_","ce_" or "sp_" but "{kern}" given'
            assert kern in self._allowed_kernels[kern[:3]], f'gp_params_convert(): `{kern[:2]}` kernel to convert must be one of {self._allowed_kernels[kern[:3]]} but "{kern}" given'

            # call class function with the name kern
            if kern[:3]=="sp_" and gp_dim>1:
                if i == 0:
                    p,alpha,beta = self.__getattribute__(kern)(data,*pars[i*5:i*5+5])
                    conv_pars.extend([1]+p)
                    alphas.append(alpha)
                    betas.append(beta)
                else:
                    alpha = pars[i*5]*1e-6 if data=="lc" else pars[i*5]
                    beta  = pars[i*5+4]*1e-6 if data=="lc" else pars[i*5+4]
                    alphas.append(alpha)
                    betas.append(beta)
            else:
                npars = 4 if data=='lc' else 5
                p = self.__getattribute__(kern)(data,*pars[i*npars:i*npars+npars])
                conv_pars.append(p)

        if kern[:3]=="sp_" and gp_dim>1:
            return np.array(conv_pars+alphas+betas)
        else:    
            return np.concatenate(conv_pars)
            
    
    #======spleaf kernels=======
    # the kernels here are a direct function of the distance between points
    def sp_expsq(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        exponential sine kernel. spleaf ESKernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        sig        = amplitude
        rho        = lengthscale

        if h5==None or np.isnan(h5): 
            return sig, rho
        else:
            h5 = h5*1e-6 if data == "lc" else h5
            return [rho], sig, h5
    
    def sp_exp(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        spleaf ExponentialKernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
    
        a          = amplitude**2  #variance
        la         = 1/lengthscale
        
        if h5==None or np.isnan(h5): 
            return a, la
        else:
            h5 = h5*1e-6 if data == "lc" else h5
            return [la], a, h5
    
    def sp_sho(self, data, amplitude, lengthscale, h3, h4=None, h5=None):
        """
        simple harmonic oscillator kernel. spleaf SHOKernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        sig        = amplitude
        P0         = lengthscale
        Q          = h3
        
        if h5==None or np.isnan(h5): 
            return sig, P0, Q
        else:
            h5 = h5*1e-6 if data == "lc" else h5
            return [P0, Q], sig, h5

    def sp_mat32(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        Matern32 kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        sig        = amplitude
        rho        = lengthscale

        if h5==None or np.isnan(h5): 
            return sig, rho
        else:
            h5 = h5*1e-6 if data == "lc" else h5
            return [rho], sig, h5
    
    def sp_mat52(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        Matern52 kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
                
        sig        = amplitude
        rho        = lengthscale

        if h5==None or np.isnan(h5): 
            return sig, rho
        else:
            h5 = h5*1e-6 if data == "lc" else h5
            return [rho], sig, h5
            
    def sp_cos(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        Cosine kernel built from spleaf QuasiperiodicKernel, with b and la set to 0
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
                
        variance   = amplitude**2
        nu         = 2*np.pi/lengthscale

        if h5==None or np.isnan(h5):
            return variance, nu  
        else:
            h5 = h5*1e-6 if data == "lc" else h5
            return [nu], variance, h5
        
    def sp_exps2(self, data, amplitude, lengthscale, h3, h4=None, h5=None):
        """
        expsine2 kernel. derived from spleaf ESPKernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        sig        = amplitude
        P          = lengthscale
        eta        = h3

        if h5==None or np.isnan(h5):
            return sig, P, eta
        else:
            h5 = h5*1e-6 if data == "lc" else h5
            return [P, eta], sig, h5
    
    def sp_qp(self, data, amplitude, lengthscale, h3, h4, h5=None):
        """
        Exponential-sine periodic (ESP) kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        sig        = amplitude
        rho        = lengthscale
        eta        = h3
        P          = h4

        if h5==None or np.isnan(h5):      
            return sig, rho, eta, P
        else:
            h5     = h5*1e-6 if data == "lc" else h5
            return  [rho, eta, P], sig, h5

    
    def sp_qp_mp(self, data, amplitude, lengthscale, h3, h4, h5=None):
        """
        MEP kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
                
        sig        = amplitude
        rho        = lengthscale
        eta        = h3
        P          = h4

        if h5==None or np.isnan(h5):  
            return sig, rho, eta, P
        else:
            h5     = h5*1e-6 if data == "lc" else h5
            return [rho, eta, P], sig, h5
    
    def sp_qp_sc(self, data, amplitude, lengthscale, h3, h4, h5=None):  
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

        if h5==None or np.isnan(h5):
            return a, la, b, nu
        else:
            h5     = h5*1e-6 if data == "lc" else h5
            return [la,b,nu], a, h5
    
    #=======celerite kernels =======
    # the kernels here are directly a function of the distance between points, 
    # so we do not square the lengthscale
    def ce_sho(self, data, amplitude, lengthscale, h3, h4=None, h5=None):
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

    def ce_cos(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        CosineKernel implementation in celerite
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var   = np.log(amplitude**2)
        log_nu    = np.log(2*np.pi/lengthscale)
        return log_var, log_nu

    def ce_exp(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        celerite real term: same as an exponential kernel like in George
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        c     = 1/lengthscale
        log_c = np.log(c)
        log_a = np.log(amplitude**2)     #log_variance
        return log_a, log_c
    
    def ce_mat32(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        celerite mat32
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        log_sigma  = np.log(amplitude)
        rho        = lengthscale
        log_rho    = np.log(rho)
        return log_sigma, log_rho
    
    def ce_qp_ce(self, data, amplitude, lengthscale, h3, h4, h5=None):
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
    def ge_mat32(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        George mat32, stationary kernel
        """
        
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude

        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_cos(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        George CosineKernel, non-stationary kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        log_period = np.log(lengthscale)
        return log_var, log_period

    def ge_mat52(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        George mat52, stationary kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_expsq(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        George expsq, stationary kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_exp(self, data, amplitude, lengthscale, h3=None, h4=None, h5=None):
        """
        George exp, stationary kernel
        """
        if amplitude==-1: amplitude = 1
        else: amplitude  = amplitude*1e-6 if data == "lc" else amplitude
        
        log_var    = np.log(amplitude**2)
        metric     = lengthscale**2
        log_metric = np.log(metric)
        return log_var, log_metric
    
    def ge_exps2(self, data, amplitude, lengthscale, h3, h4=None, h5=None):
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

    def ge_qp(self, data, amplitude, lengthscale, h3, h4, h5=None):
        """
        George quasiperiodic. ge_qp = ge_expsq *  ge_exps2
        """
        eta, P = h3, h4 
        log_var, log_metric         = self.ge_expsq(data, amplitude, lengthscale)
        log_var2, gamma, log_period = self.ge_exps2(data, -1, P, eta)

        return log_var, log_metric, log_var2, gamma, log_period


    def ge_rquad(self, data, amplitude, lengthscale, h3, h4=None, h5=None):
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
        

class GPSaveObj:
    """
    object to save the state of a GP kernel for post-fit manipulation.
    """
    def __init__(self, fname,gp_pck, gp, x, gp_pars, ndim_gp, gp_cols, gp_cols_err, residuals, kernel=None):
        """
        Parameters
        ----------
        fname: str, optional
            The filename where the GP object is saved.
        gp_pck: str, optional
            The package used for the GP, e.g., 'ge' for George, 'ce' for celerite, 'sp' for spleaf.
        gp: object, optional
            The GP object itself
        x: array-like, optional
            The x values at which the GP is evaluated.
        gp_pars: dict
            The parameters of the GP, such as amplitude, lengthscale, h3, h4, h5
        ndim_gp: int, optional
            The number of dimensions of the GP.
        gp_cols: list, optional
            The columns of the GP data.
        gp_cols_err: list, optional
            The columns of the GP data errors.
        residuals: array-like, optional
            The residuals to which the GP is fit.
        kernel: str, optional
            The kernel used for the GP.
        """

        self.fname          = fname
        self.gp             = gp
        self.gp_pck         = gp_pck
        self.x              = x
        self.ndim_gp        = ndim_gp
        self.resid          = residuals
        self.kernel         = kernel
        self.gp_cols        = gp_cols
        self.gp_cols_err    = gp_cols_err
        self.gp_pars        = gp_pars


    def get_param_dict(self, original=False):
        """
        Get the GP parameters as a dictionary.

        Parameters
        ----------
        original: bool, optional
            If True, return the original package values. If False, return the converted values.

        Returns
        -------
        dict
            A dictionary of the GP parameters.
        """
        if original:
            if self.gp_pck in ['ge','ce']:
                return dict(self.gp.get_parameter_dict())
            else:
                return {k:v for k,v in zip(self.gp.param, self.gp.get_param())}
        else:
            return self.gp_pars

    def predict(self, x=None,return_var=False, series_id=0):
        """
        Predict the GP at the given x values.

        Parameters
        ----------
        x: array-like, optional
            The x values at which to predict the GP. If None, use the saved x values.
        return_var: bool, optional
            If True, return the variance of the GP prediction.
        series_id: int, optional
            For multiseries GP fit with Spleaf, the series ID (dimension) for the GP prediction. Default is 0.

        Returns
        -------
        array-like
            The GP prediction at the given x values.
        If return_var is True, returns a tuple of (mean, variance).
        If return_var is False, returns only the mean.
        """

        x = self.x if x is None else x

        if self.gp_pck in ['ge','ce']:
            srt_x   = np.argsort(x) if x.ndim==1 else np.argsort(x[:,0]) 
            unsrt_x = np.argsort(srt_x)  #indices to unsort the gp axis
            return self.gp.predict(self.resid, t=x[srt_x], return_var=return_var, return_cov=False)[unsrt_x]

        elif self.gp_pck == 'sp':
            if self.ndim_gp == 1:
                srt_x   = np.argsort(x)
                unsrt_x = np.argsort(srt_x)  #indices to unsort the gp axis
                return self.gp.conditional(self.resid, x[srt_x], calc_cov=return_var)[unsrt_x]
            else:
                self.gp.kernel["GP"].set_conditional_coef(series_id=series_id)
                calc_cov = "diag" if return_var else False
                return self.gp.conditional(self.resid, x, calc_cov=calc_cov)

    def __repr__(self):
        return f"GPSaveObj({self.fname}: kernel={self.kernel}, ndim={self.ndim_gp}, cols={self.gp_cols}, cols_err={self.gp_cols_err})"

    def __str__(self):
        return self.__repr__()
