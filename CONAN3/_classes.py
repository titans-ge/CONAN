import numpy as np
import matplotlib.pyplot as plt
import corner
import pickle
from types import SimpleNamespace
import os
import matplotlib
import pandas as pd
from lmfit import minimize, Parameters, Parameter
import batman

__all__ = ["load_lightcurves", "load_rvs", "setup_fit", "__default_backend__","load_result_array"]

#helper functions
__default_backend__ = matplotlib.get_backend()
matplotlib.use(__default_backend__)

def _plot_data(obj, plot_cols, col_labels, nrow_ncols=None, figsize=None, 
                fit_order=0, model_overplot=None):
    """
    Takes a data object (containing light-curves or RVs) and plots them.
    """

    n_data = len(obj._names)
    cols = plot_cols+(1,) if len(plot_cols)==2 else plot_cols

    if n_data == 1:
        p1, p2, p3 = np.loadtxt(obj._fpath+obj._names[0], usecols=cols, unpack=True )
        if len(plot_cols)==2: p3 = None
        if figsize is None: figsize=(8,5)
        fig = plt.figure(figsize=figsize)
        plt.errorbar(p1,p2,yerr=p3, fmt=".", color="b", ecolor="gray",label=f'{obj._names[0]}')
        if model_overplot:
            plt.plot(p1,model_overplot[0][0],"r",zorder=3,label="detrend_model")
            plt.plot(p1,model_overplot[0][1],"c",zorder=3,label="tra/occ_model")

        if fit_order>0:
            pfit = np.polyfit(p1,p2,fit_order)
            srt = np.argsort(p1)
            plt.plot(p1[srt],np.polyval(pfit,p1[srt]),"r",zorder=3)
        plt.legend(fontsize=12)

    else:
        if nrow_ncols is None: 
            nrow_ncols = (int(n_data/2), 2) if n_data%2 ==0 else (int(np.ceil(n_data/3)), 3)
        if figsize is None: figsize=(14,3*nrow_ncols[0])
        fig, ax = plt.subplots(nrow_ncols[0], nrow_ncols[1], figsize=figsize)
        ax = ax.reshape(-1)

        for i, d in enumerate(obj._names):
            p1,p2,p3 = np.loadtxt(obj._fpath+d,usecols=cols, unpack=True )
            if len(plot_cols)==2: p3 = None
            ax[i].errorbar(p1,p2,yerr=p3, fmt=".", color="b", ecolor="gray",label=f'{obj._names[i]}')
            if model_overplot:
                ax[i].plot(p1,model_overplot[i][0],"r",zorder=3,label="detrend_model")
                ax[i].plot(p1,model_overplot[i][1],"c",zorder=3,label="tra/occ_model")

            if fit_order>0:
                pfit = np.polyfit(p1,p2,fit_order)
                srt = np.argsort(p1)
                ax[i].plot(p1[srt],np.polyval(pfit,p1[srt]),"r",zorder=3)
            
            ax[i].legend(fontsize=12)
    fig.suptitle(f"{col_labels[0]} against {col_labels[1]}", y=0.99, fontsize=18)

    plt.show()
    return fig

def _skip_lines(file, n):
    """
    takes a open file object and skips the reading of lines by n lines
    
    """
    for i in range(n):
        dump = file.readline()
        
def _reversed_dict(d):
    """
    reverse the order of dictionary keys
    """
    dd = dict()
    for key in reversed(list(d.keys())):
        dd[key] = d[key]
    return dd

def _raise(exception_type, msg):
    raise exception_type(msg)

def _decorr(file, T_0=None, P=None, dur=None, L=0, b=0, rp=None, q1=0, q2=0,
                 mask=False, decorr_bound=(-1,1),
                offset=None, A0=None, B0=None, A3=None, B3=None,
                A4=None, B4=None, A5=None, B5=None,
                A6=None, B6=None, A7=None, B7=None,
                A5_2=None, B5_2=None, A5_3=None,B5_3=None,
                cheops=False, return_models=False):
    """
    linear decorrelation with different columns of data file. It performs a linear model fit to the 3rd column of the file.
    It uses columns 0,3,4,5,6,7 to construct the linear trend model.
    
    Parameters:
    -----------
    file : str;
        path to data file with columns 0 to 8 (col0-col8).
    
    T_0, P, dur, L, b, rp : floats, None;
        transit/eclipse parameters of the planet. T_0, P, and dur must be in same units as the time axis (cols0) in the data file.
        if float/int, the values are held fixed. if tuple/list of len 2 implies [min,max] while len 3 implies [min,start_val,max].
        
    q1,q2 : float  (optional);
        quadratic limb darkening parameters according to kipping 2013. values are in the range (0,1).

    mask : bool ;
        if True, transits and eclipses are masked using T_0, P and dur which must be float/int.                    
        
    offset, Ai, Bi; floats [-1,1] or None;
        coefficients of linear model where offset is the intercept. they have large bounds [-1,1].
        Ai, Bi are the linear and quadratic term of the model against column i. A0*col0 + A0*col0**2 for time trend
    
    cheops : Bool;
        True, if data is from CHEOPS with col5 being the roll-angle. 
        In this case, a linear fourier model up to 3rd harmonic in roll-angle  is used for col5 
        
    return_models : Bool;
        True to return trend model and transit/eclipse model.
         
    Returns:
    -------
    result: object;
        result object from fit with several attributes such as result.bestfit, result.params, result.bic, ...
        if return_models = True, returns (trend_model, transit/eclipse model)
    """
    in_pars = locals().copy()
    _       = in_pars.pop("file")
    tr_pars = {}

    ff = np.loadtxt(file,usecols=(0,1,2,3,4,5,6,7,8))
    dict_ff = {}
    for i in range(8): dict_ff[f"cols{i}"] = ff[:,i]
    df = pd.DataFrame(dict_ff)  #pandas dataframe

    cols0_med = np.median(df["cols0"])
                          
    if mask:
        print("masking transit/eclipse phases")
        for tp in ["T_0", "P", "dur"]:
            assert isinstance(in_pars[tp], (float,int)),f"{tp} must be  float/int for masking transit/eclipses"
        #use periodicity of 0.5*P to catch both transits and eclipses
        E = np.round(( cols0_med - T_0)/(0.5*P) )
        Tc = E*(0.5*P) + T_0
        mask = abs(df["cols0"] - Tc) > 0.5*dur
        df = df[mask]
        
    for p in ["T_0", "P", "dur", "L","b","rp","q1","q2"]: tr_pars[p]= in_pars[p]    #transit/eclipse pars
    #remove non-decorr variables    
    _ = [in_pars.pop(item) for item in ["T_0", "P", "dur", "L","b","rp","q1","q2",
                                        "mask","cheops", "return_models","decorr_bound"]]

    #decorr params
    params = Parameters()
    for key in in_pars.keys():
        val  = in_pars[key] if in_pars[key] != None else 0
        vary = True if in_pars[key] != None else False
        if not cheops and key in ["A5_2", "B5_2", "A5_3","B5_3"]:
            val=0; vary = False 
        params.add(key, value=val, min=decorr_bound[0], max=decorr_bound[1], vary=vary)
    
    #transit/eclipseparameters
    tr_params = Parameters()
    for key in tr_pars.keys():
        if isinstance(tr_pars[key], (list,tuple)):
            assert len(tr_pars[key]) in [2,3],f"{key} must be float/int or tuple of length 2/3"
            if len(tr_pars[key])==3:  #uniform prior (min, start, max)
                val = tr_pars[key]
                tr_params.add(key, value=val[1], min=val[0], max=val[2], vary=True)
            if len(tr_pars[key])==2: #normal prior (mean, std)
                tr_params[key] = Parameter(key, value=tr_pars[key][0], vary=True, user_data = tr_pars[key] )
        if isinstance(tr_pars[key], (float,int)):
            tr_params.add(key, value=tr_pars[key], vary=False)
        if tr_pars[key] == None:
            val = 1e-10 if key in ["rp","P","dur"] else 0 #allows to obtain transit/eclipse with zero depth
            tr_params.add(key, value=val, vary=False)
                
    
    def transit_occ_model(tr_params):
        bt = batman.TransitParams()

        bt.per = tr_params["P"]
        bt.t0  = tr_params["T_0"]
        bt.fp  = tr_params["L"]
        bt.rp  = tr_params["rp"]
        b      = tr_params["b"]
        dur    = tr_params["dur"]
        bt.a   = np.sqrt( (1+bt.rp)**2 - b**2)/(np.pi*dur/bt.per)
        bt.inc = np.rad2deg(np.arccos(b/bt.a))
        bt.limb_dark = "quadratic"
        u1 = 2 *tr_params["q1"]**0.5 *tr_params["q2"]  #convert back to quadratic lds
        u2 = tr_params["q1"]**0.5*(1-2*tr_params["q2"])
        bt.u   = [u1,u2]
        bt.ecc, bt.w = 0,90

        bt.t_secondary = bt.t0 + 0.5*bt.per

        m_tra = batman.TransitModel(bt, df["cols0"].values,transittype="primary")
        m_ecl = batman.TransitModel(bt, df["cols0"].values,transittype="secondary")

        f_tra = m_tra.light_curve(bt)
        f_occ = (m_ecl.light_curve(bt)-bt.fp)
        model_flux = f_tra*f_occ #transit and eclipse model
        return np.array(model_flux)


    def trend_model(params):
        trend = 1 + params["offset"]       #offset
        trend += params["A0"]*(df["cols0"]-cols0_med)  + params["B0"]*(df["cols0"]-cols0_med)**2 #time trend
        trend += params["A3"]*df["cols3"]  + params["B3"]*df["cols3"]**2 #x
        trend += params["A4"]*df["cols4"]  + params["B4"]*df["cols4"]**2 #y
        trend += params["A6"]*df["cols6"]  + params["B6"]*df["cols6"]**2 #bg
        trend += params["A7"]*df["cols7"] + params["B7"]*df["cols7"]**2 #conta
        
        if cheops is False:
            trend += params["A5"]*df["cols5"]  + params["B5"]*df["cols5"]**2 
        else: #roll
            sin_col5,  cos_col5  = np.sin(np.deg2rad(df["cols5"])), np.cos(np.rad2deg(df["cols5"]))
            sin_2col5, cos_2col5 = np.sin(2*np.deg2rad(df["cols5"])), np.cos(2*np.rad2deg(df["cols5"]))
            sin_3col5, cos_3col5 = np.sin(3*np.deg2rad(df["cols5"])), np.cos(3*np.rad2deg(df["cols5"]))

            trend+= params["A5"]*sin_col5 + params["B5"]*cos_col5
            trend+= params["A5_2"]*sin_2col5 + params["B5_2"]*cos_2col5
            trend+= params["A5_3"]*sin_3col5 + params["B5_3"]*cos_3col5
        return np.array(trend)
    

    if return_models:
        return trend_model(params),transit_occ_model(tr_params)    
    
    #perform fitting 
    def chisqr(fit_params):
        flux_model = trend_model(fit_params)*transit_occ_model(fit_params)
        res = (df["cols1"] - flux_model)/df["cols2"]
        for p in fit_params:
            u = fit_params[p].user_data  #obtain tuple specifying the normal prior if defined
            if u:  #modify residual to account for how far the value is from mean of prior
                res = np.append(res, (u[0]-fit_params[p].value)/u[1] )
#         print(f"chi-square:{np.sum(res**2)}")
        return res
    
    fit_params = params+tr_params
    out = minimize(chisqr, fit_params, nan_policy='propagate')
    
    #modify output object
    out.bestfit = trend_model(out.params)*transit_occ_model(out.params)
    out.trend   = trend_model(out.params)
    out.transit = transit_occ_model(out.params)
    out.time    = np.array(df["cols0"])
    out.flux    = np.array(df["cols1"])
    out.flux_err= np.array(df["cols2"])
    out.ndata   = len(out.time)
    out.residual= out.residual[:out.ndata]
    out.nfree   = out.ndata - out.nvarys
    out.chisqr  = np.sum(out.residual**2)
    out.redchi  = out.chisqr/out.nfree
    out.bic     = out.ndata*np.log(out.chisqr/out.ndata) + out.nvarys*np.log(out.ndata)

    return out


def _print_output(self, section: str, file=None):
    """function to print to screen/file the different sections of CONAN setup"""

    lc_possible_sections= ["lc_baseline", "gp", "transit_rv_pars", "depth_variation",
                            "occultations", "limb_darkening", "contamination", "stellar_pars"]
    if self._obj_type == "lc_obj":
        assert section in lc_possible_sections, f"{section} not a valid section of `lc_data`. \
            section must be one of {lc_possible_sections}."
    if self._obj_type == "rv_obj":
        assert section == "rv_baseline", f"The only valid section for an RV data object is 'rv_baseline' but {section} given."
    if self._obj_type == "mcmc_obj":
        assert section == "mcmc",  f"The only valid section for an mcmc object is 'mcmc' but {section} given."

    if section == "lc_baseline":    
        _print_lc_baseline = f"""#--------------------------------------------- \n# Input lightcurves filters baseline function--------------""" +\
                            f""" \n{"name":15s}\t{"fil":3s}\t {"lamda":5s}\t {"time":4s}\t {"roll":3s}\t x\t y\t {"conta":5s}\t sky\t sin\t group\t id\t GP"""
        #define print out format
        txtfmt = "\n{0:15s}\t{1:3s}\t{2:5.1f}\t {3:4d}\t {4:3d}\t {5}\t {6}\t {7:5d}\t {8:3d}\t {9:3d}\t {10:5d}\t {11:2d}\t {12:2s}"        
        for i in range(len(self._names)):
            t = txtfmt.format(self._names[i], self._filters[i], self._lamdas[i], *self._bases[i], self._groups[i], self._useGPphot[i])
            _print_lc_baseline += t
        print(_print_lc_baseline, file=file)   

    if section == "gp":
        DA = self._GP_dict
        _print_gp = f"""# -------- photometry GP input properties: komplex kernel -> several lines -------------- """+\
                     f"""\n{'name':13s} {'para':5s} kernel WN {'scale':7s} s_step {'s_pri':5s} s_pri_wid {'s_up':5s} """+\
                         f"""{'s_lo':5s} {'metric':7s} m_step {'m_pri':6s} m_pri_wid {'m_up':4s} {'m_lo':4s}"""
        #define gp print out format
        if DA["lc_list"] != []:
            txtfmt = "\n{0:13s} {1:5s} {2:6s} {3:2s} {4:5.1e} {5:6.4f} {6:5.1f} {7:9.2e} {8:4.1f} {9:4.1f} {10:5.1e} {11:6.4f} {12:5.2f} {13:9.2e} {14:4.1f} {15:4.1f}"        
            for i in range(len(DA["lc_list"])):
                t = txtfmt.format(DA["lc_list"][i], DA["pars"][i],DA["kernels"][i],
                                    DA["WN"][i], DA["scale"][i], DA["s_step"][i], 
                                    DA["s_pri"][i], DA["s_pri_wid"][i], DA["s_up"][i],
                                    DA["s_lo"][i],DA["metric"][i],DA["m_step"][i], DA["m_pri"][i], 
                                    DA["m_pri_wid"][i],DA["m_up"][i],DA["m_lo"][i])
                _print_gp += t
        print(_print_gp, file=file)

    if section == "transit_rv_pars":
        DA = self._config_par
        _print_transit_rv_pars = f"""#=========== jump parameters (Jump0value step lower_limit upper_limit priors) ====================== """+\
                                  f"""\n{'name':12s}\tfit\tstart_val\tstepsize\tlow_lim\tup_lim\tprior\tvalue\tsig_lo\tsig_hi"""

        #define print out format
        txtfmt = "\n{0:12s}\t{1:3s}\t{2:8.5f}\t{3:.7f}\t{4:4.2f}\t{5:4.2f}\t{6}\t{7:.5f}\t{8:4.1e}\t{9:4.1e} "        
        for i,p in enumerate(self._parnames):
            t = txtfmt.format(  p, DA[p].to_fit, DA[p].start_value,
                                DA[p].step_size, DA[p].bounds_lo, 
                                DA[p].bounds_hi, DA[p].prior, DA[p].prior_mean,
                                DA[p].prior_width_lo, DA[p].prior_width_hi)
            _print_transit_rv_pars += t
        print(_print_transit_rv_pars, file=file)

    if section == "depth_variation":
        grnames    = np.array(list(sorted(set(self._groups))))
        ngroup     = len(grnames)
        _print_depth_variation = f"""#=========== ddF setup ============================================================================== """+\
                                    f"""\nFit_ddFs  step\t low_lim   up_lim   prior   sig_lo   sig_hi   div_white"""

        #define print out format
        txtfmt = "\n{0:8s}  {1:.3f}\t {2:.4f}   {3:.4f}   {4:5s}   {5:.5f}   {6:.5f}   {7:3s}"        
        t = txtfmt.format(self._ddfs.ddfYN,*self._ddfs.drprs_op[1:4],
                            self._ddfs.prior, self._ddfs.prior_width_lo,
                            self._ddfs.prior_width_hi,self._ddfs.divwhite)
        _print_depth_variation += t
        _print_depth_variation += "\ngroup_ID   RpRs_0   err\t\tdwfile"
        txtfmt = "\n{0:6d}\t   {1:.4f}   {2:.2e}   {3}"
        for i in range(ngroup):
            t2 = txtfmt.format( grnames[i] , self._ddfs.depth_per_group[i],
                                self._ddfs.depth_err_per_group[i],f"dw_00{grnames[i]}.dat")
            _print_depth_variation += t2

        print(_print_depth_variation, file=file)

    if section == "occultations":
        DA = self._occ_dict
        _print_occultations = f"""#=========== occultation setup ============================================================================= """+\
                                f"""\n{'filters':7s}\tfit start_val\tstepsize  {'low_lim':8s}  {'up_lim':8s}  prior  {'value':8s}  {'sig_lo':8s}\t{'sig_hi':8s}"""

        #define print out format
        # txtfmt = "\n{0:7s}\t{1:3s} {2:.8f}\t{3:.6f}  {4:7.6f}  {5:6.6f}  {6:5s}  {7:4.3e}  {8:4.2e}\t{9:4.2e} "       
        txtfmt = "\n{0:7s}\t{1:3s} {2:4.3e}\t{3:3.2e}  {4:3.2e}  {5:3.2e}  {6:5s}  {7:3.2e}  {8:3.2e}\t{9:3.2e} "       
        for i in range(len(self._filnames)):
            t = txtfmt.format(  DA["filters_occ"][i], DA["filt_to_fit"][i],
                                DA["start_value"][i], DA["step_size"][i],
                                DA["bounds_lo"][i],DA["bounds_hi"][i], 
                                DA["prior"][i], DA["prior_mean"][i],
                                DA["prior_width_lo"][i], DA["prior_width_hi"][i])
            _print_occultations += t
        print(_print_occultations, file=file)

    if section == "limb_darkening":
        DA = self._ld_dict
        _print_limb_darkening = f"""#=========== Limb darkending setup ==================================================================="""+\
                                f"""\n{'filters':7s} priors\t{'c_1':4s} {'step1':5s}  sig_lo1  sig_hi1  lo_lim1 hi_lim1\t{'c_2':4s} {'step2':5s} sig_lo2 sig_hi2  lo_lim2 hi_lim2"""

        #define print out format
        txtfmt = "\n{0:7s} {1:6s}\t{2:4.3f} {3:5.3f} {4:7.4f} {5:7.4f}  {6:7.4f} {7:7.4f}\t{8:4.3f} {9:5.3f} {10:7.4f} {11:7.4f}  {12:7.4f} {13:7.4f}"       
        for i in range(len(self._filnames)):
            t = txtfmt.format(self._filnames[i],DA["priors"][i], 
                            DA["c1"][i], DA["step1"][i], DA["sig_lo1"][i], DA["sig_hi1"][i], DA["bound_lo1"][i], DA["bound_hi1"][i], 
                            DA["c2"][i], DA["step2"][i], DA["sig_lo2"][i], DA["sig_hi2"][i], DA["bound_lo2"][i], DA["bound_hi2"][i],)
            _print_limb_darkening += t

        print(_print_limb_darkening, file=file)

    if section == "contamination":
        DA = self._contfact_dict
        _print_contamination = f"""#=========== contamination setup === give contamination as flux ratio ================================"""+\
                                f"""\n{'filters':7s}\tcontam\terr"""
        #define print out format
        txtfmt = "\n{0:7s}\t{1:.4f}\t{2:.4f}"       
        for i in range(len(self._filnames)):
            t = txtfmt.format(self._filnames[i],DA["cont_ratio"][i], 
                                DA["err"][i])
            _print_contamination += t
        print(_print_contamination, file=file)

    if section == "stellar_pars":
        DA = self._stellar_dict
        _print_stellar_pars = f"""#=========== Stellar input properties ================================================================"""+\
        f"""\n{'# parameter':13s}  value  sig_lo  sig_hi """+\
        f"""\n{'Radius_[Rsun]':13s}  {DA['R_st'][0]:.3f}  {DA['R_st'][1]:.3f}  {DA['R_st'][2]:.3f} """+\
            f"""\n{'Mass_[Msun]':13s}  {DA['M_st'][0]:.3f}  {DA['M_st'][1]:.3f}  {DA['M_st'][2]:.3f}"""+\
            f"""\nStellar_para_input_method:_R+rho_(Rrho),_M+rho_(Mrho),_M+R_(MR): {DA['par_input']}"""
        print(_print_stellar_pars, file=file)           

    if section == "mcmc":
        DA = self._mcmc_dict
        _print_mcmc_pars = f"""#=========== MCMC setup =============================================================================="""+\
        f"""\n{'Total_no_steps':23s}  {DA['n_steps']*DA['n_chains']} \n{'Number_chains':23s}  {DA['n_chains']} \n{'Number_of_processes':23s}  {DA['n_cpus']} """+\
            f"""\n{'Burnin_length':23s}  {DA['n_burn']} \n{'Walk_(snooker/demc/mrw)':23s}  {DA['sampler']} \n{'GR_test_(y/n)':23s}  {DA['GR_test']} """+\
                f"""\n{'Make_plots_(y/n)':23s}  {DA['make_plots']} \n{'leastsq_(y/n)':23s}  {DA['leastsq']} \n{'Savefile':23s}  {DA['savefile']} \n{'Savemodel':23s}  {DA['savemodel']} """+\
                    f"""\n{'Adapt_base_stepsize':23s}  {DA['adapt_base_stepsize']} \n{'Remove_param_for_CNM':23s}  {DA['remove_param_for_CNM']} \n{'leastsq_for_basepar':23s}  {DA['leastsq_for_basepar']} """+\
                        f"""\n{'lssq_use_Lev-Marq':23s}  {DA['lssq_use_Lev_Marq']} \n{'apply_CFs':23s}  {DA['apply_CFs']} \n{'apply_jitter':23s}  {DA['apply_jitter']}"""

        print(_print_mcmc_pars, file=file)

    if section == "rv_baseline":
        _print_rv_baseline = f"""# ------------------------------------------------------------\n# Input RV curves, baseline function, gamma  """+\
                                    f"""\n{'name':13s}   time  bis  fwhm  contrast  sinPs  gamma_kms  stepsize  prior  value  sig_lo  sig_hi"""
        
        if self._names != []:
            #define gp print out format
            txtfmt = "\n{0:13s}   {1:4d}  {2:3d}  {3:4d}  {4:8d}  {5:5d}  {6:9.4f}  {7:8.4f}  {8:5s}  {9:6.4f}  {10:6.4f}  {11:6.4f}"         
            for i in range(self._nRV):
                t = txtfmt.format(self._names[i],*self._RVbases[i],self._gammas[i], 
                                self._gamsteps[i], self._prior[i], self._gampri[i],
                                self._siglo[i], self._sighi[i])
                _print_rv_baseline += t

        print(_print_rv_baseline, file=file)

class _param_obj():
    def __init__(self,par_list):
        """
            par_list: list of len =9;
                list of configuration values for the specified model parameter.
        """
        assert len(par_list) == 9, f"length of input list must be 9 ({len(par_list)} given)."
        for i in range(len(par_list)): assert isinstance(par_list[i], (int,float,str)), \
            f"par_list[{i}] must be of type: int, float or str."
            
        self.to_fit = par_list[0] if (par_list[0] in ["n","y"]) \
                                    else _raise(ValueError, "to_fit (par_list[0]) must be 'n' or 'y'")
        self.start_value = par_list[1]
        self.step_size = par_list[2]
        self.prior = par_list[3] if (par_list[3] in ["n","p"]) \
                                    else _raise(ValueError, "prior (par_list[3]) must be 'n' or 'p'")
        self.prior_mean = par_list[4]
        self.prior_width_lo = par_list[5]
        self.prior_width_hi = par_list[6]
        self.bounds_lo = par_list[7]
        self.bounds_hi = par_list[8]
        
    def _set(self, par_list):
        return self.__init__(par_list)
    
    @classmethod
    def re_init(cls, par_list):
        _ = cls(par_list)
        
    def __repr__(self):
        return f"{self.__dict__}"
    
    def _get_list(self):
        return [p for p in self.__dict__.values()]

class _text_format:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


#========================================================================
class load_lightcurves:
    """
        lightcurve object to hold lightcurves for analysis
        
        Parameters:
        -----------
        
        data_filepath : str;
            Filepath where lightcurves files are located. Default is None which implies the data is in the current working directory.

            
        file_list : list;
            List of filenames for the lightcurves.
            
        filter : list, str, None;
            filter for each lightcurve in file_list. if a str is given, it is used for all lightcurves,
            if None, the default of "V" is used for all.
            
        lamdas : list, int, float, None;
            central wavelength for each lightcurve in file_list. if a int or float is given, it is used for all lightcurves,
            if None, the default of 6000.0 is used for all.
        
        Returns:
        --------
        lc_data : light curve object
        
    """
    def __init__(self, file_list, data_filepath=None, filters=None, lamdas=None,
                 verbose=True, show_guide=False):
        self._obj_type = "lc_obj"
        self._fpath = os.getcwd() if data_filepath is None else data_filepath
        self._names = [file_list] if isinstance(file_list, str) else file_list
        for lc in self._names: assert os.path.exists(self._fpath+lc), f"file {lc} does not exist in the path {self._fpath}."
        
        assert filters is None or isinstance(filters, (list, str)), \
            f"filters is of type {type(filters)}, it should be a list, a string or None."
        assert lamdas  is None or isinstance(lamdas, (list, int, float)), \
            f"lamdas is of type {type(lamdas)}, it should be a list, int or float."
        
        if isinstance(filters, str): filters = [filters]
        if isinstance(lamdas, (int, float)): lamdas = [float(lamdas)]

        if filters is not None and len(filters) == 1: filters = filters*len(self._names)
        if lamdas is  not None and len(lamdas)  == 1: lamdas  = lamdas *len(self._names)

        self._filters = ["V"]*len(self._names) if filters is None else [f for f in filters]
        self._lamdas  = [6000.0]*len(self._names) if lamdas is None else [l for l in lamdas]
        
        assert len(self._names) == len(self._filters) == len(self._lamdas), \
            f"filters and lamdas must be a list with same length as file_list (={len(self._names)})"
        self._filnames   = np.array(list(sorted(set(self._filters),key=self._filters.index)))

        if verbose: 
            print(f"Filters: {self._filters}")
            print(f"Order of unique filters: {list(self._filnames)}")

        self._show_guide = show_guide
        self.lc_baseline(verbose=False)

        if self._show_guide: print("\nNext: use method `lc_baseline` to define baseline model for each lc or method " + \
            "`get_decorr` to obtain best best baseline model parameters according bayes factor comparison")

    def get_decorr(self, T_0=None, P=None, dur=None, L=0, b=0, rp=1e-5,q1=0, q2=0, mask=False, decorr_bound =(-1,1),
                     cheops=False, verbose=True, show_steps=False, plot_model=True, use_result=True):
        """
            Function to obtain best decorrelation parameters for each light-curve file using the forward selection method.
            It compares a model with only an offset to a polynomial model constructed with the other columns of the data.
            It uses columns 0,3,4,5,6,7 to construct the polynomial trend model. The temporary decorr parameters are labelled Ai,Bi for 1st & 2nd order in column i.
            If cheops is True, A5, B5 are the sin and cos of the roll-angle while A5_i, B5_i are the corresponding harmonics with i=2,3. If these are significant, a gp in roll-angle will be needed.
            Decorrelation parameters that reduces the BIC (favored with Bayes factor > 1) are iteratively selected.
            The result can then be used to populate the `lc_baseline` method, if use_result is set to True.

            Parameters:
            -----------
            T_0, P, dur, L, b, rp : floats, None;
                transit/eclipse parameters of the planet. T_0, P, and dur must be in same units as the time axis (cols0) in the data file.
                if float/int, the values are held fixed. if tuple/list of len 2 implies gaussian prior as (mean,std) while len 3 implies [min,start_val,max].
                
            q1,q2 : float  (optional);
                quadratic limb darkening parameters according to kipping 2013. values are in the range (0,1).
    
            mask : bool ;
                If True, transits and eclipses are masked using T_0, P and dur which must be float/int.
        
            decorr_bound: tuple of size 2;
                bounds when fitting decorrelation parameters. Default is (-1,1)

            cheops : Bool or list of Bool;
                Flag to specify if data is from CHEOPS with col5 being the roll-angle. if True, a linear + \
                    fourier model (sin and cos) up to 3rd harmonic in roll-angle is used for col5.
                If Bool is given, the same is used for all input lc, else a list specifying bool for each lc is required.
                Default is False.

            verbose : Bool, optional;
                Whether to show the table of baseline model obtained. Defaults to True.

            show_steps : Bool, optional;
                Whether to show the steps of the forward selection of decorr parameters. Default is False
            
            plot_model : Bool, optional;
                Whether to overplot suggested trend model on the data. Defaults to True.

            use_result : Bool, optional;
                whether to use result/input to setup the baseline model and transit/eclipse models. Default is True.
        """
        
        blpars = {"dt":[], "dphi":[],"dx":[], "dy":[], "dconta":[], "dsky":[],"gp":[]}  #inputs to lc_baseline method
        self._decorr_result = []   #list of decorr result for each lc
        self._tra_occ_pars = dict(T_0=T_0, P=P, dur=dur, L=L, b=b, rp=rp, q1=q1,q2=q2)  #transit/occultation parameters
        
        #if input transit par is iterable, make no of elements=3 [min, start, max]
        # for p in self._tra_occ_pars.keys():
        #     if isinstance(self._tra_occ_pars[p],(list, tuple)) and len(self._tra_occ_pars[p])==2:
        #         val = self._tra_occ_pars[p]
        #         self._tra_occ_pars[p] = (val[0], np.median(val), val[1])    
        
        #check cheops input
        if isinstance(cheops, bool): cheops_flag = [cheops]*len(self._names)
        elif isinstance(cheops, list):
            assert len(cheops) == len(self._names),f"list given for cheops must have same +\
                length as number of input lcs but {len(cheops)} given."
            for flag in cheops:
                assert isinstance(flag, bool), f"all elements in cheops list must be bool: +\
                     True or False, but {flag} given"
            cheops_flag = cheops
        else: _raise(TypeError, f"`cheops` must be bool or list of bool with same length as +\
            number of input files but type{cheops} given.")


        t_model = []  #list to hold determined trendmodel for each lc
        for j,file in enumerate(self._names):
            if verbose: print(_text_format.BOLD + f"\ngetting decorrelation parameters for lc: {file} (cheops={cheops_flag[j]})" + _text_format.END)
            all_par = [f"{L}{i}" for i in [0,3,4,5,6,7] for L in ["A","B"]] 
            if cheops_flag[j]: all_par += ["A5_2","B5_2","A5_3","B5_3"]

            out = _decorr(self._fpath+file, **self._tra_occ_pars, mask=mask,
                            offset=0,cheops=cheops_flag[j], decorr_bound=decorr_bound)    #no trend, only offset
            best_bic = out.bic
            best_pars = {"offset":0}
            if show_steps: print(f"{'Param':7s} : {'BIC':6s} N_pars \n---------------------------")

            # bic_ratio = 0 
            bf = np.inf
            while  bf > 1:
                if show_steps: print(f"{'Best':7s} : {best_bic:.2f} {len(best_pars.keys())} {list(best_pars.keys())}\n---------------------")
                pars_bic = {}
                for p in all_par:
                    dtmp = best_pars.copy()   #always include offset
                    dtmp[p] = 0
                    out = _decorr(self._fpath+file, **self._tra_occ_pars,**dtmp,
                                    cheops=cheops_flag[j], decorr_bound=decorr_bound)
                    if show_steps: print(f"{p:7s} : {out.bic:.2f} {out.nvarys}")
                    pars_bic[p] = out.bic

                par_in = min(pars_bic,key=pars_bic.get)
                par_in_bic = pars_bic[par_in]
                # bic_ratio = par_in_bic/best_bic
                del_BIC = par_in_bic - best_bic
                bf = np.exp(-0.5*(del_BIC))
                if show_steps: print(f"+{par_in} -> BF:{bf:.2f}, del_BIC:{del_BIC:.2f}")
            #     if bic_ratio < 1:
                if bf>1:
                    if show_steps: print(f"adding {par_in} lowers BIC to {par_in_bic:.2f}\n" )
                    best_pars[par_in]=0
                    best_bic = par_in_bic
                    all_par.remove(par_in)            
                      
            result = _decorr(self._fpath+file, **self._tra_occ_pars,
                                **best_pars, cheops=cheops_flag[j], decorr_bound=decorr_bound)
            self._decorr_result.append(result)
            print(f"BEST BIC:{result.bic:.2f}, pars:{list(best_pars.keys())}")
            
            #calculate determined trend and tra/occ model over all data(no mask)
            pps = result.params.valuesdict()
            t_model.append(_decorr(self._fpath+file,**pps, cheops=cheops_flag[j], return_models=True))

            #set-up lc_baseline model from obtained configuration
            blpars["dt"].append( 2 if pps["B0"]!=0 else 1 if  pps["A0"]!=0 else 0)
            blpars["dx"].append( 2 if pps["B3"]!=0 else 1 if  pps["A3"]!=0 else 0)
            blpars["dy"].append( 2 if pps["B4"]!=0 else 1 if  pps["A4"]!=0 else 0)
            blpars["dconta"].append( 2 if pps["B6"]!=0 else 1 if  pps["A6"]!=0 else 0)
            blpars["dsky"].append( 2 if pps["B7"]!=0 else 1 if  pps["A7"]!=0 else 0)
            if not cheops_flag[j]:
                blpars["dphi"].append( 2 if pps["B5"]!=0 else 1 if  pps["A5"]!=0 else 0)
                blpars["gp"].append("n")
            else:
                blpars["dphi"].append(0)
                blpars["gp"].append("y")  #for gp in roll-angle (mostly needed)

        if plot_model:
            _plot_data(self,plot_cols=(0,1,2),col_labels=("time","flux"),model_overplot=t_model)

        #prefill other light curve setup from the results here or inputs given here.
        if use_result:
            if verbose: print(_text_format.BOLD + "Setting-up baseline model from result" +_text_format.END)
            self.lc_baseline(**blpars, verbose=verbose)
            print(_text_format.RED + f"\n Note: GP flag for each lc has been set to {self._useGPphot}. "+\
                    "Use `._useGPphot` method to modify this list with 'y' or 'n' for each loaded lc\n" + _text_format.END)

            if isinstance(self._tra_occ_pars["L"], (list, tuple)):
                if verbose: print(_text_format.BOLD + "\nSetting-up occultation pars from input values" +_text_format.END)
                self.setup_occultation("all",start_depth=tuple(self._tra_occ_pars["L"]), verbose=verbose)
            
            if all([p in self._tra_occ_pars for p in["P","dur","b","rp","T_0"]]):
                if verbose: print(_text_format.BOLD + "\nSetting-up transit pars from input values" +_text_format.END)
                self.setup_transit_rv(RpRs=self._tra_occ_pars["rp"], Impact_para=self._tra_occ_pars["b"], T_0=self._tra_occ_pars["T_0"],
                                    Period=self._tra_occ_pars["P"], Duration=self._tra_occ_pars["dur"], verbose=verbose)


    def lc_baseline(self, dt=None,  dphi=None, dx=None, dy=None, dconta=None, 
                 dsky=None, dsin=None, grp=None, grp_id=None, gp="n", verbose=True):
        """
            Define lightcurve baseline model parameters to fit.
            Each baseline decorrelation parameter should be a list of integers specifying the polynomial order for each light curve.
            e.g. Given 3 input light curves, if one wishes to fit a 2nd order time trend to only the first and third lightcurves,
            then dt = [2, 0, 2].
            The decorrelation parameters depend on the columns of the input light curve. Any desired array can be put in these columns to decorrelate against them irrespective of the name here (which would be modified soon).
            The columns are:
            * dt: column 0
            * dx: colums 3
            * dy: colums 4
            * dphi: colums 5
            * dconta: colums 6
            * dsky: colums 7

            Parameters:
            -----------
            dt, dx,dy,dphi,dconta,dsky : time, x_pos,roll_angle(col5)
            grp_id : list (same length as file_list);
                group the different input lightcurves by id so that different transit depths can be fitted for each group.

            gp : list (same length as file_list); 
                list containing 'y', 'n', or 'ce' to specify if a gp will be fitted to a light curve. +\
                    'ce' indicates that the celerite package will be used for the gp. 

        """
        dict_args = locals().copy()     #get a dictionary of the input arguments for easy manipulation
        _ = dict_args.pop("self")            #remove self from dictionary
        _ = dict_args.pop("verbose")

        n_lc = len(self._names)

        for par in dict_args.keys():
            assert dict_args[par] is None or isinstance(dict_args[par], (int,str)) or \
                (isinstance(dict_args[par], (list,np.ndarray)) and len(dict_args[par]) == n_lc), \
                    f"parameter {par} must be a list of length {n_lc} or int (if same degree is to be used for all LCs) or None (if not used in decorrelation)."
            
            if isinstance(dict_args[par], (int,str)): dict_args[par] = [dict_args[par]]*n_lc
            elif dict_args[par] is None: dict_args[par] = [0]*n_lc

        dict_args["grp_id"] = list(np.arange(1,n_lc+1))

        self._bases = [ [dict_args["dt"][i], dict_args["dphi"][i], dict_args["dx"][i], dict_args["dy"][i],
                        dict_args["dconta"][i], dict_args["dsky"][i], dict_args["dsin"][i], 
                        dict_args["grp"][i]] for i in range(n_lc) ]

        self._groups = dict_args["grp_id"]
        self._grbases = dict_args["grp"]
        self._useGPphot= dict_args["gp"]
        self._gp_lcs = np.array(self._names)[np.array(self._useGPphot) != "n"]     #lcs with gp == "y" or "ce"

        if verbose: _print_output(self,"lc_baseline")

        if np.all(np.array(self._useGPphot) == "n"):        #if gp is "n" for all input lightcurves, run add_GP with None
            self.add_GP(None, verbose=verbose)
            if self._show_guide: print("\nNo GPs required.\nNext: use method `setup_transit_rv` to configure transit an rv model parameters.")
        else: 
            if self._show_guide: print("\nNext: use method `add_GP` to include GPs for the specified lcs. Get names of lcs with GPs using `._gp_lcs` attribute of the lightcurve object.")

        #initialize other methods to empty incase they are not called/have not been called
        if not hasattr(self,"_config_par"):    self.setup_transit_rv(verbose=False)
        if not hasattr(self,"_ddfs"):          self.transit_depth_variation(verbose=False)
        if not hasattr(self,"_occ_dict"):      self.setup_occultation(verbose=False)
        if not hasattr(self,"_contfact_dict"): self.contamination_factors(verbose=False)
        if not hasattr(self,"_ld_dict"):       self.limb_darkening(verbose=False)
        if not hasattr(self,"_stellar_dict"):  self.stellar_parameters(verbose=False)
   
    def add_GP(self, lc_list=None, pars="time", kernels="mat32", WN="y", 
               log_scale=[(-25,-15.2,-5)], s_step=0.1,
               log_metric=[(-10,6.9,15)],  m_step=0.1,
               verbose=True):
        """
            Model variations in light curve with a GP (using george GP package)
            
            Parameters:
            
            lc_list : list of strings, None;
                list of lightcurve filenames to which a GP is to be applied.
                If n-dimensional GP is to be applied to a lightcurve, the filename should be listed n times consecutively (corresponding to each dimension).
            
            pars : list of strings;
                independent variable of the GP for each lightcurve name in lc_list. 
                If a lightcurve filename is listed more than once in lc_list, par is used to apply a GP along a different axis.
                For each lightcurve, `par` can be any of ["time", "xshift", "yshift", "air", "fwhm", "sky", "eti"]
                
            kernel : list of strings;
                GP kernel for each lightcuve file in lc_list. Options: "mat32"
                
            WN : list;
                list containing "y" or "n" to specify whether to fit a white noise component for each GP. 
                
            log_scale, log_metric : list of tuples;
                Prior of log_scale (variance) and log_metric (lengthscale) of the GP kernel applied for each lc in lc_list.
                * if tuple is of len 2, set normal prior with index[0] as prior mean and index[1] as prior width.
                * if tuple is of len 3, set uniform prior with between index[0] and index[2], index[1] is the initial value.
                if a single tuple is given, same prior is used for all specified lcs 

               
            s_step, m_step : list of floats;
                step sizes of the scale and metric parameter of the GP kernel.
        
        """
        assert isinstance(log_scale, (tuple,list)), f"log_scale must be a list of tuples specifying value for each lc or single tuple if same for all lcs."
        assert isinstance(log_metric, (tuple,list)), f"log_metric must be a list of tuples specifying value for each lc or single tuple if same for all lcs."

        if isinstance(log_scale, tuple): log_scale= [log_scale]
        if isinstance(log_metric, tuple): log_metric= [log_metric]

        #unpack scale and metric to the expected CONAN parameters
        scale, s_pri, s_pri_wid, s_lo, s_up = [], [], [], [], []
        for s in log_scale:
            if isinstance(s,tuple) and len(s)==2:
                s_pri.append(s[0])
                scale.append( np.exp(s[0]) )
                s_pri_wid.append(s[1])
                s_up.append( np.max((s[0]+10, s[0]+5*s[1])) )    #set bounds at +/- 10 from prior mean or 5stdev (the larger value)
                s_lo.append( np.min((s[0]-10, s[0]-5*s[1])) )

            elif isinstance(s,tuple) and len(s)==3:
                s_pri_wid.append(0)          #using uniform prior so set width = 0
                s_lo.append(s[0])
                scale.append(np.exp(s[1]))
                s_pri.append(0.0)
                s_up.append(s[2])
            
            else: _raise(TypeError, f"tuple of len 2 or 3 was expected but got the value {s} in log_scale.")

        metric, m_pri, m_pri_wid, m_lo, m_up  = [], [], [], [], []
        for m in log_metric:
            if isinstance(m,tuple) and len(m)==2:
                m_pri.append(m[0])
                metric.append( np.exp(m[0]) )
                m_pri_wid.append(m[1])
                m_up.append( np.max((m[0]+10,m[0]+5*m[1])) )    #set uniform bounds at _+/- 10 from prior mean
                m_lo.append( np.min((m[0]-10, m[0]-5*m[1])) )
                
            elif isinstance(m,tuple) and len(m)==3:
                m_pri_wid.append(0)       
                m_lo.append(m[0])
                metric.append( np.exp(m[1]) )
                m_pri.append(0.0)
                m_up.append(m[2])

            else: _raise(TypeError, f"tuple of len 2 or 3 was expected but got the value {m} in log_metric.")


        DA = locals().copy()
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")
        if "m" in DA: DA.pop("m")
        if "s" in DA: DA.pop("s")
        _ = [DA.pop(item) for item in ["log_metric", "log_scale"]]
        

        if lc_list is None: 
            self._GP_dict = {"lc_list":[]}
            if len(self._gp_lcs)>0: print(f"\nWarning: GP was expected for the following lcs {self._gp_lcs} \nMoving on ...")
            if verbose: _print_output(self,"gp")
            return 
        elif isinstance(lc_list, str): lc_list = [lc_list]

        if 'all' not in lc_list:
            for lc in self._gp_lcs: 
                assert lc in lc_list,f"GP was expected for {lc} but was not given in lc_list."   

            for lc in lc_list: 
                assert lc in self._names,f"{lc} is not one of the loaded lightcurve files"
                assert lc in self._gp_lcs, f"while defining baseline model in the `lc_baseline` method, gp = 'y' was not specified for {lc}."
        n_list = len(lc_list)
        
        #transform        
        for key in DA.keys():
            if (isinstance(DA[key],list) and len(DA[key])==1): 
                DA[key]= DA[key]*n_list
            if isinstance(DA[key], list):
                assert len(DA[key]) == n_list, f"{key} must have same length as lc_list"
            if isinstance(DA[key],(float,int,str)):  
                DA[key] = [DA[key]]*n_list
                
        
        for p in DA["pars"]: 
            assert p in ["time", "xshift", "yshift", "air", "fwhm", "sky", "eti"], \
                f"pars {p} cannot be the GP independent variable"             
        
        
        assert len(DA["pars"]) == len(DA["kernels"]) == len(DA["WN"]) == n_list, f"pars and kernels must have same length as lc_list (={len(lc_list)})"
                                            
        self._GP_dict = DA     #save dict of gp pars in lc object

        if verbose: _print_output(self,"gp")

        if self._show_guide: print("\nNext: use method `setup_transit_rv` to configure transit parameters.")

    def setup_transit_rv(self, RpRs=0.1, Impact_para=0, Duration=0.1245, T_0=0, Period=3, 
                 Eccentricity=0, omega=90, K=0, verbose=True):
        """
            Define parameters an priors of model parameters.
            By default, the parameters are fixed to the given values. To fit a parameter use the `to_fit` method to change it from 'n' to 'y'.
            The parameters can be defined in following ways:
            
            * fixed value as float or int, e.g Period = 3.4
            * free parameter with gaussian prior given as tuple of len 2, e.g. T_0 = (5678, 0.1)
            * free parameters with uniform prior interval and initial value given as tuple of length 3, e.g. RpRs = (0,0.1,0.2) with 0.1 being the initial value.
        """
        
        DA = locals().copy()         #dict of arguments (DA)
        _ = DA.pop("self")                            #remove self from dictionary
        _ = DA.pop("verbose")
        #sort to specific order
        key_order = ["RpRs","Impact_para","Duration", "T_0", "Period", "Eccentricity","omega", "K"]
        DA = {key:DA[key] for key in key_order if key in DA} 
            
        self._parnames  = [n for n in DA.keys()]
        self._npars = 8

        for par in DA.keys():
            if par in ["RpRs","Impact_para","Duration", "Eccentricity"]: up_lim = 1
            elif par == "omega": up_lim = 360
            else: up_lim = 10000

            #fitting parameter
            if isinstance(DA[par], tuple):
                #gaussian       
                if len(DA[par]) == 2:        
                    DA[par] = _param_obj(["y", DA[par][0], 0.01*DA[par][1], "p", DA[par][0],
                                  DA[par][1], DA[par][1], 0, up_lim])
                #uniform
                elif len(DA[par]) == 3: 
                    DA[par] = _param_obj(["y", DA[par][1], 0.01*np.ptp(DA[par]), "n", DA[par][1],
                                       0, 0, DA[par][0], DA[par][2]])
                
                else: _raise(ValueError, f"length of tuple is {len(DA[par])} but it must be 2 or 3 such that it follows (lo_limit, start_value, up_limit).")
            #fixing parameter
            elif isinstance(DA[par], (int, float)):
                DA[par] = _param_obj(["n", DA[par], 0.00, "n", DA[par],
                                       0,  0, 0, up_lim])

            else: _raise(TypeError, f"{par} must be one of [tuple(of len 2 or 3), int, float] but is {type(DA[par])}")

        self._config_par = DA      #add to object
        self._items = DA["RpRs"].__dict__.keys()
        
        if verbose: _print_output(self,"transit_rv_pars")


        if self._show_guide: print("\nNext: use method transit_depth_variation` to include variation of RpRs for the different filters or \n`setup_occultation` to fit the occultation depth or \n`limb_darkening` for fit or fix LDCs or `contamination_factors` to add contamination.")

    def transit_depth_variation(self, ddFs="n", transit_depth_per_group=[(0.1,0.0001)], divwhite="n",
                        step=0.001, bounds=(-1,1), prior="n", prior_width=(0,0),
                       verbose=True):
        """
            Include transit depth variation between the different lcs or lc groups. Note RpRs must be fixed to a reference value and not a jump parameter.
            
            Parameters:
            ----------

            ddFs : str ("y" or "n");
                specify if to fit depth variation or not. default is "n"

            transit_depth_per_group : list of size2-tuples;
                the reference depth (and uncertainty) to compare the transit depth of each lightcurve group with.
                Usually from fit to the white (or total) available light-curves. 
                The length should be equal to the length of unique groups defined in lc_baseline.
                if list contains only 1 tuple, then same value of depth and uncertainty is used for all groups.

            divwhite : str ("y" or "n");
                flag to divide each light-curve by the white lightcurve. Default is "n"

            step: float;
                stepsize when fitting for depth variation

            bounds: tuple of len 2;
                tuple with lower and upper bound of the deviation of depth. Default is (-1,1).

            prior: str ("y" or "n"):
                use gaussian prior or not on the depth deviation

            prior_width: tuple of len 2;
                if using gaussian prior, set the width of the priors. 

            verbose: bool;
                print output
                  
        """
        
        self._ddfs= SimpleNamespace()

        if ddFs == "y":
            assert self._config_par["RpRs"].to_fit == "n",'Fix `RpRs` in `setup_transit_rv` to a reference value in order to setup depth variation.'
        
        assert isinstance(transit_depth_per_group, (tuple,list)),f"transit_depth_per_group must be type tuple or list of tuples."
        if isinstance(transit_depth_per_group,tuple): transit_depth_per_group = [transit_depth_per_group]
        depth_per_group     = [d[0] for d in transit_depth_per_group]
        depth_err_per_group = [d[1] for d in transit_depth_per_group]

        assert isinstance(prior_width, tuple),f"prior_width must be tuple with lower and upper widths."
        prior_width_lo, prior_width_hi = prior_width

        assert isinstance(bounds, tuple),f"bounds must be tuple with lower and upper values."
        bounds_lo, bounds_hi = bounds


        width_lo = (0 if (prior == 'n' or ddFs == 'n' or bounds_lo == 0.) else prior_width_lo)
        width_hi = (0 if (prior == 'n' or ddFs == 'n' or bounds_hi == 0.) else prior_width_hi)

        self._ddfs.drprs_op=[0., step, bounds_lo, bounds_hi, 0., width_lo, width_hi]  # the dRpRs options
        
        grnames    = np.array(list(sorted(set(self._groups))))
        ngroup     = len(grnames)
        
        if len(depth_per_group)==1: depth_per_group = depth_per_group * ngroup     #depth for each group
        if len(depth_err_per_group)==1: depth_err_per_group = depth_err_per_group * ngroup

        
        assert len(depth_per_group)== len(depth_err_per_group)== ngroup, \
            f"length of depth_per_group and depth_err_per_group must be equal to the number of unique groups (={ngroup}) defined in `lc_baseline`"
        
        nphot      = len(self._names)             # the number of photometry input files

        self._ddfs.depth_per_group     = depth_per_group
        self._ddfs.depth_err_per_group = depth_err_per_group
        self._ddfs.divwhite            = divwhite
        self._ddfs.prior               = prior
        self._ddfs.ddfYN               = ddFs
        self._ddfs.prior_width_lo      = prior_width_lo
        self._ddfs.prior_width_hi      = prior_width_hi
        if divwhite=="y":
            assert ddFs=='n', 'you can not do divide-white and not fit ddfs!'
            
            for i in range(nphot):
                if (self._bases[i][6]>0):
                    _raise(ValueError, 'you can not have CNMs active and do divide-white')
        

        if (ddFs=='n' and np.max(self._grbases)>0):
            print('no ddFs but groups? Not a good idea!')
            print(base)
            
        if verbose: _print_output(self,"depth_variation")
                
    def setup_occultation(self, filters_occ=None, start_depth=[(0,500e-6,1000e-6)], step_size=0.00001,verbose=True):
        """
            setup fitting for occultation depth
            
            Parameters:
            -----------
            
            filters_occ : list;
                List of unique filters to fit. 
                If "all", occultation depth is fit for all filters given in `lc.load_lightcurves`. 
                use `lc_data._filnames` to obtain the list of unique filters.
                If None, will not fit occultation.
            
            start_depth : list of tuples, tuple;
                define start value for occultation depth in each filter and the priors/bounds.
                * if tuple is of len 2, set normal prior with index[0] as prior mean and index[1] as prior width. \
                    hard bounds are set between 0 and 1
                * if tuple is of len 3, set uniform prior with between index[0] and index[2], index[1] is the initial value.
              
            
            step_size : list, float;
                step size for each filter. If float, the same step size is used for all filters.
                
            verbose: bool;
                print output configuration or not.
            
        """
        if isinstance(filters_occ, str):
            if filters_occ == "all": filters_occ = list(self._filnames)
            else: filters_occ= [filters_occ]
        if filters_occ is None: filters_occ = []

        assert isinstance(start_depth,(tuple,list)), f"start depth must be list of tuple for depth in each filter or tuple for same in all filters."
        if isinstance(start_depth, tuple): start_depth= [start_depth]
        # unpack start_depth input
        start_value, prior, prior_mean, prior_width_hi, prior_width_lo, bounds_hi, bounds_lo = [],[],[],[],[],[],[]
        for dp in start_depth:
            if isinstance(dp,tuple) and len(dp)==2:
                start_value.append(dp[0])
                prior.append("y" if dp[1] else "n")
                prior_mean.append(dp[0])
                prior_width_hi.append(dp[1])
                prior_width_lo.append(dp[1])
                bounds_lo.append(0)
                bounds_hi.append(1)

            elif isinstance(dp,tuple) and len(dp)==3:
                start_value.append(dp[1])
                prior.append("n")
                prior_mean.append(0)
                prior_width_hi.append(0)
                prior_width_lo.append(0)
                bounds_lo.append(dp[0])
                bounds_hi.append(dp[2])

            else: _raise(TypeError, f"tuple of len 2 or 3 was expected but got the value {dp} in start_depth.")


        DA = _reversed_dict(locals().copy())
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")
        _ = DA.pop("dp")
        _ = DA.pop("start_depth")
        
        if verbose: 
            if filters_occ != [] : print(f"fitting occultation depth for filters: {filters_occ}\n")
            else: print("Not fitting occultation\n")

        nfilt  = len(self._filnames)    #length of unique filters 
        nocc   = len(filters_occ)        #length of given filters to fit
        
                      

        if filters_occ != []:
            for f in filters_occ: assert f in self._filnames, \
                f"{f} is not in list of defined filters"
            
            for par in DA.keys():
                assert isinstance(DA[par], (int,float,str)) or \
                    (isinstance(DA[par], list) and ( (len(DA[par]) == nocc) or (len(DA[par]) == 1))), \
                    f"length of input {par} must be equal to the length of filters_occ (={nocc}) or float or None."

                if (isinstance(DA[par], list) and len(DA[par]) == 1):  DA[par] = DA[par]*nocc
                if isinstance(DA[par], (int,float,str)):             DA[par] = [DA[par]]*nocc


        
        DA2 = {}    # expand dictionary to also include specifications for non-fitted filters
        DA2["filt_to_fit"] = [("y" if f in filters_occ else "n") for f in self._filnames]

        indx = [ list(self._filnames).index(f) for f in filters_occ]    #index of given filters_occ in unique filter names
        for par in DA.keys():
            if par == "prior": DA2[par] = ["n"]*nfilt
            elif par == "filters_occ": DA2[par] = list(self._filnames)
            else: DA2[par] = [0]*nfilt

            for i,j in zip(indx, range(nocc)):                
                DA2[par][i] = DA[par][j]

        self._occ_dict =  DA = DA2
        if verbose: _print_output(self,"occultations")

    def limb_darkening(self, c1=0, c2 = 0,verbose=True):
        """
            Setup quadratic limb darkening LD parameters (c1, c2) for transit light curves. 
            Different LD parameters are required if observations of different filters are used.

            Parameters:
            ---------
            c1,c2 : float/tuple or list of float/tuple for each filter;
                Stellar quadratic limb darkening coefficients.
                if tuple, must be of - length 2 for normal prior (mean,std) or length 3 for uniform prior defined as (lo_lim, val, uplim).
                **recall the conditions: c1+c2<1, c1>0, c1+c2>0  (https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract)\n
                This implies the a broad uniform prior of [0,2] for c1 and [-1,1] for c2. However, it is highly recommended to use gaussian priors on c1 and c2. 
                    
        """
        #defaults
        c3 = c4 = 0
        bound_lo1 = bound_lo2= bound_lo3 = bound_lo4 = 0
        bound_hi1 = bound_hi2= bound_hi3 = bound_hi4 = 0
        sig_lo1 = sig_lo2 = sig_lo3 = sig_lo4 = 0
        sig_hi1 = sig_hi2 = sig_hi3 = sig_hi4 = 0
        step1 = step2 = step3 = step4 = 0     

        DA = _reversed_dict(locals().copy())
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")

        nfilt = len(self._filnames)

        for par in DA.keys():
            if isinstance(DA[par], (int,float)): DA[par] = [DA[par]]*nfilt
            elif isinstance(DA[par], tuple): 
                if len(DA[par])==2 or len(DA[par])==3: DA[par] = [DA[par]]*nfilt
            elif isinstance(DA[par], list): assert len(DA[par]) == nfilt,f"length of list {par} must be equal to number of unique filters (={nfilt})."
            else: _raise(TypeError, f"{par} must be int/float, or tuple of len 2 (for gaussian prior) or 3 (for uniform prior) but {DA[par]} is given.")
        
        for par in ["c1","c2","c3","c4"]:
            for i,d in enumerate(DA[par]):
                if isinstance(d, (int,float)):  #fixed
                    DA[par][i] = d
                    DA[f"step{par[-1]}"][i] = DA[f"bound_lo{par[-1]}"][i] = DA[f"bound_hi{par[-1]}"][i] = 0
                elif isinstance(d, tuple):
                    if len(d) == 2:  #normal prior
                        DA[par][i] = d[0]
                        DA[f"sig_lo{par[-1]}"][i] = DA[f"sig_hi{par[-1]}"][i] = d[1]
                        DA[f"bound_lo{par[-1]}"][i] = 0 if par=="c1" else -1
                        DA[f"bound_hi{par[-1]}"][i] = 2 if par=="c1" else 1
                        DA[f"step{par[-1]}"][i] = 0.001 if d[1] else 0  #if width is > 0


                    if len(d) == 3:  #uniform prior
                        DA[par][i] = d[1]
                        DA[f"bound_lo{par[-1]}"][i] = d[0]
                        DA[f"bound_hi{par[-1]}"][i] = d[2]
                        DA[f"sig_lo{par[-1]}"][i] = DA[f"sig_hi{par[-1]}"][i] = 0
                        DA[f"step{par[-1]}"][i] = 0.001 if (d[0] or d[2]) else 0 #if bounds !=  0
  
        DA["priors"] = [0]*nfilt
        for i in range(nfilt):
            DA["priors"][i] = "y" if np.any( [DA["sig_lo1"][i], DA["sig_lo2"][i],DA["sig_lo3"][i], DA["sig_lo4"][i] ]) else "n"

        self._ld_dict = DA
        if verbose: _print_output(self,"limb_darkening")

    def contamination_factors(self, cont_ratio=0, err = 0, verbose=True):
        """
            add contamination factor for each unique filter defined from load_lightcurves().

            Paramters:
            ----------
            cont_ratio: list, float;
                ratio of contamination flux to target flux in aperture for each filter. The order of list follows lc_data._filnames.
                very unlikely but if float, same cont_ratio is used for all filters.

            err : list, float;
                error of the contamination flux

        """


        DA = _reversed_dict(locals().copy())
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")

        nfilt = len(self._filnames)

        for par in DA.keys():
            assert isinstance(DA[par], (int,float)) or (isinstance(DA[par], list) and len(DA[par]) == nfilt), f"length of input {par} must be equal to the length of unique filters (={nfilt}) or float."
            if isinstance(DA[par], (int,float)): DA[par] = [DA[par]]*nfilt

        self._contfact_dict = DA
        if verbose: _print_output(self,"contamination")

    def stellar_parameters(self,R_st=None, M_st=None, par_input = "MR", verbose=True):
        """
            input parameters of the star

            Parameters:
            -----------

            R_st, Mst : tuple of length 2 or 3;
                stellar radius and mass (in solar units) to use for calculating absolute dimensions.
                First tuple element is the value and the second is the uncertainty. use a third element if asymmetric uncertainty
            
            par_input : str;
                input method of stellar parameters. It can be "Rrho","Mrho" or "MR", to use the combination of 2 stellar params to get the third.

        """


        DA = _reversed_dict(locals().copy())
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")
        
        for par in ["R_st", "M_st"]:
            assert DA[par] is None or isinstance(DA[par],tuple), f"{par} must be either None or tuple of length 2 or 3 "
            if DA[par] is None: DA[par] = (1,0.01)
            if isinstance(DA[par],tuple):
                assert len(DA[par])==2 or len(DA[par]) <=3, f"length of {par} tuple must be 2 or 3 "
                if len(DA[par])== 2: DA[par]= (DA[par][0], DA[par][1], DA[par][1])
        
        assert DA["par_input"] in ["Rrho","Mrho", "MR"], f"par_input must be one of 'Rrho','Mrho' or 'MR'. "
            
        self._stellar_dict = DA
         
        if verbose: _print_output(self,"stellar_pars")

    def __repr__(self):
        data_type = str(self.__class__).split("load_")[1].split("'>")[0]
        return f'Object containing {len(self._names)} {data_type}\nFiles:{self._names}\nFilepath: {self._fpath}'
 
    def print(self, section="all"):
        """
            Print out all input configuration (or particular section) for the light curve object. 
            It is printed out in the format of the legacy CONAN config file.
            Parameters:
            ------------
            section : str (optional) ;
                section of configuration to print.
                Must be one of ["lc_baseline", "gp", "transit_rv_pars", "depth_variation", "occultations", "limb_darkening", "contamination", "stellar_pars"].
                Default is 'all' to print all sections.
        """
        if section=="all":
            _print_output(self,"lc_baseline")
            _print_output(self,"gp")
            _print_output(self,"transit_rv_pars")
            _print_output(self,"depth_variation")
            _print_output(self,"occultations")
            _print_output(self,"limb_darkening")
            _print_output(self,"contamination")
            _print_output(self,"stellar_pars")
        else:
            possible_sections= ["lc_baseline", "gp", "transit_rv_pars", "depth_variation",
                                 "occultations", "limb_darkening", "contamination", "stellar_pars"]
            assert section in possible_sections, f"{section} not a valid section of `lc_data`. \
                section must be one of {possible_sections}."
            _print_output(self, section)

    def plot(self, plot_cols=(0,1,2), col_labels=None, nrow_ncols=None, figsize=None, fit_order=0, return_fig=False):
        """
            visualize data

            Parameters:
            -----------
            plot_cols : tuple of length 2 or 3;
                Tuple specifying which columns in input file to plot. 
                Default is (0,1,2) to plot column 0 against 1, and 2 as errorbar (i.e. time against flux with fluxerr). 
                Use (3,1,2) to show the correlation between column 3 and the flux. 
                Using tuple of length 2 does not plot errorbars. e.g (3,1).

            col_labels : tuple of length 2;
                label of the given columns in plot_cols. Default is ("time", "flux").
            
            nrow_ncols : tuple of length 2;
                Number of rows and columns to plot the input files. 
                Default is None to find the best layout.

            fit_order : int;
                order of polynomial to fit to the plotted data columns to visualize correlation.
            
            figsize: tuple of length 2;
                Figure size. If None, (8,5) is used for a single input file and optimally determined for more inputs.

            return_fig  : bool;
                return figure object for saving to file.
        """
        if not (isinstance(plot_cols, tuple) and len(plot_cols) in [2,3]): 
            raise TypeError(f"plot_cols must be tuple of length 2 or 3, but is {type(plot_cols)} and length of {len(plot_cols)}.")
        
        assert col_labels is None or ((isinstance(col_labels, tuple) and len(col_labels)==2)), \
            f"col_labels must be tuple of length 2, but is {type(col_labels)} and length of {len(col_labels)}."
        
        assert isinstance(fit_order,int),f'fit_order must be an integer'
        
        if col_labels is None:
            col_labels = ("time", "flux") if plot_cols[:2] == (0,1) else (f"column[{plot_cols[0]}]",f"column[{plot_cols[1]}]")
        
        if self._names != []:
            fig = _plot_data(self, plot_cols=plot_cols, col_labels = col_labels, nrow_ncols=nrow_ncols, figsize=figsize, fit_order=fit_order)
            if return_fig: return fig
        else: print("No data to plot")
    
        
    
        
#rv data
class load_rvs:
    """
        rv object to hold lightcurves for analysis
        
        Parameters:
        -----------
        
        data_filepath : str;
            filepath where rvs files are located
            
        file_list : list;
            list of filenames for the rvs

        Returns:
        --------
        rv_data : rv object
    """
    def __init__(self, file_list=None, data_filepath=None):
        self._obj_type = "rv_obj"
        self._fpath = os.getcwd() if data_filepath is None else data_filepath
        self._names   = [] if file_list is None else file_list  
        if self._names == []:
            self.rv_baseline(verbose=False)
        else: 
            for rv in self._names: assert os.path.exists(self._fpath+rv), f"file {rv} does not exist in the path {self._fpath}."
            print("Next: use method `rv_baseline` to define baseline model for for the each rv")

        self._nRV = len(self._names)

    def rv_baseline(self, dt=None, dbis=None, dfwhm=None, dcont=None,sinPs=None,
                    gammas_kms=0.0, gam_steps=0.01, 
                    verbose=True):
        
        """
            Define rv baseline model parameters to fit.
            Each baseline model parameter should be a list of numbers specifying the polynomial order for each rv data.
            e.g. Given 3 input rvs, and one wishes to fit a 2nd order time trend to only the first and third lightcurves,
            then dt = [2, 0, 2].

            dt, dbis, dfwhm,dcont: list of ints;
                decorrelatation paramters: time, bis, fwhm, contrast
                
            gammas_kms: tuple,floats or list of tuple/float;
                specify if to fit for gamma. if float/int, it is fixed to this value. If tuple of len 2 it is fitted gaussian prior as (prior_mean, width). 
        """

        # assert self._names != [], "No rv files given"
        # assert 
        if self._names == []: 
            if verbose: _print_output(self,"rv_baseline")
            return 
        
        if isinstance(gammas_kms, list): assert len(gammas_kms) == self._nRV, f"gammas_kms must be type tuple/int or list of tuples/floats/ints of len {self._nRV}."
        elif isinstance(gammas_kms, (tuple,float,int)): gammas_kms=[gammas_kms]*self._nRV
        else: _raise(TypeError, f"gammas_kms must be type tuple/int or list of tuples/floats/ints of len {self._nRV}." )
        
        gammas,prior,gam_pri,sig_lo,sig_hi = [],[],[],[],[]
        for g in gammas_kms:
            #fixed gammas
            if isinstance(g, (float,int)):
                prior.append("n")
                gammas.append(g)
                gam_pri.append(g)
                sig_lo.append(0)
                sig_hi.append(0)
            #fit gammas
            elif isinstance(g, tuple) and len(g)==2:
                prior.append("y")
                gammas.append(g[0])
                gam_pri.append(g[0])
                sig_lo.append(g[1])
                sig_hi.append(g[1])   
            else: _raise(TypeError, f"a tuple of len 2, float or int  was expected but got the value {g} in gammas_kms.")

        dict_args = locals().copy()     #get a dictionary of the input/variables arguments for easy manipulation
        _ = dict_args.pop("self")            #remove self from dictionary
        _ = [dict_args.pop(item) for item in ["verbose","gammas_kms","g"]]


        for par in dict_args.keys():
            assert dict_args[par] is None or isinstance(dict_args[par], (int,float)) or (isinstance(dict_args[par], (list,np.ndarray)) and len(dict_args[par]) == self._nRV), f"parameter {par} must be a list of length {self._nRV} or int (if same degree is to be used for all RVs) or None (if not used in decorrelation)."
            
            if dict_args[par] is None: dict_args[par] = [0]*self._nRV
            elif isinstance(dict_args[par], (int,float,str)): dict_args[par] = [dict_args[par]]*self._nRV
            

        self._RVbases = [ [dict_args["dt"][i], dict_args["dbis"][i], dict_args["dfwhm"][i], dict_args["dcont"][i],dict_args["sinPs"][i]] for i in range(self._nRV) ]

        self._gammas = dict_args["gammas"]
        self._gamsteps = dict_args["gam_steps"]
        self._gampri = dict_args["gam_pri"]
        
        self._prior = dict_args["prior"]
        self._siglo = dict_args["sig_lo"]
        self._sighi = dict_args["sig_hi"]
        
        gampriloa=[]
        gamprihia=[]
        for i in range(self._nRV):
            gampriloa.append( 0. if (self._prior[i] == 'n' or self._gamsteps[i] == 0.) else self._siglo[i])
            gamprihia.append( 0. if (self._prior[i] == 'n' or self._gamsteps[i] == 0.) else self._sighi[i])
        
        self._gamprilo = gampriloa                
        self._gamprihi = gamprihia                
        self._sinPs = dict_args["sinPs"]
        
        if verbose: _print_output(self,"rv_baseline")
    
    def __repr__(self):
        data_type = str(self.__class__).split("load_")[1].split("'>")[0]
        return f'Object containing {len(self._names)} {data_type}\nFiles:{self._names}\nFilepath: {self._fpath}'
        
    def plot(self, plot_cols=(0,1,2), col_labels=None, nrow_ncols=None, figsize=None, fit_order=0, return_fig=False):
        """
            visualize data

            Parameters:
            -----------
            plot_cols : tuple of length 3;
                Tuple specifying which columns in input file to plot. 
                Default is (0,1,2) to plot time, flux with fluxerr. 
                Use (3,1,2) to show the correlation between the 4th column and the flux. 

            col_labels : tuple of length 2;
                label of the given columns in plot_cols. Default is ("time", "rv").

            nrow_ncols : tuple of length 2;
                Number of rows and columns to plot the input files. 
                Default is (None, None) to find the best layout.
            
            figsize: tuple of length 2;
                Figure size. If None, (8,5) is used for a single input file and optimally determined for more inputs.

            return_fig  : bool;
                return figure object for saving to file.
        """

        if not (isinstance(plot_cols, tuple) and len(plot_cols) in [2,3]): 
            raise TypeError(f"plot_cols must be tuple of length 2 or 3, but is {type(plot_cols)} and length of {len(plot_cols)}.")
        
        assert col_labels is None or ((isinstance(col_labels, tuple) and len(col_labels)==2)), \
            f"col_labels must be tuple of length 2, but is {type(col_labels)} and length of {len(col_labels)}."
        
        assert isinstance(fit_order,int),f'fit_order must be an integer'
        
        if col_labels is None:
            col_labels = ("time", "rv") if plot_cols[:2] == (0,1) else (f"column[{plot_cols[0]}]",f"column[{plot_cols[1]}]")
        
        if self._names != []:
            fig = _plot_data(self, plot_cols=plot_cols, col_labels = col_labels, nrow_ncols=nrow_ncols, fit_order=fit_order, figsize=figsize)
            if return_fig: return fig
        else: print("No data to plot")
    
    def print(self):
        _print_output(self, "rv_baseline")
    
class mcmc_setup:
    """
        class to setup fitting
    """
    def __init__(self, n_chains=64, n_steps=2000, n_burn=500, n_cpus=2, sampler=None,
                    leastsq_for_basepar="n", apply_CFs="y",apply_jitter="y",
                    verbose=True, remove_param_for_CNM="n", lssq_use_Lev_Marq="n",
                    GR_test="y", make_plots="n", leastsq="y", savefile="output_ex1.npy",
                    savemodel="n", adapt_base_stepsize="y"):
        """
            configure mcmc run
            
            Parameters:
            ----------
            n_chains: int;
                number of chains/walkers
            
            n_steps: int;
                length of each chain. the effective total steps becomes n_steps*n_chains.

            n_burn: int;
                number of steps to discard as burn-in
            
            n_cpus: int;
                number of cpus to use for parallelization.
            
            sampler: int;
                sampler algorithm to use in traversing the parameter space. Options are ["demc","snooker"].
                if None, the default emcee StretchMove is used.

            leastsq_for_basepar: "y" or "n";
                whether to use least-squares fit within the mcmc to fit for the baseline. This reduces +\
                the computation time especially in cases with several input files. Default is "n".

            apply_jitter: "y" or "n";
                whether to apply a jitter term for the fit of RV data. Default is "y".
            
            Other keyword arguments to the emcee sampler function `run_mcmc` can be given in the call to `CONAN3.fit_data`.
        """
        
        DA = _reversed_dict(locals().copy())
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")
        self._obj_type = "mcmc_obj"
        self._mcmc_dict = DA
            
        if verbose: _print_output(self,"mcmc")

    def __repr__(self):
        return f"mcmc setup: steps:{self._mcmc_dict['n_steps']} \nchains: {self._mcmc_dict['n_chains']}"

    def print(self):
        _print_output(self, "mcmc")

                   


def create_configfile(lc, rv, mcmc, filename="input_config.dat"): 
    """
        create configuration file that of lc, rv, amd mcmc setup.
        
        Parameters:
        -----------
        lc : object;
            Instance of CONAN.load_lightcurve() object and its attributes.

        rv : object, None;
            Instance of CONAN.load_rvs() object and its attributes.
        
        mcmc : object;
            Instance of CONAN.setup_fit() object and its attributes.
    """
    f = open(filename,"w")
    f.write("#=========== MCMC input file =======================\n")
    f.write("Path_of_input_lightcurves:\n")
    f.write(lc._fpath+"\n")

    _print_output(lc,"lc_baseline",file=f)
    _print_output(lc,"gp",file=f)
    _print_output(rv,"rv_baseline",file=f)
    _print_output(lc,"transit_rv_pars",file=f)
    _print_output(lc,"depth_variation",file=f)
    _print_output(lc,"occultations",file=f)
    _print_output(lc,"limb_darkening",file=f)
    _print_output(lc,"contamination",file=f)
    _print_output(lc,"stellar_pars",file=f)
    _print_output(mcmc, "mcmc",file=f)

    f.close()


def load_configfile(configfile="input_config.dat", return_fit=False, verbose=True):
    """
        configure conan from specified configfile.
        
        Parameters:
        -----------
        configfile: filepath;
            path to configuration file.

        return_fit: bool;
            whether to immediately perform the fit from this function call.
            if True, the result object from the fit is also returned

        verbose: bool;
            show print statements

        Returns:
        --------
        lc_data, rv_data, mcmc. if return_fit is True, the result object of fit is also returned

        lc_data: object;
            light curve data object generated from `conan3.load_lighturves`.
        
        rv_data: object;
            rv data object generated from `conan3.load_rvs`
            
        mcmc: object;
            fitting object generated from `conan3.setup_fit`.

        result: object;
            result object containing chains of the mcmc fit.
    
    """
    _file = open(configfile,"r")
    _skip_lines(_file,2)                       #remove first 2 comment lines
    fpath= _file.readline().rstrip()           # the path where the files are
    _skip_lines(_file,2)                       #remove 2 comment lines


 # ========== Lightcurve input ====================
    _names=[]                    # array where the LC filenames are supposed to go
    _filters=[]                  # array where the filter names are supposed to go
    _lamdas=[]
    _bases=[]                    # array where the baseline exponents are supposed to go
    _groups=[]                   # array where the group indices are supposed to go
    _grbases=[]
    _useGPphot=[]
    _skip_lines(_file,1)

    #read specification for each listed light-curve file
    dump = _file.readline() 
    while dump[0] != '#':           # if it is not starting with # then
        _adump = dump.split()          # split it

        _names.append(_adump[0])      # append the first field to the name array
        _filters.append(_adump[1])    # append the second field to the filters array
        _lamdas.append(float(_adump[2]))    # append the second field to the filters array
        strbase=_adump[3:11]         # string array of the baseline function exponents
        base = [int(i) for i in strbase]
        _bases.append(base)
        group = int(_adump[11])
        _groups.append(group)
        grbase=int(_adump[10])
        _useGPphot.append(_adump[12])
        _grbases.append(grbase)

        #move to next LC
        dump =_file.readline() 

    
    dump=_file.readline()        # read the next line
    dump=_file.readline()        # read the next line

    # ========== GP input ====================
    gp_namelist, gp_pars, kernels, WN = [],[],[],[]
    log_scale, log_metric, s_step, m_step = [],[],[],[]
    

    while dump[0] != '#':  
        adump=dump.split()   
        gp_namelist.append(adump[0]) 
        gp_pars.append(adump[1])
        kernels.append(adump[2])
        WN.append(adump[3])

        s_step.append(float(adump[5]))
        m_step.append(float(adump[11]))

        #gp scale
        if float(adump[7]) == 0.0:    #prior width ==0
            #uniform prior
            lo_lim = float(adump[9])
            up_lim = float(adump[8])
            scale  = float(adump[4])
            log_scale.append( (lo_lim, np.log(scale), up_lim) )
        else:
            #gaussian prior
            prior_mean = float(adump[6])
            width = float(adump[7])
            log_scale.append( (prior_mean, width) )

        #gp metric
        if float(adump[13]) == 0.0:    #prior width ==0
            #uniform prior
            lo_lim = float(adump[15])
            up_lim = float(adump[14])
            metric = float(adump[10])
            log_metric.append( (lo_lim, np.log(metric), up_lim) )
        else:
            #gaussian prior
            prior_mean = float(adump[12])
            width = float(adump[13])
            log_metric.append( (prior_mean, width) )

        dump=_file.readline()        # read the next line

    lc_data = load_lightcurves(_names, fpath, _filters, _lamdas,verbose)
    lc_data.lc_baseline(*np.array(_bases).T, grp_id=_groups, gp=_useGPphot,verbose=verbose )
    lc_data.add_GP(gp_namelist,gp_pars,kernels,WN, 
                    log_scale, s_step, log_metric, m_step,verbose=verbose)

    _skip_lines(_file,2)
    dump=_file.readline()

 # ========== RV input ====================

    RVnames=[]
    RVbases=[]
    gammas=[]
    gamsteps=[]
    gampri=[]
    gamprilo=[]
    gamprihi=[]
    sinPs=[]    

    while dump[0] != '#':           # if it is not starting with # then
        adump=dump.split()
        RVnames.append(adump[0])      # append the first field to the RVname array
        strbase=adump[1:6]         # string array of the baseline function exponents 
        base = [int(i) for i in strbase]
        RVbases.append(base)
        gammas.append(float(adump[6]))
        gamsteps.append(float(adump[7]))
        gampri.append(float(adump[9]))
        gampriloa = (0. if (adump[8] == 'n' or adump[7] == 0.) else float(adump[10]))
        gamprilo.append(gampriloa)
        gamprihia = (0. if (adump[8] == 'n' or adump[7] == 0.) else float(adump[11]))
        gamprihi.append(gamprihia)
        sinPs.append(adump[5])
        dump=_file.readline()

    gamm = [((g,e) if e!=0 else g) for g,e in zip(gampri,gamprilo)]


    rv_data = load_rvs(RVnames,fpath)
    rv_data.rv_baseline(*np.array(RVbases).T, gammas_kms=gamm,
                        gam_steps=gamsteps,verbose=verbose)  
    
 #========== transit and rv model paramters=====================
    dump=_file.readline()
    dump=_file.readline()

    model_par = {}
    for _ in range(8):
        adump=dump.split()
        
        par_name = adump[0]
        fit   = adump[1]
        val   = float(adump[2])
        step  = float(adump[3])
        lo_lim= float(adump[4])
        up_lim= float(adump[5])
        prior = adump[6]
        pr_width_lo= float(adump[8])
        pr_width_hi= float(adump[9])

        if par_name == "K_[m/s]":  par_name = "K"

        if fit == "n" or step==0: model_par[par_name] = val
        else:
            model_par[par_name] = ( (lo_lim,val,up_lim) if prior =="n"  else (val,pr_width_lo)  ) #unform if prior is n else gaussian
            
        dump=_file.readline()

    lc_data.setup_transit_rv(**model_par,verbose=verbose)

 #========== depth variation=====================
    dump=_file.readline()
    dump=_file.readline()
    adump=dump.split()   
    
    ddf = adump[0]
    step = float(adump[1])
    bounds=(float(adump[2]), float(adump[3]))
    prior = adump[4]
    pr_width = (float(adump[5]),float(adump[6]))
    div_white= adump[7]

    dump=_file.readline()
    dump=_file.readline()

    depth_per_group = []
    while dump[0] != '#':           # if it is not starting with # then
        adump=dump.split()   
        depth_per_group.append( (float(adump[1]),float(adump[2])))
        dump=_file.readline()


    lc_data.transit_depth_variation(ddf,depth_per_group,div_white,
                                    step,bounds,prior,pr_width,
                                    verbose)

 #=========== occultation setup ===========================
    dump=_file.readline()
    dump=_file.readline()

    filts,depths,step = [],[],[]

    while dump[0] != '#':           # if it is not starting with # then
        adump=dump.split()   
        filts.append(adump[0])
        val = float(adump[2])
        step.append (float(adump[3]))
        lo_lim= float(adump[4])
        up_lim= float(adump[5])
        prior = adump[6]
        pr_width = float(adump[8])
        
        depths.append( (lo_lim,val,up_lim) if prior=="n" else (val,pr_width) )
        
        dump=_file.readline()
        
    lc_data.setup_occultation(filts, depths, step,verbose)

 #=========== Limb darkening setup ==================
    dump=_file.readline()
    dump=_file.readline()

    c1,c2,step =[],[],[]
    while dump[0] != '#':
        adump=dump.split()
        step    = ( float(adump[3]), float(adump[9]))

        sig_lo1 = float(adump[4])
        lo_lim1 = float(adump[6])
        hi_lim1 = float(adump[7])

        sig_lo2 = float(adump[10])
        lo_lim2 = float(adump[12])
        hi_lim2 = float(adump[13])

        if step[0]: c1.append( (float(adump[2]),sig_lo1) if sig_lo1 else (lo_lim1,float(adump[2]),hi_lim1) )
        else: c1.append(float(adump[2]))

        if step[1]: c2.append( (float(adump[8]),sig_lo2) if sig_lo2 else (lo_lim2,float(adump[8]),hi_lim2) )
        else: c2.append(float(adump[8]))

        dump=_file.readline()

    lc_data.limb_darkening(c1,c2,verbose )

 #=========== contamination setup === 
    dump=_file.readline()
    dump=_file.readline() 

    cont, err = [],[]
    while dump[0] != '#':
        adump=dump.split()
        cont.append(float(adump[1]))
        err.append(float(adump[2]))
        dump=_file.readline()

    lc_data.contamination_factors(cont,err,verbose)

 #=========== Stellar input properties ===========================
    dump=_file.readline()
    dump=_file.readline() 
    
    adump=dump.split()
    Rst = ((float(adump[1]),float(adump[2]),float(adump[3])))
    dump=_file.readline()
    adump=dump.split() 
    Mst = ((float(adump[1]),float(adump[2]),float(adump[3])))
    dump=_file.readline()
    adump=dump.split()
    howstellar = adump[1]

    lc_data.stellar_parameters(Rst,Mst,howstellar,verbose)

 #=========== MCMC setup ======================================
    dump=_file.readline()
    dump=_file.readline()

    adump=dump.split() 
    nsamples=int(adump[1])   # total number of integrations
    
    dump=_file.readline()
    adump=dump.split()
    nchains=int(adump[1])  #  number of chains
    ppchain = int(nsamples/nchains)  # number of points per chain
    
    dump=_file.readline()
    adump=dump.split()
    nproc=int(adump[1])   #  number of processes
    
    dump=_file.readline()
    adump=dump.split()
    burnin=int(adump[1])    # Length of bun-in
    
    dump=_file.readline()
    adump=dump.split()
    walk=adump[1]            # Differential Evolution?

    dump = _file.readline()
    adump=dump.split()
    grtest=adump[1]         #GRTest?

    dump = _file.readline()
    adump=dump.split()
    makeplots=adump[1]         #Make plots?

    dump = _file.readline()
    adump=dump.split()
    least_sq=adump[1]         #Least squares??

    dump = _file.readline()
    adump=dump.split()
    save_file=adump[1]         #Output file?

    dump = _file.readline()
    adump=dump.split()
    save_model=adump[1]         #Save the model??

    dump = _file.readline()
    adump=dump.split()
    adaptbasestepsize=adump[1]         #Adapt the stepsize of bases?

    dump = _file.readline()
    adump=dump.split()
    removeparamforCNM=adump[1]         #Remove paramameter for CNM?

    dump = _file.readline()
    adump=dump.split()
    leastsqforbasepar=adump[1]         #Least-squares for base parameters?

    dump = _file.readline()
    adump=dump.split()
    lssquseLevMarq=adump[1]         #Use Lev-Marq for least squares?

    dump = _file.readline()
    adump=dump.split()
    applyCFs=adump[1]         #GRTest?

    dump = _file.readline()
    adump=dump.split()
    applyjitter=adump[1]         #GRTest?
 
    mcmc = mcmc_setup(n_chains=nchains, n_steps=ppchain, n_burn=burnin, n_cpus=nproc, sampler=walk, 
                        GR_test=grtest, make_plots=makeplots, leastsq=least_sq, savefile=save_file,
                         savemodel=save_model, adapt_base_stepsize=adaptbasestepsize, 
                         remove_param_for_CNM=removeparamforCNM,leastsq_for_basepar=leastsqforbasepar, 
                         lssq_use_Lev_Marq=lssquseLevMarq, apply_CFs=applyCFs,apply_jitter=applyjitter,
                         verbose=verbose)

    _file.close()

    if return_fit:
        from .fit_data import fit_data
        result =   fit_data(lc_data, rv_data, mcmc) 
        return lc_data,rv_data,mcmc,result

    return lc_data,rv_data,mcmc



class load_chains:
    def __init__(self,chain_file = "chains_dict.pkl", burnin_chain_file="burnin_chains_dict.pkl"):
        assert os.path.exists(chain_file) or os.path.exists(burnin_chain_file) , f"file {chain_file} or {burnin_chain_file}  does not exist in this directory"

        if os.path.exists(chain_file):
            self._chains = pickle.load(open(chain_file,"rb"))
        if os.path.exists(burnin_chain_file):
            self._burnin_chains = pickle.load(open(burnin_chain_file,"rb"))

        self._par_names = self._chains.keys() if os.path.exists(chain_file) else self._burnin_chains
        
    def __repr__(self):
        return f'Object containing chains (main or burn-in) from mcmc. \
                \nParameters in chain are:\n\t {self._par_names} \
                \n\nuse `plot_chains`, `plot_burnin_chains`, `plot_corner` or `plot_posterior` methods on selected parameters to visualize results.'
        
    def plot_chains(self, pars=None, figsize = None, thin=1, discard=0, alpha=0.05,
                    color=None, label_size=12, force_plot = False):
        """
            Plot chains of selected parameters.
              
            Parameters:
            ----------
            pars: list of str;
                parameter names to plot. Plot less than 20 parameters at a time for clarity.
        
            thin : int;
                factor by which to thin the chains in order to reduce correlation.

            discard : int;
                to discard first couple of steps within the chains. 
            
        """
        assert pars is None or isinstance(pars, list) or pars == "all", \
             f'pars must be None, "all", or list of iu8999relevant parameters.'
        if pars is None or pars == "all": pars = [p for p in self._par_names]
        for p in pars:
            assert p in self._par_names, f'{p} is not one of the parameter labels in the mcmc run.'
        
        ndim = len(pars)
        if not force_plot: assert ndim < 21, f'number of parameter chain to plot should be <=20 for clarity. Use force_plot = True to continue anyways.'

        if figsize is None: figsize = (12,6+int(ndim/2))
        fig, axes = plt.subplots(ndim, sharex=True, figsize=figsize)
        if ndim == 1: axes = np.array([axes])
            
        if thin > 1 and discard > 0:
            axes[0].set_title(f"Discarded first {discard} steps & thinned by {thin}", fontsize=14)
        elif thin > 1 and discard == 0:
            axes[0].set_title(f"Thinned by {thin}", fontsize=14)
        else:
            axes[0].set_title(f"Discarded first {discard} steps", fontsize=14)
            
        
        for i,p in enumerate(pars):
            ax = axes[i]
            ax.plot(self._chains[p][:,discard::thin].T,c = color, alpha=alpha)
            ax.legend([pars[i]],loc="upper left")
            ax.autoscale(enable=True, axis='x', tight=True)
        plt.subplots_adjust(hspace=0.0)
        axes[-1].set_xlabel("step number", fontsize=label_size);

        return fig

            
    def plot_burnin_chains(self, pars=None, figsize = None, thin=1, discard=0, alpha=0.05,
                    color=None, label_size=12, force_plot = False):
        """
            Plot chains of selected parameters.
              
            Parameters:
            ----------
            pars: list of str;
                parameter names to plot. Plot less than 20 parameters at a time for clarity.
        
            thin : int;
                factor by which to thin the chains in order to reduce correlation.

            discard : int;
                to discard first couple of steps within the chains. 
        
        """
        assert pars is None or isinstance(pars, list) or pars == "all", \
             f'pars must be None, "all", or list of relevant parameters.'
        if pars is None or pars == "all": pars = [p for p in self._par_names]
        for p in pars:
            assert p in self._par_names, f'{p} is not one of the parameter labels in the mcmc run.'
        
        ndim = len(pars)
        if not force_plot: assert ndim < 21, f'number of parameter chain to plot should be <=20 for clarity. Use force_plot = True to continue anyways.'

        if figsize is None: figsize = (12,6+int(ndim/2))
        fig, axes = plt.subplots(ndim, sharex=True, figsize=figsize)
        if ndim == 1: axes = np.array([axes])
            
        if thin > 1 and discard > 0:
            axes[0].set_title(f"Burn-in\nDiscarded first {discard} steps & thinned by {thin}", fontsize=14)
        elif thin > 1 and discard == 0:
            axes[0].set_title(f"Burn-in\nThinned by {thin}", fontsize=14)
        else:
            axes[0].set_title(f"Burn-in\nDiscarded first {discard} steps", fontsize=14)
            
        
        for i,p in enumerate(pars):
            ax = axes[i]
            ax.plot(self._burnin_chains[p][:,discard::thin].T,c = color, alpha=alpha)
            ax.legend([pars[i]],loc="upper left")
            ax.autoscale(enable=True, axis='x', tight=True)
        plt.subplots_adjust(hspace=0.0)
        axes[-1].set_xlabel("step number", fontsize=label_size);

        return fig
        
    def plot_corner(self, pars=None, bins=20, thin=1, discard=0,
                    q=[0.16,0.5,0.84], range=None,show_titles=True, title_fmt =".3f", titlesize=14,
                    labelsize=20, multiply_by=1, add_value= 0, force_plot = False ):
        """
            Corner plot of selected parameters.
              
            Parameters:
            ----------
            pars : list of str;
                parameter names to plot. Ideally less than 12 pars for clarity of plot

            bins : int;
                number of bins in 1d histogram

            thin : int;
                factor by which to thin the chains in order to reduce correlation.

            discard : int;
                to discard first couple of steps within the chains. 
            
            q : list of floats;
                quantiles to show on the 1d histograms. defaults correspoind to +/-1 sigma
                
            range : iterable (same length as pars);
                A list where each element is either a length 2 tuple containing
                lower and upper bounds or a float in range (0., 1.)
                giving the fraction of samples to include in bounds, e.g.,
                [(0.,10.), (1.,5), 0.999, etc.].
                If a fraction, the bounds are chosen to be equal-tailed.
        """
        assert pars is None or isinstance(pars, list) or pars == "all", \
             f'pars must be None, "all", or list of relevant parameters.'
        if pars is None or pars == "all": pars = [p for p in self._par_names]

        ndim = len(pars)

        if not force_plot: assert ndim <= 12, \
            f'number of parameters to plot should be <=12 for clarity. Use force_plot = True to continue anyways.'

        lsamp = len(self._chains[pars[0]][:,discard::thin].flatten())
        samples = np.empty((lsamp,ndim))

        #adjustments to make values more readable
        if isinstance(multiply_by, (int,float)): multiply_by = [multiply_by]*ndim
        elif isinstance(multiply_by, list): assert len(multiply_by) == ndim
        if isinstance(add_value, (int,float)): add_value = [add_value]*ndim
        elif isinstance(add_value, list): assert len(add_value) == ndim


        for i,p in enumerate(pars):
            assert p in self._par_names, f'{p} is not one of the parameter labels in the mcmc run.'
            samples[:,i] = self._chains[p][:,discard::thin].reshape(-1) * multiply_by[i] + add_value[i]
        
        
        fig = corner.corner(samples, bins=bins, labels=pars, show_titles=show_titles, range=range,
                    title_fmt=title_fmt,quantiles=q,title_kwargs={"fontsize": titlesize},
                    label_kwargs={"fontsize":labelsize})
        
        return fig


    def plot_posterior(self, par, thin=1, discard=0, bins=20, density=True, range=None,
                        q = [0.0015,0.16,0.5,0.85,0.9985], multiply_by=1, add_value=0, 
                        return_values=False):
        """
        Plot the posterior distribution of a single input parameter, par.
        if return_values = True, the summary statistic for the parameter is also returned as an output.  

        Parameters:
        -----------
        par : str;
            parameter posterior to plot

        thin : int;
            thin samples by factor of 'thin'

        discard : int;
            to discard first couple of steps within the chains. 


        Returns:
        --------
        fig: figure object

        result: tuple of len 3;
            summary statistic for the parameter, par, in the order [median, -1sigma, +1sigma] 

        """
        assert isinstance(par, str), 'par must be a single parameter of type str'
        assert par in self._par_names, f'{par} is not one of the parameter labels in the mcmc run.'
        assert isinstance(q, (float, list)),"q must be either a single float or list of length 1, 3 or 5"
        if isinstance(q,float): q = [q]
        
        par_samples = self._chains[par][:,discard::thin].flatten() * multiply_by + add_value
        quants = np.quantile(par_samples,q)

        if len(q)==1:
            ls = ['-']; c=["r"]
            med = quants[0]
        elif len(q)==3: 
            ls = ["--","-","--"]; c = ["r"]*3 
            med = quants[1]; sigma_1 = np.diff(quants)
        elif len(q)==5: 
            ls = ["--","--","-","--","--"]; c =["k",*["r"]*3,"k"] 
            med=quants[2]; sigma_1 = np.diff(quants[1:4])
        else: _raise(ValueError, "q must be either a single float or list of length 1, 3 or 5")

        fig  = plt.figure()
        plt.hist(par_samples, bins=bins, density=density, range=range);
        [plt.axvline(quants[i], ls = ls[i], c=c[i], zorder=3) for i in np.arange(len(quants))]
        if len(q)==1:
            plt.title(f"{par}={med:.4f}")
        else:
            plt.title(f"{par}={med:.4f}$^{{+{sigma_1[1]:.4f}}}_{{-{sigma_1[0]:.4f}}}$")

        plt.xlabel(par);

        if return_values:
            result = [med, sigma_1[0],sigma_1[1]]
            return fig, result

        return fig



def load_result_array():
    """
        Load result array from CONAN3 fit allowing for customised plots.
        All files with '_out_full.dat' are loaded. 

        Returns:
        --------
            results : dict;
                dictionary of holding the arrays for each output file.
            
        Examples
        --------
        >>> import CONAN3
        >>> results = CONAN3.load_result_array()
        >>> list(results.keys())
        ['lc8det_out_full.dat', 'lc6bjd_out_full.dat']

        >>> df1 = results['lc8det_out_full.dat']
        >>> df1.keys()
        ['time', 'flux', 'error', 'full_mod', 'gp*base', 'transit', 'det_flux']

        >>> #plot arrays
        >>> plt.plot(df["time"], df["flux"],"b.")
        >>> plt.plot(df["time"], df["gp*base"],"r")
        >>> plt.plot(df["time"], df["transit"],"g")
        
    """
    out_files = [ f  for f in os.listdir() if '_out_full.dat' in f]
    results = {}
    for f in out_files:
        df = pd.read_fwf(f, header=0)
        df = df.rename(columns={'# time': 'time'})
        results[f] = df
    print(f"Output files loaded are: {out_files} ")
    return results