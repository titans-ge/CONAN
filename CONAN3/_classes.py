from typing import Type
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
from warnings import warn
from ldtk import SVOFilter
from .utils import outlier_clipping, rho_to_aR, rho_to_tdur
from copy import deepcopy

__all__ = ["load_lightcurves", "load_rvs", "mcmc_setup", "load_result", "__default_backend__","load_result_array"]

#helper functions
__default_backend__ = matplotlib.get_backend()
matplotlib.use(__default_backend__)

def _plot_data(obj, plot_cols, col_labels, nrow_ncols=None, figsize=None, 
                fit_order=0, model_overplot=None):
    """
    Takes a data object (containing light-curves or RVs) and plots them.
    """

    n_data = len(obj._names)
    tsm= True if plot_cols[0]==0 else False
    cols = plot_cols+(1,) if len(plot_cols)==2 else plot_cols

    if n_data == 1:
        p1, p2, p3 = np.loadtxt(obj._fpath+obj._names[0], usecols=cols, unpack=True )
        if len(plot_cols)==2: p3 = None
        if figsize is None: figsize=(8,5)
        fig = plt.figure(figsize=figsize)
        plt.errorbar(p1,p2,yerr=p3, fmt=".", color="b", ecolor="gray",label=f'{obj._names[0]}')
        if model_overplot:
            plt.plot(p1,model_overplot[0][0],"r",zorder=3,label="detrend_model")
            if tsm: plt.plot(model_overplot[0][2],model_overplot[0][3],"c",zorder=3,label="tra/rv_model")   #smooth model plot if time on x axis
            else: plt.plot(p1,model_overplot[0][1],"c",zorder=3,label="tra/rv_model")


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
                if tsm: ax[i].plot(model_overplot[i][2],model_overplot[i][3],"c",zorder=3,label="tra/occ_model")
                else: ax[i].plot(p1,model_overplot[i][1],"c",zorder=3,label="tra/occ_model")

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

def _decorr(file, T_0=None, Period=None, rho_star=None, L=0, Impact_para=0, RpRs=None, Eccentricity=0, omega=90, u1=0, u2=0,
                mask=False, decorr_bound=(-1,1),
                offset=None, A0=None, B0=None, A3=None, B3=None,
                A4=None, B4=None, A5=None, B5=None,
                A6=None, B6=None, A7=None, B7=None,
                A5_2=None, B5_2=None, A5_3=None,B5_3=None,
                cheops=False, npl=1,return_models=False):
    """
    linear decorrelation with different columns of data file. It performs a linear model fit to the 3rd column of the file.
    It uses columns 0,3,4,5,6,7 to construct the linear trend model.
    
    Parameters:
    -----------
    file : str;
        path to data file with columns 0 to 8 (col0-col8).
    
    T_0, Period, rho_star, L, Impact_para, RpRs, Eccentricity, omega : floats, None;
        transit/eclipse parameters of the planet. T_0 and P must be in same units as the time axis (cols0) in the data file. rho_star is the stellar density in g/cm^3.
        if float/int, the values are held fixed. if tuple/list of len 2 implies [min,max] while len 3 implies [min,start_val,max].
        
    u1,u2 : float  (optional);
        standard quadratic limb darkening parameters.

    mask : bool ;
        if True, transits and eclipses are masked using T_0, P and rho_star which must be float/int.                    
        
    offset, Ai, Bi; floats [-1,1] or None;
        coefficients of linear model where offset is the intercept. they have large bounds [-1,1].
        Ai, Bi are the linear and quadratic term of the model against column i. A0*col0 + A0*col0**2 for time trend
    
    cheops : Bool;
        True, if data is from CHEOPS with col5 being the roll-angle. 
        In this case, a linear fourier model up to 3rd harmonic in roll-angle  is used for col5 

    npl : int; 
        number of planets in the system. default is 1.
        
    return_models : Bool;
        True to return trend model and transit/eclipse model.
         
    Returns:
    -------
    result: object;
        result object from fit with several attributes such as result.bestfit, result.params, result.bic, ...
        if return_models = True, returns (trend_model, transit/eclipse model)
    """
    DA = locals().copy()
    _       = DA.pop("file")
    tr_pars = {}

    ff = np.loadtxt(file,usecols=(0,1,2,3,4,5,6,7,8))
    dict_ff = {}
    for i in range(8): dict_ff[f"cols{i}"] = ff[:,i]
    df = pd.DataFrame(dict_ff)  #pandas dataframe

    cols0_med = np.median(df["cols0"])
                          
    if mask:
        print("masking transit/eclipse phases")
        for tp in ["T_0", "Period", "rho_star"]:
            if isinstance(in_pars[tp], tuple):
                if len(in_pars[tp])==2:   in_pars[tp]=in_pars[tp][0]
                elif len(in_pars[tp])==3: in_pars[tp]=in_pars[tp][1]
            # assert isinstance(in_pars[tp], (float,int)),f"{tp} must be  float/int for masking transit/eclipses"
        #use periodicity of 0.5*P to catch both transits and eclipses
        E = np.round(( cols0_med - in_pars['T_0'])/(0.5*in_pars["Period"]) )
        Tc = E*(0.5*in_pars["Period"]) + in_pars['T_0']
        duration = rho_to_tdur(in_pars["rho_star"], in_pars["Impact_para"], in_pars["RpRs"],in_pars["Period"], in_pars["Eccentricity"], in_pars["omega"])
        mask = abs(df["cols0"] - Tc) > 0.5*duration
        df = df[mask]
        
    

    #transit variables
    for p in ["T_0", "Period", "rho_star", "L", "Impact_para","RpRs", "Eccentricity", "omega", "u1","u2"]:
        for n in range(npl):
            lbl = f"_{n+1}" if npl>1 else ""      # numbering to add to parameter names of each planet
            if p not in ["u1","u2","rho_star"]:   # parameters common to all planet
                tr_pars[p+lbl]= DA[p][n]  #transit/eclipse pars
            else:
                tr_pars[p] = DA[p]        #limb darkening pars

    #decorr variables    
    decorr_vars = [f"{L}{i}" for i in [0,3,4,5,6,7] for L in ["A","B"]]  + ["A5_2","B5_2","A5_3","B5_3","offset"]
    in_pars = {k:v for k,v in DA.items() if k in decorr_vars}

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
            vs = ["RpRs","Period","rho_star"]
            vs = [v+(f"_{n+1}" if npl>1 else "") for n in range(npl) for v in vs]    
            val = 1e-10 if key in vs else 0 #allows to obtain transit/eclipse with zero depth
            tr_params.add(key, value=val, vary=False)
                
    
    def transit_occ_model(tr_params,t=None,npl=1):
        if t is None: t = df["cols0"].values
        model_flux = np.zeros_like(t)

        for n in range(1,npl+1):
            lbl = f"_{n}" if npl>1 else ""
            
            bt = batman.TransitParams()
            bt.per = tr_params["Period"+lbl]
            bt.t0  = tr_params["T_0"+lbl]
            bt.fp  = tr_params["L"+lbl]
            bt.rp  = tr_params["RpRs"+lbl]
            b      = tr_params["Impact_para"+lbl]
            bt.ecc = tr_params["Eccentricity"+lbl]
            bt.w   = tr_params["omega"+lbl]
            bt.a   = rho_to_aR(tr_params["rho_star"], bt.per)
            bt.fp  = tr_params["L"+lbl]                                        
            ecc_factor=(1-bt.ecc**2)/(1+bt.ecc*np.sin(np.deg2rad(bt.w)))  
            
            bt.inc = np.rad2deg(np.arccos(b/(bt.a * ecc_factor)))
            bt.limb_dark = "quadratic"
            
            u1 = tr_params["u1"]
            u2 = tr_params["u2"]
            bt.u   = [u1,u2]

            bt.t_secondary = bt.t0 + 0.5*bt.per*(1 + 4/np.pi * bt.ecc * np.cos(np.deg2rad(bt.w))) #eqn 33 (http://arxiv.org/abs/1001.2010)
            m_tra = batman.TransitModel(bt, t,transittype="primary")
            m_ecl = batman.TransitModel(bt, t,transittype="secondary")

            f_tra = m_tra.light_curve(bt)-1
            f_occ = (m_ecl.light_curve(bt)-bt.fp)-1
            model_flux += f_tra+f_occ           #transit and eclipse model

        return np.array(1+model_flux)


    def trend_model(params):
        trend = 1 + params["offset"]       #offset
        trend += params["A0"]*(df["cols0"]-cols0_med)  + params["B0"]*(df["cols0"]-cols0_med)**2 #time trend
        trend += params["A3"]*df["cols3"]  + params["B3"]*df["cols3"]**2 #x
        trend += params["A4"]*df["cols4"]  + params["B4"]*df["cols4"]**2 #y
        trend += params["A6"]*df["cols6"]  + params["B6"]*df["cols6"]**2 #bg
        trend += params["A7"]*df["cols7"]  + params["B7"]*df["cols7"]**2 #conta
        
        if cheops is False:
            trend += params["A5"]*df["cols5"]  + params["B5"]*df["cols5"]**2 
        else: #roll
            sin_col5,  cos_col5  = np.sin(np.deg2rad(df["cols5"])),   np.cos(np.rad2deg(df["cols5"]))
            sin_2col5, cos_2col5 = np.sin(2*np.deg2rad(df["cols5"])), np.cos(2*np.rad2deg(df["cols5"]))
            sin_3col5, cos_3col5 = np.sin(3*np.deg2rad(df["cols5"])), np.cos(3*np.rad2deg(df["cols5"]))

            trend+= params["A5"]*sin_col5 + params["B5"]*cos_col5
            trend+= params["A5_2"]*sin_2col5 + params["B5_2"]*cos_2col5
            trend+= params["A5_3"]*sin_3col5 + params["B5_3"]*cos_3col5
        return np.array(trend)
    

    if return_models:
        tsm = np.linspace(min(df["cols0"]),max(df["cols0"]),len(df["cols0"])*3)
        return trend_model(params),transit_occ_model(tr_params,npl=npl),tsm,transit_occ_model(tr_params,tsm,npl=npl)   
    
    #perform fitting 
    def chisqr(fit_params):
        flux_model = trend_model(fit_params)*transit_occ_model(fit_params,npl=npl)
        res = (df["cols1"] - flux_model)/df["cols2"]
        for p in fit_params:
            u = fit_params[p].user_data  #obtain tuple specifying the normal prior if defined
            if u:  #modify residual to account for how far the value is from mean of prior
                res = np.append(res, (u[0]-fit_params[p].value)/u[1] )
        # print(f"chi-square:{np.sum(res**2)}")
        return res
    
    fit_params = params+tr_params
    out = minimize(chisqr, fit_params, nan_policy='propagate')
    
    #modify output object
    out.bestfit = trend_model(out.params)*transit_occ_model(out.params,npl=npl)
    out.trend   = trend_model(out.params)
    out.transit = transit_occ_model(out.params,npl=npl)
    out.time    = np.array(df["cols0"])
    out.flux    = np.array(df["cols1"])
    out.flux_err= np.array(df["cols2"])
    out.data    = df
    out.rms     = np.std(out.flux - out.bestfit)
    out.ndata   = len(out.time)
    out.residual= out.residual[:out.ndata]
    out.nfree   = out.ndata - out.nvarys
    out.chisqr  = np.sum(out.residual**2)
    out.redchi  = out.chisqr/out.nfree
    out.lnlike  = -0.5*np.sum(out.residual**2 + np.log(2*np.pi*out.flux_err**2))
    out.bic    = out.chisqr + out.nvarys*np.log(out.ndata)

    return out


def _decorr_RV(file, T_0=None, Period=None, K=None, sesinw=0, secosw=0, gamma=None, decorr_bound=(-1000,1000),
                 A0=None, B0=None, A3=None, B3=None, A4=None, B4=None, A5=None, B5=None, npl=1,return_models=False):
    """
    linear decorrelation with different columns of data file. It performs a linear model fit to the 3rd column of the file.
    It uses columns 0,3,4,5 to construct the linear trend model.
    
    Parameters:
    -----------
    file : str;
        path to data file with columns 0 to 8 (col0-col8).
    
    T_0, Period, K, Eccentricity, omega : floats, None;
        RV parameters of the planet. T_0 and P must be in same units as the time axis (cols0) in the data file.
        if float/int, the values are held fixed. if tuple/list of len 2 implies [min,max] while len 3 implies [min,start_val,max].

    offset, Ai, Bi; floats [-1,1] or None;
        coefficients of linear model where offset is the systemic velocity. They have large bounds [-1000,1000].
        Ai, Bi are the linear and quadratic term of the model against column i. A0*col0 + A0*col0**2 for time trend
    
    npl : int; 
        number of planets in the system. default is 1.
        
    return_models : Bool;
        True to return trend model and transit/eclipse model.
         
    Returns:
    -------
    result: object;
        result object from fit with several attributes such as result.bestfit, result.params, result.bic, ...
        if return_models = True, returns (trend_model, transit/eclipse model)
    """
    from CONAN3.RVmodel_v3 import get_RVmod

    DA      = locals().copy()
    _       = DA.pop("file")
    rv_pars = {}

    ff = np.loadtxt(file,usecols=(0,1,2,3,4,5))
    dict_ff = {}
    for i in range(6): dict_ff[f"cols{i}"] = ff[:,i]
    df = pd.DataFrame(dict_ff)  #pandas dataframe
    cols0_med = np.median(df["cols0"])
                          

    #add indices to parameters if npl>1
    for p in ["T_0", "Period", "K", "sesinw", "secosw"]:
        for n in range(npl):
            lbl = f"_{n+1}" if npl>1 else ""   # numbering to add to parameter names of each planet
            rv_pars[p+lbl]= DA[p][n]           # rv pars
    rv_pars["gamma"] = DA["gamma"]   #same for all planets

    #decorr variables    
    decorr_vars = [f"{L}{i}" for i in [0,3,4,5] for L in ["A","B"]] 
    in_pars = {k:v for k,v in DA.items() if k in decorr_vars}  #dictionary of decorr parameters Ai,Bi

    #decorr params to fit or fix
    params = Parameters()
    for key in in_pars.keys():
        val  = in_pars[key] if in_pars[key] != None else 0
        vary = True if in_pars[key] != None else False
        params.add(key, value=val, min=decorr_bound[0], max=decorr_bound[1], vary=vary)
    
    #rvparameters
    rv_params = Parameters()
    for key in rv_pars.keys():
        if isinstance(rv_pars[key], (list,tuple)):
            assert len(rv_pars[key]) in [2,3],f"{key} must be float/int or tuple of length 2/3"
            if len(rv_pars[key])==3:  #uniform prior (min, start, max)
                val = rv_pars[key]
                rv_params.add(key, value=val[1], min=val[0], max=val[2], vary=True)
            if len(rv_pars[key])==2: #normal prior (mean, std)
                rv_params[key] = Parameter(key, value=rv_pars[key][0], vary=True, user_data = rv_pars[key] )
        if isinstance(rv_pars[key], (float,int)):
            rv_params.add(key, value=rv_pars[key], vary=False)
        if rv_pars[key] is None:
            rv_params.add(key, value=0, vary=False)
                
    
    def rv_model(rv_params,t=None,npl=1):
        if t is None: t = df["cols0"].values
        rvmod_ms = np.zeros_like(t)

        for n in range(1,npl+1):
            lbl = f"_{n}" if npl>1 else ""

            per     = [rv_params["Period"+lbl]]
            t0      = [rv_params["T_0"+lbl]]
            K       = [rv_params["K"+lbl]]
            sesinw  = [rv_params["sesinw"+lbl]]
            secosw  = [rv_params["secosw"+lbl]]
            mod,_   = get_RVmod(t, t0, per, K, sesinw, secosw, get_model=True)  
            rvmod_ms += mod
        
        rvmod_kms = rvmod_ms/1000 # to km/s
        return rvmod_kms + rv_params["gamma"]


    def trend_model(params):
        trend  = params["A0"]*(df["cols0"]-cols0_med)  + params["B0"]*(df["cols0"]-cols0_med)**2 #time trend
        trend += params["A3"]*df["cols3"]  + params["B3"]*df["cols3"]**2 #bisector
        trend += params["A4"]*df["cols4"]  + params["B4"]*df["cols4"]**2 #fwhm
        trend += params["A5"]*df["cols5"]  + params["B5"]*df["cols5"]**2 #contrast
        return np.array(trend)
    

    if return_models:
        tsm = np.linspace(min(df["cols0"]),max(df["cols0"]),len(df["cols0"])*3)
        return trend_model(params)+rv_params["gamma"], rv_model(rv_params,npl=npl), tsm,rv_model(rv_params,tsm,npl=npl)   
    
    #perform fitting 
    def chisqr(fit_params):
        rvmod = trend_model(fit_params)+rv_model(fit_params,npl=npl)
        res = (df["cols1"] - rvmod)/df["cols2"]
        for p in fit_params:
            u = fit_params[p].user_data  #obtain tuple specifying the normal prior if defined
            if u:  #modify residual to account for how far the value is from mean of prior
                res = np.append(res, (u[0]-fit_params[p].value)/u[1] )
        # print(f"chi-square:{np.sum(res**2)}")
        return res
    
    fit_params = params+rv_params
    out = minimize(chisqr, fit_params, nan_policy='propagate')
    
    #modify output object
    out.bestfit = trend_model(out.params)+rv_model(out.params,npl=npl)
    out.trend   = trend_model(out.params)+out.params["gamma"]
    out.rvmodel = rv_model(out.params,npl=npl)
    out.time    = np.array(df["cols0"])
    out.rv      = np.array(df["cols1"])
    out.rv_err  = np.array(df["cols2"])
    out.data    = df
    out.rms     = np.std(out.rv - out.bestfit)
    out.ndata   = len(out.time)
    out.residual= out.residual[:out.ndata]    #out.residual = chisqr(out.params)
    out.nfree   = out.ndata - out.nvarys
    out.chisqr  = np.sum(out.residual**2)
    out.redchi  = out.chisqr/out.nfree
    out.lnlike  = -0.5*np.sum(out.residual**2 + np.log(2*np.pi*out.rv_err**2))
    out.bic     = out.chisqr + out.nvarys*np.log(out.ndata)

    return out


def _print_output(self, section: str, file=None):
    """function to print to screen/file the different sections of CONAN setup"""

    lc_possible_sections= ["lc_baseline", "gp", "transit_rv_pars", "depth_variation",
                            "occultations", "limb_darkening", "contamination", "stellar_pars"]
    if self._obj_type == "lc_obj":
        assert section in lc_possible_sections, f"{section} not a valid section of `lc_data`. Section must be one of {lc_possible_sections}."
        max_name_len = max([len(n) for n in self._names]+[len("name")])      #max length of lc filename
        max_filt_len = max([len(n) for n in self._filters]+[len("filt")])  #max length of lc filter name
    if self._obj_type == "rv_obj":
        assert section == "rv_baseline", f"The only valid section for an RV data object is 'rv_baseline' but {section} given."
        max_name_len = max([len(n) for n in self._names]+[len("name")])      #max length of lc filename
    if self._obj_type == "mcmc_obj":
        assert section == "mcmc",  f"The only valid section for an mcmc object is 'mcmc' but {section} given."

    if section == "lc_baseline":
        _print_lc_baseline = f"""#--------------------------------------------- \n# Input lightcurves filters baseline function--------------""" +\
                            f""" \n{"name":{max_name_len}s}  {"filt":{max_filt_len}s}  {"lamda":8s} {"col0":4s}  {"col3":4s}  {"col4":4s}  {"col5":4s}  {"col6":4s}  {"col7":4s}  {"sin":3s}  {"group":5s}  {"id":2s}  {"GP":2s}  {"spl_config     ":20s}"""
        #define print out format
        txtfmt = f"\n{{0:{max_name_len}s}}  {{1:{max_filt_len}s}}"+"  {2:8s} {3:4d}  {4:4d}  {5:4d}  {6:4d}  {7:4d}  {8:4d}  {9:3d}  {10:5d}  {11:2d}  {12:2s}  {13:20s}"        
        for i in range(len(self._names)):
            t = txtfmt.format(self._names[i], self._filters[i], str(self._lamdas[i]), *self._bases[i], self._groups[i], self._useGPphot[i],self._lcspline[i].conf)
            _print_lc_baseline += t
        print(_print_lc_baseline, file=file)   

    if section == "gp":
        DA = self._GP_dict
        _print_gp = f"""# -------- photometry GP input properties: komplex kernel -> several lines -------------- """+\
                     f"""\n{"name":{max_name_len}s} {'para':4s} {"kernel":6s} {"WN":2s} {'scale':7s} {"s_step":6s} {'s_pri':5s} {"s_pri_wid":9s} {'s_up':5s} """+\
                         f"""{'s_lo':5s} {'metric':7s} {"m_step":6s} {'m_pri':6s} {"m_pri_wid":9s} {'m_up':4s} {'m_lo':4s}"""
        #define gp print out format
        if DA["lc_list"] != []:
            txtfmt = f"\n{{0:{max_name_len}s}}"+" {1:4s} {2:6s} {3:2s} {4:7.1e} {5:6.4f} {6:5.1f} {7:9.2e} {8:5.1f} {9:5.1f} {10:6.1e} {11:6.4f} {12:6.2f} {13:9.2e} {14:4.1f} {15:4.1f}"        
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
                                  f"""\n{'name':12s}\tfit\tstart_val\tstepsize\tlow_lim\tup_lim\tprior\t{"value   ":8s}\tsig_lo\tsig_hi"""
        #define print out format
        txtfmt = "\n{0:12s}\t{1:3s}\t{2:8.5f}\t{3:.7f}\t{4:4.2f}\t{5:4.2f}\t{6:5s}\t{7:8.5f}\t{8:6.1e}\t{9:6.1e} "
        #print line for stellar density
        p = "rho_star"
        _print_transit_rv_pars +=  txtfmt.format( p,
                                                DA[f'pl{1}'][p].to_fit, DA[f'pl{1}'][p].start_value,
                                                DA[f'pl{1}'][p].step_size, DA[f'pl{1}'][p].bounds_lo, 
                                                DA[f'pl{1}'][p].bounds_hi, DA[f'pl{1}'][p].prior, DA[f'pl{1}'][p].prior_mean,
                                                DA[f'pl{1}'][p].prior_width_lo, DA[f'pl{1}'][p].prior_width_hi)
        _print_transit_rv_pars +=  "---------------------"
        #then cycle through parameters for each planet       
        for n in range(1,self._nplanet+1):        
            for i,p in enumerate(self._TR_RV_parnames):
                if p != "rho_star":
                    t = txtfmt.format(  p+(f"_{n}" if self._nplanet>1 else ""),
                                        DA[f'pl{n}'][p].to_fit, DA[f'pl{n}'][p].start_value,
                                        DA[f'pl{n}'][p].step_size, DA[f'pl{n}'][p].bounds_lo, 
                                        DA[f'pl{n}'][p].bounds_hi, DA[f'pl{n}'][p].prior, DA[f'pl{n}'][p].prior_mean,
                                        DA[f'pl{n}'][p].prior_width_lo, DA[f'pl{n}'][p].prior_width_hi)
                    _print_transit_rv_pars += t
            if n!=self._nplanet: _print_transit_rv_pars += "---------------------"
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
            f"""\nStellar_para_input_method:_R+rho_(Rrho),_M+rho_(Mrho): {DA['par_input']}"""
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
                                    f"""\n{'name':{max_name_len}s} {'col0':4s} {'col3':4s} {'col4':4s} {"col5":4s} {'sin':3s} {"spl_config     ":20s} | {'gamma_kms':9s} {'stepsize':8s} {'prior':5s} {'    value':9s} {'sig_lo':6s} {'sig_hi':6s}"""
        
        if self._names != []:
            #define gp print out format
            txtfmt = f"\n{{0:{max_name_len}s}}"+" {1:4d} {2:4d} {3:4d} {4:4d} {5:3d} {6:20s} | {7:9.4f} {8:8.4f} {9:5s} {10:9.4f} {11:6.4f} {12:6.4f}"         
            for i in range(self._nRV):
                t = txtfmt.format(self._names[i],*self._RVbases[i],self._rvspline[i].conf,self._gammas[i], 
                                self._gamsteps[i], self._prior[i], self._gampri[i],self._siglo[i], self._sighi[i])
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
            
        self.to_fit         = par_list[0] if (par_list[0] in ["n","y"]) else _raise(ValueError, "to_fit (par_list[0]) must be 'n' or 'y'")
        self.start_value    = par_list[1]
        self.step_size      = par_list[2]
        self.prior          = par_list[3] if (par_list[3] in ["n","p"]) else _raise(ValueError, "prior (par_list[3]) must be 'n' or 'p'")
        self.prior_mean     = par_list[4]
        self.prior_width_lo = par_list[5]
        self.prior_width_hi = par_list[6]
        self.bounds_lo      = par_list[7]
        self.bounds_hi      = par_list[8]
        
    def _set(self, par_list):
        return self.__init__(par_list)
    
    @classmethod
    def re_init(cls, par_list):      #re-initialize class
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

filter_shortcuts = dict(kepler='Kepler/Kepler.k',
                    tess='TESS/TESS.Red',
                    cheops='CHEOPS/CHEOPS.band',
                    wfc3_g141='HST/WFC3_IR.G141',
                    wfc3_g102='HST/WFC3_IR.G102',
                    sp36='Spitzer/IRAC.I1',
                    sp45='Spitzer/IRAC.I2',
                    ug='Geneva/Geneva.U',
                    b1='Geneva/Geneva.B1',
                    b2='Geneva/Geneva.B2',
                    bg='Geneva/Geneva.B',
                    gg='Geneva/Geneva.G',
                    v1='Geneva/Geneva.V2',
                    vg='Geneva/Geneva.V',
                    sdss_g='SLOAN/SDSS.g',
                    sdss_r='SLOAN/SDSS.r',
                    sdss_i='SLOAN/SDSS.i',
                    sdss_z='SLOAN/SDSS.z')

#========================================================================
class load_lightcurves:
    """
        lightcurve object to hold lightcurves for analysis
        
        Parameters:
        -----------
        
        data_filepath : str;
            Filepath where lightcurves files are located. Default is None which implies the data is in the current working directory.

            
        file_list : list;
            List of filenames for the lightcurves. Files must have 9 columns: time,flux,err,xc,xc,xc,xc,xc,xc. 
            where xc are columns that can be used in decorrelating the flux. Arrays of zeroes are put in xc if file contains less than 9 columns.
            
        filter : list, str, None;
            filter for each lightcurve in file_list. if a str is given, it is used for all lightcurves,
            if None, the default of "V" is used for all.
            
        lamdas : list, int, float, None;
            central wavelength for each lightcurve in file_list. if a int or float is given, it is used for all lightcurves,
            if None, the default of 6000.0 is used for all.
        
        Returns:
        --------
        lc_data : light curve object

        Example:
        --------
        >>> lc_data = load_lightcurves(file_list=["lc1.dat","lc2.dat"], filters=["V","I"], lamdas=[6000,8000])
        
    """
    def __init__(self, file_list, data_filepath=None, filters=None, lamdas=None, nplanet=1,
                 verbose=True, show_guide=False):
        self._obj_type = "lc_obj"
        self._nplanet = nplanet
        self._fpath = os.getcwd()+'/' if data_filepath is None else data_filepath
        self._names = [file_list] if isinstance(file_list, str) else [] if file_list is None else file_list
        for lc in self._names: assert os.path.exists(self._fpath+lc), f"file {lc} does not exist in the path {self._fpath}."
        
        assert filters is None or isinstance(filters, (list, str)), \
            f"filters is of type {type(filters)}, it should be a list, a string or None."
        assert lamdas  is None or isinstance(lamdas, (list, int, float)), \
            f"lamdas is of type {type(lamdas)}, it should be a list, int or float."
        
        if isinstance(filters, str): filters = [filters]
        if isinstance(lamdas, (int, float)): lamdas = [float(lamdas)]

        self._nphot = len(self._names)
        if filters is not None and len(filters) == 1: filters = filters*self._nphot
        if lamdas is  not None and len(lamdas)  == 1: lamdas  = lamdas *self._nphot

        self._filters = ["V"]*self._nphot if filters is None else [f for f in filters]
        self._lamdas  = [6000.0]*self._nphot if lamdas is None else [l for l in lamdas]
        self._filter_shortcuts = filter_shortcuts
        
        assert self._nphot == len(self._filters) == len(self._lamdas), \
            f"filters and lamdas must be a list with same length as file_list (={self._nphot})"
        self._filnames   = np.array(list(sorted(set(self._filters),key=self._filters.index)))

        #modify input files to have 9 columns as CONAN expects
        for f in self._names:
            fdata = np.loadtxt(self._fpath+f)
            nrow,ncol = fdata.shape
            if ncol < 9:
                print(f"writing ones to the missing columns of file: {f}")
                new_cols = np.ones((nrow,9-ncol))
                ndata = np.hstack((fdata,new_cols))
                np.savetxt(self._fpath+f,ndata,fmt='%.8f')

        if verbose: 
            print(f"Filters: {self._filters}")
            print(f"Order of unique filters: {list(self._filnames)}")

        self._show_guide = show_guide
        self._clipped_data = False
        self.lc_baseline(re_init = hasattr(self,"_bases"), verbose=False)

        if self._show_guide: print("\nNext: use method `lc_baseline` to define baseline model for each lc or method " + \
            "`get_decorr` to obtain best best baseline model parameters according bayes factor comparison")

    def get_decorr(self, T_0=None, Period=None, rho_star=None, L=0, Impact_para=0, RpRs=1e-5,Eccentricity=0, omega=90, K=0,u1=0, u2=0, 
                   mask=False, delta_BIC=-3, decorr_bound =(-1,1),cheops=False, exclude=[],enforce=[],verbose=True, 
                   show_steps=False, plot_model=True, use_result=True):
        """
            Function to obtain best decorrelation parameters for each light-curve file using the forward selection method.
            It compares a model with only an offset to a polynomial model constructed with the other columns of the data.
            It uses columns 0,3,4,5,6,7 to construct the polynomial trend model. The temporary decorr parameters are labelled Ai,Bi for 1st & 2nd order in column i.
            If cheops is True, A5, B5 are the sin and cos of the roll-angle while A5_i, B5_i are the corresponding harmonics with i=2,3. If these are significant, a gp or spline in roll-angle will be needed.
            Decorrelation parameters that reduces the BIC by 3(i.e delta_BIC = -3) are iteratively selected.
            The result can then be used to populate the `lc_baseline` method, if use_result is set to True.

            Parameters:
            -----------
            T_0, Period, rho_star, L, Impact_para, RpRs, Eccentricity, omega: floats,tuple, None;
                transit/eclipse parameters of the planet. T_0 and Period must be in same units as the time axis (cols0) in the data file.
                if float/int, the values are held fixed. if tuple/list of len 2 implies gaussian prior as (mean,std) while len 3 implies [min,start_val,max].
                
            u1,u2 : float,tuple, list  (optional);
                standard quadratic limb darkening parameters. if float, the values are held fixed. if tuple/list of len 2 implies gaussian prior as (mean,std) while len 3 implies [min,start_val,max].
                give list of values for assign value to each unique filter in the data, or one value to be used for all filtets. Default is 0 for all filters.
    
            delta_BIC : float (negative);
                BIC improvement a parameter needs to provide in order to be considered relevant for decorrelation. + \
                    Default is conservative and set to -3 i.e, parameters needs to lower the BIC by 3 to be included as decorrelation parameter.

            mask : bool ;
                If True, transits and eclipses are masked using T_0, P and dur which must be float/int.
        
            decorr_bound: tuple of size 2;
                bounds when fitting decorrelation parameters. Default is (-1,1)

            cheops : Bool or list of Bool;
                Flag to specify if data is from CHEOPS with col5 being the roll-angle. if True, a linear + \
                fourier model (sin and cos) up to 3rd harmonic in roll-angle is used for col5.
                If Bool is given, the same is used for all input lc, else a list specifying bool for each lc is required.
                Default is False.

            exclude : list of int;
                list of column numbers (e.g. [3,4]) to exclude from decorrelation. Default is [].

            enforce : list of int;
                list of decorr params (e.g. ['B3', 'A5']) to enforce in decorrelation. Default is [].

            verbose : Bool, optional;
                Whether to show the table of baseline model obtained. Defaults to True.

            show_steps : Bool, optional;
                Whether to show the steps of the forward selection of decorr parameters. Default is False
            
            plot_model : Bool, optional;
                Whether to overplot suggested trend model on the data. Defaults to True.

            use_result : Bool, optional;
                whether to use result/input to setup the baseline model and transit/eclipse models. Default is True.
        
            Returns
            -------
            decorr_result: list of result object
                list containing result object for each lc.
        """
        assert isinstance(exclude, list), f"get_decorr: exclude must be a list of column numbers to exclude from decorrelation but {exclude} given."
        for c in exclude: assert isinstance(c, int), f"get_decorr: column number to exclude from decorrelation must be an integer but {c} given in exclude."

        nfilt = len(self._filnames)
        if isinstance(u1, np.ndarray): u1 = list(u1)
        if isinstance(u1, list): assert len(u1) == nfilt, f"get_decorr(): u1 must be a list of same length as number of unique filters {nfilt} but {len(u1)} given." 
        else: u1=[u1]*nfilt
        if isinstance(u2, np.ndarray): u2 = list(u2)
        if isinstance(u2, list): assert len(u2) == nfilt, f"get_decorr(): u2 must be a list of same length as number of unique filters {nfilt} but {len(u2)} given." 
        else: u2=[u2]*nfilt

        blpars = {"dcol0":[], "dcol3":[],"dcol4":[], "dcol5":[], "dcol6":[], "dcol7":[],"gp":[]}  #inputs to lc_baseline method
        self._decorr_result = []   #list of decorr result for each lc.
        
        input_pars = dict(T_0=T_0, Period=Period, rho_star=rho_star, Impact_para=Impact_para, \
                        RpRs=RpRs, Eccentricity=Eccentricity, omega=omega, K=K)

        self._tra_occ_pars = dict(T_0=T_0, Period=Period, rho_star=rho_star, L=L, Impact_para=Impact_para, \
                              RpRs=RpRs, Eccentricity=Eccentricity, omega=omega)#, u1=u1,u2=u2)  #transit/occultation parameters
        
        for p in self._tra_occ_pars:
            if p != "rho_star":
                if isinstance(self._tra_occ_pars[p], (int,float,tuple)): self._tra_occ_pars[p] = [self._tra_occ_pars[p]]*self._nplanet
                if isinstance(self._tra_occ_pars[p], (list)): assert len(self._tra_occ_pars[p]) == self._nplanet, \
                    f"get_decorr(): {p} must be a list of same length as number of planets {self._nplanet} but {len(self._tra_occ_pars[p])} given."
            else:
                assert isinstance(self._tra_occ_pars[p],(int,float,tuple)),f"get_decorr(): {p} must be one of int/float/tuple but {self._tra_occ_pars[p]} given "

        ld_u1, ld_u2 = {},{}
        for i,fil in enumerate(self._filnames):
            ld_u1[fil] = u1[i]
            ld_u2[fil] = u2[i]
            #check that ld values give realistic profiles for each filter, folling kipping2013 triangular test
            u1_ = u1[i] if isinstance(u1[i], (int,float)) else u1[i][1] if len(u1[i])==3 else u1[i][0]
            u2_ = u2[i] if isinstance(u2[i], (int,float)) else u2[i][1] if len(u2[i])==3 else u2[i][0]
            q1 = (u1_ + u2_)**2
            q2 = u1_/(2*(u1_+u2_))
            if (0<q1<1) and (0<=q2<1): pass
            else: print(f"get_decorr(): Warning!!! converting u1,u2={u1_:.3f},{u2_:.3f} to Kipping parameterization q1,q2={q1:.3f},{q2:.3f} for filter {fil}. The conditions 0<q1<1, 0<=q2<1 are not met.")

        
        #check cheops input
        assert delta_BIC<0,f'get_decorr(): delta_BIC must be negative for parameters to provide improved fit but {delta_BIC} given.'
        if isinstance(cheops, bool): cheops_flag = [cheops]*self._nphot
        elif isinstance(cheops, list):
            assert len(cheops) == self._nphot,f"list given for cheops must have same +\
                length as number of input lcs but {len(cheops)} given."
            for flag in cheops:
                assert isinstance(flag, bool), f"get_decorr(): all elements in cheops list must be bool: +\
                     True or False, but {flag} given"
            cheops_flag = cheops
        else: _raise(TypeError, f"get_decorr(): `cheops` must be bool or list of bool with same length as +\
            number of input files but type{cheops} given.")


        self._tmodel = []  #list to hold determined trendmodel for each lc
        decorr_cols = [0,3,4,5,6,7]
        for c in exclude: assert c in decorr_cols, f"get_decorr(): column number to exclude from decorrelation must be in {decorr_cols} but {c} given in exclude." 
        _ = [decorr_cols.remove(c) for c in exclude]  #remove excluded columns from decorr_cols

        for j,file in enumerate(self._names):
            if verbose: print(_text_format.BOLD + f"\ngetting decorrelation parameters for lc: {file} (cheops={cheops_flag[j]})" + _text_format.END)
            all_par = [f"{L}{i}" for i in decorr_cols for L in ["A","B"]] 
            if cheops_flag[j]: all_par += ["A5_2","B5_2","A5_3","B5_3"]

            out = _decorr(self._fpath+file, **self._tra_occ_pars, u1=ld_u1[self._filters[j]],u2=ld_u2[self._filters[j]], mask=mask,
                            offset=0,cheops=cheops_flag[j], decorr_bound=decorr_bound, npl=self._nplanet)    #no trend, only offset
            best_bic = out.bic
            best_pars = {"offset":0}                      #parameter salways included
            for cp in enforce: best_pars[cp]=0            #add enforced parameters
            _ = [all_par.remove(cp) for cp in enforce if cp in all_par]    #remove enforced parameters from all_par

            if show_steps: print(f"{'Param':7s} : {'BIC':6s} N_pars \n---------------------------")

            del_BIC = -np.inf # bic_ratio = 0 # bf = np.inf
            while del_BIC < delta_BIC:#while  bf > 1:
                if show_steps: print(f"{'Best':7s} : {best_bic:.2f} {len(best_pars.keys())} {list(best_pars.keys())}\n---------------------")
                pars_bic = {}
                for p in all_par:
                    dtmp = best_pars.copy()   #always include offset
                    dtmp[p] = 0
                    out = _decorr(self._fpath+file, **self._tra_occ_pars, u1=ld_u1[self._filters[j]],u2=ld_u2[self._filters[j]],**dtmp,
                                    cheops=cheops_flag[j], decorr_bound=decorr_bound, npl=self._nplanet)
                    if show_steps: print(f"{p:7s} : {out.bic:.2f} {out.nvarys}")
                    pars_bic[p] = out.bic

                par_in = min(pars_bic,key=pars_bic.get)   #parameter that gives lowest BIC
                par_in_bic = pars_bic[par_in]
                del_BIC = par_in_bic - best_bic
                bf = np.exp(-0.5*(del_BIC))
                if show_steps: print(f"+{par_in} -> BF:{bf:.2f}, del_BIC:{del_BIC:.2f}")

                if del_BIC < delta_BIC:# if bf>1:
                    if show_steps: print(f"adding {par_in} lowers BIC to {par_in_bic:.2f}\n" )
                    best_pars[par_in]=0
                    best_bic = par_in_bic
                    all_par.remove(par_in)            
                      
            result = _decorr(self._fpath+file, **self._tra_occ_pars, u1=ld_u1[self._filters[j]],u2=ld_u2[self._filters[j]],
                                **best_pars, cheops=cheops_flag[j], decorr_bound=decorr_bound, npl=self._nplanet)
            self._decorr_result.append(result)
            print(f"BEST BIC:{result.bic:.2f}, pars:{list(best_pars.keys())}")
            
            #calculate determined trend and tra/occ model over all data(no mask)
            pps = result.params.valuesdict()
            #convert result transit parameters to back to a list
            for p in ['RpRs', 'Impact_para', 'T_0', 'Period', 'Eccentricity', 'omega','L']:
                if self._nplanet==1:
                    pps[p] = [pps[p]]  
                else:      
                    pps[p] = [pps[p+f"_{n}"] for n in range(1,self._nplanet+1)]
                    _ = [pps.pop(f"{p}_{n}") for n in range(1,self._nplanet+1)]
    
            self._tmodel.append(_decorr(self._fpath+file,**pps, cheops=cheops_flag[j], npl=self._nplanet, return_models=True))

            #set-up lc_baseline model from obtained configuration
            blpars["dcol0"].append( 2 if pps["B0"]!=0 else 1 if  pps["A0"]!=0 else 0)
            blpars["dcol3"].append( 2 if pps["B3"]!=0 else 1 if  pps["A3"]!=0 else 0)
            blpars["dcol4"].append( 2 if pps["B4"]!=0 else 1 if  pps["A4"]!=0 else 0)
            blpars["dcol6"].append( 2 if pps["B6"]!=0 else 1 if  pps["A6"]!=0 else 0)
            blpars["dcol7"].append( 2 if pps["B7"]!=0 else 1 if  pps["A7"]!=0 else 0)
            blpars["gp"].append("n")
            if not cheops_flag[j]:
                blpars["dcol5"].append( 2 if pps["B5"]!=0 else 1 if  pps["A5"]!=0 else 0)
            else:
                blpars["dcol5"].append(0)
                # blpars["gp"].append("y")  #for gp in roll-angle (mostly needed)

        if plot_model:
            _plot_data(self,plot_cols=(0,1,2),col_labels=("time","flux"),model_overplot=self._tmodel)
        
        if np.any(cheops): 
            print(_text_format.BOLD + f"\nSetting-up spline for roll-angle decorrelation."+ _text_format.END +\
                    " Use `.add_spline(None)` method to remove")
            self.add_spline()

        #prefill other light curve setup from the results here or inputs given here.
        if use_result:
            if verbose: print(_text_format.BOLD + "Setting-up baseline model from result" +_text_format.END)
            self.lc_baseline(**blpars, verbose=verbose)
            # print(_text_format.RED + f"\n Note: GP flag for each lc has been set to {self._useGPphot}. "+\
            #         "Use `._useGPphot` method to modify this list with 'y' or 'n' for each loaded lc\n" + _text_format.END)
        
            if verbose: print(_text_format.BOLD + "\nSetting-up occultation pars from input values" +_text_format.END)
            if isinstance(self._tra_occ_pars["L"], tuple):   #TODO: not printed when occultation fixed
                self.setup_occultation("all",start_depth=self._tra_occ_pars["L"], verbose=verbose)
            else:
                self.setup_occultation(verbose=False)
            
            if all([p in self._tra_occ_pars for p in["Period","rho_star","Impact_para","RpRs","Eccentricity", "omega", "T_0"]]):
                if verbose: print(_text_format.BOLD + "\nSetting-up transit pars from input values" +_text_format.END)
                self.setup_transit_rv(**input_pars, verbose=verbose)
            
            # if all([p in self._tra_occ_pars for p in ["u1","u2"]]):
            if verbose: print(_text_format.BOLD + "\nSetting-up Limb darkening pars from input values" +_text_format.END)
            self.limb_darkening(c1=u1, c2=u2, verbose=verbose)


        return self._decorr_result
    
    
    def clip_outliers(self, lc_list="all", clip=5, width=15, return_clipped_indices = False, create_new_file = True, verbose=True):

        """
        First divide the data by its median to normalise it.
        Then remove outliers using a running median method. Points > clip*M.A.D are removed
        where M.A.D is the mean absolute deviation from the median in each window

        Parameters:
        ------------
        lc_list: list of string, None, 'all';
            list of lightcurve filenames on which perform outlier clipping. Default is 'all' which clips all lightcurves in the object.
        
        clip: float;
            cut off value above the median. Default is 5

        width: int;
            Number of points in window to use when computing the running median. Must be odd. Default is 15
        
        verbose: bool;
            Prints number of points that have been cut. Default is True
        
        return_clipped_indices: bool;
            Whether to return an array that indicates which points have been clipped. Default is False
            
        create_new_file: bool;
            Whether to replace the original file or create a new one without the clipped points. Default behaviour is True. \
                If set to True, the original file is kept in its folder, while a new folder and file with the clipped data is created. 'clp' is appended to the new folder and file names.

        Returns:
        --------
        Nothing by default.
        if return_clipped_indices is True, returns a list containing array of indices of the points removed for each given lc.

        """
        if self._clipped_data:
            print("Data has already been clipped. run `load_lightcurves()` again to reset.")
            return None

        clipped_indices = []
        if lc_list == None: 
            print("lc_list is None: No lightcurve to clip outliers.")
            return None
        if isinstance(lc_list, str) and (lc_list != 'all'): lc_list = [lc_list]
        if lc_list == "all": lc_list = self._names
        
        if width%2 == 0:    #if width is even, make it odd
            width += 1

        if create_new_file:     # create new path with _clp/ appended if it does not exist
            new_fpath = self._fpath[:-1]+"_clp/"
            if not os.path.exists(new_fpath): os.mkdir(new_fpath)
            new_fnames = []
        
        for j,file in enumerate(self._names):
            if file in lc_list:
                data = np.loadtxt(self._fpath+file)

                _,_,clpd_mask = outlier_clipping(x=data[:,0],y=data[:,1],clip=clip,width=width,verbose=verbose,
                                 return_clipped_indices=True)   #returns mask of the clipped points
                ok = ~clpd_mask     #invert mask to get indices of points that are not clipped
                
                med_baseline = np.median(np.sort(data[:,1])[-int(0.4*len(data[:,1])):])  #estimated median of the baseline
                data[:,1:3]  = data[:,1:3]/med_baseline  #normalise data by median of baseline
            
                clipped_data = data[ok]
            
                if verbose:
                    print(f'\n{file}: Rejected {sum(~ok)} points more than {clip:0.1f} x MAD from the median')

                clipped_filename = os.path.splitext(file)
                clipped_filename = clipped_filename[0] + "_clp" + clipped_filename[1]
            else:
                data = np.loadtxt(self._fpath+file)
                clipped_data = data
                clipped_filename = file
                ok = data[:,0] == data[:,0]

            new_fnames.append(clipped_filename)

            if create_new_file:
                np.savetxt(new_fpath+clipped_filename,clipped_data,fmt='%.8f')
                print(f'\n{"" if file in lc_list else "un"}Clipped data saved in file: {new_fpath+clipped_filename}')
            else:
                np.savetxt(self._fpath+file,clipped_data,fmt='%.8f')
                print(f'\nrOriginal data file replaced by {"" if file in lc_list else "un"}clipped data file: {file}')
                
            
            fig = plt.figure(figsize=(15,3))
            plt.title(f"{file} --> {clipped_filename}")
            plt.plot(data[:,0][ok], data[:,1][ok], '.b')
            plt.plot(data[:,0][~ok], data[:,1][~ok], '.r')
            plt.show()
            
            clipped_indices.append(~ok)
            
        if create_new_file: 
            self._fpath = new_fpath
            self._names = new_fnames

        self._clipped_data = True

        if return_clipped_indices:
            return clipped_indices
           
    
    def split_transits(self, filename=None, P=None, t_ref=None, baseline_amount=0.3, input_t0s=None, show_plot=True, save_separate=True, same_filter=True):
    
        """
        Function to split the transits in the data into individual transits and save them in separate files or to remove a certain amount of data points around the transits while keeping them in the original file.
        Recommended to set show_plot=True to visually ensure that transits are well separated.

        Parameters:
        -----------

        filename: string;
                name of the lightcurve file to clip. Default is None

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
            
        save_separate: bool;
            set True to separately save each transit and its baseline_amount around it in a new file. Default is True.

        same_filter: bool; 
            set True to save the split transits in the same filter as the original file. Default is True.
        """

        if filename==None: 
            print("No lightcurve filename given.")
            return None
        assert filename in self._names, f"split_transits(): filename {filename} not in loaded lightcurves."
        
        data = np.loadtxt(self._fpath+filename)
        
        t = data.transpose()[0]
        flux = data.transpose()[1]
        
        if baseline_amount < 0.05 :
            baseline_amount = 0.05
            print("Baseline amount defaulted to minimum 0.05")
        elif baseline_amount > 0.5 :
            baseline_amount = 0.5
            print("Baseline amount defaulted to maximum 0.5")
            
        
        if input_t0s is not None:
            t0s = list(input_t0s)

        else:
            tref = t_ref
            if t_ref < t.min() or t.max() < t_ref:        #if reference time t0 is not within this timeseries
                #find transit time that falls around middle of the data
                ntrans = int((np.median(t) - tref)/P)   
                tref = t_ref + ntrans*P

            nt = int( (tref-t.min())/P )                            #how many transits behind tref is the first transit
            tr_first = tref - nt*P                                    #time of first transit in data
            tr_last = tr_first + int((t.max() - tr_first)/P)*P        #time of last transit in data

            n_tot_tr = int((tr_last - tr_first)/P)                  #total nmumber of transits in data_range
            t0s = [tr_first + P*n for n in range(n_tot_tr+1) ]        #expected tmid of transits in data (if no TTV)
            #remove tmid without sufficient transit data around it
            t0s = list(filter(lambda t0: ( t[ (t<t0+0.1*P) & (t>t0-0.1*P)] ).size>0, t0s))


        #split data into individual transits. taking points around each tmid    
        tr_times= []
        fluxes= []
        indz = []
        for i in range(len(t0s)):
            higher_than = t>(t0s[i]-baseline_amount*P)
            lower_than = t<(t0s[i]+baseline_amount*P)
            if i==0: 
                tr_times.append(t[higher_than & lower_than])
                indz.append( np.argwhere(higher_than & lower_than).reshape(-1) )
            elif i == len(t0s)-1: 
                tr_times.append(t[higher_than & lower_than])
                indz.append( np.argwhere(higher_than & lower_than).reshape(-1) )
            else: 
                tr_times.append(t[higher_than & lower_than])
                indz.append( np.argwhere(higher_than & lower_than).reshape(-1))
            fluxes.append(flux[indz[i]])
            
        tr_edges = [(tr_t[0], tr_t[-1]) for tr_t in tr_times]    #take last time in each timebin as breakpts
        
        tr_times = np.concatenate(tr_times)
        fluxes = np.concatenate(fluxes)
            
        if show_plot:
            assert fluxes is not None, f"plotting requires input flux"
            plt.figure(figsize=(15,3))
            plt.plot(tr_times,fluxes,".")
            for edg in tr_edges:
                plt.axvline(edg[0], ls="dashed", c="k", alpha=0.3)
                plt.axvline(edg[1], ls="dashed", c="k", alpha=0.3)
            plt.plot(t0s, (0.997*np.min(flux))*np.ones_like(t0s),"k^")
            plt.xlabel("Time (days)", fontsize=14)
            plt.title("Using t_ref: dashed vertical lines = transit splitting times;  triangles = identified transits");

        if save_separate:
            #replace filter names
            _flt = self._filters[self._names.index(filename)]
            _lbd = self._lamdas[self._names.index(filename)]
            self._filters.remove(_flt)
            self._lamdas.remove(_lbd)
            self._names.remove(filename)

            for i in range(len(t0s)):
                
                tr_data = data[indz[i]]
                
                tr_filename = os.path.splitext(filename)
                tr_filename = tr_filename[0] + "_tr" + str(i) + tr_filename[1]
            
                np.savetxt(self._fpath+tr_filename,tr_data,fmt='%.8f')
                print("Saved " + self._fpath + tr_filename)

                self._names.append(tr_filename)
                if same_filter: self._filters.append(_flt)
                else: self._filters.append(_flt+str(i))
                # self._filnames   = np.array(list(sorted(set(self._filters),key=self._filters.index)))

                self._lamdas.append(_lbd)
        # self.lc_baseline(re_init=True,verbose=False)
        self.__init__(self._names, self._fpath, self._filters, self._lamdas, self._nplanet)

            
    def lc_baseline(self, dcol0=None, dcol3=None, dcol4=None,  dcol5=None, dcol6=None, 
                 dcol7=None, dsin=None, grp=None, grp_id=None, gp="n", re_init=False,verbose=True):
        """
            Define baseline model parameters to fit for each light curve using the columns of the input data. dcol0 refers to decorrelation parameters for column 0, dcol3 for column 3 and so on.
            Each baseline decorrelation parameter (dcolx) should be a list of integers specifying the polynomial order for column x for each light curve.
            e.g. Given 3 input light curves, if one wishes to fit a 2nd order trend in column 0 to the first and third lightcurves,
            then dcol0 = [2, 0, 2].
            The decorrelation parameters depend on the columns (col) of the input light curve. Any desired array can be put in these columns to decorrelate against them. Note that col0 is usually the time array.
            The columns are:


            Parameters:
            -----------
            dcol0, dcol3,dcol4,dcol5,dcol6,dcol7 : list of int;
                polynomial order to fit to each column. Default is 0 for all columns.

            grp_id : list (same length as file_list);
                group the different input lightcurves by id so that different transit depths can be fitted for each group.

            gp : list (same length as file_list); 
                list containing 'y', 'n', or 'ce' to specify if a gp will be fitted to a light curve. +\
                    'y' indicates that the george gp package will be used while 'ce' uses the celerite package.
            
            re_init : bool;
                if True, re-initialize all other methods to empty. Default is False.

        """
        DA = locals().copy()     #get a dictionary of the input arguments for easy manipulation
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("re_init")            #remove self from dictionary
        _ = DA.pop("verbose")


        for par in DA.keys():
            if isinstance(DA[par], (int,str)): DA[par] = [DA[par]]*self._nphot      #use same for all lcs
            elif DA[par] is None: DA[par] = ["n"]*self._nphot if par=="gp" else [0]*self._nphot   #no decorr or gp for all lcs
            elif isinstance(DA[par], (list,np.ndarray)):
                if par=="gp": assert len(DA[par]) == self._nphot, f"lc_baseline: parameter `{par}` must be a list of length {self._nphot} or str (if same is to be used for all LCs) or None."
                else: assert len(DA[par]) == self._nphot, f"lc_baseline: parameter `{par}` must be a list of length {self._nphot} or int (if same degree is to be used for all LCs) or None (if not used in decorrelation)."

            for p in DA[par]:
                if par=="gp": assert p in ["y","n","ce"], f"lc_baseline: gp must be a list of 'y', 'n', or 'ce' for each lc but {p} given."
                else: assert isinstance(p, int) and p<3, f"lc_baseline: decorrelation parameters must be a list of integers (max integer value = 2) but {p} given."

        DA["grp_id"] = list(np.arange(1,self._nphot+1)) if grp_id is None else grp_id

        self._bases = [ [DA["dcol0"][i], DA["dcol3"][i], DA["dcol4"][i], DA["dcol5"][i],
                        DA["dcol6"][i], DA["dcol7"][i], DA["dsin"][i], 
                        DA["grp"][i]] for i in range(self._nphot) ]

        self._groups    = DA["grp_id"]
        self._grbases   = DA["grp"]
        self._useGPphot = DA["gp"]
        self._gp_lcs = np.array(self._names)[np.array(self._useGPphot) != "n"]     #lcs with gp == "y" or "ce"

        if verbose: _print_output(self,"lc_baseline")

        if np.all(np.array(self._useGPphot) == "n") or self._useGPphot==[]:        #if gp is "n" for all input lightcurves, run add_GP with None
            self.add_GP(None, verbose=verbose)
            if self._show_guide: print("\nNo GPs required.\nNext: use method `setup_transit_rv` to configure transit an rv model parameters.")
        else: 
            if self._show_guide: print("\nNext: use method `add_GP` to include GPs for the specified lcs. Get names of lcs with GPs using `._gp_lcs` attribute of the lightcurve object.")

        #initialize other methods to empty incase they are not called/have not been called
        if not hasattr(self,"_lcspline") or re_init:      self.add_spline(None, verbose=False)
        if not hasattr(self,"_config_par") or re_init:    self.setup_transit_rv(verbose=False)
        if not hasattr(self,"_ddfs") or re_init:          self.transit_depth_variation(verbose=False)
        if not hasattr(self,"_occ_dict") or re_init:      self.setup_occultation(verbose=False)
        if not hasattr(self,"_contfact_dict") or re_init: self.contamination_factors(verbose=False)
        if not hasattr(self,"_ld_dict") or re_init:       self.limb_darkening(verbose=False)
        if not hasattr(self,"_stellar_dict") or re_init:  self.stellar_parameters(verbose=False)

    def add_spline(self, lc_list= None, par = None, degree=3, knots_every=None, periodicity=0,verbose=True):
        """
            add spline to fit correlation along 1 or 2 columns of the data. This splits the data at the defined knots interval and fits a spline to each section. 
            scipy's LSQUnivariateSpline() and LSQBivariateSpline() functions are used for 1D spline and 2D splines respectively.
            All arguments can be given as a list to specify config for each lc file in lc_list.

            Parameters
            ----------
            lc_list : list, str, optional
                list of lc files to fit a spline to. set to "all" to use spline for all lc files. Default is None for no splines.

            par : str,tuple,list, optional
                column of input data to which to fit the spline. must be one/two of ["col0","col3","col4","col5","col6","col7"]. Default is None.
                Give list of columns if different for each lc file. e.g. ["col0","col3"] for spline in col0 for lc1.dat and col3 for lc2.dat. 
                For 2D spline for an lc file, use tuple of length 2. e.g. ("col0","col3") for simultaneous spline fit to col0 and col3.

            degree : int, tuple, list optional
                Degree of the smoothing spline. Must be 1 <= degree <= 5. Default is 3 for a cubic spline.
            
            knots_every : float, tuple, list
                distance between knots of the spline, by default 15 degrees for cheops data roll-angle 

            periodicity : float, tuple, list optional
                periodicity of the spline in that axis  of the data. e.g for cheops roll angle in degrees, the periodicity should be set to 360 degrees. 
                Default is zero for no periodicity.
            
            verbose : bool, optional
                print output. Default is True.

            Examples
            --------
            To use different spline configuration for 2 lc files: 2D spline for the first file and 1D for the second.
            >>> lc_data.add_spline(lc_list=["lc1.dat","lc2.dat"], par=[("col3","col4"),"col4"], degree=[(3,3),2], knots_every=[(5,3),2], periodicity=0)
            
            For same spline configuration for all loaded lc files
            >>> lc_data.add_spline(lc_list="all", par="col3", degree=3, knots_every=5, periodicity=0)
        """  

        #default spline config -- None
        self._lcspline = [None]*self._nphot                   #list to hold spline configuration for each lc
        for i in range(self._nphot):
            self._lcspline[i]        = SimpleNamespace()    #create empty namespace for each lc
            self._lcspline[i].name   = self._names[i]
            self._lcspline[i].dim    = 0
            self._lcspline[i].par    = None
            self._lcspline[i].use    = False
            self._lcspline[i].deg    = None
            self._lcspline[i].knots  = None
            self._lcspline[i].period = None
            self._lcspline[i].conf   = "None"

        if lc_list is None:
            print("No spline\n")
            return
        elif lc_list == "all":
            lc_list = self._names
        else:
            if isinstance(lc_list, str): lc_list = [lc_list]
        
        nlc_spl = len(lc_list)   #number of lcs to add spline to
        for lc in lc_list:
            assert lc in self._names, f"add_spline(): {lc} not in loaded lc files: {self._names}."
        
        DA = locals().copy()
        _ = [DA.pop(item) for item in ["self", "verbose","lc"]]  

        for p in ["par","degree","knots_every","periodicity"]:
            if DA[p] is None: DA[p] = [None]*nlc_spl
            if isinstance(DA[p], (str,int,float,tuple)): DA[p] = [DA[p]]*nlc_spl
            if isinstance(DA[p], list): assert len(DA[p])==nlc_spl, f"add_spline(): {p} must be a list of length {nlc_spl} or length 1 (if same is to be used for all lcs)."
            
            #check if inputs are valid
            for list_item in DA[p]:
                if p=="par":
                    if isinstance(list_item, str): assert list_item in ["col0","col3","col4","col5","col6","col7",None],f'add_spline(): {p} must be in ["col0","col3","col4","col5"] but {list_item} given.'
                    if isinstance(list_item, tuple): 
                        for tup_item in list_item: assert tup_item in ["col0","col3","col4","col5","col6","col7",None],f'add_spline(): {p} must be in ["col0","col3","col4","col5"] but {tup_item} given.'
                if p=="degree": 
                    assert isinstance(list_item, (int,tuple)),f'add_spline(): {p} must be an integer but {list_item} given.'
                    if isinstance(list_item, tuple):
                        for tup_item in list_item: assert isinstance(tup_item, int),f'add_spline(): {p} must be an integer but {tup_item} given.'

        for i,lc in enumerate(lc_list):
            ind = self._names.index(lc)    #index of lc in self._names
            par, deg, knots, period =  DA["par"][i], DA["degree"][i], DA["knots_every"][i], DA["periodicity"][i]
            dim = 1 if isinstance(par,str) else len(par)
            assert dim <=2, f"add_spline(): dimension of spline must be 1 or 2 but {par} (dim {dim}) given for {lc}."
            if dim==2:   #if 2d spline 
                if isinstance(deg, int): deg = (deg,deg)  #if degree is int, make it a tuple
                if isinstance(knots, int): knots = (knots,knots)
                if isinstance(period, int): period = (period,period)

            self._lcspline[ind].name   = lc
            self._lcspline[ind].dim    = dim
            self._lcspline[ind].par    = par
            self._lcspline[ind].use    = True if par else False
            self._lcspline[ind].deg    = deg
            self._lcspline[ind].knots  = knots
            self._lcspline[ind].period = period
                
            if dim==1:
                self._lcspline[ind].conf   = f"c{par[-1]}:d{deg}K{knots}P{period}"
            else:
                self._lcspline[ind].conf   = f"c{par[0][-1]}:d{deg[0]}K{knots[0]}P{period[0]}|c{par[1][-1]}:d{deg[1]}K{knots[1]}P{period[1]}"

            if verbose: 
                print(f"{lc} – Adding a spline to fit {par}: knots = {knots}, periodicity={period}")
        
        if verbose: _print_output(self,"lc_baseline")

    def add_GP(self, lc_list=None, pars="col0", kernels="mat32", WN="y", 
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
                For each lightcurve, `par` can be any of "col0", "col3", "col4", "col5", "col6", "col7", "col8". Right now only "col0" is supported for celerite
                
            kernel : list of strings;
                GP kernel for each lightcuve file in lc_list. Options: ["mat32","sqexp] for george, ["mat32","sho"] for celerite.
                
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
        assert hasattr(self,"_bases"), f"add_GP(): need to run lc_baseline() function before adding GP."
        assert isinstance(log_scale, (tuple,list)), f"add_GP(): log_scale must be a list of tuples specifying value for each lc or single tuple if same for all lcs."
        assert isinstance(log_metric, (tuple,list)), f"add_GP(): log_metric must be a list of tuples specifying value for each lc or single tuple if same for all lcs."

        if isinstance(log_scale, tuple):  log_scale  = [log_scale]
        if isinstance(log_metric, tuple): log_metric = [log_metric]

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
            
            else: _raise(TypeError, f"add_GP(): tuple of len 2 or 3 was expected but got the value {s} in log_scale.")

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

            else: _raise(TypeError, f"add_GP: tuple of len 2 or 3 was expected but got the value {m} in log_metric.")


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
                assert lc in lc_list,f"add_GP(): GP was expected for {lc} but was not given in lc_list."   

            for lc in lc_list: 
                assert lc in self._names,f"add_GP(): {lc} is not one of the loaded lightcurve files"
                assert lc in self._gp_lcs, f"add_GP(): while defining baseline model in the `lc_baseline` method, gp = 'y' or 'ce' was not specified for {lc}."
        n_list = len(lc_list)
        
        #transform        
        for key in DA.keys():
            if (isinstance(DA[key],list) and len(DA[key])==1): 
                DA[key]= DA[key]*n_list
            if isinstance(DA[key], list):
                assert len(DA[key]) == n_list, f"add_GP(): {key} must have same length as lc_list"
            if isinstance(DA[key],(float,int,str)):  
                DA[key] = [DA[key]]*n_list
                
        
        for p in DA["pars"]: 
            assert p in ["col0", "col3", "col4", "col5", "col6", "col7", "col8"], \
                f"add_GP(): pars `{p}` cannot be the GP independent variable. Must be one of ['col0', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8'] but only col0 supported for celerite"             
        
        
        assert len(DA["pars"]) == len(DA["kernels"]) == len(DA["WN"]) == n_list, f"add_GP():pars and kernels must have same length as lc_list (={len(lc_list)})"
                                            
        self._GP_dict = DA     #save dict of gp pars in lc object

        if verbose: _print_output(self,"gp")

        if self._show_guide: print("\nNext: use method `setup_transit_rv` to configure transit parameters.")

    def setup_transit_rv(self, RpRs=0., Impact_para=0, rho_star=0, T_0=0, Period=0, 
                 Eccentricity=0, omega=90, K=0, verbose=True):
        """
            Define parameters an priors of model parameters.
            By default, the parameters are fixed to the given values. To fit a parameter use the `to_fit` method to change it from 'n' to 'y'.
            The parameters can be defined in following ways:
            
            * fixed value as float or int, e.g Period = 3.4
            * free parameter with gaussian prior given as tuple of len 2, e.g. T_0 = (5678, 0.1)
            * free parameters with uniform prior interval and initial value given as tuple of length 3, e.g. RpRs = (0,0.1,0.2) with 0.1 being the initial value.

            Parameters:
            -----------
            RpRs : float, tuple;
                Ratio of planet to stellar radius. Default is 0.

            Impact_para : float, tuple;
                Impact parameter of the transit. Default is 0.

            rho_star : float, tuple;
                density of the star in g/cm^3. Default is 0.

            T_0 : float, tuple;
                Mid-transit time in days. Default is 0.

            Period : float, tuple;
                Orbital period of the planet in days. Default is 0.

            Eccentricity : float, tuple;
                Eccentricity of the orbit. Default is 0.

            omega : float, tuple;
                Argument of periastron. Default is 90.

            K : float, tuple;
                Radial velocity semi-amplitude in m/s. Default is 0.

            verbose : bool;
                print output. Default is True.
        """
        
        DA = locals().copy()         #dict of arguments (DA)
        _ = DA.pop("self")                            #remove self from dictionary
        _ = DA.pop("verbose")
        #sort to specific order
        key_order = ["RpRs","Impact_para","rho_star", "T_0", "Period", "Eccentricity","omega", "K"]
        DA = {key:DA[key] for key in key_order if key in DA} 
            
        self._TR_RV_parnames  = [nm for nm in DA.keys()] 
        # self._npars = 8
        self._config_par = {}

        for par in DA.keys():
            if isinstance(DA[par], (float,int,tuple)): DA[par] = [DA[par]]*self._nplanet
            if isinstance(DA[par], list): assert len(DA[par])==self._nplanet, f"setup_transit_rv: {par} must be a list of length {self._nplanet} or float/int/tuple."

        for n in range(self._nplanet):    #n is planet number
            self._config_par[f"pl{n+1}"] = {}

            for par in DA.keys():
                if par in ["RpRs", "Eccentricity"]: up_lim = 1
                elif par == "rho_star":    up_lim = 4
                elif par == "Impact_para": up_lim = 1.5
                elif par == "omega":       up_lim = 360
                else: up_lim = 10000

                #fitting parameter
                #_param_obj([to_fit, start_value, step_size,prior, prior_mean,pr_width_lo, 
                #                                      prior_width_hi, bounds_lo, bounds_hi])
                if isinstance(DA[par][n], tuple):
                    #gaussian       
                    if len(DA[par][n]) == 2:
                        if par in ["T_0","rho_star"]: up_lim = DA[par][n][0]+20*DA[par][n][1]    #uplim is mean+20*sigma   
                        DA[par][n] = _param_obj(["y", DA[par][n][0], 0.1*DA[par][n][1], "p", DA[par][n][0],    #TODO: for code clarity, use argument names instead of index (e.g. to_fit="y",)
                                                        DA[par][n][1], DA[par][n][1], 0, up_lim])
                    #uniform
                    elif len(DA[par][n]) == 3: 
                        DA[par][n] = _param_obj(["y", DA[par][n][1], min(0.01,0.01*np.ptp(DA[par][n])), "n", DA[par][n][1],
                                                        0, 0, DA[par][n][0], DA[par][n][2]])
                    
                    else: _raise(ValueError, f"setup_transit_rv: length of tuple {par} is {len(DA[par][n])} but it must be 2 for gaussian or 3 for uniform priors")
                #fixing parameter
                elif isinstance(DA[par][n], (int, float)):
                    DA[par][n] = _param_obj(["n", DA[par][n], 0.00, "n", DA[par][n],
                                                    0,  0, 0, 1.1*DA[par][n]])

                else: _raise(TypeError, f"setup_transit_rv: {par} for planet{n} must be one of [tuple(of len 2 or 3), int, float] but is {type(DA[par][n])}")

                self._config_par[f"pl{n+1}"][par] = DA[par][n]      #add to object
        # self._items = DA["RpRs"].__dict__.keys()
        
        if verbose: _print_output(self,"transit_rv_pars")


        if self._show_guide: print("\nNext: use method transit_depth_variation` to include variation of RpRs for the different filters or \n`setup_occultation` to fit the occultation depth or \n`limb_darkening` for fit or fix LDCs or `contamination_factors` to add contamination.")


    def update_setup_transit_rv(self, RpRs=0., Impact_para=0, rho_star=0, T_0=0, Period=0, 
                 Eccentricity=0, omega=90, K=0, verbose=True):
        """
            Define parameters an priors of model parameters.
            By default, the parameters are fixed to the given values. To fit a parameter use the `to_fit` method to change it from 'n' to 'y'.
            The parameters can be defined in following ways:
            
            * fixed value as float or int, e.g Period = 3.4
            * free parameter with gaussian prior given as tuple of len 2, e.g. T_0 = (5678, 0.1)
            * free parameters with uniform prior interval and initial value given as tuple of length 3, e.g. RpRs = (0,0.1,0.2) with 0.1 being the initial value.

            Parameters:
            -----------
            RpRs : float, tuple;
                Ratio of planet to stellar radius. Default is 0.

            Impact_para : float, tuple;
                Impact parameter of the transit. Default is 0.

            rho_star : float, tuple;
                density of the star in g/cm^3. Default is 0.

            T_0 : float, tuple;
                Mid-transit time in days. Default is 0.

            Period : float, tuple;
                Orbital period of the planet in days. Default is 0.

            Eccentricity : float, tuple;
                Eccentricity of the orbit. Default is 0.

            omega : float, tuple;
                Argument of periastron. Default is 90.

            K : float, tuple;
                Radial velocity semi-amplitude in m/s. Default is 0.

            verbose : bool;
                print output. Default is True.
        """
        
        DA = locals().copy()         #dict of arguments (DA)
        _ = DA.pop("self")                            #remove self from dictionary
        _ = DA.pop("verbose")

        rm_par= []
        for par in DA.keys():
            if DA[par] == 0: rm_par.append(par)
        _ = [DA.pop(p) for p in rm_par]
        

        for par in DA.keys():
            if isinstance(DA[par], (float,int,tuple)): DA[par] = [DA[par]]*self._nplanet
            if isinstance(DA[par], list): assert len(DA[par])==self._nplanet, f"setup_transit_rv: {par} must be a list of length {self._nplanet} or float/int/tuple."

        for n in range(self._nplanet):    #n is planet number

            for par in DA.keys():
                if par in ["RpRs","rho_star", "Eccentricity"]: up_lim = 1
                elif par == "Impact_para": up_lim = 1.5
                elif par == "omega":       up_lim = 360
                else: up_lim = 10000

                #fitting parameter
                #_param_obj([to_fit, start_value, step_size,prior, prior_mean,pr_width_lo, 
                #                                      prior_width_hi, bounds_lo, bounds_hi])
                if isinstance(DA[par][n], tuple):
                    #gaussian       
                    if len(DA[par][n]) == 2:
                        if par in ["T_0","rho_star"]: up_lim = DA[par][n][0]+20*DA[par][n][1]    #uplim is mean+20*sigma   
                        DA[par][n] = _param_obj(["y", DA[par][n][0], 0.1*DA[par][n][1], "p", DA[par][n][0],    #TODO: for code clarity, use argument names instead of index (e.g. to_fit="y",)
                                                        DA[par][n][1], DA[par][n][1], 0, up_lim])
                    #uniform
                    elif len(DA[par][n]) == 3: 
                        DA[par][n] = _param_obj(["y", DA[par][n][1], min(0.01,0.01*np.ptp(DA[par][n])), "n", DA[par][n][1],
                                                        0, 0, DA[par][n][0], DA[par][n][2]])
                    
                    else: _raise(ValueError, f"setup_transit_rv: length of tuple {par} is {len(DA[par][n])} but it must be 2 for gaussian or 3 for uniform priors")
                #fixing parameter
                elif isinstance(DA[par][n], (int, float)):
                    DA[par][n] = _param_obj(["n", DA[par][n], 0.00, "n", DA[par][n],
                                                    0,  0, 0, 1.1*DA[par][n]])

                else: _raise(TypeError, f"setup_transit_rv: {par} for planet{n} must be one of [tuple(of len 2 or 3), int, float] but is {type(DA[par][n])}")

                self._config_par[f"pl{n+1}"][par] = DA[par][n]      #add to object
        # self._items = DA["RpRs"].__dict__.keys()
        
        if verbose: _print_output(self,"transit_rv_pars")


        if self._show_guide: print("\nNext: use method transit_depth_variation` to include variation of RpRs for the different filters or \n`setup_occultation` to fit the occultation depth or \n`limb_darkening` for fit or fix LDCs or `contamination_factors` to add contamination.")


    def transit_depth_variation(self, ddFs="n", transit_depth_per_group=[(0.1,0.0001)], divwhite="n",
                        step=0.001, bounds=(-1,1), prior="n", prior_width=(0,0),
                       verbose=True):
        """
            Include transit depth variation between the different lcs or lc groups. Note RpRs must be fixed to a reference value and not a jump parameter.
            
            Parameters:
            -----------

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
        
        assert isinstance(transit_depth_per_group, (tuple,list)),f"transit_depth_variation: transit_depth_per_group must be type tuple or list of tuples."
        if isinstance(transit_depth_per_group,tuple): transit_depth_per_group = [transit_depth_per_group]
        depth_per_group     = [d[0] for d in transit_depth_per_group]
        depth_err_per_group = [d[1] for d in transit_depth_per_group]

        assert isinstance(prior_width, tuple),f"transit_depth_variation(): prior_width must be tuple with lower and upper widths."
        prior_width_lo, prior_width_hi = prior_width

        assert isinstance(bounds, tuple),f"transit_depth_variation(): bounds must be tuple with lower and upper values."
        bounds_lo, bounds_hi = bounds


        width_lo = (0 if (prior == 'n' or ddFs == 'n' or bounds_lo == 0.) else prior_width_lo)
        width_hi = (0 if (prior == 'n' or ddFs == 'n' or bounds_hi == 0.) else prior_width_hi)

        self._ddfs.drprs_op=[0., step, bounds_lo, bounds_hi, 0., width_lo, width_hi]  # the dRpRs options
        
        grnames    = np.array(list(sorted(set(self._groups))))
        ngroup     = len(grnames)
        
        if len(depth_per_group)==1: depth_per_group = depth_per_group * ngroup     #depth for each group
        if len(depth_err_per_group)==1: depth_err_per_group = depth_err_per_group * ngroup

        
        assert len(depth_per_group)== len(depth_err_per_group)== ngroup, \
            f"transit_depth_variation(): length of depth_per_group and depth_err_per_group must be equal to the number of unique groups (={ngroup}) defined in `lc_baseline`"
        
        self._ddfs.depth_per_group     = depth_per_group
        self._ddfs.depth_err_per_group = depth_err_per_group
        self._ddfs.divwhite            = divwhite
        self._ddfs.prior               = prior
        self._ddfs.ddfYN               = ddFs
        self._ddfs.prior_width_lo      = prior_width_lo
        self._ddfs.prior_width_hi      = prior_width_hi
        if divwhite=="y":
            assert ddFs=='n', 'transit_depth_variation(): you can not do divide-white and not fit ddfs!'
            
            for i in range(self._nphot):
                if (self._bases[i][6]>0):
                    _raise(ValueError, 'transit_depth_variation(): you can not have CNMs active and do divide-white')
        

        if self._nphot>0: 
            if (ddFs=='n' and np.max(self._grbases)>0): _raise(ValueError,'no ddFs but groups? Not a good idea!')
            
        if verbose: _print_output(self,"depth_variation")
                
    def setup_occultation(self, filters_occ=None, start_depth=[(0,20e-6,1000e-6)], step_size=0.00001,verbose=True):
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

        assert isinstance(start_depth,(int,float,tuple,list)), f"setup_occulation():start depth must be list of tuple/float for depth in each filter or tuple/float for same in all filters."
        if isinstance(start_depth, (int,float,tuple)): start_depth= [start_depth]
        # unpack start_depth input
        start_value, prior, prior_mean, prior_width_hi, prior_width_lo, bounds_hi, bounds_lo = [],[],[],[],[],[],[]
        for dp in start_depth:
            if isinstance(dp, (int,float)):
                start_value.append(dp)
                prior.append("n")
                prior_mean.append(0)
                prior_width_hi.append(0)
                prior_width_lo.append(0)
                bounds_lo.append(0)
                bounds_hi.append(0)

            elif isinstance(dp,tuple) and len(dp)==2:
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

            else: _raise(TypeError, f"setup_occultation(): float or tuple (of len 2 or 3) was expected but got the value {dp} in start_depth.")


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
                f"setup_occultation(): {f} is not in list of defined filters"
            
            for par in DA.keys():
                assert isinstance(DA[par], (int,float,str)) or \
                    (isinstance(DA[par], list) and ( (len(DA[par]) == nocc) or (len(DA[par]) == 1))), \
                    f"setup_occultation(): length of input {par} must be equal to the length of filters_occ (={nocc}) or float or None."

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

        DA2["filt_to_fit"] = [("y" if step else "n") for step in DA2["step_size"]]
        self._occ_dict =  DA = DA2
        if verbose: _print_output(self,"occultations")

    def get_LDs(self,Teff,logg,Z, filter_names, unc_mult=10, use_result=True, verbose=True):
        """
        get quadratic limb darkening parameters using ldtk (requires internet connection).

        Parameters
        ----------
        Teff : tuple
            (value, std) of stellar effective temperature

        logg : tuple
            (value, std) of stellar logg

        Z : tuple
            (value, std) of stellar metallicity

        filter_names : list of str
            SVO filter name such as 'Spitzer/IRAC.I1' or a name shortcut such as "TESS", "CHEOPS","kepler". 
            use `lc_data._filter_shortcuts` to get list of filter shortcut names. Filter names can be obtained from http://svo2.cab.inta-csic.es/theory/fps/

        unc_mult : int/float, optional
            value by which to multiply ldtk uncertainties which are usually underestimated, by default 10

        use_result : bool, optional
            whether to use the result to setup limb darkening priors, by default True

        Returns
        -------
        u1, u2 : arrays
            each coefficient is an array of only values (no uncertainity) for each filter. 
            These can be fed to the `limb_darkening()` function to fix the coefficients
        """

        from ldtk import LDPSetCreator, BoxcarFilter
        u1, u2 = [], []
        
        if isinstance(filter_names,list): 
            assert len(filter_names)==len(self._filnames),\
                f"get_LDs: number of unique filters in input lc_list = {len(self._filnames)} but trying to obtain LDs for {len(filter_names)} filters."
        elif isinstance(filter_names, str): filter_names = [filter_names]
        else: raise TypeError(f"get_LDs: filter_names must be str for a single filter or list of str but {type(filter_names)} given.")

        for i,f in enumerate(filter_names):
            if f.lower() in self._filter_shortcuts.keys(): ft = self._filter_shortcuts[f.lower()]
            else: ft=f
            flt = SVOFilter(ft)
            ds  = 'visir-lowres' if np.any(flt.wavelength > 1000) else 'vis-lowres'

            sc  = LDPSetCreator(teff=Teff, logg=logg, z=Z,    # spectra from the Husser et al.
                                filters=[flt], dataset=ds)      # FTP server automatically.

            ps = sc.create_profiles(100)                      # Create the limb darkening profiles\
            ps.set_uncertainty_multiplier(unc_mult)
            ps.resample_linear_z(300)

            #calculate ld profiles
            c, e = ps.coeffs_qd(do_mc=True, n_mc_samples=10000,mc_burn=1000) 
            ld1 = (round(c[0][0],4),round(e[0][0],4))
            ld2 = (round(c[0][1],4),round(e[0][1],4))
            u1.append(ld1)
            u2.append(ld2)
            if verbose: print(f"{f:10s}({self._filters[i]}): u1={ld1}, u2={ld2}")

        if use_result: 
            u_1 = deepcopy(u1)
            u_2 = deepcopy(u2)
            if verbose: print(_text_format.BOLD + "\nSetting-up limb-darkening priors from LDTk result" +_text_format.END)
            self.limb_darkening(u_1,u_2)
        return u1,u2



    def limb_darkening(self, c1=0, c2 = 0,verbose=True):
        """
            Setup quadratic limb darkening LD coefficient (c1, c2) for transit light curves.
            c1 and c2 stand for coefficient 1 and 2 (Note they are usually denoted u1 and u2 in literature) 
            Different LD coefficients are required if observations of different filters are used.

            Parameters:
            -----------
            c1,c2 : float/tuple or list of float/tuple for each filter;
                Stellar quadratic limb darkening coefficients.
                if tuple, must be of - length 2 for normal prior (mean,std) or length 3 for uniform prior defined as (lo_lim, val, uplim).
                **recall the conditions: c1+c2<1, c1>0, c1+c2>0  (https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract)\n
                This implies the a broad uniform prior of [0,2] for c1 and [-1,1] for c2. However, it is highly recommended to use gaussian priors on c1 and c2. 

            Note: c1,c2 are reparameterised in the mcmc fitting to: 2*c1+c2 and c1-c2 
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
            elif isinstance(DA[par], list): assert len(DA[par]) == nfilt,f"limb_darkening: length of list {par} must be equal to number of unique filters (={nfilt})."
            else: _raise(TypeError, f"limb_darkening: {par} must be int/float, or tuple of len 2 (for gaussian prior) or 3 (for uniform prior) but {DA[par]} is given.")
        
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
                        DA[f"step{par[-1]}"][i] = 0.1*DA[f"sig_lo{par[-1]}"][i] if d[1] else 0  #if width is > 0


                    if len(d) == 3:  #uniform prior
                        if d[0]!= 0 and d[2]!=0: assert d[0]<d[1]<d[2],f'limb_darkening: uniform prior be (lo_lim, val, uplim) where lo_lim < val < uplim but {d} given.'
                        DA[par][i] = d[1]
                        DA[f"bound_lo{par[-1]}"][i] = d[0]
                        DA[f"bound_hi{par[-1]}"][i] = d[2]
                        DA[f"sig_lo{par[-1]}"][i] = DA[f"sig_hi{par[-1]}"][i] = 0
                        DA[f"step{par[-1]}"][i] = min(0.001, np.ptp([d[0],d[2]])) if (d[0] or d[2]) else 0 #if bounds !=  0
  
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

    def stellar_parameters(self,R_st=None, M_st=None, par_input = "Rrho", verbose=True):
        """
            input parameters of the star

            Parameters:
            -----------

            R_st, Mst : tuple of length 2 or 3;
                stellar radius and mass (in solar units) to use for calculating absolute dimensions.
                First tuple element is the value and the second is the uncertainty. use a third element if asymmetric uncertainty
            
            par_input : str;
                input method of stellar parameters. It can be one of  ["Rrho","Mrho"], to use the fitted stellar density and one stellar parameter (M_st or R_st) to compute the other stellar parameter (R_st or M_st).
                Default is 'Rrho' to use the fitted stellar density and stellar radius to compute the stellar mass.

        """
    

        DA = _reversed_dict(locals().copy())
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")
        
        for par in ["R_st", "M_st"]:
            assert DA[par] is None or isinstance(DA[par],tuple), f"stellar_parameters: {par} must be either None or tuple of length 2 or 3 "
            if DA[par] is None: DA[par] = (1,0.01)
            if isinstance(DA[par],tuple):
                assert len(DA[par])==2 or len(DA[par]) <=3, f"stellar_parameters: length of {par} tuple must be 2 or 3 "
                if len(DA[par])== 2: DA[par]= (DA[par][0], DA[par][1], DA[par][1])
        
        assert DA["par_input"] in ["Rrho","Mrho"], f"stellar_parameters: par_input must be one of ['Rrho','Mrho']."
            
        self._stellar_dict = DA
         
        if verbose: _print_output(self,"stellar_pars")

    def __repr__(self):
        data_type = str(self.__class__).split("load_")[1].split("'>")[0]
        return f'Object containing {self._nphot} {data_type}.\n{self._nplanet} transiting planet(s)\nFiles:{self._names}\nFilepath: {self._fpath}'
 
    def print(self, section="all"):
        """
            Print out all input configuration (or particular section) for the light curve object. 
            It is printed out in the format of the legacy CONAN config file.
            Parameters:
            ------------
            section : str (optional) ;
                section of configuration to print.Must be one of ["lc_baseline", "gp", "transit_rv_pars", "depth_variation", "occultations", "limb_darkening", "contamination", "stellar_pars"].
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
            assert section in possible_sections, f"print: {section} not a valid section of `lc_data`. \
                section must be one of {possible_sections}."
            _print_output(self, section)

    def plot(self, plot_cols=(0,1,2), col_labels=None, nrow_ncols=None, figsize=None, fit_order=0, show_decorr_model=False, return_fig=False):
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

            show_decorr_model : bool;
                show decorrelation model if decorrelation has been done.
            
            figsize: tuple of length 2;
                Figure size. If None, (8,5) is used for a single input file and optimally determined for more inputs.

            return_fig  : bool;
                return figure object for saving to file.
        """
        if not (isinstance(plot_cols, tuple) and len(plot_cols) in [2,3]): 
            raise TypeError(f"plot: plot_cols must be tuple of length 2 or 3, but is {type(plot_cols)} and length of {len(plot_cols)}.")
        
        assert col_labels is None or ((isinstance(col_labels, tuple) and len(col_labels)==2)), \
            f"plot: col_labels must be tuple of length 2, but is {type(col_labels)} and length of {len(col_labels)}."
        
        assert isinstance(fit_order,int),f'fit_order must be an integer'

        if show_decorr_model:
            if not hasattr(self,"_tmodel"): 
                print("cannot show decorr model since decorrelation has not been done. First, use `lc_data.get_decorr()` to launch decorrelation.")
                show_decorr_model = False
        
        if col_labels is None:
            col_labels = ("time", "flux") if plot_cols[:2] == (0,1) else (f"column[{plot_cols[0]}]",f"column[{plot_cols[1]}]")
        
        if self._names != []:
            fig = _plot_data(self, plot_cols=plot_cols, col_labels = col_labels, nrow_ncols=nrow_ncols, figsize=figsize, fit_order=fit_order,
                            model_overplot=self._tmodel if show_decorr_model else None)
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

        nplanet : int;
            number of planets in the system. Default is 1.

        rv_unit : str;
            unit of the rv data. Must be one of ["m_s","km_s"]. Default is "km_s".

        show_guide : bool;
            print output to guide the user. Default is False.

        Returns:
        --------
        rv_data : rv object

        Examples:
        ---------
        >>> rv_data = load_rvs(file_list=["rv1.dat","rv2.dat"], data_filepath="/path/to/data/", rv_unit="km_s")
    """
    def __init__(self, file_list=None, data_filepath=None, nplanet=1, rv_unit="km_s",show_guide =False):
        self._obj_type = "rv_obj"
        self._nplanet = nplanet
        self._fpath = os.getcwd()+"/" if data_filepath is None else data_filepath
        self._names   = [] if file_list is None else file_list 
        assert rv_unit in ["m_s","km_s"], f"load_rvs(): rv_unit must be one of ['m_s','km_s'] but {rv_unit} given." 
        if self._names == []:
            self.rv_baseline(verbose=False)
        else: 
            for rv in self._names: assert os.path.exists(self._fpath+rv), f"file {rv} does not exist in the path {self._fpath}."
            if show_guide: print("Next: use method `rv_baseline` to define baseline model for for the each rv")
            #modify input files to have 6 columns as CONAN expects
            for f in self._names:
                fdata = np.loadtxt(self._fpath+f)
                nrow,ncol = fdata.shape
                if ncol < 6:
                    print(f"Expected at least 6 columns for RV file: writing ones to the missing columns of file: {f}")
                    new_cols = np.ones((nrow,6-ncol))
                    fdata = np.hstack((fdata,new_cols))
                    np.savetxt(self._fpath+f,fdata,fmt='%.8f')
                if rv_unit == "m_s":
                    print(f"Converting cols[1:] of RV file {f} from m/s to km/s.\nNew file is: {f.split('.')[0]}_kms.{f.split('.')[1]}")
                    fdata[:,1:] = fdata[:,1:]/1e3
                    #save new file with _kms suffix
                    fsplit = f.split(".")
                    nf = fsplit[0]+"_kms."+fsplit[1]
                    np.savetxt(self._fpath+nf,fdata,fmt='%.8f')
                    self._names[self._names.index(f)] = nf
        
        self._nRV = len(self._names)
        self.rv_baseline(verbose=False)

    # def update_setup_transit_rv(self, Eccentricity=0, omega=90, K=0, verbose=True):
    #     """
    #         Define parameters for Eccentricity 
    #         By default, the parameters are fixed to the given values. To fit a parameter use the `to_fit` method to change it from 'n' to 'y'.
    #         The parameters can be defined in following ways:
            
    #         * fixed value as float or int, e.g Period = 3.4
    #         * free parameter with gaussian prior given as tuple of len 2, e.g. K = (20, 5)
    #         * free parameters with uniform prior interval and initial value given as tuple of length 3, e.g. omega = (0,90,360) with 90 being the initial value.

    #         Parameters:
    #         -----------
    #         Eccentricity : float, tuple;
    #             Eccentricity of the orbit. Default is 0.

    #         omega : float, tuple;
    #             Argument of periastron. Default is 90.

    #         K : float, tuple;
    #             Radial velocity semi-amplitude in m/s. Default is 0.

    #         verbose : bool;
    #             print output. Default is True.
    #     """
        
    #     DA = locals().copy()         #dict of arguments (DA)
    #     _ = DA.pop("self")                            #remove self from dictionary
    #     _ = DA.pop("verbose")
    #     #sort to specific order

    #     self._rvconfig_par = {}

    #     for par in DA.keys():
    #         if isinstance(DA[par], (float,int,tuple)): DA[par] = [DA[par]]*self._nplanet
    #         if isinstance(DA[par], list): assert len(DA[par])==self._nplanet, f"setup_transit_rv: {par} must be a list of length {self._nplanet} or float/int/tuple."

    #     for n in range(self._nplanet):    #n is planet number
    #         self._rvconfig_par[f"pl{n+1}"] = {}

    #         for par in DA.keys():
    #             if par =="Eccentricity": up_lim = 1
    #             elif par == "omega":       up_lim = 360
    #             else: up_lim = 10000

    #             #fitting parameter
    #             #_param_obj([to_fit, start_value, step_size,prior, prior_mean,pr_width_lo, 
    #             #                                      prior_width_hi, bounds_lo, bounds_hi])
    #             if isinstance(DA[par][n], tuple):
    #                 #gaussian       
    #                 if len(DA[par][n]) == 2:
    #                     if par in ["T_0","Duration"]: up_lim = DA[par][n][0]+20*DA[par][n][1]    #uplim is mean+20*sigma   
    #                     DA[par][n] = _param_obj(["y", DA[par][n][0], 0.1*DA[par][n][1], "p", DA[par][n][0],    #TODO: for code clarity, use argument names instead of index (e.g. to_fit="y",)
    #                                                     DA[par][n][1], DA[par][n][1], 0, up_lim])
    #                 #uniform
    #                 elif len(DA[par][n]) == 3: 
    #                     DA[par][n] = _param_obj(["y", DA[par][n][1], min(0.01,0.01*np.ptp(DA[par][n])), "n", DA[par][n][1],
    #                                                     0, 0, DA[par][n][0], DA[par][n][2]])
                    
    #                 else: _raise(ValueError, f"setup_transit_rv: length of tuple {par} is {len(DA[par][n])} but it must be 2 for gaussian or 3 for uniform priors")
    #             #fixing parameter
    #             elif isinstance(DA[par][n], (int, float)):
    #                 DA[par][n] = _param_obj(["n", DA[par][n], 0.00, "n", DA[par][n],
    #                                                 0,  0, 0, 1.1*DA[par][n]])

    #             else: _raise(TypeError, f"setup_transit_rv: {par} for planet{n} must be one of [tuple(of len 2 or 3), int, float] but is {type(DA[par][n])}")

    #             self._rvconfig_par[f"pl{n+1}"][par] = DA[par][n]      #add to object
    #     # self._items = DA["RpRs"].__dict__.keys()
        
    #     if verbose: _print_output(self,"transit_rv_pars")



    def get_decorr(self, T_0=None, Period=None, K=None, sesinw=0, secosw=0, gamma=0,
                    delta_BIC=-5, decorr_bound =(-1000,1000), exclude=[],enforce=[],verbose=True, 
                        show_steps=False, plot_model=True, use_result=True):
        """
            Function to obtain best decorrelation parameters for each rv file using the forward selection method.
            It compares a model with only an offset to a polynomial model constructed with the other columns of the data.
            It uses columns 0,3,4,5 to construct the polynomial trend model. The temporary decorr parameters are labelled Ai,Bi for 1st & 2nd order in column i.
            Decorrelation parameters that reduces the BIC by 5(i.e delta_BIC = -5, 12X more probable) are iteratively selected.
            The result can then be used to populate the `rv_baseline()` method, if use_result is set to True.

            Parameters:
            -----------
            T_0, Period, K, Eccentricity, omega : floats, None;
                RV parameters of the planet. T_0 and P must be in same units as the time axis (cols0) in the data file.
                if float/int, the values are held fixed. if tuple/list of len 2 implies [min,max] while len 3 implies [min,start_val,max].

            delta_BIC : float (negative);
                BIC improvement a parameter needs to provide in order to be considered relevant for decorrelation. + \
                    Default is conservative and set to -5 i.e, parameters needs to lower the BIC by 5 to be included as decorrelation parameter.

            decorr_bound: tuple of size 2;
                bounds when fitting decorrelation parameters. Default is (-1000,1000)
                
            exclude : list of int;
                list of column numbers (e.g. [3,4]) to exclude from decorrelation. Default is [].

            enforce : list of int;
                list of decorr params (e.g. ['B3', 'A5']) to enforce in decorrelation. Default is [].

            verbose : Bool, optional;
                Whether to show the table of baseline model obtained. Defaults to True.

            show_steps : Bool, optional;
                Whether to show the steps of the forward selection of decorr parameters. Default is False
            
            plot_model : Bool, optional;
                Whether to overplot suggested trend model on the data. Defaults to True.

            use_result : Bool, optional;
                whether to use result/input to setup the baseline model and transit/eclipse models. Default is True.
        
            Returns
            -------
            decorr_result: list of result object
                list containing result object for each lc.
        """
        assert isinstance(exclude, list), f"get_decorr: exclude must be a list of column numbers to exclude from decorrelation but {exclude} given."
        for c in exclude: assert isinstance(c, int), f"get_decorr: column number to exclude from decorrelation must be an integer but {c} given in exclude."
        assert delta_BIC<0,f'get_decorr(): delta_BIC must be negative for parameters to provide improved fit but {delta_BIC} given.'
        if isinstance(gamma, tuple):
            assert len(gamma)==2,f"get_decorr(): gamma must be float or tuple of length 2, but {gamma} given."
        

        blpars = {"dcol0":[], "dcol3":[],"dcol4":[], "dcol5":[]}  #inputs to rv_baseline method
        self._rvdecorr_result = []   #list of decorr result for each lc.
        self._rvmodel = []  #list to hold determined trendmodel for each rv


        self._rv_pars = dict(T_0=T_0, Period=Period, K=K, sesinw=sesinw, secosw=secosw, gamma=gamma) #rv parameters
        for p in self._rv_pars:
            if p != "gamma":
                if isinstance(self._rv_pars[p], (int,float,tuple)): self._rv_pars[p] = [self._rv_pars[p]]*self._nplanet
                if isinstance(self._rv_pars[p], (list)): assert len(self._rv_pars[p]) == self._nplanet, \
                    f"get_decorr(): {p} must be a list of same length as number of planets {self._nplanet} but {len(self._rv_pars[p])} given."


        decorr_cols = [0,3,4,5]
        for c in exclude: assert c in decorr_cols, f"get_decorr(): column number to exclude from decorrelation must be in {decorr_cols} but {c} given in exclude." 
        _ = [decorr_cols.remove(c) for c in exclude]  #remove excluded columns from decorr_cols

        for j,file in enumerate(self._names):
            if verbose: print(_text_format.BOLD + f"\ngetting decorrelation parameters for rv: {file}" + _text_format.END)
            all_par = [f"{L}{i}" for i in decorr_cols for L in ["A","B"]] 

            out = _decorr_RV(self._fpath+file, **self._rv_pars, decorr_bound=decorr_bound, npl=self._nplanet)    #no trend, only offset
            best_bic = out.bic
            best_pars = {}                      #parameter salways included
            for cp in enforce: best_pars[cp]=0            #add enforced parameters
            _ = [all_par.remove(cp) for cp in enforce if cp in all_par]    #remove enforced parameters from all_par

            if show_steps: print(f"{'Param':7s} : {'BIC':6s} N_pars \n---------------------------")
            del_BIC = -np.inf 
            while del_BIC < delta_BIC:
                if show_steps: print(f"{'Best':7s} : {best_bic:.2f} {len(best_pars.keys())} {list(best_pars.keys())}\n---------------------")
                pars_bic = {}
                for p in all_par:
                    dtmp = best_pars.copy()  #temporary dict to hold parameters to test
                    dtmp[p] = 0
                    out = _decorr_RV(self._fpath+file, **self._rv_pars,**dtmp, decorr_bound=decorr_bound, npl=self._nplanet)
                    if show_steps: print(f"{p:7s} : {out.bic:.2f} {out.nvarys}")
                    pars_bic[p] = out.bic

                par_in = min(pars_bic,key=pars_bic.get)   #parameter that gives lowest BIC
                par_in_bic = pars_bic[par_in]
                del_BIC = par_in_bic - best_bic
                bf = np.exp(-0.5*(del_BIC))
                if show_steps: print(f"+{par_in} -> BF:{bf:.2f}, del_BIC:{del_BIC:.2f}")

                if del_BIC < delta_BIC:# if bf>1:
                    if show_steps: print(f"adding {par_in} lowers BIC to {par_in_bic:.2f}\n" )
                    best_pars[par_in]=0
                    best_bic = par_in_bic
                    all_par.remove(par_in)            
                      
            result = _decorr_RV(self._fpath+file, **self._rv_pars,**best_pars, decorr_bound=decorr_bound, npl=self._nplanet)
            self._rvdecorr_result.append(result)
            print(f"\nBEST BIC:{result.bic:.2f}, pars:{list(best_pars.keys())}")
            
            #calculate determined trend and rv model over all data
            pps = result.params.valuesdict()
            #convert result transit parameters to back to a list
            for p in ["T_0", "Period", "K", "sesinw", "secosw"]:
                if self._nplanet==1:
                    pps[p] = [pps[p]]  
                else:      
                    pps[p] = [pps[p+f"_{n}"] for n in range(1,self._nplanet+1)]
                    _      = [pps.pop(f"{p}_{n}") for n in range(1,self._nplanet+1)]
    
            self._rvmodel.append(_decorr_RV(self._fpath+file,**pps,decorr_bound=decorr_bound, npl=self._nplanet, return_models=True))

            #set-up lc_baseline model from obtained configuration
            blpars["dcol0"].append( 2 if pps["B0"]!=0 else 1 if  pps["A0"]!=0 else 0)
            blpars["dcol3"].append( 2 if pps["B3"]!=0 else 1 if  pps["A3"]!=0 else 0)
            blpars["dcol4"].append( 2 if pps["B4"]!=0 else 1 if  pps["A4"]!=0 else 0)
            blpars["dcol5"].append( 2 if pps["B5"]!=0 else 1 if  pps["A5"]!=0 else 0)

        if plot_model:
            _plot_data(self,plot_cols=(0,1,2),col_labels=("time","rv"),model_overplot=self._rvmodel)
        

        #prefill other light curve setup from the results here or inputs given here.
        if use_result:
            if verbose: print(_text_format.BOLD + "Setting-up rv baseline model from result" +_text_format.END)
            self.rv_baseline(dt = blpars["dcol0"], dbis=blpars["dcol3"], dfwhm=blpars["dcol4"],
                             dcont=blpars["dcol5"], gammas_kms= gamma, verbose=verbose)

        return self._rvdecorr_result
    
    def rv_baseline(self, dt=None, dbis=None, dfwhm=None, dcont=None,sinPs=None,
                    gammas_kms=0.0, gam_steps=0.001, 
                    verbose=True):
        
        """
            Define rv baseline model parameters to fit.
            Each baseline model parameter should be a list of numbers specifying the polynomial order for each rv data.
            e.g. Given 3 input rvs, and one wishes to fit a 2nd order time trend to only the first and third lightcurves,
            then dt = [2, 0, 2].

            dt, dbis, dfwhm,dcont: list of ints;
                decorrelatation parameters: time, bis, fwhm, contrast
                
            gammas_kms: tuple,floats or list of tuple/float;
                specify if to fit for gamma. if float/int, it is fixed to this value. If tuple of len 2 it is fitted gaussian prior as (prior_mean, width). Uniform prior cannot be defined for gamma. use wide gaussian prior instead.
        """

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
            else: _raise(TypeError, f"a tuple of len 2, float or int was expected but got the value {g} of len {len(g)} in gammas_kms. instead of uniform priors wide gaussian priors should be used")

        DA = locals().copy()     #get a dictionary of the input/variables arguments for easy manipulation
        _ = DA.pop("self")            #remove self from dictionary
        _ = [DA.pop(item) for item in ["verbose","gammas_kms","g"]]


        for par in DA.keys():
            assert DA[par] is None or isinstance(DA[par], (int,float)) or (isinstance(DA[par], (list,np.ndarray)) and len(DA[par]) == self._nRV), f"parameter {par} must be a list of length {self._nRV} or int (if same degree is to be used for all RVs) or None (if not used in decorrelation)."
            
            if DA[par] is None: DA[par] = [0]*self._nRV
            elif isinstance(DA[par], (int,float,str)): DA[par] = [DA[par]]*self._nRV
            

        self._RVbases = [ [DA["dt"][i], DA["dbis"][i], DA["dfwhm"][i], DA["dcont"][i],DA["sinPs"][i]] for i in range(self._nRV) ]

        self._gammas = DA["gammas"]
        self._gamsteps = DA["gam_steps"]
        self._gampri = DA["gam_pri"]
        
        self._prior = DA["prior"]
        self._siglo = DA["sig_lo"]
        self._sighi = DA["sig_hi"]
        
        gampriloa=[]
        gamprihia=[]
        for i in range(self._nRV):
            gampriloa.append( 0. if (self._prior[i] == 'n' or self._gamsteps[i] == 0.) else self._siglo[i])
            gamprihia.append( 0. if (self._prior[i] == 'n' or self._gamsteps[i] == 0.) else self._sighi[i])
        
        self._gamprilo = DA["gampriloa"] = gampriloa                
        self._gamprihi = DA["gamprihia"] = gamprihia                
        self._sinPs    = DA["sinPs"]
        
        self._rvdict   = DA
        if not hasattr(self,"_rvspline"):        self.add_spline(None, verbose=False)

        if verbose: _print_output(self,"rv_baseline")
    

    def add_spline(self, rv_list=None, par = None, degree=3, knots_every=None, periodicity=0,verbose=True):
        """
            add spline to fit correlation along 1 or 2 columns of the data. This splits the data at the defined knots interval and fits a spline to each section. 
            scipy's LSQUnivariateSpline() and LSQBivariateSpline() functions are used for 1D spline and 2D splines respectively.
            All arguments can be given as a list to specify config for each rv file in rv_list.

            Parameters
            ----------
            rv_list : list, str, optional
                list of rv files to fit a spline to. set to "all" to use spline for all rv files. Default is None for no splines.

            par : str,tuple,list, optional
                column of input data to which to fit the spline. must be one/two of ["col0","col3","col4","col5"]. Default is None.
                Give list of columns if different for each rv file. e.g. ["col0","col3"] for spline in col0 for rv1.dat and col3 for rv2.dat. 
                For 2D spline for an rv file, use tuple of length 2. e.g. ("col0","col3") for simultaneous spline fit to col0 and col3.

            degree : int, tuple, list optional
                Degree of the smoothing spline. Must be 1 <= degree <= 5. Default is 3 for a cubic spline.
            
            knots_every : float, tuple, list
                distance between knots of the spline, by default 15 degrees for cheops data roll-angle 

            periodicity : float, tuple, list optional
                periodicity of the spline in that axis  of the data. e.g for cheops roll angle in degrees, the periodicity should be set to 360 degrees. 
                Default is zero for no periodicity.
            
            verbose : bool, optional
                print output. Default is True.

            Examples
            --------
            To use different spline configuration for 2 rv files: 2D spline for the first file and 1D for the second.
            >>> rv_data.add_spline(rv_list=["rv1.dat","rv2.dat"], par=[("col3","col4"),"col4"], degree=[(3,3),2], knots_every=[(5,3),2], periodicity=0)
            
            For same spline configuration for all loaded RV files
            >>> rv_data.add_spline(rv_list="all", par="col3", degree=3, knots_every=5, periodicity=0)
        """  
        #default spline config -- None
        self._rvspline = [None]*self._nRV                   #list to hold spline configuration for each rv
        for i in range(self._nRV):
            self._rvspline[i]        = SimpleNamespace()    #create empty namespace for each rv
            self._rvspline[i].name   = self._names[i]
            self._rvspline[i].dim    = 0
            self._rvspline[i].par    = None
            self._rvspline[i].use    = False
            self._rvspline[i].deg    = None
            self._rvspline[i].knots  = None
            self._rvspline[i].period = None
            self._rvspline[i].conf   = "None"

        if rv_list is None:
            print("No spline\n")
            return
        elif rv_list == "all":
            rv_list = self._names
        else:
            if isinstance(rv_list, str): rv_list = [rv_list]
        
        nrv_spl = len(rv_list)   #number of rvs to add spline to
        for rv in rv_list:
            assert rv in self._names, f"add_spline(): {rv} not in loaded rv files: {self._names}."
        
        DA = locals().copy()
        _ = [DA.pop(item) for item in ["self", "verbose","rv"]]  

        for p in ["par","degree","knots_every","periodicity"]:
            if DA[p] is None: DA[p] = [None]*nrv_spl
            if isinstance(DA[p], (str,int,float,tuple)): DA[p] = [DA[p]]*nrv_spl
            if isinstance(DA[p], list): assert len(DA[p])==nrv_spl, f"add_spline(): {p} must be a list of length {nrv_spl} or length 1 (if same is to be used for all RVs)."
            
            #check if inputs are valid
            for list_item in DA[p]:
                if p=="par":
                    if isinstance(list_item, str): assert list_item in ["col0","col3","col4","col5",None],f'add_spline(): {p} must be in ["col0","col3","col4","col5"] but {list_item} given.'
                    if isinstance(list_item, tuple): 
                        for tup_item in list_item: assert tup_item in ["col0","col3","col4","col5",None],f'add_spline(): {p} must be in ["col0","col3","col4","col5"] but {tup_item} given.'
                if p=="degree": 
                    assert isinstance(list_item, (int,tuple)),f'add_spline(): {p} must be an integer but {list_item} given.'
                    if isinstance(list_item, tuple):
                        for tup_item in list_item: assert isinstance(tup_item, int),f'add_spline(): {p} must be an integer but {tup_item} given.'

        for i,rv in enumerate(rv_list):
            ind = self._names.index(rv)    #index of rv in self._names
            par, deg, knots, period =  DA["par"][i], DA["degree"][i], DA["knots_every"][i], DA["periodicity"][i]
            dim = 1 if isinstance(par,str) else len(par)
            assert dim <=2, f"add_spline(): dimension of spline must be 1 or 2 but {par} (dim {dim}) given for {rv}."
            if dim==2:   #if 2d spline 
                if isinstance(deg, int): deg = (deg,deg)  #if degree is int, make it a tuple
                if isinstance(knots, int): knots = (knots,knots)
                if isinstance(period, int): period = (period,period)

            self._rvspline[ind].name   = rv
            self._rvspline[ind].dim    = dim
            self._rvspline[ind].par    = par
            self._rvspline[ind].use    = True if par else False
            self._rvspline[ind].deg    = deg
            self._rvspline[ind].knots  = knots
            self._rvspline[ind].period = period
            
        if dim==1:
            self._rvspline[ind].conf   = f"c{par[-1]}:d{deg}:K{knots}:P{period}" if par else "None"
        else:
            self._rvspline[ind].conf   = f"c{par[0][-1]}:d{deg[0]}K{knots[0]}P{period[0]}|c{par[1][-1]}:d{deg[1]}K{knots[1]}P{period[1]}"

        if verbose: 
            print(f"Adding a spline to fit {par}: knots = {knots}, periodicity={period}")
            _print_output(self,"rv_baseline")
    
    def __repr__(self):
        data_type = str(self.__class__).split("load_")[1].split("'>")[0]
        return f'Object containing {len(self._names)} {data_type}\nFiles:{self._names}\nFilepath: {self._fpath}'
        
    def plot(self, plot_cols=(0,1,2), col_labels=None, nrow_ncols=None, figsize=None, fit_order=0, show_decorr_model=False,return_fig=False):
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

            fit_order : int;
                order of polynomial to fit to the plotted data columns to visualize correlation.
                
            show_decorr_model : bool;
                show decorrelation model if decorrelation has been done.
            
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

        if show_decorr_model:
            if not hasattr(self,"_rvmodel"): 
                print("cannot show decorr model since decorrelation has not been done. First, use `rv_data.get_decorr()` to launch decorrelation.")
                show_decorr_model = False
        
        if col_labels is None:
            col_labels = ("time", "rv") if plot_cols[:2] == (0,1) else (f"column[{plot_cols[0]}]",f"column[{plot_cols[1]}]")
        
        if self._names != []:
            fig = _plot_data(self, plot_cols=plot_cols, col_labels = col_labels, nrow_ncols=nrow_ncols, fit_order=fit_order, figsize=figsize,
                             model_overplot=self._rvmodel if show_decorr_model else None)
            if return_fig: return fig
        else: print("No data to plot")
    
    def print(self):
        _print_output(self, "rv_baseline")
    
class mcmc_setup:
    """
        class to configure mcmc run
            
        Parameters:
        ------------
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
        
        Returns:
        --------
        mcmc : mcmc object

        Examples:
        ---------
        >>> mcmc = CONAN3.mcmc_setup(n_chains=64, n_steps=2000, n_burn=500, n_cpus=2)

    """
    def __init__(self, n_chains=64, n_steps=2000, n_burn=500, n_cpus=2, sampler=None,
                    leastsq_for_basepar="n", apply_CFs="y",apply_jitter="n",
                    verbose=True, remove_param_for_CNM="n", lssq_use_Lev_Marq="n",
                    GR_test="y", make_plots="n", leastsq="y", savefile="output_ex1.npy",
                    savemodel="n", adapt_base_stepsize="y"):
        
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

                   

class load_result:
    """
        Load results from mcmc run
        
        Parameters:
        ------------
        folder: str;
            folder where the output files are located. Default is "output".
        
        chain_file: str;
            name of the file containing the chains. Default is "chains_dict.pkl".
        
        burnin_chain_file: str;
            name of the file containing the burn-in chains. Default is "burnin_chains_dict.pkl".

        Returns:
        --------
        load_result : load_result object

        Examples:
        ---------
        >>> result = CONAN3.load_result(folder="output")
    """

    def __init__(self, folder="output",chain_file = "chains_dict.pkl", burnin_chain_file="burnin_chains_dict.pkl",verbose=True):

        chain_file        = folder+"/"+chain_file
        burnin_chain_file = folder+"/"+burnin_chain_file
        self._folder = folder
        assert os.path.exists(chain_file) or os.path.exists(burnin_chain_file) , f"file {chain_file} or {burnin_chain_file}  does not exist in the given directory"

        if os.path.exists(chain_file):
            self._chains = pickle.load(open(chain_file,"rb"))
        if os.path.exists(burnin_chain_file):
            self._burnin_chains = pickle.load(open(burnin_chain_file,"rb"))

        self._par_names     = self._chains.keys() if os.path.exists(chain_file) else self._burnin_chains.keys()
        self.params_names   = list(self._par_names)

        #reconstruct posterior from dictionary of chains
        if hasattr(self,"_chains"):
            posterior = np.array([ch for k,ch in self._chains.items()])
            posterior = np.moveaxis(posterior,0,-1)
            s = posterior.shape

            #FLATTEN posterior
            self.flat_posterior = posterior.reshape((s[0]*s[1],s[2]))

        #retrieve summary statistics of the fit
        try:
            self._ind_para      = pickle.load(open(folder+"/.par_config.pkl","rb"))
            self._stat_vals     = pickle.load(open(folder+"/.stat_vals.pkl","rb"))
            self.params_median  = self._stat_vals["med"]
            self.params_max     = self._stat_vals["max"]
            self.params_bestfit = self._stat_vals["bf"]
            self.params_bfdict  = {k:v for k,v in zip(self.params_names, self.params_median)}

            self.lc             = SimpleNamespace(names    = self._ind_para[31],
                                                  filters  = self._ind_para[12],
                                                  evaluate = self._evaluate_lc,
                                                  outdata  = self.load_result_array(["lc"],verbose=verbose),
                                                  indata   = None)
            self.rv             = SimpleNamespace(names    = self._ind_para[32],
                                                  filters  = self._ind_para[12],
                                                  evaluate = self._evaluate_rv,
                                                  outdata  = self.load_result_array(["rv"],verbose=verbose),
                                                  indata   = None)

        except:
            pass
        
    def __repr__(self):
        return f'Object containing chains (main or burn-in) from mcmc. \
                \nParameters in chain are:\n\t {self.params_names} \
                \n\nuse `plot_chains()`, `plot_burnin_chains()`, `plot_corner()` or `plot_posterior()` methods on selected parameters to visualize results.'
        
    def plot_chains(self, pars=None, figsize = None, thin=1, discard=0, alpha=0.05,
                    color=None, label_size=12, force_plot = False):
        """
            Plot chains of selected parameters.
              
            Parameters:
            ------------
            pars: list of str;
                parameter names to plot. Plot less than 20 parameters at a time for clarity.

            figsize: tuple of length 2;
                Figure size. If None, optimally determined.
        
            thin : int;
                factor by which to thin the chains in order to reduce correlation.

            discard : int;
                to discard first couple of steps within the chains. 

            alpha : float;
                transparency of the lines in the plot.

            color : str;
                color of the lines in the plot.

            label_size : int;
                size of the labels in the plot.

            force_plot : bool;
                if True, plot more than 20 parameters at a time.

            
            
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
            ------------
            pars: list of str;
                parameter names to plot. Plot less than 20 parameters at a time for clarity.
        
            thin : int;
                factor by which to thin the chains in order to reduce correlation.

            discard : int;
                to discard first couple of steps within the chains. 
        
        """
        self._par_names = self._burnin_chains.keys()
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
            ------------
            pars : list of str;
                parameter names to plot. Ideally less than 14 pars for clarity of plot

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

        if not force_plot: assert ndim <= 14, \
            f'number of parameters to plot should be <=14 for clarity. Use force_plot = True to continue anyways.'

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


    def load_result_array(self, data=["lc","rv"],verbose=True):
        """
            Load result array from CONAN3 fit allowing for customised plots.
            All files with '_**out.dat' are loaded. 

            Returns:
            --------
                results : dict;
                    dictionary of holding the arrays for each output file.
                
            Examples
            --------
            >>> import CONAN3
            >>> res=CONAN3.load_result()
            >>> results = res.load_result_array()
            >>> list(results.keys())
            ['lc8det_lcout.dat', 'lc6bjd_lcout.dat']

            >>> df1 = results['lc8det_lcout.dat']
            >>> df1.keys()
            ['time', 'flux', 'error', 'full_mod', 'gp*base', 'transit', 'det_flux']

            >>> #plot arrays
            >>> plt.plot(df["time"], df["flux"],"b.")
            >>> plt.plot(df["time"], df["gp*base"],"r")
            >>> plt.plot(df["time"], df["transit"],"g")
            
        """
        out_files_lc = sorted([ f  for f in os.listdir(self._folder) if '_lcout.dat' in f])
        out_files_rv = sorted([ f  for f in os.listdir(self._folder) if '_rvout.dat' in f])
        all_files    = []
        if "lc" in data: all_files.extend(out_files_lc)
        if "rv" in data: all_files.extend(out_files_rv)
        
        results = {}
        for f in all_files:
            df = pd.read_fwf(self._folder+"/"+f, header=0)
            df = df.rename(columns={'# time': 'time'})
            keyname = f[:-10]+f[-4:]   #remove _rvout or _lcout from filename
            results[keyname] = df
        if verbose: print(f"{data} Output files, {all_files}, loaded into result object")
        return results

    def make_output_file(self, stat="median"):
        """
        make output model file ("*_??out.dat") from parameters obtained using different summary statistic on the posterior.
        if a *_??out.dat file already exists, it is overwritten (so be sure!!!).

        Parameters
        ----------
        stat : str, optional
            posterior summary statistic to use for model calculation, must be one of ["median","max","bestfit"], by default "median".
            "max" and "median" calculate the maximum and median of each parameter posterior respectively while "bestfit" \
            is the parameter combination that gives the maximum joint posterior probability.
        """
        
        from CONAN3.logprob_multi_sin_v4 import logprob_multi
        from CONAN3.plots_v12 import mcmc_plots

        assert stat in ["median","max","bestfit"],f'make_output_file: stat must be of ["median","max","bestfit"] but {stat} given'
        if   stat == "median":  stat = "med"
        elif stat == "bestfit": stat = "bf"

        mval2, merr2, T0_post, p_post = logprob_multi(self._stat_vals[stat],*self._ind_para,make_out_file=True, verbose=True)

        return
        
    def _evaluate_lc(self, file, time=None,params=None, nsamp=5000,return_std=False, return_components=False):
        """
        Compute transit model from CONAN3 fit for a given input file at the given times using specified parameters.

        Parameters:
        -----------
        file : str;
            name of the LC file  for which to evaluate the LC model.

        time : array-like;
            times at which to evaluate the model

        params : array-like;
            parameters to evaluate the model at. The median posterior parameters from the fit are used if params is None

        nsamp : int;
            number of posterior samples to use for computing the 1sigma quantiles of the model. Default is 5000.

        return_std : bool;
            if True, return the 1sigma quantiles of the model as well as the model itself. If False, return only the model.

        return_components : bool;
            if True, return transit model for each planet in the system. If False, return only the total lc model.

        Returns:
        --------
        lcmodel, lc_comps, 1sigma_lo, 1sigma_hi : array, dict, array, array,  respectively;
            Transit model array evaluated at the given times, for a specific file. transit_components are return if return_components is True.
            if return_std is True, 1sigma quantiles (lo and hi) of the model is returned.
        """

        from CONAN3.logprob_multi_sin_v4 import logprob_multi
        
        if params is None: params = self.params_median
        mod  = logprob_multi(params,*self._ind_para,t=time,get_model=True)
        keys = mod.lc.keys() 

        if not return_std:    #return only the model
            return mod.lc[file] if return_components else mod.lc[file][0]
        else:                 #return model and quantiles
            lenpost = len(self.flat_posterior)
            mods    = []  #store model realization for each parameter combination

            for p in self.flat_posterior[np.random.randint(0,lenpost,int(min(nsamp,0.2*lenpost)))]:   #at most 5000 random posterior samples 
                temp = logprob_multi(p,*self._ind_para,t=time,get_model=True)
                mods.append(temp.lc[file][0])

            qs = np.quantile(mods,q=[0.16,0.5,0.84],axis=0) #compute 68% percentiles

            return (mod.lc[file][0],mod.lc[file][1],qs[0],qs[1]) if return_components else (mod.lc[file][0],qs[0],qs[1])
        
    def _evaluate_rv(self, file, time=None,params=None, nsamp=5000,return_std=False, return_components=False):
        """
        Compute RV model from CONAN3 fit for a given input file at the given times using specified parameters.

        Parameters:
        -----------
        file : str;
            name of the RV file for which to evaluate the RVmodel.

        time : array-like;
            times at which to evaluate the model

        params : array-like;
            parameters to evaluate the model at. The median posterior parameters from the fit are used if params is None

        nsamp : int;    
            number of posterior samples to use for computing the 1sigma quantiles of the model. Default is 5000.

        return_std : bool;
            if True, return the 1sigma quantiles of the model as well as the model itself. If False, return only the model.

        return_components : bool;
            if True, return rv model for each planet in the system. If False, return only the total rv model.

        Returns:
        --------
        rvmodel, rv_comps, 1sigma_lo, 1sigma_hi : array, dict, array, array,  respectively;
            RV model array evaluated at the given times, for a specific file. rv_components are return if return_components is True.
            if return_std is True, 1sigma quantiles (lo and hi) of the model is returned.
        """

        from CONAN3.logprob_multi_sin_v4 import logprob_multi

        if params is None: params = self.params_median
        mod  = logprob_multi(params,*self._ind_para,t=time,get_model=True)

        if not return_std:     #return only the model
            return mod.rv[file] if return_components else mod.rv[file][0]
        else:                 #return model and quantiles
            lenpost = len(self.flat_posterior)
            mods    = []

            for p in self.flat_posterior[np.random.randint(0,lenpost,int(min(5000,0.2*lenpost)))]:   #at most 5000 random posterior samples
                temp = logprob_multi(p,*self._ind_para,t=time,get_model=True)
                mods.append(temp.rv[file][0])

            qs = np.quantile(mods,q=[0.16,0.5,0.84],axis=0) #compute 68% percentiles
            
            return (mod.rv[file][0],mod.rv[file][1],qs[0],qs[1]) if return_components else (mod.rv[file][0],qs[0],qs[1])



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

    if ddf != "n":
        lc_data.transit_depth_variation(ddf,depth_per_group,div_white,
                                        step,bounds,prior,pr_width,
                                        verbose)
    else: lc_data.transit_depth_variation(verbose=verbose)

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




def fit_EULER_lc(planet_name, filename, filter):

    get_pars = pd.read.csv("exoplanets.csv")
    get_pars = get_pars[get_pars["pl_name"]==planet_name]
    pl_pars = get_pars["T0","Period","ecc","omega",]