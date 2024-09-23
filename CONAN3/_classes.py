import numpy as np
import matplotlib.pyplot as plt
import corner
import dill as pickle
from types import SimpleNamespace
import os
import matplotlib
import pandas as pd
from lmfit import minimize, Parameters, Parameter
import batman

from os.path import splitext
from ldtk import SVOFilter
from CONAN3.models import RadialVelocity_Model, Transit_Model
from .utils import outlier_clipping, rho_to_aR, Tdur_to_aR, rho_to_tdur, rescale0_1
from .utils import rescale_minus1_1, cosine_atm_variation, split_transits,sinusoid,gp_params_convert 
from .utils import phase_fold, supersampling, convert_LD, get_transit_time, bin_data_with_gaps
from copy import deepcopy
from scipy.interpolate import LSQUnivariateSpline,LSQBivariateSpline
from uncertainties import ufloat
import inspect
import celerite
import warnings

__all__ = ["load_lightcurves", "load_rvs", "fit_setup", "load_result", "__default_backend__"]

#helper functions
__default_backend__ = "Agg" if matplotlib.get_backend()=="TkAgg" else matplotlib.get_backend()
matplotlib.use(__default_backend__)

def _plot_data(obj, plot_cols, col_labels, nrow_ncols=None, figsize=None, fit_order=0, 
                model_overplot=None, detrend=False, hspace=None, wspace=None,binsize=0.0104):
    """
    Takes a data object (containing light-curves or RVs) and plots them.
    """

    tsm= True if plot_cols[0]==0 else False   #if time is being plotted,use regular time grid
    cols = plot_cols+(2,) if len(plot_cols)==2 else plot_cols

    if isinstance(obj, SimpleNamespace): input_data = obj.indata  # if result object
    else: input_data = obj._input_lc if obj._obj_type=="lc_obj" else obj._input_rv   # if lc_obj or rv_obj
        
    fnames = list(input_data.keys())
    n_data = len(fnames)

    if plot_cols[1] == "res":   # if plotting residuals only
        cols = (cols[0],1,2)
        col_labels = (col_labels[0],"residuals")
        
    if nrow_ncols is None: 
        nrow_ncols = (1,1) if n_data==1 else (int(n_data/2), 2) if n_data%2==0 else (int(np.ceil(n_data/3)), 3)
    if figsize is None: figsize=(8,5) if n_data==1 else (14,3.5*nrow_ncols[0])

    fig, ax = plt.subplots(nrow_ncols[0], nrow_ncols[1], figsize=figsize)
    ax = [ax] if n_data==1 else ax.reshape(-1)

    for i, d in enumerate(fnames):
        p1,p2,p3 = [input_data[d][f"col{n}"] for n in cols]   #select columns to plot
        if plot_cols[1] == "res": p2 = model_overplot[i].residual

        if len(plot_cols)==2: p3 = None
        ax[i].set_title(f'{fnames[i]} (detrended)') if (model_overplot and detrend) else ax[i].set_title(f'{fnames[i]}')

        if model_overplot and plot_cols[1] != "res":
            if detrend:     #remove trend model from data
                dt_flux           = (p2/model_overplot[i].tot_trnd_mod) if obj._obj_type=="lc_obj" else (p2-model_overplot[i].tot_trnd_mod)
                planet_mod        = model_overplot[i].planet_mod if obj._obj_type=="lc_obj" else model_overplot[i].planet_mod-model_overplot[i].gamma
                planet_mod_smooth = model_overplot[i].planet_mod_smooth if obj._obj_type=="lc_obj" else model_overplot[i].planet_mod_smooth-model_overplot[i].gamma
                
                if plot_cols[0]==0 and binsize!=0: 
                    ax[i].plot(p1,dt_flux, "C0.", ms=4, alpha=0.6)
                    if p3 is not None: t_bin,y_bin,e_bin = bin_data_with_gaps(p1,dt_flux,p3,binsize=binsize)
                    else: t_bin,y_bin = bin_data_with_gaps(p1,dt_flux,binsize=binsize); e_bin=None
                    ax[i].errorbar(t_bin,y_bin,yerr=e_bin, fmt="o", color='midnightblue', capsize=2, zorder=3)
                else: ax[i].errorbar(p1,dt_flux, p3,fmt=".", ms=6, color="C0", alpha=0.6, capsize=2)

                if tsm: ax[i].plot(model_overplot[i].time_smooth, planet_mod_smooth,"r",zorder=4,label="planet_model")
                else: ax[i].plot(p1,planet_mod,"r",zorder=5,label="planet_model")
            
            else: 
                if plot_cols[0]==0 and binsize!=0: 
                    ax[i].plot(p1,p2,"C0.",ms=4,alpha=0.6)      #data
                    if p3 is not None: t_bin,y_bin,e_bin = bin_data_with_gaps(p1,p2,p3,binsize=binsize)
                    else: t_bin,y_bin = bin_data_with_gaps(p1,p2,binsize=binsize); e_bin=None
                    ax[i].errorbar(t_bin,y_bin,yerr=e_bin, fmt="o", color='midnightblue', capsize=2, zorder=3)
                else: ax[i].errorbar(p1,p2,yerr=p3, fmt=".",ms=6, color="C0", alpha=0.6, capsize=2)
                ax[i].plot(p1,model_overplot[i].tot_trnd_mod,c="darkgoldenrod",zorder=4,label="detrend_model")  #detrend model plot

                if tsm: ax[i].plot(model_overplot[i].time_smooth,model_overplot[i].planet_mod_smooth,"r",zorder=4,label="planet_model")
                else: ax[i].plot(p1,model_overplot[i].planet_mod,"r",zorder=5,label="planet_model")
            
            xmin    = ax[i].get_ylim()[0]
            res_lvl = xmin - max(model_overplot[i].residual) #np.ptp(model_overplot[i].residual)
            ax[i].axhline(res_lvl, color="k", ls="--", alpha=0.2)
            if plot_cols[0]==0 and binsize!=0: 
                ax[i].plot(p1,model_overplot[i].residual+res_lvl,".",ms=3,c="gray",alpha=0.3)
                t_bin,res_bin = bin_data_with_gaps(p1,model_overplot[i].residual,binsize=binsize)
                ax[i].errorbar(t_bin,res_bin+res_lvl, fmt="o",ms=5, color="k", capsize=2, zorder=3)
            else:
                ax[i].plot(p1,model_overplot[i].residual+res_lvl,".",ms=5,c="gray")

            ax[i].text(min(p1), max(model_overplot[i].residual+res_lvl),"residuals",va="bottom")
            ax[i].axhline(max(model_overplot[i].residual+res_lvl), color="k", ls="-",lw=1)
            ax[i].legend(fontsize=10)
        else:
            ax[i].errorbar(p1,p2,yerr=p3, fmt=".", color="C0", ms=5, ecolor="gray")
            

        if fit_order>0:
            pfit = np.polyfit(p1,p2,fit_order)
            srt = np.argsort(p1)
            ax[i].plot(p1[srt],np.polyval(pfit,p1[srt]),"r",zorder=3)
    plt.subplots_adjust(hspace=0.3 if hspace is None else hspace , wspace = wspace if wspace!=None else None)
    for i in range(len(fnames),np.prod(nrow_ncols)): ax[i].axis("off")   #remove unused subplots

    fig.suptitle(f"{col_labels[0]} against {col_labels[1]}", y=0.99, fontsize=18)
    plt.tight_layout()

    plt.show()
    return fig

def _raise(exception_type, msg):
    raise exception_type(msg)

def _decorr(df, T_0=None, Period=None, rho_star=None, Duration=None, Impact_para=0, RpRs=None, 
                Eccentricity=0, omega=90, D_occ=0, Fn=None, ph_off=None, A_ev=0, A_db=0,q1=0, q2=0,
                mask=False, decorr_bound=(-1,1), spline=None,sinus=None,gp=None,ss_exp=None,
                offset=None, A0=None, B0=None, A3=None, B3=None,A4=None, B4=None, 
                A5=None, B5=None,A6=None, B6=None, A7=None, B7=None, A8=None, B8=None,
                sin_Amp=0, sin2_Amp=0, sin3_Amp=0, cos_Amp=0, cos2_Amp=0, cos3_Amp=0, sin_P=0,  sin_x0=0,
                log_GP_amp1=0, log_GP_amp2=0, log_GP_len1=0, log_GP_len2=0,
                npl=1,jitter=0,Rstar=None,custom_LCfunc=None,return_models=False):
    """
    linear decorrelation with different columns of data file. It performs a linear model fit to the columns of the file.
    It uses columns 0,3,4,5,6,7,8 to construct the linear trend model. A spline can also be included to decorrelate against any column.
    
    Parameters:
    -----------
    df : dataframe/dict;
        data file with columns 0 to 8 (col0-col8).

    T_0, Period, rho_star, D_occ, Impact_para, RpRs, Eccentricity, omega,Fn,ph_off,A_ev,A_db : floats, None;
        transit/eclipse parameters of the planet. T_0 and P must be in same units as the time axis (cols0) in the data file. rho_star is the stellar density in g/cm^3.
        if float/int, the values are held fixed. if tuple/list of len 2 implies [min,max] while len 3 implies [min,start_val,max].
        
    q1,q2 : float  (optional);
        Kipping quadratic limb darkening parameters.

    mask : bool ;
        if True, transits and eclipses are masked using T_0, P and rho_star which must be float/int.                    
        
    offset, Ai, Bi; floats [-1,1] or None;
        coefficients of linear model where offset is the intercept. they have large bounds [-1,1].
        Ai, Bi are the linear and quadratic term of the model against column i. A0*col0 + A0*col0**2 for time trend
    
    spline : dict;
        spline configuration to use in decorrelation an axis of the data. Default is None which implies no spline is used.
        the config is given as a dict with keys "col", "knot_spacing", "degree" specifying the column, the knot spacing and the degree of the spline.
        e.g. spline = {"col":0, "knot_spacing":0.1, "degree":3} will fit a spline the flux as a function of column 0 with knot spacing of 0.1 and degree 3.

    sinus : dict;
        sinusoid configuration to use in decorrelation. Default is None which implies no sinusoid is used.
    npl : int; 
        number of planets in the system. default is 1.

    jitter : float;
        jitter value to quadratically add to the errorbars of the data.

    Rstar : float;
        stellar radius in solar radii. Required for light travel time correction. Default is None

    custom_LCsignal : array, None;
        custom light curve signal to add to the planet model. Default is None.
        
    return_models : Bool;
        True to return trend model and transit/eclipse model.

    Returns:
    -------
    result: object;
        result object from fit with several attributes such as result.bestfit, result.params, result.bic, ...
        if return_models = True, returns (trend_model, transit/eclipse model)
    """
    DA = locals().copy()
    if Fn!=None and ph_off!=None: model_phasevar=True
    else:                         model_phasevar=False
    
    assert isinstance(spline, dict) or spline is None, "spline must be a dictionary"
    flux_err = (np.array(df["col2"])**2 + jitter**2)**0.5

    #transit variables
    pl_vars = ["T_0", "Period", "rho_star" if rho_star!=None else "Duration", "D_occ", "Impact_para","RpRs", "Eccentricity", "omega", "Fn", "ph_off", "A_ev","A_db","q1","q2"]
    tr_pars = {}
    for p in pl_vars:
        for n in range(npl):
            lbl = f"_{n+1}" if npl>1 else ""                      # numbering to add to parameter names of each planet
            if p not in ["q1","q2","rho_star","Duration","Fn","ph_off","A_ev","A_db","D_occ"]:   # parameters common to all planets or not used in multi-planet fit
                tr_pars[p+lbl]= DA[p][n]  #transit/eclipse pars
            else:
                tr_pars[p] = DA[p]   #common parameters

    # add custom function parameters to transit parameters
    if custom_LCfunc is not None: 
        for p in custom_LCfunc.func_args.keys():
            tr_pars[p] = custom_LCfunc.func_args[p] 

    df       = pd.DataFrame(df)  #pandas dataframe
    col0_med = np.median(df["col0"])

    if mask:
        assert npl==1, "masking transits/eclipses only works for single planet systems"
        print("masking transit/eclipse phases")
        for tp in pl_vars:
            if isinstance(tr_pars[tp], tuple):
                if len(tr_pars[tp])==2:   tr_pars[tp]=tr_pars[tp][0]
                elif len(tr_pars[tp])==3: tr_pars[tp]=tr_pars[tp][1]
        #use periodicity of 0.5*P to catch both transits and eclipses. this does not work for eccentric orbits
        E = np.round(( col0_med - tr_pars['T_0'])/(0.5*tr_pars["Period"]) )
        Tc = E*(0.5*tr_pars["Period"]) + tr_pars['T_0']
        duration = rho_to_tdur(tr_pars["rho_star"], tr_pars["Impact_para"], tr_pars["RpRs"],tr_pars["Period"], tr_pars["Eccentricity"], tr_pars["omega"])
        phase = phase_fold(df["col0"], 0.5*tr_pars["Period"], Tc,-0.25)
        mask = abs(phase) > 0.5*duration/(0.5*tr_pars["Period"])
        df = df[mask]


    #PARAMETRIC DECORRELATION PARAMETERS
    #decorr variables    
    decorr_vars = [f"{L}{i}" for i in [0,3,4,5,6,7,8] for L in ["A","B"]]  + ["offset"] # creates list: ['A0','B0','A3','B3','A4','B4','A5','B5','A6','B6','A7','B7','A8','B8','offset']
    in_pars     = {k:v for k,v in DA.items() if k in decorr_vars}

    params = Parameters()
    for key in in_pars.keys():
        val  = in_pars[key] if in_pars[key] != None else 0    #val is set to 0 or the value of the parameter
        vary = False if in_pars[key] is None else True
        if key=="offset":      #set lims at min, max of flux 
            params.add(key, value=val, min=df["col1"].min()-1, max=df["col1"].max()-1, vary=vary)
        else: params.add(key, value=val, min=decorr_bound[0], max=decorr_bound[1], vary=vary)
    
    #SINUSOIDAL DECORRELATION PARAMS
    if sinus is not None:
        sin_decorr_vars = ['sin_Amp','sin2_Amp','sin3_Amp','cos_Amp','cos2_Amp','cos3_Amp','sin_P', 'sin_x0']
        sin_pars        = {k:v  for k,v in DA.items() if k in sin_decorr_vars}   #input values/priors for sinusoid parameters

        sin_params = Parameters()
        for key in sin_pars.keys():
            if isinstance(sin_pars[key], (float,int)):
                sin_params.add(key, value=sin_pars[key], vary=False)
            if isinstance(sin_pars[key], tuple):
                assert len(sin_pars[key]) in [2,3,4],f"{key} must be float/int or tuple of length 2/3"
                if len(sin_pars[key])==2:
                    sin_params[key] = Parameter(key, value=sin_pars[key][0], vary=True, user_data = sin_pars[key] )
                if len(sin_pars[key])==3:
                    sin_params.add(key, value=sin_pars[key][1], min=sin_pars[key][0], max=sin_pars[key][2], vary=True)
                if len(sin_pars[key])==4:
                    sin_params[key] = Parameter(key, value=sin_pars[key][2], vary=True, min=sin_pars[key][0],max=sin_pars[key][1], user_data = sin_pars[key][-2:] )

        params = params+sin_params 

    #GP DECORRELATION PARAMETERS
    if gp is not None:
        gp_decorr_vars = list(gp.params.keys())
        gp_pars        = {k:v  for k,v in DA.items() if k in gp_decorr_vars}   #input values/priors for GP parameters

        gp_params = Parameters()
        for key in gp_pars.keys():
            if isinstance(gp_pars[key], (float,int)):
                gp_params.add(key, value=gp_pars[key], vary=False)
            if isinstance(gp_pars[key], tuple):
                assert len(gp_pars[key]) in [2,3,4],f"{key} must be float/int or tuple of length 2/3"
                if len(gp_pars[key])==2:
                    gp_params[key] = Parameter(key, value=gp_pars[key][0], vary=True, user_data = gp_pars[key] )
                if len(gp_pars[key])==3:
                    gp_params.add(key, value=gp_pars[key][1], min=gp_pars[key][0], max=gp_pars[key][2], vary=True)
                if len(gp_pars[key])==4:
                    gp_params[key] = Parameter(key, value=gp_pars[key][2], vary=True, min=gp_pars[key][0],max=gp_pars[key][1], user_data = gp_pars[key][-2:] )
    
        params = params+gp_params

    #transit/eclipseparameters
    tr_params = Parameters()
    for key in tr_pars.keys():
        if isinstance(tr_pars[key], (list,tuple)):
            assert len(tr_pars[key]) in [2,3,4],f"{key} must be float/int or tuple of length 2/3"
            if len(tr_pars[key])==3:  #uniform prior (min, start, max)
                val = tr_pars[key]
                tr_params.add(key, value=val[1], min=val[0], max=val[2], vary=True)
            if len(tr_pars[key])==2: #normal prior (mean, std)  #TODO min/max of normal prior of each transit par should also be added min=min_max[key][0], max=min_max[key][1]
                tr_params[key] = Parameter(key, value=tr_pars[key][0], vary=True, user_data = tr_pars[key] )
            if len(tr_pars[key])==4: #trunc normal prior (min,max, mean, std) 
                tr_params[key] = Parameter(key, value=tr_pars[key][2], vary=True, min=tr_pars[key][0],max=tr_pars[key][1], user_data = tr_pars[key][-2:] )
        if isinstance(tr_pars[key], (float,int)):
            tr_params.add(key, value=tr_pars[key], vary=False)
        if tr_pars[key] == None:
            vs = ["RpRs","Period","rho_star"]
            vs = [v+(f"_{n+1}" if npl>1 else "") for n in range(npl) for v in vs]    
            val = 1e-10 if key in vs else 0 #allows to obtain transit/eclipse with zero depth
            tr_params.add(key, value=val, vary=False)

    
    #transit model based on TransitModel       
    def transit_occ_model(tr_params,t=None,ss_exp=ss_exp,Rstar=None,custom_LCfunc=custom_LCfunc,npl=1):
        if t is None: t = df["col0"].values
        ss = supersampling(ss_exp/(60*24),int(ss_exp)) if ss_exp is not None else None
        pl_ind = [(f"_{n}" if npl>1 else "") for n in range(1,npl+1)]

        per = [tr_params["Period"+lbl].value for lbl in pl_ind]
        t0  = [tr_params["T_0"+lbl].value for lbl in pl_ind]
        rp  = [tr_params["RpRs"+lbl].value for lbl in pl_ind]
        b   = [tr_params["Impact_para"+lbl].value for lbl in pl_ind]
        ecc = [tr_params["Eccentricity"+lbl].value for lbl in pl_ind]
        w   = [tr_params["omega"+lbl].value for lbl in pl_ind]
        sesinw = [np.sqrt(e)*np.sin(np.deg2rad(om)) for e,om in zip(ecc,w)]
        secosw = [np.sqrt(e)*np.cos(np.deg2rad(om)) for e,om in zip(ecc,w)]

        rho_star = tr_params["rho_star"].value if "rho_star" in tr_params.keys() else None
        dur      = tr_params["Duration"].value if "Duration" in tr_params.keys() else None 

        cst_pars = {p:tr_params[p].value for p in custom_LCfunc.func_args.keys()} if custom_LCfunc is not None else {}

        TM  = Transit_Model(rho_star, dur, t0, rp, b, per, sesinw, secosw,ddf=0,q1=tr_params["q1"].value,q2=tr_params["q2"].value,occ=tr_params["D_occ"].value,
                            Fn=tr_params["Fn"].value,delta=tr_params["ph_off"].value,A_ev=tr_params["A_ev"].value,A_db=tr_params["A_db"].value,cst_pars=cst_pars,npl=npl)
        model_flux,_ = TM.get_value(t,ss=ss,Rstar=Rstar, model_phasevar=model_phasevar,custom_LCfunc=custom_LCfunc)

        return model_flux


    def trend_model(params):    #parametric + sinusoid
        trend = 1 + params["offset"]       #offset
        trend += params["A0"]*(df["col0"]-col0_med)  + params["B0"]*(df["col0"]-col0_med)**2 #time trend
        trend += params["A3"]*df["col3"]  + params["B3"]*df["col3"]**2 #x
        trend += params["A4"]*df["col4"]  + params["B4"]*df["col4"]**2 #y
        trend += params["A5"]*df["col5"]  + params["B5"]*df["col5"]**2
        trend += params["A6"]*df["col6"]  + params["B6"]*df["col6"]**2 #bg
        trend += params["A7"]*df["col7"]  + params["B7"]*df["col7"]**2 #conta
        trend += params["A8"]*df["col8"]  + params["B8"]*df["col8"]**2
        if sinus is not None:
            amps = [params[k] for k in sin_pars.keys() if "Amp" in k]
            sin_model = sinusoid(df[sinus.par],A=amps,x0=params['sin_x0'],P=params['sin_P'],n=3, trig="sincos")
            trend += sin_model
        return np.array(trend)
    
    def gp_model(params,resid,ll=False):
        gp_x     = df[gp.column[0]]
        srt_gp   = np.argsort(gp_x)
        unsrt_gp = np.argsort(srt_gp)  #indices to unsort the gp axis
        
        gppars  = [params[p].value for p in gp_decorr_vars]
        gp_conv = gp_params_convert()   #class containing functions to convert gp amplitude and lengthscale to the required values for the different kernels 
        gppars  = gp_conv.get_values(kernels=gp.kern, data="lc", pars=np.exp(gppars))
        gp.GPobj.set_parameter_vector(gppars)
        gp.GPobj.compute(gp_x[srt_gp], yerr=flux_err[srt_gp])
        return gp.GPobj.log_likelihood(resid[srt_gp],quiet=True) if ll==True else gp.GPobj.predict(resid[srt_gp],t=gp_x[srt_gp],return_cov=False,return_var=False)[unsrt_gp]


    if spline is not None:
        spl_col = spline["col"]
        spl_kn  = spline["knot_spacing"]
        spl_deg = spline["degree"]

        if not isinstance(spline["col"], tuple):  #1D spline
            assert spl_col in [0,3,4,5,6,7,8], f'_decorr(): spline["col"] must be one of [0,3,4,5,6,7,8]'

            spl_x = df["col"+str(spl_col)]
            srt   = np.argsort(spl_x)
            if spl_kn=='r': spl_kn = np.ptp(spl_x) #range of the array
            knots = np.arange(min(spl_x)+spl_kn,max(spl_x),spl_kn)
        else:
            assert spl_col[0] in [0,3,4,5,6,7,8], f'_decorr(): spline["col"] must be one of [0,3,4,5,6,7,8] but {spl_col[0]} given'
            assert spl_col[1] in [0,3,4,5,6,7,8], f'_decorr(): spline["col"] must be one of [0,3,4,5,6,7,8] but {spl_col[1]} given'

            spl_x  = (df["col"+str(spl_col[0])], df["col"+str(spl_col[1])])
            for ii in range(2):
                spl_kn = list(spl_kn)
                if spl_kn[ii]=='r': spl_kn[ii] = np.ptp(spl_x[ii]) #range of the array of each column
            knots1 = np.arange(min(spl_x[0])+spl_kn[0],max(spl_x[0]),spl_kn[0])
            knots2 = np.arange(min(spl_x[1])+spl_kn[1],max(spl_x[1]),spl_kn[1])

    if return_models:
        tra_occ_mod = transit_occ_model(tr_params,npl=npl)
        trnd_mod    = trend_model(params)
        fl_mod      = tra_occ_mod*trnd_mod

        if spline is not None:
            if not isinstance(spline["col"], tuple):  #1D spline
                splfunc    = LSQUnivariateSpline(spl_x[srt],(df["col1"]-fl_mod)[srt],t=knots,k=spl_deg)
                spl_mod    = splfunc(spl_x)
            else:  #2D
                splfunc = LSQBivariateSpline(spl_x[0], spl_x[1], df["col1"]-fl_mod,
                                                knots1, knots2, kx=spl_deg[0], ky=spl_deg[1])
                spl_mod = splfunc(spl_x[0],spl_x[1],grid=False)
        else: spl_mod  = 0

        if gp is not None:
            gp_mod = gp_model(params,df["col1"]-fl_mod-spl_mod)
        else: gp_mod = 0
        
        tsm = np.linspace(min(df["col0"]),max(df["col0"]),len(df["col0"])*3)
        mods = SimpleNamespace(tot_trnd_mod       = trnd_mod+spl_mod+gp_mod, 
                                planet_mod        = tra_occ_mod, 
                                time_smooth       = tsm, 
                                planet_mod_smooth = transit_occ_model(tr_params,tsm,npl=npl), 
                                residual          = df["col1"] - fl_mod - spl_mod - gp_mod
                                ) 
        return mods
    
    #perform fitting 
    def chisqr(fit_params):
        flux_model = trend_model(fit_params)*transit_occ_model(fit_params,npl=npl)
        resid = df["col1"] - flux_model
        if spline is not None:
            if not isinstance(spline["col"], tuple):  #1D spline
                splfunc = LSQUnivariateSpline(spl_x[srt],resid[srt],t=knots,k=spl_deg)
                spl     = splfunc(spl_x)
            else:  #2D spline
                splfunc = LSQBivariateSpline(spl_x[0], spl_x[1], resid, knots1, knots2, kx=spl_deg[0], ky=spl_deg[1])
                spl     = splfunc(spl_x[0],spl_x[1],grid=False)
        else:
            spl = 0
        resid = resid - spl

        if gp is not None:
            gp_ll = gp_model(fit_params,resid,True)
            res   = -2*gp_ll - np.sum(np.log(2*np.pi*flux_err**2)) #gp chi2 from LL
        else:
            res = resid/flux_err

        for p in fit_params:
            u = fit_params[p].user_data  #obtain tuple specifying the normal prior if defined
            if u:  #modify residual to account for how far the value is from mean of prior
                res_mod = (u[0]-fit_params[p].value)/u[1]
                res = np.append(res, res_mod ) if gp==None else res+res_mod**2
            # print(f"chi-square:{np.sum(res**2)}")
        return res
    
    fit_params = params+tr_params
    out = minimize(chisqr, fit_params, nan_policy='propagate', method='lbfgsb' if gp!=None else 'leastsq')  #lbfgsb does not give uncertainties on the parameters
    
    #modify output object
    tra_occ_mod = transit_occ_model(out.params,npl=npl)
    trnd_mod    = trend_model(out.params)
    if spline is None: spl_mod = 0
    else: 
        if not isinstance(spline["col"], tuple): #1D spline
            spl_mod = LSQUnivariateSpline(spl_x[srt],(df["col1"]-tra_occ_mod*trnd_mod)[srt],t=knots,k=spl_deg)(spl_x)
        else: #2D spline
            spl_mod = LSQBivariateSpline(spl_x[0], spl_x[1], df["col1"]-tra_occ_mod*trnd_mod, knots1, knots2, kx=spl_deg[0], ky=spl_deg[1])(spl_x[0],spl_x[1],grid=False)
    if gp is not None: gp_mod = gp_model(out.params,df["col1"]-tra_occ_mod*trnd_mod-spl_mod)
    else: gp_mod = 0
    out.bestfit    = tra_occ_mod*trnd_mod + spl_mod + gp_mod
    out.poly_trend = trnd_mod   
    out.trend      = trnd_mod+spl_mod+gp_mod
    out.transit    = tra_occ_mod
    out.spl_mod    = spl_mod
    out.spl_x      = 0 if spline is None else spl_x
    
    out.time       = np.array(df["col0"])
    out.flux       = np.array(df["col1"])
    out.flux_err   = flux_err
    out.data       = df

    out.rms        = np.std(out.flux - out.bestfit)
    out.ndata      = len(out.time)
    if gp!=None: out.residual = (out.flux - out.bestfit)/flux_err
    else:        out.residual   = out.residual[:out.ndata]    #note that residual = (y-mod)/err
    out.nfree      = out.ndata - out.nvarys
    out.chisqr     = np.sum(out.residual**2)
    out.redchi     = out.chisqr/out.nfree
    out.lnlike     = -0.5*np.sum(out.residual**2 + np.log(2*np.pi*out.flux_err**2))
    out.bic        = out.chisqr + out.nvarys*np.log(out.ndata)

    return out


def _decorr_RV(df, T_0=None, Period=None, K=None, sesinw=0, secosw=0, gamma=None, decorr_bound=(-1000,1000),
                A0=None, B0=None, A3=None, B3=None, A4=None, B4=None, A5=None, B5=None, npl=1,jitter=0,return_models=False):
    """
    linear decorrelation with different columns of data file. It performs a linear model fit to the 3rd column of the file.
    It uses columns 0,3,4,5 to construct the linear trend model.
    
    Parameters:
    -----------
    df : dataframe/dict;
        data file with columns 0 to 5 (col0-col5).
    T_0, Period, K, Eccentricity, omega : floats, None;
        RV parameters of the planet. T_0 and P must be in same units as the time axis (cols0) in the data file.
        if float/int, the values are held fixed. if tuple/list of len 2 implies [min,max] while len 3 implies [min,start_val,max].
    offset, Ai, Bi; floats [-1,1] or None;
        coefficients of linear model where offset is the systemic velocity. They have large bounds [-1000,1000].
        Ai, Bi are the linear and quadratic term of the model against column i. A0*col0 + A0*col0**2 for time trend
    npl : int; 
        number of planets in the system. default is 1.
    jitter : float;
        jitter value to quadratically add to the errorbars of the data.  
    return_models : Bool;
        True to return trend model and transit/eclipse model.
    Returns:
    -------
    result: object;
        result object from fit with several attributes such as result.bestfit, result.params, result.bic, ...
        if return_models = True, returns (trend_model, transit/eclipse model)
    """
    DA      = locals().copy()
    rv_pars = {}

    df       = pd.DataFrame(df)      #pandas dataframe
    col0_med = np.median(df["col0"])

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
        if t is None: t = df["col0"].values
        rvmod = np.zeros_like(t)

        for n in range(1,npl+1):
            lbl = f"_{n}" if npl>1 else ""

            per    = [rv_params["Period"+lbl]]
            t0     = [rv_params["T_0"+lbl]]
            K      = [rv_params["K"+lbl]]
            sesinw = [rv_params["sesinw"+lbl]]
            secosw = [rv_params["secosw"+lbl]]
            mod,_  = RadialVelocity_Model(t, t0, per, K, sesinw, secosw)  
            rvmod += mod
        
        return rvmod + rv_params["gamma"]

    def trend_model(params):
        trend  = params["A0"]*(df["col0"]-col0_med)  + params["B0"]*(df["col0"]-col0_med)**2 #time trend
        trend += params["A3"]*df["col3"]  + params["B3"]*df["col3"]**2 #bisector
        trend += params["A4"]*df["col4"]  + params["B4"]*df["col4"]**2 #fwhm
        trend += params["A5"]*df["col5"]  + params["B5"]*df["col5"]**2 #contrast
        return np.array(trend)
    

    if return_models:
        tsm = np.linspace(min(df["col0"]),max(df["col0"]),max(500,len(df["col0"])*3))
        mods = SimpleNamespace(tot_trnd_mod = trend_model(params)+rv_params["gamma"],
                                gamma       = rv_params["gamma"], 
                                planet_mod  = rv_model(rv_params,npl=npl), 
                                time_smooth = tsm, 
                                planet_mod_smooth = rv_model(rv_params,tsm,npl=npl), 
                                residual    = df["col1"] - trend_model(params) - rv_model(rv_params,npl=npl)
                                )
        return mods
        
    #perform fitting 
    def chisqr(fit_params):
        rvmod = trend_model(fit_params)+rv_model(fit_params,npl=npl)
        res = (df["col1"] - rvmod)/(df["col2"]**2 + jitter**2)**0.5
        for p in fit_params:
            u = fit_params[p].user_data  #obtain tuple specifying the normal prior if defined
            if u:  #modify residual to account for how far the value is from mean of prior
                res = np.append(res, (u[0]-fit_params[p].value)/u[1] )
        return res
    
    fit_params = params+rv_params
    out = minimize(chisqr, fit_params, nan_policy='propagate')
    
    #modify output object
    out.bestfit = trend_model(out.params)+rv_model(out.params,npl=npl)
    out.trend   = out.poly_trend = trend_model(out.params)+out.params["gamma"]
    out.rvmodel = rv_model(out.params,npl=npl)
    out.time    = np.array(df["col0"])
    out.rv      = np.array(df["col1"])
    out.rv_err  = (np.array(df["col2"])**2 + jitter**2)**0.5
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
    
    prior_str = lambda v: 'None' if v==None else f'F({v})' if isinstance(v,(int,float)) else f"N({v[0]},{v[1]})" if len(v)==2 else f"U({v[0]},{v[1]},{v[2]})" if len(v)==3 else f"TN({v[0]},{v[1]},{v[2]},{v[3]})"

    lc_possible_sections = ["lc_baseline", "sinusoid", "gp", "planet_parameters", "custom_LCfunction","depth_variation","timing_variation",
                            "phasecurve", "limb_darkening", "contamination","sinusoid"]
    rv_possible_sections = ["rv_baseline", "rv_gp", "custom_RVfunction"]
    fit_possible_sections = ["fit", "stellar_pars"]
    spacing = "" if file is None else "\t"

    if self._obj_type == "lc_obj":
        assert section in lc_possible_sections, f"{section} not a valid section of `lc_obj`. Section must be one of {lc_possible_sections}."
        max_name_len = max([len(n) for n in self._names]+[len("name")])      #max length of lc filename
        max_filt_len = max([len(n) for n in self._filters]+[len("filt")])  #max length of lc filter name
    if self._obj_type == "rv_obj":
        assert section in rv_possible_sections, f"{section} not a valid section of `rv_obj`. Section must be one of {rv_possible_sections}."
        max_name_len = max([len(n) for n in self._names]+[len("name")])      #max length of lc filename
    if self._obj_type == "fit_obj":
        assert section in fit_possible_sections, f"{section} not a valid section of `fit_obj`. Section must be one of {fit_possible_sections}."

    if section == "lc_baseline":
        _print_lc_baseline = """# ============ Input lightcurves, filters baseline function =======================================================""" +\
                            f"""\n{spacing}{"name":{max_name_len}s} {"filt":{max_filt_len}s} {"𝜆_𝜇m":5s}|{"s_samp ":7s} {"clip_outliers":13s} {"scl_col":8s}|{"col0":4s} {"col3":4s} {"col4":4s} {"col5":4s} {"col6":4s} {"col7":4s} {"col8":4s}|{"sin":3s} {"id":2s} {"GP":2s} {"spline_config  ":15s}"""
        #define print out format
        txtfmt = f"\n{spacing}{{0:{max_name_len}s}} {{1:{max_filt_len}s}}"+" {2:5s}|{3:7s} {4:13s} {5:8s}|{6:4d} {7:4d} {8:4d} {9:4d} {10:4d} {11:4d} {12:4d}|{13:3s} {14:2d} {15:2s} {16:15s}"        
        for i in range(len(self._names)):
            t = txtfmt.format(self._names[i], self._filters[i], str(self._wl[i]), self._ss[i].config,self._clipped_data.config[i], self._rescaled_data.config[i],
                              *self._bases[i][:-1], self._groups[i], self._useGPphot[i],self._lcspline[i].conf, 
                                )
            _print_lc_baseline += t
        print(_print_lc_baseline, file=file)

    if section == "sinusoid":
        DA = self._sine_dict
        _print_sinusoid = f"""# ============ Sinusoidal signals: Amp*trig(2𝜋/P*(x-x0)) - trig=sin or cos or both added==========================""" +\
                            f"""\n{spacing}{"name/filt":{max_name_len}s} {"trig":7s}  {"n_harmonics":11s}  {"x":4s}  {"Amp[ppm]":18s}  {"P":18s}  {"x0":18s}"""
        #define print out format
        max_namefilt_len = max([len(n) for n in self._names+self._filters]+[9])      #max length of lcname/filtname
        txtfmt = f"\n{spacing}{{0:{max_namefilt_len}s}}"+" {1:7s}  {2:11d}  {3:4s}  {4:18s}  {5:18s}  {6:18s}"

        for k,v in DA.items():
            if v.trig is not None:
                amp_pri = f"F({v.Amp.user_input})" if isinstance(v.Amp.user_input, (float,int)) else f"N({v.Amp.user_input[0]},{v.Amp.user_input[1]})" if len(v.Amp.user_input)==2 else f"U({v.Amp.user_input[0]},{v.Amp.user_input[1]},{v.Amp.user_input[2]})"
                P_pri   = "None" if v.P.user_input==None else f"F({v.P.user_input})" if isinstance(v.P.user_input, (float,int)) else f"N({v.P.user_input[0]},{v.P.user_input[1]})" if len(v.P.user_input)==2 else f"U({v.P.user_input[0]},{v.P.user_input[1]},{v.P.user_input[2]})"
                x0_pri  = "None" if v.x0.user_input==None else f"F({v.x0.user_input})" if isinstance(v.x0.user_input, (float,int)) else f"N({v.x0.user_input[0]},{v.x0.user_input[1]})" if len(v.x0.user_input)==2 else f"U({v.x0.user_input[0]},{v.x0.user_input[1]},{v.x0.user_input[2]})"
                t = txtfmt.format(v.name, v.trig, v.n, v.par, amp_pri, P_pri, x0_pri)
                _print_sinusoid += t
        print(_print_sinusoid, file=file)


    if section == "gp":
        DA = self._GP_dict
        _print_gp = f"""# ============ Photometry GP properties ==========================================================================="""
        # _print_gp += f"""\nsame_GP: {self._sameLCgp.flag}"""
        _print_gp += f"""\n{spacing}{"name":{max_name_len}s} {'par1':4s} {"kern1":5s} {'Amplitude1_ppm':18s} {'length_scale':17s} |{'op':2s}| {'par2':4s} {"kern2":5s} {'Amplitude2_ppm':18s} {'length_scale2':17s}"""
        if DA != {}: 
            #define gp print out format
            txtfmt = f"\n{spacing}{{0:{max_name_len}s}}"+" {1:4s} {2:5s} {3:18s} {4:17s} |{5:2s}| {6:4s} {7:5s} {8:18s} {9:17s} "        

            for lc in DA.keys():
                ngp = DA[lc]["ngp"]
                prior={}
                for p in ["amplitude", "lengthscale"]:
                    for j in range(ngp):
                        if DA[lc][p+str(j)].to_fit == "n":
                            prior[p+str(j)] = f"F({DA[lc][p+str(j)].start_value})"
                        elif DA[lc][p+str(j)].to_fit == "y" and DA[lc][p+str(j)].prior == "n":
                            b_lo = 0 if DA[lc][p+str(j)].bounds_lo==1e-20 else DA[lc][p+str(j)].bounds_lo
                            prior[p+str(j)] = f"LU({b_lo},{DA[lc][p+str(j)].start_value},{DA[lc][p+str(j)].bounds_hi})"
                        elif DA[lc][p+str(j)].to_fit == "y" and DA[lc][p+str(j)].prior == "p":
                            prior[p+str(j)] = f"N({DA[lc][p+str(j)].prior_mean},{DA[lc][p+str(j)].prior_width_lo})"

                if ngp == 2:
                    t = txtfmt.format('same' if self._sameLCgp.flag else lc,DA[lc]["amplitude0"].user_data[1], DA[lc]["amplitude0"].user_data[0],  
                                                    prior["amplitude0"], prior["lengthscale0"], DA[lc]["op"], 
                                        DA[lc]["amplitude1"].user_data[1], DA[lc]["amplitude1"].user_data[0],
                                                    prior["amplitude1"], prior["lengthscale1"])
                else:
                    t = txtfmt.format('same' if self._sameLCgp.flag else lc,DA[lc]["amplitude0"].user_data[1], DA[lc]["amplitude0"].user_data[0],  
                                                    prior["amplitude0"], prior["lengthscale0"], "--", 
                                        "None", "None", "None", "None")
                _print_gp += t
                if self._sameLCgp.flag:      #dont print the other GPs if same_GP is True
                    break
        print(_print_gp, file=file)

    if section == "planet_parameters":
        DA = self._config_par
        notes = dict(RpRs="#range[-0.5,0.5]",Impact_para="#range[0,2]",K="#unit(same as RVdata)",T_0="#unit(days)",Period="#range[0,inf]days",
                        Eccentricity="#range[0,1]",omega="#range[0,360]deg")
        _print_planet_parameters = f"""# ============ Planet parameters (Transit and RV) setup ========================================================== """+\
                                    f"""\n{spacing}{'name':20s}\t{'fit':3s} \t{'prior':35s}\tnote"""
        #define print out format
        txtfmt = f"\n{spacing}"+"{0:20s}\t{1:3s} \t{2:35s}\t{3}"
        #print line for stellar density or duration
        p    = "rho_star" if "rho_star" in DA[f'pl{1}'].keys() else "Duration"
        popt = "[rho_star]/Duration" if "rho_star" in DA[f'pl{1}'].keys() else "rho_star/[Duration]"
        pri_par = f"N({DA[f'pl{1}'][p].prior_mean},{DA[f'pl{1}'][p].prior_width_lo})" if DA[f'pl{1}'][p].prior == "p" else f"LU({DA[f'pl{1}'][p].bounds_lo},{DA[f'pl{1}'][p].start_value},{DA[f'pl{1}'][p].bounds_hi})" if DA[f'pl{1}'][p].bounds_hi else f"F({DA[f'pl{1}'][p].start_value})"
        _print_planet_parameters +=  txtfmt.format( popt, DA[f'pl{1}'][p].to_fit, pri_par, "#choice in []|unit(gcm^-3/days)")
        _print_planet_parameters +=  f"\n{spacing}--------repeat this line & params below for multisystem, adding planet number to the names e.g RpRs_1 for planet 1, ..."
        #then cycle through parameters for each planet       
        for n in range(1,self._nplanet+1):        
            for i,p in enumerate(self._TR_RV_parnames):
                if p not in ["rho_star","Duration"]:
                    pri_par = f"N({DA[f'pl{n}'][p].prior_mean},{DA[f'pl{n}'][p].prior_width_lo})" if DA[f'pl{n}'][p].prior == "p" else f"U({DA[f'pl{n}'][p].bounds_lo},{DA[f'pl{n}'][p].start_value},{DA[f'pl{n}'][p].bounds_hi})" if DA[f'pl{n}'][p].bounds_hi else f"F({DA[f'pl{n}'][p].start_value})"
                    t = txtfmt.format(  p+(f"_{n}" if self._nplanet>1 else ""), DA[f'pl{n}'][p].to_fit, pri_par, notes[p])
                    _print_planet_parameters += t
            if n!=self._nplanet: _print_planet_parameters += f"\n{spacing}------------"
        print(_print_planet_parameters, file=file)

    if section == "custom_LCfunction":
        DA = self._custom_LCfunc
        flag = False if DA.func is None else True
        _print_custom_function = f"""#============ Custom LC function (read from custom_LCfunc.py file)================================================"""
        #define print out format
        txtfmt = f"\n{spacing}{{0:16s}}: {{1:40s}}\t{{2}}"
        _print_custom_function += txtfmt.format("function", DA.func.__code__.co_name if flag else 'None', "#custom function name to combine with LCmodel")
        _print_custom_function += txtfmt.format("x",DA.x if flag else 'None',"#independent variable [time, phase_angle]")
        if flag:
            fa      = DA.func_args
            fa_str  = []
            for k in fa.keys():
                if isinstance(fa[k],(int,float)):
                    fa_str.append(f'{k}:F({fa[k]})')
                if isinstance(fa[k],tuple):
                    fa_str.append(f"{k}:{'U' if len(fa[k])==3 else 'N' if len(fa[k])==2 else 'TN'}{str(fa[k]).replace(' ','')}" )
            fa_str  = ",".join(fa_str)
        else: fa_str = 'None'
        _print_custom_function += txtfmt.format("func_pars",fa_str,"#param names&priors e.g. A:U(0,1,2),P:N(2,1)")
        exa_str = [f"{k}:{v}" for k,v in DA.extra_args.items()]
        exa_str = ",".join(exa_str) if exa_str!=[] else 'None'
        _print_custom_function += txtfmt.format("extra_args",exa_str,"#extra args to func as a dict e.g ld_law:quad")
        _print_custom_function += txtfmt.format("op_func",'None' if (DA.replace_LCmodel or not flag) else DA.op_func.__code__.co_name ,"#function to combine the LC and custom models")
        _print_custom_function += txtfmt.format("replace_LCmodel",str(DA.replace_LCmodel) if flag else 'False',"#if the custom function replaces the LC model")
        print(_print_custom_function, file=file)

    if section == "custom_RVfunction":
        DA = self._custom_RVfunc
        flag = False if DA.func is None else True
        _print_custom_function = f"""#============ Custom RV function (read from custom_RVfunc.py file)================================================"""
        #define print out format
        txtfmt = f"\n{spacing}{{0:16s}}: {{1:40s}}\t{{2}}"
        _print_custom_function += txtfmt.format("function", DA.func.__code__.co_name if flag else 'None', "#custom function name to combine with RVmodel")
        _print_custom_function += txtfmt.format("x",DA.x if flag else 'None',"#independent variable [time, true_anomaly]")
        if flag:
            fa      = DA.func_args
            fa_str  = []
            for k in fa.keys():
                if isinstance(fa[k],(int,float)):
                    fa_str.append(f'{k}:F({fa[k]})')
                if isinstance(fa[k],tuple):
                    fa_str.append(f"{k}:{'U' if len(fa[k])==3 else 'N' if len(fa[k])==2 else 'TN'}{str(fa[k]).replace(' ','')}" )
            fa_str  = ",".join(fa_str)
        else: fa_str = 'None'
        _print_custom_function += txtfmt.format("func_pars",fa_str,"#param names&priors e.g. A:U(0,1,2),P:N(2,1)")
        exa_str = [f"{k}:{v}" for k,v in DA.extra_args.items()]
        exa_str = ",".join(exa_str) if exa_str!=[] else 'None'
        _print_custom_function += txtfmt.format("extra_args",exa_str,"#extra args to func as a dict")
        _print_custom_function += txtfmt.format("op_func",'None' if (DA.replace_RVmodel or not flag) else DA.op_func.__code__.co_name ,"#function to combine the RV and custom models")
        _print_custom_function += txtfmt.format("replace_RVmodel",str(DA.replace_RVmodel) if flag else 'False',"#if the custom function replaces the RV model")
        print(_print_custom_function, file=file)


    if section == "depth_variation":
        grnames    = np.array(list(sorted(set(self._groups))))
        ngroup     = len(grnames)
        _print_depth_variation = f"""# ============ ddF setup ========================================================================================"""+\
                                    f"""\n{spacing}{"Fit_ddFs":8s}\t{"dRpRs":16s}\tdiv_white"""

        #define print out format
        txtfmt = f"\n{spacing}"+"{0:8s}\t{1:16s}\t{2:3s}"        
        # pri_ddf = f"N({self._ddfs.drprs.prior_mean},{self._ddfs.drprs.prior_width_lo})" if self._ddfs.drprs.prior=="p" else f"U({self._ddfs.drprs.bounds_lo},{self._ddfs.drprs.start_value},{self._ddfs.drprs.bounds_hi})"
        pri_ddf = prior_str(self._ddfs.drprs.user_input)
        t = txtfmt.format(self._ddfs.ddfYN, pri_ddf, self._ddfs.divwhite)
        _print_depth_variation += t

        print(_print_depth_variation, file=file)

    if section == "timing_variation":
        _print_timing_variation = f"""# ============ TTV setup ========================================================================================"""+\
                                    f"""\n{spacing}{"Fit_TTVs":8s}\t{"dt_priors(deviation from linear T0)":35s}\t\t{"transit_baseline[P]":19s}\t\tper_LC_T0"""
        #define print out format
        txtfmt = f"\n{spacing}"+"{0:8s}\t{1:35s}\t\t{2:19.4f}\t\t{3}"
        pri_ttv = f"N{self._ttvs.dt}" if len(self._ttvs.dt)==2 else f"U{self._ttvs.dt}"
        t = txtfmt.format(self._ttvs.to_fit, pri_ttv.replace(" ",""),self._ttvs.baseline, self._ttvs.per_LC_T0)
        _print_timing_variation += t
        print(_print_timing_variation, file=file)

    if section == "phasecurve":
        pars  = ["D_occ", "Fn", "ph_off","A_ev","A_db"]
        # descr = ["occultation depth", "atmospheric amplitude", "phase offset in degrees","ellipsoidal variation"]
        _print_phasecurve = f"""# ============ Phase curve setup ================================================================================ """+\
                                f"""\n{spacing}{'filt':{max_filt_len}s}  {'D_occ[ppm]':20s} {'Fn[ppm]':20s} {'ph_off[deg]':20s} {'A_ev[ppm]':20s} {'A_db[ppm]':20s}"""
        #define print out format
        txtfmt = f"\n{spacing}{{0:{max_filt_len}s}}"+"  {1:20s} {2:20s} {3:20s} {4:20s} {5:20s}"       
        
        DA = self._PC_dict
        for i,f in enumerate(self._filnames):
            pri_Docc  = prior_str(DA['D_occ'][f].user_input)
            pri_Fn    = prior_str(DA['Fn'][f].user_input)
            pri_phoff = prior_str(DA['ph_off'][f].user_input)
            pri_Aev   = prior_str(DA['A_ev'][f].user_input)
            pri_Adb   = prior_str(DA['A_db'][f].user_input)
            
            t = txtfmt.format(f, pri_Docc, pri_Fn, pri_phoff, pri_Aev, pri_Adb)
            _print_phasecurve += t
        print(_print_phasecurve, file=file)

    if section == "limb_darkening":
        DA = self._ld_dict
        _print_limb_darkening = f"""# ============ Limb darkening setup ============================================================================= """+\
                                f"""\n{spacing}{'filters':7s}\tfit\t{'q1':17s}\t{'q2':17s}"""
        #define print out format
        txtfmt = f"\n{spacing}"+"{0:7s}\t{1:3s}\t{2:17s}\t{3:17s}"       
        for i in range(len(self._filnames)):
            pri_q1 = f"N({DA['q1'][i]},{DA['sig_lo1'][i]})" if DA['sig_lo1'][i] else f"U({DA['bound_lo1'][i]},{DA['q1'][i]},{DA['bound_hi1'][i]})"  if DA['bound_hi1'][i] else f"F({DA['q1'][i]})"
            pri_q2 = f"N({DA['q2'][i]},{DA['sig_lo2'][i]})" if DA['sig_lo2'][i] else f"U({DA['bound_lo2'][i]},{DA['q2'][i]},{DA['bound_hi2'][i]})" if DA['bound_hi2'][i] else f"F({DA['q2'][i]})"
            to_fit = "y" if (pri_q1[0]!="F" or pri_q2[0]!="F") else "n"
            t = txtfmt.format(self._filnames[i], to_fit, pri_q1, pri_q2) 
            _print_limb_darkening += t

        print(_print_limb_darkening, file=file)

    if section == "contamination":
        DA = self._contfact_dict
        _print_contamination = f"""# ============ contamination setup (give contamination as flux ratio) ======================================== """+\
                                f"""\n{spacing}{'filters':7s}\tcontam_factor"""
        #define print out format
        txtfmt = f"\n{spacing}{{0:{max_filt_len}s}}"+"\t{1:20s}"       
        for i,f in enumerate(self._filnames):
            t = txtfmt.format(f,prior_str(DA[f].user_input))
            _print_contamination += t
        print(_print_contamination, file=file)

    if section == "stellar_pars":
        DA = self._stellar_dict
        _print_stellar_pars = f"""# ============ Stellar input properties ======================================================================"""+\
        f"""\n{spacing}{'# parameter':13s}   value """+\
        f"""\n{spacing}{'Radius_[Rsun]':13s}  N({DA['R_st'][0]},{DA['R_st'][1]})"""+\
        f"""\n{spacing}{'Mass_[Msun]':13s}  N({DA['M_st'][0]},{DA['M_st'][1]})"""+\
            f"""\n{spacing}Input_method:[R+rho(Rrho), M+rho(Mrho)]: {DA['par_input']}"""
        print(_print_stellar_pars, file=file)           

    if section == "fit":
        DA = self._fit_dict
        _print_fit_pars = f"""# ============ FIT setup ====================================================================================="""+\
        f"""\n{spacing}{'Number_steps':40s}  {DA['n_steps']} \n{spacing}{'Number_chains':40s}  {DA['n_chains']} \n{spacing}{'Number_of_processes':40s}  {DA['n_cpus']} """+\
            f"""\n{spacing}{'Burnin_length':40s}  {DA['n_burn']} \n{spacing}{'n_live':40s}  {DA['n_live']} \n{spacing}{'force_nlive':40s}  {DA['force_nlive']} \n{spacing}{'d_logz':40s}  {DA['dyn_dlogz']} """+\
                    f"""\n{spacing}{'Sampler(emcee/dynesty)':40s}  {DA['sampler']} \n{spacing}{'emcee_move(stretch/demc/snooker)':40s}  {DA['emcee_move']} """+\
                    f"""\n{spacing}{'nested_sampling(static/dynamic[pfrac])':40s}  {DA['nested_sampling']} \n{spacing}{'leastsq_for_basepar(y/n)':40s}  {DA['leastsq_for_basepar']} """+\
                        f"""\n{spacing}{'apply_LCjitter(y/n)':40s}  {DA['apply_LCjitter']} \n{spacing}{'apply_RVjitter(y/n)':40s}  {DA['apply_RVjitter']} """+\
                            f"""\n{spacing}{'LCjitter_loglims(auto/[lo,hi])':40s}  {DA['LCjitter_loglims']} \n{spacing}{'RVjitter_lims(auto/[lo,hi])':40s}  {DA['RVjitter_lims']} """+\
                                f"""\n{spacing}{'LCbasecoeff_lims(auto/[lo,hi])':40s}  {DA['LCbasecoeff_lims']} \n{spacing}{'RVbasecoeff_lims(auto/[lo,hi])':40s}  {DA['RVbasecoeff_lims']} """+\
                                    f"""\n{spacing}{'Light_Travel_Time_correction(y/n)':40s}  {DA['LTT_corr']}"""

        
        print(_print_fit_pars, file=file)

    if section == "rv_baseline":
        _print_rv_baseline = """# ============ Input RV curves, baseline function, GP, spline,  gamma ============================================ """+\
                                f"""\n{spacing}{'name':{max_name_len}s} {'RVunit':6s} {"scl_col":7s} |{'col0':4s} {'col3':4s} {'col4':4s} {"col5":4s}| {'sin':3s} {"GP":2s} {"spline_config  ":15s} | {f'gamma_{self._RVunit}':14s} """
        if self._names != []:
            DA = self._rvdict
            txtfmt = f"\n{spacing}{{0:{max_name_len}s}}"+" {1:6s} {2:7s} |{3:4d} {4:4d} {5:4d} {6:4d}| {7:3d} {8:2s} {9:15s} | {10:14s}"         
            for i in range(self._nRV):
                # gam_pri_ = f'N({DA["gammas"][i]},{DA["sig_lo"][i]})' if DA["sig_lo"][i] else f'U({DA["bound_lo"][i]},{DA["gammas"][i]},{DA["bound_hi"][i]})' if DA["bound_hi"][i] else f"F({DA['gammas'][i]})"
                gam_pri_ = prior_str(DA["gamma"][i].user_input)
                t = txtfmt.format(self._names[i],self._RVunit,self._rescaled_data.config[i], *self._RVbases[i],
                                    self._useGPrv[i],self._rvspline[i].conf,gam_pri_)
                _print_rv_baseline += t
        print(_print_rv_baseline, file=file)

    if section == "rv_gp":
        DA = self._rvGP_dict
        _print_gp = f"""# ============ RV GP properties ================================================================================== """
        # _print_gp += f"""\nsame_GP: {self._sameRVgp.flag}"""
        _print_gp += f"""\n{spacing}{"name":{max_name_len}s} {'par1':4s} {"kern1":5s} {'Amplitude1':18s} {'length_scale':17s} |{'op':2s}| {'par2':4s} {"kern2":5s} {'Amplitude2':18s} {'length_scale2':15s}"""
        if DA != {}: 
            #define gp print out format
            txtfmt = f"\n{spacing}{{0:{max_name_len}s}}"+" {1:4s} {2:5s} {3:18s} {4:17s} |{5:2s}| {6:4s} {7:5s} {8:18s} {9:15s} "        

            for rv in DA.keys():
                ngp = DA[rv]["ngp"]
                prior={}
                for p in ["amplitude", "lengthscale"]:
                    for j in range(ngp):
                        if DA[rv][p+str(j)].to_fit == "n":
                            prior[p+str(j)] = f"F({DA[rv][p+str(j)].start_value})"
                        elif DA[rv][p+str(j)].to_fit == "y" and DA[rv][p+str(j)].prior == "n":
                            b_lo = 0 if DA[rv][p+str(j)].bounds_lo==1e-20 else DA[rv][p+str(j)].bounds_lo
                            prior[p+str(j)] = f"LU({b_lo},{DA[rv][p+str(j)].start_value},{DA[rv][p+str(j)].bounds_hi})"
                        elif DA[rv][p+str(j)].to_fit == "y" and DA[rv][p+str(j)].prior == "p":
                            prior[p+str(j)] = f"N({DA[rv][p+str(j)].prior_mean},{DA[rv][p+str(j)].prior_width_lo})"

                if ngp == 2:
                    t = txtfmt.format('same' if self._sameRVgp.flag else rv,DA[rv]["amplitude0"].user_data[1], DA[rv]["amplitude0"].user_data[0],  
                                                    prior["amplitude0"], prior["lengthscale0"], DA[rv]["op"], 
                                        DA[rv]["amplitude1"].user_data[1], DA[rv]["amplitude1"].user_data[0],
                                                    prior["amplitude1"], prior["lengthscale1"])
                else:
                    t = txtfmt.format('same' if self._sameRVgp.flag else rv,DA[rv]["amplitude0"].user_data[1], DA[rv]["amplitude0"].user_data[0],  
                                                    prior["amplitude0"], prior["lengthscale0"], "--", 
                                        "None", "None", "None", "None")
                _print_gp += t
                if self._sameRVgp.flag:      #dont print the other GPs if same_GP is True
                    break
        print(_print_gp, file=file)

class _param_obj():
    def __init__(self,to_fit,start_value,step_size,
                    prior, prior_mean, prior_width_lo, prior_width_hi,
                    bounds_lo, bounds_hi,user_input=None,user_data=None):
        """  
        convenience class to create a parameter object with the following attributes:

        Parameters:
        -----------
        to_fit : str;
            'y' or 'n' to fit or not fit the parameter.
        start_value : float;
            starting value for the parameter.
        step_size : float;
            step size for the parameter.
        prior : str;
            'n' or 'p' to not use (n) or use (p) a normal prior.
        prior_mean : float;
            mean of the normal prior.
        prior_width_lo : float;
            lower sigma of the normal prior.
        prior_width_hi : float;
            upper sigma of the normal prior.
        bounds_lo : float;
            lower bound for the parameter.
        bounds_hi : float;
            upper bound for the parameter.
        user_data : any;
            any data to be stored in the parameter object.
        user_info: tuple, int, float:
            stores prior input given by the user

        Returns:
        --------
        param_obj : object;
            object with the parameters.
        """
    
        self.to_fit         = to_fit if (to_fit in ["n","y"]) else _raise(ValueError, "to_fit (to_fit) must be 'n' or 'y'")
        self.start_value    = start_value
        self.step_size      = step_size
        self.prior          = prior if (prior in ["n","p"]) else _raise(ValueError, "prior (prior) must be 'n' or 'p'")
        self.prior_mean     = prior_mean
        self.prior_width_lo = prior_width_lo
        self.prior_width_hi = prior_width_hi
        self.bounds_lo      = bounds_lo
        self.bounds_hi      = bounds_hi
        self.user_input     = user_input
        self.user_data      = user_data

    @classmethod
    def from_tuple(cls, param_in, step=None,lo=None, hi=None, user_input=None,func_call=""):
        """
        alternative method to initialize _param_obj using from a tuple.
        * if int/float is given returns: (to_fit="n",start_value=param_in,step_size=0,prior="n",prior_mean=param_in,prior_width_lo=0,prior_width_hi=0,bounds_lo=0,bounds_hi=0,user_input=None)
        * if tuple of len 2 it returns:  (to_fit="y",start_value=param_in[0],step_size=0.1*param_in[1],prior="p",prior_mean=param_in[0],prior_width_lo=param_in[1],prior_width_hi=param_in[1],bounds_lo=param_in[0]-10*param_in[1],bounds_hi=param_in[0]+10*param_in[1],user_input=None)
        * if tuple of len 3 it returns:  (to_fit="y",start_value=param_in[0],step_size=0.001*np.ptp(param_in),prior="n",prior_mean=param_in[0],prior_width_lo=0,prior_width_hi=0,bounds_lo=param_in[0],bounds_hi=param_in[2],user_input=None)
        * if tuple of len 4 it returns:  (to_fit="y",start_value=param_in[2],step_size=0.1*param_in[3],prior="p",prior_mean=param_in[2],prior_width_lo=param_in[3],prior_width_hi=param_in[3],bounds_lo=param_in[0],bounds_hi=param_in[1],user_input=None)

        Parameters:
        -----------
        param_in : int, float,tuple,None;
            input float/tuple with the parameters for the object.
        step : float,None;
            step size for the parameter to override the default value derived from param
        lo : float,None;
            lower bound for the parameter to override the default value derived from param_in
        hi : float;
            upper bound for the parameter to override the default value derived from param_in
        user_input; tuple, int, float:
            stores prior input given by the user
        func_call : str;
            name of the function calling this method, to be used in error messages.
    
        Returns:
        --------
        param_obj : object;
            object with the parameters.

        Example:
        --------
        >>> RpRs = (0.1,0.002)
        >>> param_obj = _param_obj.from_tuple(RpRs,func_call="planet_parameters():") 
        """
        assert isinstance(func_call,str),f"_param_obj.from_tuple() func_call must be a string but {func_call} given."
        
        v = param_in
        if isinstance(v, (int, float)): #fixed parameter
            params = ["n",v,0,"n",v,0,0,0,0,user_input]
        elif isinstance(v, tuple):
            if len(v)==2:  #normal prior
                step   = 0.1*v[1] if step==None else step
                lo_lim = v[0]-10*v[1] if lo==None else lo
                hi_lim = v[0]+10*v[1] if hi==None else hi
                params = ["y",v[0],step,"p",v[0],v[1],v[1],lo_lim,hi_lim,user_input]
            elif len(v)==3: #uniform prior
                assert v[0]<=v[1]<=v[2],f"{func_call} wrongly defined uniform prior. must be of form (min,start,max) with min<=start<=max but {v} given."
                step = min(0.001,0.001*np.ptp(v)) if step==None else step
                lo_lim = v[0] if lo==None else lo
                hi_lim = v[2] if hi==None else hi
                params = ["y",v[1],step,"n",v[1],0,0,lo_lim,hi_lim,user_input]
            elif len(v)==4: #truncated normal prior
                assert v[0]<v[1],f"{func_call} wrongly defined trucated normal prior. must be of form (min,max,mean,std) with min<max but {v} given. "
                step = 0.1*v[3] if step==None else step
                params = ["y",v[2],step,"p",v[2],v[3],v[3],v[0],v[1],user_input]
            else:
                raise TypeError(f"{func_call} tuple must have 2,3 or 4 elements")
        elif v==None:
            params = ["n",None,0,"n",None,0,0,0,0,user_input]
        else:
            raise TypeError(f"{func_call} input must be an int, float, tuple or None")
        return cls(*params)

    def _set(self, par_list):
        return self.__init__(*par_list)
    
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
        file_list : list;
            List of filenames for the lightcurves. Files must have 9 columns: time,flux,err,xc,xc,xc,xc,xc,xc. 
            where xc are columns that can be used in decorrelating the flux. Arrays of zeroes are put in xc if file contains less than 9 columns.
        data_filepath : str;
            Filepath where lightcurves files are located. Default is None which implies the data is in the current working directory.
        filter : list, str, None;
            filter for each lightcurve in file_list. if a str is given, it is used for all lightcurves,
            if None, the default of "V" is used for all.
        wl : list, int, float, None;
            central wavelength in microns for each lightcurve in file_list. if int or float is given, it is used for all lightcurves,
            if None, the default of 0.6 is used for all.
        nplanet : int;
            number of planets in the system. Default is 1.
        verbose : bool;
            if True, prints out information about the lightcurves. Default is True.
        
        Returns:
        --------
        lc_obj : light curve object

        Example:
        --------
        >>> lc_obj = load_lightcurves(file_list=["lc1.dat","lc2.dat"], filters=["V","I"], wl=[0.6,0.8])
        
    """
    def __init__(self, file_list=None, data_filepath=None, filters=None, wl=None, nplanet=1,
                    verbose=True, show_guide=False,lamdas=None):
        self._obj_type = "lc_obj"
        self._nplanet  = nplanet
        self._fpath    = os.getcwd()+'/' if data_filepath is None else data_filepath
        self._names    = [file_list] if isinstance(file_list, str) else [] if file_list is None else file_list
        for lc in self._names: assert os.path.exists(self._fpath+lc), f"file {lc} does not exist in the path {self._fpath}."
        
        if lamdas is not None:
            warnings.warn("The 'lamdas' parameter  in `load_lightcurves()` is deprecated and will be discontinued in future versions. Use 'wl' instead.", DeprecationWarning)
            if wl is None: wl = lamdas
        
        assert filters is None or isinstance(filters, (list, str)), f"filters is of type {type(filters)}, it should be a list, a string or None."
        assert wl  is None or isinstance(wl, (list, int, float)), f"wl is of type {type(wl)}, it should be a list, int or float."
        
        if isinstance(filters, str): filters = [filters]
        if isinstance(wl, (int, float)): wl = [float(wl)]

        self._nphot = len(self._names)
        if filters is not None and len(filters) == 1: filters = filters*self._nphot
        if wl is not None and len(wl)==1: wl  = wl *self._nphot

        self._filters = ["V"]*self._nphot if filters is None else [f for f in filters]
        self._wl      = [0.6]*self._nphot if wl is None else [l for l in wl]
        self._filter_shortcuts = filter_shortcuts
        
        assert self._nphot == len(self._filters) == len(self._wl), f"filters and wl must be a list with same length as file_list (={self._nphot})"
        self._filnames  = np.array(list(sorted(set(self._filters),key=self._filters.index))) #unique filters

        #modify input files to have 9 columns as CONAN expects then save as attribute of self
        self._input_lc = {}     #dictionary to hold input lightcurves
        self._rms_estimate, self._jitt_estimate = [], []
        for f in self._names:
            fdata = np.loadtxt(self._fpath+f)
            nrow,ncol = fdata.shape
            if ncol < 9:
                # if verbose: print(f"writing ones to the missing columns of file: {f}")
                new_cols = np.ones((nrow,9-ncol))
                fdata = np.hstack((fdata,new_cols))

            #remove nan rows from fdata and print number of removed rows
            n_nan = np.sum(np.isnan(fdata).any(axis=1))
            if n_nan > 0: print(f"removed {n_nan} row(s) with NaN values from file: {f}")            
            fdata = fdata[~np.isnan(fdata).any(axis=1)]
            #store input files in lc object
            self._input_lc[f] = {}
            for i in range(9): self._input_lc[f][f"col{i}"] = fdata[:,i]
            #compute rms and multiplicative jitter
            self._rms_estimate.append( np.std(np.diff(fdata[:,1]))/np.sqrt(2) )      #std(diff(flux))/√2 is a good estimate of the rms noise
            self._jitt_estimate.append( np.sqrt(self._rms_estimate[-1]**2 - np.mean(fdata[:,2]**2)) ) # √(rms^2 - mean(err^2)) is a good estimate of the required jitter to add quadratically
            if np.isnan(self._jitt_estimate[-1]): self._jitt_estimate[-1] = 1e-20

        #list to hold initial baseline model coefficients for each lc
        self._bases_init =  [dict(off=1, A0=0, B0= 0, C0=0, D0=0,A3=0, B3=0, A4=0, B4=0,A5=0, B5=0, 
                                    A6=0, B6=0,A7=0, B7=0, A8=0, B8=0, amp=0,freq=0,phi=0,ACNM=1,BCNM=0) 
                                for _ in range(self._nphot)]

        self._show_guide    = show_guide
        self._masked_points = False
        self._clipped_data  = SimpleNamespace(flag=False, lc_list=self._names, config=["None"]*self._nphot)
        self._rescaled_data = SimpleNamespace(flag=False, config=["None"]*self._nphot)
        self.lc_baseline(re_init = hasattr(self,"_bases"), verbose=False)  
        self.add_custom_LC_function(verbose=False) 

        if self._show_guide: print("\nNext: use method `lc_baseline` to define baseline model for each lc or method " + \
            "`get_decorr` to obtain best best baseline model parameters according bayes factor comparison")

    def rescale_data_columns(self, method="med_sub", verbose=True):

        """
            Function to rescale the data columns of the lightcurves. This can be important when decorrelating the data with polynomials.
            The operation is not performed on columns 0,1,2. It is only performed on columns whose values do not span zero.
            Function can only be run once on the loaded datasets but can be reset by running `load_lightcurves()` again. 

            The method can be one of ["med_sub", "rs0to1", "rs-1to1","None"] which subtracts the median, rescales t0 [0-1], rescales to [-1,1], or does nothing, respectively.
            The default is "med_sub" which subtracts the median from each column.
        """
        
        if self._rescaled_data.flag:
            print("Data columns have already been rescaled. run `load_lightcurves()` again to reset.")
            return None
        
        if isinstance(method,str): method = [method]*self._nphot
        elif isinstance(method, list):
            assert len(method)==1 or len(method)==self._nphot, f'rescale_data_columns(): method must be either str or list of same length as number of input lcs ({self._nphot})'
        else: _raise(TypeError,'rescale_data_columns(): method must be either str or list of same length as number of input lcs ({self._nphot})')

        if method == ["None"]*self._nphot:
            self._rescaled_data.flag = False
            return None

        for j,lc in enumerate(self._names):
            assert method[j] in ["med_sub", "rs0to1", "rs-1to1","None"], f"method must be one of ['med_sub','rs0to1','rs-1to1','None'] but {method[j]} given"
            if verbose: print(f"No rescaling for {lc}") if method[j]=="None" else print(f"Rescaled data columns of {lc} with method:{method[j]}")
            for i in range(9):
                if i not in [0,1,2]:
                    if not (min(self._input_lc[lc][f"col{i}"]) <= 0 <=  max(self._input_lc[lc][f"col{i}"])):     #if zero not in array
                        if method[j] == "med_sub":
                            self._input_lc[lc][f"col{i}"] -= np.median(self._input_lc[lc][f"col{i}"])
                        elif method[j] == "rs0to1":
                            self._input_lc[lc][f"col{i}"] = rescale0_1(self._input_lc[lc][f"col{i}"])
                        elif method[j] == "rs-1to1":
                            self._input_lc[lc][f"col{i}"] = rescale_minus1_1(self._input_lc[lc][f"col{i}"])
                        else: pass

        self._rescaled_data = SimpleNamespace(flag=True, config=method)

    def get_decorr(self, T_0=None, Period=None, rho_star=None, Duration=None, D_occ=0, Impact_para=0, RpRs=1e-5,
                    Eccentricity=0, omega=90, Fn=None, ph_off=None, A_ev=0, A_db=0, K=0, q1=0, q2=0, 
                    mask=False, ss_exp=None,Rstar=None,delta_BIC=-5, decorr_bound =(-10,10),
                    exclude_cols=[], enforce_pars=[],show_steps=False, plot_model=True, use_jitter_est=False,
                    setup_baseline=True, setup_planet=False, custom_LCfunc=None, verbose=True):
        """
            Function to obtain best decorrelation parameters for each light-curve file using the forward selection method.
            It compares a model with only an offset to a polynomial model constructed with the other columns of the data.
            It uses columns 0,3,4,5,6,7,8 to construct the polynomial trend model. The temporary decorr parameters are labelled Ai,Bi for 1st & 2nd order coefficients in column i.
            if a spline, sinusoid or gp has been setup for the LC object, it is varied also during the decorrelation process.
            
            Decorrelation parameters that reduce the BIC by 5(i.e delta_BIC = -5) are iteratively selected. This implies bayes_factor=exp(-0.5*-5) = 12 or more is required for a parameter to be selected.
            The result can then be used to populate the `lc_baseline` method, if use_result is set to True. The transit, limb darkening and phase curve parameters can also be setup from the inputs to this function.

            Parameters:
            -----------
            T_0, Period, rho_star/Duration, D_occ, Impact_para, RpRs, Eccentricity, omega, Fn, ph_off,A_ev, A_db: floats,tuple, None;
                transit/eclipse parameters of the planet. T_0 and Period must be in same units as the time axis (col0) in the data file.
                D_occ, Fn, A_ev and A_db are in ppm
                if float/int, the values are held fixed. if tuple/list of len 2 implies gaussian prior as (mean,std) while len 3 implies [min,start_val,max].
            q1,q2 : float,tuple, list  (optional);
                Kipping quadratic limb darkening parameters. if float, the values are held fixed. if tuple/list of len 2 implies gaussian prior as (mean,std) while len 3 implies [min,start_val,max].
                Give list of values to assign value to each unique filter in the data, or one value to be used for all filtets. Default is 0 for all filters.
            delta_BIC : float (negative);
                BIC improvement a parameter needs to provide in order to be considered relevant for decorrelation. + \
                    Default is conservative and set to -5 i.e, parameters needs to lower the BIC by 5 to be included as decorrelation parameter.
            mask : bool ;
                If True, transits and eclipses are masked using T_0, P and rho_star (duration).
            decorr_bound: tuple of size 2;
                bounds when fitting decorrelation parameters. Default is (-1,1)
            ss_exp : list, None;
                exposure time of the lcs to configure supersampline. Default is None which implies no supersampling.
            Rstar : float, None;
                Stellar radius in solar radii, required for performing light travel time correction. Default is None
            exclude_cols : list of int;
                list of column numbers (e.g. [3,4]) to exclude from decorrelation. Default is []. Can also specify "all" to only fit an offset
            enforce_pars : list of int;
                list of decorr params (e.g. ['B3', 'A5']) to enforce in decorrelation. Default is [].
            show_steps : Bool, optional;
                Whether to show the steps of the forward selection of decorr parameters. Default is False
            plot_model : Bool, optional;
                Whether to overplot suggested trend model on the data. Defaults to True.
            use_jitter_est : Bool, optional;
                Whether to use the estimated jitter from the data in the decorrelation model. Defaults to False.
            setup_baseline : Bool, optional;
                whether to use result to setup the baseline model and transit/eclipse models. Default is True.
            setup_planet : Bool, optional;
                whether to use input to setup the transit model(planet_parameters/phasecurve/LD functions). Default is False.
            custom_LCfunc : object, optional;
                namespace object created from `lc_obj.add_custom_LC_function()`. 
                It contains a custom function with a parameter dictionary and an operation function defining how to combine the output with the LC model.
                It can also specify if the custom function replaces the native CONAN LC model.
            verbose : Bool, optional;
                Whether to show the table of baseline model obtained. Defaults to True.
        
            Returns
            -------
            decorr_result: list of result object
                list containing result object for each lc.
        """
        if exclude_cols=="all": exclude_cols = [0,3,4,5,6,7,8]
        elif isinstance(exclude_cols, list):
            for c in exclude_cols: assert isinstance(c, int), f"get_decorr(): column number to exclude from decorrelation must be an integer but {c} given in exclude_cols."
        else: _raise(TypeError, "get_decorr(): exclude_cols must be a list of column numbers to exclude from decorrelation or 'all' to only fit an offset.")

        if custom_LCfunc is not None: assert callable(custom_LCfunc.func), "get_decorr(): custom_LCfunc must be a callable function"
        
        nfilt = len(self._filnames)
        if isinstance(q1, np.ndarray): q1 = list(q1)
        if isinstance(q1, list): assert len(q1) == nfilt, f"get_decorr(): q1 must be a list of same length as number of unique filters {nfilt} but {len(q1)} given." 
        else: q1=[q1]*nfilt
        if isinstance(q2, np.ndarray): q2 = list(q2)
        if isinstance(q2, list): assert len(q2) == nfilt, f"get_decorr(): q2 must be a list of same length as number of unique filters {nfilt} but {len(q2)} given." 
        else: q2=[q2]*nfilt

        blpars = {"dcol0":[], "dcol3":[],"dcol4":[], "dcol5":[], "dcol6":[], "dcol7":[], "dcol8":[]}  #inputs to lc_baseline method
        self._decorr_result = []   #list of decorr result for each lc.
        
        if self._nplanet > 1:
            assert rho_star is not None, f"get_decorr(): rho_star must be given for multiplanet system but {rho_star} given."
            assert  Duration==None, f"get_decorr(): Duration must be None for multiplanet systems, since transit model uses rho_star but {Duration=} given."
        else:
            #check that rho_star and Duration are not both given
            if rho_star is not None: assert Duration is None, "get_decorr(): Duration must be None if rho_star is given."
            if Duration is not None: assert rho_star is None, "get_decorr(): rho_star must be None if Duration is given."

        input_pars = dict(T_0=T_0, Period=Period, Impact_para=Impact_para, RpRs=RpRs, Eccentricity=Eccentricity, omega=omega, K=K)
        self._tra_occ_pars = dict(T_0=T_0, Period=Period, D_occ=D_occ, Impact_para=Impact_para, RpRs=RpRs, Eccentricity=Eccentricity,\
                                    omega=omega, Fn=Fn, ph_off=ph_off,A_ev=A_ev,A_db=A_db) #transit/occultation parameters
        # add rho_star/Duration to input_pars and self._tra_occ_pars if given
        if rho_star is not None: input_pars["rho_star"] = rho_star; self._tra_occ_pars["rho_star"] = rho_star
        if Duration is not None: input_pars["Duration"] = Duration; self._tra_occ_pars["Duration"] = Duration

        
        for p in self._tra_occ_pars:
            if p not in ["rho_star","Duration","Fn","ph_off","D_occ","A_ev","A_db"]:
                if isinstance(self._tra_occ_pars[p], (int,float,tuple)): self._tra_occ_pars[p] = [self._tra_occ_pars[p]]*self._nplanet
                if isinstance(self._tra_occ_pars[p], (list)): assert len(self._tra_occ_pars[p]) == self._nplanet, \
                    f"get_decorr(): {p} must be a list of same length as number of planets {self._nplanet} but {len(self._tra_occ_pars[p])} given."
            else:
                assert isinstance(self._tra_occ_pars[p],(int,float,tuple,type(None))),f"get_decorr(): {p} must be one of int/float/tuple/None but {self._tra_occ_pars[p]} given "

        ld_q1, ld_q2 = {},{}
        for i,fil in enumerate(self._filnames):
            ld_q1[fil] = q1[i]
            ld_q2[fil] = q2[i]
        
        assert delta_BIC<0,f'get_decorr(): delta_BIC must be negative for parameters to provide improved fit but {delta_BIC} given.'
        
        #check spline setup
        if [self._lcspline[i].conf for i in range(self._nphot)] == ["None"]*self._nphot: #if no input spline in lc_obj, set to None
            spline = [None]*self._nphot
        else:
            spline = [self._lcspline[i].conf for i in range(self._nphot)]
            for i,sp in enumerate(spline):
                if sp != "None":
                    if "|" not in sp:   #1D spline
                        ks = sp.split("k")[-1]
                        spline[i]= dict(col          = int(sp.split("d")[0][1]) , 
                                        knot_spacing = float(ks) if  ks!='r' else ks, 
                                        degree       = int(sp.split("k")[0].split("d")[-1])) 
                    else:  #2D spline
                        sp = sp.split("|")
                        ks = [sp[0].split("k")[-1], sp[1].split("k")[-1] ]
                        spline[i]= dict(col          = (int(sp[0].split("d")[0][1]),int(sp[1].split("d")[0][1])), 
                                        knot_spacing = (float(ks[0]) if ks[0]!='r' else ks[0], float(ks[1]) if ks[1]!='r' else ks[1]), 
                                        degree       = (int(sp[0].split("k")[0].split("d")[-1]),int(sp[1].split("k")[0].split("d")[-1])) )
                else:
                    spline[i] = None

        #check sinusoid model
        if any([v.trig!= None for v in self._sine_dict.values()]):    #if any sinusoid model is defined
            sine_dict = deepcopy(self._sine_dict)
            fit_type   = [v.fit for v in sine_dict.values()][0]
            if fit_type=="same": 
                temp = {k:sine_dict["same"] for k in self._names}
            if fit_type=="filt": 
                temp = {k:sine_dict[f] for k,f in zip(self._names,self._filters)}
            if fit_type in ["slct","all"]:
                temp = {k:sine_dict[k] for k in self._names }

            sinusoid = {k:(v if v.trig else None) for k,v in temp.items() }
            
            # get sinus parameters for each each lc
            for k in self._names:
                sinus = sinusoid[k]
                if sinus!=None:
                    sinus.params = {}
                    trig_funcs = ["sin","cos"] if sinus.trig=="sincos" else [sinus.trig]
                    for p in ["Amp","P","x0"]:
                        for trig in trig_funcs:
                            for n in range(1,sinus.n+1):
                                pnm = f"{trig}{f'{n}' if n>1 else ''}_{p}"   #sinus parameter name
                                sinus.params[pnm] = sinus.__dict__[p].user_input  #update with user-defined priors
                                if p!="Amp": break
                            if p!="Amp": break
        else:
            sinusoid = {k:None for k in self._names}
        
        #check gp
        if self._GP_dict != {}:
            celerite_kernel = dict(mat32=celerite.terms.Matern32Term, sho=celerite.terms.SHOTerm,real=celerite.terms.RealTerm)
            GP = deepcopy(self._GP_dict)
            for k in self._names:
                if k not in GP.keys(): GP[k] = None
            
            for k in self._names:
                if GP[k]!=None:
                    geepee = GP[k] = SimpleNamespace(**GP[k])
                    geepee.params  = {  "log_GP_amp1":tuple(np.log(geepee.amplitude0.user_input)),    #difficult to set loguniform priors for a least-square fit, so we fit the log of the amplitude and lengthscale 
                                        "log_GP_len1":tuple(np.log(geepee.lengthscale0.user_input))}
                    geepee.kern    = [geepee.amplitude0.user_data[0]]
                    geepee.column  = [geepee.amplitude0.user_data[1]]
                    del geepee.amplitude0, geepee.lengthscale0            #remove extracted attributes
                    if geepee.ngp==2:   # if 2nd GP kernel is defined
                        geepee.params["log_GP_amp2"]=np.log(geepee.amplitude1.user_input)
                        geepee.params["log_GP_len2"]=np.log(geepee.lengthscale1.user_input)
                        geepee.kern.append(geepee.amplitude1.user_data[0])
                        geepee.column.append(geepee.amplitude1.user_data[1])
                        del geepee.amplitude1, geepee.lengthscale1        # remove extracted attributes
                
                    #instantiate kernels with dummy parameters
                    kernels = []
                    gp_conv = gp_params_convert()   #class containing functions to convert gp amplitude and lengthscale to the required values for the different kernels 
                    for i in range(geepee.ngp):
                        gpkern = geepee.kern[i]
                        if gpkern=='sho':
                            kernels.append(celerite_kernel[gpkern](log_S0 =-10, log_Q=np.log(1/np.sqrt(2)), log_omega0=1)) #dummy initialization
                            kernels[i].freeze_parameter("log_Q")
                        else:
                            kernels.append(celerite_kernel[gpkern](-10, 1)) #dummy initialization

                        gppar1, gppar2 =  gp_conv.get_values(kernels=gpkern, data="lc", pars=[10,0.1])
                        kernels[i].set_parameter_vector([gppar1, gppar2])
                    
                    if geepee.ngp==1:
                        kernel = kernels[0]
                    else: 
                        kernel = kernels[0]+kernels[1] if geepee.op=="+" else kernels[0]*kernels[1]

                    geepee.GPobj = celerite.GP(kernel, mean=0, fit_mean = False)

        else:
            GP = {k:None for k in self._names}
            
        #check supersampling input. If no input check if already defined in .supersample() method
        if [self._ss[i].config for i in range(self._nphot)] == ["None"]*self._nphot: 
            ss_exp = [None]*self._nphot
        else: 
            ss_exp = [self._ss[i].config for i in range(self._nphot)]
            ss_exp = [(float(exp[1:]) if exp!="None" else None) for exp in ss_exp]

        ### begin computation
        self._tmodel = []              #list to hold determined trendmodel for each lc
        decorr_cols = [0,3,4,5,6,7,8]  #decorrelation columns
        for c in exclude_cols: assert c in decorr_cols, f"get_decorr(): column number to exclude from decorrelation must be in {decorr_cols} but {c} given in exclude_cols." 
        _ = [decorr_cols.remove(c) for c in exclude_cols]  #remove excluded columns from decorr_cols

        for j,file in enumerate(self._names):
            df = self._input_lc[file]
            if verbose: 
                print(_text_format.BOLD + f"\ngetting decorr params for lc: {file} (spline={spline[j]!=None}, sine={sinusoid[file]!=None}, gp={GP[file]!=None}, s_samp={ss_exp[j]!=None}, jitt={self._jitt_estimate[j]*1e6 if use_jitter_est else 0:.1f}ppm)" + _text_format.END)
            
            all_par  = [f"{L}{i}" for i in decorr_cols for L in ["A","B"]]     #A0,B0,A3,B3,...
            sin_pars = sinusoid[file].params if sinusoid[file]!=None else {}   #sin(C5)_Amp, sin(C5)_P,...
            gp_pars  = GP[file].params if GP[file]!=None else {}               #log_GP_Amp, log_GP_len,...
    
            #perform first fit of all jump parameters(astro,gp,sine,spline) with offset as only decorr par
            out = _decorr(df, **self._tra_occ_pars, **sin_pars, **gp_pars, q1=ld_q1[self._filters[j]],q2=ld_q2[self._filters[j]], mask=mask,
                            offset=0 if spline[j]==None else None, decorr_bound=decorr_bound,spline=spline[j],sinus=sinusoid[file],gp=GP[file],ss_exp=ss_exp[j], 
                            jitter=self._jitt_estimate[j] if use_jitter_est else 0, Rstar=Rstar, 
                            custom_LCfunc=custom_LCfunc, npl=self._nplanet)    #no trend, only offset if no spline
            if set(exclude_cols) == set([0,3,4,5,6,7,8]):
                best_pars ={"offset":0} if spline[j] is None else {}        #setting to 0 means it will be varied in the fit
            else:
                best_bic  = out.bic                                         #best bic from first fit
                best_pars = {"offset":0} if spline[j] is None else {}               #offset always included if no spline
                for cp in enforce_pars: best_pars[cp]=0                             #add enforced parameters
                _ = [all_par.remove(cp) for cp in enforce_pars if cp in all_par]    #remove enforced parameters from all_par to test

                if show_steps: print(f"{'Param':7s} : {'BIC':18s} N_pars \n---------------------------")

                del_BIC = -np.inf  #initialize del_BIC to -inf to start the while loop
                while del_BIC < delta_BIC:
                    if show_steps: print(f"{'Best':7s} : {best_bic:<18.2f} {len(best_pars.keys())} {list(best_pars.keys())}\n---------------------")
                    pars_bic = {}
                    for p in all_par:
                        dtmp = best_pars.copy()   #always include offset if no spline
                        dtmp[p] = 0               #setting the par p to 0 means it will be varied in the fit
                        out = _decorr(self._input_lc[file], **self._tra_occ_pars, **sin_pars, **gp_pars, q1=ld_q1[self._filters[j]],q2=ld_q2[self._filters[j]],**dtmp,
                                        decorr_bound=decorr_bound,spline=spline[j],sinus=sinusoid[file],gp=GP[file],ss_exp=ss_exp[j], 
                                        jitter=self._jitt_estimate[j] if use_jitter_est else 0, Rstar=Rstar,
                                        custom_LCfunc=custom_LCfunc, npl=self._nplanet)
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

            result = _decorr(df, **self._tra_occ_pars, **sin_pars, **gp_pars, q1=ld_q1[self._filters[j]],q2=ld_q2[self._filters[j]],
                                **best_pars, decorr_bound=decorr_bound,spline=spline[j],sinus=sinusoid[file],gp=GP[file],ss_exp=ss_exp[j], 
                                jitter=self._jitt_estimate[j] if use_jitter_est else 0, Rstar=Rstar, 
                                custom_LCfunc=custom_LCfunc, npl=self._nplanet)

            self._decorr_result.append(result)
            if verbose: print(f"BEST BIC:{result.bic:.2f}, pars:{list(best_pars.keys())}")
            
            #calculate determined trend and tra/occ model over all data(no mask)
            pps = result.params.valuesdict()
            #convert result transit parameters to back to a list
            for p in ['RpRs', 'Impact_para', 'T_0', 'Period', 'Eccentricity', 'omega']:
                if self._nplanet==1:
                    pps[p] = [pps[p]]  
                else:      
                    pps[p] = [pps[p+f"_{n}"] for n in range(1,self._nplanet+1)]
                    _ = [pps.pop(f"{p}_{n}") for n in range(1,self._nplanet+1)]
            if custom_LCfunc is not None:
                best_custom_LCfunc = deepcopy(custom_LCfunc)
                best_custom_LCfunc.func_args = {p:pps[p] for p in best_custom_LCfunc.func_args.keys()} #update best_custom_LCfunc parameters to values from fit
                _ = [pps.pop(p) for p in custom_LCfunc.func_args.keys()] # remove custom_LCfunc parameters from pps
                
    
            self._tmodel.append(_decorr(df,**pps, spline=spline[j],sinus=sinusoid[file],gp=GP[file],ss_exp=ss_exp[j], Rstar=Rstar, custom_LCfunc=custom_LCfunc,npl=self._nplanet, return_models=True))

            #set-up lc_baseline model from obtained configuration
            blpars["dcol0"].append( 2 if pps["B0"]!=0 else 1 if  pps["A0"]!=0 else 0)
            blpars["dcol3"].append( 2 if pps["B3"]!=0 else 1 if  pps["A3"]!=0 else 0)
            blpars["dcol4"].append( 2 if pps["B4"]!=0 else 1 if  pps["A4"]!=0 else 0)
            blpars["dcol5"].append( 2 if pps["B5"]!=0 else 1 if  pps["A5"]!=0 else 0)
            blpars["dcol6"].append( 2 if pps["B6"]!=0 else 1 if  pps["A6"]!=0 else 0)
            blpars["dcol7"].append( 2 if pps["B7"]!=0 else 1 if  pps["A7"]!=0 else 0)
            blpars["dcol8"].append( 2 if pps["B8"]!=0 else 1 if  pps["A8"]!=0 else 0)

            if setup_baseline:
                # store baseline model coefficients for each lc, to used as start values of mcmc
                self._bases_init[j] = dict(off=1+pps["offset"], 
                                            A0=pps["A0"], B0=pps["B0"], C0=0, D0=0,
                                            A3=pps["A3"], B3=pps["B3"], 
                                            A4=pps["A4"], B4=pps["B4"],
                                            A5=pps["A5"], B5=pps["B5"], 
                                            A6=pps["A6"], B6=pps["B6"],
                                            A7=pps["A7"], B7=pps["B7"],
                                            A8=pps["A8"], B8=pps["B8"], 
                                            amp=0,freq=0,phi=0,ACNM=1,BCNM=0)
                
                # if fit_type in ["slct","all"] and sinusoid[file]!=None:
                #     sin_decorr_vars = ['sin_Amp','sin2_Amp','sin3_Amp','cos_Amp','cos2_Amp','cos3_Amp','sin_P', 'sin_x0']
                #     self._sine_dict[file].init_pars = {}
                #     for p in sin_decorr_vars: 
                #         self._sine_dict[file].init_pars[p] = pps[p]


        if plot_model:
            _plot_data(self,plot_cols=(0,1,2),col_labels=("time","flux"),model_overplot=self._tmodel)

        #prefill other light curve setup from the results here or inputs given here.
        if setup_baseline:       
            # baseline
            if verbose: print(_text_format.BOLD + "\nSetting-up parametric baseline model from decorr result" +_text_format.END)
            self.lc_baseline(**blpars, sin=[self._bases[i][7] for i in range(self._nphot)],gp=self._useGPphot, verbose=verbose)
            if verbose: print(_text_format.RED + f"\nNote: GP flag for the lcs has been set to {self._useGPphot}. "+\
                    "Use `._useGPphot` attribute to modify this list with 'y','ce' or 'n' for each loaded lc\n" + _text_format.END)

        if setup_planet:
            # transit/RV
            if verbose: print(_text_format.BOLD + "\nSetting-up transit pars from input values" +_text_format.END)
            self.planet_parameters(**input_pars, verbose=verbose)

            # phasecurve
            if np.any([self._tra_occ_pars["D_occ"],self._tra_occ_pars["Fn"],self._tra_occ_pars["ph_off"],self._tra_occ_pars["A_ev"],self._tra_occ_pars["A_db"]] != 0): 
                if verbose: print(_text_format.BOLD + "\nSetting-up Phasecurve pars from input values" +_text_format.END)
                self.setup_phasecurve(D_occ=self._tra_occ_pars["D_occ"], Fn=self._tra_occ_pars["Fn"],
                                        ph_off=self._tra_occ_pars["ph_off"], A_ev=self._tra_occ_pars["A_ev"], 
                                        A_db=self._tra_occ_pars["A_db"], verbose=verbose)
            else:
                self.setup_phasecurve(verbose=False)
            
            # limb darkening
            if verbose: print(_text_format.BOLD + "\nSetting-up Limb darkening pars from input values" +_text_format.END)
            self.limb_darkening(q1=q1, q2=q2, verbose=verbose)


        return self._decorr_result
    

    def mask_points(self,lc_list=None,condition="lc['col0']<lc['col0'][10]",show_plot=False,verbose=True):
        """
        Function to mask points in the lightcurves based on a condition. These points are removed from the injested data
 
        Parameters:
        -----------
        lc_list: list of string, None, 'all';
            list of lightcurve filenames on which to mask points. Default is 'all' which masks all lightcurves in the object.
        condition: str, list of str, None;
            The condition is a string that can be evaluated as a python boolean expression based on any column of the lc e.g. "lc['col1']>1.0002" will mask points where flux is greater than 1.0002.
            The condition can be a combination of columns and logical operators e.g. "lc['col1']>1.0002 and lc['col0']<lc['col0'][10]" will mask points where flux is greater than 1.0002 and time is less than the 10th time point.
            Default is None which does nothing.
        show_plot: bool;
            set True to plot the data and show masked points.
        verbose: bool;
            Prints number of points that have been masked. Default is True
        """
        if self._masked_points:
            print("Data has already been masked. run `load_lightcurves()` again to reset.")
            return None
        
        if lc_list == None or lc_list == []: 
            print("lc_list is None: No lightcurve to mask.")
            return None
        if isinstance(lc_list, str) and (lc_list != 'all'): lc_list = [lc_list]
        if lc_list == "all": lc_list = self._names
        if condition is not None:
            if isinstance(condition,str): condition = [condition]*len(lc_list)
            assert len(condition) == len(lc_list), f"mask_points(): condition must be a string/None or list of strings/None with same length as lc_list but {len(condition)} given."

        if show_plot:
            n_data = len(lc_list)
            nrow_ncols = (1,1) if n_data==1 else (int(n_data/2), 2) if n_data%2==0 else (int(np.ceil(n_data/3)), 3)
            figsize=(8,5) if n_data==1 else (14,3.5*nrow_ncols[0])
            fig, ax = plt.subplots(nrow_ncols[0], nrow_ncols[1], figsize=figsize)
            ax = [ax] if n_data==1 else ax.reshape(-1)
            fig.suptitle("Masking Points",y=0.99)
            plt.subplots_adjust(hspace=0.3)

        for i,file in enumerate(lc_list):
            assert file in self._names, f"mask_points(): filename {file} not in loaded lightcurves."
            if condition[i] is None: 
                if verbose: print(f"No condition to mask points in {file}")
                continue

            lc = self._input_lc[file]
            mask = eval(condition[i])
            if show_plot:
                ax[i].set_title(f"Masked {np.sum(mask)} points in {file}")
                ax[i].plot(lc["col0"][ ~mask],lc["col1"][~mask], '.C0', ms=5, label="data")
                ax[i].plot(lc["col0"][mask],lc["col1"][mask], '.r', ms=5,label="masked")
                ax[i].legend()

            if verbose and (not show_plot): print(f"Masked {np.sum(~mask)} points in {file}")
            self._input_lc[file] = {k:lc[k][~mask] for k in lc.keys()}
        
        self._masked_points = True

    def clip_outliers(self, lc_list="all", clip=5, width=15, select_column=["col1"], niter=1, show_plot=False, verbose=True):

        """
        Remove outliers using a running median method. Points > clip*M.A.D are removed
        where M.A.D is the mean absolute deviation from the median in each window

        Parameters:
        ------------
        lc_list: list of string, None, 'all';
            list of lightcurve filenames on which perform outlier clipping. Default is 'all' which clips all lightcurves in the object.
        clip: int/float,list;
            cut off value above the median deviation. Default is 5. If list, must be same length as lc_list.
        width: int,list;
            Number of points in window to use when computing the running median. Must be odd. Default is 15. If list, must be same length as lc_list.
        select_column: list of str, str;
            list of column names on which to perform clipping. Default is only ["col1"] which is the flux column. 
            possible columns are: ["col1","col3","col4","col5","col6","col7","col8"]. "all" can be used to clip all columns. 
        niter: int,list;
            Number of iterations to perform clipping. Default is 1
        show_plot: bool;
            set True to plot the data and show clipped points.
        verbose: bool;
            Prints number of points that have been cut. Default is True

        """
        if self._clipped_data.flag:
            print("Data has already been clipped. run `load_lightcurves()` again to reset.")
            return None

        if lc_list == None or lc_list == []: 
            print("lc_list is None: No outlier clipping.")
            return None
        
        if isinstance(select_column,str):
            if select_column == "all": select_column = ["col1","col3","col4","col5","col6","col7","col8"]
            else: select_column = [select_column]
        if isinstance(select_column, list):
            for col in select_column: 
                    assert col in ["col1","col3","col4","col5","col6","col7","col8"],\
                            f'clip_outliers(): elements of select_column must be in ["col1","col3","col4","col5","col6","col7","col8"] but "{col}" given.'
        
        if isinstance(lc_list, str) and (lc_list != 'all'): lc_list = [lc_list]
        if lc_list == "all": lc_list = self._names

        if isinstance(width, int): width = [width]*len(lc_list)
        elif isinstance(width, list): 
            if len(width)==1: width = width*len(lc_list)
            for wid in width: assert isinstance(wid, int), f"clip_outliers(): width must be an int or list of int but {width=} given."
        else: _raise(TypeError, f"clip_outliers(): width must be an int or list of int but {clip=} given.")
            
        if isinstance(clip, (int,float)): clip = [clip]*len(lc_list)
        elif isinstance(clip, list): 
            if len(clip)==1: clip = clip*len(lc_list)
        else: _raise(TypeError, f"clip_outliers(): width must be an int/float or list of int/float but {clip=} given.")
            
        if isinstance(niter, (int)): niter = [niter]*len(lc_list)
        elif isinstance(niter, list): 
            if len(niter)==1: niter = niter*len(lc_list)
            for ni in niter: assert isinstance(ni, int), f"clip_outliers(): niter must be an int or list of int but {niter=} given."
        else: _raise(TypeError, f"clip_outliers(): width must be an int or list of int but {niter=} given.")
            

        assert len(width) == len(clip) == len(niter) == len(lc_list), f"clip_outliers(): width, clip, niter and lc_list must have same length but {len(width)=}, {len(clip)=} and {len(lc_list)=} given."

        if show_plot:
            n_data = len(lc_list)
            nrow_ncols = (1,1) if n_data==1 else (int(n_data/2), 2) if n_data%2==0 else (int(np.ceil(n_data/3)), 3)
            figsize=(8,5) if n_data==1 else (14,3.5*nrow_ncols[0])
            fig, ax = plt.subplots(nrow_ncols[0], nrow_ncols[1], figsize=figsize)
            ax = [ax] if n_data==1 else ax.reshape(-1)
            fig.suptitle("Outlier clipping",y=0.99)
            plt.subplots_adjust(hspace=0.3)

        for i,file in enumerate(lc_list):
            assert file in self._names, f"clip_outliers(): filename {file} not in loaded lightcurves."
            
            if width[i]%2 == 0: width[i] += 1   #if width is even, make it odd
            
            sel_cols = "c"+"".join([col[-1] for col in select_column])+":"  #get last index of select_column to get e.g. c135:
            if sel_cols == "c1345678:" : sel_cols = "ca:"
            self._clipped_data.config[self._names.index(file)] = f"{sel_cols}W{width[i]}C{clip[i]}n{niter[i]}"
            #join all last index of select_column to get e.g. c135


            thisLCdata = deepcopy(self._input_lc[file])
            ok      = np.ones(len(thisLCdata["col0"]), dtype=bool)  #initialize mask to all True, used to store indices of points that are not clipped
            ok_iter = np.ones(len(thisLCdata["col0"]), dtype=bool)  #initialize mask to all True, ok points for each iteration

            for col in select_column:
                if np.ptp(thisLCdata[col])==0.0: continue  #skip column if all values are the same (nothing to clip)
                for _ in range(niter[i]):
                    thisLCdata  = {k:v[ok_iter] for k,v in thisLCdata.items()}   #remove clipped points from previous iteration
                    _,_,clpd_mask = outlier_clipping(x=thisLCdata["col0"],y=thisLCdata[col],clip=clip[i],width=width[i],
                                                        verbose=False, return_clipped_indices=True)   #returns mask of the clipped points
                    ok_iter = ~clpd_mask     #invert mask to get indices of points that are not clipped
                    ok[ok] &= ok_iter        #update points in ok mask with the new iteration clipping
            if verbose and (not show_plot): print(f'\n{file}: Rejected {sum(~ok)}pts > {clip[i]:0.1f}MAD from the median of columns {select_column}')

            if show_plot:
                ax[i].set_title(f'{file}: Rejected {sum(~ok)}pts>{clip[i]:0.1f}MAD')
                ax[i].plot(self._input_lc[file]["col0"][ok],  self._input_lc[file]["col1"][ok], '.C0', ms=5)
                ax[i].plot(self._input_lc[file]["col0"][~ok], self._input_lc[file]["col1"][~ok], '.r', ms=5)

            
            #replace all columns of input file with the clipped data
            self._input_lc[file] = {k:v[ok] for k,v in self._input_lc[file].items()}

            #recompute rms estimate and multiplicative jitter
            self._rms_estimate[self._names.index(file)]  = np.std(np.diff(self._input_lc[file]["col1"]))/np.sqrt(2)
            self._jitt_estimate[self._names.index(file)] = np.sqrt(self._rms_estimate[self._names.index(file)]**2 - np.mean(self._input_lc[file]["col2"]**2))
            if np.isnan(self._jitt_estimate[self._names.index(file)]): self._jitt_estimate[self._names.index(file)] = 1e-20
        
        self._clipped_data.flag = True # SimpleNamespace(flag=True, width=width, clip=clip, lc_list=lc_list, config=conf)
        if show_plot: plt.tight_layout; plt.show()

    def lc_baseline(self, dcol0=None, dcol3=None, dcol4=None,  dcol5=None, dcol6=None, dcol7=None, 
                    dcol8=None, sin="n",grp=None, grp_id=None, gp="n", re_init=False,verbose=True):
        """
            Define baseline model parameters to fit for each light curve using the columns of the input data. `dcol0` refers to decorrelation setup for column 0, `dcol3` for column 3 and so on.
            Each baseline decorrelation parameter (dcolx) should be a list of integers specifying the polynomial order for column x for each light curve.
            e.g. Given 3 input light curves, if one wishes to fit a 2nd order trend in column 0 to the first and third lightcurves,
            then `dcol0` = [2, 0, 2].
            The decorrelation parameters depend on the columns (col) of the input light curve. Any desired array can be put in these columns to decorrelate against them. 
            Note that col0 is usually the time array.


            Parameters:
            -----------
            dcol0, dcol3,dcol4,dcol5,dcol6,dcol7,dcol8 : list of ints;
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
                if par=="gp": assert len(DA[par]) == self._nphot, f"lc_baseline(): parameter `{par}` must be a list of length {self._nphot} or str (if same is to be used for all LCs) or None."
                else: assert len(DA[par]) == self._nphot, f"lc_baseline(): parameter `{par}` must be a list of length {self._nphot} or int (if same degree is to be used for all LCs) or None (if not used in decorrelation)."

            for p in DA[par]:
                if par=="gp": assert p in ["y","n","ce"], f"lc_baseline(): gp must be a list of 'y', 'n', or 'ce' for each lc but {p} given."
                elif par=="sin": assert p in ["y","n"], f"lc_baseline(): sin must be a list of 'y' or 'n' for each lc but {p} given."
                else: assert isinstance(p, (int,np.int64)) and p<3, f"lc_baseline(): decorrelation parameters must be a list of integers (max int value = 2) but {type(p)} {p} given for {par}."

        DA["grp_id"] = list(np.arange(1,self._nphot+1)) if grp_id is None else grp_id

        self._bases = [ [DA["dcol0"][i], DA["dcol3"][i], DA["dcol4"][i], DA["dcol5"][i],
                        DA["dcol6"][i], DA["dcol7"][i], DA["dcol8"][i], DA["sin"][i], 
                        DA["grp"][i]] for i in range(self._nphot) ]

        self._groups    = DA["grp_id"]
        self._grbases   = DA["grp"]    #TODO: never used, remove instances of it
        self._useGPphot = DA["gp"]
        self._gp_lcs    = lambda : np.array(self._names)[np.array(self._useGPphot) != "n"]

        if verbose: _print_output(self,"lc_baseline")
        if np.all(np.array(self._useGPphot) == "n") or len(self._useGPphot)==0:        #if gp is "n" for all input lightcurves, run add_GP with None
            self.add_GP(None, verbose=False)

        #initialize other methods to empty incase they are not called/have not been called
        if not hasattr(self,"_lcspline") or re_init:      self.add_spline(None, verbose=False)
        if not hasattr(self,"_sine_dict") or re_init:     self.add_sinusoid(None, verbose=False)
        if not hasattr(self,"_ss") or re_init:            self.supersample(None, verbose=False)
        if not hasattr(self,"_config_par") or re_init:    self.planet_parameters(verbose=False)
        if not hasattr(self,"_ddfs") or re_init:          self.transit_depth_variation(verbose=False)
        if not hasattr(self,"_ttvs") or re_init:          self.transit_timing_variation(verbose=False)
        if not hasattr(self,"_PC_dict") or re_init:       self.setup_phasecurve(verbose=False)
        if not hasattr(self,"_contfact_dict") or re_init: self.contamination_factors(verbose=False)
        if not hasattr(self,"_ld_dict") or re_init:       self.limb_darkening(verbose=False)

    def supersample(self, lc_list=None,exp_time=0,verbose=True):
        """
        Supersample long intergration time of lcs in lc_list. This divides each exposure of the lc into  int(exp_time) subexposures to attain ~1min sampling. 
        e.g a lc with 30 minute exp_time will be divided into 30 subexposures of 1 minute each.

        Parameters
        ----------
        lc_list : list, str, optional
            list of lc files to supersample. set to "all" to use supersampling for all lc files. Default is None.

        exp_time : float, tuple, list, optional
            exposure time of each lc to supersample in minutes. if different for each lc in lc_list, give list with exp_time for each lc.
            Default is 0 for no exposure time, which means no supersampling

        verbose : bool, optional
            print output. Default is True.

        Examples
        --------
        To supersample a light curve that has a long cadence of 30mins (0.0208days) to 1 min, 30 points are needed to subdivide each exposure.
        >>> lc_obj.supersample(lc_list="lc1.dat",exp_time=30)
        """
                
        #default supersampling config -- None
        self._ss = [None]*self._nphot
        for i in range(self._nphot):
            self._ss[i] = supersampling()

        if lc_list is None:
            if verbose: print("No supersampling\n")
            return
        elif lc_list == "all":
            lc_list = self._names
        else:
            if isinstance(lc_list, str): lc_list = [lc_list]

        nlc_ss = len(lc_list)   #number of lcs to supersample
        for lc in lc_list:
            assert lc in self._names, f"supersample(): {lc} not in loaded lc files: {self._names}."

        if isinstance(exp_time, (int,float)): exp_time = [exp_time]*nlc_ss
        elif isinstance(exp_time, list): 
            if len(exp_time) == 1: exp_time = exp_time*nlc_ss
            assert len(exp_time)==nlc_ss, f"supersample(): exp_time must be a list of length {nlc_ss} or length 1 (if same is to be used for all lcs)."
        else: _raise(TypeError, f"supersample(): exp_time must be int/float/list but {exp_time} given.")   

        supersample_factor = [int(exp) for exp in exp_time]    #supersample to around 1minute

        for i,lc in enumerate(lc_list):
            ind = self._names.index(lc)  #index of lc in self._names
            self._ss[ind]= supersampling(exp_time=exp_time[i]/(60*24), supersample_factor=supersample_factor[i])

            if verbose: print(f"Supersampling {lc} with exp_time={exp_time[i]:.2f}mins each divided into {supersample_factor[i]} subexposures")
            
        if verbose: _print_output(self,"lc_baseline")

    
    def add_spline(self, lc_list= None, par = None, degree=3, knot_spacing=None,verbose=True):
        """
        add spline to fit correlation along 1 or 2 columns of the data. This splits the data at the defined knots interval and fits a spline to each section. 
        scipy's LSQUnivariateSpline() and LSQBivariateSpline() functions are used for 1D spline and 2D splines respectively.
        All arguments can be given as a list to specify config for each lc file in lc_list.

        Parameters
        ----------
        lc_list : list, str, optional
            list of lc files to fit a spline to. set to "all" to use spline for all lc files. Default is None for no splines.

        par : str,tuple,list, optional
            column of input data to which to fit the spline. must be one/two of ["col0","col3","col4","col5","col6","col7","col8"]. Default is None.
            Give list of columns if different for each lc file. e.g. ["col0","col3"] for spline in col0 for lc1.dat and col3 for lc2.dat. 
            For 2D spline for an lc file, use tuple of length 2. e.g. ("col0","col3") for simultaneous spline fit to col0 and col3.

        degree : int, tuple, list optional
            Degree of the smoothing spline. Must be 1 <= degree <= 5. Default is 3 for a cubic spline.
        
        knot_spacing : float, tuple, list
            distance between knots of the spline, in units of the desired column array. E.g 15 degrees for roll angle in CHEOPS data.
            If 'r' is given, the full range of the array is fit by a single spline of the specified order. this is useful if the range of the array varies for different datasets
        
        verbose : bool, optional
            print output. Default is True.

        Examples
        --------
        To use different spline configuration for 2 lc files: 2D spline for the first file and 1D for the second.
        >>> lc_obj.add_spline(lc_list=["lc1.dat","lc2.dat"], par=[("col3","col4"),"col4"], degree=[(3,3),2], knot_spacing=[(5,3),2])
        
        For same spline configuration for all loaded lc files
        >>> lc_obj.add_spline(lc_list="all", par="col3", degree=3, knot_spacing=5)
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
            self._lcspline[i].conf   = "None"

        if lc_list is None:
            if verbose: print("No spline\n")
            return
        elif lc_list == "all":
            lc_list = self._names
        else:
            if isinstance(lc_list, str): lc_list = [lc_list]
        
        nlc_spl = len(lc_list)   #number of lcs to add spline to
        for lc in lc_list:
            assert lc in self._names, f"add_spline(): {lc} not in loaded lc files: {self._names}."
        
        DA = locals().copy()

        for p in ["par","degree","knot_spacing"]:
            if DA[p] is None: DA[p] = [None]*nlc_spl
            if isinstance(DA[p], (str,int,float,tuple)): DA[p] = [DA[p]]*nlc_spl
            if isinstance(DA[p], list): assert len(DA[p])==nlc_spl, f"add_spline(): {p} must be a list of length {nlc_spl} or length 1 (if same is to be used for all lcs)."
            
            #check if inputs are valid
            for list_item in DA[p]:
                if p=="par":
                    if isinstance(list_item, str): assert list_item in ["col0","col3","col4","col5","col6","col7","col8",None],f'add_spline(): {p} must be in ["col0","col3","col4","col5"] but {list_item} given.'
                    if isinstance(list_item, tuple): 
                        for tup_item in list_item: assert tup_item in ["col0","col3","col4","col5","col6","col7","col8",None],f'add_spline(): {p} must be in ["col0","col3","col4","col5"] but {tup_item} given.'
                if p=="degree": 
                    assert isinstance(list_item, (int,tuple)),f'add_spline(): {p} must be an integer but {list_item} given.'
                    if isinstance(list_item, tuple):
                        for tup_item in list_item: assert isinstance(tup_item, int),f'add_spline(): {p} must be an integer but {tup_item} given.'

        for i,lc in enumerate(lc_list):
            ind = self._names.index(lc)    #index of lc in self._names
            par, deg, knots =  DA["par"][i], DA["degree"][i], DA["knot_spacing"][i]
            dim = 1 if isinstance(par,str) else len(par)
            assert dim <=2, f"add_spline(): dimension of spline must be 1 or 2 but {par} (dim {dim}) given for {lc}."
            if dim==2:   #if 2d spline 
                if isinstance(deg, int): deg = (deg,deg)  #if degree is int, make it a tuple
                if isinstance(knots, (int,float,str)): knots = (knots,knots)

            self._lcspline[ind].name   = lc
            self._lcspline[ind].dim    = dim
            self._lcspline[ind].par    = par
            self._lcspline[ind].use    = True if par else False
            self._lcspline[ind].deg    = deg
            self._lcspline[ind].knots  = knots
                
            if dim==1:
                assert knots=='r' or knots <= np.ptp(self._input_lc[lc][par]), f"add_spline():{lc} – knot_spacing must be <= the range of the column array but {knots} given for {par} with a range of {np.ptp(self._input_lc[lc][par])}."
                assert deg <= 5, f"add_spline():{lc} – degree must be <=5 but {deg} given for {par}."
                self._lcspline[ind].conf   = f"c{par[-1]}:d{deg}k{knots}"
            else:
                for j in range(2):
                    assert deg[j] <= 5, f"add_spline():{lc} – degree must be <=5 but {deg[j]} given for {par[j]}." 
                    assert knots[j]=='r' or knots[j] <= np.ptp(self._input_lc[lc][par[j]]), f"add_spline():{lc} – knot_spacing must be <= the range of the column array but {knots[j]} given for {par[j]} with range of {np.ptp(self._input_lc[lc][par[j]])}."
                self._lcspline[ind].conf   = f"c{par[0][-1]}:d{deg[0]}k{knots[0]}|c{par[1][-1]}:d{deg[1]}k{knots[1]}"

            if verbose: print(f"{lc} – degree {deg} spline to fit {par}: knot spacing={knots} --> [{self._lcspline[ind].conf}]") 
        
        if verbose: 
            print("\n")
            _print_output(self,"lc_baseline")


    def add_sinusoid(self, lc_list=None, trig='sin', n=1, par="col0", Amp=0,  P=2*np.pi, x0=0, verbose=True):
        """
        Add sinusoid to fit correlation along a column of the data. This fits a sinusoid to the column data using the given period, amplitude and zero phase.
        sinusoid = ∑[n=1,2,3] Amp*sin(n*2𝜋/P*(x-x0)) for trig='sin'. similar for 'cos' while 'sincos' is sinusoid = ∑  Amp*sin(n*2𝜋/P*(x-x0)) + Amp*cos(n*2𝜋/P*(x-x0))
        All arguments can be given as a list to specify config for each lc file in lc_list.
        To directly fit the trig function to the column array i.e sin(colx), set x0=0, and P=2*np.pi.

        Parameters
        ----------
        lc_list : list, str, optional
            list of lc files to fit a sinusoid to. set to "all" to use sinusoid for all lc files. Default is None for no sinusoid.
        trig : str, list, optional
            trigonometric function to fit. must be one of ['sin','cos','sincos']. Default is 'sin'. Give list of trig functions if different for each lc file. e.g. ["sin","cos"] for sin(x) for lc1.dat and cos(x) for lc2.dat.
        n : int, tuple, list, optional
            number of harmonics of the sinusoid to fit. Default is 1 for only sin(x) term, 2 for sin(2x), .... max value is 3
        par : str, list, optional
            column of input data representing the independent variable x of the sinusoid. must be one of ["col0","col3","col4","col5","col6","col7","col8"]. Default is "col0".
            Give list of columns if different for each lc file. e.g. ["col0","col3"] for sinusoid in col0 for lc1.dat and col3 for lc2.dat.
        P : float, tuple, list, optional
            period of the sinusoid in the same units as the column array specified in `par`. Default is 2𝜋.
        amp : float, tuple, list, optional
            amplitude of the sinusoid function in ppm. Default is None.
        x0: float, tuple, list, optional
            zero phase of the sinusoid in units of column array specified in `par`. Default is 0.
        verbose : bool, optional
            print output. Default is True.

        Examples
        --------
        To use different sinusoid configuration for 2 lc files
        >>> lc_obj.add_sinusoid(lc_list=["lc1.dat","lc2.dat"], par=["col0","col4"], P=[1,2], amp=[0.1,0.2], phase=[0,1])
        """
        DA = locals().copy()
        # function to set default sinusoid config -- None
        init_sine = lambda name,fit : SimpleNamespace(name=name, fit=fit, trig=None, n=1, par="col0", npars=3, nfree=0, 
                                                        Amp=_param_obj.from_tuple(0), P=_param_obj.from_tuple(0), 
                                                        x0=_param_obj.from_tuple(0))
        for i in range(self._nphot): self._bases[i][7]="n"

        if lc_list is None or lc_list==[]:
            self._sine_dict = {"same":init_sine("same","same")}#{k:None for k in self._names}
            if verbose: print("No sinusoid\n")
            return

        elif isinstance(lc_list, str):
            if lc_list in ["all","same","filt"]:
                for i in range(self._nphot): self._bases[i][7]="y"

            if lc_list == "all": 
                sin_names = self._names
                if verbose: print("fitting individual sinusoids to each LC.")
            elif lc_list == "same": 
                sin_names = ["same"]
                if verbose: print("fitting same sinusoid to all LCs.")
            elif lc_list == "filt": 
                sin_names = list(self._filnames) 
                if verbose: print("fitting same sinusoids to each filter.")
            else:
                if lc_list in self._names:
                    self._bases[self._names.index(lc_list)][7]="y"
                    sin_names = [lc_list] 
                    lc_list = "slct"  # one selected lc 
                elif lc_list in self._filnames:
                    for i in np.where(np.array(self._filters)==lc_list)[0]: self._bases[i][7]="y"
                    sin_names = [lc_list]
                    lc_list = "filt"
                else:
                    _raise(ValueError, f"add_sinusoid(): {lc_list} not in loaded lc files: {self._names} or filters: {self._filnames}.")
        
        elif isinstance(lc_list, list):
            for lc in lc_list: 
                assert lc in self._names or lc in self._filnames, f"add_sinusoid(): {lc} not in loaded lc files: {self._names} or filters: {self._filnames}." 
            if all([lc in self._names for lc in lc_list]): 
                for i in [self._names.index(lc) for lc in lc_list]: self._bases[i][7]="y"
                sin_names = lc_list
                lc_list = "slct"    #list of selected lcs
            elif all([lc in self._filnames for lc in lc_list]):
                for i in np.where(np.array(self._filters)==lc_list)[0]: self._bases[i][7]="y"
                sin_names = lc_list
                lc_list = "filt"
            else:
                _raise(ValueError, f"add_sinusoid(): elements of lc_list must be either all LC filenames or all filter names but {lc_list=} given.")

        else:
            raise TypeError(f"add_sinusoid(): lc_list must be either a str or list but {lc_list} given")

        nLC_sin = len(sin_names)
        #check that inputs are valid
        for p in ["trig","n","par","Amp","P","x0"]:
            if DA[p] is None: DA[p] = [None]*nLC_sin
            if isinstance(DA[p], (str,int,float,tuple)): DA[p] = [DA[p]]*nLC_sin
            if isinstance(DA[p], list): assert len(DA[p])==nLC_sin, f"add_sinusoid(): {p} must be a list of length {nLC_sin} to specify value for each lc/filter or length 1 to use same value for all lcs/filters)."
        
            for list_item in DA[p]:
                if p=="trig": assert list_item in ["sin","cos","sincos"], f"add_sinusoid(): {p} must be in ['sin','cos','sincos'] but {list_item} given."
                if p=="n": assert list_item in [1,2,3], f"add_sinusoid(): {p} must be an integer (<= 3) but {list_item} given."
                if p=="par": assert list_item in ["col0","col3","col4","col5","col6","col7","col8"], f"add_sinusoid(): {p} must be in ['col0','col3','col4','col5','col6','col7','col8'] but {list_item} given."
                if p=="Amp": 
                    if isinstance(list_item, (int,float)): assert list_item >= 0, f"add_sinusoid(): {p} must be a positive float/int/tuple but {list_item} given."
                    if isinstance(list_item, tuple): 
                        assert len(list_item) in [2,3], f"add_sinusoid(): {p} must be a tuple of length 2/3 but {list_item} given."
                        if len(list_item)==3: assert list_item[0]<list_item[1]<list_item[2], f"add_sinusoid(): uniform prior for {p} must be in increasing order but {list_item} given."
                if p in ["P", "x0"]: 
                    assert isinstance(list_item, (int,float,tuple)) or list_item is None, f"add_sinusoid(): {p} must be a float/int/tuple or None but {list_item} given."
                    if p=="P":
                        if isinstance(list_item, (int,float)): assert list_item > 0, f"add_sinusoid(): {p} must be a positive float/int/tuple but {list_item} given."
                    if isinstance(list_item, tuple):
                        assert len(list_item) in [2,3], f"add_sinusoid(): {p} must be a tuple of length 2/3 but {list_item} given."
                        if len(list_item)==3: assert list_item[0]<list_item[1]<list_item[2], f"add_sinusoid(): uniform prior for {p} must be in increasing order but {list_item} given."
        
        #initialize sinusoid to None for all lcs or all filters
        # self._sine_dict = {k:None for k in self._names} if lc_list=="slct" else {k:None for k in self._filnames} if lc_list=="filt" else {k:None for k in sin_names}
        self._sine_dict = {lc:init_sine(lc,"slct") for lc in self._names} if lc_list=="slct" else {lc:init_sine(lc,"filt") for lc in self._filnames} if lc_list=="filt" else {lc:init_sine(lc,"same") for lc in sin_names}
        
        for i,lc in enumerate(sin_names):
            self._sine_dict[lc]        = SimpleNamespace()
            self._sine_dict[lc].name   = lc
            self._sine_dict[lc].fit    = lc_list
            self._sine_dict[lc].trig   = DA["trig"][i]
            self._sine_dict[lc].n      = DA["n"][i]
            self._sine_dict[lc].par    = DA["par"][i]
            self._sine_dict[lc].Amp    = _param_obj.from_tuple(DA["Amp"][i],user_input=DA["Amp"][i],func_call="add_sinusoid():")
            self._sine_dict[lc].P      = _param_obj.from_tuple(DA["P"][i],lo=1e-5,user_input=DA["P"][i],func_call="add_sinusoid():")
            self._sine_dict[lc].x0     = _param_obj.from_tuple(DA["x0"][i],user_input=DA["x0"][i],func_call="add_sinusoid():")
            self._sine_dict[lc].npars  = 0     #number of total parameters for this sinusoid
            self._sine_dict[lc].nfree  = 0     #number of free parameters for this sinusoid
            for p in ["Amp","P","x0"]:
                if p=="Amp":   # first count number of amplitudes for the trigs
                    self._sine_dict[lc].npars += 1  if DA["trig"][i] in ["sin","cos"] else 2
                    self._sine_dict[lc].npars *= self._sine_dict[lc].n
                    if isinstance(DA[p][i], tuple): self._sine_dict[lc].nfree = self._sine_dict[lc].npars
                else: 
                    self._sine_dict[lc].npars += 1
                    if isinstance(DA[p][i], tuple): self._sine_dict[lc].nfree += 1

        if verbose: 
            _print_output(self,"lc_baseline")
            _print_output(self,"sinusoid")

    def add_GP(self ,lc_list=None, par=["col0"], kernel=["mat32"], operation=[""],
                 amplitude=[], lengthscale=[], verbose=True):
        """  
            Define GP parameters for each lc.
            The GP parameters, amplitude in ppm  and lengthscale in unit of the desired column can be defined in following ways:
            * fixed value as float or int, e.g amplitude = 2
            * free parameter with lognormal prior given as tuple of len 2, e.g. amplitude = (2, 1)
            * free parameters with loguniform prior interval and initial value given as tuple of length 3, e.g. amplitude = (1,2,5) with 2 being the initial value.
            
            Here the amplitude corresponds to the standard deviation of the noise process and the lengthscale corresponds to the characteristic timescale of the noise process.
            
            For the celerite sho kernel, the quality factor Q has been fixed to 1/sqrt(2) which is commonly used to model stellar oscillations/granulation noise (eqn 24 celerite paper).
            this lengthscale here is the undamped period of the oscillator.

            For the cosine kernel in George, lengthscale is the period. This kernel should probably always be multiplied by a stationary kernel (e.g. ExpSquaredKernel) to allow quasi-periodic variations.

            Parameters:
            -----------
            lc_list : str, list;
                list of lc files to add GP to. Default is None for no GP. if "all" is given, GP is added to all lc files where gp use has been indicated in ``lc_baseline()``. 
                If "same" is given, a global (same) GP is used for all indicated lc files in ``lc_baseline()``.
            par : str, tuple, list;
                column of the input data to use as the GP independent variable. a list is expected if different columns are to be used for the lc files given in lc_list.
                to use 2 different kernels on a single lc file, give column name for each kernel as a tuple of length 2. 
                e.g. lc_list=["lc1.dat","lc2.dat"], par = [("col0","col0"),"col3"] to use col0 for both kernels of lc1, and col3 for lc2.
            kernel : str, tuple, list;
                kernel to use for the GP. Must be one of ["mat32","mat52","exp","expsq","cos"] if George package is selected  and one of ["real","mat32","sho","cos"] if using Celerite package
                A list is expected if different kernels are to be used for the lc files given in lc_list.
                to use 2 different kernels on a single lc file, give kernel name for each kernel as a tuple of length 2.
                e.g. lc_list=["lc1.dat","lc2.dat"], kernel = [("mat32","expsq"),"exp"] to use mat32 and expsq for lc1, and exp for lc2.
            operation : str, tuple, list;
                operation to combine 2 kernels. Must be one of ["+","*"]. Default is "" for no combination.
            amplitude : float, tuple, list;
                amplitude of the GP kernel in ppm. Must be list containing int/float or tuple of length 2 or 3
            lengthscale : float, tuple, list;
                lengthscale of the GP kernel in units of the column array specified in `par`. Must be list containing int/float or tuple of length 2 or 3
            verbose : bool;
                print output. Default is True.        
        """
        # supported 2-hyperparameter kernels
        george_allowed   = dict(kernels = ["mat32","mat52","exp","expsq","cos"],columns= ["col0","col3","col4","col5","col6","col7","col8"])
        celerite_allowed = dict(kernels = ["real","mat32","sho","cos"], columns= ["col0","col3","col4","col5","col6","col7","col8"])

        self._GP_dict  = {}
        self._sameLCgp  = SimpleNamespace(flag = False, first_index =None) #flag to indicate if same GP is to be used for all lcs

        if lc_list is None or lc_list == []:
            if self._nphot>0:
                if len(self._gp_lcs())>0: print(f"\nWarning: GP was expected for the following lcs {self._gp_lcs()} \nMoving on ...")
                if verbose:_print_output(self,"gp")
            return
        elif isinstance(lc_list, str): 
            if lc_list == "same":
                self._sameLCgp.flag        = True
                self._sameLCgp.first_index = self._names.index(self._gp_lcs()[0])
            if lc_list in ["all","same"]:
                lc_list = self._gp_lcs()
            else: lc_list=[lc_list]


        for lc in self._gp_lcs(): assert lc in lc_list,f"add_GP(): GP was expected for {lc} but was not given in lc_list."
        for lc in lc_list: 
            assert lc in self._names,f"add_GP(): {lc} not in loaded lc files."
            # assert lc in self._gp_lcs(),f"add_GP(): GP was not expected for {lc} but was given in lc_list. Use `._useGPphot` attribute to modify this list with 'y','ce' or 'n' for each loaded lc"
            if lc not in self._gp_lcs():
                self._useGPphot[self._names.index(lc)] = "ce"
                print(f"add_GP(): GP was not expected for {lc} but was given in lc_list, but now adding 'ce' GP for this lc.")

        lc_ind = [self._names.index(lc) for lc in lc_list]
        gp_pck = [self._useGPphot[i] for i in lc_ind]   #gp_pck is a list of "y" or "ce" for each lc in lc_list

        DA = locals().copy()
        _  = [DA.pop(item) for item in ["self","verbose"]]

        for p in ["par","kernel","operation","amplitude","lengthscale"]:
            if isinstance(DA[p], (str,int,float,tuple)): DA[p] = [DA[p]]
            if isinstance(DA[p], list): 
                if self._sameLCgp.flag: 
                    assert len(DA[p])==1 or len(DA[p])==len(lc_list), f"add_GP(): {p} must be a list of length {len(lc_list)} or length 1 (if same is to be used for all LCs)."
                    if len(DA[p])==2: assert DA[p][0]==DA[p][1], f"add_GP(): {p} must be same for all lc files if sameGP is used."
                if len(DA[p])==1: DA[p] = DA[p]*len(lc_list)

            assert len(DA[p])==len(lc_list), f"add_GP(): {p} must be a list of length {len(lc_list)} or length 1 (if same is to be used for all LCs)."
            
        for p in ["par","kernel","operation","amplitude","lengthscale"]:
            #check if inputs for p are valid
            for i,list_item in enumerate(DA[p]):
                if p=="par":
                    if isinstance(list_item, str): 
                        if gp_pck[i]=="y":  assert list_item in george_allowed["columns"],  f'add_GP(): inputs of {p} must be in {george_allowed["columns"]}   but {list_item} given.'
                        if gp_pck[i]=="ce": assert list_item in celerite_allowed["columns"],f'add_GP(): inputs of {p} must be in {celerite_allowed["columns"]} but {list_item} given.'
                        DA["operation"][i] = ""
                    elif isinstance(list_item, tuple): 
                        assert len(list_item)==2,f'add_GP(): max of 2 gp kernels can be combined, but {list_item} given in {p}.'
                        assert DA["operation"][i] in ["+","*"],f'add_GP(): operation must be one of ["+","*"] to combine 2 kernels but {DA["operation"][i]} given.'
                        for tup_item in list_item: 
                            if gp_pck[i]=="y":  assert tup_item in george_allowed["columns"],  f'add_GP(): {p} must be in {george_allowed["columns"]}   but {tup_item} given.'
                            if gp_pck[i]=="ce": assert tup_item in celerite_allowed["columns"],f'add_GP(): {p} must be in {celerite_allowed["columns"]} but {tup_item} given.'
                        # assert that a tuple of length 2 is also given for kernels, amplitude and lengthscale.
                        for chk_p in ["kernel","amplitude","lengthscale"]:
                            assert isinstance(DA[chk_p][i], tuple) and len(DA[chk_p][i])==2,f'add_GP(): expected tuple of len 2 for {chk_p} element {i} but {DA[chk_p][i]} given.'
                            
                    else: _raise(TypeError, f"add_GP(): elements of {p} must be a tuple of length 2 or str but {list_item} given.")
                
                if p=="kernel":
                    if isinstance(list_item, str): 
                        if gp_pck[i]=="y":  assert list_item in george_allowed["kernels"],  f'add_GP(): {p} must be one of {george_allowed["kernels"]}   but {list_item} given.'
                        if gp_pck[i]=="ce": assert list_item in celerite_allowed["kernels"],f'add_GP(): {p} must be one of {celerite_allowed["kernels"]} but {list_item} given.'
                    elif isinstance(list_item, tuple):
                        for tup_item in list_item: 
                            if gp_pck[i]=="y":  assert tup_item in george_allowed["kernels"],  f'add_GP(): {p} must be one of {george_allowed["kernels"]}   but {tup_item} given.'
                            if gp_pck[i]=="ce": assert tup_item in celerite_allowed["kernels"],f'add_GP(): {p} must be one of {celerite_allowed["kernels"]} but {tup_item} given.'
                    else: _raise(TypeError, f"add_GP(): elements of {p} must be a tuple of length 2 or str but {list_item} given.")

                if p=="operation":
                    assert list_item in ["+","*",""],f'add_GP(): {p} must be one of ["+","*",""] but {list_item} given.'

                if p in ["amplitude", "lengthscale"]:
                    if isinstance(list_item, (int,float)): pass
                    elif isinstance(list_item, tuple):
                        if isinstance(DA["par"][i],tuple):
                            for tup in list_item:
                                if isinstance(tup, (int,float)): pass
                                elif isinstance(tup, tuple): 
                                    assert len(tup) in [2,3,4],f'add_GP(): {p} must be a float/int or tuple of length 2/3/4 but {tup} given.'
                                    if len(tup)==3: assert tup[0]<tup[1]<tup[2],f'add_GP(): uniform prior for {p} must follow (min, start, max) but {tup} given.'
                                else: _raise(TypeError, f"add_GP(): elements of {p} must be a tuple of length 2/3 or float/int but {tup} given.")
                        else:
                            assert len(list_item) in [2,3],f'add_GP(): {p} must be a float/int or tuple of length 2/3/4 but {tup} given.'
                            if len(list_item)==3: assert list_item[0]<list_item[1]<list_item[2],f'add_GP(): uniform prior for {p} must follow (min, start, max) but {list_item} given.'
                    else: _raise(TypeError, f"add_GP(): elements of {p} must be a tuple of length 2/3 or float/int but {list_item} given.")


        #setup parameter objects
        for i,lc in enumerate(lc_list):
            self._GP_dict[lc] = {}
            ngp = 2 if isinstance(DA["kernel"][i],tuple) else 1
            self._GP_dict[lc]["ngp"] = ngp
            self._GP_dict[lc]["op"]  = DA["operation"][i]

            if self._GP_dict[lc]["op"] == "*" and self._GP_dict[lc]["ngp"] == 2:
                assert DA["amplitude"][i][1] == 1, f"add_GP(): for multiplication of 2 kernels, the second amplitude must be fixed to 1 but {DA['amplitude'][i][1]} given."

            for p in ["amplitude", "lengthscale"]:
                for j in range(ngp):
                    if ngp==1: 
                        v = DA[p][i]
                        this_kern, this_par = DA["kernel"][i], DA["par"][i]
                    else:
                        v = DA[p][i][j]
                        this_kern, this_par = DA["kernel"][i][j], DA["par"][i][j]

                    if isinstance(v, (int,float)):
                        self._GP_dict[lc][p+str(j)]     = _param_obj(to_fit="n", start_value=v,step_size=0,
                                                                        prior="n", prior_mean=v, prior_width_lo=0,
                                                                        prior_width_hi=0, bounds_lo=0.01, bounds_hi=0,
                                                                        user_input=v, user_data = [this_kern, this_par])
                    elif isinstance(v, tuple):
                        if len(v)==2:
                            steps = 0 if (self._sameLCgp.flag and i!=0) else 0.1*v[1]   #if sameLCgp is set, only first pars will jump and be used for all rvs
                            self._GP_dict[lc][p+str(j)] = _param_obj(to_fit="y", start_value=v[0],step_size=steps,prior="p", 
                                                                        prior_mean=v[0], prior_width_lo=v[1], prior_width_hi=v[1], 
                                                                        bounds_lo=v[0]-10*v[1], bounds_hi=v[0]+10*v[1],     #10sigma cutoff
                                                                        user_input=v, user_data=[this_kern, this_par])
                        elif len(v)==3:
                            steps = 0 if (self._sameLCgp.flag and i!=0) else min(0.001,0.001*np.ptp(v))
                            self._GP_dict[lc][p+str(j)] = _param_obj(to_fit="y", start_value=v[1],step_size=steps,
                                                                        prior="n", prior_mean=v[1], prior_width_lo=0,
                                                                        prior_width_hi=0, bounds_lo=v[0] if v[0]>0 else 0.007, bounds_hi=v[2],
                                                                        user_input=v, user_data=[this_kern, this_par])
                        elif len(v)==4:
                            steps = 0 if (self._sameLCgp.flag and i!=0) else 0.1*v[3]   #if sameLCgp is set, only first pars will jump and be used for all rvs
                            self._GP_dict[lc][p+str(j)] = _param_obj(to_fit="y", start_value=v[2],step_size=steps,prior="p", 
                                                                        prior_mean=v[2], prior_width_lo=v[3], prior_width_hi=v[3], 
                                                                        bounds_lo=v[0], bounds_hi=v[1], 
                                                                        user_input=v, user_data=[this_kern, this_par])
                    else: _raise(TypeError, f"add_GP(): elements of {p} must be a tuple of length 2/3/4 or float/int but {v} given.")

        if verbose: _print_output(self,"gp")
    
    
    def planet_parameters(self, RpRs=0, Impact_para=0, rho_star=None, Duration=None, T_0=0, Period=0, 
                            Eccentricity=0, omega=90, K=0, verbose=True):
        """
            Define parameters and priors of model parameters. By default, the parameters are fixed to the given values. 
            The parameters can be defined in following ways:
            
            * fixed value as float or int, e.g Period = 3.4
            * free parameter with gaussian prior given as tuple of len 2, e.g. T_0 = (5678, 0.1)
            * free parameters with uniform prior interval and initial value given as tuple of length 3, e.g. RpRs = (0,0.1,0.2) with 0.1 being the initial value.
            if uniform is specified for rho_star or Duration, loguniform is used instead following literature convention (https://iopscience.iop.org/article/10.3847/1538-3881/ac7f2f).

            Parameters:
            -----------
            RpRs : float, tuple;
                Ratio of planet to stellar radius. Default is 0.
            Impact_para : float, tuple;
                Impact parameter of the transit. Default is 0.
            rho_star : float, tuple;
                density of the star in g/cm^3. Default is None.
            Duration : float, tuple;
                Duration of the transit in days. Default is None.
            T_0 : float, tuple;
                Mid-transit time in days. Default is 0.
            Period : float, tuple;
                Orbital period of the planet in days. Default is 0.
            Eccentricity : float, tuple;
                Eccentricity of the orbit. Default is 0.
            omega : float, tuple;
                Argument of periastron om degrees. Default is 90.
            K : float, tuple;
                Radial velocity semi-amplitude in same unit as the data. Default is 0.
            verbose : bool;
                print output. Default is True.
        """
        if rho_star is None and Duration is None: rho_star = 0
        if self._nplanet > 1: 
            rho_star = rho_star if rho_star is not None else 0
            assert isinstance(rho_star, (int,float,tuple)), "planet_parameters(): rho_star must be a float/int/tuple for multiplanet systems."
            assert Duration==None, "planet_parameters(): Duration must be None for multiplanet systems, since transit model uses rho_star."

        DA = deepcopy(locals())         #dict of arguments (DA)
        _ = DA.pop("self")                            #remove self from dictionary
        _ = DA.pop("verbose")
        if Duration==None: _ = DA.pop("Duration")
        if rho_star==None: _ = DA.pop("rho_star")

        #sort to specific order
        # key_order = ["RpRs","Impact_para","rho_star", "T_0", "Period", "Eccentricity","omega", "K"]
        # DA = {key:DA[key] for key in key_order if key in DA} 
            
        self._TR_RV_parnames  = [nm for nm in DA.keys()] 
        self._config_par = {}

        for par in DA.keys():
            if isinstance(DA[par], (float,int,tuple)): DA[par] = [DA[par]]*self._nplanet
            if isinstance(DA[par], list): assert len(DA[par])==self._nplanet, f"planet_parameters: {par} must be a list of length {self._nplanet} or float/int/tuple."

        for n in range(self._nplanet):    #n is planet number
            self._config_par[f"pl{n+1}"] = {}

            for par in DA.keys():
                if par == "rho_star":    lo_lim,up_lim = 0,8
                elif par in ["Eccentricity","Duration"]: lo_lim,up_lim = 0,1
                elif par == "RpRs": lo_lim, up_lim = -1,1
                elif par == "Impact_para": lo_lim,up_lim = 0,2
                elif par == "omega":       lo_lim,up_lim = 0,360

                #fitting parameter
                if isinstance(DA[par][n], tuple):
                    #gaussian       
                    if len(DA[par][n]) == 2:
                        if par in ["T_0","rho_star","Duration","Period","Impact_para","K","Eccentricity"]: 
                            lo_lim = DA[par][n][0]-20*DA[par][n][1] if par=="T_0" else max(0,DA[par][n][0]-20*DA[par][n][1])    #lowlim is mean-20*sigma
                            up_lim = DA[par][n][0]+20*DA[par][n][1]    #uplim is mean+20*sigma   
                        DA[par][n] = _param_obj(to_fit="y", start_value=DA[par][n][0], step_size=0.1*DA[par][n][1],
                                                prior="p", prior_mean=DA[par][n][0],  
                                                prior_width_lo=DA[par][n][1], prior_width_hi=DA[par][n][1], 
                                                bounds_lo=lo_lim, bounds_hi=up_lim,user_input=DA[par][n])
                    #uniform
                    elif len(DA[par][n]) == 3: 
                        DA[par][n] = _param_obj(*["y", DA[par][n][1], min(0.001,0.001*np.ptp(DA[par][n])), "n", DA[par][n][1],
                                                        0, 0, DA[par][n][0], DA[par][n][2],DA[par][n]])
                    #trunc gaussian       
                    elif len(DA[par][n]) == 4:  
                        DA[par][n] = _param_obj(to_fit="y", start_value=DA[par][n][2], step_size=0.1*DA[par][n][3],
                                                prior="p", prior_mean=DA[par][n][2],  
                                                prior_width_lo=DA[par][n][3], prior_width_hi=DA[par][n][3], 
                                                bounds_lo=DA[par][n][0], bounds_hi=DA[par][n][1],user_input=DA[par][n])
                    
                    else: _raise(ValueError, f"planet_parameters: length of tuple {par} is {len(DA[par][n])} but it must be 2 for gaussian or 3 for uniform priors")
                #fixing parameter
                elif isinstance(DA[par][n], (int, float)):
                    DA[par][n] = _param_obj(*["n", DA[par][n], 0.00, "n", DA[par][n], 0, 0, 0, 0,DA[par][n]])

                else: _raise(TypeError, f"planet_parameters(): {par} for planet{n} must be one of [tuple(of len 2 or 3), int, float] but is {type(DA[par][n])}")

                self._config_par[f"pl{n+1}"][par] = DA[par][n]      #add to object
        
        if verbose: _print_output(self,"planet_parameters")

        if self._show_guide: print("\nNext: use method transit_depth_variation` to include variation of RpRs for the different filters or \n`setup_phasecurve` to fit the occultation depth or \n`limb_darkening` for fit or fix LDCs or `contamination_factors` to add contamination.")


    def update_planet_parameters(self, RpRs=None, Impact_para=None, rho_star=None, Duration=None, T_0=None, Period=None, 
                                    Eccentricity=None, omega=None, K=None, verbose=True):
        """
            Update parameters and priors of model parameters. By default, the parameters are all set to None for no update on them.
            The parameters to update can be defined in following ways:
            
            * fixed value as float or int, e.g Period = 3.4
            * free parameter with gaussian prior given as tuple of len 2, e.g. T_0 = (5678, 0.1)
            * free parameters with uniform prior interval and initial value given as tuple of length 3, e.g. RpRs = (0,0.1,0.2) with 0.1 being the initial value.

            Parameters:
            -----------
            RpRs : float, tuple;
                Ratio of planet to stellar radius.
            Impact_para : float, tuple;
                Impact parameter of the transit.
            rho_star : float, tuple;
                density of the star in g/cm^3.
            Duration : float, tuple;
                Duration of the transit in days. Default is None.
            T_0 : float, tuple;
                Mid-transit time in days.
            Period : float, tuple;
                Orbital period of the planet in days.
            Eccentricity : float, tuple;
                Eccentricity of the orbit.
            omega : float, tuple;
                Argument of periastron.
            K : float, tuple;
                Radial velocity semi-amplitude in data unit.
            verbose : bool;
                print output. Default is True.
        """
        
        DA = locals().copy()         #dict of arguments (DA)
        _ = DA.pop("self")                            #remove self from dictionary
        _ = DA.pop("verbose")

        if "rho_star" not in self._config_par[f"pl{1}"].keys():
            assert rho_star==None, "update_planet_parameters(): cannot update 'rho_star' since 'Duration' selected in .planet_parameters()"
            _ = DA.pop("rho_star")
        if "Duration" not in self._config_par[f"pl{1}"].keys():
            assert Duration==None, "update_planet_parameters(): cannot update 'Duration' since 'rho_star' selected in .planet_parameters()"
            _ = DA.pop("Duration")

        rm_par= []
        for par in DA.keys():
            if DA[par] == None: rm_par.append(par)
        _ = [DA.pop(p) for p in rm_par]
        

        for par in DA.keys():
            if isinstance(DA[par], (float,int,tuple)): DA[par] = [DA[par]]*self._nplanet
            if isinstance(DA[par], list): assert len(DA[par])==self._nplanet, f"planet_parameters: {par} must be a list of length {self._nplanet} or float/int/tuple."

        for n in range(self._nplanet):    #n is planet number

            for par in DA.keys():
                if par in ["RpRs", "Eccentricity,Duration"]: lo_lim,up_lim = 0,1
                elif par == "rho_star":    lo_lim,up_lim = 0,8
                elif par == "Impact_para": lo_lim,up_lim = 0,2
                elif par == "omega":       lo_lim,up_lim = 0,360

                #fitting parameter
                if isinstance(DA[par][n], tuple):
                    #gaussian       
                    if len(DA[par][n]) == 2:
                        if par in ["T_0","rho_star","Duration","Period","Impact_para","K","Eccentricity"]: 
                            lo_lim = DA[par][n][0]-20*DA[par][n][1] if par=="T_0" else max(0,DA[par][n][0]-20*DA[par][n][1])    #lowlim is mean-20*sigma
                            up_lim = DA[par][n][0]+20*DA[par][n][1]    #uplim is mean+20*sigma   
                        DA[par][n] = _param_obj(to_fit="y", start_value=DA[par][n][0], step_size=0.1*DA[par][n][1],
                                                prior="p", prior_mean=DA[par][n][0],  
                                                prior_width_lo=DA[par][n][1], prior_width_hi=DA[par][n][1], 
                                                bounds_lo=lo_lim, bounds_hi=up_lim, user_input=DA[par][n])
                    #uniform
                    elif len(DA[par][n]) == 3: 
                        DA[par][n] = _param_obj(*["y", DA[par][n][1], min(0.001,0.001*np.ptp(DA[par][n])), "n", DA[par][n][1],
                                                        0, 0, DA[par][n][0], DA[par][n][2],DA[par][n]])
                    #trunc gaussian       
                    elif len(DA[par][n]) == 4:  
                        DA[par][n] = _param_obj(to_fit="y", start_value=DA[par][n][2], step_size=0.1*DA[par][n][3],
                                                prior="p", prior_mean=DA[par][n][2],  
                                                prior_width_lo=DA[par][n][3], prior_width_hi=DA[par][n][3], 
                                                bounds_lo=DA[par][n][0], bounds_hi=DA[par][n][1])
                    else: _raise(ValueError, f"update_planet_parameters(): length of tuple {par} is {len(DA[par][n])} but it must be 2 for gaussian or 3 for uniform priors")
                #fixing parameter
                elif isinstance(DA[par][n], (int, float)):
                    DA[par][n] = _param_obj(*["n", DA[par][n], 0.00, "n", DA[par][n], 0, 0, 0, 0,DA[par][n]])

                else: _raise(TypeError, f"update_planet_parameters(): {par} for planet{n} must be one of [tuple(of len 2,3,or 4), int, float] but is {type(DA[par][n])}")

                self._config_par[f"pl{n+1}"][par] = DA[par][n]      #add to object
        
        if verbose: _print_output(self,"planet_parameters")

        if self._show_guide: print("\nNext: use method transit_depth_variation` to include variation of RpRs for the different filters or \n`setup_phasecurve` to fit the occultation depth or \n`limb_darkening` for fit or fix LDCs or `contamination_factors` to add contamination.")


    def transit_depth_variation(self, ddFs="n", dRpRs=(-0.5,0,0.5), divwhite="n",  verbose=True):
        """
            Include transit depth variation between the different lcs or lc groups. Note: "RpRs" must be fixed to a reference value  in `.planet_parameters()` and not a jump parameter .
            transit depth variation is calculated as the deviation of each group's transit depth from the fixed RpRs
            
            Parameters:
            -----------
            ddFs : str ("y" or "n");
                specify if to fit depth variation or not. default is "n"
            dRpRs : tuple of len 2 or 3;
                deviation of depth in each group from the reference values. Must be tuple of len 2/3 specifying (mu,std)/(min,start,max) 
            divwhite : str ("y" or "n");
                flag to divide each light-curve by the white lightcurve. Default is "n"
            verbose: bool;
                print output

        """
        assert ddFs in ["y","n"], "transit_depth_variation(): ddFs must be 'y' or 'n'."
        if ddFs == "y": 
            assert self._config_par["pl1"]["Period"].start_value != 0, "transit_depth_variation(): planet_parameters() must be called before transit_depth_variation()."
            assert self._config_par["pl1"]["RpRs"].to_fit == "n" or self._config_par["pl1"]["RpRs"].step_size ==0,'Fix `RpRs` in `.planet_parameters()` to a reference value in order to setup depth variation.'
        assert isinstance(dRpRs, tuple),f"transit_depth_variation(): dRpRs must be tuple of len 2/3 specifying (mu,std)/(min,start,max)."

        grnames    = np.array(list(sorted(set(self._groups))))
        ngroup     = len(grnames)

        transit_depth_per_group = [(self._config_par["pl1"]["RpRs"].start_value,0)] if ddFs=="y" else [(0,0)]
        depth_per_group         = [d[0] for d in transit_depth_per_group] * ngroup  # depth for each group
        depth_err_per_group     = [d[1] for d in transit_depth_per_group] * ngroup 
        step = 0.001

        self._ddfs= SimpleNamespace()
        self._ddfs.drprs = _param_obj.from_tuple(dRpRs, user_input=dRpRs,func_call="transit_depth_variation():")

        self._ddfs.depth_per_group     = depth_per_group
        self._ddfs.depth_err_per_group = depth_err_per_group
        self._ddfs.divwhite            = divwhite
        self._ddfs.ddfYN               = ddFs

        if divwhite=="y":
            assert ddFs=='n', 'transit_depth_variation(): you can not do divide-white and not fit ddfs!'
            
            # for i in range(self._nphot):
            #     if (self._bases[i][6]>0):
            #         _raise(ValueError, 'transit_depth_variation(): you can not have CNMs active and do divide-white')
        
        if self._nphot>0: 
            if (ddFs=='n' and np.max(self._grbases)>0): _raise(ValueError,'no ddFs but groups? Not a good idea!')
            
        if verbose: _print_output(self,"depth_variation")
                
    
    def transit_timing_variation(self, ttvs="n", dt=(-0.125,0,0.125), baseline_amount=0.25,per_LC_T0=False, 
                                    show_plot=False, print_linear_eph=False, verbose=True):
        """
        include transit timing variation between the transit. Note: "T_0" and "P" must be fixed to reference values  in `.planet_parameters()` and not jump parameters.
        transit times are of each transit of a planet is calculated 
        
        Parameters:
        -----------
        ttvs : str ("y" or "n");
            specify if to fit transit timing variation or not. default is "n"
        dt : tuple of len 2 or 3;
            deviation of transit times from the expected values. Must be tuple of len 2/3 specifying (mu,std)/(min,start,max)
            this deviation is added to the expected linear ephemeris time, where n is the transit number. Default is 180mins around expected T0 i.e (-0.125,0,0.125)
            e.g T0_linear = 1406, dt=(-0.125,0,0.125) --> T0_prior = U(1406-0.125, 1406, 1406+0.125)
            dt = (0,0.125) --> T0_prior = N(1406, 0.125)
        baseline_amount : float;
            amount of baseline around each transit to use in extracting the transit. Default is 0.25 times the planet period.
            Only used to split transits when `per_LC_T0`=False.
        per_LC_T0: bool;
            whether to fit a single T0 per LC file as opposed to a T0 per transit.
        show_plot : bool;
            show plot of the extracted transits. Default is False.
        print_linear_eph : bool;
            print linear ephemeris. Default is False.
        """
        T0s, Ps = [], []
        assert isinstance(dt, tuple) and (len(dt) in [2,3]),f"transit_timing_variation(): dt must be tuple of len 2/3 specifying (mu,std)/(min,start,max) but {dt} given."
        assert isinstance(baseline_amount, (int,float,type(None))),f"transit_timing_variation(): baseline_amount must be a float/int but {baseline_amount} given."
        
        self._ttvs = SimpleNamespace(to_fit = "n", conf=[], fit_t0s=[], lc_names=[], pl_num=[],
                                        fit_labels=[],prior=[],dt=dt,baseline=baseline_amount,per_LC_T0=per_LC_T0)
        assert ttvs in ["y","n"], "transit_timing_variation(): ttvs must be 'y' or 'n'."
        if ttvs == "n":
            if verbose: _print_output(self,"timing_variation")
            return
        if ttvs == "y": 
            assert self._config_par["pl1"]["Period"].start_value != 0, "transit_timing_variation(): planet_parameters() must be called before transit_timing_variation()."
            
            self._ttvs.to_fit="y"
            for n in range(1,self._nplanet+1):
                assert self._config_par[f"pl{n}"]["T_0"].to_fit == "n" or self._config_par[f"pl{n}"]["T_0"].step_size ==0,'Fix `T_0` in `.planet_parameters()` to a reference value in order to setup TTVs.'
                assert self._config_par[f"pl{n}"]["Period"].to_fit == "n" or self._config_par[f"pl{n}"]["Period"].step_size ==0,'Fix `Period` in `.planet_parameters()` to a reference value in order to setup TTVs.'
                T0s.append(self._config_par[f"pl{n}"]["T_0"].start_value)
                Ps.append(self._config_par[f"pl{n}"]["Period"].start_value)
        
        lcnum = []
        for i,nm in enumerate(self._names):
            t, f = self._input_lc[nm]["col0"], self._input_lc[nm]["col1"]
            if per_LC_T0:    #get only one T0 within this LC
                lc_T0s = []
                for j in range(self._nplanet): lc_T0s.append([get_transit_time(t=t,per=Ps[j],t0=T0s[j])])
                self._ttvs.conf.append(split_transits(t=t, P=Ps, input_t0s=lc_T0s,flux=f, baseline_amount=None, 
                                                        show_plot=show_plot))     #returns
            else:   #get all T0s in the LC
                self._ttvs.conf.append(split_transits(t=t, P=Ps, t_ref= T0s,flux=f, baseline_amount=baseline_amount, 
                                                        show_plot=show_plot))   
            self._ttvs.conf[i].lcname = [nm]*len(self._ttvs.conf[i].t0s)
            self._ttvs.fit_t0s.extend(self._ttvs.conf[i].t0s)
            self._ttvs.pl_num.extend(self._ttvs.conf[i].plnum)
            self._ttvs.lc_names.extend([nm]*len(self._ttvs.conf[i].t0s))
            lcnum.extend([i+1]*len(self._ttvs.conf[i].t0s))

        self._ttvs.fit_labels = [f"ttv{j:02d}-lc{lcnum[j]}-T0_pl{self._ttvs.pl_num[j]+1}" for j in range(len(self._ttvs.fit_t0s))]
        for j in range(len(self._ttvs.fit_t0s)):
            if isinstance(dt, tuple):
                if len(dt) == 2:
                    assert dt[0]==0, "transit_timing_variation(): for N(mu,std) prior, mu must be 0."
                    self._ttvs.prior.append(_param_obj(*["y", self._ttvs.fit_t0s[j], 2e-4, "p",self._ttvs.fit_t0s[j], dt[1], dt[1], 
                                                    self._ttvs.fit_t0s[j]-20*dt[1], self._ttvs.fit_t0s[j]+20*dt[1]]))
                elif len(dt) == 3:
                    assert dt[1]==0 and dt[0]<dt[2], "transit_timing_variation(): for U(min,start,max) prior, start must be 0 and min<max."
                    self._ttvs.prior.append(_param_obj(*["y", self._ttvs.fit_t0s[j], 2e-4, "n", self._ttvs.fit_t0s[j], 0, 0, 
                                                            self._ttvs.fit_t0s[j]+dt[0], self._ttvs.fit_t0s[j]+dt[2]]))
        self._ttvs.lin_eph = {}
        for i in range(len(self._ttvs.fit_t0s)):
            self._ttvs.lin_eph[self._ttvs.fit_labels[i]] = self._ttvs.fit_t0s[i]
            
        if verbose: _print_output(self,"timing_variation")
        if print_linear_eph:
            _print_lin = f"""\n======(linear ephemeris estimate)===============\n{"label":20s}\t{"T0s (ordered)":16s}\t{"T0s priors"}"""
            txtfmt = "\n{0:20s}\t{1:.8f}\t{2}"
            for i in range(len(self._ttvs.fit_t0s)):
                ttv_pri = f"N({self._ttvs.fit_t0s[i]:.4f},{dt})" if len(dt)==2 else f"U({self._ttvs.fit_t0s[i]+dt[0]:.4f},{self._ttvs.fit_t0s[i]:.4f},{self._ttvs.fit_t0s[i]+dt[2]:.4f})"
                t = txtfmt.format(self._ttvs.fit_labels[i], self._ttvs.fit_t0s[i],ttv_pri)
                _print_lin += t
            print(_print_lin)
    
    def add_custom_LC_function(self,func=None, x="time",func_args=dict(), extra_args=dict(), op_func=None,replace_LCmodel=False,
                               verbose=True):
        """
        Define custom function to be combined with the light-curve model. The function must be of the form `func(x, **func_args,extra_args)` where `x` is the independent variable of the function.
        The function must return the signal to be combined with the light-curve model using `op_func`.

        Parameters:
        -----------
        func : callable;
            custom function to add to the light-curve model. Must be of the form `func(x,**func_args,extra_args)` where `x` is the independent variable of the function. Default is None.
            if this function replaces the native lightcurve model(`replace_LCmodel=True`), an additional dictionary argument of zero initilized LC parameters should be given to use in computing the new LCmodel i.e
            `func(x,**func_args,extra_args, LC_pars=dict(Duration=0,rho_star=0,RpRs=0,Impact_para=0,T_0=0,Period=0,Eccentricity=0,omega=90,q1=0,q2=0,D_occ=0,Fn=0,ph_off=0,A_ev=0,A_db=0))`
        x : str;
            the independent variable of the custom function. can be 'time' or 'phase_angle'. Default is "time".
            if 'time', the independent variable is the time array of each light-curve. if 'phase_angle', the independent variable is the phase angle computed within the transit model.
            the phase angle is generally given by true_anom+omega-pi/2 or simply 2pi*phase for circular orbit. If `replace_LCmodel=True`, then x needs to be time. 
        func_args : dict;
            dictionary of arguments to pass to the custom function. The keys must be the argument names for the custom function.
            each parameter can be fixed to a value(float/int) or set as a jump parameter (tuple of length 2/3/4). Default is an empty dictionary.
            tuple of length 2:nnormal prior (mean, std), tuple of length 3: uniform prior (min, start, max), tuple of length 4: truncated normal prior (min,max,mean, std)
        extra_args : dict;
            dictionary of extra arguments to pass to the custom function.  this arguments can be strings or any data type needed to be specified in the custom function. Default is an empty dictionary.
        op_func : callable;
            operation function to apply to the output of custom function and transit model to obtain the desired model. 
            Must be of the form `op_func(transit_model,custom_model)`. Default is None. This function is not required if `replace_LCmodel=True`. 
        replace_LCmodel : bool;
            replace the transit model with the custom model. Default is False.
        
        Returns:
        --------
        custom_LCfunc : SimpleNamespace;
            namespace object containing the custom function, operation function and parameter dictionary.

        Examples:
        --------
        # 1. create a custom function in phase angle that adds sinusoidal components to the light curve model
        >>> def custom_func(phase_angle, A, B, C,extra_args={}):     # A,B,C are the custom parameters and no extra arguments
        >>>     return A*np.sin(phase_angle) + B*np.cos(2*phase_angle) + C

        >>> def op_func(transit_model, custom_model):   # operation function to combine the custom model with the transit model
        >>>     return transit_model + custom_model

        >>> # A is fixed, B has uniform prior, C has gaussian prior
        >>> custom_lcfunc = lc_obj.add_custom_LC_function(func=custom_func, x="phase_angle",func_args=dict(A=0.1, B=(0.2,0,0.01), C=(0.3,0.01), op_func=op_func)
        >>> # this custom function has now been registered and will be used in the light-curve model.

        # 2. replace the transit model with a custom model that has two new parameters beyond the standard transit model parameters.
        # here we create a custom model from `catwoman` that models asymmetric transit light curves using two new parameters rp2 and phi.
        # the limb darkening law to use in catwoman  can be passed in the extra_args dictionary.
        >>> LC_pars=dict(Duration=0,rho_star=0,RpRs=0,Impact_para=0,T_0=0,Period=0,Eccentricity=0,omega=90,q1=0,q2=0,D_occ=0,Fn=0,ph_off=0,A_ev=0,A_db=0)
        >>> def catwoman_func(t, rp2, phi, extra_args=dict(ld_law="quadratic"),LC_pars=LC_pars):
        >>>     import catwoman 
        >>>     import numpy as np
        >>>     import astropy.constants as c
        >>>     import astropy.units as u
        >>>     
        >>>     # create a catwoman model. CONAN transit pars in LDpars can be passed to the function
        >>>     params  = catwoman.TransitParams()
        >>>     params.t0  = LD_pars["T_0"]          
        >>>     params.per = LD_pars["Period"]    
        >>>     params.rp  = LD_pars["RpRs"] 
        >>>     #convert stellar density to a/R*
        >>>     G  = (c.G.to(u.cm**3/(u.g*u.second**2))).value
        >>>     aR = ( rho*G*(P*(u.day.to(u.second)))**2 / (3*np.pi)) **(1/3.)
        >>>     params.a   =  aR                          
        >>>     params.inc = np.arccos(LD_pars["Impact_para"]/aR)   
        >>>     params.ecc = LD_pars["Eccentricity"]   
        >>>     params.w   = LD_pars["omega"]             
        >>>     params.limb_dark = extra_args["ld_law"]
        >>>     #convert from kipping parameterisation to quadratic
        >>>     u1 = 2*np.sqrt(LD_pars["q1"])*LD_pars["q2"]  
        >>>     u2 = np.sqrt(LD_pars["q1"])*(1-2*LD_pars["q2"]) 
        >>>     params.u = [u1, u2]    

        >>>     params.phi = phi                        #angle of rotation of top semi-circle (in degrees)
        >>>     params.rp2 = rp2                        #bottom semi-circle radius (in units of stellar radii)
        >>>     
        >>>     model = catwoman.TransitModel(params,t)         #initalises model
        >>>     return  model.light_curve(params)                #calculates light curve
        >>> 
        >>> lc_obj.add_custom_LC_function(func=catwoman_func, x="time",func_args=dict(rp2=(0.1,0.01), phi=(-90,0,90)),extra_args=dict(ld_law="quadratic") op_func=None,replace_LCmodel=True)
        
        """

        if func!=None:
            assert callable(func), "add_custom_LC_function(): func must be a callable function."
            assert isinstance(func_args, dict), "add_custom_LC_function(): func_args must be a dictionary."
            if extra_args == None: extra_args=dict()
            assert isinstance(extra_args, dict), "add_custom_LC_function(): extra_args must be a dictionary."
            #check that opfunc takes two array arguments
            if not replace_LCmodel: 
                assert x in ["time","phase_angle"], "add_custom_LC_function(): x must be 'time' or 'phase_angle'."
                assert callable(op_func), "add_custom_LC_function(): op_func must be a callable function."
                assert len(inspect.signature(op_func).parameters)==2, f"add_custom_LC_function(): op_func must take two arguments but {len(inspect.signature(op_func).parameters)} given."
                #assert that number of arguments in func() is equal to the number of parameters in func_args + the independent variable x + extra_args dict
                assert len(inspect.signature(func).parameters)==len(func_args)+2, f"add_custom_LC_function(): number of arguments in func must be equal to number of parameters in func_args + the independent variable + extra_args."
            else:
                assert x=="time", "add_custom_LC_function(): x must be 'time' if replace_LCmodel=True."
                # assert LC_pars argument in func
                assert "LC_pars" in inspect.signature(func).parameters, "add_custom_LC_function(): LC_pars dictionary argument must be in func in order to replace native transit model."
                #assert LC_pars argument is a dict with keys in planet_parameters
                tp_arg = inspect.signature(func).parameters["LC_pars"].default
                assert isinstance(tp_arg, dict) and len(tp_arg)==15, "add_custom_LC_function(): LC_pars argument in func must be a dictionary with 15 keys."
                assert all([k in tp_arg.keys() for k in ["Duration","rho_star","RpRs","Impact_para","T_0","Period","Eccentricity","omega","q1","q2","D_occ","Fn","ph_off","A_ev","A_db"]]), \
                    'add_custom_LC_function(): LC_pars argument in func must same keys as planet_parameters ["Duration","rho_star","RpRs","Impact_para","T_0","Period","Eccentricity","omega","q1","q2","D_occ","Fn","ph_off","A_ev","A_db"].'
                #assert that number of arguments in func() is equal to the number of parameters in func_args + the independent variable x + extra_args + LC_pars
                assert len(inspect.signature(func).parameters)==len(func_args)+3, f"add_custom_LC_function(): number of arguments in func_args must be equal to number of arguments in func minus 2."
        else: 
            self._custom_LCfunc = SimpleNamespace(func=None,x=None,op_func=None, func_args={},extra_args={},par_dict={},npars=0,nfree=0,replace_LCmodel=False)
            if verbose: _print_output(self,"custom_LCfunction")
            return None
        
        par_dict = {}   #dictionary to store parameter objects
        nfree    = 0
        for k in func_args.keys():
            if isinstance(func_args[k], (int,float)): 
                par_dict[k]  = _param_obj(*["n", func_args[k], 0.00, "n", func_args[k], 0, 0, 0, 0])
            elif isinstance(func_args[k], tuple):
                assert len(func_args[k]) in [2,3,4], f"add_custom_LC_function(): func_args[{k}] must be a tuple of length 2/3/4."
                if len(func_args[k])==2: #gaussian prior
                    par_dict[k] = _param_obj(*["y", func_args[k][0], 0.1*func_args[k][1], "p", func_args[k][0], func_args[k][1], func_args[k][1], func_args[k][0]-10*func_args[k][1], func_args[k][0]+10*func_args[k][1]])
                if len(func_args[k])==3: #uniform prior
                    par_dict[k] = _param_obj(*["y", func_args[k][1], 0.001*np.ptp(func_args[k]), "n", func_args[k][1], 0, 0, func_args[k][0], func_args[k][2]])
                if len(func_args[k])==4: #trunc gaussian prior
                    par_dict[k] = _param_obj(*["y", func_args[k][2], 0.1*func_args[k][3], "p", func_args[k][2], func_args[k][3], func_args[k][3], func_args[k][0], func_args[k][1]])
                nfree += 1
            else:
                _raise(TypeError, f"add_custom_LC_function(): func_args[{k}] must be a float/int/tuple.")
        self._custom_LCfunc = SimpleNamespace(func=func, x=x,op_func=op_func, func_args=func_args,extra_args=extra_args,par_dict=par_dict, npars=len(par_dict),nfree=nfree,replace_LCmodel=replace_LCmodel)

        if verbose: _print_output(self,"custom_LCfunction")
        return deepcopy(self._custom_LCfunc)
    
    def setup_phasecurve(self, D_occ=0, Fn=None, ph_off=None, A_ev=0, A_db=0, verbose=True ):
        """
            Setup phase curve parameters for each unique filter in loaded lightcurve. use `._filnames` attribute to see the unique filter order.
            Fn and ph_off are set to None by default which means the planet's phase variation is not included in the model (so we have just the occultation model if Docc is given). setting Fn and ph_off to a value, even if fixed to zero turns on the phase variation model.             
            Parameters:
            -----------
            D_occ : float, tuple, list;
                Occultation depth/dayside flux in ppm. Default is 0.
            Fn : float, tuple, list,None;
                Planet Nightside flux in ppm. Default is None which implies that a phase curve model is not included 
            ph_off : float, tuple,list;
                Offset of the hotspot in degrees. Default is None.
            A_ev : float, tuple, list;
                semi-amplitude of ellipsoidal variation in ppm. Default is 0.
            A_db : float, tuple, list;
                semi-amplitude of Doppler boosting in ppm. Default is 0.
            verbose : bool;
                print output. Default is True.
        """

        DA = locals().copy()
        _  = [DA.pop(item) for item in ["self","verbose"]]

        nfilt  = len(self._filnames)    #length of unique filters

        for par in DA.keys():
            if isinstance(DA[par], (int,float,tuple)) or DA[par]==None: DA[par] = [DA[par]]*nfilt
            if isinstance(DA[par], list):
                if len(DA[par])==1: DA[par] = DA[par]*nfilt
            assert len(DA[par])==nfilt, \
                        f"setup_phasecurve(): {par} must be a list of length {nfilt} (for filters {list(self._filnames)}) or float/int/tuple."
        
        self._model_phasevar = [False]*nfilt
        for i in range(nfilt):
            if DA['Fn'][i]!=None and DA['ph_off'][i]!=None: 
                self._model_phasevar[i] = True
                if verbose: print(f"{self._filnames[i]}: modeling planet's phase_variation with the occultation signal")
            else:                         
                self._model_phasevar[i] = False
                if verbose: print(f"{self._filnames[i]}: modeling only occultation signal")

        self._PC_dict = dict(D_occ = {}, Fn = {}, ph_off = {}, A_ev = {}, A_db={})     #dictionary to store phase curve parameters
        for par in DA.keys():      #D_occ, Fn, ph_off,A_ev, A_db
            for i,f in enumerate(self._filnames):    
                v = DA[par][i]
                self._PC_dict[par][f] = _param_obj.from_tuple(v,step=1,user_input=v,func_call="setup_phasecurve():")

        if verbose: _print_output(self,"phasecurve")

    def get_LDs(self,Teff,logg,Z, filter_names, unc_mult=10, fixed_unc=None, use_result=False, verbose=True):
        """
        get Kipping quadratic limb darkening parameters (q1,q2) using ldtk (requires internet connection).

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
            use `lc_obj._filter_shortcuts` to get list of filter shortcut names. Filter names can be obtained from http://svo2.cab.inta-csic.es/theory/fps/
        unc_mult : int/float, optional
            value by which to multiply ldtk uncertainties which are usually underestimated, by default 10
        fixed_unc : float/list, optional
            fixed uncertainty value to use for each filter, by default None and unc_mult is used
        use_result : bool, optional
            whether to use the result to setup limb darkening priors, by default True

        Returns
        -------
        q1, q2 : arrays
            each coefficient is an array of only values (no uncertainity) for each filter. 
            These can be fed to the `limb_darkening()` function to fix the coefficients
        """

        from ldtk import LDPSetCreator, BoxcarFilter
        q1, q2 = [], []
        
        if isinstance(filter_names,list): 
            assert len(filter_names)==len(self._filnames),\
                f"get_LDs: number of unique filters in input lc_list = {len(self._filnames)} but trying to obtain LDs for {len(filter_names)} filters."
        elif isinstance(filter_names, str): filter_names = [filter_names]
        else: raise TypeError(f"get_LDs: filter_names must be str for a single filter or list of str but {type(filter_names)} given.")

        if isinstance(fixed_unc, float): fixed_unc = [fixed_unc]*len(filter_names)
        if isinstance(fixed_unc, list): assert len(fixed_unc)==len(filter_names),\
            f"get_LDs: length of fixed_unc must be equal to number of filters (={len(filter_names)})."

        for i,f in enumerate(filter_names):
            if isinstance(f, str):
                if f.lower() in self._filter_shortcuts.keys(): ft = self._filter_shortcuts[f.lower()]
                else: ft=f
                flt = SVOFilter(ft)
            if isinstance(f,(SVOFilter, SimpleNamespace)): 
                flt = f
                f   = flt.name
            ds  = 'visir-lowres' if np.any(flt.wavelength > 1000) else 'vis-lowres'

            sc  = LDPSetCreator(teff=Teff, logg=logg, z=Z,    # spectra from the Husser et al.
                                filters=[flt], dataset=ds)      # FTP server automatically.

            ps = sc.create_profiles(100)                      # Create the limb darkening profiles\
            ps.set_uncertainty_multiplier(unc_mult)
            ps.resample_linear_z(300)

            #calculate ld profiles
            c, e = ps.coeffs_tq(do_mc=True, n_mc_samples=10000,mc_burn=1000)
            if fixed_unc: e[0][0] = e[0][1] = fixed_unc[i]
            ld1 = (round(c[0][0],4),round(e[0][0],4))
            ld2 = (round(c[0][1],4),round(e[0][1],4))
            q1.append(ld1)
            q2.append(ld2)
            if verbose: print(f"{f:10s}({self._filters[i]}): q1={ld1}, q2={ld2}")

        if use_result: 
            q_1 = deepcopy(q1)
            q_2 = deepcopy(q2)
            if verbose: print(_text_format.BOLD + "\nSetting-up limb-darkening priors from LDTk result" +_text_format.END)
            self.limb_darkening(q_1,q_2)
        return q1,q2



    def limb_darkening(self, q1=0, q2 = 0,verbose=True):
        """
            Setup Kipping quadratic limb darkening LD coefficient (q1, q2) for transit light curves. 
            Different LD coefficients are required if observations of different filters are used.

            Parameters:
            -----------
            q1,q2 : float/tuple or list of float/tuple for each filter;
                Stellar quadratic limb darkening coefficients.
                if tuple, must be of - length 2 for normal prior (mean,std) or length 3 for uniform prior defined as (lo_lim, val, uplim).
                The values must obey: (0<q1<1) and (0<=q2<1)

            Example:
            --------
            # set the limb darkening coefficients for each filter
            >>> lc_obj.limb_darkening(q1=0.5, q2=0.2)  # fixed values for all filters
            >>> lc_obj.limb_darkening(q1=[0.5,0.6], q2=[0.2,0.3])  # different fixed values for each filter (2 filters)  

            >>> lc_obj.limb_darkening(q1=(0.5,0.1), q2=(0.2,0.05))  # normal prior for all filters
            >>> lc_obj.limb_darkening(q1=[(0.5,0.1),(0.6,0.05)], q2=[(0.2,0.05),(0.3,0.1)])  # different normal prior for each filter (2 filters)
            
            >>> lc_obj.limb_darkening(q1=(0,0.1,1), q2=(0,0.05,1))  # uniform prior for all filters
            >>> lc_obj.limb_darkening(q1=[(0,0.1,1),(0.6,0.05,1)], q2=[(0,0.05,1),(0.3,0.1,1)])  # different uniform prior for each filter (2 filters

        """
        #defaults
        bound_lo1 = bound_lo2 = 0
        bound_hi1 = bound_hi2 = 0
        sig_lo1 = sig_lo2 = 0
        sig_hi1 = sig_hi2 = 0
        step1 = step2 = 0 

        DA = deepcopy(locals())
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")

        nfilt = len(self._filnames)

        for par in DA.keys():
            if isinstance(DA[par], (int,float,tuple,str)): DA[par] = [DA[par]]*nfilt
            elif isinstance(DA[par], list): assert len(DA[par]) == nfilt,f"limb_darkening(): length of list {par} must be equal to number of unique filters (={nfilt})."
            else: _raise(TypeError, f"limb_darkening(): {par} must be int/float, or tuple of len 2 (for gaussian prior) or 3 (for uniform prior) but {DA[par]} is given.")
        
        for par in ["q1","q2"]:
            for i,d in enumerate(DA[par]):
                if isinstance(d, (int,float)):  #fixed
                    DA[par][i] = d
                    DA[f"step{par[-1]}"][i] = DA[f"bound_lo{par[-1]}"][i] = DA[f"bound_hi{par[-1]}"][i] = 0
                elif isinstance(d, tuple):
                    if len(d) == 2:  #normal prior
                        DA[par][i] = d[0]
                        DA[f"sig_lo{par[-1]}"][i] = DA[f"sig_hi{par[-1]}"][i] = d[1]
                        DA[f"bound_lo{par[-1]}"][i] = 0
                        DA[f"bound_hi{par[-1]}"][i] = 1
                        DA[f"step{par[-1]}"][i] = 0.1*DA[f"sig_lo{par[-1]}"][i]

                    if len(d) == 3:  #uniform prior
                        assert d[0]<=d[1]<=d[2],f'limb_darkening(): uniform prior be (lo_lim, start_val, uplim) where lo_lim <= start_val <= uplim but {d} given.'
                        assert (d[0]>=0  and d[2]<=1),f'limb_darkening(): uniform prior must be (lo_lim, val, uplim) where lo_lim>=0 and uplim<=1 but {d} given.'
                        DA[par][i] = d[1]
                        DA[f"bound_lo{par[-1]}"][i] = d[0]
                        DA[f"bound_hi{par[-1]}"][i] = d[2]
                        DA[f"sig_lo{par[-1]}"][i] = DA[f"sig_hi{par[-1]}"][i] = 0
                        DA[f"step{par[-1]}"][i] = min(0.001, np.ptp([d[0],d[2]]))

        DA["priors"] = [0]*nfilt
        for i in range(nfilt):
            DA["priors"][i] = "y" if np.any( [DA["sig_lo1"][i], DA["sig_lo2"][i] ]) else "n"

        self._ld_dict = DA
        if verbose: _print_output(self,"limb_darkening")

    def contamination_factors(self, cont_ratio=0, verbose=True):
        """
            add contamination factor for each unique filter defined in load_lightcurves().

            Paramters:
            ----------
            cont_ratio: tuple, float;
                ratio of contamination flux to target flux in aperture for each filter. The order of list follows lc_obj._filnames.
                Very unlikely but if a single float/tuple is given for several filters, same cont_ratio is used for all.
            
            Example:
            --------
            >>> lc_obj.contamination_factors(cont_ratio=0.1)  # fixed contamination ratio for all filters
            >>> lc_obj.contamination_factors(cont_ratio=[0.1,0.08])  # different contamination ratio for each filter
            >>> lc_obj.contamination_factors(cont_ratio=(0.1,0.01))  # normal prior for all filters
        """
        nfilt = len(self._filnames)

        if isinstance(cont_ratio, (int,float,tuple)): cont_ratio = [cont_ratio]*nfilt
        elif isinstance(cont_ratio, list):
            if len(cont_ratio)==1: cont_ratio = cont_ratio*nfilt
            assert len(cont_ratio)==nfilt, f"contamination(): cont_ratio must be a list of length {nfilt}  (for filters {list(self._filnames)}) or float/int/tuple."
        else:
            _raise(TypeError, f"contamination(): cont_ratio must be one of [tuple(of len 2/3/4), float, or list of float/tuple] but {cont_ratio} is given.")
        
        self._contfact_dict = {}
        for i,f in enumerate(self._filnames):
                self._contfact_dict[f] = _param_obj.from_tuple(cont_ratio[i],user_input=cont_ratio[i],func_call="contamination_factors():")

        if verbose: _print_output(self,"contamination")

    def __repr__(self):
        data_type = str(self.__class__).split("load_")[1].split("'>")[0]
        self.print("lc_baseline")
        if self._nphot>0:
            return f'{data_type} from filepath: {self._fpath}\n{self._nplanet} transiting planet(s)\nOrder of unique filters: {list(self._filnames)}'
        else:
            return ""
    def print(self, section="all"):
        """
            Print out all input configuration (or particular section) for the light curve object. 
            It is printed out in the format of the legacy CONAN config file.
            
            Parameters:
            ------------
            section : str (optional) ;
                section of configuration to print.Must be one of ["lc_baseline", "sinusoid","gp", "planet_parameters", "custom_LCfunction", "depth_variation", "occultations", "limb_darkening", "contamination", "stellar_pars"].
                Default is 'all' to print all sections.
        """
        if section=="all":
            _print_output(self,"lc_baseline")
            _print_output(self,"sinusoid")
            _print_output(self,"gp")
            _print_output(self,"planet_parameters")
            _print_output(self,"limb_darkening")
            _print_output(self,"depth_variation")
            _print_output(self,"timing_variation")
            _print_output(self,"phasecurve")
            _print_output(self,"custom_LCfunction")
            _print_output(self,"contamination")
        else:
            possible_sections= ["lc_baseline", "sinusoid", "gp", "planet_parameters", "custom_LCfunction", "depth_variation","timing_variation",  
                                "phasecurve", "limb_darkening", "contamination", "stellar_pars"]
            assert section in possible_sections, f"print: {section} not a valid section of `lc_obj`. \
                section must be one of {possible_sections}."
            _print_output(self, section)

    def save_LCs(self, save_path=None, suffix="",overwrite=False, verbose=True):
        """
            Save each of the loaded light curves to a .dat file  

            Parameters:
            -----------
            save_path : str;
                path to save the LC data. Default is None to save to a folder 'data_preproc/' in current working directory.
            suffix : str;
                suffix to add to the original name of each file. Default is "".
            overwrite : bool;
                overwrite the existing file. Default is False.
            verbose : bool;
                print output. Default is True.

            Example:
            --------
            >>> lc_obj.save_LCs(save_path="data_preproc/", suffix="_clpd")
        """
        if save_path is None: save_path = 'data_preproc/'
        for k,lc in self._input_lc.items():
            if not os.path.exists(save_path): os.makedirs(save_path)
            _fname  = splitext(k)[0] + suffix + splitext(k)[1]
            if not overwrite:
                if os.path.exists(save_path+_fname): 
                    print(f"save_LCs(): {save_path+_fname} already exists. Set `overwrite=True` to overwrite the file.")
                    return
            #save dict lc as a .dat file 
            pd.DataFrame(lc).to_csv(save_path+_fname, index=False, sep="\t", header=False,float_format='%-16.6f')
            if verbose: print(f"save_LCs(): {save_path+_fname} saved.")


    def plot(self, plot_cols=(0,1,2), col_labels=None, nrow_ncols=None, figsize=None, fit_order=0, 
                show_decorr_model=False, detrend=False, hspace=None, wspace=None, binsize=0.0104, return_fig=False):
        """
            visualize data

            Parameters:
            -----------
            plot_cols : tuple of length 2 or 3;
                Tuple specifying which columns in input file to plot. 
                Default is (0,1,2) to plot column 0 against 1, and 2 as errorbar (i.e. time against flux with fluxerr). 
                Use (3,1,2) to show the correlation between column 3 and the flux. Using tuple of length 2 does not plot errorbars. e.g (3,1).
                if decorrelation has been done with `lmfit`, the "res" can also be given to plot a column against the residual of the fit.
            col_labels : tuple of length 2;
                label of the given columns in plot_cols. Default is ("time", "flux").
            nrow_ncols : tuple of length 2;
                Number of rows and columns to plot the input files. 
                Default is None to find the best layout.
            fit_order : int;
                order of polynomial to fit to the plotted data columns to visualize correlation.
            show_decorr_model : bool;
                show decorrelation model if decorrelation has been done.
            detrend : bool;
                plot the detrended data. Default is False.
            hspace, wspace: float;
                height and width space between subplots. Default is None to use matplotlib defaults.
            binsize : float;
                binsize to use for binning the data in time. Default is 0.0104 (15mins).
            figsize: tuple of length 2;
                Figure size. If None, (8,5) is used for a single input file and optimally determined for more inputs.

            return_fig  : bool;
                return figure object for saving to file.
        """
        if not (isinstance(plot_cols, tuple) and len(plot_cols) in [2,3]): 
            raise TypeError(f"plot: plot_cols must be tuple of length 2 or 3, but is {type(plot_cols)} and length of {len(plot_cols)}.")
        if detrend: assert show_decorr_model, "plot(): detrend can only be True if decorrelation has been done."
        if plot_cols[1] == "res": assert show_decorr_model, "plot(): plot_cols[1] can only be 'res' if decorrelation has been done, and show_decorr_model=True."

        assert col_labels is None or ((isinstance(col_labels, tuple) and len(col_labels)==2)), \
            f"plot: col_labels must be tuple of length 2, but is {type(col_labels)} and length of {len(col_labels)}."
        
        assert isinstance(fit_order,int),f'fit_order must be an integer'

        if show_decorr_model:
            if not hasattr(self,"_tmodel"): 
                print("cannot show decorr model since decorrelation has not been done. First, use `lc_obj.get_decorr()` to launch decorrelation.")
                show_decorr_model = False
        
        if col_labels is None:
            col_labels = ("time", "flux") if plot_cols[:2] == (0,1) else (f"column[{plot_cols[0]}]",f"column[{plot_cols[1]}]")
        
        if self._names != []:
            fig = _plot_data(self, plot_cols=plot_cols, col_labels = col_labels, nrow_ncols=nrow_ncols, figsize=figsize, fit_order=fit_order,
                            hspace=hspace, wspace=wspace, model_overplot=self._tmodel if show_decorr_model else None, detrend=detrend,binsize=binsize)
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
            unit of the rv data. Must be one of ["m/s","km/s"]. Default is "km/s".
        lc_obj : object;
            lightcurve object to modify rv parameters
        show_guide : bool;
            print output to guide the user. Default is False.

        Returns:
        --------
        rv_obj : rv object

        Examples:
        ---------
        >>> rv_obj = load_rvs(file_list=["rv1.dat","rv2.dat"], data_filepath="/path/to/data/", rv_unit="km/s")
    """
    def __init__(self, file_list=None, data_filepath=None, nplanet=1, rv_unit="km/s",lc_obj=None,show_guide =False):
        self._obj_type = "rv_obj"
        self._nplanet  = nplanet
        self._fpath    = os.getcwd()+"/" if data_filepath is None else data_filepath
        self._names    = [] if file_list is None else file_list 
        self._input_rv = {}
        self._RVunit   = rv_unit
        self._nRV      = len(self._names)
        self._lcobj    = lc_obj
        
        assert rv_unit in ["m/s","km/s"], f"load_rvs(): rv_unit must be one of ['m/s','km/s'] but {rv_unit} given." 

        for rv in self._names: assert os.path.exists(self._fpath+rv), f"file {rv} does not exist in the path {self._fpath}."
        if show_guide: print("Next: use method `rv_baseline` to define baseline model for for the each rv")
        
        #modify input files to have 6 columns as CONAN expects
        self._rms_estimate, self._jitt_estimate = [], []
        for f in self._names:
            fdata = np.loadtxt(self._fpath+f)
            nrow,ncol = fdata.shape
            if ncol < 6:
                # if verbose: print(f"Expected at least 6 columns for RV file: writing ones to the missing columns of file: {f}")
                new_cols = np.ones((nrow,6-ncol))
                fdata = np.hstack((fdata,new_cols))
                np.savetxt(self._fpath+f,fdata,fmt='%.8f')
            #remove nan rows
            n_nan = np.sum(np.isnan(fdata).any(axis=1))
            if n_nan > 0: print(f"removed {n_nan} row(s) with NaN values from file: {f}")
            fdata = fdata[~np.isnan(fdata).any(axis=1)]
            #store input files in rv object
            self._input_rv[f] = {}
            for i in range(6): self._input_rv[f][f"col{i}"] = fdata[:,i]
            self._rms_estimate.append(np.std(fdata[:,1]))   #std of rv
            self._jitt_estimate.append( np.sqrt(self._rms_estimate[-1]**2 - np.mean(fdata[:,2]**2)) )
            if np.isnan(self._jitt_estimate[-1]): self._jitt_estimate[-1] = 0 

        #list to hold initial baseline model coefficients for each rv
        self._RVbases_init = [dict( A0=0, B0=0, A3=0, B3=0, A4=0, B4=0, A5=0, B5=0, 
                                    amp=0,freq=0,phi=0,phi2=0)
                                for _ in range(self._nRV)]
            
        self._rescaled_data = SimpleNamespace(flag=False, config=["None"]*self._nRV)
        self.rv_baseline(verbose=False)
        self.add_custom_RV_function(verbose=False)


    def update_planet_parameters(self, T_0=None, Period=None, Eccentricity=None, omega=None, K=None, verbose=True):
        """
            Update the rv planet parameters defined in the lightcurve object. 

            Parameters:
            -----------
            T_0 : float, tuple;
                Mid-transit time in days. Default is 0.
            Period : float, tuple;
                Orbital period of the planet in days. Default is 0.
            Eccentricity : float, tuple;
                Eccentricity of the orbit. Default is 0.
            omega : float, tuple;
                Argument of periastron. Default is 90.
            K : float, tuple;
                Radial velocity semi-amplitude in same unit as the data. Default is 0.
            verbose : bool;
                print output. Default is True.
        """
        assert self._lcobj is not None, "update_planet_parameters(): lightcurve object not defined. Use `lc_obj` argument in `load_rvs()` to define lightcurve object."
        self._lcobj.update_planet_parameters(T_0=T_0, Period=Period, Eccentricity=Eccentricity, omega=omega, K=K, verbose=verbose)

    def rescale_data_columns(self,method="med_sub", verbose=True):

        """
            Function to rescale the data columns of the RVs. This can be important when decorrelating the data with polynomials.
            The operation is not performed on columns 0,1,2. It is only performed on columns whose values do not span zero.
            Function can only be run once on the loaded datasets but can be reset by running `load_rvs()` again. 

            The method can be one of ["med_sub", "rs0to1", "rs-1to1","None"] which subtracts the median, rescales to [0,1], rescales to [-1,1], or does nothing, respectively.
        """
        
        if self._rescaled_data.flag:
            print("Data columns have already been rescaled. run `load_rvs()` again to reset.")
            return None
        
        if isinstance(method,str): method = [method]*self._nRV
        elif isinstance(method, list):
            assert len(method)==1 or len(method)==self._nRV, f'rescale_data_columns(): method must be either str or list of same length as number of input lcs ({self._nRV})'
        else: _raise(TypeError,'rescale_data_columns(): method must be either str or list of same length as number of input RVs ({self._nRV})')

        if method == ["None"]*self._nRV:
            self._rescaled_data.flag = False
            return None
        
        for j,rv in enumerate(self._names):
            assert method[j] in ["med_sub", "rs0to1", "rs-1to1","None"], f"method must be one of ['med_sub','rs0to1','rs-1to1','None'] but {method[j]} given"
            if verbose: print(f"No rescaling for {rv}") if method[j]=="None" else print(f"Rescaled data columns of {rv} with method:{method[j]}")
            for i in range(6):
                if i not in [0,1,2]:
                    if not (min(self._input_rv[rv][f"col{i}"]) <= 0 <=  max(self._input_rv[rv][f"col{i}"])):     #if zero not in array
                        if method[j] == "med_sub":
                            self._input_rv[rv][f"col{i}"] -= np.median(self._input_rv[rv][f"col{i}"])
                        elif method[j] == "rs0to1":
                            self._input_rv[rv][f"col{i}"] = rescale0_1(self._input_rv[rv][f"col{i}"])
                        elif method[j] == "rs-1_1":
                            self._input_rv[rv][f"col{i}"] = rescale_minus1_1(self._input_rv[rv][f"col{i}"])
                        else: pass
        self._rescaled_data = SimpleNamespace(flag=True, config=method)

    def get_decorr(self, T_0=None, Period=None, K=None, sesinw=0, secosw=0, gamma=0,
                    delta_BIC=-5, decorr_bound =(-1000,1000), exclude_cols=[],enforce_pars=[],
                    show_steps=False, plot_model=True, use_jitter_est=False, setup_baseline=True,verbose=True ):
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
            exclude_cols : list of int;
                list of column numbers (e.g. [3,4]) to exclude from decorrelation. Default is [].
            enforce_pars : list of int;
                list of decorr params (e.g. ['B3', 'A5']) to enforce in decorrelation. Default is [].
            show_steps : Bool, optional;
                Whether to show the steps of the forward selection of decorr parameters. Default is False
            plot_model : Bool, optional;
                Whether to plot data and suggested trend model. Defaults to True.
            use_jitter_est : Bool, optional;
                Whether to use the jitter estimate to setup the baseline model. Default is False.
            setup_baseline : Bool, optional;
                whether to use result to setup the baseline model. Default is True.
            verbose : Bool, optional;
                Whether to show the table of baseline model obtained. Defaults to True.
        
            Returns
            -------
            decorr_result: list of result object
                list containing result object for each lc.
        """
        assert isinstance(exclude_cols, list), f"get_decorr(): exclude_cols must be a list of column numbers to exclude from decorrelation but {exclude_cols} given."
        for c in exclude_cols: assert isinstance(c, int), f"get_decorr(): column number to exclude from decorrelation must be an integer but {c} given in exclude_cols."
        assert delta_BIC<0,f'get_decorr(): delta_BIC must be negative for parameters to provide improved fit but {delta_BIC} given.'
        if isinstance(gamma, tuple):
            assert len(gamma)==2 or len(gamma)==3,f"get_decorr(): gamma must be float or tuple of length 2/3, but {gamma} given."
        else: assert isinstance(gamma, (int,float)),f"get_decorr(): gamma must be float or tuple of length 2/3, but {gamma} given."

        blpars = {"dcol0":[], "dcol3":[],"dcol4":[], "dcol5":[]}  #inputs to rv_baseline method
        self._rvdecorr_result = []   #list of decorr result for each lc.
        self._rvmodel = []  #list to hold determined trendmodel for each rv
        gamma_init    = []  #list to update gamma prior for each rv based on the lmfit

        self._rv_pars = dict(T_0=T_0, Period=Period, K=K, sesinw=sesinw, secosw=secosw, gamma=gamma) #rv parameters
        for p in self._rv_pars:
            if p != "gamma":
                if isinstance(self._rv_pars[p], (int,float,tuple)): self._rv_pars[p] = [self._rv_pars[p]]*self._nplanet
                if isinstance(self._rv_pars[p], (list)): assert len(self._rv_pars[p]) == self._nplanet, \
                    f"get_decorr(): {p} must be a list of same length as number of planets {self._nplanet} but {len(self._rv_pars[p])} given."


        decorr_cols = [0,3,4,5]
        for c in exclude_cols: assert c in decorr_cols, f"get_decorr(): column number to exclude from decorrelation must be in {decorr_cols} but {c} given in exclude_cols." 
        _ = [decorr_cols.remove(c) for c in exclude_cols]  #remove excluded columns from decorr_cols

        for j,file in enumerate(self._names):
            df = self._input_rv[file]
            if verbose: print(_text_format.BOLD + f"\ngetting decorrelation parameters for rv: {file} (jitt={self._jitt_estimate[j]*1e6 if use_jitter_est else 0:.2f}{self._RVunit})" + _text_format.END)
            all_par = [f"{L}{i}" for i in decorr_cols for L in ["A","B"]] 

            out = _decorr_RV(df, **self._rv_pars, decorr_bound=decorr_bound, npl=self._nplanet,
                            jitter=self._jitt_estimate[j] if use_jitter_est else 0)    #no trend, only offset
            best_bic = out.bic
            best_pars = {}                      #parameter salways included
            for cp in enforce_pars: best_pars[cp]=0            #add enforced parameters
            _ = [all_par.remove(cp) for cp in enforce_pars if cp in all_par]    #remove enforced parameters from all_par

            if show_steps: print(f"{'Param':7s} : {'BIC':6s} N_pars \n---------------------------")
            del_BIC = -np.inf 
            while del_BIC < delta_BIC:
                if show_steps: print(f"{'Best':7s} : {best_bic:.2f} {len(best_pars.keys())} {list(best_pars.keys())}\n---------------------")
                pars_bic = {}
                for p in all_par:
                    dtmp = best_pars.copy()  #temporary dict to hold parameters to test
                    dtmp[p] = 0
                    out = _decorr_RV(df, **self._rv_pars,**dtmp, decorr_bound=decorr_bound, npl=self._nplanet,
                                    jitter=self._jitt_estimate[j] if use_jitter_est else 0)
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

            result = _decorr_RV(df, **self._rv_pars,**best_pars, decorr_bound=decorr_bound, npl=self._nplanet,
                                jitter=self._jitt_estimate[j] if use_jitter_est else 0)
            self._rvdecorr_result.append(result)
            if verbose: print(f"\nBEST BIC:{result.bic:.2f}, pars:{list(best_pars.keys())}")
            
            #calculate determined trend and rv model over all data
            pps = result.params.valuesdict()
            #convert result transit parameters to back to a list
            for p in ["T_0", "Period", "K", "sesinw", "secosw"]:
                if self._nplanet==1:
                    pps[p] = [pps[p]]  
                else:      
                    pps[p] = [pps[p+f"_{n}"] for n in range(1,self._nplanet+1)]
                    _      = [pps.pop(f"{p}_{n}") for n in range(1,self._nplanet+1)]
    
            self._rvmodel.append(_decorr_RV(df,**pps,decorr_bound=decorr_bound, npl=self._nplanet, return_models=True))

            #set-up lc_baseline model from obtained configuration
            blpars["dcol0"].append( 2 if pps["B0"]!=0 else 1 if  pps["A0"]!=0 else 0)
            blpars["dcol3"].append( 2 if pps["B3"]!=0 else 1 if  pps["A3"]!=0 else 0)
            blpars["dcol4"].append( 2 if pps["B4"]!=0 else 1 if  pps["A4"]!=0 else 0)
            blpars["dcol5"].append( 2 if pps["B5"]!=0 else 1 if  pps["A5"]!=0 else 0)
            # store baseline model coefficients for each lc, to used as start values of mcmc
            self._RVbases_init[j] = dict( A0=pps["A0"], B0=pps["B0"],
                                        A3=pps["A3"], B3=pps["B3"], 
                                        A4=pps["A4"], B4=pps["B4"],
                                        A5=pps["A5"], B5=pps["B5"], 
                                        amp=0,freq=0,phi=0,phi2=0)
            
            # adjust gamma prior based on result of the fit to each rv
            if isinstance(gamma, tuple): 
                if len(gamma) ==2: gamma_init.append( (pps["gamma"], gamma[1]))     
                if len(gamma) ==3: gamma_init.append( (gamma[0], pps["gamma"], gamma[2]) )  

        if plot_model:
            _plot_data(self,plot_cols=(0,1,2),col_labels=("time","rv"),binsize=0,model_overplot=self._rvmodel)
        

        #prefill other light curve setup from the results here or inputs given here.
        if setup_baseline:
            if verbose: print(_text_format.BOLD + "Setting-up rv baseline model from result" +_text_format.END)
            self.rv_baseline(dcol0 = blpars["dcol0"], dcol3=blpars["dcol3"], dcol4=blpars["dcol4"],
                                dcol5=blpars["dcol5"], gamma= gamma_init, verbose=verbose)

        return self._rvdecorr_result
    
    def rv_baseline(self, dcol0=None, dcol3=None, dcol4=None, dcol5=None, sinPs=None,gamma=0.0, gp="n",verbose=True):
        
        """
            Define rv baseline model parameters to fit.
            Each baseline model parameter should be a list of numbers specifying the polynomial order for each rv data.
            e.g. Given 3 input rvs, and one wishes to fit a 2nd order time trend to only the first and third lightcurves,
            then dcol0 = [2, 0, 2].

            Parameters:
            -----------
            dcol0,dcol3,dcol4,dcol5,: list of ints;
                polynomial order to fit to each column. Default is 0 for all columns. max order is 2
            gamma: tuple,floats or list of tuple/float;
                specify if to fit for gamma. if float/int, it is fixed to this value. If tuple of len 2 it assumes gaussian prior as (prior_mean, width) and if len 3 uniform as (min,start,max) with min<start<max.
        """

        # if self._names == []: 
        #     if verbose: _print_output(self,"rv_baseline")
        #     return 
        
        if isinstance(gamma, list): assert len(gamma) == self._nRV, f"gamma must be type tuple/int or list of tuples/floats/ints of len {self._nRV}."
        elif isinstance(gamma, (tuple,float,int)): gamma=[gamma]*self._nRV
        else: _raise(TypeError, f"gamma must be type tuple/int or list of tuples/floats/ints of len {self._nRV}." )
        sinPs = [0]*self._nRV

        DA = locals().copy()     #get a dictionary of the input/variables arguments for easy manipulation
        _ = [DA.pop(item) for item in ["self","gamma","verbose"]]

        for par in DA.keys():            
            if DA[par] is None: DA[par] = ["n"]*self._nRV if par=="gp" else [0]*self._nRV
            elif isinstance(DA[par], (int,float,str)): DA[par] = [DA[par]]*self._nRV
            elif isinstance(DA[par],list): 
                if len(DA[par])==1: DA[par]*self._nRV
                assert len(DA[par]) == self._nRV, f"parameter {par} must be a list of length {self._nRV} or 1 if same is to be used for all RVs or None"

            if par=="gp":
                for p in DA[par]: assert p in ["ce","y","n"], f"rv_baseline(): gp must be one of ['y','ce','n'] but {p} given."
            if 'dcol' in par:
                for p in DA[par]: assert isinstance(p, (int,np.int64)) and p<3, f"rv_baseline(): polynomial order must be int of max 2 but {p} given."

        self._RVbases = [ [DA["dcol0"][i], DA["dcol3"][i], DA["dcol4"][i], DA["dcol5"][i],DA["sinPs"][i]] for i in range(self._nRV) ]
        self._useGPrv = DA["gp"]
        self._gp_rvs  = lambda : np.array(self._names)[np.array(self._useGPrv) != "n"]     #rvs with gp == "y" or "ce"
        
        DA["gamma"] = []
        for g in gamma:
            DA["gamma"].append(_param_obj.from_tuple(g, user_input=g, func_call="rv_baseline():"))
                    
        _ = [DA.pop(item) for item in ["dcol0","dcol3","dcol4","dcol5"]]
        self._rvdict   = DA
        
        if not hasattr(self,"_rvspline"):  self.add_spline(None, verbose=False)
        if np.all(np.array(self._useGPrv) == "n") or len(self._useGPrv)==0:        #if gp is "n" for all input rvs, run add_rvGP with None
            self.add_rvGP(None, verbose=False)

        if verbose: _print_output(self,"rv_baseline")
    

   
    def add_custom_RV_function(self,func=None, x="time",func_args=dict(), extra_args=dict(), op_func=None,replace_RVmodel=False,
                               verbose=True):
        """
        Define custom function to be combined with the RV model. The function must be of the form `func(x, **func_args,extra_args)` where `x` is the independent variable of the function.
        The function must return the signal to be combined with the RV model using `op_func`.

        Parameters:
        -----------
        func : callable;
            custom function to add to the RV model. Must be of the form `func(x,**func_args,extra_args)` where `x` is the independent variable of the function. Default is None.
            if this function replaces the native CONAN RV model(`replace_RVmodel=True`), an additional dictionary argument of zero-initialized RV parameters should be given to use in computing the new RVmodel i.e
            `func(x,**func_args,extra_args, RV_pars=dict(T_0=0,Period=0,Eccentricity=0,omega=90,K=0))`
        x : str;
            the independent variable of the custom function. can be 'time' or 'true_anomaly'. Default is "time".
            if 'time', the independent variable is the time array of each RV. if 'true_anomaly', the independent variable is the true anomaly computed within the RV model using time, T0, Period, Eccentricity and omega.
        func_args : dict;
            dictionary of arguments to pass to the custom function. The keys must be the argument names for the custom function.
            each parameter can be fixed to a value(float/int) or set as a jump parameter (tuple of length 2/3/4). Default is an empty dictionary.
            tuple of length 2:normal prior (mean, std), tuple of length 3: uniform prior (min, start, max), tuple of length 4: truncated normal prior (min,max,mean, std)
        extra_args : dict;
            dictionary of extra arguments to pass to the custom function.  this arguments can be strings or any data type needed to be specified in the custom function. Default is an empty dictionary.
        op_func : callable;
            operation function to apply to the output of custom function and transit model to obtain the desired model. 
            Must be of the form `op_func(transit_model,custom_model)`. Default is None. This function is not required if `replace_RVmodel=True`. 
        replace_RVmodel : bool;
            replace the transit model with the custom model. Default is False.
        
        Returns:
        --------
        custom_RVfunc : SimpleNamespace;
            namespace object containing the custom function, operation function and parameter dictionary.

        Examples:
        --------
        # 1. create a custom function in phase angle that adds sinusoidal components to the light curve model
        >>> def custom_func(phase_angle, A, B, C,extra_args={}):     # A,B,C are the custom parameters and no extra arguments
        >>>     return A*np.sin(phase_angle) + B*np.cos(2*phase_angle) + C

        >>> def op_func(transit_model, custom_model):   # operation function to combine the custom model with the transit model
        >>>     return transit_model + custom_model

        >>> # A is fixed, B has uniform prior, C has gaussian prior
        >>> custom_lcfunc = lc_obj.add_custom_LC_function(func=custom_func, x="phase_angle",func_args=dict(A=0.1, B=(0.2,0,0.01), C=(0.3,0.01), op_func=op_func)
        >>> # this custom function has now been registered and will be used in the light-curve model.

        # 2. replace the transit model with a custom model that has two new parameters beyond the standard transit model parameters.
        # here we create a custom model from `catwoman` that models asymmetric transit light curves using two new parameters rp2 and phi.
        # the limb darkening law to use in catwoman  can be passed in the extra_args dictionary.
        >>> LC_pars=dict(Duration=0,rho_star=0,RpRs=0,Impact_para=0,T_0=0,Period=0,Eccentricity=0,omega=90,q1=0,q2=0,D_occ=0,Fn=0,ph_off=0,A_ev=0,A_db=0)
        >>> def catwoman_func(t, rp2, phi, extra_args=dict(ld_law="quadratic"),LC_pars=LC_pars):
        >>>     import catwoman 
        >>>     import numpy as np
        >>>     import astropy.constants as c
        >>>     import astropy.units as u
        >>>     
        >>>     # create a catwoman model. CONAN transit pars in LDpars can be passed to the function
        >>>     params  = catwoman.TransitParams()
        >>>     params.t0  = LD_pars["T_0"]          
        >>>     params.per = LD_pars["Period"]    
        >>>     params.rp  = LD_pars["RpRs"] 
        >>>     #convert stellar density to a/R*
        >>>     G  = (c.G.to(u.cm**3/(u.g*u.second**2))).value
        >>>     aR = ( rho*G*(P*(u.day.to(u.second)))**2 / (3*np.pi)) **(1/3.)
        >>>     params.a   =  aR                          
        >>>     params.inc = np.arccos(LD_pars["Impact_para"]/aR)   
        >>>     params.ecc = LD_pars["Eccentricity"]   
        >>>     params.w   = LD_pars["omega"]             
        >>>     params.limb_dark = extra_args["ld_law"]
        >>>     #convert from kipping parameterisation to quadratic
        >>>     u1 = 2*np.sqrt(LD_pars["q1"])*LD_pars["q2"]  
        >>>     u2 = np.sqrt(LD_pars["q1"])*(1-2*LD_pars["q2"]) 
        >>>     params.u = [u1, u2]    

        >>>     params.phi = phi                        #angle of rotation of top semi-circle (in degrees)
        >>>     params.rp2 = rp2                        #bottom semi-circle radius (in units of stellar radii)
        >>>     
        >>>     model = catwoman.TransitModel(params,t)         #initalises model
        >>>     return  model.light_curve(params)                #calculates light curve
        >>> 
        >>> lc_obj.add_custom_LC_function(func=catwoman_func, x="time",func_args=dict(rp2=(0.1,0.01), phi=(-90,0,90)),extra_args=dict(ld_law="quadratic") op_func=None,replace_RVmodel=True)
        
        """

        if func!=None:
            assert callable(func), "add_custom_RV_function(): func must be a callable function."
            assert isinstance(func_args, dict), "add_custom_RV_function(): func_args must be a dictionary."
            if extra_args == None: extra_args=dict()
            assert isinstance(extra_args, dict), "add_custom_RV_function(): extra_args must be a dictionary."
            assert x in ["time","true_anomaly"], "add_custom_RV_function(): x must be 'time' or 'true_anomaly'."

            #check that opfunc takes two array arguments
            if not replace_RVmodel: 
                assert callable(op_func), "add_custom_RV_function(): op_func must be a callable function."
                assert len(inspect.signature(op_func).parameters)==2, f"add_custom_RV_function(): op_func must take two arguments but {len(inspect.signature(op_func).parameters)} given."
                #assert that number of arguments in func() is equal to the number of parameters in func_args + the independent variable x + extra_args dict
                assert len(inspect.signature(func).parameters)==len(func_args)+2, f"add_custom_RV_function(): number of arguments in func must be equal to number of parameters in func_args + the independent variable + extra_args."
            else:
                # assert RV_pars argument in func
                assert "RV_pars" in inspect.signature(func).parameters, "add_custom_RV_function(): RV_pars dictionary argument must be in func in order to replace native transit model."
                #assert RV_pars argument is a dict with keys in planet_parameters
                tp_arg = inspect.signature(func).parameters["RV_pars"].default
                assert isinstance(tp_arg, dict) and len(tp_arg)==5, "add_custom_RV_function(): RV_pars argument in func must be a dictionary with 5 keys i.e. RV_pars=dict(T_0=0,Period=0,Eccentricity=0,omega=90,K=0)"
                assert all([k in tp_arg.keys() for k in ["T_0","Period","Eccentricity","omega","K"]]), \
                    'add_custom_RV_function(): RV_pars argument in func must same keys as planet_parameters i.e. RV_pars=dict(T_0=0,Period=0,Eccentricity=0,omega=90,K=0).'
                #assert that number of arguments in func() is equal to the number of parameters in func_args + the independent variable x + extra_args + RV_pars
                assert len(inspect.signature(func).parameters)==len(func_args)+3, f"add_custom_RV_function(): number of arguments in func_args must be equal to number of arguments in func minus 2."
        else: 
            self._custom_RVfunc = SimpleNamespace(func=None,x=None,op_func=None, func_args={},extra_args={},par_dict={},npars=0,nfree=0,replace_RVmodel=False)
            if verbose: _print_output(self,"custom_RVfunction")
            return None
        
        par_dict = {}   #dictionary to store parameter objects
        nfree    = 0
        for k in func_args.keys():
            if isinstance(func_args[k], tuple): nfree += 1
            par_dict[k] = _param_obj.from_tuple(func_args[k], user_input=func_args[k], func_call="add_custom_RV_function():")

        self._custom_RVfunc = SimpleNamespace(func=func, x=x,op_func=op_func, func_args=func_args,extra_args=extra_args,par_dict=par_dict, npars=len(par_dict),nfree=nfree,replace_RVmodel=replace_RVmodel)

        if verbose: _print_output(self,"custom_RVfunction")
        return deepcopy(self._custom_RVfunc)


    def add_rvGP(self ,rv_list=None, par=[], kernel=[], operation=[""],
                    amplitude=[], lengthscale=[], verbose=True):
        """  
            Define GP parameters for each RV.
            The GP parameters, amplitude in RV unit  and lengthscale in unit of the desired column) can be defined in following ways:
            * fixed value as float or int, e.g amplitude = 2
            * free parameter with lognormal prior given as tuple of len 2, e.g. amplitude = (2, 1)
            * free parameters with loguniform prior interval and initial value given as tuple of length 3, e.g. amplitude = (1,2,5) with 2 being the initial value.
            
            Here the amplitude corresponds to the standard deviation of the noise process and the lengthscale corresponds to the characteristic timescale of the noise process.
            
            For the celerite sho kernel, the quality factor Q has been fixed to 1/sqrt(2) which is commonly used to model stellar oscillations/granulation noise (eqn 24 celerite paper).
            this lengthscale here is the undamped period of the oscillator.

            For the cosine kernel in George, lengthscale is the period. This kernel should probably always be multiplied by a stationary kernel (e.g. ExpSquaredKernel) to allow quasi-periodic variations.

            Parameters:
            -----------
            rv_list : str, list;
                list of rv files to add GP to. Default is None for no GP. if "all" is given, GP is added to all rv files where gp use has been indicated in ``rv_baseline()``. 
                If "same" is given, a global (same) GP is used for all indicated RV files in ``rv_baseline()``.
            par : str, tuple, list;
                column of the input data to use as the GP independent variable. a list is expected if different columns are to be used for the rv files given in rv_list.
                to use 2 different kernels on a single rv file, give column name for each kernel as a tuple of length 2. 
                e.g. rv_list=["rv1.dat","rv2.dat"], par = [("col0","col0"),"col3"] to use col0 for both kernels of rv1, and col3 for rv2.
            kernel : str, tuple, list;
                kernel to use for the GP. Must be one of ["mat32","mat52","exp","expsq","cos"] if George package is selected  and one of ["real","mat32","sho"] if using Celerite package
                A list is expected if different kernels are to be used for the rv files given in rv_list.
                to use 2 different kernels on a single rv file, give kernel name for each kernel as a tuple of length 2.
                e.g. rv_list=["rv1.dat","rv2.dat"], kernel = [("mat32","expsq"),"exp"] to use mat32 and expsq for rv1, and exp for rv2.
            operation : str, tuple, list;
                operation to combine 2 kernels. Must be one of ["+","*"]. Default is "" for no combination.
            amplitude : float, tuple, list;
                amplitude of the GP kernel. Must be int/float or tuple of length 2 or 3
            lengthscale : float, tuple, list;
                lengthscale of the GP kernel. Must be int/float or tuple of length 2 or 3
            verbose : bool;
                print output. Default is True.        
        """
        # supported 2-hyperparameter kernels
        george_allowed   = dict(kernels = ["mat32","mat52","exp","expsq","cos"],columns= ["col0","col3","col4","col5"])
        celerite_allowed = dict(kernels = ["real","mat32","sho"], columns=["col0"])

        self._rvGP_dict = {}
        self._sameRVgp  = SimpleNamespace(flag = False, first_index =None)

        if rv_list is None or rv_list == []:
            if self._nRV>0:
                if len(self._gp_rvs())>0: print(f"\nWarning: GP was expected for the following rvs {self._gp_rvs()} \nMoving on ...")
                if verbose:_print_output(self,"rv_gp")
            return
        elif isinstance(rv_list, str):
            if rv_list == "same": 
                self._sameRVgp.flag        = True 
                self._sameRVgp.first_index = self._names.index(self._gp_rvs()[0])
            if rv_list in ["all","same"]: 
                rv_list = self._gp_rvs()
            else: rv_list=[rv_list]


        for rv in self._gp_rvs(): assert rv in rv_list,f"add_rvGP(): GP was expected for {rv} but was not given in rv_list {rv_list}."
        for rv in rv_list: 
            assert rv in self._names,f"add_rvGP(): {rv} not in loaded rv files."
            assert rv in self._gp_rvs(),f"add_rvGP(): GP was not expected for {rv} but was given in rv_list."
        
        rv_ind = [self._names.index(rv) for rv in rv_list]
        gp_pck = [self._useGPrv[i] for i in rv_ind]   #gp_pck is a list of "y" or "ce" for each rv in rv_list

        DA = locals().copy()
        _  = [DA.pop(item) for item in ["self","verbose"]]

        for p in ["par","kernel","operation","amplitude","lengthscale"]:
            if isinstance(DA[p], (str,int,float,tuple)): DA[p] = [DA[p]]   #convert to list
            if isinstance(DA[p], list): 
                if self._sameRVgp.flag:    #ensure same inputs for all rvs with indicated gp
                    assert len(DA[p])==1 or len(DA[p])==len(rv_list), f"add_rvGP(): {p} must be a list of length {len(rv_list)} or length 1 (if same is to be used for all RVs)."
                    if len(DA[p])==2: assert DA[p][0] == DA[p][1],f"add_rvGP(): {p} must be same for all rv files if sameGP is used."
                if len(DA[p])==1: DA[p] = DA[p]*len(rv_list)
            
            assert len(DA[p])==len(rv_list), f"add_rvGP(): {p} must be a list of length {len(rv_list)} or length 1 (if same is to be used for all RVs)."
            
        for p in ["par","kernel","operation","amplitude","lengthscale"]:
            #check if inputs for p are valid
            for i,list_item in enumerate(DA[p]):
                if p=="par":
                    if isinstance(list_item, str): 
                        if gp_pck[i]=="y":  assert list_item in george_allowed["columns"],  f'add_rvGP(): inputs of {p} must be in {george_allowed["columns"]}   but {list_item} given.'
                        if gp_pck[i]=="ce": assert list_item in celerite_allowed["columns"],f'add_rvGP(): inputs of {p} must be in {celerite_allowed["columns"]} but {list_item} given.'
                        DA["operation"][i] = ""
                    elif isinstance(list_item, tuple): 
                        assert len(list_item)==2,f'add_rvGP(): max of 2 gp kernels can be combined, but {list_item} given in {p}.'
                        assert DA["operation"][i] in ["+","*"],f'add_rvGP(): operation must be one of ["+","*"] to combine 2 kernels but {DA["operation"][i]} given.'
                        for tup_item in list_item: 
                            if gp_pck[i]=="y":  assert tup_item in george_allowed["columns"],  f'add_rvGP(): {p} must be in {george_allowed["columns"]}   but {tup_item} given.'
                            if gp_pck[i]=="ce": assert tup_item in celerite_allowed["columns"],f'add_rvGP(): {p} must be in {celerite_allowed["columns"]} but {tup_item} given.'
                        # assert that a tuple of length 2 is also given for kernels, amplitude and lengthscale.
                        for chk_p in ["kernel","amplitude","lengthscale"]:
                            assert isinstance(DA[chk_p][i], tuple) and len(DA[chk_p][i])==2,f'add_rvGP(): expected tuple of len 2 for {chk_p} element {i} but {DA[chk_p][i]} given.'
                            
                    else: _raise(TypeError, f"add_rvGP(): elements of {p} must be a tuple of length 2 or str but {list_item} given.")
                
                if p=="kernel":
                    if isinstance(list_item, str): 
                        if gp_pck[i]=="y":  assert list_item in george_allowed["kernels"],  f'add_rvGP(): {p} must be one of {george_allowed["kernels"]}   but {list_item} given.'
                        if gp_pck[i]=="ce": assert list_item in celerite_allowed["kernels"],f'add_rvGP(): {p} must be one of {celerite_allowed["kernels"]} but {list_item} given.'
                    elif isinstance(list_item, tuple):
                        for tup_item in list_item: 
                            if gp_pck[i]=="y":  assert tup_item in george_allowed["kernels"],  f'add_rvGP(): {p} must be one of {george_allowed["kernels"]}   but {tup_item} given.'
                            if gp_pck[i]=="ce": assert tup_item in celerite_allowed["kernels"],f'add_rvGP(): {p} must be one of {celerite_allowed["kernels"]} but {tup_item} given.'
                    else: _raise(TypeError, f"add_rvGP(): elements of {p} must be a tuple of length 2 or str but {list_item} given.")

                if p=="operation":
                    assert list_item in ["+","*",""],f'add_rvGP(): {p} must be one of ["+","*",""] but {list_item} given.'

                if p in ["amplitude", "lengthscale"]:
                    if isinstance(list_item, (int,float)): pass
                    elif isinstance(list_item, tuple):
                        if isinstance(DA["par"][i],tuple):
                            for tup in list_item:
                                if isinstance(tup, (int,float)): pass
                                elif isinstance(tup, tuple): 
                                    assert len(tup) in [2,3],f'add_rvGP(): {p} must be a float/int or tuple of length 2/3 but {tup} given.'
                                    if len(tup)==3: assert tup[0]<tup[1]<tup[2],f'add_rvGP(): uniform prior for {p} must follow (min, start, max) but {tup} given.'
                                else: _raise(TypeError, f"add_rvGP(): elements of {p} must be a tuple of length 23 or float/int but {tup} given.")
                        else:
                            assert len(list_item) in [2,3],f'add_rvGP(): {p} must be a float/int or tuple of length 2/3 but {tup} given.'
                            if len(list_item)==3: assert list_item[0]<list_item[1]<list_item[2],f'add_rvGP(): uniform prior for {p} must follow (min, start, max) but {list_item} given.'
                    else: _raise(TypeError, f"add_rvGP(): elements of {p} must be a tuple of length 2/3 or float/int but {list_item} given.")


        #setup parameter objects
        for i,rv in enumerate(rv_list):
            self._rvGP_dict[rv] = {}
            ngp = 2 if isinstance(DA["kernel"][i],tuple) else 1
            self._rvGP_dict[rv]["ngp"] = ngp
            self._rvGP_dict[rv]["op"]  = DA["operation"][i]

            if self._rvGP_dict[rv]["op"] == "*" and self._rvGP_dict[rv]["ngp"] == 2:
                assert DA["amplitude"][i][1] == 1, f"add_rvGP(): for multiplication of 2 kernels, the second amplitude must be fixed to 1 to avoid degeneracy, but {DA['amplitude'][i][1]} given."

            for p in ["amplitude", "lengthscale"]:
                for j in range(ngp):
                    if ngp==1: 
                        v = DA[p][i]
                        this_kern, this_par = DA["kernel"][i], DA["par"][i]
                    else:
                        v = DA[p][i][j]
                        this_kern, this_par = DA["kernel"][i][j], DA["par"][i][j]

                    if isinstance(v, (int,float)):
                        self._rvGP_dict[rv][p+str(j)]     = _param_obj(to_fit="n", start_value=v,step_size=0,
                                                                        prior="n", prior_mean=v, prior_width_lo=0,
                                                                        prior_width_hi=0, bounds_lo=0.01, bounds_hi=0,
                                                                        user_data = [this_kern, this_par])
                    elif isinstance(v, tuple):
                        if len(v)==2:
                            steps = 0 if (self._sameRVgp.flag and i!=0) else 0.1*v[1]   #if sameRVgp is set, only first pars will jump and be used for all rvs
                            self._rvGP_dict[rv][p+str(j)] = _param_obj(to_fit="y", start_value=v[0],step_size=steps,prior="p", 
                                                                        prior_mean=v[0], prior_width_lo=v[1], prior_width_hi=v[1], 
                                                                        bounds_lo=v[0]-10*v[1], bounds_hi=v[0]+10*v[1],
                                                                        user_data=[this_kern, this_par])
                        elif len(v)==3:
                            steps = 0 if (self._sameRVgp.flag and i!=0) else min(0.001,0.001*np.ptp(v))
                            self._rvGP_dict[rv][p+str(j)] = _param_obj(to_fit="y", start_value=v[1],step_size=steps,
                                                                        prior="n", prior_mean=v[1], prior_width_lo=0,
                                                                        prior_width_hi=0, bounds_lo=v[0] if v[0]>0 else 0.007, bounds_hi=v[2],
                                                                        user_data=[this_kern, this_par])

                    else: _raise(TypeError, f"add_rvGP(): elements of {p} must be a tuple of length 2/3 or float/int but {v} given.")

        if verbose: _print_output(self,"rv_gp")

    def add_custom_rvGP( self, rv_list=None, module=None, kernel=[], par=[], GP_params=dict(), verbose=True):
        
        #user will need to provide the GPmodule to import, relevant kernel and GP parameters, how to get the loglikelihood and get samples(predict)

        #import module using the string name
        GPmodule = __import__(module)

        raise NotImplementedError

    
    def add_spline(self, rv_list=None, par = None, degree=3, knot_spacing=None, verbose=True):
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
            knot_spacing : float, tuple, list
                distance between knots of the spline, by default 15 degrees for cheops data roll-angle 
            verbose : bool, optional
                print output. Default is True.

            Examples
            --------
            To use different spline configuration for 2 rv files: 2D spline for the first file and 1D for the second.
            >>> rv_obj.add_spline(rv_list=["rv1.dat","rv2.dat"], par=[("col3","col4"),"col4"], degree=[(3,3),2], knot_spacing=[(5,3),2])
            
            For same spline configuration for all loaded RV files
            >>> rv_obj.add_spline(rv_list="all", par="col3", degree=3, knot_spacing=5)
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
            self._rvspline[i].conf   = "None"

        if rv_list is None:
            if verbose: print("No spline\n")
            return
        elif rv_list == "all":
            rv_list = self._names
        else:
            if isinstance(rv_list, str): rv_list = [rv_list]
        
        nrv_spl = len(rv_list)   #number of rvs to add spline to
        for rv in rv_list:
            assert rv in self._names, f"add_spline(): {rv} not in loaded rv files: {self._names}."
        
        DA = locals().copy()
        _ = [DA.pop(item) for item in ["self", "verbose"]]  

        for p in ["par","degree","knot_spacing"]:
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
            par, deg, knots =  DA["par"][i], DA["degree"][i], DA["knot_spacing"][i]
            dim = 1 if isinstance(par,str) else len(par)
            assert dim <=2, f"add_spline(): dimension of spline must be 1 or 2 but {par} (dim {dim}) given for {rv}."
            if dim==2:   #if 2d spline 
                if isinstance(deg, int): deg = (deg,deg)  #if degree is int, make it a tuple
                if isinstance(knots, int): knots = (knots,knots)

            self._rvspline[ind].name   = rv
            self._rvspline[ind].dim    = dim
            self._rvspline[ind].par    = par
            self._rvspline[ind].use    = True if par else False
            self._rvspline[ind].deg    = deg
            self._rvspline[ind].knots  = knots
            
            if dim==1:
                assert knots <= np.ptp(self._input_rv[rv][par]), f"add_spline():{rv} – knot_spacing must be less than the range of the column array but {knots} given for {par} with range of {np.ptp(self._input_rv[rv][par])}."
                assert deg <=5, f"add_spline():{rv} – degree must be <=5 but {deg} given for {par}."
                self._rvspline[ind].conf   = f"c{par[-1]}:d{deg}:k{knots}"
            else:
                for j in range(2):
                    assert deg[j] <=5, f"add_spline():{rv} – degree must be <=5 but {deg[j]} given for {par[j]}."
                    assert knots[j] <= np.ptp(self._input_rv[rv][par[j]]), f"add_spline():{rv} – knot_spacing must be less than the range of the column array but {knots[j]} given for {par[j]} with range of {np.ptp(self._input_rv[rv][par[j]])}."
                self._rvspline[ind].conf   = f"c{par[0][-1]}:d{deg[0]}k{knots[0]}|c{par[1][-1]}:d{deg[1]}k{knots[1]}"

            if verbose: print(f"{rv} – degree {deg} spline to fit {par}: knot spacing= {knots}")
        
        if verbose: _print_output(self,"rv_baseline")
    
    def __repr__(self):
        data_type = str(self.__class__).split("load_")[1].split("'>")[0]
        self.print("rv_baseline")
        if self._nRV>0:
            return f'{data_type} from filepath: {self._fpath}\n{self._nplanet} planet(s)\n'
        else:
            return ""
        
    def plot(self, plot_cols=(0,1,2), col_labels=None, nrow_ncols=None, figsize=None, fit_order=0, 
                show_decorr_model=False, detrend=False, hspace=None, wspace=None, binsize=0.,return_fig=False):
        """
            visualize data

            Parameters:
            -----------
            plot_cols : tuple of length 3;
                Tuple specifying which columns in input file to plot. Default is (0,1,2) to plot time, flux with fluxerr. 
                Use (3,1,2) to show the correlation between the 4th column and the flux. 
                if decorrelation has been done with `lmfit`, the "res" can also be given to plot a column against the residual of the fit.
            col_labels : tuple of length 2;
                label of the given columns in plot_cols. Default is ("time", "rv").
            nrow_ncols : tuple of length 2;
                Number of rows and columns to plot the input files. 
                Default is (None, None) to find the best layout.
            fit_order : int;
                order of polynomial to fit to the plotted data columns to visualize correlation.
            show_decorr_model : bool;
                show decorrelation model if decorrelation has been done.
            detrend : bool;
                plot the detrended data. Default is False.
            hspace, wspace: float;
                height and width space between subplots. Default is None to use matplotlib defaults.
            binsize : float;
                binsize to use for binning the data in time. Default is 0.0104 (15mins).
            figsize: tuple of length 2;
                Figure size. If None, (8,5) is used for a single input file and optimally determined for more inputs.
            return_fig  : bool;
                return figure object for saving to file.
        """

        if not (isinstance(plot_cols, tuple) and len(plot_cols) in [2,3]): 
            raise TypeError(f"plot_cols must be tuple of length 2 or 3, but is {type(plot_cols)} and length of {len(plot_cols)}.")
        if detrend: assert show_decorr_model, "plot(): detrend can only be True if decorrelation has been done."
        if plot_cols[1] == "res": assert show_decorr_model, "plot(): plot_cols[1] can only be 'res' if decorrelation has been done, and show_decorr_model=True."
        
        assert col_labels is None or ((isinstance(col_labels, tuple) and len(col_labels)==2)), \
            f"col_labels must be tuple of length 2, but is {type(col_labels)} and length of {len(col_labels)}."
        
        assert isinstance(fit_order,int),f'fit_order must be an integer'

        if show_decorr_model:
            if not hasattr(self,"_rvmodel"): 
                print("cannot show decorr model since decorrelation has not been done. First, use `rv_obj.get_decorr()` to launch decorrelation.")
                show_decorr_model = False
        
        if col_labels is None:
            col_labels = ("time", "rv") if plot_cols[:2] == (0,1) else (f"column[{plot_cols[0]}]",f"column[{plot_cols[1]}]")
        
        if self._names != []:
            fig = _plot_data(self, plot_cols=plot_cols, col_labels = col_labels, nrow_ncols=nrow_ncols, fit_order=fit_order, figsize=figsize,
                            hspace=hspace, wspace=wspace, model_overplot=self._rvmodel if show_decorr_model else None, detrend=detrend, binsize=binsize)
            if return_fig: return fig
        else: print("No data to plot")

    def save_RVs(self, save_path=None, overwrite=False, verbose=True):
        """
            Save the loaded rvs to file.

            Parameters:
            -----------
            save_path : str;
                path to save the rv data. Default is None to save to a folder 'data_preproc/' in current working directory.
            overwrite : bool;
                overwrite the existing file. Default is False.
            verbose : bool;
                print output. Default is True.
        """
        if save_path is None: save_path = 'data_preproc/'
        for k,rv in self._input_rv.items():
            if not os.path.exists(save_path): os.makedirs(save_path)
            if not overwrite:
                if os.path.exists(save_path+k): 
                    print(f"save_RVs(): {save_path+k} already exists. Set `overwrite=True` to overwrite the file.")
                    return
            #save dict lc as a .dat file 
            pd.DataFrame(rv).to_csv(save_path+k, index=False, sep="\t", header=False,float_format='%-16.6f')
            if verbose: print(f"save_RVs(): {save_path+k} saved.")


    
    def print(self, section="all"):
        """
            Print out all input configuration (or particular section) for the RV object. 
            It is printed out in the format of the legacy CONAN config file.
            
            Parameters:
            ------------
            section : str (optional) ;
                section of configuration to print. Must be one of ["rv_baseline", "rv_gp", "custom_RVfunction"].
                Default is 'all' to print all sections.
        """
        if section=="all":
            _print_output(self,"rv_baseline")
            _print_output(self,"rv_gp")
            if hasattr(self,"_lcobj"): _print_output(self._lcobj,"planet_parameters")
            _print_output(self,"custom_RVfunction")
        else:
            possible_sections = ["rv_baseline", "rv_gp", "custom_RVfunction","planet_parameters"]
            assert section in possible_sections, f"print(): section must be one of {possible_sections} but {section} given."
            if section == "planet_parameters":
                assert hasattr(self,"_lcobj"), "print(): planet_parameters section can only be printed for RVs if light curve data has been loaded. Use `lc_obj` argument in `load_rvs()` to load in the  lightcurve object."
                _print_output(self._lcobj,section)
            else:
                _print_output(self,section)

    
class fit_setup:
    """
        class to configure mcmc run
            
        Parameters:
        ------------
        R_st, Mst : tuple of length 2 ;
            stellar radius and mass (in solar units) to use for calculating absolute dimensions. R_st is also used in calculating light travel time correction.
            Only one of these is needed, preferrably R_st. First tuple element is the value and the second is the uncertainty
        par_input : str;
            input method of stellar parameters. It can be one of  ["Rrho","Mrho"], to use the fitted stellar density and one stellar parameter (M_st or R_st) to compute the other stellar parameter (R_st or M_st).
            Default is 'Rrho' to use the fitted stellar density and stellar radius to compute the stellar mass.   
        leastsq_for_basepar: "y" or "n";
            whether to use least-squares fit within the mcmc to fit for the baseline. This reduces +\
            the computation time especially in cases with several input files. Default is "n".
        apply_RVjitter: "y" or "n";
            whether to apply a jitter term for the fit of RV data. Default is "y".
        apply_LCjitter: "y" or "n";
            whether to apply a jitter term for the fit of LC data. Default is "y".
        LCjitter_loglims: "auto" or list of length 2: [lo_lim,hi_lim];
            log limits of uniform prior for the LC jitter term. 
            Default is "auto" which automatically determines the limits for each lcfile as [-15,log(10*mean(LCerr))].
        RVjitter_lims: "auto" or list of length 2:[lo_lim,hi_lim];
            limits of uniform prior for the RV jitter term. 
            Default is "auto" which automatically determines the limits for each rvfile as [0,10*mean(RVerr)].
        LCbasecoeff_lims: "auto" or list of length 2: [lo_lim,hi_lim];
            limits of uniform prior for the LC baseline coefficients default.
            Dafault is "auto" which automatically determines the limits from data properties.
        RVbasecoeff_lims: "auto" or list of length 2: [lo_lim,hi_lim];
            limits of uniform prior for the RV baseline coefficients. 
            Dafault is "auto" which automatically determines the limits from data properties.
    
        Other keyword arguments to the emcee or dynesty sampler functions (`run_mcmc()` or `run_nested()`) can be given in the call to `CONAN3.fit_data`.
        
        Returns:
        --------
        fit_obj : fit object

        Examples:
        ---------
        >>> fit_obj = CONAN3.fit_setup(R_st=(1,0.01), M_st=(1,0.01), par_input="Rrho", apply_LCjitter="y", apply_RVjitter="y")
            fit_obj.sampling(sampler="emcee", ncpus=2,n_chains=64, n_steps=2000, n_burn=500)
    """

    def __init__(self, R_st=None, M_st=None, par_input = "Rrho",
                    apply_LCjitter="y", apply_RVjitter="y", 
                    LCjitter_loglims="auto", RVjitter_lims="auto",
                    LCbasecoeff_lims = "auto", RVbasecoeff_lims = "auto", 
                    leastsq_for_basepar="n", LTT_corr = "n", verbose=True):
        
        self._obj_type = "fit_obj"
        if LTT_corr=="y": assert R_st!=None, "fit_setup(): R_st is needed to compute light travel time correction."
        
        self._stellar_parameters(R_st=R_st, M_st=M_st, par_input = par_input, verbose=verbose)
        for val in [LCjitter_loglims,RVjitter_lims,LCbasecoeff_lims, RVbasecoeff_lims]:
            assert isinstance(val, (str,list)), f"fit_setup(): inputs for the different limits must be a list or 'auto' but {val} given."
            if isinstance(val, list): assert len(val)==2, f"fit_setup(): inputs for the different limits must be a list of length 2 or 'auto' but {val} given."
            if isinstance(val, str): assert val=="auto", f"fit_setup(): inputs for the different limits must be a list of length 2 or 'auto' but {val} given."

        DA = locals().copy()
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")
        
        self._fit_dict = DA
        self.sampling(verbose=False)

    def sampling(self, sampler="dynesty", n_cpus=4,
                    n_chains=64, n_steps=2000, n_burn=500, emcee_move="stretch",  
                    n_live=300, dyn_dlogz=0.1, force_nlive=False, nested_sampling="static",
                    verbose=True):
        """   
        configure sampling

        Parameters:
        -----------
        sampler: str;
            sampler to use. Default is "dynesty". Options are ["emcee","dynesty"].
        n_cpus: int;
            number of cpus to use for parallelization.
        n_chains: int;
            number of chains/walkers
        n_steps: int;
            length of each chain. the effective total steps becomes n_steps*n_chains.
        n_burn: int;
            number of steps to discard as burn-in
        emcee_move: str;
            sampler algorithm to use in traversing the parameter space. Options are ["demc","snooker",stretch].
            The default is stretch to use the emcee StretchMove.
        n_live: int;
            number of live points to use for dynesty sampler. Default is 300.
        dyn_dlogz: float;
            stopping criterion for dynesty sampler. Default is 0.1.
        force_nlive: bool;
            force dynesty to use n_live even if less than the required ndim*(ndim+1)//2. Default is False.  
        nested_sampling: str;
            type of nested sampling to use. Default is "static". 
            Options are ["static","dynamic[pfrac]"] where pfrac is a float from [0, 1] that determine the posterior.evidence fraction.
            "dynamic[1.0]" performs sampling optimized for 100% posterior evaluation and "dynamic[0.8]" is 80% posterior, 20% evidence.
        verbose: bool;
            print output. Default is True.
        """
        apply_CFs="y"
        remove_param_for_CNM="n"

        assert sampler in ["emcee","dynesty"],f'sampler must be one of ["emcee","dynesty"] but {sampler} given.'
        assert emcee_move in ["demc","snooker","stretch"],f'emcee_move must be one of ["demc","snooker","stretch] but {emcee_move} given.'
        assert nested_sampling=="static" or "dynamic" in nested_sampling,f'nested_sampling must be one of ["static","dynamic[pfrac]"] but {nested_sampling} given.'
        if "dynamic" in nested_sampling:
            assert "[" in nested_sampling and "]" in nested_sampling,f'for dynamic nested_sampling must specified in the form "dynamic[pfrac]" but {nested_sampling} given.'
            pfrac = float(nested_sampling.split("[")[1].split("]")[0])
            assert 0<=pfrac<=1,f'pfrac in nested_sampling must be a float from [0,1] but {pfrac} given.'
        
        DA = locals().copy()
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")
        self._obj_type = "fit_obj"
        self._fit_dict.update(DA)

        if verbose: _print_output(self,"fit")

    def _stellar_parameters(self,R_st=None, M_st=None, par_input = "Rrho", verbose=True):
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

        DA = locals().copy()
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")
        
        for par in ["R_st", "M_st"]:
            assert DA[par] is None or isinstance(DA[par],tuple), f"stellar_parameters: {par} must be either None or tuple of length 2 "
            if DA[par] is None: DA[par] = (None,None) if par=="M_st" else (1,None) # set R_st to 1 if not given
            elif isinstance(DA[par],tuple):
                assert len(DA[par])==2, f"stellar_parameters(): length of {par} tuple must be 2"
                DA[par]= (DA[par][0], DA[par][1], DA[par][1])
            else: _raise(TypeError, f"stellar_parameters(): {par} must be either None or tuple of length 2 but {DA[par]} given.")
        
        assert DA["par_input"] in ["Rrho","Mrho"], f"stellar_parameters: par_input must be one of ['Rrho','Mrho']."
        self._stellar_dict = DA

        if verbose: _print_output(self,"stellar_pars")

    def __repr__(self):
        if self._fit_dict['sampler'] == "emcee":
            return f"Sampling setup:\n  sampler: {self._fit_dict['sampler']}\n  steps: {self._fit_dict['n_steps']}\n  chains: {self._fit_dict['n_chains']}"
        else:
            return f"Sampling setup:\n  sampler: {self._fit_dict['sampler']}\n  live points: {self._fit_dict['n_live']}\n  dlogz: {self._fit_dict['dyn_dlogz']}"

    def print(self, section="fit"):
        if section == "all":
            _print_output(self,"stellar_pars")
            _print_output(self,"fit")
        else:
            _print_output(self, section)


class load_result:
    """
        Load results from emcee/dynesty run
        
        Parameters:
        ------------
        folder: str;
            folder where the output files are located. Default is "output".
        chain_file: str;
            name of the file containing the posterior chains. Default is "chains_dict.pkl".
        burnin_chain_file: str;
            name of the file containing the burn-in chains. Default is "burnin_chains_dict.pkl".
        
        Returns:
        --------
        load_result : load_result object

        Examples:
        ---------
        >>> result = CONAN3.load_result(folder="output")
        # different plots from the result object
        >>> fig    = result.plot_corner()                       # corner plot
        >>> fig    = result.plot_burnin_chains()                # burn-in chains
        >>> fig    = result.plot_chains()                       # posterior chains
        >>> fig    = result.lc.plot_bestfit(detrend=True)        # model of the light curves
        >>> fig    = result.rv.plot_bestfit(detrend=True)        # model of the RV curves
        >>> fig    = result.lc.plot_ttv()                        # plot the TTVs
        >>> fig    = result.lc.plot_lcttv()                      # plot the light curves showing the TTVs

        #get the best-fit parameters
        >>> med_pars = result.params.median        # median values of the fitted parameters
        >>> stdev    = result.params.stdev         # standard deviation of the fitted parameters
        >>> pars_dict= result.get_all_params_dict(stat="med")              # get all parameters (fitted, derived, and fixed) as a dictionary
        
        # load files
        >>> out_lc  = result.lc.out_data()            # output data of the light curves i.e *_lcout.dat files
        >>> out_rv  = result.rv.out_data()            # output data of the RV curves i.e *_rvout.dat files
        >>> in_lc   = result.lc.in_data()             # input light curves
        >>> in_rv   = result.rv.in_data()             # input RV data 

        # evaluate model (lc or rv) at user-defined times
        >>> t = np.linspace(0,1,1000)
        >>> model  = result.lc.evaluate(file="lc1.dat", time=t, params= result.params.median, return_std=True)                # model of the light curve "lc1.dat" at user time t
        >>> lc_mod = model.planet_model      # model of the planet
        >>> comps  = model.components        # for multiplanet fit, this will be a dict with lc_mod for each planet. i.e. comps["pl_1"] for planet 1 
        >>> sigma_low, sigma_hi = model.sigma_low, model.sigma_hi    # lower and upper 1-sigma model uncertainties that can be plotted along with lc_mod
    """

    def __init__(self, folder="output",chain_file = "chains_dict.pkl", burnin_chain_file="burnin_chains_dict.pkl",verbose=True):

        chain_file        = folder+"/"+chain_file
        burnin_chain_file = folder+"/"+burnin_chain_file
        self._folder      = folder
        assert os.path.exists(chain_file) or os.path.exists(burnin_chain_file) , f"file {chain_file} or {burnin_chain_file}  does not exist in the given directory"

        if os.path.exists(chain_file):
            self._chains = pickle.load(open(chain_file,"rb"))
        if os.path.exists(burnin_chain_file):
            self._burnin_chains = pickle.load(open(burnin_chain_file,"rb"))

        self._obj_type      = "result_obj"
        self._par_names     = self._chains.keys() if os.path.exists(chain_file) else self._burnin_chains.keys()

        #retrieve configration of the fit
        self._ind_para      = pickle.load(open(folder+"/.par_config.pkl","rb"))
        self._lcnames       = self._ind_para["LCnames"]
        self._rvnames       = self._ind_para["RVnames"]
        self._nplanet       = self._ind_para["npl"]
        input_lcs           = self._ind_para["input_lcs"]
        input_rvs           = self._ind_para["input_rvs"]
        self.fit_sampler    = self._ind_para["fit_sampler"]
        self._ttvs          = self._ind_para["ttv_conf"]

        assert list(self._par_names) == list(self._ind_para["jnames"]),'load_result(): the fitting parameters do not match those saved in the chains_dict.pkl file' + \
            f'\nThey differ in these parameters: {list(set(self._par_names).symmetric_difference(set(self._ind_para["jnames"])))}.'

        if not hasattr(self,"_chains"):
            return

        #reconstruct posterior from dictionary of chains
        if self.fit_sampler=="emcee":
            posterior = np.array([ch for k,ch in self._chains.items()])
            posterior = np.moveaxis(posterior,0,-1)
            s = posterior.shape
            self.flat_posterior = posterior.reshape((s[0]*s[1],s[2])) #FLATTEN posterior
        else:
            self.flat_posterior = np.array([ch for k,ch in self._chains.items()]).T

        try:
            # assert os.path.exists(chain_file)  #chain file must exist to compute the correct .stat_vals
            self._stat_vals = pickle.load(open(folder+"/.stat_vals.pkl","rb"))    #load summary statistics of the fit
            self.params     = SimpleNamespace(  names   = list(self._par_names),
                                                median  = self._stat_vals["med"],
                                                max     = self._stat_vals["max"],
                                                stdev   = self._stat_vals["stdev"] if "stdev" in self._stat_vals.keys() else np.zeros_like(self._stat_vals["med"]),
                                                bestfit = self._stat_vals["bf"],
                                                T0      = self._stat_vals["T0"],
                                                P       = self._stat_vals["P"],
                                                dur     = self._stat_vals["dur"])
            assert len(self.params.median)==len(self.params.names), "load_result(): number of parameter names and values do not match."
        except:
            self.params     = SimpleNamespace(  names   = list(self._par_names),
                                                median  = np.median(self.flat_posterior,axis=0),
                                                stdev   = np.std(self.flat_posterior,axis=0))
            
        self.evidence   = self._stat_vals["evidence"] if hasattr(self,"_stat_vals") and "evidence" in self._stat_vals.keys() else None
        self.params_dict  = {k:ufloat(v,e) for k,v,e in zip(self.params.names, self.params.median,self.params.stdev)}

        if hasattr(self,"_stat_vals"):     #summary statistics are available only if fit completed
            # evaluate model of each lc at a smooth time grid
            self._lc_smooth_time_mod = {}
            for lc in self._lcnames:
                self._lc_smooth_time_mod[lc] = SimpleNamespace()
                if self._nplanet == 1:
                    this_T0 = get_transit_time(t=input_lcs[lc]["col0"],per=self.params.P[0],t0=self.params.T0[0])
                    if this_T0 < input_lcs[lc]["col0"].min(): #occultation
                        this_T0 += 0.5*self.params.P[0]

                    if input_lcs[lc]["col0"].min() >= this_T0-0.75*self.params.dur[0]:     #data starts after/too close to ingress
                        tmin   = this_T0 - 0.75*self.params.dur[0]
                    else: tmin = input_lcs[lc]["col0"].min()
                    if input_lcs[lc]["col0"].max() <= this_T0+0.75*self.params.dur[0]:     #data finishes b4/too close to egress
                        tmax   = this_T0 + 0.75*self.params.dur[0]
                    else: tmax = input_lcs[lc]["col0"].max()
                else:
                    tmin, tmax = input_lcs[lc]["col0"].min(), input_lcs[lc]["col0"].max()

                self._lc_smooth_time_mod[lc].time    = np.linspace(tmin,tmax,max(2000, len(input_lcs[lc]["col0"])))
                self._lc_smooth_time_mod[lc].model   = self._evaluate_lc(file=lc, time=self._lc_smooth_time_mod[lc].time).planet_model


            # evaluate model of each rv at a smooth time grid
            self._rv_smooth_time_mod = {}
            for i,rv in enumerate(self._rvnames):
                self._rv_smooth_time_mod[rv] = SimpleNamespace()
                tmin, tmax = input_rvs[rv]["col0"].min(), input_rvs[rv]["col0"].max()
                t_sm  = np.linspace(tmin,tmax,max(2000, len(input_rvs[rv]["col0"])))
                gam = self.params_dict[f"rv{i+1}_gamma"].n
                self._rv_smooth_time_mod[rv].time    = t_sm
                self._rv_smooth_time_mod[rv].model   = self._evaluate_rv(file=rv, time=self._rv_smooth_time_mod[rv].time).planet_model + gam
                self._rv_smooth_time_mod[rv].gamma   = gam
        

            #LC data and functions
            self.lc = SimpleNamespace(  names     = self._lcnames,
                                        filters   = self._ind_para["filters"],
                                        evaluate  = self._evaluate_lc,
                                        outdata   = self._load_result_array(["lc"],verbose=verbose),
                                        #load each lcfile as a pandas dataframe and store all in dictionary
                                        indata    = {fname:pd.DataFrame(df) for fname,df in input_lcs.items()}, 
                                        _obj_type = "lc_obj"
                                        )
            self.lc.plot_bestfit = self._plot_bestfit_lc
            self.lc.plot_ttv     = self._ttvplot
            self.lc.plot_lcttv   = self._ttv_lcplot
            
            #RV data and functions
            self.rv = SimpleNamespace(  names     = self._rvnames,
                                        filters   = self._ind_para["filters"],
                                        evaluate  = self._evaluate_rv,
                                        outdata   = self._load_result_array(["rv"],verbose=verbose),
                                        #load each rvfile as a pandas dataframe and store all in dictionary
                                        indata    = {fname:pd.DataFrame(df) for fname,df in input_rvs.items()},
                                        _obj_type = "rv_obj"
                                        )
            self.rv.plot_bestfit = self._plot_bestfit_rv

    def __repr__(self):
        return f'Object containing posterior from emcee/dynesty sampling \
                \nParameters in chain are:\n\t {self.params.names} \
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
        if self.fit_sampler=="dynesty":
            print("chains are not available for dynesty sampler. instead see dynesty_trace_*.png plot in the output folder.")
            return
        
        assert pars is None or isinstance(pars, list) or pars == "all", f'pars must be None, "all", or list of relevant parameters.'
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
        if self.fit_sampler=="dynesty":
            print("chains are not available for dynesty sampler")
            return
        assert pars is None or isinstance(pars, list) or pars == "all", f'pars must be None, "all", or list of relevant parameters.'
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
                    labelsize=20, multiply_by=1, add_value= 0, force_plot = False, kwargs={}):
        """
            Corner plot of selected parameters.
            
            Parameters:
            ------------
            pars : list of str;
                parameter names to plot. Ideally less than 15 pars for clarity of plot
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
            show_titles : bool;
                whether to show titles for 1d histograms
            title_fmt : str;
                format string for the quantiles given in titles. Default is ".3f" for 3 decimal places.
            titlesize : int;
                size of the titles in the plot.
            labelsize : int;
                size of the labels in the plot.
            multiply_by : float or list of floats;
                factor by which to multiply the chains. Default is 1. If list, must be same length as pars.
            add_value : float or list of floats;
                value to add to the chains. Default is 0. If list, must be same length as pars.
        """
        assert pars is None or isinstance(pars, list) or pars == "all", f'pars must be None, "all", or list of relevant parameters.'
        if pars is None or pars == "all": pars = [p for p in self._par_names]

        ndim = len(pars)

        if not force_plot: assert ndim <= 15, f'number of parameters to plot should be <=15 for clarity. Use force_plot = True to continue anyways.'

        lsamp = len(self._chains[pars[0]][:,discard::thin].flatten()) if self.fit_sampler=="emcee" else len(self._chains[pars[0]])
        samples = np.empty((lsamp,ndim))

        #adjustments to make values more readable
        if isinstance(multiply_by, (int,float)): multiply_by = [multiply_by]*ndim
        elif isinstance(multiply_by, list): assert len(multiply_by) == ndim
        if isinstance(add_value, (int,float)): add_value = [add_value]*ndim
        elif isinstance(add_value, list): assert len(add_value) == ndim


        for i,p in enumerate(pars):
            assert p in self._par_names, f'{p} is not one of the parameter labels in the mcmc run.'
            if self.fit_sampler=="emcee":
                samples[:,i] = self._chains[p][:,discard::thin].reshape(-1) * multiply_by[i] + add_value[i]
            else:
                samples[:,i] = self._chains[p]* multiply_by[i] + add_value[i]
        
        fig = corner.corner(samples, bins=bins, labels=pars, show_titles=show_titles, range=range,
                    title_fmt=title_fmt,quantiles=q,title_kwargs={"fontsize": titlesize},
                    label_kwargs={"fontsize":labelsize},**kwargs)
        
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
        
        if self.fit_sampler=="emcee":
            par_samples = self._chains[par][:,discard::thin].flatten() * multiply_by + add_value
        else:
            par_samples = self._chains[par] * multiply_by + add_value

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

    def _plot_bestfit_lc(self, plot_cols=(0,1,2), detrend=False, col_labels=None, nrow_ncols=None, figsize=None, 
                        hspace=None, wspace=None, binsize=0.0104, return_fig=True):
        """
            Plot the best-fit model of the input data. 

            Parameters:
            -----------
            plot_cols : tuple of length 2 or 3;
                Tuple specifying which columns in input file to plot. 
                Default is (0,1,2) to plot column 0 against 1, and 2 as errorbar (i.e. time against flux with fluxerr). 
                Use (3,1,2) to show the correlation between column 3 and the flux. 
                Using tuple of length 2 does not plot errorbars. e.g (3,1).
            detrend : bool;
                plot the detrended data. Default is False.
            col_labels : tuple of length 2;
                label of the given columns in plot_cols. Default is ("time", "flux").
            nrow_ncols : tuple of length 2;
                Number of rows and columns to plot the input files. 
                Default is None to find the best layout.
            figsize: tuple of length 2;
                Figure size. If None, (8,5) is used for a single input file and optimally determined for more inputs.
            binsize : float;
                binsize to use for binning the data in time. Default is  None which gives 10 bin points in transit.
            hspace, wspace: float;
                height and width space between subplots. Default is None to use matplotlib defaults.
            return_fig  : bool;
                return figure object for saving to file.
        """
        if binsize==None: binsize = min(self.params.dur)/10

        obj = self.lc
        if col_labels is None:
            col_labels = ("time", "flux") if plot_cols[:2] == (0,1) else (f"column[{plot_cols[0]}]",f"column[{plot_cols[1]}]")
        
        if obj.names != []:
            model_overplot = []
            for lc in obj.names:
                df = obj.outdata[lc]
                # bl = list(df.keys())[4] #baseline name
                mop = SimpleNamespace(tot_trnd_mod=df["base_total"], time_smooth=self._lc_smooth_time_mod[lc].time,
                                    planet_mod=df["transit"], planet_mod_smooth=self._lc_smooth_time_mod[lc].model,
                                    residual=df["flux"]-df["full_mod"])
                model_overplot.append(mop)

            fig = _plot_data(obj, plot_cols=plot_cols, col_labels = col_labels, nrow_ncols=nrow_ncols, figsize=figsize, fit_order=0,
                            hspace=hspace, wspace=wspace, model_overplot = model_overplot, detrend=detrend, binsize=binsize)

            if return_fig: return fig


    def _plot_bestfit_rv(self, plot_cols=(0,1,2), detrend=False, col_labels=None, nrow_ncols=None, figsize=None, 
                            hspace=None, wspace=None, binsize=0, return_fig=True):
        """
            Plot the best-fit model of the input data. 

            Parameters:
            -----------
            plot_cols : tuple of length 2 or 3;
                Tuple specifying which columns in input file to plot. 
                Default is (0,1,2) to plot column 0 against 1, and 2 as errorbar (i.e. time against flux with fluxerr). 
                Use (3,1,2) to show the correlation between column 3 and the flux. 
                Using tuple of length 2 does not plot errorbars. e.g (3,1).
            detrend : bool;
                plot the detrended data. Default is False.
            col_labels : tuple of length 2;
                label of the given columns in plot_cols. Default is ("time", "rv").
            nrow_ncols : tuple of length 2;
                Number of rows and columns to plot the input files. 
                Default is None to find the best layout.
            figsize: tuple of length 2;
                Figure size. If None, (8,5) is used for a single input file and optimally determined for more inputs.
            hspace, wspace: float;
                height and width space between subplots. Default is None to use matplotlib defaults.
            binsize : float;
                binsize to use for binning the data in time. Default is 0.0104 (15mins).
            return_fig  : bool;
                return figure object for saving to file.
        """
        
        obj = self.rv
        if col_labels is None:
            col_labels = ("time", "rv") if plot_cols[:2] == (0,1) else (f"column[{plot_cols[0]}]",f"column[{plot_cols[1]}]")
        
        if obj.names != []:
            model_overplot = []
            for rv in obj.names:
                df = obj.outdata[rv]
                # bl = list(df.keys())[4] #baseline name 
                mop = SimpleNamespace(tot_trnd_mod=df["base_total"], time_smooth=self._rv_smooth_time_mod[rv].time, gamma=self._rv_smooth_time_mod[rv].gamma,
                                        planet_mod=df["Rvmodel"]+self._rv_smooth_time_mod[rv].gamma, planet_mod_smooth=self._rv_smooth_time_mod[rv].model,
                                        residual=df["RV"]-df["full_mod"])
                model_overplot.append(mop)

            fig = _plot_data(obj, plot_cols=plot_cols, col_labels = col_labels, nrow_ncols=nrow_ncols, figsize=figsize, fit_order=0,
                                hspace=hspace, wspace=wspace, model_overplot = model_overplot, detrend=detrend, binsize=binsize)

            if return_fig: return fig

    def get_all_params_dict(self, stat="med",uncertainty="1sigma", return_type="ufloat"):
        """
            Get all parameters(jumping,derived,fixed) from the result_**.dat and load in a dictionary with uncertainties.

            Parameters:
            -----------
            stat : str;
                summary statistic to load for model calculation. Must be one of ["med","max","bf"] for median, maximum and bestfit respectively.
                Default is "med" to load the 'result_med.dat' file.
            uncertainty : str;
                uncertainty to load from file. Must be one of ["1sigma","3sigma"] for 1sigma or 3sigma uncertainties.
                Default is "1sigma".
            return_type : str;
                return type of the values. Must be one of ["ufloat","array"] to return each parameter as ufloat(val,+/-sigma) or array of [val,lo_sigma,hi_sigma] .
                Default is "ufloat".
            Returns:
            --------
            results_dict : dict;
                dictionary of all parameters with uncertainties for jumping and derived parameters

        """
        results_dict = {}
        assert uncertainty in  ["1sigma","3sigma"], "get_all_params_dict(): uncertainty must be one of ['1sigma','3sigma']"
        assert stat in ["med","max","bf"], "get_all_params_dict(): stat must be one of ['med','max','bf']"
        assert return_type in ["ufloat","array"], "get_all_params_dict(): return_type must be one of ['ufloat','array']"
        
        print(f"Loading file: {self._folder}/results_{stat}.dat ...\n")

        with open(f"{self._folder}/results_{stat}.dat", 'r') as file:
            for line in file:
                # Ignore lines that start with '#'
                if not line.startswith('#'):
                    # Split the line into words
                    words = line.split()
                    # Use the first word as the key
                    val = float(words[1])
                    
                    if len(words) > 2: # jumping and derived parameters
                        if uncertainty == "1sigma":
                            lo, up = abs(float(words[2])), float(words[3])
                            sigma  = np.median([lo, up]) if return_type == "ufloat" else [lo,up]
                        else:
                            lo, up = abs(float(words[4])), float(words[5])
                            sigma  = np.median([lo, up]) if return_type == "ufloat" else [lo,up]
                        
                        results_dict[words[0]] = ufloat(val, sigma) if return_type=="ufloat" else np.array([val]+sigma)
                    
                    else:
                        results_dict[words[0]] = val
                        
        return results_dict
    
    def _load_result_array(self, data=["lc","rv"],verbose=True):
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
            >>> results = res._load_result_array()
            >>> list(results.keys())
            ['lc8det_lcout.dat', 'lc6bjd_lcout.dat']

            >>> df1 = results['lc8det_lcout.dat']
            >>> df1.keys()
            ['time', 'flux', 'error', 'full_mod', 'base_total', 'transit', 'det_flux',...]

            >>> #plot arrays
            >>> plt.plot(df["time"], df["flux"],"b.")
            >>> plt.plot(df["time"], df["base_total"],"r")
            >>> plt.plot(df["time"], df["transit"],"g")
            
        """
        out_files_lc = sorted([ f  for f in os.listdir(self._folder) if '_lcout.dat' in f])
        out_files_rv = sorted([ f  for f in os.listdir(self._folder) if '_rvout.dat' in f])
        all_files    = []
        input_fnames = []
        if "lc" in data: 
            all_files.extend(out_files_lc)
            input_fnames.extend(self._lcnames)
        if "rv" in data: 
            all_files.extend(out_files_rv)
            input_fnames.extend(self._rvnames)
        
        results = {}
        for f in all_files:
            df = pd.read_fwf(self._folder+"/"+f, header=0)
            df = df.rename(columns={'# time': 'time'})
            fname = f[:-10]         #remove _rvout.dat or _lcout.dat from filename
            fname_with_ext = [f for f in input_fnames if fname in f][0]    #take extension of the input_fname
            results[fname_with_ext] = df
        if verbose: print(f"{data} Output files, {all_files}, loaded into result object")
        return results

    def make_output_file(self, stat="median",out_folder=None):
        """
        make output model file ("*_??out.dat") from parameters obtained using different summary statistics on the posterior.
        if a *_??out.dat file already exists in the out_folder, it is overwritten (so be sure!!!).

        Parameters
        ----------
        stat : str, optional
            posterior summary statistic to use for model calculation, must be one of ["median","max","bestfit"], by default "median".
            "max" and "median" calculate the maximum and median of each parameter posterior respectively while "bestfit" \
            is the parameter combination that gives the maximum joint posterior probability.
        out_folder : str, optional
            folder to save the output files. Default is None to save in the current result directory.

        """
        
        from CONAN3.logprob_multi import logprob_multi
        from CONAN3.plotting import fit_plots

        assert stat in ["median","max","bestfit"],f'make_output_file: stat must be of ["median","max","bestfit"] but {stat} given'
        if   stat == "median":  stat = "med"
        elif stat == "bestfit": stat = "bf"

        if out_folder is None: out_folder =  self._folder
        if not os.path.exists(out_folder): os.makedirs(out_folder)
        _ = logprob_multi(self._stat_vals[stat],self._ind_para,make_outfile=True, out_folder=out_folder,verbose=True)

        return
        
    def _evaluate_lc(self, file, time=None,params=None, nsamp=500,return_std=False):
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
            number of posterior samples to use for computing the 1sigma quantiles of the model. Default is 500.
        return_std : bool;
            if True, return the 1sigma quantiles of the model as well as the model itself. If False, return only the model.
        Returns:
        --------
        object : SimpleNamespace;
            planet_model: LC model array evaluated at the given times, for a specific file. 
            components: lc_components of each planet in the system.
            sigma_low, sigma_high: if return_std is True, 1sigma quantiles of the model is returned.
        """

        from CONAN3.logprob_multi import logprob_multi
        
        if params is None: params = self.params.median
        mod  = logprob_multi(params,self._ind_para,t=time,get_model=True)

        if not return_std:     #return only the model
            output = SimpleNamespace(time=time if time is not None else self._ind_para["input_lcs"][file]["col0"],
                                    planet_model=mod.lc[file][0], components=mod.lc[file][1], sigma_low=None, sigma_high=None)
            return output
 
        else:                 #return model and quantiles
            lenpost = len(self.flat_posterior)
            mods    = []  #store model realization for each parameter combination

            for p in self.flat_posterior[np.random.randint(0,lenpost,int(min(nsamp,0.2*lenpost)))]:   #at most 5000 random posterior samples 
                temp = logprob_multi(p,self._ind_para,t=time,get_model=True)
                mods.append(temp.lc[file][0])

            qs = np.quantile(mods,q=[0.16,0.5,0.84],axis=0) #compute 68% percentiles

            output = SimpleNamespace(time=time if time is not None else self._ind_para["input_lcs"][file]["col0"],
                                    planet_model=mod.lc[file][0], components=mod.lc[file][1], sigma_low=qs[0], sigma_high=qs[2])
            return output
                
    def _evaluate_rv(self, file, time=None,params=None, nsamp=500,return_std=False):
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
            number of posterior samples to use for computing the 1sigma quantiles of the model. Default is 500.
        return_std : bool;
            if True, return the 1sigma quantiles of the model as well as the model itself. If False, return only the model.

        Returns:
        --------
        object : SimpleNamespace;
            planet_model: RV model array evaluated at the given times, for a specific file. 
            components: rv_components of each planet in the system.
            sigma_low, sigma_high: if return_std is True, 1sigma quantiles of the model is returned.
        """

        from CONAN3.logprob_multi import logprob_multi

        if params is None: params = self.params.median
        mod  = logprob_multi(params,self._ind_para,t=time,get_model=True)

        if not return_std:     #return only the model
            output = SimpleNamespace(time=time if time is not None else self._ind_para["input_rvs"][file]["col0"],
                                    planet_model=mod.rv[file][0], components=mod.rv[file][1], sigma_low=None, sigma_high=None)
            return output
        
        else:                 #return model and quantiles
            lenpost = len(self.flat_posterior)
            mods    = []

            for p in self.flat_posterior[np.random.randint(0,lenpost,int(min(nsamp,0.2*lenpost)))]:   #at most 5000 random posterior samples
                temp = logprob_multi(p,self._ind_para,t=time,get_model=True)
                mods.append(temp.rv[file][0])

            qs = np.quantile(mods,q=[0.16,0.5,0.84],axis=0) #compute 68% percentiles
            
            output = SimpleNamespace(time=time if time is not None else self._ind_para["input_lcs"][file]["col0"],
                                    planet_model=mod.rv[file][0], components=mod.rv[file][1], sigma_low=qs[0], sigma_high=qs[2])
            return output

    def _ttvplot(self,figsize=None):
        """  
        plot the transit times of the individual planets in the system having subtracted the best-fit linear ephemeris model.

        Parameters:
        -----------
        figsize : tuple of length 2;
            Figure size. If None, (10,6) used.

        Returns:
        --------
        fig : figure object
        """
        if self._ttvs==[]:
            return print("TTV fit not performed for this system.")
        npl = self._nplanet
        fig = plt.figure(figsize=(10,6) if figsize is None else figsize)
        ntr = {}
        for n in range(1,npl+1):
            fit_t0s     = np.array([v.n for k,v in self.params_dict.items() if f'T0_pl{n}' in k])
            fit_t0s_err = np.array([v.s for k,v in self.params_dict.items() if f'T0_pl{n}' in k])
            epoch       = np.round((fit_t0s - self.params.T0[n-1])/self.params.P[n-1])
            ntr[n]      = len(epoch)

            pp      = np.polyfit(epoch, fit_t0s,1)
            lin_t0s = np.polyval(pp,epoch)

            plt.errorbar(lin_t0s, (fit_t0s - lin_t0s)*24*60, fit_t0s_err*24*60, fmt="o",label=f"Planet {n} (P={self.params.P[n-1]:.2f}d)",capsize=3 )
        plt.axhline(0,ls="--",color="k")
        plt.title("TTV plot")
        plt.legend()
        plt.ylabel("O – C [mins]")
        plt.xlabel("Time [mBJD]")
        return fig


    def _ttv_lcplot(self, figsize=None, binsize=None,offset=None):
        """  
        plot the stacked individual transits of each planet in the system using linear ephemeris revealing the presence of TTVs.

        Parameters:
        -----------
        figsize : tuple of length 2;
            Figure size. If None, optimally determined.
        binsize : float or list of floats;
            binsize to use for binning the data in time. Default is None to use dur/10.
        offset : float or list of floats;
            vertical offset between the transits to use for plotting. Default is None to use 5*stdev transit data.

        Returns:
        --------
        fig : figure object
        """
        if self._ttvs==[]:
            return print("TTV fit not performed for this system.")
        npl         = self._nplanet
        lc          = pd.concat(self.lc.outdata,ignore_index=True)   # join all LC data into one dataframe
        tsm,modsm  = np.array([]), np.array([])                     # join all smooth model data into one array
        for nm,v in self._lc_smooth_time_mod.items():
            tsm    = np.append(tsm, v.time)
            modsm = np.append(modsm, v.model)

        max_ntra = max([len([v.n for k,v in self.params_dict.items() if f'T0_pl{n}' in k]) for n in range(1,npl+1)])
        if figsize==None: figsize = (6 if npl==1 else 10, max_ntra)


        if isinstance(offset,(float,type(None))): offset=[offset]*npl
        elif isinstance(offset,list): assert len(offset)==npl
        else: raise TypeError("offset can only be float, None or a list of these types")
            
        if isinstance(binsize,(float,type(None))): binsize=[binsize]*npl
        elif isinstance(binsize,list): assert len(binsize)==npl, "binsize must be a list of length equal to the number of planets"
        else: raise TypeError("binsize can only be float, None or a list of these types")
            
        fig,ax = plt.subplots(int(np.ceil(npl/2)),npl, figsize=figsize)
        ax = [ax] if npl==1 else ax.reshape(-1)
        for n in range(1,npl+1):
            fit_t0s = np.array([v.n for k,v in self.params_dict.items() if f'T0_pl{n}' in k]) # get t0 results for planet n
            epoch   = np.round((fit_t0s - self.params.T0[n-1])/self.params.P[n-1])            # get epoch for each t0
            pp      = np.polyfit(epoch, fit_t0s,1)                                            # fit linear ephemeris
            lin_t0s = np.polyval(pp,epoch)                                                     # get linear ephemeris
            dur     = self.params.dur[n-1]                                                     # get duration of planet n
            per     = self.params.P[n-1]                                                        # get period of planet n
            
            # lc_names = list(np.concatenate([[nm]*len(self._ttvs[i].t0s) for i,nm in enumerate(self.lc.names)]))
            # lc_names_ind = [self.lc.names.index(nm) for nm in lc_names] 
            lc_ind  = [int(k.split(f"-T0_pl{n}")[0].split("lc")[-1])-1 for k in self.params_dict.keys() if f'T0_pl{n}' in k] # get the index of the lc data for the transit times of planet n
            edges =[]
            for i in range(len(self._ttvs)):
                if i in lc_ind:
                    pl_ind = np.array(self._ttvs[i].plnum)==(n-1)  # get indices of edges for planet n in this lc
                    edges.append(tuple(np.array(self._ttvs[i].tr_edges)[pl_ind][0]))
            
            # edges   = np.concatenate([self._ttvs[i].tr_edges for i in range(len(self._ttvs)) if i in lc_ind]) # get the transit edges for the transits times of  planet n

            ax[n-1].axvline(0,ls="--",color="gray")
            for i,t0 in enumerate(lin_t0s):
                # lc     = self.lc.outdata[self.lc.names[i]]
                edg    = edges[i]
                cut    = (lc["time"]>=edg[0]) & (lc["time"]<=edg[1]) #cut the data to the transit edges
                cut_sm = (tsm>=edg[0]) & (tsm<=edg[1])                  #cut the smooth model data to the transit edges

                t,f,e  = lc["time"][cut],lc["det_flux"][cut],lc["error"][cut]       #get the data for the transit
                t_sm,mod_sm  = tsm[cut_sm], modsm[cut_sm]
                ph     = phase_fold(t,per,t0,-0.25)*per    #phase fold but convert back to days
                ph_sm  = phase_fold(t_sm,per,t0,-0.25)*per

                ind    = abs(ph)<1.2*dur  #cut the data to 1.2 transit duration
                t,f,e  = ph[ind],f[ind],e[ind]
                ind_sm = abs(ph_sm)<1.2*dur 
                ts, ms = ph_sm[ind_sm], mod_sm[ind_sm]

                if offset[n-1]==None: offset[n-1] = 5* np.std(np.diff(f))/np.sqrt(2)
                if binsize[n-1]==None: binsize[n-1] = dur/10
                t_,f_,e_ = bin_data_with_gaps(t,f,e,binsize=binsize[n-1])

                # to hrs
                x  = t*24 #(t-t0)*24
                xs = ts*24 #(ts-t0)*24
                x_ = t_*24 #(t_-t0)*24
                srt_xs = np.argsort(xs)

                ax[n-1].plot(x,f+i*offset[n-1],".",ms=3)
                ax[n-1].errorbar(x_,f_+i*offset[n-1],e_,fmt="k.",ms=8,mfc="w",label=f"{binsize[n-1]*24*60:.0f}min bins") 

                ax[n-1].plot(xs[srt_xs],ms[srt_xs]+i*offset[n-1],"k")
                ax[n-1].set_title(f"Planet {n} (P={self.params.P[n-1]:.2f})")
                ax[n-1].set_yticks([])
                ax[n-1].set_xlabel("time from linear ephemeris [hrs]")
                if i==0: ax[n-1].legend()
                
        plt.subplots_adjust(wspace=0.02)
        return fig



class compare_results:
    def __init__(self,result_list):
        """
        Compare the results of multiple CONAN3 fits.

        Parameters:
        -----------
        result_list : list of CONAN3 objects or result folder names;
            list of CONAN3 objects to compare.

        Examples:
        ---------
        #load the result objects before comparing
        >>> import CONAN3
        >>> res1 = CONAN3.load_result("result1")
        >>> res2 = CONAN3.load_result("result2")
        >>> res3 = CONAN3.load_result("result3")
        >>> comp = CONAN3.compare([res1,res2,res3])

        #or load the result folders directly for comparing
        >>> comp = CONAN3.compare(["result1","result2","result3"])

        """
        for r in result_list:
            assert isinstance(r, (load_result,str)), "compare: all elements in result_list must be CONAN3 objects or strings of folder names."
        
        if isinstance(result_list[0],str):
            self.results_list = [load_result(r) for r in result_list]
        else:
            self.results_list = result_list

    def plot_corner(self, pars=None, bins=20, thin=1, discard=0, q=[0.5], range=None,show_titles=True, title_fmt =".3f", titlesize=14,save_fig=False):
        """  
        make corner plot of loaded results object

        Parameters:
        -----------
        pars : list of str;
            parameter names to plot. Ideally less than 15 pars for clarity of plot
        bins : int;
            number of bins in 1d histogram
        thin : int;
            factor by which to thin the chains in order to reduce correlation.
        discard : int;
            to discard first couple of steps within the chains.
        q : list of floats;
            quantiles to show on the 1d histograms. defaults is 0.5 to show just the median
        range : dict or list with the same length as pars;
            A dict/list where each key/element is either a length 2 tuple containing
            lower and upper bounds or a float in range (0., 1.) giving the fraction of samples to include in bounds, e.g.,
            [(0.,10.), (1.,5), 0.999, etc.] or {"P":(1.,5.), "rho_star":(1.,5), "Impact_para":0.999,...}
        
        """
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        all_pars = set()
        for r in self.results_list:
            all_pars = all_pars.union(r.params.names)
            
        for p in all_pars:
                for r in self.results_list:
                    nsamp = len(r._chains[r.params.names[0]])
                    if p not in r.params.names:
                        r._chains[p] = np.random.normal(0,1e-10,nsamp)
        
        if range!=None:
            if isinstance(range,list):
                assert len(range) == len(pars), "range must be a list (of tuples/floats) with length equal to the number of parameters in pars."
                prange = range
            elif isinstance(range,dict):
                for p in range.keys(): assert p in pars, f"{p} in range is not a parameter in in pars"
                prange = [None]*len(pars)
                for i,p in enumerate(pars):
                    if p in range.keys():
                        prange[i] = range[p]
                    else:
                        xs = np.concatenate([r._chains[p] for r in self.results_list])
                        prange[i] = (xs.min(),xs.max())
            else:
                raise TypeError("range must be a dictionary or list  containing the range (tuple/float) for each parameter in par")
        else: 
            prange = None

        for i,r in enumerate(self.results_list):
            if i ==0: f = r.plot_corner(pars=pars, bins=bins, thin=thin, discard=discard, q=q, range=prange,show_titles=False, title_fmt =title_fmt, titlesize=titlesize, kwargs={"color":color_cycle[i]})
            else: r.plot_corner(pars=pars, bins=bins, thin=thin, discard=discard, q=q, range=prange,show_titles=False, title_fmt =title_fmt, titlesize=titlesize,kwargs={"fig":f,"color":color_cycle[i]})
        logz= np.array([r.evidence for r in self.results_list])
        dlogz = (logz-logz[0]) if all(logz) else None
        
        f.suptitle(f"Comparing all results to {self.results_list[0]._folder} gives $\Delta$logZ={list(dlogz)}", fontsize=20)
        f.subplots_adjust(top=0.95)

        return f

    def plot_param_sigma_diff(self, pars):
        p1 = self.results_list[0].get_all_params_dict()
        p2 = self.results_list[1].get_all_params_dict()
        fig = plt.figure(figsize=(12, 5))
        for i,p in enumerate(pars):
            assert p in p1 and p in p2, f"{p} not in both results"
            diff = p1[p]-p2[p]
            assert isinstance(diff, type(ufloat(0,1)-0)), f"parameter {p} is not a ufloat"
            plt.plot(i, diff.n/diff.s, "o")
        
        plt.xticks(np.arange(len(pars)), pars, rotation=45)
        plt.grid()
        plt.axhline(0,ls=":",color="k")
        plt.ylabel("sigma_difference: (p1-p2)/err(p1-p2)")
        return fig

    def plot_lc(self, plot_cols=(0, 1, 2),detrend=False,col_labels=None,nrow_ncols=None,figsize=None,hspace=None,wspace=None,
                binsize=0.0104):
        """
        side-by-side plot of the best-fit model of the LCs of the loaded results objects.

        Parameters:
        -----------
        plot_cols : tuple of length 2 or 3;
            Tuple specifying which columns in input file to plot. 
            Default is (0,1,2) to plot column 0 against 1, and 2 as errorbar (i.e. time against flux with fluxerr). 
            Use (3,1,2) to show the correlation between column 3 and the flux. 
            Using tuple of length 2 does not plot errorbars. e.g (3,1).
        detrend : bool;
            plot the detrended data. Default is False.
        col_labels : tuple of length 2;
            label of the given columns in plot_cols. Default is ("time", "flux").
        nrow_ncols : tuple of length 2;
            Number of rows and columns to plot the input files of each result object. 
            Default is None which creates layout of (nfiles x 1) for each result object
        figsize: tuple of length 2;
            Figure size to plot the LCs of each result object. If None, (7,3.5*nfiles) is used.
        hspace, wspace: float;
            height and width space between subplots. Default is None to use matplotlib defaults.
        binsize : float;
            binsize to use for binning the data in time. Default is  None which gives 10 bin points in transit.
        
        Examples:
        ---------
        >>> comp = CONAN3.compare_results(["results1_folder","results2_folder"])
        >>> fig  = comp.plot_lc(plot_cols=(0,1,2),detrend=False)
        >>> fig.savefig("compare_LC.png", dpi=200)
        """
        from PIL import Image
        nfiles = len(self.results_list[0].lc.names)
        if nrow_ncols is None: nrow_ncols = (nfiles,1)
        if figsize is None: figsize = (7,3.5*nfiles)

        imgs = []
        ev   = []
        for i,r in enumerate(self.results_list):
            fig_ = r._plot_bestfit_lc(plot_cols=plot_cols,detrend=detrend,col_labels=col_labels,nrow_ncols=nrow_ncols,
                                            figsize=figsize,hspace=hspace,wspace=wspace,binsize=binsize,return_fig=True)
            # Save fig1 and fig2 as temporary PNG files
            fig_.savefig(f".temp{i}.png",bbox_inches='tight',dpi=200)
            ev.append(f"{r.evidence:.2f}" if r.evidence else "")
            # Load the temporary PNG files
            imgs.append(Image.open(f".temp{i}.png") ) 
            os.remove(f".temp{i}.png")

        # Create a matplotlib figure
        fig, axes = plt.subplots(1, len(self.results_list), figsize=(figsize[0]*len(self.results_list),3.5*nfiles ))
        # Display the images
        for i,img in enumerate(imgs):
            axes[i].imshow(img)
            # axes[i].axis('off')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(f"{self.results_list[i]._folder}: ev={ev[i]}")
        # Adjust layout
        plt.tight_layout()
        # Clean up temporary file
        

        return fig

    def plot_rv():
        pass
