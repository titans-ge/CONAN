import numpy as np
import matplotlib.pyplot as plt
import corner
import pickle
from types import SimpleNamespace
import os
import matplotlib

__all__ = ["load_lightcurves", "load_rvs", "setup_fit", "__default_backend__"]

#helper functions
__default_backend__ = matplotlib.get_backend()
matplotlib.use(__default_backend__)

def _plot_data(obj, plot_cols, col_labels, nrow_ncols=None, figsize=None):
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

        if self._show_guide: print("\nNext: use method `lc_baseline` to define baseline model for each lc")

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
                list containing 'y' or 'n' to specify if a gp will be fitted to a light curve.

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
        self._gp_lcs = np.array(self._names)[np.array(self._useGPphot) == "y"]     #lcs with gp == "y"

        #create lc_baseline print out variable
        self._print_lc_baseline = f"""#--------------------------------------------- \n# Input lightcurves filters baseline function-------------- \n{"name":15s}\t{"fil":3s}\t {"lamda":5s}\t {"time":4s}\t {"roll":3s}\t x\t y\t {"conta":5s}\t sky\t sin\t group\t id\t GP"""

        #define print out format
        txtfmt = "\n{0:15s}\t{1:3s}\t{2:5.1f}\t {3:4d}\t {4:3d}\t {5}\t {6}\t {7:5d}\t {8:3d}\t {9:3d}\t {10:5d}\t {11:2d}\t {12:2s}"        
        for i in range(n_lc):
            t = txtfmt.format(self._names[i], self._filters[i], self._lamdas[i], *self._bases[i], self._groups[i], self._useGPphot[i])
            
            self._print_lc_baseline += t

        if verbose: print(self._print_lc_baseline)

        if np.all(np.array(self._useGPphot) == "n"):        #if gp is "n" for all input lightcurves, run add_GP with None
            self.add_GP(None, verbose=verbose)
            if self._show_guide: print("\nNo GPs required.\nNext: use method `setup_transit_rv` to configure transit an rv model parameters.")
        else: 
            if self._show_guide: print("\nNext: use method `add_GP` to include GPs for the specified lcs. Get names of lcs with GPs using `._gp_lcs` attribute of the lightcurve object.")

        #initialize other methods to empty incase they are not called
        self.setup_transit_rv(verbose=False)
        self.transit_depth_variation(verbose=False)
        self.setup_occultation(verbose=False)
        self.contamination_factors(verbose=False)
        self.limb_darkening(verbose=False)
        self.stellar_parameters(verbose=False)
   
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
                s_up.append( np.max(s[0]+10, s[0]+5*s[1]) )    #set bounds at +/- 10 from prior mean or 5stdev (the larger value)
                s_lo.append( np.min(s[0]-10, s[0]-5*s[1]) )

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
                m_up.append( np.max(m[0]+10,m[0]+5*m[1]) )    #set uniform bounds at _+/- 10 from prior mean
                m_lo.append( np.min(m[0]-10, m[0]-5*m[1]) )
                
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
        _ = [DA.pop(item) for item in ["log_metric", "log_scale","m","s"]]
        

        #create gp print variable
        self._print_gp = f"""# -------- photometry GP input properties: komplex kernel -> several lines -------------- \n{'name':13s} {'para':5s} kernel WN {'scale':7s} s_step {'s_pri':5s} s_pri_wid {'s_up':5s} {'s_lo':5s} {'metric':7s} m_step {'m_pri':6s} m_pri_wid {'m_up':4s} {'m_lo':4s}"""


        if lc_list is None: 
            self._GP_dict = {"lc_list":[]}
            if len(self._gp_lcs)>0: print(f"\nWarning: GP was expected for the following lcs {self._gp_lcs} \nMoving on ...")
            if verbose: print(self._print_gp)
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

        #define gp print out format
        txtfmt = "\n{0:13s} {1:5s} {2:6s} {3:2s} {4:5.1e} {5:6.4f} {6:5.1f} {7:9.2e} {8:4.1f} {9:4.1f} {10:5.1e} {11:6.4f} {12:5.2f} {13:9.2e} {14:4.1f} {15:4.1f}"        
        for i in range(n_list):
            t = txtfmt.format(DA["lc_list"][i], DA["pars"][i],DA["kernels"][i],
                                DA["WN"][i], DA["scale"][i], DA["s_step"][i], 
                                DA["s_pri"][i], DA["s_pri_wid"][i], DA["s_up"][i],DA["s_lo"][i],DA["metric"][i],DA["m_step"][i], DA["m_pri"][i], DA["m_pri_wid"][i],DA["m_up"][i],DA["m_lo"][i])
            self._print_gp += t

        if verbose: print(self._print_gp)

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
        

        #create transit_rv print out variable
        self._print_transit_rv_pars = f"""#=========== jump parameters (Jump0value step lower_limit upper_limit priors) ====================== \n{'name':12s}\tfit\tstart_val\tstepsize\tlow_lim\tup_lim\tprior\tvalue\tsig_lo\tsig_hi"""

        #define print out format
        txtfmt = "\n{0:12s}\t{1:3s}\t{2:8.5f}\t{3:.7f}\t{4:4.2f}\t{5:4.2f}\t{6}\t{7:.5f}\t{8:4.1e}\t{9:4.1e} "        
        for i,p in enumerate(self._parnames):
            t = txtfmt.format(  p, DA[p].to_fit, DA[p].start_value,
                                DA[p].step_size, DA[p].bounds_lo, 
                                DA[p].bounds_hi, DA[p].prior, DA[p].prior_mean,
                                DA[p].prior_width_lo, DA[p].prior_width_hi)
            self._print_transit_rv_pars += t

        
        if verbose: print(self._print_transit_rv_pars)



        if self._show_guide: print("\nNext: use method transit_depth_variation` to include variation of RpRs for the different filters or \n`setup_occultation` to fit the occultation depth or \n`limb_darkening` for fit or fix LDCs or `contamination_factors` to add contamination.")

    def transit_depth_variation(self, transit_depth_per_group=[(0.1,0.0001)], divwhite="n",
                        ddFs="n", step=0.001, bounds=(-1,1), prior="n", prior_width=(0,0),
                       verbose=True):
        """
            Include transit depth variation between the lightcurves.
            
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
            

        #create depth_variation print out variable
        self._print_depth_variation = f"""#=========== ddF setup ============================================================================== \nFit_ddFs  step\t low_lim   up_lim   prior   sig_lo   sig_hi   div_white"""

        #define print out format
        txtfmt = "\n{0:8s}  {1:.3f}\t {2:.4f}   {3:.4f}   {4:5s}   {5:.5f}   {6:.5f}   {7:3s}"        
        t = txtfmt.format(self._ddfs.ddfYN,*self._ddfs.drprs_op[1:4],
                            self._ddfs.prior, self._ddfs.prior_width_lo,
                            self._ddfs.prior_width_hi,self._ddfs.divwhite)
        self._print_depth_variation += t

        self._print_depth_variation += "\ngroup_ID   RpRs_0   err\t\tdwfile"
        
        txtfmt = "\n{0:6d}\t   {1:.4f}   {2:.2e}   {3}"
        for i in range(ngroup):
            t2 = txtfmt.format( grnames[i] , self._ddfs.depth_per_group[i],
                                self._ddfs.depth_err_per_group[i],f"dw_00{grnames[i]}.dat")
            self._print_depth_variation += t2

        
        if verbose: print(self._print_depth_variation)

                
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
                prior.append("y")
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

        #create occultations print out variable
        self._print_occulations = f"""#=========== occultation setup ============================================================================= \n{'filters':7s}\tfit start_val\tstepsize  {'low_lim':8s}  {'up_lim':8s}  prior  {'value':8s}  {'sig_lo':8s}\t{'sig_hi':8s}"""

        #define print out format
        # txtfmt = "\n{0:7s}\t{1:3s} {2:.8f}\t{3:.6f}  {4:7.6f}  {5:6.6f}  {6:5s}  {7:4.3e}  {8:4.2e}\t{9:4.2e} "       
        txtfmt = "\n{0:7s}\t{1:3s} {2:4.3e}\t{3:3.2e}  {4:3.2e}  {5:3.2e}  {6:5s}  {7:3.2e}  {8:3.2e}\t{9:3.2e} "       
        for i in range(nfilt):
            t = txtfmt.format(  DA["filters_occ"][i], DA["filt_to_fit"][i],
                                DA["start_value"][i], DA["step_size"][i],
                                DA["bounds_lo"][i],DA["bounds_hi"][i], 
                                DA["prior"][i], DA["prior_mean"][i],
                                DA["prior_width_lo"][i], DA["prior_width_hi"][i])
            self._print_occulations += t
        
        if verbose: print(self._print_occulations)


    def limb_darkening(self, priors="n",
                             c1=0, step1=0.001,
                             c2=0, step2=0.001,
                             verbose=True):
        """
            Setup quadratic limb darkening LD parameters (c1, c2) for transit light curves. 
            Different LD parameters are required if observations of different filters are used.

            Parameters:
            ---------
            priors : "y" or "n":
                specify if the ldc should be fitted/

            c1,c2 : float/tuple or list of float/tuple for each filter;
                 limb darkening coefficient.
                 *if tuple, must be of length 3 defined as (lo_lim, val, uplim) 

            step1,step2 : float or list of floats;
                stepsize for fitting        
                    
        """
        #not used yet
        c3 = step3 = 0
        c4 = step4 = 0

        DA = _reversed_dict(locals().copy())
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")

        nfilt = len(self._filnames)
        
        temp_DA = {}
        for par in DA.keys():
            if isinstance(DA[par], (int,float,str)): DA[par] = [DA[par]]*nfilt
            elif (isinstance(DA[par], tuple) and len(DA[par])==3): DA[par] = [DA[par]]*nfilt 
            elif isinstance(DA[par], list): assert len(DA[par]) == nfilt,f"length of list {DA[par]} must be equal to number of unique filters (={nfilt})."
            else: _raise(TypeError, f"{par} must be int/float or tuple of len 3 but {DA[par]} is given.")

            if par in ["c1","c2","c3","c4"]:
                for i,d in enumerate(DA[par]):
                    if isinstance(d, (int,float)):  
                        DA[par][i] = (0,d,0)
                        DA[f"step{par[-1]}"][i] = 0
                    elif isinstance(d, tuple):
                        DA[par][i] = d

                temp_DA[f"bound_lo{par[-1]}"] = [b[0] for b in DA[par]]
                temp_DA[f"bound_hi{par[-1]}"] = [b[2] for b in DA[par]]
                DA[par] = [b[1] for b in DA[par]]

        for k in temp_DA.keys():
            DA[k] = temp_DA[k]

        for i in range(nfilt):
            DA["priors"][i] = "y" if np.any( [DA["step1"][i], DA["step2"][i],DA["step3"][i], DA["step4"][i] ]) else "n"
        


        self._ld_dict = DA


        #create limb_darkening print out variable
        self._print_limb_darkening = f"""#=========== Limb darkending setup ===================================================================\n{'filters':7s} priors\t{'c_1':4s} {'step1':5s}  low_lim1  up_lim1\t{'c_2':4s} {'step2':5s} low_lim2 up_lim2"""

        #define print out format
        txtfmt = "\n{0:7s} {1:6s}\t{2:4.3f} {3:5.3f} {4:7.4f} {5:7.4f}\t{6:4.3f} {7:5.3f} {8:7.4f} {9:7.4f}"       
        for i in range(nfilt):
            t = txtfmt.format(self._filnames[i],DA["priors"][i], 
                            DA["c1"][i], DA["step1"][i], DA["bound_lo1"][i], 
                            DA["bound_hi1"][i], DA["c2"][i], DA["step2"][i], 
                            DA["bound_lo2"][i], DA["bound_hi2"][i])
            self._print_limb_darkening += t
        
        if verbose: print(self._print_limb_darkening)


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

        #create contamination print out variable
        self._print_contamination = f"""#=========== contamination setup === give contamination as flux ratio ================================\n{'filters':7s}\tcontam\terr"""

        #define print out format
        txtfmt = "\n{0:7s}\t{1:.4f}\t{2:.4f}"       
        for i in range(nfilt):
            t = txtfmt.format(self._filnames[i],DA["cont_ratio"][i], 
                                DA["err"][i])
            self._print_contamination += t
        
        if verbose: print(self._print_contamination)

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

        #create stellar_pars print out variable
        self._print_stellar_pars = f"""#=========== Stellar input properties ================================================================\n{'# parameter':13s}  value  sig_lo  sig_hi \n{'Radius_[Rsun]':13s}  {DA['R_st'][0]:.3f}  {DA['R_st'][1]:.3f}  {DA['R_st'][2]:.3f} \n{'Mass_[Msun]':13s}  {DA['M_st'][0]:.3f}  {DA['M_st'][1]:.3f}  {DA['M_st'][2]:.3f}\nStellar_para_input_method:_R+rho_(Rrho),_M+rho_(Mrho),_M+R_(MR): {DA['par_input']}"""

        if verbose: print(self._print_stellar_pars)           
    
    def __repr__(self):
        data_type = str(self.__class__).split("load_")[1].split("'>")[0]
        return f'Object containing {len(self._names)} {data_type}\nFiles:{self._names}\nFilepath: {self._fpath}'

    
    def print(self):
        """
            Print out all input configuration for the light curve object. It is printed out in the format of the legacy config file.
        
        """
        print(self._print_lc_baseline)
        print(self._print_gp)
        print(self._print_transit_rv_pars)
        print(self._print_depth_variation)
        print(self._print_occulations)
        print(self._print_limb_darkening)
        print(self._print_contamination)
        print(self._print_stellar_pars)


    def plot(self, plot_cols=(0,1,2), col_labels=None, nrow_ncols=None, figsize=None, return_fig=False):
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
            
            figsize: tuple of length 2;
                Figure size. If None, (8,5) is used for a single input file and optimally determined for more inputs.

            return_fig  : bool;
                return figure object for saving to file.
        """
        if not (isinstance(plot_cols, tuple) and len(plot_cols) in [2,3]): 
            raise TypeError(f"plot_cols must be tuple of length 2 or 3, but is {type(plot_cols)} and length of {len(plot_cols)}.")
        
        assert col_labels is None or ((isinstance(col_labels, tuple) and len(col_labels)==2)), \
            f"col_labels must be tuple of length 2, but is {type(col_labels)} and length of {len(col_labels)}."
        
        
        if col_labels is None:
            col_labels = ("time", "flux") if plot_cols[:2] == (0,1) else (f"column[{plot_cols[0]}]",f"column[{plot_cols[1]}]")
        
        if self._names != []:
            fig = _plot_data(self, plot_cols=plot_cols, col_labels = col_labels, nrow_ncols=nrow_ncols, figsize=figsize)
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
        self._fpath = os.getcwd() if data_filepath is None else data_filepath
        self._names   = [] if file_list is None else file_list  
        if self._names == []:
            self.rv_baseline(verbose=False)
        else: 
            for rv in self._names: assert os.path.exists(self._fpath+rv), f"file {rv} does not exist in the path {self._fpath}."
            print("Next: use method `rv_baseline` to define baseline model for for the each rv")

        self._nRV = len(self._names)

    def rv_baseline(self, dt=None, dbis=None, dfwhm=None, dcont=None,
                    gammas_kms=0.0, gam_steps=0.01, sinPs=None,
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

        #create gp print variable
        self._print_rv_baseline = f"""# ------------------------------------------------------------\n# Input RV curves, baseline function, gamma  \n{'name':13s}   time  bis  fwhm  contrast  gamma_kms  stepsize  prior  value  sig_lo  sig_hi"""

        # assert self._names != [], "No rv files given"
        # assert 
        if self._names == []: 
            if verbose: print(self._print_rv_baseline)  
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
        _ = [dict_args.pop(item) for item in ["verbose","gammas_kms","g","txtfmt"]]


        for par in dict_args.keys():
            assert dict_args[par] is None or isinstance(dict_args[par], (int,float)) or (isinstance(dict_args[par], (list,np.ndarray)) and len(dict_args[par]) == self._nRV), f"parameter {par} must be a list of length {self._nRV} or int (if same degree is to be used for all RVs) or None (if not used in decorrelation)."
            
            if dict_args[par] is None: dict_args[par] = [0]*self._nRV
            elif isinstance(dict_args[par], (int,float,str)): dict_args[par] = [dict_args[par]]*self._nRV
            

        self._RVbases = [ [dict_args["dt"][i], dict_args["dbis"][i], dict_args["dfwhm"][i], dict_args["dcont"][i]] for i in range(self._nRV) ]

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
        
        
        #define gp print out format
        txtfmt = "\n{0:13s}   {1:4d}  {2:3d}  {3:4d}  {4:8d}  {5:9.4f}  {6:8.4f}  {7:5s}  {8:6.4f}  {9:6.4f}  {10:6.4f}"         
        for i in range(self._nRV):
            t = txtfmt.format(self._names[i],*self._RVbases[i],self._gammas[i], 
                            self._gamsteps[i], self._prior[i], self._gampri[i],
                            self._siglo[i], self._sighi[i])
            self._print_rv_baseline += t

        if verbose: print(self._print_rv_baseline)
    
    def __repr__(self):
        data_type = str(self.__class__).split("load_")[1].split("'>")[0]
        return f'Object containing {len(self._names)} {data_type}\nFiles:{self._names}\nFilepath: {self._fpath}'
        
    def plot(self, plot_cols=(0,1,2), col_labels=None, nrow_ncols=None, figsize=None, return_fig=False):
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
        
        
        if col_labels is None:
            col_labels = ("time", "rv") if plot_cols[:2] == (0,1) else (f"column[{plot_cols[0]}]",f"column[{plot_cols[1]}]")
        
        if self._names != []:
            fig = _plot_data(self, plot_cols=plot_cols, col_labels = col_labels, nrow_ncols=nrow_ncols, figsize=figsize)
            if return_fig: return fig
        else: print("No data to plot")
    
    def print(self):
        print(self._print_rv_baseline)
    
class mcmc_setup:
    """
        class to setup fitting
    """
    def __init__(self, n_chains=64, n_steps=2000, n_burn=500, n_cpus=2, sampler=None,
                         GR_test="y", make_plots="n", leastsq="y", savefile="output_ex1.npy",
                         savemodel="n", adapt_base_stepsize="y", remove_param_for_CNM="n",
                         leastsq_for_basepar="n", lssq_use_Lev_Marq="n", apply_CFs="y",apply_jitter="y",
                         verbose=True):
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
                sampler algorithm to use in traversing the parameter space. Options are ["demc","snooker"]
        """
        
        DA = _reversed_dict(locals().copy())
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")

        self._mcmc_dict = DA

        #create stellar_pars print out variable
        self._print_mcmc_pars = f"""#=========== MCMC setup ==============================================================================\n{'Total_no_steps':23s}  {DA['n_steps']*DA['n_chains']} \n{'Number_chains':23s}  {DA['n_chains']} \n{'Number_of_processes':23s}  {DA['n_cpus']} \n{'Burnin_length':23s}  {DA['n_burn']} \n{'Walk_(snooker/demc/mrw)':23s}  {DA['sampler']} \n{'GR_test_(y/n)':23s}  {DA['GR_test']} \n{'Make_plots_(y/n)':23s}  {DA['make_plots']} \n{'leastsq_(y/n)':23s}  {DA['leastsq']} \n{'Savefile':23s}  {DA['savefile']} \n{'Savemodel':23s}  {DA['savemodel']} \n{'Adapt_base_stepsize':23s}  {DA['adapt_base_stepsize']} \n{'Remove_param_for_CNM':23s}  {DA['remove_param_for_CNM']} \n{'leastsq_for_basepar':23s}  {DA['leastsq_for_basepar']} \n{'lssq_use_Lev-Marq':23s}  {DA['lssq_use_Lev_Marq']} \n{'apply_CFs':23s}  {DA['apply_CFs']} \n{'apply_jitter':23s}  {DA['apply_jitter']}"""

        if verbose: print(self._print_mcmc_pars)              
            
    def __repr__(self):
        return f"mcmc setup: steps:{self._mcmc_dict['n_steps']} \nchains: {self._mcmc_dict['n_chains']}"

    def print(self):
        print(self._print_mcmc_pars)

                   


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

    print(lc._print_lc_baseline,file=f)
    print(lc._print_gp,file=f)
    print(rv._print_rv_baseline, file=f)
    print(lc._print_transit_rv_pars,file=f)
    print(lc._print_depth_variation,file=f)
    print(lc._print_occulations,file=f)
    print(lc._print_limb_darkening,file=f)
    print(lc._print_contamination,file=f)
    print(lc._print_stellar_pars,file=f)
    print(mcmc._print_mcmc_pars, file=f)

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
        strbase=adump[1:5]         # string array of the baseline function exponents 
        base = [int(i) for i in strbase]
        RVbases.append(base)
        gammas.append(float(adump[5]))
        gamsteps.append(float(adump[6]))
        gampri.append(float(adump[8]))
        gampriloa = (0. if (adump[7] == 'n' or adump[6] == 0.) else float(adump[9]))
        gamprilo.append(gampriloa)
        gamprihia = (0. if (adump[7] == 'n' or adump[6] == 0.) else float(adump[10]))
        gamprihi.append(gamprihia)
        # sinPs.append(adump[12])
        dump=_file.readline()

    gamm = [((g,e) if e!=0 else g) for g,e in zip(gampri,gamprilo)]


    rv_data = load_rvs(RVnames,fpath)
    rv_data.rv_baseline(*np.array(RVbases).T, gammas_kms=gamm,
                        gam_steps=gamsteps,sinPs=None,verbose=verbose)  
    
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


    lc_data.transit_depth_variation(depth_per_group,div_white,
                                    ddf,step,bounds,prior,pr_width,
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

    priors,c1,c2,step1,step2 =[],[],[],[],[]
    while dump[0] != '#':
        adump=dump.split()
        priors.append(adump[1])
        step1.append(float(adump[3]))
        step2.append(float(adump[7]))
        c1.append( (float(adump[4]),float(adump[2]),float(adump[5])))
        c2.append( (float(adump[8]),float(adump[6]),float(adump[9])))

        dump=_file.readline()

    lc_data.limb_darkening(priors,c1,step1,c2,step2,verbose )

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
    def __init__(self,chain_file = "chains_dict.pkl"):
        self._chains = pickle.load(open(chain_file,"rb"))
        self._par_names = self._chains.keys()
        
    def __repr__(self):
        return f'Object containing chains from mcmc. \
                \nParameters in chain are:\n\t {self._par_names} \
                \n\nuse `plot_chains`, `plot_corner` or `plot_posterior` methods on selected parameters to visualize results.'
        
    def plot_chains(self, pars=None, figsize = None, thin=1, discard=0, alpha=0.05,
                    color=None, label_size=12, force_plot = False):
        """
            Plot chains of selected parameters.
              
            Parameters:
            ----------
            pars: list of str;
                parameter names to plot. Plot less than 20 parameters at a time for clarity.
        """
        assert pars is None or isinstance(pars, list) or pars == "all", \
             f'pars must be None, "all", or list of relevant parameters.'
        if pars is None or pars == "all": pars = [p for p in self._par_names]
        for p in pars:
            assert p in self._par_names, f'{p} is not one of the parameter labels in the mcmc run.'
        
        ndim = len(pars)
        if not force_plot: assert ndim < 21, f'number of parameter chain to plot should be <=20 for clarity. \
            Use force_plot = True to continue anyways.'

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
        # plt.show()
        return fig
        
    def plot_corner(self, pars=None, bins=20, thin=1, discard=0,
                    q=[0.16,0.5,0.84], show_titles=True, title_fmt =".3f",
                    multiply_by=1, add_value= 0, force_plot = False ):
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
        
        
        fig = corner.corner(samples, bins=bins, labels=pars, show_titles=show_titles,
                    title_fmt=title_fmt,quantiles=q,title_kwargs={"fontsize": 14},
                    label_kwargs={"fontsize":20})
        
        # plt.show()
        return fig


    def plot_posterior(self, par, thin=1, discard=0, bins=20, density=True, range=None,
                        q = [0.0015,0.16,0.5,0.85,0.9985], multiply_by=1, add_value=0):
        """
        Plot the posterior distribution of a single input parameter, par.
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

        return fig