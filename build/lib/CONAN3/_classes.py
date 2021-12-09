import numpy as np
import matplotlib.pyplot as plt
import corner
import pickle
from types import SimpleNamespace
import os
import matplotlib

__all__ = ["load_lightcurves", "load_rvs", "fit_setup"]


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
    def __init__(self, file_list, data_filepath=None, filters=None, lamdas=None):
        self._fpath = os.getcwd() if data_filepath is None else data_filepath
        self._names   = [file_list] if isinstance(file_list, str) else file_list
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
        print(f"Filters: {self._filters}")
        print(f"Order of unique filters: {list(self._filnames)}")

        print("\nNext: use method `lc_baseline` to define baseline model for each lc")

    def lc_baseline(self, dt=None, dphi=None, dx=None, dy=None, dconta=None, 
                 dsky=None, dsin=None, grp=None, grp_id=None, gp="n", verbose=True):
        """
            Define lightcurve baseline model parameters to fit.
            Each baseline model parameter should be a list of numbers specifying the polynomial order for each light curve.
            e.g. Given 3 input light curves, and one wishes to fit a 2nd order time trend to only the first and third lightcurves,
            then dt = [2, 0, 2].

            Parameters:
            -----------
            grp_id : list (same length as file_list);
                group the different input lightcurves by id so that different transit depths can be fitted for each group.

            gp : list (same length as file_list); 
                list containing 'y' or 'n' to specify if a gp will be fitted to a light curve.

        """
        dict_args = locals().copy()     #get a dictionary of the input arguments for easy manipulation
        _ = dict_args.pop("self")            #remove self from dictionary
        _ = dict_args.pop("verbose")
        # print(dict_args)    
        n_lc = len(self._names)

        for par in dict_args.keys():
            assert dict_args[par] is None or isinstance(dict_args[par], (int,str)) or \
                (isinstance(dict_args[par], list) and len(dict_args[par]) == n_lc), \
                    f"parameter {par} must be a list of length {n_lc} or \
                        int (if same degree is to be used for all LCs) or \
                            None (if not used in decorrelation)."
            
            if isinstance(dict_args[par], (int,str)): dict_args[par] = [dict_args[par]]*n_lc
            elif dict_args[par] is None: dict_args[par] = [0]*n_lc

        dict_args["grp_id"] = list(np.arange(1,n_lc+1))

        self._bases = [ [dict_args["dt"][i], dict_args["dphi"][i], dict_args["dx"][i], dict_args["dy"][i],
                        dict_args["dconta"][i], dict_args["dsky"][i], dict_args["dsin"][i], 
                        dict_args["grp"][i]] for i in range(n_lc) ]

        self._groups = dict_args["grp_id"]
        self._grbases = dict_args["grp"]
        self._useGPphot= dict_args["gp"]

        if verbose:
            print("#---------------------------------------------------")
            print("# Input lightcurves filters baseline function")
            print(f"{'name':15s}\t{'fil':3s}\t {'lamda':5s}\t {'time':4s}\t {'roll':3s}\t x\t y\t {'conta':5s}\t sky\t sin\t group\t id\t GP")
            txtfmt = "{0:15s}\t{1:3s}\t{2:5.1f}\t {3:4d}\t {4:3d}\t {5}\t {6}\t {7:5d}\t {8:3d}\t {9:3d}\t {10:5d}\t {11:2d}\t {12:2s}"        
            for i in range(n_lc):
                out_txt = txtfmt.format(self._names[i], self._filters[i], self._lamdas[i], *self._bases[i], self._groups[i], self._useGPphot[i])
                print(out_txt)

        if np.all(np.array(self._useGPphot) == "n"):        #if gp is "n" for all input lightcurves, run add_GP with None
            self.add_GP(None, verbose=verbose)
        else: print("\nNext: use method `add_GP` to include GPs for the desired lcs")

        #initialize other methods to empty incase they are not called
        self.depth_variation(verbose=False)
        self.setup_occultation(verbose=False)
        self.contamination_factors(verbose=False)
        self.limb_darkening(verbose=False)
   
    def add_GP(self, lc_list=None, pars="time", kernels="mat32", WN="y", 
               log_scale=[(-25,-15.2,-5)], s_step=0.001,
               log_metric=[(-10,6.9,15)],  m_step=0.001,
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
                log of the scale (variance) and metric (lengthscale) of the GP kernel applied for each lc in lc_list.
                * if tuple is of len 2, set normal prior with index[0] as prior mean and index[1] as prior width. \
                    hard bounds are set at 10X the prior width
                * if tuple is of len 3, set uniform prior with between index[0] and index[2], index[1] is the initial value.

               
            s_step, m_step : list of floats;
                step sizes of the scale and metric parameter of the GP kernel.
  
                
        """
        #unpack scale and metric to the expected CONAN parameters
        scale, s_pri, s_pri_wid, s_lo, s_up = [], [], [], [], []
        for s in log_scale:
            if isinstance(s,tuple) and len(s)==2:
                s_pri.append(s[0])
                scale.append( np.exp(s[0]) )
                s_pri_wid.append(s[1])
                s_up.append(s[0]+5*s[1])    #set bounds as 10X the width of the normal
                s_lo.append(s[0]-5*s[1])

            elif isinstance(s,tuple) and len(s)==3:
                s_pri_wid.append(0)          #using uniform prior so set width = 0
                s_lo.append(s[0])
                scale.append(np.exp(s[1]))
                s_pri.append(s[1])
                s_up.append(s[2])
            
            else: _raise(TypeError, f"tuple of len 2 or 3 was expected but got the value {s} in log_scale.")

        metric, m_pri, m_pri_wid, m_lo, m_up  = [], [], [], [], []
        for m in log_metric:
            if isinstance(m,tuple) and len(m)==2:
                m_pri.append(m[0])
                metric.append( np.exp(m[0]) )
                m_pri_wid.append(m[1])
                m_up.append(m[0]+5*m[1])    #set bounds as 10X the width of the normal
                m_lo.append(m[0]-5*m[1])
                
            elif isinstance(m,tuple) and len(m)==3:
                m_pri_wid.append(0)       
                m_lo.append(m[0])
                metric.append( np.exp(m[1]) )
                m_pri.append(m[1])
                m_up.append(m[2])

            else: _raise(TypeError, f"tuple of len 2 or 3 was expected but got the value {m} in log_metric.")


        dict_args = locals().copy()
        _ = dict_args.pop("self")            #remove self from dictionary
        _ = dict_args.pop("verbose")
        _ = [dict_args.pop(item) for item in ["log_metric", "log_scale","m","s"]]
        
        if verbose:
            print("# -------- photometry GP input properties: komplex kernel -> several lines --------------")
            print(f"{'name':13s} {'para':5s} kernel WN {'scale':7s} s_step {'s_pri':5s} s_pri_wid {'s_up':4s} {'s_lo':5s}   {'metric':7s} m_step {'m_pri':6s} m_pri_wid {'m_up':4s} {'m_lo':4s}")
            txtfmt = "{0:13s} {1:5s} {2:6s} {3:2s} {4:5.2e} {5:6.4f} {6:5.1f} {7:9.2e} {8:4.1f} {9:4.1f}   {10:5.1e} {11:6.4f} {12:5.2f} {13:9.2e} {14:4.1f} {15:4.1f}"

        gp_lcs = np.array(self._names)[np.array(self._useGPphot) == "y"]
        # assert 
        if lc_list is None: 
            self._GP_dict = {"lc_list":[]}
            print(f"\nWarning: GP was expected for the following lcs {gp_lcs} \nMoving on ...")
            return 
        elif isinstance(lc_list, str): lc_list = [lc_list]
        

        for lc in lc_list: 
            assert lc in self._names,f"{lc} is not one of the loaded lightcurve files"
            assert lc in gp_lcs, f"while defining baseline in the `lc_baseline` method, gp = 'y' was not specified for {lc}."

           
        n_list = len(lc_list)
        
        #transform        
        for key in dict_args.keys():
            if (isinstance(dict_args[key],list) and len(dict_args[key])==1): 
                dict_args[key]= dict_args[key]*n_list
            if isinstance(dict_args[key], list):
                assert len(dict_args[key]) == n_list, f"{key} must have same length as lc_list"
            if isinstance(dict_args[key],(float,int,str)):  
                dict_args[key] = [dict_args[key]]*n_list
                
        
        for p in dict_args["pars"]: 
            assert p in ["time", "xshift", "yshift", "air", "fwhm", "sky", "eti"], \
                f"pars {p} cannot be the GP independent variable"             
        
        
        assert len(dict_args["pars"]) == len(dict_args["kernels"]) == len(dict_args["WN"]) == n_list, f"pars and kernels must have same length as lc_list (={len(lc_list)})"
                                            
        self._GP_dict = dict_args

        if verbose:
            ff = self._GP_dict
        
            for i in range(n_list):
                out_txt = txtfmt.format(ff["lc_list"][i], ff["pars"][i], ff["kernels"][i], ff["WN"][i],
                                        ff["scale"][i], ff["s_step"][i], ff["s_pri"][i],  ff["s_pri_wid"][i],
                                        ff["s_up"][i],ff["s_lo"][i],ff["metric"][i],ff["m_step"][i],
                                        ff["m_pri"][i], ff["m_pri_wid"][i],ff["m_up"][i],ff["m_lo"][i])
                print(out_txt)


        print("\nNext: use method `depth_variation` to include variation of RpRs for the different filters or \
            \n`setup_occultation` to fit the occultation depth")

    
    def depth_variation(self, ddFs="n", step=0.001, bounds_lo=-1, bounds_hi=1,
                       prior="n", prior_width_lo=0, prior_width_hi=0, divwhite="n",
                       transit_depth_per_group=[(0.1,0.0001)], verbose=True):
        """
            Include transit depth variation between the lightcurves.
            
            Parameters:
            ----------
            
            divwhite : str ("y" or "n"):
                flag to divide each light-curve by the white lightcurve
                
            depth_per_group : list of size2-tuples;
                the reference depth (and error) to compare the transit depth of each lightcurve group with.
                Usually from fit to the white (or total) available light-curves. The length should be equal to the length of unique groups defined in lc_baseline.
                if float, then same value is used for all groups.
            
        """
        
        self._ddfs= SimpleNamespace()

        depth_per_group     = [d[0] for d in transit_depth_per_group]
        depth_err_per_group = [d[1] for d in transit_depth_per_group]

        width_lo = (0 if (prior == 'n' or ddFs == 'n' or bounds_lo == 0.) else prior_width_lo)
        width_hi = (0 if (prior == 'n' or ddFs == 'n' or bounds_lo == 0.) else prior_width_hi)

        self._ddfs.drprs_op=[0., step, bounds_lo, bounds_hi, 0., width_lo, width_hi]  # the dRpRs options
        
        grnames    = np.array(list(sorted(set(self._groups))))
        ngroup     = len(grnames)
        
        if isinstance(depth_per_group, float): depth_per_group = [depth_per_group]*ngroup
        if isinstance(depth_err_per_group, float): depth_err_per_group = [depth_err_per_group]*ngroup

        
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
            
        if verbose:
            print("Fit_ddFs  step\t low_lim   up_lim   prior   sig_lo   sig_hi   div_white")
            txtfmt = "{0:8s}  {1:.3f}\t {2:.4f}   {3:.4f}   {4:5s}   {5:.5f}   {6:.5f}   {7:3s}"
            out_txt = txtfmt.format(self._ddfs.ddfYN,*self._ddfs.drprs_op[1:4],self._ddfs.prior,
                                    self._ddfs.prior_width_lo,self._ddfs.prior_width_lo,self._ddfs.divwhite)
            print(out_txt)
            print("group_ID   RpRs_0   err\t\tdwfile")
            txtfmt = "{0:6d}\t   {1:.4f}   {2:.2e}   {3}"
            for i in range(ngroup):
                out_txt = txtfmt.format( grnames[i] , self._ddfs.depth_per_group[i], self._ddfs.depth_err_per_group[i],
                                      f"dw_00{grnames[i]}.dat" )
                print(out_txt)
                
    def setup_occultation(self, filters_occ=None, start_value= 0.0005, step_size=0.00001,
                             prior ="n", prior_mean=None, prior_width_lo=None, prior_width_hi=None,
                             bounds_lo=None, bounds_hi=None, verbose=True):
        """
            setup fitting for occultation depth
            
            Parameters:
            -----------
            
            filters_occ : list;
                List of unique filters to fit. 
                If "all", occultation depth is fit for all filters given in `lc.load_lightcurves`. 
                use `lc_data._filnames` to obtain the list of unique filters.
                If None, will not fit occultation.
            
            start_value : list, float, None;
                define start value for occultation depth in each filter. If float is given, the same is used for all filters.
                If None, defaults to 0.0005 (500ppm) for all filters.
            
            step_size : list, float;
                step size for each filter. If float, the same step size is used for all filters.
                
            prior : list containing "y" or "n", or str;
                Flag specify whether to use gaussian priors on the depth. If a single str is given, all filters are set to the given string.
                
            prior_mean : list of floats, float;
                mean value of the gaussian prior for the depth of each filter. 
                    
            prior_width_lo, prior_width_hi : list, None;
                corresponding to the lower and upper width of the priors. 
            
            bounds_lo, bounds_hi : list, None;
                corresponding to the lower and upper bound of the depth in each filter.
                if None, the lower bound is 0 and the upper bound is 1.
                
            verbose: bool;
                print output configuration or not.
            
        """
        if isinstance(filters_occ, str):
            if filters_occ is "all": filters_occ = list(self._filnames)
            else: filters_occ= [filters_occ]
        if filters_occ is None: filters_occ = []

        if start_value is None: start_value = 0.0005
        if prior_mean is None: prior_mean = 0
        if prior_width_lo is None: prior_width_lo=0
        if prior_width_hi is None: prior_width_hi=0
        if bounds_lo is None: bounds_lo=0
        if bounds_hi is None: bounds_hi=1
        

        DA = _reversed_dict(locals().copy())
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")
        
        if verbose: 
            if filters_occ != [] : print(f"fitting occultation depth for filters: {filters_occ}\n")
            else: print("Not fitting occultation\n")

        nfilt  = len(self._filnames)    #length of unique filters 
        nocc   = len(filters_occ)        #length of given filters to fit
        
                      

        if filters_occ != []:
            for f in filters_occ: assert f in self._filnames, \
                f"{f} is not in list of defined filters"
            
            for par in DA.keys():
                assert isinstance(DA[par], (int,float,str)) or (isinstance(DA[par], list) and len(DA[par]) == nocc), \
                    f"length of input {par} must be equal to the length of filters_occ (={nocc}) or float or None."
                if isinstance(DA[par], (int,float,str)): DA[par] = [DA[par]]*nocc

        for i in range(nocc):
            if DA["prior"][i] is "y":
                assert DA["prior_width_lo"][i] != 0 and DA["prior_width_hi"][i] != 0, \
                    f"setting prior[{i}]='y' requires to define prior_width(lo and hi)."
            
            if (DA["prior"][i]=='n' or DA["step_size"][i] == 0.): 
                DA["prior_mean"][i] = DA["prior_width_lo"][i] =  DA["prior_width_hi"][i] = 0  

        
        DA2 = {}    # expand dictionary to also include specifications for non-fitted filters
        DA2["filt_to_fit"] = [("y" if f in filters_occ else "n") for f in self._filnames]

        indx = [ list(self._filnames).index(f) for f in filters_occ]    #index of given filters_occ in unique filter names
        for par in DA.keys():
            if par is "prior": DA2[par] = ["n"]*nfilt
            elif par is "filters_occ": DA2[par] = list(self._filnames)
            else: DA2[par] = [0]*nfilt

            for i,j in zip(indx, range(nocc)):                
                DA2[par][i] = DA[par][j]

        self._occ_dict =  DA = DA2

        if verbose:
            print(f"{'filters':7s}\tfit\tstart_val\tstepsize\tlow_lim\tup_lim\tprior\tvalue\tsig_lo\tsig_hi")
            txtfmt = "{0:7s}\t{1:3s}\t{2:.8f}\t{3:.7f}\t{4:.3f}\t{5:.3f}\t{6:5s}\t{7:.3f}\t{8:.3f}\t{9:.3f} "
            for i in range(nfilt):
                out_txt = txtfmt.format(DA["filters_occ"][i], DA["filt_to_fit"][i], DA["start_value"][i],DA["step_size"][i],
                                        DA["bounds_lo"][i],DA["bounds_hi"][i], DA["prior"][i], DA["prior_mean"][i],
                                        DA["prior_width_lo"][i], DA["prior_width_hi"][i])
                print(out_txt)


    def limb_darkening(self, priors="n",
                             c1=0, step1=0.000, bound_lo1=0, bound_hi1=0,
                             c2=0, step2=0.000, bound_lo2=0, bound_hi2=0,
                             verbose=True):
        """
            Setup quadratic limb darkening LD parameters (c1, c2) for transit light curves. 
            Different LD parameters are required if observations of different filters are used.
        """
        #not used yet
        c3 = step3 = bound_lo3 = bound_hi3 = 0
        c4 = step4 = bound_lo4 = bound_hi4 = 0

        DA = _reversed_dict(locals().copy())
        _ = DA.pop("self")            #remove self from dictionary
        _ = DA.pop("verbose")

        nfilt = len(self._filnames)

        for par in DA.keys():
            assert isinstance(DA[par], (int,float,str)) or (isinstance(DA[par], list) and len(DA[par]) == nfilt), f"length of input {par} must be equal to the length of unique filters (={nfilt}) or float."
            if isinstance(DA[par], (int,float,str)): DA[par] = [DA[par]]*nfilt


        self._ld_dict = DA
        if verbose:
            print(f"{'filters':7s} priors\t{'c_1':4s} {'step1':5s}  low_lim1  up_lim1\t{'c_2':4s} {'step2':5s} low_lim2 up_lim2")
            txtfmt = "{0:7s} {1:6s}\t{2:4.3f} {3:5.3f} {4:7.4f} {5:7.4f}\t{6:4.3f} {7:5.3f} {8:7.4f} {9:7.4f}"
            
            for i in range(nfilt):
                out_txt = txtfmt.format(self._filnames[i],DA["priors"][i], 
                                        DA["c1"][i], DA["step1"][i], DA["bound_lo1"][i], DA["bound_hi1"][i],
                                        DA["c2"][i], DA["step2"][i], DA["bound_lo2"][i], DA["bound_hi2"][i])
                print(out_txt)

         

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

        if verbose:
            print(f"{'filters':7s}\tcontam\terr")
            txtfmt = "{0:7s}\t{1:.4f}\t{2:.4f}"
        
            for i in range(nfilt):
                out_txt = txtfmt.format(self._filnames[i],DA["cont_ratio"][i], DA["err"][i])
                print(out_txt)

    def __repr__(self):
        data_type = str(self.__class__).split("load_")[1].split("'>")[0]
        return f'Object containing {len(self._names)} {data_type}\nFiles:{self._names}\nFilepath: {self._fpath}'
        
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
            print("Next: use method ```rv_baseline``` to define baseline model for for the each rv")

        
    def rv_baseline(self, dt=None, dbis=None, dfwhm=None, dcont=None, dgamma=None,
                   gam_steps= 0.01, prior="n", gam_pri=None, sig_lo=None, sig_hi=None, sinPs=None,
                   verbose=True):
        
        """
            Define rv baseline model parameters to fit.
            Each baseline model parameter should be a list of numbers specifying the polynomial order for each rv data.
            e.g. Given 3 input rvs, and one wishes to fit a 2nd order time trend to only the first and third lightcurves,
            then dt = [2, 0, 2].
        """

        if verbose:
            print("# --------------------------------------------------------------")
            print("# Input RV curves, baseline function, gamma")
            print(f"{'name':13s}   time  bis  fwhm  contrast  gamma_kms  stepsize  prior  value  sig_lo  sig_hi")
            txtfmt = "{0:13s}   {1:4d}  {2:3d}  {3:4d}  {4:8d}  {5:9.4f}  {6:8.4f}  {7:5s}  {8:6.4f}  {9:6.4f}  {10:6.4f}" 

        # assert self._names != [], "No rv files given"
        # assert 
        if self._names == []: 
            return 

        dict_args = locals().copy()     #get a dictionary of the input arguments for easy manipulation
        _ = dict_args.pop("self")            #remove self from dictionary
        _ = dict_args.pop("verbose")

        nRV = len(self._names)

        for par in dict_args.keys():
            assert dict_args[par] is None or isinstance(dict_args[par], (int,float,str)) or (isinstance(dict_args[par], list) and len(dict_args[par]) == nRV), f"parameter {par} must be a list of length {nRV} or int (if same degree is to be used for all RVs) or None (if not used in decorrelation)."
            
            if dict_args[par] is None: dict_args[par] = [0]*nRV
            elif isinstance(dict_args[par], (int,float,str)): dict_args[par] = [dict_args[par]]*nRV
            

        self._RVbases = [ [dict_args["dt"][i], dict_args["dbis"][i], dict_args["dfwhm"][i], dict_args["dcont"][i]] for i in range(nRV) ]

        self._gammas = dict_args["dgamma"]
        self._gamsteps = dict_args["gam_steps"]
        self._gampri = dict_args["gam_pri"]
        
        self._prior = dict_args["prior"]
        self._siglo = dict_args["sig_lo"]
        self._sighi = dict_args["sig_hi"]
        
        gampriloa=[]
        gamprihia=[]
        for i in range(nRV):
            gampriloa.append( 0. if (self._prior[i] == 'n' or self._gamsteps[i] == 0.) else self._siglo[i])
            gamprihia.append( 0. if (self._prior[i] == 'n' or self._gamsteps[i] == 0.) else self._sighi[i])
        
        self._gamprilo = gampriloa                
        self._gamprihi = gamprihia                
        self._sinPs = dict_args["sinPs"]
        
        
        if verbose:
            for i in range(nRV):
                out_txt = txtfmt.format(self._names[i], *self._RVbases[i], self._gammas[i], self._gamsteps[i],
                                       self._prior[i], self._gampri[i], self._siglo[i], self._sighi[i])
                print(out_txt)
    

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
    
    
class setup_fit:
    """
        class to setup fitting
    """
    def __init__(self, RpRs=0.1, Impact_para=0, Duration=0.1245, T_0=0, Period=3, 
                 Eccentricity=0, Omega=90, K=0, verbose=True):
        """
            Define parameters of the signals to be fitted. configure the parameters using the `configure_parameters` method.
            By default, the parameters are fixed the given values. To fit a parameter use the `to_fit` method to change it from 'n' to 'y'.
            parameters can be defined in following ways:
            
            * fixed value as float or int, e.g Period = 3.4
            * free parameter with gaussian prior given as ufloat, e.g. T_0 = ufloat(5678, 0.1)
            * free parameters with uniform prior interval and initial value given as tuple of length 3, e.g. RpRs = (0,0.1,0.2) with 0.1 being the initial value.
        """
        
        DA = _reversed_dict(locals().copy() )         #dict of arguments (DA)
        _ = DA.pop("self")                            #remove self from dictionary
        _ = DA.pop("verbose")

        self._parnames  = [n for n in DA.keys()]
        self._npars = 8

        for par in DA.keys():
            if par in ["RpRs","Impact_para","Duration", "Eccentricity"]: up_lim = 1
            elif par == "Omega": up_lim = 360
            else: up_lim = 10000

            if isinstance(DA[par], tuple):
                if len(DA[par]) == 2:
                    DA[par] = _param_obj(["y", DA[par][0], 0.1*DA[par][1], "p", DA[par][0],
                                  DA[par][1], DA[par][1], 0, up_lim])

                elif len(DA[par]) == 3: 
                    DA[par] = _param_obj(["y", DA[par][1], 0.00001, "n", DA[par][1],
                                       0, 0, DA[par][0], DA[par][2]])
                
                else: _raise(ValueError, f"length of tuple is {len(DA[par])} but it must be 2 or 3 such that it follows (lo_limit, start_value, up_limit).")

            elif isinstance(DA[par], (int, float)):
                DA[par] = _param_obj(["n", DA[par], 0.00, "n", DA[par],
                                       0,  0, 0, up_lim])

            else: _raise(TypeError, f"{par} must be one of [tuple(of len 2 or 3), int, float] but is {type(DA[par])}")

        self._config_par = DA      #add to object
        self._items = DA["RpRs"].__dict__.keys()
        
        
        if verbose:
            print(f"{'name':12s}\tfit\tstart_val\tstepsize\tlow_lim\tup_lim\tprior\tvalue\tsig_lo\tsig_hi")
            txtfmt = "{0:12s}\t{1:3s}\t{2:8.5f}\t{3:.7f}\t{4:4.2f}\t{5:4.2f}\t{6}\t{7:.5f}\t{8:4.1e}\t{9:4.1e} "
            for i,p in enumerate(self._parnames):
                out_txt = txtfmt.format(p, DA[p].to_fit, DA[p].start_value, DA[p].step_size,
                                        DA[p].bounds_lo, DA[p].bounds_hi, DA[p].prior, DA[p].prior_mean,
                                        DA[p].prior_width_lo, DA[p].prior_width_hi)
                print(out_txt)
        
    
        self.stellar_parameters(verbose=False)
        self.mcmc_setup(verbose=False)
    
        
    def __repr__(self):
        fit_flag = np.array([self._config_par[p].to_fit for p in self._parnames]) == "y"
        return f"fit configuration for model parameters: {list(np.array(self._parnames)[fit_flag])}"

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
                assert 2 == len(DA[par]) <=3, f"length of {par} tuple must be 2 or 3 "
                if len(DA[par])== 2: DA[par]= (DA[par][0], DA[par][1], DA[par][1])
        
        assert DA["par_input"] in ["Rrho","Mrho", "MR"], f"par_input must be one of 'Rrho','Mrho' or 'MR'. "
            
        self._stellar_dict = DA

        if verbose:
            print("#=========== Stellar input properties ================================================================")
            print(f"{'# parameter':13s}  value  sig_lo  sig_hi")
            print(f"{'Radius_[Rsun]':13s}  {DA['R_st'][0]:.3f}  {DA['R_st'][1]:.3f}  {DA['R_st'][2]:.3f}")
            print(f"{'Mass_[Msun]':13s}  {DA['M_st'][0]:.3f}  {DA['M_st'][1]:.3f}  {DA['M_st'][2]:.3f}")
            print(f"Stellar_para_input_method:_R+rho_(Rrho),_M+rho_(Mrho),_M+R_(MR): {DA['par_input']}")

        

    def mcmc_setup(self, n_chains=64, n_steps=2000, n_burn=500, n_cpus=2, sampler=None,
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

        if verbose:
            print("#=========== MCMC setup ==============================================================================")
            print(f"{'Total_no_steps':23s}  {DA['n_steps']*DA['n_chains']}")
            print(f"{'Number_chains':23s}  {DA['n_chains']}")
            print(f"{'Number_of_processes':23s}  {DA['n_cpus']}")    
            print(f"{'Burnin_length':23s}  {DA['n_burn']}")          
            print(f"{'Walk_(snooker/demc/mrw)':23s}  {DA['sampler']}")
            print(f"{'GR_test_(y/n)':23s}  {DA['GR_test']}")          
            print(f"{'Make_plots_(y/n)':23s}  {DA['make_plots']}")       
            print(f"{'leastsq_(y/n)':23s}  {DA['leastsq']}")          
            print(f"{'Savefile':23s}  {DA['savefile']}")  
            print(f"{'Savemodel':23s}  {DA['savemodel']}")              
            print(f"{'Adapt_base_stepsize':23s}  {DA['adapt_base_stepsize']}")    
            print(f"{'Remove_param_for_CNM':23s}  {DA['remove_param_for_CNM']}")  
            print(f"{'leastsq_for_basepar':23s}  {DA['leastsq_for_basepar']}")    
            print(f"{'lssq_use_Lev-Marq':23s}  {DA['lssq_use_Lev_Marq']}")      
            print(f"{'apply_CFs':23s}  {DA['apply_CFs']}")              
            print(f"{'apply_jitter':23s}  {DA['apply_jitter']}")           


def create_configfile(lc, rv, fitter, filename="input_config.dat"):
    """
        create configuration file that of lc, rv, amd fitter setup.
        
        Parameters:
        -----------
        lc : object;
            Instance of CONAN.load_lightcurve() object and its attributes.

        rv : object, None;
            Instance of CONAN.load_rvs() object and its attributes.
        
        fitter : object;
            Instance of CONAN.setup_fit() object and its attributes.
    """
    f = open(filename,"w")
    f.write("#=========== MCMC input file =======================\n")
    f.write("Path_of_input_lightcurves:\n")
    f.write(lc._fpath)
    #LC
    f.write("\n#---------------------------------------------------\n")
    f.write("# Input lightcurves filters baseline function\n")

    f.write(f"{'name':13s}\t{'fil':3s}\t {'lamda':5s}\t {'time':4s}\t {'roll':3s}\t x\t y\t {'conta':5s}\t sky\t sin\t group\t id\t GP\n")
    txtfmt = "{0:13s}\t{1:3s}\t{2:5.1f}\t {3:4d}\t {4:3d}\t {5}\t {6}\t {7:5d}\t {8:3d}\t {9:3d}\t {10:5d}\t {11:2d}\t {12:2s}"        
    for i in range(len(lc._names)):
        out_txt = txtfmt.format(lc._names[i], lc._filters[i], lc._lamdas[i], *lc._bases[i], lc._groups[i], lc._useGPphot[i])
        f.write(out_txt+"\n")
    
    #GP
    f.write("# -------- photometry GP input properties: komplex kernel -> several lines --------------\n")
    f.write(f"{'name':13s} {'para':5s} kernel WN {'scale':7s} s_step {'s_pri':5s} s_pri_wid {'s_up':4s} {'s_lo':5s}   {'metric':7s} m_step {'m_pri':6s} m_pri_wid {'m_up':4s} {'m_lo':4s}\n")
    txtfmt = "{0:13s} {1:5s} {2:6s} {3:2s} {4:5.1e} {5:6.4f} {6:5.1f} {7:9.2e} {8:4.1f} {9:4.1f}   {10:5.1e} {11:6.4f} {12:5.1f} {13:9.2e} {14:4.1f} {15:4.1f}"
    
    gp = lc._GP_dict    
    for i in range(len(gp["lc_list"])):
        out_txt = txtfmt.format(gp["lc_list"][i], gp["pars"][i], gp["kernels"][i], gp["WN"][i],
                                gp["scale"][i], gp["s_step"][i], gp["s_pri"][i],  gp["s_pri_wid"][i],
                                gp["s_up"][i],gp["s_lo"][i],gp["metric"][i],gp["m_step"][i],
                                gp["m_pri"][i], gp["m_pri_wid"][i],gp["m_up"][i],gp["m_lo"][i])
        f.write(out_txt+"\n")     

    #RV
    f.write("# --------------------------------------------------------------\n")
    f.write("# Input RV curves, baseline function, gamma\n")
    f.write(f"{'name':13s}   time  bis  fwhm  contrast  gamma_kms  stepsize  prior  value  sig_lo  sig_hi\n")
    txtfmt = "{0:13s}   {1:4d}  {2:3d}  {3:4d}  {4:8d}  {5:9.4f}  {6:8.4f}  {7:5s}  {8:6.4f}  {9:6.4f}  {10:6.4f}"  

    for i in range(len(rv._names)):
        out_txt = txtfmt.format(rv._names[i], *rv._RVbases[i], rv._gammas[i], rv._gamsteps[i],
                                    rv._prior[i], rv._gampri[i], rv._siglo[i], rv._sighi[i])
        f.write(out_txt+"\n")

    #jump parameters
    f.write("#=========== jump parameters (Jump0value step lower_limit upper_limit priors) ======================\n")
    f.write(f"{'name':12s}\tfit\tstart_val\tstepsize\tlow_lim\tup_lim\tprior\tvalue\tsig_lo\tsig_hi\n")
    txtfmt = "{0:12s}\t{1:3s}\t{2:8.5f}\t{3:.7f}\t{4:4.2f}\t{5:4.2f}\t{6}\t{7:.5f}\t{8:4.1e}\t{9:4.1e} "


    jps = fitter._config_par

    for i,p in enumerate(fitter._parnames):
        out_txt = txtfmt.format(p, jps[p].to_fit, jps[p].start_value, jps[p].step_size,
                                jps[p].bounds_lo, jps[p].bounds_hi, jps[p].prior, jps[p].prior_mean,
                                jps[p].prior_width_lo, jps[p].prior_width_hi)
        f.write(out_txt+"\n")



    #ddf
    DF= lc._ddfs
    f.write("#=========== ddF setup ==============================================================================\n")
    f.write("Fit_ddFs  step\t low_lim   up_lim   prior   sig_lo   sig_hi   div_white\n")
    txtfmt = "{0:8s}  {1:.3f}\t {2:.4f}   {3:.4f}   {4:5s}   {5:.5f}   {6:.5f}   {7:3s}"
    out_txt = txtfmt.format(DF.ddfYN,*DF.drprs_op[1:4],DF.prior,
                            DF.prior_width_lo,DF.prior_width_lo,DF.divwhite)
    f.write(out_txt+"\n")
    f.write("group_ID   RpRs_0   err\t\tdwfile\n")
    txtfmt = "{0:6d}\t   {1:.4f}   {2:.2e}   {3}"

    grnames    = np.array(list(sorted(set(lc._groups))))
    for i in range(len(grnames)):
        out_txt = txtfmt.format( grnames[i] , DF.depth_per_group[i], DF.depth_err_per_group[i],
                                      f"dw_00{grnames[i]}.dat" )
        f.write(out_txt+"\n")

    #occ
    f.write("#=========== occultation setup =============================================================================\n")
    f.write(f"{'filters':7s}\tfit\tstart_val\tstepsize\tlow_lim\tup_lim\tprior\tvalue\tsig_lo\tsig_hi\n")
    txtfmt = "{0:7s}\t{1:3s}\t{2:.8f}\t{3:.7f}\t{4:.3f}\t{5:.3f}\t{6:5s}\t{7:.3f}\t{8:.3f}\t{9:.3f} "

    occ = lc._occ_dict
    for i in range(len(occ["filters_occ"])):
        out_txt = txtfmt.format(occ["filters_occ"][i], occ["filt_to_fit"][i], occ["start_value"][i],occ["step_size"][i],
                                occ["bounds_lo"][i],occ["bounds_hi"][i], occ["prior"][i], occ["prior_mean"][i],
                                occ["prior_width_lo"][i], occ["prior_width_hi"][i])
        f.write(out_txt+"\n")

    #limb darkening
    f.write("#=========== Limb darkending setup ===================================================================\n")
    f.write(f"{'filters':7s} priors\t{'c_1':4s} {'step1':5s}  low_lim1  up_lim1\t{'c_2':4s} {'step2':5s} low_lim2 up_lim2\
        \t{'c_3':4s} {'step3':5s} low_lim3 up_lim3\t{'c_4':4s} {'step4':5s} low_lim4 up_lim4\n")
    txtfmt = "{0:7s} {1:6s}\t{2:4.3f} {3:5.3f} {4:7.4f} {5:7.4f}\t{6:4.3f} {7:5.3f} {8:7.4f} {9:7.4f}\
        \t{6:4.3f} {7:5.3f} {8:7.4f} {9:7.4f}\t{6:4.3f} {7:5.3f} {8:7.4f} {9:7.4f}"

    ld = lc._ld_dict
    for i in range(len(lc._filnames)):
        out_txt = txtfmt.format(lc._filnames[i],ld["priors"][i], 
                                ld["c1"][i], ld["step1"][i], ld["bound_lo1"][i], ld["bound_hi1"][i],
                                ld["c2"][i], ld["step2"][i], ld["bound_lo2"][i], ld["bound_hi2"][i],
                                0.,0.,0.,0.,0.,0.,0.,0.)
        f.write(out_txt+"\n")

       
    #contamination
    f.write("#=========== contamination setup === give contamination as flux ratio ================================\n")
    f.write(f"{'filters':7s}\tcontam\terr\n")
    txtfmt = "{0:7s}\t{1:.4f}\t{2:.4f}"

    cnt = lc._contfact_dict
    for i in range(len(lc._filnames)):
        out_txt = txtfmt.format(lc._filnames[i],cnt["cont_ratio"][i], cnt["err"][i])
        f.write(out_txt+"\n")



    #stellar input
    f.write("#=========== Stellar input properties ================================================================\n")
    f.write(f"{'# parameter':13s}  value  sig_lo  sig_hi\n")

    stlr = fitter._stellar_dict
    f.write(f"{'Radius_[Rsun]':13s}  {stlr['R_st'][0]:.3f}  {stlr['R_st'][1]:.3f}  {stlr['R_st'][2]:.3f}\n")
    f.write(f"{'Mass_[Msun]':13s}  {stlr['M_st'][0]:.3f}  {stlr['M_st'][1]:.3f}  {stlr['M_st'][2]:.3f}\n")
    f.write(f"Stellar_para_input_method:_R+rho_(Rrho),_M+rho_(Mrho),_M+R_(MR): {stlr['par_input']}\n")



    #mcmc
    f.write("#=========== MCMC setup ==============================================================================\n")

    mc = fitter._mcmc_dict    
    f.write(f"{'Total_no_steps':23s}  {mc['n_steps']*mc['n_chains']}\n")
    f.write(f"{'Number_chains':23s}  {mc['n_chains']}\n")
    f.write(f"{'Number_of_processes':23s}  {mc['n_cpus']}\n")    
    f.write(f"{'Burnin_length':23s}  {mc['n_burn']}\n")          
    f.write(f"{'Walk_(snooker/demc/mrw)':23s}  {mc['sampler']}\n")
    f.write(f"{'GR_test_(y/n)':23s}  {mc['GR_test']}\n")          
    f.write(f"{'Make_plots_(y/n)':23s}  {mc['make_plots']}\n")       
    f.write(f"{'leastsq_(y/n)':23s}  {mc['leastsq']}\n")          
    f.write(f"{'Savefile':23s}  {mc['savefile']}\n")  
    f.write(f"{'Savemodel':23s}  {mc['savemodel']}\n")              
    f.write(f"{'Adapt_base_stepsize':23s}  {mc['adapt_base_stepsize']}\n")    
    f.write(f"{'Remove_param_for_CNM':23s}  {mc['remove_param_for_CNM']}\n")  
    f.write(f"{'leastsq_for_basepar':23s}  {mc['leastsq_for_basepar']}\n")    
    f.write(f"{'lssq_use_Lev-Marq':23s}  {mc['lssq_use_Lev_Marq']}\n")      
    f.write(f"{'apply_CFs':23s}  {mc['apply_CFs']}\n")              
    f.write(f"{'apply_jitter':23s}  {mc['apply_jitter']}\n")  
     
    f.close()


class load_chains:
    def __init__(self,chain_file = "chains_dict.pkl"):
        self._chains = pickle.load(open(chain_file,"rb"))
        self._par_names = self._chains.keys()
        
    def __repr__(self):
        return f'Object containing chains from mcmc. \
                \nParameters in chain are:\n\t {self._par_names} \
                \n\nuse `plot_chains` method on selected parameters to plot the chains.'
        
    def plot_chains(self, pars=None, figsize = None, thin=1, discard=0, alpha=0.05,
                    color=None, label_size=12, force_plot = False):
        """
            Plot chains of selected parameters.
              
            Parameters:
            ----------
            pars: list of str;
                parameter names to plot. Plot less than 20 parameters at a time for clarity.
        """
        assert pars is None or isinstance(pars, list) or pars is "all", \
             f'pars must be None, "all", or list of relevant parameters.'
        if pars is None or pars is "all": pars = [p for p in self._par_names]
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
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.set_ylabel(pars[i], fontsize=label_size)
        plt.subplots_adjust(hspace=0.0)
        axes[-1].set_xlabel("step number", fontsize=label_size);
        plt.show()
        
    def plot_corner(self, pars=None, bins=20, thin=1, discard=0,
                    q=[0.16,0.5,0.84], show_titles=True, title_fmt =".3f",
                    multiply_by=1, add_value= 0, force_plot = False ):
        """
            Corner plot of selected parameters.
              
            Parameters:
            ----------
            pars : list of str;
                parameter names to plot

            bins : int;
                number of bins in 1d histogram

            thin : int;
                factor by which to thin the chains in order to reduce correlation.

            discard : int;
                to discard first couple of steps within the chains. 
            
            q : list of floats;
                quantiles to show on the 1d histograms. defaults correspoind to +/-1 sigma
                
             
        """
        assert pars is None or isinstance(pars, list) or pars is "all", \
             f'pars must be None, "all", or list of relevant parameters.'
        if pars is None or pars is "all": pars = [p for p in self._par_names]

        ndim = len(pars)

        if not force_plot: assert ndim <= 10, f'number of parameters to plot should be <=10 for clarity. \
            Use force_plot = True to continue anyways.'

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
        
        plt.show()



class _param_obj():
    def __init__(self,par_list):
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
    

        
    def __repr__(self):
        return f"{self.__dict__}"
    
    def _get_list(self):
        return [p for p in self.__dict__.values()]
    
                                      
    def _raise(exception_type, msg):
        raise exception_type(msg)
        
        
    #     def __init__(self, RpRs=0.1, Impact_para=0, Duration=0.1245, T_0=0, Period=3, 
    #              Eccentricity=0, Omega=90, K=0):
    #     """
    #         Define parameters of the signals to be fitted. configure the parameters using the `configure_parameters` method.
    #     """
        
    #     DA = _reversed_dict(locals().copy() )         #dict of arguments (DA)
    #     _ = DA.pop("self")                            #remove self from dictionary
        
    #     for n in DA.keys():
    #         if DA[n] is None: DA[n] = 0
                
    #     self._parname = [n for n in DA.keys()]
    #     self._value = [DA[n] for n in DA.keys()]
        
    #     self._npars = 8
    #     self._to_fit_flag = False
    #     # self.stellar_parameters()
    #     # self.mcmc_setup()
    
    # def configure_parameters(self, to_fit="n", start_value=None, step_size=0.001,
    #                          prior ="n", prior_mean=None, prior_width_lo=None, prior_width_hi=None,
    #                          bounds_lo=None, bounds_hi=None, verbose=True):
    #     """
    #         setup fit parameters
            
    #         Parameters:
    #         -----------
            
    #         to_fit : list containing "y" or "n", or str ;
    #             specify parameters to fit where each element in the list relates to the paramters in the order
    #             ['RpRs', 'Impact_para', 'Duration', 'T_0', 'Period', 'Eccentricity', 'Omega', 'K'].
    #             If a single str is given, all parameters are set to the given string.
            
    #         start_value : list, None;
    #             define start value for each parameter. If None, it takes takes the values defined/defaults of  `fit_setup()`.
            
    #         step_size : list, float;
    #             step size for each paramter. if type float, the same step size is used for all parameters.
                
    #         prior : list containing "y" or "n", or str;
    #             specify if to use gaussian priors on the parameters. If a single str is given, all parameters are set to the given string.
                
    #         prior_mean : list of floats;
    #             mean value of the gaussian prior for each parameter
            

            
    #         prior_width_lo, prior_width_hi : list, None;
    #             corresponding to the lower and upper width of the parameter priors. 
            
    #         bounds_lo, bounds_hi : list, None;
    #             corresponsing to the lower and upper bound of each parameter.
                
    #         verbose: bool;
    #             print output configuration or not.
            
    #     """
    #     #todo allow user input paramters to fit and their values
        
    #     if prior_mean is None: prior_mean = self._value
    #     if start_value is None: start_value = self._value
            
    #     DA = _reversed_dict(locals().copy())
    #     _ = DA.pop("self")            #remove self from dictionary
    #     _ = DA.pop("verbose")
        
    #     for par in DA.keys():
    #         assert DA[par] is None or isinstance(DA[par], (int,float,str)) or (isinstance(DA[par], list) and len(DA[par]) == self._npars), f"parameter {par} must be a list of length {self._npars} or float or None."
    #         if isinstance(DA[par], (int,float,str)): DA[par] = [DA[par]]*self._npars
        
    #     if DA["prior_width_lo"] is None:
    #         DA["prior_width_lo"] = [0.2*pr for pr in prior_mean]
    #     if DA["prior_width_hi"] is None:
    #         DA["prior_width_hi"] = [0.2*pr for pr in prior_mean]
            
    #     if DA["bounds_lo"] is None: DA["bounds_lo"] = [0.4*pr for pr in prior_mean]
    #     if DA["bounds_hi"] is None: DA["bounds_hi"] = [1.5*pr for pr in prior_mean]
            
    #     for i in range(self._npars):    
    #         if DA["to_fit"][i] == 'n': DA["step_size"][i] = 0.
    #         if (DA["prior"][i]=='n' or DA["to_fit"][i]=='n' or DA["step_size"][i] == 0.): 
    #             DA["prior_mean"][i] = 0.
    #             DA["prior_width_lo"][i] = 0
    #             DA["prior_width_hi"][i] = 0
        
    #     self._to_fit_flag = np.array(DA["to_fit"]) == "y"   
    #     self._fit_config = DA
        
    #     if verbose:
    #         print(f"{'name':12s}\tfit\tstart_val\tstepsize\tlow_lim\tup_lim\tprior\tvalue\tsig_lo\tsig_hi")
    #         txtfmt = "{0:12s}\t{1:3s}\t{2:8.4f}\t{3:.7f}\t{4:4.2f}\t{5:4.2f}\t{6}\t{7:.3f}\t{8:.3f}\t{9:.3f} "
    #         for i in range(self._npars):
    #             out_txt = txtfmt.format(self._parname[i], DA["to_fit"][i], DA["start_value"][i],DA["step_size"][i],
    #                                     DA["bounds_lo"][i],DA["bounds_hi"][i], DA["prior"][i], DA["prior_mean"][i],
    #                                     DA["prior_width_lo"][i], DA["prior_width_hi"][i])
    #             print(out_txt)
    