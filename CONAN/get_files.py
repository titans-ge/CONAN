# %%
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astroquery.mast import Observations
from .utils import rho_to_tdur
from uncertainties import ufloat


try:
    import lightkurve as lk
except:
    print("Warning: LightKurve not installed, won't be able to download TESS data. run `pip install lightkurve`")
try:
    from pycheops import Dataset
except:
    print("Warning: pycheops not installed, won't be able to download CHEOPS data. run `pip install pycheops`")
try:
    from dace_query.cheops import Cheops
except:
    print("Warning: dace_query not installed, won't be able to download CHEOPS data. run `pip install dace_query`")


class get_TESS_data(object):
    """
    Class to download and save TESS light curves from MAST (using the `lightkurve` package).

    Parameters:
    ----------
    planet_name : str
        Name of the planet.

    Examples
    --------
    >>> from CONAN.get_files import get_TESS_data
    >>> df = get_TESS_data("WASP-121")
    >>> df.search(author="SPOC", exptime=120)
    >>> df.download(sectors= [7,33,34,61], author="SPOC", exptime=120)
    >>> df.scatter()
    >>> df.save_CONAN_lcfile(bjd_ref = 2450000, folder="data")
    """

    def __init__(self, planet_name):
        self.planet_name = planet_name
        self.lc          = {}
        self.contam      = {}

    def search(self,sectors=None,author=None,exptime=None):
        """
        Search for TESS light curves on MAST using the lightkurve package.
        see documentation for lightkurve.search_lightcurve for more information.

        Parameters:
        ----------
        sectors : int or list of ints
            TESS sectors to search for light curves.
        author : str
            Author of the light curve. Default is "SPOC".
        exptime : float
            Exposure time of the light curve.
        """
        self.author  = author
        self.sectors = sectors
        self.exptime = exptime

        print(lk.search_lightcurve(self.planet_name,author=self.author,sector=self.sectors,
                                    exptime=self.exptime, mission="TESS"))
        
    def download(self,sectors=None,author=None,exptime=None, select_flux="pdcsap_flux",quality_bitmask="default"):
        """  
        Download TESS light curves from MAST using the lightkurve package

        Parameters:
        ----------
        sectors : int or list of ints
            TESS sector lightcurve to download.
        author : str
            Author of the light curve. Default is "SPOC".
        exptime : float
            Exposure time of the light curve.
        quality_bitmask : str
            Quality bitmask to use for the lightkurve package. options are ["none","default","hard","hardest"]. Default is "default".
        """
        if sectors is not None: self.sectors = sectors
        if author  is not None: self.author  = author
        if exptime is not None: self.exptime = exptime
        # assert select_flux in ["pdcsap_flux","sap_flux"], "select_flux must be either 'pdcsap_flux' or 'sap_flux'"

        if isinstance(self.sectors,int): self.sectors = [self.sectors]

        for s in self.sectors:
            self.lc[s] = lk.search_lightcurve(self.planet_name,author=self.author,sector=s,
                                                exptime=self.exptime).download(quality_bitmask=quality_bitmask)
            try:
                self.lc[s] = self.lc[s].select_flux(select_flux)
            except:
                print(f"{select_flux} is not available. Using 'sap_flux' instead. other options include ['kspsap_flux','det_flux'] ")
                self.lc[s] = self.lc[s].select_flux("sap_flux")
                

            self.lc[s]= self.lc[s].remove_nans().normalize()
            print(f"downloaded lightcurve for sector {s}")

            self.contam[s] = 1 - self.lc[s].hdu[1].header["CROWDSAP"]

        if hasattr(self,"_ok_mask"): del self._ok_mask
        self.data_splitted = False
        

    def discard_ramp(self,length=0.25, gap_size=1, start=True, end=True):
        """
        Discard data at the start/end of the orbits (or large gaps) that typically feature ramps

        Parameters:
        ----------
        length : float,list
            length of data (in days) to discard at beginning of each orbit. give list of length for each sector or single value for all sectors.

        gap_size : float
            minimum size of gap (in days) to detect for separating the orbits.

        start : bool
            discard data of length days at the start of the orbit. Default is True.

        end : bool
            discard data of length days at the end of the orbit. Default is True.    
        """
        if hasattr(self,"_ok_mask"): 
            print("points have already been discarded. Run `.download()` again to restart.")
            return
        assert self.lc != {}, "No light curves downloaded yet. Run `download()` first."
        if isinstance(length,(int,float)): length = [length]*len(self.sectors)
        assert len(length)==len(self.sectors), "length must be a single value or list of the same length as sectors"

        for i,s in enumerate(self.sectors):
            tt = self.lc[s].time.value
            gap = np.diff(tt) 
            gap = np.insert(gap,0,0)   #insert diff of 0 at the beginning
            
            gap_bool = gap > gap_size
            print(f"sector {s}: {sum(gap_bool)} gap(s)>{gap_size}d detected",end="; ")
            chunk_start_ind = np.append(0, np.where(gap_bool)[0])
            chunk_end_ind   = np.append(np.where(gap_bool)[0]-1, len(tt)-1)

            #mask points that are length days from chunk_start_ind
            start_mask, end_mask = [], []
            for st_ind,end_ind in zip(chunk_start_ind, chunk_end_ind):
                if start: start_mask.append((tt >= tt[st_ind]) & (tt < tt[st_ind]+length[i]))
                if end: end_mask.append((tt <= tt[end_ind]) & (tt > tt[end_ind]-length[i]))
            try:
                start_mask = np.logical_or(*start_mask) if start else np.array([False]*len(tt))
                end_mask   = np.logical_or(*end_mask) if end else np.array([False]*len(tt))
                self._nok_mask = np.logical_or(start_mask, end_mask)
                self._ok_mask  = ~np.logical_or(start_mask, end_mask)
                self.lc[s] = self.lc[s][self._ok_mask]
                print(f"discarded {sum(self._nok_mask)} points")
            except:
                print("could not discard points")


    def scatter(self):
        """
        Plot the scatter of the light curves.
        """
        assert self.lc != {}, "No light curves downloaded yet. Run `download()` first."
        for s in self.sectors:
            ax = self.lc[s].scatter()
            ax.set_title(f"sector {s}")

    def split_data(self, gap_size=None, split_times=None, show_plot=True):
        """
        Split the light curves into sections based on (1)gaps in the data, (2)given time intervals (3)given split time(s).

        Parameters:
        ----------
        gap_size : float
            minimum gap size at which data will be splitted. same unit as the time axis. Default is None.
        split_times : float,list
            Time to split the data. Default is None.
        show_plot : bool
            Show plot of the split times. Default is True.
        """

        assert self.lc != {}, "No light curves downloaded yet. Run `download()` first."
        if isinstance(split_times,(int,float)): split_times = [split_times] #convert to list
        if self.data_splitted: 
            print("data has already being split. Run `.download()` again to restart.")
            return

        remove_sectors = []
        curr_sector = self.sectors.copy()
        for s in curr_sector:
            t = self.lc[s].time.value
    
            if gap_size is not None:
                self._find_gaps(t,gap_size)
                if show_plot:
                    ax = self.lc[s].scatter()
                    [ax.axvline(t[i], color='r', linestyle='--') for i in self.gap_ind[1:-1]]
                nsplit = len(self.gap_ind)-1
                for i in range(len(self.gap_ind)-1):
                    self.lc[f"{s}_{i+1}"] = self.lc[s][(t >= t[self.gap_ind[i]]) & (t < t[self.gap_ind[i+1]])]
            
            if split_times is not None:
                split_times = np.array([0] +split_times +[t[-1]])
                if show_plot:
                    ax = self.lc[s].scatter()
                    [ax.axvline(ti, color='r', linestyle='--') for ti in split_times[1:-1]]
                nsplit = len(split_times)-1
                for i in range(len(split_times)-1):
                    self.lc[f"{s}_{i+1}"] = self.lc[s][(t >= split_times[i]) & (t < split_times[i+1])]
            
            if split_times is None and gap_size is None:
                raise ValueError("either gap_size or split_time must be given")
            
            remove_sectors.append(s)
            self.sectors += [f"{s}_{i+1}" for i in range(nsplit)]
        
        print(f"Sector {s} data splitted into {nsplit} chunks: {list(self.sectors)}")
        #remove sectors that have been split        
        _ = [self.sectors.pop(self.sectors.index(s)) for s in remove_sectors]
        self.data_splitted = True
        

    def _find_gaps(self,t,gap_size):
        """
        find gaps in the data
        """
        gap = np.diff(t) 
        gap = np.insert(gap,len(gap),0) #insert diff of 0 at the beginning
        self.gap_ind = np.where(gap > gap_size)[0]
        self.gap_ind = np.array([0]+list(self.gap_ind)+[len(t)-1])
        # self.gap_ind = np.append(0,self.gap_ind)
        # self.gap_ind = np.append(self.gap_ind, len(t)-1)

    
    def save_CONAN_lcfile(self,bjd_ref = 2450000, folder="data", out_filename=None):
        """
        Save TESS light curves as a CONAN light curve file.

        Parameters:
        ----------
        bjd_ref : float
            BJD reference time to use for the light curve file.
        folder : str
            Folder to save the light curve file in.
        out_filename : str
            Name of the output file. Default is None in which case the file will be named as "{planet_name}_S{sector}.dat"
        """

        assert self.lc != {}, "No light curves downloaded yet. Run `download()` first."
        if not os.path.exists(folder): os.mkdir(folder)

        for s in self.sectors:
            try:
                t, f, e = self.lc[s]["time"].value, self.lc[s]["flux"].value.unmasked, self.lc[s]["flux_err"].value.unmasked   
            except AttributeError:
                t, f, e = self.lc[s]["time"].value, self.lc[s]["flux"].value, self.lc[s]["flux_err"].value
            t = t + 2457000 - bjd_ref
            file = f"{folder}/{self.planet_name}_S{s}.dat" if out_filename is None else f"{folder}/{out_filename}"
            np.savetxt(file, np.stack((t,f,e),axis=1),fmt='%.8f',
                        header=f'time-{bjd_ref} {"flux":10s} {"flux_err":10s}')
            print(f"saved file as: {file}")

class get_Kepler_data(object):
    """
    Class to download and save Kepler light curves from MAST (using the `lightkurve` package).

    Parameters:
    ----------
    planet_name : str
        Name of the planet.

    Examples
    --------
    >>> from CONAN.get_files import get_TESS_data
    >>> df = get_TESS_data("`Kepler-1658")
    >>> df.search(quarter=[2,4,7], exptime=60)
    >>> df.download(quarter=[2,4,7], exptime=60)
    >>> df.scatter()
    >>> df.save_CONAN_lcfile(bjd_ref = 2450000, folder="data")
    """

    def __init__(self, planet_name):
        self.planet_name = planet_name
        self.lc          = {}
        self.contam      = {}

    def search(self,quarters=None,exptime=None):
        """
        Search for TESS light curves on MAST using the lightkurve package.
        see documentation for lightkurve.search_lightcurve for more information.

        Parameters:
        ----------
        quarters : int or list of ints
            quarters to search for light curves.
        exptime : float
            Exposure time of the light curve.
        """
        search_res = lk.search_lightcurve(self.planet_name,author="Kepler",quarter=quarters,
                                    exptime=exptime, mission="Kepler")
        print(search_res)
        self.quarters = quarters if quarters!=None else [int(q[-2:]) for q in search_res.mission]
        self.quarters = list(np.unique(self.quarters))
        self.exptime = exptime
        
    def download(self,quarters=None,exptime=None, select_flux="pdcsap_flux", merge_same_quarter=True,quality_bitmask="default"):
        """  
        Download Keplerlight curves from MAST using the lightkurve package

        Parameters:
        ----------
        quarters : int or list of ints
            Quarter lightcurve to download.
        exptime : float
            Exposure time of the light curve.
        select_flux : str
            Flux to select from the light curve. One of ["pdcsap_flux","sap_flux"]. Default is "pdcsap_flux".
        merge_same_quarter : bool
            Merge light curves from the same quarter. Default is True.
        quality_bitmask : str
            Quality bitmask to use for the lightkurve package. options are ["none","default","hard","hardest"]. Default is "default".
        """
        self.quarters = self.quarters if quarters is None else quarters
        if exptime is not None: self.exptime = exptime
        assert select_flux in ["pdcsap_flux","sap_flux"], "select_flux must be either 'pdcsap_flux' or 'sap_flux'"

        if isinstance(self.quarters,int): self.quarters = [self.quarters]
        curr_quarters = self.quarters.copy()
        for s in curr_quarters:
            if merge_same_quarter:
                self.lc[s] = lk.search_lightcurve(self.planet_name,author="Kepler",quarter=s,exptime=self.exptime
                                                    ).download_all(quality_bitmask=quality_bitmask
                                                                    ).stitch(lambda x: x.select_flux(select_flux).normalize())
                self.contam[s] = None # 1 - self.lc[s].hdu[1].header["CROWDSAP"]
                print(f"downloaded lightcurve for quarter {s}")

            else:
                temp_data = lk.search_lightcurve(self.planet_name,author="Kepler",quarter=s,exptime=self.exptime
                                                    ).download_all(quality_bitmask=quality_bitmask)
                new_s = s
                for lc in temp_data:
                    new_s = round(new_s+0.1,1)
                    self.lc[new_s] = lc.select_flux(select_flux).normalize()
                    self.quarters += [new_s]
                    self.contam[new_s] = None # 1 - self.lc[s].hdu[1].header["CROWDSAP"]

                self.quarters.remove(s)
                print(f"downloaded lightcurve for quarter {s}, split into {len(temp_data)} part(s)")

        if hasattr(self,"_ok_mask"): del self._ok_mask
        self.data_splitted = False
        

    def discard_ramp(self,length=0.25, gap_size=1, start=True, end=True):
        """
        Discard data at the start/end of the orbits (or large gaps) that typically feature ramps

        Parameters:
        ----------
        length : float,list
            length of data (in days) to discard at beginning of each orbit. give list of length for each quarter or single value for all quarters.

        gap_size : float
            minimum size of gap (in days) to detect for separating the orbits.

        start : bool
            discard data of length days at the start of the orbit. Default is True.

        end : bool
            discard data of length days at the end of the orbit. Default is True.    
        """
        if hasattr(self,"_ok_mask"): 
            print("points have already been discarded. Run `.download()` again to restart.")
            return
        assert self.lc != {}, "No light curves downloaded yet. Run `download()` first."
        if isinstance(length,(int,float)): length = [length]*len(self.quarters)
        assert len(length)==len(self.quarters), "length must be a single value or list of the same length as quarters"

        for i,s in enumerate(self.quarters):
            tt = self.lc[s].time.value
            gap = np.diff(tt) 
            gap = np.insert(gap,0,0)   #insert diff of 0 at the beginning
            
            gap_bool = gap > gap_size
            print(f"quarter {s}: {sum(gap_bool)} gap(s)>{gap_size}d detected",end="; ")
            chunk_start_ind = np.append(0, np.where(gap_bool)[0])
            chunk_end_ind   = np.append(np.where(gap_bool)[0]-1, len(tt)-1)

            #mask points that are length days from chunk_start_ind
            start_mask, end_mask = [], []
            for st_ind,end_ind in zip(chunk_start_ind, chunk_end_ind):
                if start: start_mask.append((tt >= tt[st_ind]) & (tt < tt[st_ind]+length[i]))
                if end: end_mask.append((tt <= tt[end_ind]) & (tt > tt[end_ind]-length[i]))
            try:
                start_mask = np.logical_or(*start_mask) if start else np.array([False]*len(tt))
                end_mask   = np.logical_or(*end_mask) if end else np.array([False]*len(tt))
                self._nok_mask = np.logical_or(start_mask, end_mask)
                self._ok_mask  = ~np.logical_or(start_mask, end_mask)
                self.lc[s] = self.lc[s][self._ok_mask]
                print(f"discarded {sum(self._nok_mask)} points")
            except:
                print("could not discard points")
        #TODO: discards ramp only when one gap is detected. fix

    def scatter(self):
        """
        Plot the scatter of the light curves.
        """
        assert self.lc != {}, "No light curves downloaded yet. Run `download()` first."
        for s in self.quarters:
            ax = self.lc[s].scatter()
            ax.set_title(f"quarter {s}")

    def split_data(self, gap_size=None, split_times=None, show_plot=True):
        """
        Split the light curves into sections based on (1)gaps in the data, (2)given time intervals (3)given split time(s).

        Parameters:
        ----------
        gap_size : float
            minimum gap size at which data will be splitted. same unit as the time axis. Default is None.
        split_times : float,list
            Time to split the data. Default is None.
        show_plot : bool
            Show plot of the split times. Default is True.
        """

        assert self.lc != {}, "No light curves downloaded yet. Run `download()` first."
        if isinstance(split_times,(int,float)): split_times = [split_times] #convert to list
        if self.data_splitted: 
            print("data has already being split. Run `.download()` again to restart.")
            return

        remove_quarters = []
        curr_quarter = self.quarters.copy()
        for s in curr_quarter:
            t = self.lc[s].time.value
    
            if gap_size is not None:
                self._find_gaps(t,gap_size)
                if show_plot:
                    ax = self.lc[s].scatter()
                    [ax.axvline(t[i], color='r', linestyle='--') for i in self.gap_ind[1:-1]]
                nsplit = len(self.gap_ind)-1
                for i in range(len(self.gap_ind)-1):
                    self.lc[f"{s}_{i+1}"] = self.lc[s][(t >= t[self.gap_ind[i]]) & (t < t[self.gap_ind[i+1]])]
            
            if split_times is not None:
                split_times = np.array([0] +split_times +[t[-1]])
                if show_plot:
                    ax = self.lc[s].scatter()
                    [ax.axvline(ti, color='r', linestyle='--') for ti in split_times[1:-1]]
                nsplit = len(split_times)-1
                for i in range(len(split_times)-1):
                    self.lc[f"{s}_{i+1}"] = self.lc[s][(t >= split_times[i]) & (t < split_times[i+1])]
            
            if split_times is None and gap_size is None:
                raise ValueError("either gap_size or split_time must be given")
            
            remove_quarters.append(s)
            self.quarters += [f"{s}_{i+1}" for i in range(nsplit)]
        
        print(f"quarter {s} data splitted into {nsplit} chunks: {list(self.quarters)}")
        #remove quarters that have been split        
        _ = [self.quarters.pop(self.quarters.index(s)) for s in remove_quarters]
        self.data_splitted = True
        

    def _find_gaps(self,t,gap_size):
        """
        find gaps in the data
        """
        gap = np.diff(t) 
        gap = np.insert(gap,len(gap),0) #insert diff of 0 at the beginning
        self.gap_ind = np.where(gap > gap_size)[0]
        self.gap_ind = np.array([0]+list(self.gap_ind)+[len(t)-1])
        # self.gap_ind = np.append(0,self.gap_ind)
        # self.gap_ind = np.append(self.gap_ind, len(t)-1)

    
    def save_CONAN_lcfile(self,bjd_ref = 2450000, folder="data", out_filename=None):
        """
        Save Kepler light curves as a CONAN light curve file.

        Parameters:
        ----------
        bjd_ref : float
            BJD reference time to use for the light curve file. Note that kepler time is in BJD-2454833
        folder : str
            Folder to save the light curve file in.
        out_filename : str
            Name of the output file. Default is None in which case the file will be named as "{planet_name}_S{quarter}.dat"
        """

        assert self.lc != {}, "No light curves downloaded yet. Run `download()` first."
        if not os.path.exists(folder): os.mkdir(folder)

        for s in self.quarters:
            # try:
            t, f, e = self.lc[s]["time"].value, np.array(self.lc[s]["flux"]), np.array(self.lc[s]["flux_err"])   
            # except AttributeError:
            #     t, f, e = self.lc[s]["time"].value, self.lc[s]["flux"].value, self.lc[s]["flux_err"].value
            t = t + 2454833 - bjd_ref
            file = f"{folder}/{self.planet_name}_Q{s}.dat" if out_filename is None else f"{folder}/{out_filename}"
            np.savetxt(file, np.stack((t,f,e),axis=1),fmt='%.8f',
                        header=f'time-{bjd_ref} {"flux":10s} {"flux_err":10s}')
            print(f"saved file as: {file}")


class get_CHEOPS_data(object):
    """
    Class to download and save CHEOPS light curves from CHEOPS archive (using `dace_query` and `pycheops`).

    Parameters:
    ----------
    planet_name : str
        Name of the planet.

    Examples
    --------
    >>> from CONAN.get_files import get_CHEOPS_data
    >>> df = get_CHEOPS_data("KELT-20")
    >>> df.search( filters = { "pi_name":{"contains":["LENDL"]}, "data_arch_rev":{"equal":[3]} })
    >>> df.download(file_keys="all", aperture="DEFAULT")
    >>> df.scatter()
    >>> df.save_CONAN_lcfile(bjd_ref = 2450000, folder="data")
    """
    
    def __init__(self, planet_name):
        self.planet_name = planet_name
        self.lc          = []

    def search(self,filters=None):
        """   
        Query the cheops database to retrieve available visits. Filters and sorting order can be applied to the query via named arguments (see query_options)
        
        Parameters:
        ----------
        filters : dict
            Dictionary of filters to apply to the query. 
            e.g. filters = {"pi_name":{"contains":["LENDL"]},"data_arch_rev":{"equal":[3]}} will return all visits of `planet_name` by PI:Lendl with data archive version 3.
        """
        self.filter = {'obj_id_catname': {'contains':[self.planet_name]}}
        self.filter.update(filters)

        data = Cheops.query_database(filters=self.filter, sort={"date_mjd_start":"asc"})
        self.file_keys = data["file_key"]
        print(pd.DataFrame(data, columns=["obj_id_catname","file_key", "pi_name", "date_mjd_start",
                                            "obs_total_exptime", "data_arch_rev",'status_published']))

    def download(self, file_keys="all", aperture="DEFAULT", decontaminate=True, reject_highpoints=True, bg_MAD_reject=3,
                 outlier_clip=4, outlier_window_width=11,outlier_clip_niter=1, verbose=True):
        """   
        Download CHEOPS light curves from the cheops database using the dace_query package

        Parameters:
        ----------
        file_keys : list of str
            List of file keys to download. if "all" is given, all file_keys shown in `.search()` result will be downloaded.
        aperture : str
            Aperture to use for the light curve. Default is "DEFAULT". The options are: "R15","R16",...,"R40","RINF","RSUP".
        decontaminate : bool
            Decontaminate the light curve. Default is True.
        reject_highpoints : bool
            Reject high points in the light curve. Default is True. 
        bg_MAD_reject : float
            Number of median absolute deviations to reject high background points. Default is 3.
        outlier_clip : float
            Number of standard deviations to clip outliers. Default is 4.
        outlier_window_width : int
            Window width to use for clipping outliers. Default is 11.
        outlier_clip_niter : int
            Number of iterations to clip outliers. Default is 3.
        """
        
        
        if isinstance(file_keys,str):
            if file_keys=="all": file_keys = self.file_keys 
            else: 
                file_keys = [file_keys]
        for f in file_keys: assert f in self.file_keys, f"file_key must be in {self.file_keys}"

        if isinstance(aperture,str): aperture = [aperture]*len(file_keys)
        elif isinstance(aperture, list): assert len(aperture)==len(file_keys), "aperture must be a single string or list of the same length as file_keys"

        for i,file_key in enumerate(file_keys):
            d     = Dataset(file_key,verbose=False)
            _,_,_ = d.get_lightcurve(aperture=aperture[i],decontaminate=decontaminate, reject_highpoints=reject_highpoints,verbose=False)
            if verbose: print(f"downloaded lightcurve with file key: {file_key}, aperture: {aperture[i]}")
            
            #remove points with high background
            mask  = d.lc["bg"] > bg_MAD_reject*np.nanmedian(d.lc["bg"])
            _,_,_ = d.mask_data(mask, verbose=False)

            #iterative sigma clipping
            for _ in range(outlier_clip_niter): t_,f_,e_ = d.clip_outliers(outlier_clip,outlier_window_width, verbose=verbose)

            self.lc.append(d)

    def scatter(self, figsize=(10,4)):
        """
        Plot the scatter of the light curves.
        """
        assert self.lc != {}, "No light curves downloaded yet. Run `.download()` first."
        for d in self.lc:
            plt.figure(figsize=figsize)
            plt.scatter(d.lc["time"],d.lc["flux"],s=1, label=d.file_key)
            plt.legend()
            plt.xlabel(f"BJD - {d.bjd_ref}")
            plt.ylabel("Flux")
            plt.show()

    def save_CONAN_lcfile(self,bjd_ref = 2450000, folder="data", out_filename=None):
        """
        Save CHEOPS light curves as a CONAN light curve file.
        the output columns are [t, f, e, x_off, y_off, roll, bg, contam, deltaT]

        Parameters:
        ----------
        bjd_ref : float
            BJD reference time to use for the light curve file.
        folder : str
            Folder to save the light curve file in.
        out_filename : str
            Name of the output file. Default is None in which case the file will be named as "{planet_name}_TG{file_key}.dat"
        """
    
        rescale   = lambda x: (x - np.min(x))/np.ptp(x)          # between 0 and 1 
        rescale_r = lambda x: (2*x-(x.min()+x.max()))/np.ptp(x)  #between -1 and 1 

        assert self.lc != {}, "No light curves downloaded yet. Run `.download()` first."
        if not os.path.exists(folder): os.mkdir(folder)
        
        for d in self.lc:
            t = d.lc["time"]+d.bjd_ref - bjd_ref
            
            lc_ = np.stack((t, d.lc["flux"], d.lc["flux_err"], rescale_r(d.lc["xoff"]),
                            rescale_r(d.lc["yoff"]), self._resort_roll(d.lc["roll_angle"]),
                            rescale(d.lc["bg"]), rescale(d.lc["contam"]), d.lc["deltaT"])).T
            
            fkey = d.file_key.split("TG")[-1].split("_V")[0]
            prefix = d.target
            file = folder+"/"+prefix+"_"+fkey+".dat" if out_filename is None else f"{folder}/{out_filename}"
            np.savetxt(file, lc_, fmt="%.8f", 
                        header=f'time-{bjd_ref} {"flux":10s} {"flux_err":10s} {"x_off":10s} {"y_off":10s} {"roll":10s} {"bg":10s} {"contam":10s} {"deltaT":10s}')
            print("columns are ordered as [0:time, 1:flux, 2:flux_err, 3:x_off, 4:y_off, 5:roll, 6:bg, 7:contam, 8:deltaT]")
            print(f"saved file as {file}")

    def _resort_roll(self,x):
        #make roll-angle continuous
        x = x.copy()
        phi = np.sort(x)
        gap = np.diff(phi)
        if max(gap)>20:
            brk_pt = phi[np.argmax(gap)]+0.5*max(gap)
        else: brk_pt = 360
        # x = (360 + x - brk_pt)%360
        x[x > brk_pt] -= 360
        return x 


class get_JWST_data(object):
    """
    Class to download and save JWST light curves from the MAST archive (using `astroquery`).

    Parameters:
    ----------
    planet_name : str
        Name of the planet.

    Examples
    --------
    >>> from CONAN.get_files import get_JWST_data
    >>> df = get_JWST_data("WASP-39")
    >>> df.search(instrument="NIRSPEC/SLIT", filters="G395H", proposal_id="1366")
    >>> df.download()   #download white light curves
    >>> df.scatter()
    >>> df.save_CONAN_lcfile(bjd_ref = 2450000, folder="data")
    """
    def __init__(self, planet_name):
        self.planet_name = planet_name
        self.lc          = []

    def search(self, instrument=[], filters=[], proposal_id=[],search_kws={}):
        """
        Search for JWST light curves on MAST using the astroquery package.

        Parameters:
        ----------
        instrument : str
            Name of the JWST instrument e.g 'NIRSPEC/SLIT','MIRI/SLITLESS','NIRCAM/GRISM','NIRCAM/IMAGE','NIRISS/SOSS'.
        filters : str
            Name of the filter e.g 'G395H','F210M;WLP8','CLEAR;PRISM','GR700XD','F444W'.
        proposal_id : str
            Proposal ID of the observation. 
        search_kws : dict
            Additional search keywords to use for the query. e.g. {"proposal_pi":"Batalha, Natalie", }       

        """
        print("Searching for JWST data ...")
        for key in ["instrument_name","filters","proposal_id","project"]:
            if key in search_kws: search_kws.pop(key)

        self.instrument  = instrument
        self.filters     = filters
        self.proposal_id = proposal_id   
        self.kwargs      = search_kws

        self.obs = Observations.query_criteria(target_name=self.planet_name, instrument_name=self.instrument, 
                                            filters=self.filters, proposal_id=self.proposal_id, project='JWST',
                                            **search_kws)
        
        print(pd.DataFrame(self.obs.to_pandas(), columns=['target_name','dataproduct_type','calib_level','intentType','provenance_name','instrument_name',
                                                        'filters','dataRights','proposal_type','proposal_pi','proposal_id','t_min','t_exptime','objID']))


    def download(self, selection_kws={},overwrite=False):
        """
        Download JWST white light curves from MAST using the astroquery package.

        Parameters:
        ----------
        selection_kws : dict
            Additional selection keywords to select specific observation e.g {"t_exptime":1200, "calib_level":3}
            Default is {} to take result from search assuming it has been fine-tuned to give one result.
        overwrite : bool
            Overwrite the downloaded light curves. Default is False.
        """
        self.kwargs.update(selection_kws)
        self.obs = Observations.query_criteria(target_name=self.planet_name, instrument_name=self.instrument, 
                                            filters=self.filters, proposal_id=self.proposal_id, project='JWST',
                                            **self.kwargs)
        
        data_products = Observations.get_product_list(self.obs)
        whtlt = Observations.filter_products(data_products, productType = 'SCIENCE', productSubGroupDescription = 'WHTLT')
        lc    = Observations.download_products(whtlt, cache=not overwrite)
        self.lc = np.loadtxt(lc["Local Path"][0],skiprows=16)

    def scatter(self):
        """
        Plot the scatter of the light curves.
        """
        assert self.lc != [], "No light curves downloaded yet. Run `download()` first."
        plt.figure(figsize=(10,4))
        plt.scatter(self.lc[:,0],self.lc[:,1],s=1)
        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.show()




def get_EULER_data(FITS_filepath, out_folder=".", planet_name=None):
    """
    create .dat file for CONAN from the EULER fits file

    Parameters:
    ----------
    FITS_filepath : str
        Path to the EULER fits file.

    Returns
    -------
    file:
        .dat file for CONAN
    """
    lc = fits.open(FITS_filepath)
    
    ti        = lc[1].data['time (BJD-TDB)'] -2450000
    fl        = lc[1].data['flux']
    fl_err    = lc[1].data['sflux']

    fwhm      = lc[1].data['fwhm']
    peak_flux = lc[1].data['peak']
    air_mass  = lc[1].data['airmass']
    sky       = lc[1].data['bkg']
    xshift    = lc[1].data['dx']
    yshift    = lc[1].data['dy']
    exptime   = lc[1].data['exptime']
    try:
        planet_name = lc[0].header["OBJECT"]
    except:
        planet_name = planet_name if planet_name is not None else ""
    
    np.savetxt(f'{out_folder}/lc_{planet_name}.dat', np.transpose([ti, fl, fl_err, xshift, yshift, air_mass, fwhm, sky, exptime]), fmt= '%3.5f')
    return

class get_EULER_data_from_server(object):
    """ 
    class to get EULER data from the server at Geneva Observatory. This requires some login access

    Parameters:
    ----------
    planet_name : str
        Name of the planet.

    Examples
    --------
    >>> from CONAN.get_files import get_EULER_data
    >>> df = get_EULER_data("WASP-121")
    >>> df.search(dates=["2020-09-01","2020-09-30"])
    >>> df.download(aperture="DEFAULT")
    >>> df.scatter()
    >>> df.save_CONAN_lcfile(bjd_ref = 2450000, folder="data")
    """
    def __init__(self, planet_name):
        self.planet_name = planet_name
        self.lc          = []
        Warning("This class is not yet implemented")

    def search(self,date_range=None):
        """
        Search for EULER light curves (pipeline fits files?) on the server at Geneva Observatory between the given dates.

        Parameters:
        ----------
        date_range : list of str
            List of dates to search for light curves. Format is ["YYYY-MM-DD","YYYY-MM-DD].
        """
        self.date_range = date_range
        self.files = []
        for date in self.date_range:
            self.files += self._get_files(date)
        self.files = np.unique(self.files)
        print(f"found {len(self.files)} files")

    def download(self, file_name, aperture="DEFAULT"):
        """
        Download EULER light curves from the server at Geneva Observatory.

        Parameters:
        ----------
        file_name : str
            Name of the file to download.
        aperture : str
            Aperture to use for the light curve. Default is "DEFAULT". The options are: "R15","R16",...,"R40","RINF","RSUP".
        """
        self.file_name = file_name
        self.aperture  = aperture
        self.lc = []
        for f in self.files:
            if f.split("/")[-1] == self.file_name:
                self.lc.append(self._get_lc(f,aperture=self.aperture))
        print(f"downloaded {len(self.lc)} light curves")

    def scatter(self, figsize=(10,4)):
        """
        Plot the scatter of the light curves.
        """
        assert self.lc != [], "No light curves downloaded yet. Run `.download()` first."
        for d in self.lc:
            plt.figure(figsize=figsize)
            plt.scatter(d["time"],d["flux"],s=1, label=d.file_key)
            plt.legend()
            plt.xlabel(f"BJD - {d.bjd_ref}")
            plt.ylabel("Flux")
            plt.show()

    def save_CONAN_lcfile(self,bjd_ref = 2450000, folder="data"):
        """
        Save EULER light curves as a CONAN light curve file.

        Parameters:
        ----------
        bjd_ref : float
            BJD reference time to use for the light curve file.
        folder : str
            Folder to save the light curve file in.
        """
        rescale   = lambda x: (x - np.min(x))/np.ptp(x)          # between 0 and 1 
        rescale_r = lambda x: (2*x-(x.min()+x.max()))/np.ptp(x)

        assert self.lc != [], "No light curves downloaded yet. Run `.download()` first."
        if not os.path.exists(folder): os.mkdir(folder)






def get_parameters(planet_name, database="exoplanetarchive", table="pscomppars", ps_select_index=None,overwrite_cache=False, check_rhostar=False):
    """
    get stellar and planet parameters from nasa exoplanet archive or exoplanet.eu

    Parameters:
    ----------
    planet_name : str
        Name of the planet.
    database : str
        Database to use. Default is "exoplanetarchive". Options are "exoplanetarchive" or "exoplanet.eu".
    table : str
        Table to use. Default is "pscomppars". Options are "pscomppars" or "ps".
    ps_select_index : int
        Index of the parameter set to use. Default is None.
    overwrite_cache : bool
        Overwrite the cached parameters. Default is True.
    check_rhostar : bool
        Check if the stellar density is consistent with the Transit duration. 
        inconsistency may be due to rho_star not being the actual stellar density but just a proxy, value may therefore not be suitable as prior in CONAN.
        Default is False.
    """
    if os.path.exists(f"{planet_name.replace(' ','')}_sysparams.pkl") and overwrite_cache is False:
        print("Loading parameters from cache ...")
        params = pd.read_pickle(f"{planet_name.replace(' ','')}_sysparams.pkl")
    
    else:
        params = {"star":{}, "planet":{}}

        assert database in ["exoplanetarchive","exoplanet.eu"], "database must be either 'exoplanetarchive' or 'exoplanet.eu'"

        if database == "exoplanetarchive":
            from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

            print("Getting system parameters from NASA exoplanet archive ...")
            if table=="pscomppars":
                df = NasaExoplanetArchive.query_object(planet_name,table=table)
            elif table=="ps":
                ref_table = NasaExoplanetArchive.query_object(planet_name,table=table)
                ref_table["pl_refname"] = [ref.split('refstr=')[1].split(' href')[0] for ref in ref_table["pl_refname"]]
                ref_table["sy_refname"] = [ref.split('refstr=')[1].split(' href')[0] for ref in ref_table["sy_refname"]]
                print("\nAvailable parameters:\n---------------------")
                print(ref_table[["pl_name","default_flag","pl_refname","sy_refname",]].to_pandas())
                if ps_select_index is None:
                    df = ref_table[ref_table['default_flag']==1]
                    print(f"\nselecting default parameters: {df['pl_refname'][0]}. Set `ps_select_index` [between 0 – {len(ref_table)-1}] to select another")
                else:
                    ref_table["index"] = np.arange(len(ref_table))
                    df = ref_table[ref_table["index"]==ps_select_index]
                    print(f"\nselecting parameters: {df['pl_refname'][0]}")
            else: 
                raise ValueError("table must be either 'pscomppars' or 'ps'")
            
            df = df.to_pandas()
            
            params["star"]["Teff"]     = (df["st_teff"][0],df["st_tefferr1"][0])
            params["star"]["logg"]     = (df["st_logg"][0],df["st_loggerr1"][0])
            params["star"]["FeH"]      = (df["st_met"][0],df["st_meterr1"][0])
            params["star"]["radius"]   = (df["st_rad"][0], df["st_raderr1"][0])
            params["star"]["mass"]     = (df["st_mass"][0],df["st_masserr1"][0])
            params["star"]["density"]  = (df["st_dens"][0],df["st_denserr1"][0])

            params["planet"]["name"]   = df["pl_name"][0]
            params["planet"]["period"] = (df["pl_orbper"][0],df["pl_orbpererr1"][0])
            params["planet"]["rprs"]   = (df["pl_ratror"][0],df["pl_ratrorerr1"][0])
            params["planet"]["mass"]   = (df["pl_bmassj"][0],df["pl_bmassjerr1"][0])
            params["planet"]["ecc"]    = (df["pl_orbeccen"][0],df["pl_orbeccenerr1"][0])
            params["planet"]["w"]      = (df["pl_orblper"][0],df["pl_orblpererr1"][0])
            params["planet"]["T0"]     = (df["pl_tranmid"][0],df["pl_tranmiderr1"][0])
            params["planet"]["b"]      = (df["pl_imppar"][0],df["pl_impparerr1"][0])
            params["planet"]["T14"]    = (df["pl_trandur"][0]/24,df["pl_trandurerr1"][0]/24)
            params["planet"]["aR"]     = (df["pl_ratdor"][0],df["pl_ratdorerr1"][0])
            params["planet"]["K[m/s]"] = (df["pl_rvamp"][0],df["pl_rvamperr1"][0])
        else:
            exo_eu = pd.read_csv("http://exoplanet.eu/catalog/csv",
                                usecols=["name",'star_teff', 'star_teff_error_max',
                                            'log_g','star_metallicity', 'star_metallicity_error_max', 'star_mass',
                                            'star_mass_error_max', 'star_radius', 'star_radius_error_max',
                                            'mass', 'mass_error_max','orbital_period', 'orbital_period_error_max',
                                            'semi_major_axis','semi_major_axis_error_max', 'eccentricity','eccentricity_error_max', 
                                            'inclination','inclination_error_max','omega', 'omega_error_max','tzero_tr','tzero_tr_error_max'
                                        ])
            raise NotImplemented

        pd.to_pickle(params, f"{planet_name.replace(' ','')}_sysparams.pkl")

    if check_rhostar: #check that rho_star conversion to tdur gives similar value to T14
        ecc_w = {}
        for p in ["ecc","w"]:
            par = params["planet"][p]
            for i in [0,1]:
                ecc_w[p]= (np.where(np.isfinite(par[0]), par[0], 0).item(), np.where(np.isfinite(par[1]), par[1], 0).item())

        tdur = rho_to_tdur(rho = ufloat(*params["star"]["density"]), b= ufloat(*params["planet"]["b"]),
                            Rp = ufloat(*params["planet"]["rprs"]), P = ufloat(*params["planet"]["period"]), 
                            e = ufloat(*ecc_w["ecc"]), w = ufloat(*ecc_w["w"]))
        print(f"{'rho_star to Tdur:'} {tdur:.6f}\n{'T14:':17s} {ufloat(*params['planet']['T14']):.6f}")

        diff = tdur - ufloat(*params["planet"]["T14"])
        if diff.n/diff.s <=1:   print("rho_star conversion to Tdur is consistent with literature T14 within 1 sigma") 
        elif diff.n/diff.s <=3: print("rho_star conversion to Tdur is consistent with literature T14 within 3 sigma")
        else:                   print("rho_star conversion to Tdur is not consistent with literature T14 within 3 sigma, rho_star may be incorrect")
    
    return params
