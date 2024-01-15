# %%
import numpy as np
import os
try:
    import lightkurve as lk
except ImportError:
    raise ImportError("lightkurve not installed. run `pip install lightkurve`")

class get_TESS_data(object):
    """
    Class to download and save TESS light curves from MAST.

    Parameters
    ----------
    planet_name : str
        Name of the planet.
    """
    def __init__(self, planet_name):
        self.planet_name = planet_name
        self.lc          = {}

    def search(self,sectors=None,author=None,exptime=None):
        """
        Search for TESS light curves on MAST using the lightkurve package.
        see documentation for lightkurve.search_lightcurve for more information.

        Parameters
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
        
    def download(self,sectors=None,author=None,exptime=None, select_flux="pdcsap_flux",quality_bitmask="hard"):
        """  
        Download TESS light curves from MAST using the lightkurve package

        Parameters
        ----------
        sectors : int or list of ints
            TESS sector lightcurve to download.
        author : str
            Author of the light curve. Default is "SPOC".
        exptime : float
            Exposure time of the light curve.
        quality_bitmask : str
            Quality bitmask to use for the lightkurve package. Default is "hard".
        """
        if sectors is not None: self.sectors = sectors
        if author  is not None: self.author  = author
        if exptime is not None: self.exptime = exptime
        assert select_flux in ["pdcsap_flux","sap_flux"], "select_flux must be either 'pdcsap_flux' or 'sap_flux'"

        if isinstance(self.sectors,int): self.sectors = [self.sectors]

        for s in self.sectors:
            self.lc[s] = lk.search_lightcurve(self.planet_name,author=self.author,sector=s,
                                              exptime=self.exptime).download(quality_bitmask=quality_bitmask)
            self.lc[s] = self.lc[s].select_flux(select_flux) 
            self.lc[s]= self.lc[s].remove_nans().normalize()
            print(f"downloaded lightcurve for sector {s}")

    def discard_ramp(self,length=0.25):
        """
        Discard data at the begining of the orbits that typically feature ramps

        Parameters
        ----------
        length : float
            length of data (in days) to discard at beginning of each orbit.
        """

        assert self.lc != {}, "No light curves downloaded yet. Run `download()` first."
        for s in self.sectors:
            gap = np.diff(self.lc[s]) > 0.5
        #TODO: implement this
        NotImplemented

    def save_CONAN_lcfile(self):
        """
        Save TESS light curves as a CONAN light curve file.
        """

        assert self.lc != {}, "No light curves downloaded yet. Run `download()` first."
        if not os.path.exists("data"): os.mkdir("data")

        for s in self.sectors:
            t, f, e = self.lc[s]["time"].value, self.lc[s]["flux"].value.unmasked, self.lc[s]["flux_err"].value.unmasked    
            file = f"data/{self.planet_name}_S{s}.dat"
            np.savetxt(file,np.stack((t,f,e),axis=1),fmt='%.8f')
            print(f"saved file as: {file}")

    # def cheops(planet_name):
    #     try:
    #         import pycheops
    #     except ImportError:
    #         raise ImportError("pycheops not installed. run `pip install pycheops`")
    #     #TODO: implement this
    #     NotImplemented

    # def create_CONAN_lcfile():
    #     #TODO: implement this
    #     NotImplemented




def get_parameters(planet_name, database="exoplanetarchive"):
    """
    get stellar and planet parameters from nasa exoplanet archive or exoplanet.eu

    """
    params = {"star":{}, "planet":{}}

    assert database in ["exoplanetarchive","exoplanet.eu"], "database must be either 'exoplanetarchive' or 'exoplanet.eu'"

    if database == "exoplanetarchive":
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        print("Getting system parameters from NASA exoplanet archive ...")
        df = NasaExoplanetArchive.query_object(planet_name,table="pscomppars")
        df = df.to_pandas()

        params["star"]["Teff"]     = (df["st_teff"][0],df["st_tefferr1"][0])
        params["star"]["logg"]     = (df["st_logg"][0],df["st_loggerr1"][0])
        params["star"]["FeH"]      = (df["st_met"][0],df["st_meterr1"][0])
        params["star"]["radius"]   = (df["st_rad"][0], df["st_raderr1"][0])
        params["star"]["mass"]     = (df["st_mass"][0],df["st_masserr1"][0])
        params["star"]["density"]  = (df["st_dens"][0],df["st_denserr1"][0])

        params["planet"]["period"] = (df["pl_orbper"][0],df["pl_orbpererr1"][0])
        params["planet"]["rprs"]   = (df["pl_ratror"][0],df["pl_ratrorerr1"][0])
        params["planet"]["mass"]   = (df["pl_bmassj"][0],df["pl_bmassjerr1"][0])
        params["planet"]["ecc"]    = (df["pl_orbeccen"][0],df["pl_orbeccenerr1"][0])
        params["planet"]["w"]      = (df["pl_orblper"][0],df["pl_orblpererr1"][0])
        params["planet"]["T0"]     = (df["pl_tranmid"][0],df["pl_tranmiderr1"][0])
        params["planet"]["b"]      = (df["pl_imppar"][0],df["pl_impparerr1"][0])
        params["planet"]["T14"]    = (df["pl_trandur"][0],df["pl_trandurerr1"][0])
        params["planet"]["aR"]     = (df["pl_ratdor"][0],df["pl_ratdorerr1"][0])
    else:
        NotImplemented

    return params
