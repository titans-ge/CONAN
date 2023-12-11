# %%
class get_files(object):
    def __init__(self,mission):
        self.mission = mission



    def tess(planet_name,sectors=None,author=None,exptime=None):
        try:
            import lightkurve as lk
        except ImportError:
            raise ImportError("lightkurve not installed. run `pip install lightkurve`")

        print(lk.search_lightcurve(planet_name,author=author,sector=sectors,exptime=exptime))
        #TODO: implement this

    def cheops(planet_name):
        try:
            import pycheops
        except ImportError:
            raise ImportError("pycheops not installed. run `pip install pycheops`")
        #TODO: implement this
        NotImplemented

    def create_CONAN_lcfile():
        #TODO: implement this
        NotImplemented