# import unittest
# from CONAN3.models import  Transit_Model
# from CONAN3.utils import phase_fold, supersampling, convert_LD, get_transit_time, rho_to_aR, rescale0_1, cosine_atm_variation
# import numpy as np


# class TestTransitModel(unittest.TestCase):
#     def setUp(self):
#         # Initialize the Transit_Model object with sample parameters
#         T0 = [0.0]
#         RpRs = [0.1]
#         b = [0.5]
#         dur = [2.0]
#         per = [10.0]
#         sesinw = [0.0]
#         secosw = [0.0]
#         ddf = [0.0]
#         occ = [0.0]
#         c1 = [0.0]
#         c2 = [0.0]
#         npl = 1
#         self.model = Transit_Model(T0, RpRs, b, dur, per, sesinw, secosw, ddf, occ, c1, c2, npl)

#     def test_get_value(self):
#         # Test the get_value method with a sample time array
#         tarr = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
#         expected_values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Replace with expected values
#         values = self.model.get_value(tarr)
#         self.assertEqual(values, expected_values)

#     def test_parameter_names(self):
#         # Test the parameter_names attribute
#         expected_names = ['T0', 'RpRs', 'b', 'dur', 'per', 'sesinw', 'secosw', 'ddf', 'occ', 'c1', 'c2']
#         names = self.model.parameter_names
#         self.assertEqual(names, expected_names)

# def conv(rho_star, T_0, RpRs, Impact_para, Period, sesinw=[0], secosw=[0], ddf=0, 
#              q1=[0], q2=[0], D_occ=0, A_atm=None, delta=None, A_ev=None):
    
#     DA = locals().copy()
#     npl = len(T_0)
    
#     pl_vars = ["T_0", "Period", "rho_star", "D_occ", "Impact_para","RpRs", "sesinw", 
#                                                "secosw", "A_atm", "delta", "A_ev","q1","q2"]
#     tr_pars = {}
#     for p in pl_vars:
#         for n in range(npl):
#             lbl = f"_{n+1}" if npl>1 else ""                      # numbering to add to parameter names of each planet
#             if p not in ["q1","q2","rho_star","A_atm","delta","A_ev","D_occ"]:   # parameters common to all planet or not used in multi-planet fit
#                 tr_pars[p+lbl]= DA[p][n]  #transit/eclipse pars
#             else:
#                 tr_pars[p] = DA[p]        #limb darkening pars
#     return tr_pars
    

# def batman_model(tr_params,t=None,ss_exp=None,npl=1):
#     import batman
#     ss = supersampling(ss_exp/(60*24),int(ss_exp)) if ss_exp is not None else None
#     tt_ss   = ss.supersample(t) if ss is not None else t

#     model_flux = np.zeros_like(tt_ss)

#     for n in range(1,npl+1):
#         lbl = f"_{n}" if npl>1 else ""

#         bt = batman.TransitParams()
#         bt.per = tr_params["Period"+lbl]
#         bt.t0  = tr_params["T_0"+lbl]
#         bt.rp  = tr_params["RpRs"+lbl]
#         b      = tr_params["Impact_para"+lbl]
#         bt.ecc = tr_params["sesinw"+lbl]**2 + tr_params["secosw"+lbl]**2
#         bt.w   = np.arctan2(tr_params["sesinw"+lbl],tr_params["secosw"+lbl])
#         if "rho_star" in tr_params.keys(): bt.a = rho_to_aR(tr_params["rho_star"], bt.per)
#         else: bt.a = Tdur_to_aR(tr_params["Duration"+lbl],b,bt.rp,bt.per, bt.ecc, bt.w)#         bt.fp  = tr_params["D_occ"]                                        
#         bt.fp  = tr_params["D_occ"]*1e-6                                       
#         ecc_factor=(1-bt.ecc**2)/(1+bt.ecc*np.sin(np.deg2rad(bt.w)))  

#         bt.inc = np.rad2deg(np.arccos(b/(bt.a * ecc_factor)))
#         bt.limb_dark = "quadratic"

#         u1,u2  = convert_LD(tr_params["q1"],tr_params["q2"],conv="q2u")
#         bt.u   = [u1,u2]

#         bt.t_secondary = bt.t0 + 0.5*bt.per*(1 + 4/np.pi * bt.ecc * np.cos(np.deg2rad(bt.w))) #eqn 33 (http://arxiv.org/abs/1001.2010)
#         m_tra = batman.TransitModel(bt, tt_ss,transittype="primary")
#         m_ecl = batman.TransitModel(bt, tt_ss,transittype="secondary")

#         f_tra = m_tra.light_curve(bt)
#         phase  = phase_fold(tt_ss, bt.per, bt.t0)
#         if tr_params["A_atm"]*1e-6 != 0:
#             f_occ = rescale0_1(m_ecl.light_curve(bt))

#             atm    = cosine_atm_variation(phase,bt.fp, tr_params["A_atm"]*1e-6, tr_params["ph_off"])
#             ellps  = tr_params["A_ev"]*1e-6 * (1 - (np.cos(2*(2*np.pi*phase))) )
#             dopp   = tr_params["A_db"]*1e-6 * np.sin(2*np.pi*phase)

#             model_flux += (f_tra*(1+ellps+dopp) + f_occ*atm.pc)-1           #transit, eclipse, atm model
#         else:
#             f_occ = m_ecl.light_curve(bt)
#             ellps  = tr_params["A_ev"]*1e-6 * (1 - (np.cos(2*(2*np.pi*phase))) )
#             dopp   = tr_params["A_db"]*1e-6 * np.sin(2*np.pi*phase)
#             model_flux += f_tra*(1+ellps+dopp) + (f_occ - (1+bt.fp)) - 1                   #transit, eclipse, no PC

#         model_flux = ss.rebin_flux(model_flux) if ss is not None else model_flux #rebin the model to the original cadence

#     return np.array(1+model_flux)


# pars = [0.6501, [0],[0.11572],[0.515],[3.474],[0],[0],[0],0,0,300e-6]
# tr_pars  = conv(*pars)
# batmodel = batman_model(tr_pars,t)
# TM = Transit_Model(*pars)
# mm,_ = TM.get_value(t)

# np.allclose(mm,batmodel)



# if __name__ == '__main__':
#     unittest.main()




#test light travel time
import numpy as np
import matplotlib.pyplot as plt
from CONAN3.utils import light_travel_time_correction
from CONAN3.models import Transit_Model
from CONAN3.utils import get_transit_time, get_orbital_elements, get_Tconjunctions,inclination,Tdur_to_aR
from CONAN3.get_files import get_parameters
sys_params = get_parameters("WASP-121 b")


P  = sys_params["planet"]["period"][0]
t0 = 0
t14= sys_params['planet']['T14'][0] 
b  = sys_params["planet"]["b"][0]
e  = 0.
w  = np.radians(90)
aR = Tdur_to_aR(sys_params["planet"]["T14"][0],b,sys_params["planet"]["rprs"][0],P,e,np.degrees(w))
inc = inclination(b,aR,e,np.degrees(w))

sesinw, secosw = np.sqrt(e)*np.sin(w), np.sqrt(e)*np.cos(w)

t = np.linspace(-0.25, 0.75*P, 3000)

tcorr = light_travel_time_correction(t,t0,aR,P,np.radians(inc),1.46,e,w)
tconj  = get_Tconjunctions(t,t0,P,e,w)


plt.figure()
plt.plot(t, 24*3600*(t-tcorr))
plt.axvline(tconj.transit,c="k",ls=":",label="mid-transit")
plt.axvline(tconj.eclipse,c="r",ls=":",label="mid-eclipse")
plt.legend()


TM = Transit_Model(dur=sys_params["planet"]["T14"][0], T0=0,
             RpRs=sys_params["planet"]["rprs"][0], b=b,per=P,
             sesinw=sesinw,secosw=secosw, occ=4000)

t = np.linspace(P/2-0.07,P/2+0.07,1500)
flux,_     = TM.get_value(t)
flux_ltt,_ = TM.get_value(t, Rstar=1.46)


fig,ax =plt.subplots(2,1,figsize=(15,3),sharex=True,  gridspec_kw={"height_ratios":(2,1)})
ax[0].plot(t,flux,label="no LTT")
ax[0].plot(t,flux_ltt,"--",label="LTT delay included")
ax[0].legend()
ax[0].set_title("LTT allows to move the occultation to later time to account for the delay")

ax[1].plot(t, 1e6*(flux_ltt-flux))
ax[1].set_ylabel("res [ppm]")
_=[ax[0].axvline(tt,c="k",ls=":") for tt in [tconj.eclipse]]

plt.subplots_adjust(hspace=0)


# TESS slight  ECCENTRICITY 

# TESS slight  ECCENTRICITY 


e  = 0.0035
w  = np.radians(83)
sesinw, secosw = np.sqrt(e)*np.sin(w), np.sqrt(e)*np.cos(w)

TM = Transit_Model(dur=sys_params["planet"]["T14"][0], T0=0,
             RpRs=sys_params["planet"]["rprs"][0], b=b,per=P,
             sesinw=0,secosw=0, occ=4000)

TM_ecc = Transit_Model(dur=sys_params["planet"]["T14"][0], T0=0,
             RpRs=sys_params["planet"]["rprs"][0], b=b,per=P,
             sesinw=sesinw,secosw=secosw, occ=4000)

t = np.linspace(P/2-0.1,P/2+0.1,1500)
flux,_     = TM.get_value(t)
flux_ecc,_ = TM_ecc.get_value(t)

tconj      = get_Tconjunctions(t,t0,P,ecc=0,omega=np.pi/2)
tconj_ecc  = get_Tconjunctions(t,t0,P,e,w)



fig,ax =plt.subplots(2,1,figsize=(15,4),sharex=True,  gridspec_kw={"height_ratios":(2,1)})
ax[0].plot(t,flux,label="no ecc")
ax[0].plot(t,flux_ecc,"--",label="ecc")
ax[0].legend()
ax[0].set_title(f"this ecc,w delays the eclipse time by {24*3600*(tconj_ecc.eclipse-tconj.eclipse):.1f}secs, and longer egress than ingress")

ax[1].plot(t, 1e6*(flux_ecc-flux))
ax[1].set_ylabel("res [ppm]")

_=[ax[0].axvline(tt,c="b",ls=":") for tt in [tconj.eclipse]]
_=[ax[0].axvline(tt,c="r",ls=":") for tt in [tconj_ecc.eclipse]]
plt.subplots_adjust(hspace=0)



#compare transit model to batman

import batman
from Planet_tools.convert_param import convert_LD_coeffs, inclination
def batman_model(pars,tobs):
    """
    T0,P,rp,aR,b,u1,u2,fp,fnight,delta
    
    Ftra+Fph+Focc
    """
    params = batman.TransitParams()       #object to store transit parameters
    
    #transit pars
    params.t0  = pars[0]                        #time of inferior conjunction
    params.per = pars[1]                       #orbital period
    params.rp  = pars[2]                       #planet radius (in units of stellar radii)
    params.a   = pars[3]                        #semi-major axis (in units of stellar radii)
    imp_par    = pars[4]
    params.ecc = 0.                       #eccentricity
    params.w   = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"        #limb darkening model
    q1,q2 = pars[5], pars[6]
    
    params.inc = inclination(imp_par, params.a)                      #orbital inclination (in degrees)
    u1,u2 = convert_LD_coeffs(q1,q2,"q2u",verify=False)
    params.u=[u1,u2]
    
    m1 = batman.TransitModel(params, t)
    trans_flux = m1.light_curve(params)                    #calculates transit

    return trans_flux


TM = Transit_Model(dur=sys_params["planet"]["T14"][0], T0=0,
             RpRs=sys_params["planet"]["rprs"][0], b=sys_params["planet"]["b"][0],per=P,
             sesinw=sesinw,secosw=secosw, occ=4000)

t = np.linspace(-0.25, 0.75*P, 3000)
flux,_     = TM.get_value(t)

batflux = batman_model([0,P,sys_params["planet"]["rprs"][0],aR,b,0,0],t)

plt.plot(t, 1e6*(flux-batflux))


