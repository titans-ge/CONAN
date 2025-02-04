import pytest
import batman
# import ellc
import numpy as np
from CONAN.utils import (rho_to_aR, convert_LD, Tdur_to_aR, inclination,sesinw_secosw_to_ecc_omega,
                            rho_to_tdur, get_orbital_elements, get_Tconjunctions)
from CONAN.models import Transit_Model
import matplotlib.pyplot as plt

rho_star = 1.5
T0       = 1880
RpRs     = 0.13
b        = 0.5
per      = 3.5
sesinw   = 0.1
secosw   = 0.1
q1       = 0.3
q2       = 0.2
occ      = 700
Fn       = None
delta    = None
A_ev     = 0
A_db     = 0

# def ellc_transit(time, rho_star=None, dur=None, T0=None, RpRs=None, b=None, per=None, sesinw=0, 
#                     secosw=0, q1=0, q2=0, occ=0, Fn=None, delta=None, A_ev=0, A_db=0,npl=1):
#     import ellc
#     sb     = occ*1e-6/RpRs**2
#     u1,u2  = convert_LD(q1,q2,"q2u")
#     ecc, w = sesinw_secosw_to_ecc_omega(sesinw, secosw,angle_unit="degrees")

#     if rho_star != None:
#         aR   = rho_to_aR(rho_star,per,ecc,w)                     #semi-major axis (in units of stellar radii)
#     else:
#         aR   = Tdur_to_aR(dur, b, RpRs, per,ecc,w)                     #semi-major axis (in units of stellar radii)

#     ellc_flux = ellc.lc(time, t_zero=T0, period=per, radius_1=1/aR, radius_2=RpRs/aR, 
#                         incl=inclination(b, aR,ecc,w) , sbratio=sb,
#                         f_c=secosw, f_s=sesinw,ld_1="quad", ldc_1=[u1,u2], 
#                         )
#     return ellc_flux

def batman_transit(time, rho_star=None, dur=None, T0=None, RpRs=None, b=None, per=None, sesinw=0, 
                    secosw=0, q1=0, q2=0, occ=0, Fn=None, delta=None, A_ev=0, A_db=0,npl=1, get_mod=False):
    from batman import TransitParams, TransitModel

    params = TransitParams()             #object to store transit parameters
    #transit pars
    params.t0  = T0                      #time of inferior conjunction
    params.per = per                       #orbital period
    params.rp  = RpRs                      #planet radius (in units of stellar radii)
    params.ecc = sesinw**2+secosw**2
    params.w   = np.arctan2(sesinw,secosw)*180/np.pi                       #longitude of periastron (in degrees)
    params.w   = np.where(params.w<0,params.w+360,params.w)
    
    params.ecc, params.w = sesinw_secosw_to_ecc_omega(sesinw, secosw,angle_unit="degrees")

    if rho_star != None:
        params.a   = rho_to_aR(rho_star,params.per,params.ecc,params.w)                     #semi-major axis (in units of stellar radii)
    else:
        params.a   = Tdur_to_aR(dur, b, RpRs, params.per,params.ecc,params.w)                     #semi-major axis (in units of stellar radii)

    params.inc       = inclination(b, params.a,params.ecc,params.w)                      #orbital inclination (in degrees)
    params.limb_dark = "quadratic"        #limb darkening model
    u1,u2            = convert_LD(q1,q2,"q2u")
    params.u         = [u1,u2]

    m1 = batman.TransitModel(params, time)
    params.t_secondary = m1.get_t_secondary(params)
    params.fp = occ*1e-6
    m2 = batman.TransitModel(params,time,transittype="secondary" )

    trans_flux = m1.light_curve(params) 
    ecl_flux   = m2.light_curve(params) -(1+params.fp)
    bat_flux   = trans_flux+ecl_flux

    return (bat_flux, m1,m2, params) if get_mod else bat_flux

def test_batman_transit(show_plot=False):

    per      = 3.5
    npl      = 1

    equiv    = []
    time     = np.linspace(T0-0.25*per, T0+0.75*per, int(1.0*per*24*60/2)) #2min cadence
    niter    = 4
    ecc_dist = np.append(0,np.random.uniform(0.,0.1, niter))
    w_dist   = np.append(90,np.random.uniform(0,1,niter)*360)

    for e, w in zip(ecc_dist, w_dist):
        sesinw = np.sqrt(e)*np.sin(w*np.pi/180)
        secosw = np.sqrt(e)*np.cos(w*np.pi/180)

        bat_mod,m1,m2,params   = batman_transit(time, rho_star=rho_star, dur=None, T0=T0, RpRs=RpRs, b=b, per=per, 
                                    sesinw=sesinw,secosw=secosw,q1=q1, q2=q2, occ=occ, Fn=None, delta=delta, A_ev=A_ev, 
                                    A_db=A_db,npl=npl, get_mod=True)

        # ellc_mod  = ellc_transit(time, rho_star=rho_star, dur=None, T0=T0, RpRs=RpRs, b=b, per=per, 
        #                             sesinw=sesinw,secosw=secosw,q1=q1, q2=q2, occ=occ, Fn=None, delta=delta, A_ev=A_ev, 
        #                             A_db=A_db,npl=npl)

        conan_mod = Transit_Model(rho_star=rho_star, dur=None, T0=T0, RpRs=RpRs, b=b, per=per,
                                    sesinw=sesinw, secosw=secosw, q1=q1, q2=q2, occ=occ, Fn=None, 
                                    delta=delta, A_ev=A_ev, A_db=A_db,npl=npl).get_value(time)[0]
        
        tconj = get_Tconjunctions(time, T0, per, e, np.radians(w))

        equiv.append( conan_mod == pytest.approx(bat_mod,abs=5e-6) )
        print(f"{e=:.2f},{w=:5.1f}, agree: {equiv[-1]}")

        if show_plot:
            fig,ax = plt.subplots(2,1, figsize=(10,7), sharex=True)
            ax[0].set_title(f"{e=:.2f},{w=:.1f}, {equiv[-1]}")
            ax[0].plot(time, conan_mod, label="CONAN")
            ax[0].plot(time, bat_mod, "--", label="Batman")
            # ax[0].plot(time, ellc_mod,":", label="ELLC")
            [ax[0].axvline(tc, color="g", linestyle=":") for tc in (tconj.transit,tconj.eclipse)]
            ax[0].set_ylabel("Flux")
            ax[0].legend()

            ax[1].plot(time, 1e6*(conan_mod - bat_mod), label="CONAN - Batman")
            # ax[1].plot(time, 1e6*(conan_mod - ellc_mod), label="CONAN - ELLC")
            ax[1].set_ylabel("diff (ppm)")
            ax[1].legend()
            
            if e==ecc_dist[-1]: plt.show()

    assert all(equiv)

def test_duration(show_plot=False):
    """ 
    Test duration calculation by comparing analytical and numerical results.
    show plot shows diagnostics comparing true anomaly and Rsky to those from batman and also,
    checking the accuracy of eccentric anomaly calculation. 
    """
    npl      = 1
    equiv    = []
    niter    = 4
    ecc_dist = [0,0.2,0.4,0.6,0.7]#np.append(0,np.random.uniform(0.,0.8, niter))
    w_dist   = [145,145,145,145,145]#np.append(90,np.random.uniform(0,1,niter)*360)

    for e, w in zip(ecc_dist, w_dist):
        sesinw = np.sqrt(e)*np.sin(w*np.pi/180)
        secosw = np.sqrt(e)*np.cos(w*np.pi/180)

        analytic_tdur = rho_to_tdur(rho_star,b,RpRs,per,e,w)            #analytical duration
        time = np.linspace(T0-0.7*analytic_tdur, T0+0.7*analytic_tdur, int(1.4*analytic_tdur*24*60*2))

        conan_mod = Transit_Model(rho_star=rho_star, dur=None, T0=T0, RpRs=RpRs, b=b, per=per,
                                sesinw=sesinw, secosw=secosw, q1=q1, q2=q2, occ=occ, Fn=None, 
                                delta=delta, A_ev=A_ev, A_db=A_db,npl=npl).get_value(time)[0]
        conan_mod2 = Transit_Model(rho_star=rho_star, dur=None, T0=T0, RpRs=RpRs, b=b, per=per,
                                sesinw=sesinw, secosw=secosw, q1=q1, q2=q2, occ=occ, Fn=None, 
                                delta=delta, A_ev=A_ev, A_db=A_db,npl=npl).get_value(time, approx_EA=True)[0]
        bat_mod,m1,m2,params   = batman_transit(time, rho_star=rho_star, dur=None, T0=T0, RpRs=RpRs, b=b, per=per, 
                                    sesinw=sesinw,secosw=secosw,q1=q1, q2=q2, occ=occ, Fn=None, delta=delta, A_ev=A_ev, 
                                    A_db=A_db,npl=npl, get_mod=True)
        # ellc_mod  = ellc_transit(time, rho_star=rho_star, dur=None, T0=T0, RpRs=RpRs, b=b, per=per, 
        #                             sesinw=sesinw,secosw=secosw,q1=q1, q2=q2, occ=occ, Fn=None, delta=delta, A_ev=A_ev, 
        #                             A_db=A_db,npl=npl)

        numeric_tdur  = np.ptp(time[conan_mod <1])+2*np.diff(time)[0]   #numerical duration
        num_t1,num_t4 = time[conan_mod <1][0]-np.diff(time)[0], time[conan_mod <1][-1]+np.diff(time)[0]

        equiv.append( numeric_tdur == pytest.approx(analytic_tdur,rel=1e-2, abs=0.002) )  #equal within 3 minutes or 1%
        print(f"{e=:.2f},{w=:.1f}, [{numeric_tdur},{analytic_tdur}],{equiv[-1]}")

        if show_plot:
            fig,ax  = plt.subplots(4,1, figsize=(10,7),sharex=True)
            ax[0].set_title(f"{e=:.2f},{w=:.1f},[{numeric_tdur:.4f},{analytic_tdur:.4f}], {equiv[-1]}")
            ax[0].plot(time, conan_mod, label="CONAN")
            ax[0].plot(time, bat_mod,"--", label="Batman")
            ax[0].plot(time, conan_mod2,":",label="CONAN_approx_EA")
            # ax[0].plot(time, ellc_mod,":", label="ELLC")
            ax[0].set_ylabel("Flux")
            # ax[0].set_xlim([num_t1-0.2*numeric_tdur, num_t4+0.2*numeric_tdur])

            orb_pars  = get_orbital_elements(time,T0,per,e,np.radians(w))
            orb_pars2 = get_orbital_elements(time,T0,per,e,np.radians(w),approx=True)
            
            ax[1].plot(time, orb_pars.ecc_anom - orb_pars._ecc*np.sin(orb_pars.ecc_anom) - orb_pars.mean_anom, label="CONAN")
            ax[1].plot(time, orb_pars2.ecc_anom - orb_pars2._ecc*np.sin(orb_pars2.ecc_anom) - orb_pars2.mean_anom, "g:",label="CONAN_approx_EA")
            ax[1].set_ylabel("(E-esinE) - M") 
            
            
            ax[2].plot(time, orb_pars.true_anom, label="CONAN")
            ax[2].plot(time,m1.get_true_anomaly(), "--",label="Batman")
            ax[2].plot(time, orb_pars2.true_anom, "g:",label="CONAN_approx_EA")
            ax[2].set_ylabel("True anomaly")

            # from pycheops.funcs import t2z
            ax[3].plot(time, orb_pars.get_Rsky(params.a,np.radians(params.inc))[0], label="CONAN")
            ax[3].plot(time,m1.ds, "--",label="Batman")
            ax[3].plot(time, orb_pars2.get_Rsky(params.a,np.radians(params.inc))[0], "g:",label="CONAN_approx_EA")
            ax[3].set_ylabel("Rsky")

            for i in [0,1,2,3]:
                ax[i].axvline(T0, color="g", linestyle=":")
                [ax[i].axvline(tt, color="k", linestyle="--") for tt in [T0+analytic_tdur/2, T0-analytic_tdur/2]]
                [ax[i].axvline(tt, color="r", linestyle=":") for tt in [num_t1, num_t4]]
                ax[i].legend()
            plt.subplots_adjust(hspace=0.02)

            if e==ecc_dist[-1]: plt.show()

    assert all(equiv)

def test_orbital_elements(show_plot=False):
    """ 
    Test calculation of orbital elements by comparing the true anom, transit time and eclipse time 
    to results from batman
    """
    npl      = 1
    per      = 38.5

    TA_equiv    = []
    Ttra_equiv  = []
    Tecl_equiv  = []

    time     = np.linspace(T0-0.25*per, T0+.75*per, int(1.5*per*24*60*2))
    niter    = 4
    ecc_dist = np.append(0,np.random.uniform(0.,0.8, niter))
    w_dist   = np.append(90,np.random.uniform(0,1,niter)*360)

    for e, w in zip(ecc_dist, w_dist):
        sesinw = np.sqrt(e)*np.sin(w*np.pi/180)
        secosw = np.sqrt(e)*np.cos(w*np.pi/180)

        conan_mod = Transit_Model(rho_star=rho_star, dur=None, T0=T0, RpRs=RpRs, b=b, per=per,
                                sesinw=sesinw, secosw=secosw, q1=q1, q2=q2, occ=occ, Fn=None, 
                                delta=delta, A_ev=A_ev, A_db=A_db,npl=npl).get_value(time)[0]
        bat_mod,m1,m2,params   = batman_transit(time, rho_star=rho_star, dur=None, T0=T0, RpRs=RpRs, b=b, per=per, 
                                    sesinw=sesinw,secosw=secosw,q1=q1, q2=q2, occ=occ, Fn=None, delta=delta, A_ev=A_ev, 
                                    A_db=A_db,npl=npl, get_mod=True)
        #test true anomaly
        orb_pars        = get_orbital_elements(time,T0,per,e,np.radians(w))
        conan_true_anom = np.mod(orb_pars.true_anom,2*np.pi)
        bat_true_anom   = np.mod(m1.get_true_anomaly(),2*np.pi)

        TA_equiv.append( conan_true_anom == pytest.approx(bat_true_anom,abs=1e-3) )

        #test time of conjunctions
        tconj = get_Tconjunctions(time, T0, per, e, np.radians(w))
        bat_Ttra = m1.get_t_conjunction(params)
        bat_Tecl = m1.get_t_secondary(params)

        Ttra_equiv.append( np.mod(tconj.transit,per) == pytest.approx(np.mod(bat_Ttra,per),abs=0.0014) ) #equal within 2mins
        Tecl_equiv.append( np.mod(tconj.eclipse,per) == pytest.approx(np.mod(bat_Tecl,per),abs=0.0014) ) #equal within 2mins

        print(f"{e=:.2f},{w=:5.1f}, agree: {all([TA_equiv[-1], Ttra_equiv[-1], Tecl_equiv[-1]])}")

        if show_plot:
            fig,ax  = plt.subplots(3,1, figsize=(10,7),sharex=True)
            ax[0].set_title(f"{e=:.2f},{w=:.1f}, {TA_equiv[-1]=}")
            ax[0].plot(time, conan_mod, label="CONAN")
            ax[0].plot(time, bat_mod,"--", label="Batman")
            [ax[0].axvline(tc, color="k", linestyle="--") for tc in (tconj.transit,tconj.eclipse)]
            [ax[0].axvline(tc, color="r", linestyle=":") for tc in (m1.get_t_conjunction(params),m1.get_t_secondary(params))]

            ax[0].set_ylabel("Flux")

            ax[1].plot(time, conan_true_anom, label="CONAN")
            ax[1].plot(time, bat_true_anom, "--",label="Batman")
            ax[1].set_ylabel("True anomaly")

            ax[2].plot(time, conan_true_anom - bat_true_anom)
            ax[2].set_ylabel("diff")

            for i in [0,1,2]:
                ax[i].axvline(T0, color="g", linestyle=":")
                ax[i].legend()

            if e==ecc_dist[-1]: plt.show()
    assert all(np.concatenate((TA_equiv, Ttra_equiv, Tecl_equiv)))

def test_LTT(show_plot=False):
    #test light travel time
    from CONAN.utils import light_travel_time_correction
    from CONAN.get_files import get_parameters
    
    
    # sys_params = get_parameters("WASP-121 b")


    P    = 1.27492504
    t0   = 0
    t14  = 0.12105416666666667
    b    = 0.1
    RpRs = 0.12355
    e    = 0.
    w    = np.radians(90)
    aR   = Tdur_to_aR(t14,b,RpRs,P,e,np.degrees(w))
    inc   = inclination(b,aR,e,np.degrees(w))

    sesinw, secosw = np.sqrt(e)*np.sin(w), np.sqrt(e)*np.cos(w)

    t = np.linspace(-0.25, 0.75*P, 3000)

    tcorr = light_travel_time_correction(t,t0,aR,P,np.radians(inc),1.46,e,w)
    tconj  = get_Tconjunctions(t,t0,P,e,w,1.46,
                                aR,np.radians(inc))
    
    if show_plot:
        plt.figure()
        plt.plot(t/P, 24*3600*(t-tcorr))
        plt.axvline(tconj.transit/P,c="k",ls=":",label="mid-transit")
        plt.axvline(tconj.eclipse/P,c="r",ls="--",label="mid-eclipse")
        plt.axvline(t[np.argmax(t-tcorr)]/P,c="g",ls=":",label="max delay")
        plt.xlabel("Phase")
        plt.ylabel("LTT delay [s]")
        plt.legend()
        plt.show()


    TM = Transit_Model(dur=t14, T0=0, RpRs=RpRs, b=b,per=P,
                        sesinw=sesinw,secosw=secosw, occ=4000)

    t = np.linspace(P/2-0.07,P/2+0.07,1500)
    flux,_     = TM.get_value(t)
    flux_ltt,_ = TM.get_value(t, Rstar=1.46)

    tconj_noLTT  = get_Tconjunctions(t,t0,P,e,w)
    dt =24*3600*(tconj.eclipse-tconj_noLTT.eclipse)

    if show_plot:
        fig,ax =plt.subplots(2,1,figsize=(15,3),sharex=True,  gridspec_kw={"height_ratios":(2,1)})
        ax[0].plot(t,flux,"b",label="no LTT")
        ax[0].plot(t,flux_ltt,"r--",label=f"LTT delay ({dt:.1f}secs) included")
        ax[0].legend()
        ax[0].set_title("LTT allows to move the occultation to later time to account for the delay")

        ax[1].plot(t, 1e6*(flux_ltt-flux))
        ax[1].set_ylabel("res [ppm]")
        ax[0].axvline(tconj_noLTT.eclipse,c="b",ls=":")
        ax[0].axvline(tconj.eclipse,c="r",ls=":")
        ax[0].set_xlim([P/2-0.07,P/2+0.07])

        plt.subplots_adjust(hspace=0)
        plt.show()

    assert dt == pytest.approx(25.80265178,abs=1e-2)

def test_ecc_occ_shift(show_plot=False):
    # Test occultation shift due to eccentricity

    P    = 1.27492504
    t0   = 0
    t14  = 0.12105416666666667
    b    = 0.1
    RpRs = 0.12355
    e    = 0.01
    w    = np.radians(45)
    equiv = []

    ecc_dist = [0., 0.01, 0.02, 0.05, 0.09 ]
    w_dist   = [ 90., 277,  180,   1.5, 102]
    exp_dt   = [0.0, 1.4244, -23.3737, 58.3939, -21.9539]   # in mins
    dt       = []

    for e, w in zip(ecc_dist, w_dist):
        sesinw = np.sqrt(e)*np.sin(w*np.pi/180)
        secosw = np.sqrt(e)*np.cos(w*np.pi/180)
        w_rad  = np.radians(w)


        TM     = Transit_Model(dur=t14, T0=0, RpRs=RpRs, b=b,per=P,
                                sesinw=0,secosw=0, occ=4000,Fn=50,delta=0)

        TM_ecc = Transit_Model(dur=t14, T0=0, RpRs=RpRs, b=b,per=P,
                                sesinw=sesinw,secosw=secosw, occ=4000,Fn=50,delta=0)

        t          = np.linspace(-0.25*P,0.75*P,5000)
        flux,_     = TM.get_value(t, model_phasevar=True)
        flux_ecc,_ = TM_ecc.get_value(t, model_phasevar=True)

        tconj      = get_Tconjunctions(t,t0,P,ecc=0,omega=np.pi/2)
        tconj_ecc  = get_Tconjunctions(t,t0,P,e,w_rad)
        dt.append(24*60*(tconj_ecc.eclipse-tconj.eclipse))

        if show_plot:
            fig,ax =plt.subplots(2,1,figsize=(15,4),sharex=True,  gridspec_kw={"height_ratios":(2,1)})
            ax[0].plot(t,flux,"b",label="no ecc")
            ax[0].plot(t,flux_ecc,"r--",label="ecc")
            ax[0].legend()
            ax[0].set_title(f"ecc={e},w={w:.1f} shifts the eclipse by {dt[-1]:.1f}mins")

            ax[1].plot(t, 1e6*(flux_ecc-flux))
            ax[1].set_ylabel("res [ppm]")

            _=[ax[0].axvline(tt,c="b",ls=":") for tt in [tconj.eclipse]]
            _=[ax[0].axvline(tt,c="r",ls=":") for tt in [tconj_ecc.eclipse]]
            plt.subplots_adjust(hspace=0)            
            if e==ecc_dist[-1]: plt.show()

    assert exp_dt == pytest.approx(dt,abs=1e-2)

def test_flat_transit():
    per      = 3.5
    npl      = 1

    equiv    = []
    time     = np.linspace(T0-0.25*per, T0+0.25*per, int(0.5*per*24*60/2)) #2min cadence

    #Non-transiting
    conan_mod = Transit_Model(rho_star=None, dur=0.112, T0=T0, RpRs=RpRs, b=1.5+RpRs, per=per,
                            sesinw=sesinw, secosw=secosw, q1=q1, q2=q2, occ=occ, Fn=None, 
                            delta=delta, A_ev=A_ev, A_db=A_db,npl=npl).get_value(time)[0]
    equiv.append(pytest.approx(conan_mod,abs=1e-6) == 1)

    #zero radius
    conan_mod = Transit_Model(rho_star=rho_star, dur=None, T0=T0, RpRs=0, b=b, per=per,
                            sesinw=sesinw, secosw=secosw, q1=q1, q2=q2, occ=occ, Fn=None, 
                            delta=delta, A_ev=A_ev, A_db=A_db,npl=npl).get_value(time)[0]
    equiv.append(pytest.approx(conan_mod,abs=1e-6) == 1)

    assert all(equiv)



if __name__ == "__main__":
    test_batman_transit(False)
    test_duration(False)
    test_orbital_elements(False)
    test_LTT(False)
    test_ecc_occ_shift(False)
    test_flat_transit()
