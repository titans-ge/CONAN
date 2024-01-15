import numpy as np

def ecc_om_par(ecc, omega):
    # This function calculates the prior values and limits for the eccentricity and omega parameters

    sesino=np.sqrt(ecc.start_value)*np.sin(omega.start_value)     # starting value
    sesinolo = -1.   # lower limit
    sesinoup = 1.   # upper limit
    
    dump1=np.sqrt(ecc.start_value+ecc.step_size)*np.sin(omega.start_value+omega.step_size)-np.sqrt(ecc.start_value)*np.sin(omega.start_value)
    dump2=np.sqrt(ecc.start_value-ecc.step_size)*np.sin(omega.start_value+omega.step_size)-np.sqrt(ecc.start_value)*np.sin(omega.start_value)
    dump3=np.sqrt(ecc.start_value+ecc.step_size)*np.sin(omega.start_value-omega.step_size)-np.sqrt(ecc.start_value)*np.sin(omega.start_value)
    dump4=np.sqrt(ecc.start_value-ecc.step_size)*np.sin(omega.start_value-omega.step_size)-np.sqrt(ecc.start_value)*np.sin(omega.start_value)

    sesinostep=np.nanmax(np.abs([dump1,dump2,dump3,dump4])) # the stepsize

    if (ecc.prior_width_lo!=0.):   # if an eccentricity prior is set
        edump= np.copy(ecc.prior_mean)
        eup  = np.copy(ecc.prior_width_lo)
        elo  = np.copy(ecc.prior_width_hi)
    else:
        edump= np.copy(ecc.start_value)
        eup=0.
        elo=0.

    if (omega.prior_width_lo!=0.):   # if an omega prior is set
        odump= np.copy(omega.prior_mean)
        oup  = np.copy(omega.prior_width_lo)
        olo  = np.copy(omega.prior_width_hi)
    else:
        odump= np.copy(omega.start_value)
        oup=0.
        olo=0.

    sesinop=np.sqrt(edump)*np.sin(odump)     # the prior value

    dump1=np.sqrt(edump+eup)*np.sin(odump+oup)-np.sqrt(edump)*np.sin(odump)
    dump2=np.sqrt(edump-elo)*np.sin(odump+oup)-np.sqrt(edump)*np.sin(odump)
    dump3=np.sqrt(edump+eup)*np.sin(odump-olo)-np.sqrt(edump)*np.sin(odump)
    dump4=np.sqrt(edump-elo)*np.sin(odump-olo)-np.sqrt(edump)*np.sin(odump)

    sesinoplo=np.abs(np.nanmin([dump1,dump2,dump3,dump4]))
    sesinopup=np.abs(np.nanmax([dump1,dump2,dump3,dump4]))
                                    
    secoso=np.sqrt(ecc.start_value)*np.cos(omega.start_value)
    secosolo=-1.   # lower limit
    secosoup=1.   # upper limit

    dump1=np.sqrt(ecc.start_value+ecc.step_size)*np.cos(omega.start_value+omega.step_size)-np.sqrt(ecc.start_value)*np.cos(omega.start_value)
    dump2=np.sqrt(ecc.start_value-ecc.step_size)*np.cos(omega.start_value+omega.step_size)-np.sqrt(ecc.start_value)*np.cos(omega.start_value)
    dump3=np.sqrt(ecc.start_value+ecc.step_size)*np.cos(omega.start_value-omega.step_size)-np.sqrt(ecc.start_value)*np.cos(omega.start_value)
    dump4=np.sqrt(ecc.start_value-ecc.step_size)*np.cos(omega.start_value-omega.step_size)-np.sqrt(ecc.start_value)*np.cos(omega.start_value)

    secosostep=np.nanmax(np.abs([dump1,dump2,dump3,dump4]))

    dump1=np.sqrt(edump+eup)*np.cos(odump+oup)-np.sqrt(edump)*np.cos(odump)
    dump2=np.sqrt(edump-elo)*np.cos(odump+oup)-np.sqrt(edump)*np.cos(odump)
    dump3=np.sqrt(edump+eup)*np.cos(odump-olo)-np.sqrt(edump)*np.cos(odump)
    dump4=np.sqrt(edump-elo)*np.cos(odump-olo)-np.sqrt(edump)*np.cos(odump)
                        
    secosoplo=np.abs(np.nanmin([dump1,dump2,dump3,dump4]))
    secosopup=np.abs(np.nanmax([dump1,dump2,dump3,dump4]))

    secosop=np.sqrt(edump)*np.cos(odump)     # the prior

    to_fit = "y" if ecc.to_fit=="y" or omega.to_fit=="y" else "n"
    pri    =  ecc.prior
    eos_in=[to_fit,sesino,sesinostep,pri,sesinop,sesinoplo,sesinopup,sesinolo,sesinoup]
    eoc_in=[to_fit,secoso,secosostep,pri,secosop,secosoplo,secosopup,secosolo,secosoup]

    from ._classes import _param_obj
    eos_in = _param_obj(*eos_in)
    eoc_in = _param_obj(*eoc_in)

    return eos_in, eoc_in
