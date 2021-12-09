import numpy as np

def ecc_om_par(ecc_in, omega_in):

    sesino=np.sqrt(ecc_in[0])*np.sin(omega_in[0])     # starting value
    sesinolo=0.   # lower limit
    sesinoup=1.   # upper limit
    
    dump1=np.sqrt(ecc_in[0]+ecc_in[1])*np.sin(omega_in[0]+omega_in[1])-np.sqrt(ecc_in[0])*np.sin(omega_in[0])
    dump2=np.sqrt(ecc_in[0]-ecc_in[1])*np.sin(omega_in[0]+omega_in[1])-np.sqrt(ecc_in[0])*np.sin(omega_in[0])
    dump3=np.sqrt(ecc_in[0]+ecc_in[1])*np.sin(omega_in[0]-omega_in[1])-np.sqrt(ecc_in[0])*np.sin(omega_in[0])
    dump4=np.sqrt(ecc_in[0]-ecc_in[1])*np.sin(omega_in[0]-omega_in[1])-np.sqrt(ecc_in[0])*np.sin(omega_in[0])

    sesinostep=np.nanmax(np.abs([dump1,dump2,dump3,dump4])) # the stepsize

    if (ecc_in[5]!=0.):   # if an eccentricity prior is set
        edump=np.copy(ecc_in[4])
        eup=np.copy(ecc_in[5])
        elo=np.copy(ecc_in[6])
    else:
        edump=np.copy(ecc_in[0])
        eup=0.
        elo=0.

    if (omega_in[5]!=0.):   # if an eccentricity prior is set
        odump=np.copy(omega_in[4])
        oup=np.copy(omega_in[5])
        olo=np.copy(omega_in[6])
    else:
        odump=np.copy(omega_in[0])
        oup=0.
        olo=0.

    sesinop=np.sqrt(edump)*np.sin(odump)     # the prior value

    dump1=np.sqrt(edump+eup)*np.sin(odump+oup)-np.sqrt(edump)*np.sin(odump)
    dump2=np.sqrt(edump-elo)*np.sin(odump+oup)-np.sqrt(edump)*np.sin(odump)
    dump3=np.sqrt(edump+eup)*np.sin(odump-olo)-np.sqrt(edump)*np.sin(odump)
    dump4=np.sqrt(edump-elo)*np.sin(odump-olo)-np.sqrt(edump)*np.sin(odump)

    sesinoplo=np.abs(np.nanmin([dump1,dump2,dump3,dump4]))
    sesinopup=np.abs(np.nanmax([dump1,dump2,dump3,dump4]))
                                    
    secoso=np.sqrt(ecc_in[0])*np.cos(omega_in[0])
    secosolo=0.   # lower limit
    secosoup=1.   # upper limit

    dump1=np.sqrt(ecc_in[0]+ecc_in[1])*np.cos(omega_in[0]+omega_in[1])-np.sqrt(ecc_in[0])*np.cos(omega_in[0])
    dump2=np.sqrt(ecc_in[0]-ecc_in[1])*np.cos(omega_in[0]+omega_in[1])-np.sqrt(ecc_in[0])*np.cos(omega_in[0])
    dump3=np.sqrt(ecc_in[0]+ecc_in[1])*np.cos(omega_in[0]-omega_in[1])-np.sqrt(ecc_in[0])*np.cos(omega_in[0])
    dump4=np.sqrt(ecc_in[0]-ecc_in[1])*np.cos(omega_in[0]-omega_in[1])-np.sqrt(ecc_in[0])*np.cos(omega_in[0])

    secosostep=np.nanmax(np.abs([dump1,dump2,dump3,dump4]))

    dump1=np.sqrt(edump+eup)*np.cos(odump+oup)-np.sqrt(edump)*np.cos(odump)
    dump2=np.sqrt(edump-elo)*np.cos(odump+oup)-np.sqrt(edump)*np.cos(odump)
    dump3=np.sqrt(edump+eup)*np.cos(odump-olo)-np.sqrt(edump)*np.cos(odump)
    dump4=np.sqrt(edump-elo)*np.cos(odump-olo)-np.sqrt(edump)*np.cos(odump)
                        
    secosoplo=np.abs(np.nanmin([dump1,dump2,dump3,dump4]))
    secosopup=np.abs(np.nanmax([dump1,dump2,dump3,dump4]))

    secosop=np.sqrt(edump)*np.cos(odump)     # the prior

    eos_in=[sesino,sesinostep,sesinolo,sesinoup,sesinop,sesinoplo,sesinopup]
    eoc_in=[secoso,secosostep,secosolo,secosoup,secosop,secosoplo,secosopup]

    return eos_in, eoc_in
