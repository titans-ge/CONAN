import sys
import numpy as np

def basecoeff(ibase,spline,init=None,lims=None, fit_offset="y"):
    # this function returns the input arrays for the baseline coefficients
    #   only those coefficients that are used are set to be jump parameters, 
    #   all others are set == 0 value, 0 stepsize, [0,0] Boundaries.
      
    # each of the arrays have 4 parameter: [initial guess, step, min, max]  -- perhaps 5 in the future if I use priors
    # nbc is the number of non-fixed baseline coefficients
    
    # set the baseline function
    #       offset 
    #       dcol0 coeff:   A0*dcol0 + B0*dcol0^2 + C0*dcol0^3 + D0*dcol0^4
    #       dcol3 coeff:   A3*dcol3 + B3*dcol3^2
    #       dcol4 coeff:   A4*dcol4 + B4*dcol4^2
    #       dcol5 coeff:   A5*dcol5 + B5*dcol5^2
    #       dcol6 coeff:   A6*dcol6 + B6*dcol6^2
    #       dcol7 coeff:   A7*dcol7 + B7*dcol7^2
    #       dcol8 coeff:   A8*dcol8 + B8*dcol8^2
    #       dsin  coeff:   Amp*sin(2pi(dcol0)/P + phi)
    #       CNM   coeff:   ACNM*?? + BCNM*??^2

    #[initial guess, step, min, max]
    nbc = 0

    offset = np.zeros((4,1), dtype=float)
    if ibase[8] > 0:                          # if we have a CNM
        offset[:,0]=[0.,0.0001,-2.,2.1]       # set the starting value and limits of the 0th-order start at 0
        nbc = nbc+1
    else:
        offset[:,0]=[init["off"],0.1*np.diff(lims["off"])[0],*lims["off"]]        # no CNM: set the starting value and limits of the offset    
        if spline.use or fit_offset=="n": offset[:,0]=[init["off"],0,0,0]   # if we use a spline, fix offset to 1 or bestvalue from get_decorr fit
        else: nbc = nbc+1
    
     # dcol0 coeff:   A0*dcol0 + B0*dcol0^2 + C0*dcol0^3 + D0*dcol0^4
    dcol0 = np.zeros((4,4), dtype=float)
    if ibase[0] > 0:  #A0
        dcol0[:,0]=[init["A0"],0.001,*lims["A0"]] 
        nbc = nbc+1
        
    if ibase[0] > 1:  #B0
        dcol0[:,1]=[init["B0"],0.001,*lims["B0"]]  
        nbc = nbc+1
         
    if ibase[0] > 2:  #C0
        dcol0[:,2]=[init["C0"],0.001,*lims["C0"]]  
        nbc = nbc+1       

    if ibase[0] > 3:   #D0
        dcol0[:,3]=[init["D0"],0.001,*lims["D0"]]  
        nbc = nbc+1      
        
    # dcol3 coeff:   A3*dcol3 + B3*dcol3^2
    dcol3=np.zeros((4,2), dtype=float)
    if ibase[1] > 0:  #A3
        dcol3[:,0]=[init["A3"],0.001,*lims["A3"]] 
        nbc = nbc+1
        
    if ibase[1] > 1: #B3
        dcol3[:,1]=[init["B3"],0.001,*lims["B3"]]  
        nbc = nbc+1

    # dcol4 coeff:   A4*dcol4 + B4*dcol4^2
    dcol4=np.zeros((4,2), dtype=float)
    if ibase[2] > 0:  #A4
        dcol4[:,0]=[init["A4"],0.001,*lims["A4"]] 
        nbc = nbc+1
        
    if ibase[2] > 1: #B4
        dcol4[:,1]=[init["B4"],0.001,*lims["B4"]] 
        nbc = nbc+1

    # dcol5 coeff:   A5*dcol5 + B5*dcol5^2
    dcol5=np.zeros((4,2), dtype=float)
    if ibase[3] > 0: #A5
        dcol5[:,0]=[init["A5"],0.001,*lims["A5"]]  # set the starting value and limits of the first-order B_in
        nbc = nbc+1
        
    if ibase[3] > 1: #B5
        dcol5[:,1]=[init["B5"],0.001,*lims["B5"]]  # set the starting value and limits of the second-order B_in
        nbc = nbc+1

    # dcol6 coeff:   A6*dcol6 + B6*dcol6^2
    dcol6=np.zeros((4,2), dtype=float)
    if ibase[4] > 0: #A6
        dcol6[:,0]=[init["A6"],0.001,*lims["A6"]]
        nbc = nbc+1

    if ibase[4] > 1: #B6
        dcol6[:,1]=[init["B6"],0.001,*lims["B6"]]
        nbc = nbc+1

    # dcol7 coeff:   A7*dcol7 + B7*dcol7^2
    dcol7=np.zeros((4,2), dtype=float)
    if ibase[5] > 0: #A7
        dcol7[:,0]=[init["A7"],0.001,*lims["A7"]]
        nbc = nbc+1

    if ibase[5] > 1: #B7
        dcol7[:,1]=[init["B7"],0.001,*lims["B7"]]
        nbc = nbc+1

    # dcol8 coeff:   A8*dcol8 + B8*dcol8^2
    dcol8=np.zeros((4,2), dtype=float)
    if ibase[6] > 0: #A8
        dcol8[:,0]=[init["A8"],0.001,*lims["A8"]]
        nbc = nbc+1

    if ibase[6] > 1: #B8
        dcol8[:,1]=[init["B8"],0.001,*lims["B8"]]
        nbc = nbc+1

    # dsin  coeff:   Amp*sin(2pi(dcol0-phi)/P) -x-> Amp*sin(freq*dcol0+phi)
    dsin=np.zeros((4,3), dtype=float)
    # if ibase[7]=="y": 
    #     dsin[:,0]=[init["amp"],0.001,0,1]
    #     dsin[:,1]=[init["freq"],0.001,1,333]
    #     dsin[:,2]=[init["phi"],0.001,0,1]
    #     nbc = nbc+3
                    
    # H coeff => CNM                
    dCNM=np.zeros((4,2), dtype=float)
    if ibase[8] > 0: #ACNM
        dCNM[:,0]=[init["ACNM"],0.001,0,1.e8] 
        nbc = nbc+1
        
    if ibase[8] > 1: #BCNM
        dCNM[:,1]=[init["BCNM"],0.0001,-1.e8,1.e8]     
        nbc = nbc+1
                    
    return offset, dcol0, dcol3, dcol4, dcol5, dcol6, dcol7, dcol8, dsin, dCNM, nbc
        

def basecoeffRV(ibaseRV,Pin,init=None,lims=None):

    nbcRV = 0

    # set the baseline function
    #       dcol0 coeff:   A0*dcol0 + B0*dcol0^2
    #       dcol3 coeff:   A3*dcol3 + B3*dcol3^2
    #       dcol4 coeff:   A4*dcol4 + B4*dcol4^2
    #       dcol5 coeff:   A5*dcol5 + B5*dcol5^2
    #       dsin  coeff:   Amp*sin(2pi(dcol0)/P + phi)

    # dcol0 coeff:   A0*dcol0 + B0*dcol0^2
    dcol0=np.zeros((4,2), dtype=float)
    if ibaseRV[0] > 0:  #A0
        dcol0[:,0]=[init["A0"],0.001,*lims["A0"] ]
        nbcRV = nbcRV+1
        
    if ibaseRV[0] > 1: #B3
        dcol0[:,1]=[init["B0"],0.001,*lims["B0"] ] 
        nbcRV = nbcRV+1

    # dcol3 coeff:   A3*dcol3 + B3*dcol3^2
    dcol3=np.zeros((4,2), dtype=float)
    if ibaseRV[1] > 0:  #A3
        dcol3[:,0]=[init["A3"],0.001,*lims["A3"] ]
        nbcRV = nbcRV+1
        
    if ibaseRV[1] > 1: #B3
        dcol3[:,1]=[init["B3"],0.001,*lims["B3"]]  
        nbcRV = nbcRV+1

    # dcol4 coeff:   A4*dcol4 + B4*dcol4^2
    dcol4=np.zeros((4,2), dtype=float)
    if ibaseRV[2] > 0:  #A4
        dcol4[:,0]=[init["A4"],0.001,*lims["A4"]]
        nbcRV = nbcRV+1

    if ibaseRV[2] > 1: #B4
        dcol4[:,1]=[init["B4"],0.001,*lims["B4"]]
        nbcRV = nbcRV+1

    # dcol5 coeff:   A5*dcol5 + B5*dcol5^2
    dcol5=np.zeros((4,2), dtype=float)
    if ibaseRV[3] > 0: #A5
        dcol5[:,0]=[init["A5"],0.001,*lims["A5"] ] # set the starting value and limits of the first-order B_in
        nbcRV = nbcRV+1

    if ibaseRV[3] > 1: #B5
        dcol5[:,1]=[init["B5"],0.001,*lims["B5"] ] # set the starting value and limits of the second-order B_in
        nbcRV = nbcRV+1

    # dsin  coeff:   Amp*sin(2pi(dcol0-phi)/P) -x-> Amp*sin(freq*dcol0+phi)
    dsin=np.zeros((4,4), dtype=float)
    if ibaseRV[4] > 0: 
        dsin[:,0]=[init["amp"],0.001,*lims["amp"]]
        dsin[:,1]=[init["freq"],0,*lims["freq"]]
        dsin[:,2]=[init["phi"],0.001,*lims["phi"]]
        dsin[:,3]=[init["phi2"],0.,*lims["phi2"]]
        nbcRV = nbcRV+4

    return dcol0, dcol3, dcol4, dcol5, dsin, nbcRV