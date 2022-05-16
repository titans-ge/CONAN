import sys
import numpy as np

def basecoeff(ibase):
    # this function returns the input arrays for the baseline coefficients
    #   only those coefficients that are used are set to be jump parameters, 
    #   all others are set == 0 value, 0 stepsize, [0,0] Boundaries.
    # NOTE that these are a lot of jump parameters if many LCs are used!
    #      perhaps I will need to implement leastsquare minimization at each step.
    #    FOR NOW, let's assume we're simple and analyze few LCs with simple baselines    
    # each of the _in arrays have 4 coefficients, for each parameter: [initial guess, step, min, max]  -- perhaps 5 in the future if I use priors
    #
    # nbc is the number of non-fixed baseline coefficients
    
    nbc = np.copy(0)
    
    # set the baseline function
    #       A coeff => time:  A[0] + A[1]*t + A[2]*t^2 + A[3]*t^3 + A[4]*t^4
    #       B coeff => AM:    B[0]*AM + B[1]*AM^2    TODDO: this should not be this simple
    #       C1 coeff => lam:  C1[0]*lam + C1[1]*lam^2 
    #       C2 coeff => y:    C2[0]*y + C1[1]*]y^2 
    #       D coeff => fwhm:  D[0]*fwhm + D[1]*fwhm^2
    #       E coeff => sky:   E[0]*sky + E[1]*sky^2
    #       G coeff => sin:   G[0]*np.sin(G[1]*ts+G[2])
    #       H coeff => CNM:   H[0]*CNM + H[1]*CNM^2
   
    # A coeff => time:  A[0] + A[1]*t + A[2]*t^2 + A[3]*t^3 + A[4]*t^4
    
    A_in=np.zeros((4,5), dtype=np.float)
    
    if ibase[6] > 0:                          # if we have a CNM
        A_in[:,0]=[0.00,0.0001,-2.,2.1]       # set the starting value and limits of the 0th-order start at 0
    else:
        A_in[:,0]=[1.,0.0001,0.,2.1]        # no CNM: set the starting value and limits of the 0th-order start at 1        
    
    nbc = nbc+1
    
    if ibase[0] > 0:
        A_in[:,1]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the first-order A_in
        nbc = nbc+1
        
    if ibase[0] > 1:
        A_in[:,2]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the second-order A_in
        nbc = nbc+1
         
    if ibase[0] > 2:
        A_in[:,3]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the second-order A_in
        nbc = nbc+1       

    if ibase[0] > 3:
        A_in[:,4]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the second-order A_in
        nbc = nbc+1      
        
    # B coeff => AM:    B[0]*AM + B[1]*AM^2  
    B_in=np.zeros((4,2), dtype=np.float)
    if ibase[1] > 0:
        B_in[:,0]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the first-order B_in
        nbc = nbc+1
        
    if ibase[1] > 1:
        B_in[:,1]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the second-order B_in
        nbc = nbc+1
     
    #  C1 coeff => lam:    C1[0]*lam + C1[1]*lam^2 
    C1_in=np.zeros((4,2), dtype=np.float)
    if ibase[2] > 0:
        C1_in[:,0]=[0.,0.0001,-1.e7,1.e7]  # set the starting value and limits of the first-order C1_in
        nbc = nbc+1
        
    if ibase[2] > 1:
        C1_in[:,1]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the second-order C1_in
        nbc = nbc+1
             
    #  C2 coeff => lam:    C2[0]*y + C2[1]*y^2 
    C2_in=np.zeros((4,2), dtype=np.float)
    if ibase[3] > 0:
        C2_in[:,0]=[0.,0.0001,-1.e7,1.e7]  # set the starting value and limits of the first-order C1_in
        nbc = nbc+1
        
    if ibase[3] > 1:
        C2_in[:,1]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the second-order C1_in
        nbc = nbc+1
    
    # D coeff => fwhm:  D[0]*fwhm + D[1]*fwhm^2
    D_in=np.zeros((4,2), dtype=np.float)
    if ibase[4] > 0:
        D_in[:,0]=[0.,0.1,-1.e7,1.e7]  # set the starting value and limits of the first-order D_in
        nbc = nbc+1
        
    if ibase[4] > 1:
        D_in[:,1]=[0.,0.01,-1.e7,1.e7]  # set the starting value and limits of the second-order D_in
        nbc = nbc+1
        
    # E coeff => sky:   E[0]*sky + E[1]*sky^2    
    E_in=np.zeros((4,2), dtype=np.float)
    if ibase[5] > 0:
        E_in[:,0]=[0.,0.00001,-1.e8,1.e8]    # set the starting value and limits of the first-order E_in
        nbc = nbc+1
        
    if ibase[5] > 1:
        E_in[:,1]=[0.,0.0000001,-1.e8,1.e8]  # set the starting value and limits of the second-order E_in
        nbc = nbc+1
    # E coeff => sky:   E[0]*sky + E[1]*sky^2    

    #       G coeff => sin:   G[0]*np.sin(G[1]*ts+G[2])
    G_in=np.zeros((4,3), dtype=np.float)
    if ibase[6] > 0:
        G_in[:,0]=[0.0001,0.0001,0,1]  # set the starting value and limits of the sinus amplitude
        G_in[:,1]=[50.,0.1,2.5,333]  # set the starting value and limits of the sinus frequency (between 30 min and 10h)
        G_in[:,2]=[np.pi,np.pi/40.,0,2.*np.pi]  # set the starting value and limits of the sinus offset (between 0 min and 2pi)
        nbc = nbc+3
                    
    # H coeff => CNM                
    H_in=np.zeros((4,2), dtype=np.float)
    if ibase[7] > 0:
        H_in[:,0]=[1.,0.001,0,1.e8]  # set the starting value and limits of the first-order H_in
        nbc = nbc+1
        
    if ibase[7] > 1:
        H_in[:,1]=[0.,0.0001,-1.e8,1.e8]  # set the starting value and limits of the second-order H_in       
        nbc = nbc+1
                    
    return A_in, B_in, C1_in, C2_in, D_in, E_in, G_in, H_in, nbc
        

def basecoeffRV(ibaseRV,Pin):
    # this function returns the input arrays for the baseline coefficients
    #   only those coefficients that are used are set to be jump parameters, 
    #   all others are set == 0 value, 0 stepsize, [0,0] Boundaries.
    # NOTE that these are a lot of jump parameters if many LCs are used!
    #      perhaps I will need to implement leastsquare minimization at each step.
    #    FOR NOW, let's assume we're simple and analyze few LCs with simple baselines    
    # each of the _in arrays have 4 coefficients, for each parameter: [initial guess, step, min, max]  -- perhaps 5 in the future if I use priors
    #
    # nbcRV is the number of non-fixed baseline coefficients
    
    nbcRV = np.copy(0)
    
    # set the baseline function
    #       W coeff => time:      W[0]*t + W[1]*t^2  NOTE: 0th order is GAMMA
    #       V coeff => bisector:  V[0]*bis + B[1]*bis^2 
    #       U coeff => fwhm:      U[0]*fwhm + U[1]*fwhm^2 
    #       S coeff => contrast:  S[0]*cont + S[1]*cont^2
    #       P coeff => sinus:     P[0]*np.sin(P[1]*ts+P[2])
   
    # W coeff => time:  W[0]*t + W[1]*t^2 
    
    W_in=np.zeros((4,2), dtype=np.float)

    if ibaseRV[0] > 0:
        W_in[:,0]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the first-order W_in
        nbcRV = nbcRV+1
        
    if ibaseRV[0] > 1:
        W_in[:,1]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the second-order W_in
        nbcRV = nbcRV+1
    
    # V coeff => bisector : V[0]*bis + B[1]*bis^2 
    V_in=np.zeros((4,2), dtype=np.float)
    
    if ibaseRV[1] > 0:
        V_in[:,0]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the first-order V_in
        nbcRV = nbcRV+1
        
    if ibaseRV[1] > 1:
        V_in[:,1]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the second-order V_in
        nbcRV = nbcRV+1
     
    #  U coeff => fwhm:     U[0]*fwhm + U[1]*fwhm^2 
    U_in=np.zeros((4,2), dtype=np.float)
    
    if ibaseRV[2] > 0:
        U_in[:,0]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the first-order U_in
        nbcRV = nbcRV+1
        
    if ibaseRV[2] > 1:
        U_in[:,1]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the second-order U_in 
        nbcRV = nbcRV+1
    
    # S coeff => contrast:  S[0]*cont + S[1]*cont^2
    S_in=np.zeros((4,2), dtype=np.float)
    if ibaseRV[3] > 0:
        S_in[:,0]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the first-order S_in
        nbcRV = nbcRV+1
        
    if ibaseRV[3] > 1:
        S_in[:,1]=[0.,0.001,-1.e7,1.e7]  # set the starting value and limits of the second-order S_in
        nbcRV = nbcRV+1

    # P coeff => sin:   P[0]*np.sin(P[1]*ts+P[2])
    P_in=np.zeros((4,4), dtype=np.float)
    if ibaseRV[4] > 0:
        P_in[:,0]=[0.01,0.001,0,1]  # set the starting value and limits of the sinus amplitude
        P_in[:,1]=[Pin,0.,0.1,100]  # set the starting value and limits of the sinus frequency
        P_in[:,2]=[np.pi,np.pi/4.,0,2.*np.pi]  # set the starting value and limits of the sinus offset (between 0 min and 2pi)
        P_in[:,3]=[0.0,0.,0,1]  # set the starting value and limits of the cosine amplitude
        nbcRV = nbcRV+4

                   
    return W_in, V_in, U_in, S_in, P_in, nbcRV
        

