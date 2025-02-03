import numpy as np
from .plotting import *
from .utils import rho_to_tdur, rho_to_aR, convert_LD,aR_to_Tdur, tdur_to_rho
from .funcs import credregionML


def mcmc_outputs(posterior, jnames, ijnames, njnames, nijnames, bp, uwl, Rs_in, Ms_in, Rs_PDF, Ms_PDF, 
                    nfilt, filnames, howstellar, extinpars, RVunit, extind_PDF,npl,out_folder):
    
    """
    Function to write out the medians, best values and distributions of the jump parameters
    and derived parameters to files. It also plots the histograms of the parameters and derived parameters
    and saves them to the output folder.

    Parameters:
    ----------
    posterior : array
        The posterior samples from the sampling.
    jnames : list
        List of the names of the jump parameters.
    ijnames : list
        List of the indices of the jump parameters.
    njnames : list
        List of the names of the non-jump parameters.
    nijnames : list
        List of the indices of the non-jump parameters.
    bp : array
        The max-probability values of the parameters.
    uwl : array
        The effective wavelengths of the filters.
    Rs_in : float
        The input value of the stellar radius.
    Ms_in : float
        The input value of the stellar mass.
    Rs_PDF : array
        The PDF of the stellar radius.
    Ms_PDF : array
        The PDF of the stellar mass.
    nfilt : int
        The number of filters.
    filnames : list
        The names of the filters.
    howstellar : str
        The method used to specify the stellar parameters.
    extinpars : list
        The names of the external input parameters.
    RVunit : str
        The unit of the radial velocity.
    extind_PDF : array
        The PDF of the external input parameters.
    npl : int
        The number of planets.
    out_folder : str
        The folder to save the outputs.

    """
    npoint,npara=posterior.shape
    #TODO: can be improved with parameter names already in posterior dict
    # =============== calculate the physical parameters taking into account the input and inferred PDFs ========
    # first, allocate the necessary jump parameters into PDF variables

    derived_pnames, derived_PDFs, derived_bp = [],[],[]
    if "rho_star" in jnames or "rho_star" in njnames:
        ind = np.where(np.char.find(jnames, 'rho_star')==0)[0]
        indn = np.where(np.char.find(njnames, 'rho_star')==0)[0]
        if len(extinpars) > 0:
            inde = np.where(np.char.find(extinpars, 'rho_star')==0)[0]
        else :
            inde = []
        if (len(ind) > 0):
            rho_PDF = posterior[:,ind[0]]
            rho_bp = bp[ijnames[0][ind]]
        else:
            rho_PDF = np.zeros(npoint)
            rho_PDF[:] = bp[nijnames[0][indn]]
            rho_bp = bp[nijnames[0][indn]]
        if (len(inde) > 0):
            rho_PDF = np.squeeze(extind_PDF[:,inde])   

    else:
        ind = np.where(np.char.find(jnames, 'Duration')==0)[0]
        indn = np.where(np.char.find(njnames, 'Duration')==0)[0]
        if len(extinpars) > 0:
            inde = np.where(np.char.find(extinpars, 'Duration')==0)[0]
        else :
            inde = []
        if (len(ind) > 0):
            dur_PDF = posterior[:,ind[0]]
            dur_bp = bp[ijnames[0][ind]]
        else:
            dur_PDF = np.zeros(npoint)
            dur_PDF[:] = bp[nijnames[0][indn]]
            dur_bp = bp[nijnames[0][indn]]
        if (len(inde) > 0):
            dur_PDF = np.squeeze(extind_PDF[:,inde])   


    q1_PDF, q2_PDF = np.zeros((npoint,nfilt)),np.zeros((npoint,nfilt))
    q1_bp,  q2_bp  = np.zeros(nfilt), np.zeros(nfilt)
    Fn_PDF, DFocc_PDF, phoff_PDF,contam_PDF,dRpRs_PDF = np.zeros((npoint,nfilt)),np.zeros((npoint,nfilt)),np.zeros((npoint,nfilt)),np.zeros((npoint,nfilt)),np.zeros((npoint,nfilt))
    Fn_bp,  DFocc_bp,  phoff_bp, contam_bp,dRpRs_bp  = np.zeros(nfilt),np.zeros(nfilt),np.zeros(nfilt),np.zeros(nfilt),np.zeros(nfilt)

    for i,fil in enumerate(filnames):
        #q1
        ind = np.where(np.char.find(jnames, fil+'_q1')==0)[0]
        indn = np.where(np.char.find(njnames, fil+'_q1')==0)[0]
        if (len(ind) > 0):
            q1_PDF[:,i] = posterior[:,ind[0]]
            q1_bp[i] = bp[ijnames[0][ind]]
        else:
            q1_PDF[:,i] = bp[nijnames[0][indn]]
            q1_bp[i] = bp[nijnames[0][indn]]
        #q2
        ind = np.where(np.char.find(jnames, fil+'_q2')==0)[0]
        indn = np.where(np.char.find(njnames, fil+'_q2')==0)[0]
        if (len(ind) > 0):
            q2_PDF[:,i] = posterior[:,ind[0]]
            q2_bp[i] = bp[ijnames[0][ind]]
        else:
            q2_PDF[:,i] = bp[nijnames[0][indn]]
            q2_bp[i] = bp[nijnames[0][indn]]   

        #Fn
        ind = np.where(np.char.find(jnames, fil+'_Fn')==0)[0]
        indn = np.where(np.char.find(njnames, fil+'_Fn')==0)[0]
        if (len(ind) > 0):
            Fn_PDF[:,i] = posterior[:,ind[0]]
            Fn_bp[i] = bp[ijnames[0][ind]]
        else:
            Fn_PDF[:,i] = bp[nijnames[0][indn]]
            Fn_bp[i] = bp[nijnames[0][indn]]

        #DFocc
        ind = np.where(np.char.find(jnames, fil+'_DFocc')==0)[0]
        indn = np.where(np.char.find(njnames, fil+'_DFocc')==0)[0]
        if (len(ind) > 0):
            DFocc_PDF[:,i] = posterior[:,ind[0]]
            DFocc_bp[i] = bp[ijnames[0][ind]]
        else:
            DFocc_PDF[:,i] = bp[nijnames[0][indn]]
            DFocc_bp[i] = bp[nijnames[0][indn]]

        #phoff
        ind  = np.where(np.char.find(jnames,  fil+'_ph_off')==0)[0]
        indn = np.where(np.char.find(njnames, fil+'_ph_off')==0)[0]
        if (len(ind) > 0):
            phoff_PDF[:,i] = posterior[:,ind[0]]
            phoff_bp[i] = bp[ijnames[0][ind]]
        else:
            phoff_PDF[:,i] = bp[nijnames[0][indn]]
            phoff_bp[i] = bp[nijnames[0][indn]]

        #contam
        ind  = np.where(np.char.find(jnames,  fil+'_cont')==0)[0]
        indn = np.where(np.char.find(njnames, fil+'_cont')==0)[0]
        if (len(ind) > 0):
            contam_PDF[:,i] = posterior[:,ind[0]]
            contam_bp[i] = bp[ijnames[0][ind]]
        else:
            contam_PDF[:,i] = bp[nijnames[0][indn]]
            contam_bp[i] = bp[nijnames[0][indn]]

        #dRpRs
        if f"{fil}_dRpRs" in jnames:
            ind = np.where(np.char.find(jnames, fil+'_dRpRs')==0)[0]
            indn = np.where(np.char.find(njnames, fil+'_dRpRs')==0)[0]
            if (len(ind) > 0):
                dRpRs_PDF[:,i] = posterior[:,ind[0]]
                dRpRs_bp[i] = bp[ijnames[0][ind]]
            else:
                dRpRs_PDF[:,i] = bp[nijnames[0][indn]]
                dRpRs_bp[i] = bp[nijnames[0][indn]]

    # mean radius deviation across all wls
    mean_RpRs_PDF = np.mean(dRpRs_PDF,axis=1)
    mean_RpRs_bp  = np.mean(dRpRs_bp)

    for n in range(npl):
        nm = f"_{n+1}" if npl>1 else ""

        ind  = np.where(np.char.find(jnames, 'T_0'+nm)==0)[0]
        indn = np.where(np.char.find(njnames, 'T_0'+nm)==0)[0]
        if len(extinpars) > 0:
            inde = np.where(np.char.find(extinpars, 'T_0'+nm)==0)[0]
        else :
            inde = []
        if (len(ind) > 0):
            T0_PDF = posterior[:,ind[0]]
            T0_bp = bp[ijnames[0][ind]]
        else:
            T0_PDF = np.zeros(npoint)
            T0_PDF[:] = bp[nijnames[0][indn]]
            T0_bp = bp[nijnames[0][indn]]
        if (len(inde) > 0):
            T0_PDF = extind_PDF[:,inde]
            posterior[:,ind[0]] = extind_PDF[:,inde]

        ind = np.where(np.char.find(jnames, 'Period'+nm)==0)[0]
        indn = np.where(np.char.find(njnames, 'Period'+nm)==0)[0]
        if len(extinpars) > 0:
            inde = np.where(np.char.find(extinpars, 'Period'+nm)==0)[0]
        else :
            inde = []
        if (len(ind) > 0):
            Period_PDF = posterior[:,ind[0]]
            Period_bp = bp[ijnames[0][ind]]
        else:
            Period_PDF = np.zeros(npoint)
            Period_PDF[:] = bp[nijnames[0][indn]]
            Period_bp = bp[nijnames[0][indn]]
        if (len(inde) > 0):
            Period_PDF = np.squeeze(extind_PDF[:,inde])
            
        ind = np.where(np.char.find(jnames, 'RpRs'+nm)==0)[0]
        indn = np.where(np.char.find(njnames, 'RpRs'+nm)==0)[0]
        if len(extinpars) > 0:
            inde = np.where(np.char.find(extinpars, 'RpRs'+nm)==0)[0]
        else :
            inde = []
        if (len(ind) > 0):
            RpRs_PDF = posterior[:,ind[0]]
            RpRs_bp = bp[ijnames[0][ind]]
        else:
            RpRs_PDF = np.zeros(npoint)
            RpRs_PDF[:] = bp[nijnames[0][indn]]
            RpRs_bp = bp[nijnames[0][indn]]
        if (len(inde) > 0):
            RpRs_PDF = np.squeeze(extind_PDF[:,inde])
        
        ind = np.where(jnames == 'Impact_para'+nm)[0] #np.where(np.char.find(jnames, 'b'+nm)==0)[0] (fixes problems when lc file name begins with b)
        indn = np.where(njnames == 'Impact_para'+nm)[0] #np.where(np.char.find(njnames, 'b')==0)[0]
        if len(extinpars) > 0:
            inde = np.where(extinpars == 'Impact_para'+nm)[0] #np.where(np.char.find(extinpars, 'b')==0)[0]
        else :
            inde = []
        if (len(ind) > 0):
            b_PDF = posterior[:,ind[0]]
            b_bp = bp[ijnames[0][ind]]
        else:
            b_PDF = np.zeros(npoint)
            b_PDF[:] = bp[nijnames[0][indn]]
            b_bp = bp[nijnames[0][indn]]
        if (len(inde) > 0):
            b_PDF = np.squeeze(extind_PDF[:,inde])        

        ind = np.where(np.char.find(jnames, 'secos(w)'+nm)==0)[0]
        indn = np.where(np.char.find(njnames, 'secos(w)'+nm)==0)[0]
        if (len(ind) > 0):
            ecosw_PDF = posterior[:,ind[0]]
            ecosw_bp = bp[ijnames[0][ind]]
        else:
            ecosw_PDF = np.zeros(npoint)
            ecosw_PDF[:] = bp[nijnames[0][indn]]
            ecosw_bp = bp[nijnames[0][indn]]
        
        ind = np.where(np.char.find(jnames, 'sesin(w)'+nm)==0)[0]
        indn = np.where(np.char.find(njnames, 'sesin(w)'+nm)==0)[0]
        if (len(ind) > 0):
            esinw_PDF = posterior[:,ind[0]]
            esinw_bp = bp[ijnames[0][ind]]
        else:
            esinw_PDF = np.zeros(npoint)
            esinw_PDF[:] = bp[nijnames[0][indn]]
            esinw_bp = bp[nijnames[0][indn]]

        ind = np.where(jnames == 'K'+nm)[0] #np.where(np.char.find(jnames, 'K'+nm)==0)[0] (fixes problems when lc file name begins with K)
        indn = np.where(njnames == 'K'+nm)[0] #np.where(np.char.find(njnames, 'K'+nm)==0)[0]
        if len(extinpars) > 0:
            inde = np.where(extinpars == 'K'+nm)[0] #np.where(np.char.find(extinpars, 'K'+nm)==0)[0]
        else :
            inde = []
        if (len(ind) > 0):
            K_PDF = posterior[:,ind[0]]
            K_bp = bp[ijnames[0][ind]]
        else:
            K_PDF = np.zeros(npoint)
            K_PDF[:] = bp[nijnames[0][indn]]
            K_bp = bp[nijnames[0][indn]]
        if (len(inde) > 0):
            K_PDF = np.squeeze(extind_PDF[:,inde])
        if RVunit == 'm/s': #convert to km/s
            K_PDF = K_PDF/1000.
            K_bp  = K_bp/1000.

        if "rho_star" in jnames or "rho_star" in njnames:
            dur_PDF = rho_to_tdur(rho_PDF, b_PDF, RpRs_PDF+mean_RpRs_PDF, Period_PDF,
                                    e=esinw_PDF**2+ecosw_PDF**2, w=np.degrees(np.arctan2(esinw_PDF,ecosw_PDF)))
            dur_bp  = rho_to_tdur(rho_bp, b_bp, RpRs_bp, Period_bp,
                                    e=esinw_bp**2+ecosw_bp**2, w=np.degrees(np.arctan2(esinw_bp,ecosw_bp)))
        else:
            rho_PDF = tdur_to_rho(dur_PDF, b_PDF, RpRs_PDF+mean_RpRs_PDF, Period_PDF,e=esinw_PDF**2+ecosw_PDF**2, 
                                    w=np.degrees(np.arctan2(esinw_PDF,ecosw_PDF)))
            rho_bp  = tdur_to_rho(dur_bp, b_bp, RpRs_bp, Period_bp,e=esinw_bp**2+ecosw_bp**2,
                                    w=np.degrees(np.arctan2(esinw_bp,ecosw_bp)))

        
        pnames, PDFs, starstring = derive_parameters(filnames, nm, Rs_PDF, Ms_PDF, RpRs_PDF+mean_RpRs_PDF, Period_PDF, b_PDF, dur_PDF,rho_PDF, ecosw_PDF, esinw_PDF, K_PDF, 
                                                        q1_PDF, q2_PDF,Fn_PDF, DFocc_PDF,phoff_PDF,howstellar) 
        _,   bp_PDFs, _          = derive_parameters(filnames, nm, Rs_in,  Ms_in,  RpRs_bp+mean_RpRs_bp,  Period_bp,  b_bp,  dur_bp, rho_bp,  ecosw_bp,  esinw_bp,  K_bp,  
                                                        q1_bp,  q2_bp, Fn_bp, DFocc_bp,phoff_bp, howstellar)

        derived_pnames.extend(pnames)
        derived_PDFs.extend(PDFs)
        derived_bp.extend(bp_PDFs)

        
        nderived = len(derived_pnames)

    # =============================================================================================================
    #                  START OUTPUT SECTION 
    # =============================================================================================================

    # =============== write out the medians, best values and distributions of the jump parameters =================
    outfile  = out_folder+"/"+'results_med.dat'
    outfile2 = out_folder+"/"+'results_max.dat'
    outfile3 = out_folder+"/"+'results_bf.dat'

    of  = open(outfile,'w')
    of2 = open(outfile2,'w')
    of3 = open(outfile3,'w')

    # posterior has the burned-in, thinned parameter states
    
    n1sig = np.round(0.34134*npoint)  # number of points for 1 sigma (on one side)
    n3sig = np.round(0.49865*npoint)  # number of points for 3 sigma (on one side)
    i1sig = np.array([np.round(npoint/2)-n1sig,np.round(npoint/2)+n1sig], dtype='int32') # indexes of the points at median -/+ 1 n1sig
    i3sig = np.array([np.round(npoint/2)-n3sig,np.round(npoint/2)+n3sig], dtype='int32') # indexes of the points at median -/+ 1 n1sig
    
    medvals  = np.zeros(npara)
    maxvals  = np.zeros(npara)
    medvalsd = np.zeros(nderived)
    maxvalsd = np.zeros(nderived)

    sig1   =  np.zeros([npara,2])   # array to contain the 1-sigma limits [lower, upper] for all parameters
    sig3   =  np.zeros([npara,2])   # array to contain the 3-sigma limits [lower, upper] for all parameters
    sig1m  =  np.zeros([npara,2])  # array to contain the 1-sigma limits [lower, upper] for all parameters
    sig3m  =  np.zeros([npara,2])  # array to contain the 3-sigma limits [lower, upper] for all parameters
    sig1s  =  np.zeros([2])       # array to contain the 1-sigma limits [lower, upper] for a single parameter
    sig3s  =  np.zeros([2]) 
    sig1ms =  np.zeros([2]) 
    sig3ms =  np.zeros([2]) 
    sig1d  =  np.zeros([nderived,2])  # array to contain the 1-sigma limits [lower, upper] for the derived parameters
    sig3d  =  np.zeros([nderived,2]) 
    sig1md =  np.zeros([nderived,2]) 
    sig3md =  np.zeros([nderived,2]) 
    
    of.write('#====================================================================================================\n')
    of.write(f'#{"Jump parameters":25s} {"median":14s} {"-1sigma":14s} {"+1sigma":14s} {"-3sigma":14s} {"+3sigma":14s}\n')
    of.write('#====================================================================================================\n')
    
    of2.write('#====================================================================================================\n')
    of2.write(f'#{"Jump parameters":25s} {"median":14s} {"-1sigma":14s} {"+1sigma":14s} {"-3sigma":14s} {"+3sigma":14s}\n')
    of2.write('#====================================================================================================\n')

    of3.write('#====================================================================================================\n')
    of3.write(f'#{"Jump parameters":25s} {"median":14s} {"-1sigma":14s} {"+1sigma":14s} {"-3sigma":14s} {"+3sigma":14s}\n')
    of3.write('#====================================================================================================\n')

    for i in range(npara):
        vals=posterior[:,i]
        # calculate median
        medvals[i] = np.median(vals)
        dval=vals-medvals[i] # the difference between vals and the median
        sval=np.sort(dval)
        sig1[i] = sval[i1sig] # the 1-sigma intervals (the left side is naturally negative) 
        sig3[i] = sval[i3sig] # the 1-sigma intervals (the left side is naturally negative) 
        #print jnames[i], medvals[i],sig1[i,0],sig1[i,1], sig3[i,0], sig3[i,1]
        of.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (jnames[i], medvals[i],sig1[i,0],sig1[i,1], sig3[i,0], sig3[i,1]))       
    
    of.write('#====================================================================================================\n')
    of.write('#Stellar input parameters: \n')
    of.write('#====================================================================================================\n')
    if howstellar == 'Rrho':
        vals=Rs_PDF
        # calculate median
        medval = np.median(vals)
        dval   = vals-medval # the difference between vals and the median
        sval   = np.sort(dval)
        sig1s  = sval[i1sig] # the 1-sigma intervals (the left side is naturally negative) 
        sig3s  = sval[i3sig] # the 1-sigma intervals (the left side is naturally negative) 
        of.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('#Rstar', medval,sig1s[0],sig1s[1], sig3s[0], sig3s[1]))  
    if howstellar == 'Mrho':
        vals   = Ms_PDF
        # calculate median
        medval = np.median(vals)
        dval   = vals-medval # the difference between vals and the median
        sval   = np.sort(dval)
        sig1s  = sval[i1sig] # the 1-sigma intervals (the left side is naturally negative) 
        sig3s  = sval[i3sig] # the 1-sigma intervals (the left side is naturally negative) 
        of.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('#Mstar', medval,sig1s[0],sig1s[1], sig3s[0], sig3s[1]))  

    for i in range(len(extinpars)):
        vals   = extind_PDF[:,i]
        medval = np.median(vals)
        dval   = vals-medval # the difference between vals and the median
        sval   = np.sort(dval)
        sig1s  = sval[i1sig] # the 1-sigma intervals (the left side is naturally negative) 
        sig3s  = sval[i3sig] # the 1-sigma intervals (the left side is naturally negative) 
        of.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (extinpars[i], medval,sig1s[0],sig1s[1], sig3s[0], sig3s[1]))  


    of.write('#====================================================================================================\n')
    of.write('#Derived parameters: ('+starstring+') \n')
    of.write('#====================================================================================================\n')
        
    for i in range(nderived):
        vals=derived_PDFs[i]
        # calculate median
        medvalsd[i] = np.median(vals)
        dval        = vals-medvalsd[i] # the difference between vals and the median
        sval        = np.sort(dval)
        sig1d[i]    = sval[i1sig] # the 1-sigma intervals (the left side is naturally negative) 
        sig3d[i]    = sval[i3sig] # the 1-sigma intervals (the left side is naturally negative) 
        of.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (derived_pnames[i], medvalsd[i],sig1d[i,0],sig1d[i,1], sig3d[i,0], sig3d[i,1])) 

    of.write('#====================================================================================================\n')     
    of.write('#Fixed parameters: \n')
    of.write('#====================================================================================================\n')

    nfix = len(njnames)
    for i in range(nfix):
        if (njnames[i] in extinpars):
            pass
        else:
            of.write('%-25s %14.8f \n' % (njnames[i], bp[nijnames[0][i]])) 
        
    of.write('#====================================================================================================\n')
    of.close()
    
    #now write out outfile2: the peak of the posterior and the area containing 68% of points
    for i in range(npara):     
        vals=posterior[:,i]
        pdf, xpdf, HPDmin, iHDP = credregionML(vals)
        maxvals[i] = xpdf[iHDP]
        sig1m[i,0] = np.amin(xpdf[pdf>HPDmin]) - maxvals[i]
        sig1m[i,1] = np.amax(xpdf[pdf>HPDmin]) - maxvals[i]
        s1bps=[np.amin(xpdf[pdf>HPDmin]) - bp[ijnames[0][i]] , np.amax(xpdf[pdf>HPDmin]) - bp[ijnames[0][i]]]
        pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
        sig3m[i,0] = np.amin(xpdf[pdf>HPDmin]) - maxvals[i]
        sig3m[i,1] = np.amax(xpdf[pdf>HPDmin]) - maxvals[i]      
        of2.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (jnames[i], maxvals[i],sig1m[i,0],sig1m[i,1], sig3m[i,0], sig3m[i,1])) 
        param_histbp(vals,jnames[i],medvals[i],sig1[i],sig3[i],maxvals[i],sig1m[i],sig3m[i],bp[ijnames[0][i]],s1bps,out_folder)

    of2.write('#====================================================================================================\n')
    of2.write('#Stellar input parameters: \n')
    of2.write('#====================================================================================================\n')
    if howstellar == 'Rrho':
        vals=Rs_PDF
        pdf, xpdf, HPDmin, iHDP = credregionML(vals)
        maxval = xpdf[iHDP]
        sig1ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig1ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval
        pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
        sig3ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig3ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval      
        of2.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('#Rstar', maxval,sig1ms[0],sig1ms[1], sig3ms[0], sig3ms[1])) 
        param_hist(vals,'Rstar',medval,sig1s,sig3s,maxval,sig1ms,sig3ms,out_folder=out_folder)
    if howstellar == 'Mrho':
        vals=Ms_PDF
        pdf, xpdf, HPDmin, iHDP = credregionML(vals)
        maxval = xpdf[iHDP]
        sig1ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig1ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval
        pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
        sig3ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig3ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval      
        of2.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('#Mstar', maxval,sig1ms[0],sig1ms[1], sig3ms[0], sig3ms[1])) 
        param_hist(vals,'Mstar',medval,sig1s,sig3s,maxval,sig1ms,sig3ms,out_folder=out_folder)

    for i in range(len(extinpars)):
        vals = extind_PDF[:,i]
        pdf, xpdf, HPDmin, iHDP = credregionML(vals)
        maxval = xpdf[iHDP]
        sig1ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig1ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval
        pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
        sig3ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig3ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval      
        of2.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (extinpars[i], maxval,sig1ms[0],sig1ms[1], sig3ms[0], sig3ms[1]))  
    
    of2.write('#====================================================================================================\n')
    of2.write('#Derived parameters: ('+starstring+') \n')
    of2.write('#====================================================================================================\n')    
    
    for i in range(nderived):     
        vals=derived_PDFs[i]
        nans = np.isnan(vals)
        lnans = len(nans[nans==True])
        if (lnans==0):
            if (min(vals) != max(vals)):
                pdf, xpdf, HPDmin, iHDP = credregionML(vals)
                maxvalsd[i] = xpdf[iHDP]
                sig1md[i,0] = np.amin(xpdf[pdf>HPDmin]) - maxvalsd[i]
                sig1md[i,1] = np.amax(xpdf[pdf>HPDmin]) - maxvalsd[i]
                pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
                sig3md[i,0] = np.amin(xpdf[pdf>HPDmin]) - maxvalsd[i]
                sig3md[i,1] = np.amax(xpdf[pdf>HPDmin]) - maxvalsd[i]    
            else:
                maxvalsd[i] = np.mean(vals)
                sig1md[i,0] = 0.
                sig1md[i,1] = 0.
                sig3md[i,0] = 0.
                sig3md[i,1] = 0.
        else:
                maxvalsd[i] = 9999999.
                sig1md[i,0] = 0.
                sig1md[i,1] = 0.
                sig3md[i,0] = 0.
                sig3md[i,1] = 0.
            
        of2.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (derived_pnames[i], maxvalsd[i], sig1md[i,0],sig1md[i,1], sig3md[i,0], sig3md[i,1])) 
        try: param_hist(vals,derived_pnames[i],medvalsd[i],sig1d[i],sig3d[i],maxvalsd[i],sig1md[i],sig3md[i],out_folder=out_folder)
        except: pass
    of2.write('#====================================================================================================\n')
    of2.write('#Fixed parameters: \n')
    of2.write('#====================================================================================================\n')

    nfix = len(njnames)
    for i in range(nfix):
        if (njnames[i] in extinpars):
            pass
        else:
            of2.write('%-25s %14.8f \n' % (njnames[i], bp[nijnames[0][i]])) 

    
    of2.write('#====================================================================================================\n')    
    of2.close()
        
        
    #now write out outfile3: the peak of the posterior and the uncertainties given by MC3
    for i in range(npara):
        
        vals=posterior[:,i]
        pdf, xpdf, HPDmin, iHDP = credregionML(vals)
        maxvals[i] = xpdf[iHDP]
        sig1m[i,0] = np.amin(xpdf[pdf>HPDmin]) - maxvals[i]
        sig1m[i,1] = np.amax(xpdf[pdf>HPDmin]) - maxvals[i]
        s1bps=[np.amin(xpdf[pdf>HPDmin]) - bp[ijnames[0][i]] , np.amax(xpdf[pdf>HPDmin]) - bp[ijnames[0][i]]]
        pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
        sig3m[i,0] = np.amin(xpdf[pdf>HPDmin]) - maxvals[i]
        sig3m[i,1] = np.amax(xpdf[pdf>HPDmin]) - maxvals[i]      
        s3bps=[np.amin(xpdf[pdf>HPDmin]) - bp[ijnames[0][i]] , np.amax(xpdf[pdf>HPDmin]) - bp[ijnames[0][i]]]
        
        of3.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (jnames[i],bp[ijnames[0][i]],s1bps[0],s1bps[1],s3bps[0],s3bps[1]))     
        
    of3.write('#====================================================================================================\n')
    of3.write('#Stellar input parameters: \n')
    of3.write('#====================================================================================================\n')
    if howstellar == 'Rrho':
        vals=Rs_PDF
        pdf, xpdf, HPDmin, iHDP = credregionML(vals)
        maxval = xpdf[iHDP]
        sig1ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig1ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval
        pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
        sig3ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig3ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval      
        of3.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('#Rstar', maxval,sig1ms[0],sig1ms[1], sig3ms[0], sig3ms[1])) 
        param_hist(vals,'Rstar',medval,sig1s,sig3s,maxval,sig1ms,sig3ms,out_folder=out_folder)
    if howstellar == 'Mrho':
        vals=Ms_PDF
        pdf, xpdf, HPDmin, iHDP = credregionML(vals)
        maxval = xpdf[iHDP]
        sig1ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig1ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval
        pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
        sig3ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig3ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval      
        of3.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('#Mstar', maxval,sig1ms[0],sig1ms[1], sig3ms[0], sig3ms[1])) 
        param_hist(vals,'Mstar',medval,sig1s,sig3s,maxval,sig1ms,sig3ms,out_folder=out_folder)
    for i in range(len(extinpars)):
        vals = extind_PDF[:,i]
        pdf, xpdf, HPDmin, iHDP = credregionML(vals)
        maxval = xpdf[iHDP]
        sig1ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig1ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval
        pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
        sig3ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
        sig3ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval      
        of3.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (extinpars[i], maxval,sig1ms[0],sig1ms[1], sig3ms[0], sig3ms[1]))  
    
    #more in here!!
    # !!! the limits are the same, but the reported center values are different
    # how to best report this here? 1. combination of the bp values
    #                               2. errors come from bp + [lower, upper] limit of the parameter region
    #                                  [err_on_bp] == bp - [maxval + sig1ms]
    
    # FIRST: derive the bp values, call them der_bp

    of3.write('#====================================================================================================\n')
    of3.write('#Derived parameters: ('+starstring+') \n')
    of3.write('#====================================================================================================\n')    
    
    for i in range(nderived):     
        
        sig1bp=sig1md[i,:] - (derived_bp[i] -maxvalsd[i])
        sig3bp=sig3md[i,:] - (derived_bp[i] -maxvalsd[i])
        
        of3.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (derived_pnames[i], derived_bp[i], sig1bp[0],sig1bp[1], sig3bp[0], sig3bp[1])) 
    
    of3.write('#====================================================================================================\n')
    of3.write('#Fixed parameters: \n')
    of3.write('#====================================================================================================\n')

    nfix = len(njnames)
    for i in range(nfix):
        if (njnames[i] in extinpars):
            pass
        else:
            of3.write('%-25s %14.8f \n' % (njnames[i], bp[nijnames[0][i]])) 
    
    of3.write('#====================================================================================================\n')    
    of3.close()        
        
        
    # ============================ plot the output transmission spec ======================================
    # first get indices of the parameters that contain a ddf value
    dRpRsres = np.array([])
    edRpRsres = np.array([])
    for i in range(npara):
        if (np.char.find(jnames[i],'dRpRs') > -1):
            dRpRsres= np.concatenate((dRpRsres, [medvals[i]]))
            edRpRsres= np.concatenate((edRpRsres, [sig1[i,1]]))
    
    if len(dRpRsres)>0:
        plot_traspec(dRpRsres, edRpRsres, uwl,out_folder)

    
    return medvals, maxvals,np.median(abs(sig1),axis=1)


def gr_print(jnames,GRvals, out_folder):
    
    outfile=out_folder+'/GRvals.dat'
    of=open(outfile,'w')
    npara=len(jnames)
    for i in range(npara):
        of.write('%-25s %10.6f \n' % (jnames[i], GRvals[i]))

    of.close()


def get_AIC_BIC(npar,ndata,chi2,out_folder):
    
    BIC = chi2 + npar * np.log(ndata)
    AIC = chi2 + npar * ndata *2. / (ndata - npar -1.)

    RCHI = chi2 /(ndata-npar)
    
    outfile=out_folder+'/AIC_BIC.dat'
    of=open(outfile,'w')
    of.write(f'{"data points:":20s} {ndata:10.0f} \n')
    of.write(f'{"free parameters:":20s} {npar:10.0f} \n')
    of.write(f'{"AIC":10s}\t{"BIC":10s}\t{"reduced CHI2"}\n')
    of.write(f'{AIC:10.3f}\t{BIC:10.3f}\t{RCHI:10.2f} \n')
    of.close()
    return


def dyn_summary(res,out_folder,ns_type):
    """Return a formatted string giving a quick summary
    of the results."""
    try: nlive = res.nlive
    except: nlive = 0
    res_print = ("nlive: {:d}\n"
                    "niter: {:d}\n"
                    "ncall: {:d}\n"
                    "eff(%): {:10.3f}\n"
                    "logz: {:10.3f} +/- {:8.3f}\n"
                    "max_logl: {:10.3f}"
                    .format(nlive, res.niter, sum(res.ncall),
                            res.eff, res.logz[-1], res.logzerr[-1], res.logl[-1]))
    f = open(f"{out_folder}/evidence.dat", "w")
    print(f'Summary [{ns_type}]\n=======\n'+res_print, file=f)
    f.close()
    return


def get_PDF_Gauss(cen,sig1,sig2,dim):
    sig = (sig1 + sig2)/2.
    npoints = int(dim[0])  # number of samples needed
    val_PDF = np.random.normal(cen, sig, int(npoints))
    return val_PDF

def derive_parameters(filnames, nm, Rs_PDF, Ms_PDF, RpRs_PDF, Period_PDF, b_PDF, dur_PDF, rhoS_PDF, ecosw_PDF, esinw_PDF, K_PDF, 
                        q1_PDF, q2_PDF, Fn_PDF, DFocc_PDF, phoff_PDF, howstellar):
    
    import scipy.constants as cn

    au = 149597870700. # IAU value
    
    secondsperday = 24.*3600.
    
    Rsolar  = 6.957e8   # IAU value
    Rjup    = 7.1492e7    # IAU value
    Rearth  = 6.3781e6  # IAU value
    
    GMsolar = 1.3271244e20 # IAU value
    GMjup   = 1.2668653e17   # IAU value
    GMearth = 3.986004e14  # IAU value
    
    Msolar = GMsolar / cn.G # IAU suggestion
    Mjup   = GMjup / cn.G     # IAU suggestion
    Mearth = GMearth / cn.G # IAU suggestion
    
    rhoSolar = 3. / (4. * np.pi) * Msolar / Rsolar**3 
    rhoJup   = 3. / (4. * np.pi) * Mjup / Rjup**3 
    rhoEarth = 3. / (4. * np.pi) * Mearth / Rearth**3 
    
    ecc_PDF = ecosw_PDF**2 + esinw_PDF**2
    ome_PDF = np.arctan2(esinw_PDF, ecosw_PDF)
    ome_PDF[ome_PDF<0] = ome_PDF[ome_PDF<0] + 2*np.pi
    # ome_PDF[ecc_PDF<1e-15] = np.pi/2.
    
    # ome_PDF[(esinw_PDF<0) & (ecosw_PDF<0)] = ome_PDF[(esinw_PDF<0) & (ecosw_PDF<0)] + np.pi
    # ome_PDF[(esinw_PDF<0) & (ecosw_PDF>0)] = 2. * np.pi - ome_PDF[(esinw_PDF<0) & (ecosw_PDF>0)]
    # ome_PDF[(esinw_PDF>0) & (ecosw_PDF<0)] = np.pi - ome_PDF[(esinw_PDF>0) & (ecosw_PDF<0)]

    # e0ind = np.where(ecc_PDF<1e-15)   # avoid NaNs for very small eccentricity
    # ome_PDF[e0ind] = 0.               # by defining omeaga == 0
    
    efac1_PDF = np.sqrt(1.-ecc_PDF**2)/(1.+ ecc_PDF*np.sin(ome_PDF))
    efac2_PDF = b_PDF*(1.-ecc_PDF**2)/(1.+ecc_PDF*np.sin(ome_PDF))
    efac3_PDF = np.sqrt(1.-ecc_PDF**2)/(1.- ecc_PDF*np.sin(ome_PDF))
    efac4_PDF = (1.-ecc_PDF**2)/(1.+ ecc_PDF*np.sin(ome_PDF))


    if (howstellar == 'Rrho'):
        Ms_PDF = rhoS_PDF * 4/3 * np.pi *  (Rs_PDF * Rsolar*100)**3 / (Msolar*1000)
        starstring = 'stellar Mass from R+rho'

    elif (howstellar == 'Mrho'):
        Rs_PDF = ((3. * Ms_PDF * Msolar*1000) / (4. * np.pi * rhoS_PDF))**(1./3.) / (Rsolar*100)
        starstring = 'stellar Radius from M+rho'
    
    Rp_PDF  = RpRs_PDF * (Rs_PDF * Rsolar) / Rjup  
    dF_PDF  = RpRs_PDF**2
    # else:
    #     rhoS_PDF = Ms_PDF / Rs_PDF**3 * rhoSolar
    #     starstring = 'rho_s from stellar parameter input'
        
    aRs_PDF = rho_to_aR(rhoS_PDF,Period_PDF,ecc_PDF,np.degrees(ome_PDF))
    a_PDF = aRs_PDF * Rs_PDF * Rsolar / au
    Rsa_PDF = 1./aRs_PDF

    rhoS_PDF = rhoS_PDF #/ (rhoSolar/1000)   #solar units

    ome_PDF = ome_PDF * 180. / np.pi

    Mp_PDF = (Period_PDF*3600.*24. / (2.*np.pi*cn.G))**(1./3.) * K_PDF*1000 * (Ms_PDF*Msolar)**(2./3.) * np.sqrt(1. - ecc_PDF**2) / Mjup

    MF_PDF = Period_PDF* secondsperday * (K_PDF*1000)**3 / (2. * np.pi * cn.G) / Msolar * (1. - ecc_PDF**2)**(3/2) # mass function in Solar mass
    #FWHM_PDF = Rsa_PDF * np.sqrt(1. - b_PDF**2)/np.pi

    if (np.any(np.isfinite(np.sqrt(1. - b_PDF**2)))==False):
        FWHM_PDF = np.zeros(len(b_PDF))
    else:
        FWHM_PDF = Rsa_PDF * np.sqrt(1. - b_PDF**2)/np.pi
        
    inc_PDF = np.arccos(b_PDF / aRs_PDF/efac4_PDF) * 180. / np.pi   #eqn 7: https://arxiv.org/pdf/1001.2010.pdf
    
    rhoP_PDF = Mp_PDF / Rp_PDF**3 
    gP_PDF =  cn.G * (Mp_PDF*Mjup) / (Rp_PDF*Rjup)**2 
    
    durocc_PDF = aR_to_Tdur(aRs_PDF,b_PDF,RpRs_PDF,Period_PDF,ecc_PDF,ome_PDF,tra_occ="occ")
    durocc_PDF[np.isfinite(durocc_PDF)==False] = 0.
    
    if len(q1_PDF.shape)<2:
        nfil = q1_PDF.shape
        pnames_LD = []
        LD_PDFs = []
        for i in range(nfil[0]):
            u1,u2 = convert_LD(q1_PDF[i], q2_PDF[i],conv="q2u")
            name1 = filnames[i]+'_u1'
            name2 = filnames[i]+'_u2'
            pnames_LD = pnames_LD + [name1] + [name2]
            LD_PDFs.append(u1)
            LD_PDFs.append(u2)
    else:
        npo, nfil = q1_PDF.shape
        pnames_LD = []
        LD_PDFs = []
    
        for i in range(nfil):
            u1,u2 = convert_LD(q1_PDF[:,i], q2_PDF[:,i],conv="q2u")
            name1 = filnames[i]+'_u1'
            name2 = filnames[i]+'_u2'
            pnames_LD = pnames_LD + [name1] + [name2]
            LD_PDFs.append(u1)
            LD_PDFs.append(u2)

    if len(Fn_PDF.shape)<2:
        nfil = Fn_PDF.shape
        pnames_Aatm = []
        Aatm_PDFs = []
        for i in range(nfil[0]):
            Aatm = (DFocc_PDF[i] - Fn_PDF[i])/(2* np.cos(np.deg2rad(phoff_PDF[i])))
            pnames_Aatm += [filnames[i]+'_Aatm']
            Aatm_PDFs.append(Aatm)
    else:
        npo, nfil = Fn_PDF.shape
        pnames_Aatm = []
        Aatm_PDFs = []
    
        for i in range(nfil):
            Aatm = (DFocc_PDF[:,i] - Fn_PDF[:,i])/(2* np.cos(np.deg2rad(phoff_PDF[:,i])))
            pnames_Aatm += [filnames[i]+'_Aatm']
            Aatm_PDFs.append(Aatm)

    derived_pnames = [f"Rp{nm}_[Rjup]",f"Mp{nm}_[Mjup]", f"rho{nm}_p_[rhoJup]", f"g_p{nm}_[SI]", f"dF{nm}", f"aRs{nm}", f"a{nm}_[au]", f"rho_star{nm}_[g_cm3]", "Ms_[Msun]", "Rs_[Rsun]",
                        f"inclination{nm}_[deg]", f"eccentricity{nm}", f"omega{nm}_[deg]", f"Occult_dur{nm}", f"Rs_a{nm}", "MF_PDF_[Msun]",f"Dur{nm}_[d]"]
    
    derived_pnames =  derived_pnames + pnames_LD + pnames_Aatm
        
    derived_PDFs = [Rp_PDF, Mp_PDF, rhoP_PDF, gP_PDF, dF_PDF, aRs_PDF, a_PDF, rhoS_PDF, Ms_PDF, Rs_PDF, inc_PDF, ecc_PDF, ome_PDF, durocc_PDF, Rsa_PDF, MF_PDF,dur_PDF]
    derived_PDFs = derived_PDFs + LD_PDFs + Aatm_PDFs
        
    return derived_pnames, derived_PDFs, starstring

# TODO: - Teq_PDF (requires Teff_star)
#       - incflux_PDF (requires Teff_star)
#       - Mplanet (requires K_RV)




