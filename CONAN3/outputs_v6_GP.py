import sys
import numpy as np
from .plots_v12 import *
import subprocess
import scipy.stats as stats
import scipy.interpolate as si

# sys.path.append("/home/lendl/Work/OurCode/transit_fit/")

from .credibleregion_ML import *


def mcmc_outputs(posterior, jnames, ijnames, njnames, nijnames, bp, ulamdas, Rs_in, Ms_in, Rs_PDF, Ms_PDF, nfilt, filnames, howstellar, extinpars, extins, extind_PDF):
   
     npoint,npara=posterior.shape
     
     # =============== calculate the physical parameters taking into account the input and inferred PDFs ========
     # first, allocate the necessary jump parameters into PDF variables
     ind = np.where(np.char.find(jnames, 'T_0')==0)[0]
     indn = np.where(np.char.find(njnames, 'T_0')==0)[0]
     if len(extinpars) > 0:
        inde = np.where(np.char.find(extinpars, 'T_0')==0)[0]
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

     ind = np.where(np.char.find(jnames, 'Period_[d]')==0)[0]
     indn = np.where(np.char.find(njnames, 'Period_[d]')==0)[0]
     if len(extinpars) > 0:
        inde = np.where(np.char.find(extinpars, 'Period_[d]')==0)[0]
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
         
     ind = np.where(np.char.find(jnames, 'RpRs')==0)[0]
     indn = np.where(np.char.find(njnames, 'RpRs')==0)[0]
     if len(extinpars) > 0:
        inde = np.where(np.char.find(extinpars, 'RpRs')==0)[0]
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
     
     ind = np.where(np.char.find(jnames, 'b')==0)[0]
     indn = np.where(np.char.find(njnames, 'b')==0)[0]
     if len(extinpars) > 0:
        inde = np.where(np.char.find(extinpars, 'b')==0)[0]
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

     ind = np.where(np.char.find(jnames, 'dur_[d]')==0)[0]
     indn = np.where(np.char.find(njnames, 'dur_[d]')==0)[0]
     if len(extinpars) > 0:
        inde = np.where(np.char.find(extinpars, 'dur_[d]')==0)[0]
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

     ind = np.where(np.char.find(jnames, 'ecos(w)')==0)[0]
     indn = np.where(np.char.find(njnames, 'ecos(w)')==0)[0]
     if (len(ind) > 0):
         ecosw_PDF = posterior[:,ind[0]]
         ecosw_bp = bp[ijnames[0][ind]]
     else:
         ecosw_PDF = np.zeros(npoint)
         ecosw_PDF[:] = bp[nijnames[0][indn]]
         ecosw_bp = bp[nijnames[0][indn]]
     
     ind = np.where(np.char.find(jnames, 'esin(w)')==0)[0]
     indn = np.where(np.char.find(njnames, 'esin(w)')==0)[0]
     if (len(ind) > 0):
         esinw_PDF = posterior[:,ind[0]]
         esinw_bp = bp[ijnames[0][ind]]
     else:
         esinw_PDF = np.zeros(npoint)
         esinw_PDF[:] = bp[nijnames[0][indn]]
         esinw_bp = bp[nijnames[0][indn]]

     ind = np.where(np.char.find(jnames, 'K')==0)[0]
     indn = np.where(np.char.find(njnames, 'K')==0)[0]
     if len(extinpars) > 0:
        inde = np.where(np.char.find(extinpars, 'K')==0)[0]
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
         
     c1_PDF = np.zeros((npoint,nfilt))
     c2_PDF = np.zeros((npoint,nfilt))
     c1_bp = np.zeros(nfilt)
     c2_bp = np.zeros(nfilt)
     
     for i in range(nfilt):
         ind = np.where(np.char.find(jnames, filnames[i]+'_c1')==0)[0]
         indn = np.where(np.char.find(njnames, filnames[i]+'_c1')==0)[0]
         if (len(ind) > 0):
             c1_PDF[:,i] = posterior[:,ind[0]]
             c1_bp[i] = bp[ijnames[0][ind]]
         else:
             c1_PDF[:,i] = bp[nijnames[0][indn]]
             c1_bp[i] = bp[nijnames[0][indn]]
         ind = np.where(np.char.find(jnames, filnames[i]+'_c2')==0)[0]
         indn = np.where(np.char.find(njnames, filnames[i]+'_c2')==0)[0]
         if (len(ind) > 0):
             c2_PDF[:,i] = posterior[:,ind[0]]
             c2_bp[i] = bp[ijnames[0][ind]]
         else:
             c2_PDF[:,i] = bp[nijnames[0][indn]]
             c2_bp[i] = bp[nijnames[0][indn]]
     
     derived_pnames, derived_PDFs, starstring = derive_parameters(filnames, Rs_PDF, Ms_PDF, RpRs_PDF, Period_PDF, b_PDF, dur_PDF, ecosw_PDF, esinw_PDF, K_PDF, c1_PDF, c2_PDF, howstellar) 
     
     derived_pnames_bp, derived_bp, starstring_bp = derive_parameters(filnames, Rs_in, Ms_in, RpRs_bp, Period_bp, b_bp, dur_bp, ecosw_bp, esinw_bp, K_bp, c1_bp, c2_bp, howstellar)
     
     nderived = len(derived_pnames)
     
   # =============================================================================================================
   #                  START OUTPUT SECTION 
   # =============================================================================================================

   
   # =============== write out the medians, best values and distributions of the jump parameters =================
     outfile='results_med.dat'
     outfile2='results_max.dat'
     outfile3='results_bf.dat'

     of=open(outfile,'w')
     of2=open(outfile2,'w')
     of3=open(outfile3,'w')

    # posterior has the burned-in, thinned parameter states
     
     n1sig = np.round(0.34134*npoint)  # number of points for 1 sigma (on one side)
     n3sig = np.round(0.49865*npoint)  # number of points for 3 sigma (on one side)
     i1sig = np.array([np.round(npoint/2)-n1sig,np.round(npoint/2)+n1sig], dtype='int32') # indexes of the points at median -/+ 1 n1sig
     i3sig = np.array([np.round(npoint/2)-n3sig,np.round(npoint/2)+n3sig], dtype='int32') # indexes of the points at median -/+ 1 n1sig
     
     medvals = np.zeros(npara)
     maxvals = np.zeros(npara)
     medvalsd = np.zeros(nderived)
     maxvalsd = np.zeros(nderived)

     sig1 =  np.zeros([npara,2])   # array to contain the 1-sigma limits [lower, upper] for all parameters
     sig3 =  np.zeros([npara,2])   # array to contain the 3-sigma limits [lower, upper] for all parameters
     sig1m =  np.zeros([npara,2])  # array to contain the 1-sigma limits [lower, upper] for all parameters
     sig3m =  np.zeros([npara,2])  # array to contain the 3-sigma limits [lower, upper] for all parameters
     sig1s  =  np.zeros([2])       # array to contain the 1-sigma limits [lower, upper] for a single parameter
     sig3s  =  np.zeros([2]) 
     sig1ms  =  np.zeros([2]) 
     sig3ms  =  np.zeros([2]) 
     sig1d  =  np.zeros([nderived,2])  # array to contain the 1-sigma limits [lower, upper] for the derived parameters
     sig3d  =  np.zeros([nderived,2]) 
     sig1md  =  np.zeros([nderived,2]) 
     sig3md  =  np.zeros([nderived,2]) 
     
     of.write('====================================================================================================\n')
     of.write('Jump parameters:\n')
     of.write('====================================================================================================\n')
     
     of2.write('====================================================================================================\n')
     of2.write('Jump parameters:\n')
     of2.write('====================================================================================================\n')

     of3.write('====================================================================================================\n')
     of3.write('Jump parameters:\n')
     of3.write('====================================================================================================\n')

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
         
     of.write('====================================================================================================\n')
     of.write('Additional input parameters: \n')
     of.write('====================================================================================================\n')
     vals=Rs_PDF
     # calculate median
     medval = np.median(vals)
     dval=vals-medval # the difference between vals and the median
     sval=np.sort(dval)
     sig1s = sval[i1sig] # the 1-sigma intervals (the left side is naturally negative) 
     sig3s = sval[i3sig] # the 1-sigma intervals (the left side is naturally negative) 
     of.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('Rstar', medval,sig1s[0],sig1s[1], sig3s[0], sig3s[1]))  
     vals=Ms_PDF
     # calculate median
     medval = np.median(vals)
     dval=vals-medval # the difference between vals and the median
     sval=np.sort(dval)
     sig1s = sval[i1sig] # the 1-sigma intervals (the left side is naturally negative) 
     sig3s = sval[i3sig] # the 1-sigma intervals (the left side is naturally negative) 
     of.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('Mstar', medval,sig1s[0],sig1s[1], sig3s[0], sig3s[1]))  

     for i in range(len(extinpars)):
         vals = extind_PDF[:,i]
         medval = np.median(vals)
         dval=vals-medval # the difference between vals and the median
         sval=np.sort(dval)
         sig1s = sval[i1sig] # the 1-sigma intervals (the left side is naturally negative) 
         sig3s = sval[i3sig] # the 1-sigma intervals (the left side is naturally negative) 
         of.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (extinpars[i], medval,sig1s[0],sig1s[1], sig3s[0], sig3s[1]))  


     of.write('====================================================================================================\n')
     of.write('Derived parameters: \n')
     of.write('====================================================================================================\n')
          
     for i in range(nderived):
         vals=derived_PDFs[i]
         # calculate median
         medvalsd[i] = np.median(vals)
         dval=vals-medvalsd[i] # the difference between vals and the median
         sval=np.sort(dval)
         sig1d[i] = sval[i1sig] # the 1-sigma intervals (the left side is naturally negative) 
         sig3d[i] = sval[i3sig] # the 1-sigma intervals (the left side is naturally negative) 
         of.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (derived_pnames[i], medvalsd[i],sig1d[i,0],sig1d[i,1], sig3d[i,0], sig3d[i,1])) 

     of.write('====================================================================================================\n')     
     of.write('Fixed parameters: \n')
     of.write('====================================================================================================\n')

     nfix = len(njnames)
     for i in range(nfix):
         if (njnames[i] in extinpars):
             pass
         else:
             of.write('%-25s %14.8f \n' % (njnames[i], bp[nijnames[0][i]])) 
         
     of.write('====================================================================================================\n')
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
         param_histbp(vals,jnames[i],medvals[i],sig1[i],sig3[i],maxvals[i],sig1m[i],sig3m[i],bp[ijnames[0][i]],s1bps)

     of2.write('====================================================================================================\n')
     of2.write('Additional input parameters: \n')
     of2.write('====================================================================================================\n')
     vals=Rs_PDF
     pdf, xpdf, HPDmin, iHDP = credregionML(vals)
     maxval = xpdf[iHDP]
     sig1ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
     sig1ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval
     pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
     sig3ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
     sig3ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval      
     of2.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('Rstar', maxval,sig1ms[0],sig1ms[1], sig3ms[0], sig3ms[1])) 
     param_hist(vals,'Rstar',medval,sig1s,sig3s,maxval,sig1ms,sig3ms)
     vals=Ms_PDF
     pdf, xpdf, HPDmin, iHDP = credregionML(vals)
     maxval = xpdf[iHDP]
     sig1ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
     sig1ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval
     pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
     sig3ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
     sig3ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval      
     of2.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('Mstar', maxval,sig1ms[0],sig1ms[1], sig3ms[0], sig3ms[1])) 
     param_hist(vals,'Mstar',medval,sig1s,sig3s,maxval,sig1ms,sig3ms)

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
     
     of2.write('====================================================================================================\n')
     of2.write('Derived parameters: ('+starstring+') \n')
     of2.write('====================================================================================================\n')    
     
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
         param_hist(vals,derived_pnames[i],medvalsd[i],sig1d[i],sig3d[i],maxvalsd[i],sig1md[i],sig3md[i])
     
     of2.write('====================================================================================================\n')
     of2.write('Fixed parameters: \n')
     of2.write('====================================================================================================\n')

     nfix = len(njnames)
     for i in range(nfix):
         if (njnames[i] in extinpars):
             pass
         else:
             of2.write('%-25s %14.8f \n' % (njnames[i], bp[nijnames[0][i]])) 
    
     
     of2.write('====================================================================================================\n')    
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
         
     of3.write('====================================================================================================\n')
     
     of3.write('Additional input parameters: \n')
     of3.write('====================================================================================================\n')
     vals=Rs_PDF
     pdf, xpdf, HPDmin, iHDP = credregionML(vals)
     maxval = xpdf[iHDP]
     sig1ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
     sig1ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval
     pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
     sig3ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
     sig3ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval      
     of3.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('Rstar', maxval,sig1ms[0],sig1ms[1], sig3ms[0], sig3ms[1])) 
     param_hist(vals,'Rstar',medval,sig1s,sig3s,maxval,sig1ms,sig3ms)
     vals=Ms_PDF
     pdf, xpdf, HPDmin, iHDP = credregionML(vals)
     maxval = xpdf[iHDP]
     sig1ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
     sig1ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval
     pdf, xpdf, HPDmin, iHDP = credregionML(vals,pdf=pdf, xpdf=xpdf, percentile=0.9973)
     sig3ms[0] = np.amin(xpdf[pdf>HPDmin]) - maxval
     sig3ms[1] = np.amax(xpdf[pdf>HPDmin]) - maxval      
     of3.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % ('Mstar', maxval,sig1ms[0],sig1ms[1], sig3ms[0], sig3ms[1])) 
     param_hist(vals,'Mstar',medval,sig1s,sig3s,maxval,sig1ms,sig3ms)
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

     of3.write('====================================================================================================\n')
     of3.write('Derived parameters: ('+starstring+') \n')
     of3.write('====================================================================================================\n')    
     
     for i in range(nderived):     
         
         sig1bp=sig1md[i,:] - (derived_bp[i] -maxvalsd[i])
         sig3bp=sig3md[i,:] - (derived_bp[i] -maxvalsd[i])
         
         of3.write('%-25s %14.8f %14.8f %14.8f %14.8f %14.8f\n' % (derived_pnames[i], derived_bp[i], sig1bp[0],sig1bp[1], sig3bp[0], sig3bp[1])) 
     
     of3.write('====================================================================================================\n')
     of3.write('Fixed parameters: \n')
     of3.write('====================================================================================================\n')

     nfix = len(njnames)
     for i in range(nfix):
         if (njnames[i] in extinpars):
             pass
         else:
             of3.write('%-25s %14.8f \n' % (njnames[i], bp[nijnames[0][i]])) 
     
     of3.write('====================================================================================================\n')    
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
         plot_traspec(dRpRsres, edRpRsres, ulamdas)

     
     return medvals, maxvals
 
 
def gr_print(jnames,GRvals):
    
    outfile='GRvals.dat'
    of=open(outfile,'w')
    npara=len(jnames)
    for i in range(npara):
        of.write('%-25s %10.6f \n' % (jnames[i], GRvals[i]))
         
    of.close()

def get_BIC(npar,ndat):
    
    chi2 = float(subprocess.check_output("cat MCMC.log | grep Best-parameter | awk '{print $3}' ", shell=True))
    BIC = chi2 + npar * np.log(ndat)
    RCHI = chi2 /(ndat-npar)
    
    outfile='BIC.dat'
    of=open(outfile,'w')
    of.write('data points: ')
    of.write('%10.0f \n' % (ndat))
    of.write('free parameters: ')
    of.write('%10.0f \n' % (npar))
    of.write('BIC            reduced CHI2\n')
    of.write('%10.3f %10.2f \n' % (BIC, RCHI))
    of.close()
    
    return BIC


def get_BIC_emcee(npar,ndat,chi2):
    
    BIC = chi2 + npar * np.log(ndat)
    RCHI = chi2 /(ndat-npar)
    
    outfile='BIC.dat'
    of=open(outfile,'w')
    of.write('data points: ')
    of.write('%10.0f \n' % (ndat))
    of.write('free parameters: ')
    of.write('%10.0f \n' % (npar))
    of.write('BIC            reduced CHI2\n')
    of.write('%10.3f %10.2f \n' % (BIC, RCHI))
    of.close()
    
    return BIC

def get_AIC_emcee(npar,ndat,chi2):
    
    AIC = chi2 + npar * ndat *2. / (ndat - npar -1.)
    
    outfile='AIC.dat'
    of=open(outfile,'w')
    of.write('data points: ')
    of.write('%10.0f \n' % (ndat))
    of.write('free parameters: ')
    of.write('%10.0f \n' % (npar))
    of.write('AIC \n')
    of.write('%10.3f \n' % (AIC))
    of.close()
    
    return AIC

def get_PDF_Gauss(cen,sig1,sig2,dim):
    
    sig = (sig1 + sig2)/2.
    npoints = np.int(dim[0])  # number of samples needed
    val_PDF = np.random.normal(cen, sig, np.int(npoints))
   
    return val_PDF

def derive_parameters(filnames, Rs_PDF, Ms_PDF, RpRs_PDF, Period_PDF, b_PDF, dur_PDF, ecosw_PDF, esinw_PDF, K_PDF, c1_PDF, c2_PDF, howstellar):
    
    import scipy.constants as cn

    au = 149597870700. # IAU value
    
    secondsperday = 24.*3600.
    
    Rsolar = 6.957e8   # IAU value
    Rjup = 7.1492e7    # IAU value
    Rearth = 6.3781e6  # IAU value
    
    GMsolar = 1.3271244e20 # IAU value
    GMjup = 1.2668653e17   # IAU value
    GMearth = 3.986004e14  # IAU value
    
    Msolar = GMsolar / cn.G # IAU suggestion
    Mjup = GMjup / cn.G     # IAU suggestion
    Mearth = GMearth / cn.G # IAU suggestion
    
    rhoSolar = 3. / (4. * np.pi) * Msolar / Rsolar**3 
    rhoJup = 3. / (4. * np.pi) * Mjup / Rjup**3 
    rhoEarth = 3. / (4. * np.pi) * Mearth / Rearth**3 
        
    Rp_PDF = RpRs_PDF * (Rs_PDF * Rsolar) / Rjup
    dF_PDF = RpRs_PDF**2
    ecc_PDF = ecosw_PDF**2 + esinw_PDF**2
    ome_PDF = np.arctan(np.abs(esinw_PDF / ecosw_PDF))
    
    ome_PDF[(esinw_PDF<0) & (ecosw_PDF<0)] = ome_PDF[(esinw_PDF<0) & (ecosw_PDF<0)] + np.pi
    ome_PDF[(esinw_PDF<0) & (ecosw_PDF>0)] = 2. * np.pi - ome_PDF[(esinw_PDF<0) & (ecosw_PDF>0)]
    ome_PDF[(esinw_PDF>0) & (ecosw_PDF<0)] = np.pi - ome_PDF[(esinw_PDF>0) & (ecosw_PDF<0)]
       
    e0ind = np.where(ecc_PDF<1e-15)   # avoid NaNs for very small eccentricity
    ome_PDF[e0ind] = 0.               # by defining omeaga == 0
    
    efac1_PDF=np.sqrt(1.-ecc_PDF**2)/(1.+ ecc_PDF*np.sin(ome_PDF))
    efac2_PDF=b_PDF*(1.-ecc_PDF**2)/(1.+ecc_PDF*np.sin(ome_PDF))
    efac3_PDF=np.sqrt(1.-ecc_PDF**2)/(1.- ecc_PDF*np.sin(ome_PDF))

    if (howstellar == 'Rrho'):
        rhoS_PDF = 3. * np.pi / ((Period_PDF*3600.*24.)**2 * cn.G) * aRs_PDF**3 
        Ms_PDF = (4. * np.pi) / 3. * rhoS_PDF * (Rs_PDF * Rsolar)**3 / Msolar
        starstring = 'stellar Mass from R+rho'
        
    elif (howstellar == 'Mrho'):
        rhoS_PDF = 3. * np.pi / ((Period_PDF*3600.*24.)**2 * cn.G) * aRs_PDF**3
        Rs_PDF = ((3. * Ms_PDF * Msolar) / (4. * np.pi * rhoS_PDF))**(1./3.) / Rsolar
        starstring = 'stellar Radius from M+rho'
    
    else:
        rhoS_PDF = Ms_PDF / Rs_PDF**3 * rhoSolar
        starstring = 'rho_s from stellar parameter input'
        
        
    rhoS_PDF = rhoS_PDF / rhoSolar

    ome_PDF = ome_PDF * 180. / np.pi

    Mp_PDF = (Period_PDF*3600.*24. / (2.*np.pi*cn.G))**(1./3.) * K_PDF*1000 * (Ms_PDF*Msolar)**(2./3.) * np.sqrt(1. - ecc_PDF**2) / Mjup
    
    test = (1.+RpRs_PDF)**2 - efac2_PDF**2 * (1.-(np.sin(dur_PDF*np.pi/Period_PDF))**2)
    
    if (np.min(test) > 0):    
        aRs_PDF = np.sqrt(((1.+RpRs_PDF)**2 - efac2_PDF**2 * (1.-(np.sin(dur_PDF*np.pi/Period_PDF))**2))/(np.sin(dur_PDF*np.pi/Period_PDF))**2 * efac1_PDF)

    else:
        aRs_PDF = ( (Period_PDF * secondsperday / (2. * np.pi))**2 * cn.G*(Ms_PDF * Msolar + Mp_PDF * Mjup) )**(1./3.) / (Rs_PDF * Rsolar)
        
    a_PDF = aRs_PDF * Rs_PDF * Rsolar / au

    Rsa_PDF = 1./aRs_PDF
    MF_PDF = Period_PDF* secondsperday * (K_PDF*1000)**3 / (2. * np.pi * cn.G) / Msolar * (1. - ecc_PDF**2)**(3/2) # mass function in Solar mass
    #FWHM_PDF = Rsa_PDF * np.sqrt(1. - b_PDF**2)/np.pi

    if (np.any(np.isfinite(np.sqrt(1. - b_PDF**2)))==False):
        FWHM_PDF = np.zeros(len(b_PDF))
        
    else:
        FWHM_PDF = Rsa_PDF * np.sqrt(1. - b_PDF**2)/np.pi
        
    inc_PDF = np.arccos(b_PDF / aRs_PDF) * 180. / np.pi
    
    rhoP_PDF = Mp_PDF / Rp_PDF**3 
    gP_PDF =  cn.G * (Mp_PDF*Mjup) / (Rp_PDF*Rjup)**2 
    
    durocc_PDF = Period_PDF / np.pi * np.arcsin(1. / aRs_PDF * np.sqrt( (1. + RpRs_PDF)**2 - b_PDF**2 ) / (np.sin(inc_PDF)) ) * efac3_PDF 
    durocc_PDF[np.isfinite(durocc_PDF)==False] = 0.
    
    if len(c1_PDF.shape)<2:
        nfil = c1_PDF.shape
        pnames_LD = []
        LD_PDFs = []
        #print(nfil[0])
        for i in range(nfil[0]):
            u1 = (c1_PDF[i] + c2_PDF[i]) / 3.
            u2 = (c1_PDF[i] - 2. * c2_PDF[i]) / 3.
            name1 = filnames[i]+'_u1'
            name2 = filnames[i]+'_u2'
            pnames_LD = pnames_LD + [name1] + [name2]
            LD_PDFs.append(u1)
            LD_PDFs.append(u2)
    else:
        npo, nfil = c1_PDF.shape
        u1_PDF = np.zeros((npo,nfil))
        u2_PDF = np.zeros((npo,nfil))
        pnames_LD = []
        LD_PDFs = []
    
        for i in range(nfil):
            u1 = (c1_PDF[:,i] + c2_PDF[:,i]) / 3.
            u2 = (c1_PDF[:,i] - 2. * c2_PDF[:,i]) / 3.
            name1 = filnames[i]+'_u1'
            name2 = filnames[i]+'_u2'
            pnames_LD = pnames_LD + [name1] + [name2]
            LD_PDFs.append(u1)
            LD_PDFs.append(u2)

     
         
    derived_pnames = ["Rp_[Rjup]","Mp_[Mjup]", "rho_p_[rhoJup]", "g_p_[SI]", "dF", "aRs", "a_[au]", "rhoS_[rhoSun]", "Ms_[Msun]", "Rs_[Rsun]", "inclination_[deg]", "eccentricity", "omega_[deg]", "Occult_dur", "Rs_a", "MF_PDF_[Msun]"]
    
    derived_pnames =  derived_pnames + pnames_LD
        
    derived_PDFs = [Rp_PDF, Mp_PDF, rhoP_PDF, gP_PDF, dF_PDF, aRs_PDF, a_PDF, rhoS_PDF, Ms_PDF, Rs_PDF, inc_PDF, ecc_PDF, ome_PDF, durocc_PDF, Rsa_PDF, MF_PDF]
    derived_PDFs = derived_PDFs + LD_PDFs
        
    return derived_pnames, derived_PDFs, starstring

# TODO: - Teq_PDF (requires Teff_star)
#       - incflux_PDF (requires Teff_star)
#       - Mplanet (requires K_RV)




