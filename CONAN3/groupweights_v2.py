import sys
import numpy as np
import matplotlib.pyplot as plt

def grweights(earr,indlist,grnames,groups,ngroup,nphot):
    # calculate the total of the error in each groups
    
    #ewarr = np.zeros(len(indlist[nphot-1][0]))
    #for j in range(len(ewarr)):
    #    ewarr[j]=earr[j]
        
    ewarr=np.copy(earr)  # error weight array
    
    for j in range(ngroup):
        jj=j+1
        # select LCs belonging to this group
        ind = np.where(np.array(groups) == jj)[0]
        nlc=len(ind)            # number of lcs in this group
        nplc=len(indlist[ind[0]][0])      # number of lc points
        errsum=np.zeros(nplc)   # this will contain the error sum for the group
        print(jj)
        # sum the errors up at each time step
        for k in range(nlc):
            if (len(indlist[ind[k]][0]) != nplc):
                print('group LCs dont have the same number of points')
            errsum=errsum+np.divide(1,np.power(earr[indlist[ind[k]][0]],2))

    # calculate the weights for each lightcurve

        for k in range(nlc):
            print(len(indlist[ind[k]][0]))
            print('ping')
            print(len(indlist[ind[k]][0]))
            print(len(ewarr))
            print(len(ewarr[indlist[ind[k]][0]]))
            print(len(errsum))
            ewarr[indlist[ind[k]][0]]=np.power(ewarr[indlist[ind[k]][0]],2)*errsum
     
 #   print np.divide(1.,ewarr[indlist[0]]), np.divide(1.,ewarr[indlist[1]]), np.divide(1.,ewarr[indlist[2]])
 #   print nothing
    return(ewarr)

    
