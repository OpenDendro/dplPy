import numpy as np #are there issues here? seeing some on my end
import pandas as pd #same, not sure these are needed since they are in the dependancies file anyway

# Date: 07/18/2024
# Author: Anne Wilce
# Title: agedepspline.py
# Description: Applies an age-dependent smoothing spline to y.


def ads_R2Py(y, nyrs0=50, pos_slope=True):
    nobs = len(y)
    nyrs = np.arange(1, nobs + 1) + nyrs0 - 1
    ySpl = np.zeros(nobs)
    
    
    if nobs < 3:
        raise ValueError("there must be at least 3 data points")
    if nobs > 10000:
        raise ValueError("y shouldn't be longer than 1e4. ask for help. the f77 code will need to be recompiled.")
    if not isinstance(nyrs0, int) or nyrs0 <= 1:
        raise ValueError("'nyrs0' must be an integer greater than 1")
    # Use 1-based indexing to better match the FORTRAN/R code and avoid confusion
    # Row/Col 0 exisits here but is not used and does not affect the result due to current index handling 
    def ads95_inPy(y, n, stiffness):
        nm2 = n - 2
        pi = np.pi
        c1 = [0, 1, -4, 6, -2] #add zero in idx 0 for future mirror index of R
        c2 = [0, 0, 1/3, 4/3] #add zero in idx 0 for future mirror index of R
        
        # Convert stiffness vector to p vector
        p = np.zeros(nm2+1)
        
        for i in range(1,nm2+1): #use R indexing
            v = stiffness[i]
            arg = (2 * pi) / v
            p[i] = (6 * (np.cos(arg) - 1)**2) / (np.cos(arg) + 2)
            #p[i]= round(p[i], 10)
        
        # Initialize arrays
        a = np.zeros((nm2+1, 5))
        #print(a.shape)
        res = np.zeros(n+1) #maybe change here AMW
        y = np.insert(y, 0, 0)

        # Fill a array
        for i in range(1,nm2+1):
            for j in range(1,4):
                a[i, j] = c1[j] + p[i] * c2[j]
            a[i, 4] = y[i] + c1[4] * y[i + 1] + y[i + 2]
        
        a[1, 1] = c2[1]
        a[1, 2] = c2[1]
        a[2, 1] = c2[1]
        
        nc = 2
        rn = 1 / (nm2 * 16)
        
        d1 = 1
        d2 = 0
        ncp1 = nc + 1
        

        # Luelpb section
        for i in range(1,nm2+1):
            #print("Working on i number", i)
            imncp1 = i - ncp1 

            i1 = max(1, 1 - imncp1)

            for j in range(i1, ncp1+1):
                l = imncp1 + j

                i2 = ncp1 - j 

                sum_val = a[i, j]

                jm1 = j-1 

                if jm1 > 0:
                    for k in range(1,jm1+1):
                        m = i2 + k 
                        sum_val -= a[i, k] * a[l, m]

                if j == (ncp1): 
                    if a[i, j] + sum_val * rn <= a[i, j]:
                        res[0] = -9999
                        return res
                    a[i, j] = 1 / np.sqrt(sum_val)

                    d1 *= sum_val
                    while abs(d1) > 1:
                        d1 *= 0.0625
                        d2 += 4
                    while abs(d1) <= 0.0625:
                        d1 *= 16
                        d2 -= 4
                else:
                    a[i, j] = sum_val * a[l, ncp1] 
                    
                    
        # Luelpb section
        nc1 = nc + 1
        iw = 0
        l = 0
        for i in range(1,nm2+1):
            sum_val = a[i, 4]
            if nc > 0:
                if iw != 0:
                    l = l + 1
                    if l > nc:
                        l = nc
                    k = nc1 - l
                    kl = i - l
                   
                    for j in range(k, nc+1):
                        sum_val = sum_val - a[kl, 4] * a[i, j]
                        kl = kl + 1
                elif sum_val != 0:
                    iw = 1
            a[i, 4] = sum_val * a[i, nc1]
        a[nm2, 4] = a[nm2, 4] * a[nm2, nc1]
        
        n1 = nm2 + 1
        for i in range(2, nm2+1):
            k = n1 - i
            sum_val = a[k, 4]
            if nc > 0:
                kl = k + 1
                k1 = min(nm2, k + nc)
                l = 1
                for j in range(kl, k1+1):
                    sum_val = sum_val - a[j, 4] * a[j, nc1 - l]
                    l = l + 1
            a[k, 4] = sum_val * a[k, nc1]
        
        for i in range(3, nm2+1):
            res[i] = a[i - 2, 4] + c1[4] * a[i - 1, 4] + a[i, 4]

        res[1] = a[1, 4]
        res[2] = c1[4] * a[1, 4] + a[2, 4]
        res[n - 1] = a[nm2 - 1, 4] + c1[4] * a[nm2, 4]
        res[n] = a[nm2, 4]
        
        for i in range(1,n+1):
            res[i] = y[i] - res[i]
        res = res[1:]
        return res
    
    ySpl = ads95_inPy(y, n=nobs, stiffness=nyrs)

    if not pos_slope:
        ySplDiff = np.diff(ySpl, prepend=0)
        ySplCutoff = np.max(np.where(ySplDiff <= 0)[0])
        ySpl[ySplCutoff:nobs] = ySpl[ySplCutoff]
        ySpl = ads95_inPy(ySpl, n=nobs, stiffness=nyrs)
    
    return ySpl