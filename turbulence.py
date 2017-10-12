#!/usr/bin/env python2
"""
Tools for statistical analysis of turbulence fields
===================================================

Created by Eliot Quon (eliot.quon@nrel.gov)

"""

def autocorrelation(n_x,n_y,r,xcoord,t,U,Uprime): 
    """Performs a spatial autocorrelation on a planar set of velocity data (for an x-range r)
    at time t and computes the integral length scale of the eddies in the flow at time t

    Written by Tyler Ambrico (tyler.ambrico@stonybrook.edu)
    """
    N=n_x*n_y
    Uavg=0.0
    Uprime=Uprime[:,:]
    Uavg=np.sum(U[:,:,0,t])/N
    Upsort=np.zeros((len(Uprime[:,0]),len(Uprime[0,:])))

    #r=n/2 #amount of domain sampled in x

    uprsqavg=0 #planar average of u for domain sized r*n (r in x, n in y)
    uprsqavg=np.sum(Uprime[:,0:r]**2) #calculate planar average of squared velocity for rth of the domain being sampled
    uprsqavg=uprsqavg+np.sum(Uprime[:,n_x-r:n_x]**2)

    f_r=np.zeros((n_x-r))
    f_r1=np.zeros((n_x-r))
    f_r2=np.zeros((n_x-r))

    j=0
    i=0

    for i in range(n_x-1,0,-1): 
        Upsort[:,j]=Uprime[:,i]
        j+=1

    for k in range(0,n_x-r):
        f_r1[k]=np.sum(Uprime[:,0:r]*Uprime[:,k:r+k])
        f_r2[k]=np.sum(Upsort[:,0:r]*Upsort[:,k:r+k])

    f_r=f_r1+f_r2

    f_r=f_r/uprsqavg

    xsmoo, f_r_smoo = smooth_f_r(f_r,xcoord,n_x,r,tol_monotonic=0.005)
    L11=scipy.integrate.simps(f_r_smoo,xsmoo) #get lengthscale at time t

    return f_r, L11, Uavg, xsmoo

def autocorrelation_st(n_x,n_y,r,Ntwindow,w,xcoord,U,Uprime,uprsqtavg_running,f_r_running,Nt_running,LB=0,RB=None):
    """
    Calculates average length scale using spatial averaging approach, but also averages over all timesteps

    Written by Tyler Ambrico (tyler.ambrico@stonybrook.edu)
    """
    
    if RB is None: RB=n_x
   
    uprsqtavg=0 #planar average of u for domain sized r*n (r in x, n in y)

    Uprime_sample=np.copy(Uprime[:,LB:RB,:,:]) #arrays get actively overwritten if you only use a simple assignment
    n_x=len(Uprime_sample[0,:,0,0])

    Upsort=np.copy(Uprime_sample)

    uprsqtavg=np.sum(Uprime_sample[:,0:r,0,w:w+Ntwindow]**2) #calculate planar average of squared velocity for left half of the domain
    uprsqtavg_running+=np.sum(Uprime_sample[:,0:r,0,w:w+Ntwindow]**2)

    uprsqtavg+=np.sum(Uprime_sample[:,n_x-r:n_x,0,w:w+Ntwindow]**2) #calculate planar average of squared velocity for right half of the domain
    uprsqtavg_running+=np.sum(Uprime_sample[:,n_x-r:n_x,0,w:w+Ntwindow]**2)
                
    f_r=np.zeros((n_x-r))
    f_r1=np.zeros((n_x-r))
    f_r2=np.zeros((n_x-r))
    f_r1_running=np.zeros((n_x-r))
    f_r2_running=np.zeros((n_x-r))
    
    j=0
    for i in range(n_x-1,0,-1): 
        Upsort[:,j,0,:]=Uprime_sample[:,i,0,:]
        j+=1

    for k in range(0,n_x-r):
        f_r1[k]=np.sum(Uprime_sample[:,0:r,0,w:w+Ntwindow]*Uprime_sample[:,k:r+k,0,w:w+Ntwindow])
        f_r1_running[k]+=np.sum(Uprime_sample[:,0:r,0,w:w+Ntwindow]*Uprime_sample[:,k:r+k,0,w:w+Ntwindow])
        f_r2[k]=np.sum(Upsort[:,0:r,0,w:w+Ntwindow]*Upsort[:,k:r+k,0,w:w+Ntwindow])
        f_r2_running[k]+=np.sum(Upsort[:,0:r,0,w:w+Ntwindow]*Upsort[:,k:r+k,0,w:w+Ntwindow])
          
    f_r=f_r1+f_r2
    f_r_running=f_r1_running+f_r2_running

    f_r=f_r/uprsqtavg
    xsmoo, f_r_smoo = smooth_f_r(f_r,xcoord,n_x,r,tol_monotonic=0.005)
    L11=scipy.integrate.simps(f_r_smoo,xsmoo) #get lengthscale at time t

    return f_r, L11,f_r_running,uprsqtavg_running, xsmoo


    def smooth_f_r(f,x,n_x,r,tol_monotonic=0.005):
        """
        Smoothes the autocorrelation function f_r(r)
        Written by Eliot Quon (eliot.quon@nrel.gov)
        """
        x=x[0:n_x-r]

        def smooth(N):
            xsmoo = x[N/2:-N/2+1]
            fsmoo = np.convolve(f, np.ones(N)/N, mode='valid')
           #ax.plot(xsmoo,fsmoo,label='N={:d}'.format(N))
            return xsmoo,fsmoo

        def is_non_increasing(fs):
            return np.all(np.diff(fs) <= tol_monotonic)

        # repeatedly smooth until we have a ~monotonically decreasing curve
        for N in range(10,len(x),10):
            xs,fs = smooth(N)
            if is_non_increasing(fs): break

        # find cutoff
        idx = np.nonzero(f <= fs[-1])[0][0]
        
        return x[:idx], f[:idx]
