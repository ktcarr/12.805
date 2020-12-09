'''Spectral analysis functions used for ENSO/MJO analysis, for class project in 12.805.
Most functions written by WHOI scientist Tom Farrar (see comments in each function).
Theo Carr, December 2020'''

import numpy as np
from scipy import fft, stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def centeredFFT(x, dt):
    """
    Computes FFT, with zero frequency in the center, and returns 
    dimensional frequency vector.
    X, freq = centeredFFT(x, dt)
    
    Parameters
    ----------
    x : numeric
        1D array to be transformed by FFT
    dt : numeric
        Time increment (used to compute dimensional freuency array)

    Returns (tuple)
    -------
    X: FFT of input x, with zero frequency in the center
    freq: Dimensional frequency vector corresponding to X

    #function [X,freq]=centeredFFT(x,dt)
    #
    # Adapted from a matlab function written by Quan Quach of blinkdagger.com 
    # Tom Farrar, 2016, 2020 jfarrar@whoi.edu
    # converted from matlab 2020
    # This code was written for MIT 12.805 class
    """
    N = len(x)
    #Generate frequency index
    if N % 2 == 0:
        m= np.arange(-N/2,N/2,1) # N even; includes start (-N/2) but not stop (+N/2)
    else:
        m= np.arange(-(N-1)/2,(N-1)/2+1,1) # N odd
    
    freq = m / (N * dt) #the dimensional frequency scale
    X = fft.fft(x)
    X = fft.fftshift(X) #swaps the halves of the FFT vector 
    return (X, freq) # Return tuple; could instead do this as dictionary or list

def band_avg(yy, num):
    '''
    Compute block averages for band averaging.

    Parameters
    ----------
    yy : np.array
        1D array to be averaged.
    num : numeric
        number of adjacent data points to average.

    Returns
    -------
    Bin-averaged version of input data, subsampled by the factor num

    # Tom Farrar, 2016, 2020 jfarrar@whoi.edu
    # This code was written for MIT 12.805 class
    '''
    yyi = 0
    for n in np.arange(0, num): # 1:num
        yyi=yy[n:-(num-n):num]+yyi;
        
    yy_avg=yyi/num
    
    return yy_avg

def confid(alpha,nu):
    """
    Computes the upper and lower 100(1-alpha)% confidence limits for 
    a chi-square variate (e.g., alpha=0.05 gives a 95% confidence interval).
    Check values (e.g., Jenkins and Watts, 1968, p. 81) are $\nu/\chi^2_{19;0.025}=0.58$
    and $\nu/\chi^2_{19;0.975}=2.11$ (I get 0.5783 and 2.1333 in MATLAB).
    
   
    Parameters
    ----------
    alpha : numeric
        Number of degrees of freedom
    nu : numeric
        Number of degrees of freedom

    Returns (tuple)
    -------
    lower: lower bound of confidence interval
    upper: upper bound of confidence interval

    # Tom Farrar, 2020, jfarrar@whoi.edu
    # converted from matlab 2020
    # This code was written for MIT 12.805 class
    """
    upperv=stats.chi2.isf(1-alpha/2,nu)
    lowerv=stats.chi2.isf(alpha/2,nu)
    lower=nu / lowerv
    upper=nu / upperv
    
    return lower, upper # Return tuple; could instead do this as dictionary or list

def autospec_density_bandavg(y, dt, M, alpha=.05, xloc=1e-2, yloc=1e1, make_plot=True, logplot=True):
    '''Get one-side band-averaged estimate of autospectral density, and plot results.
    Args: y is vector ot be transformed
          dt is time increment
          M is number of bands to average
          alpha is significance level for the confidence interval
          xloc and yloc are locations for the error bar on the plot
          make_plot is boolean that determines whether to make a plot
          logplot is a boolean that determines whether to use log scale on the plot
    Returns: dictionary in the form:
            {Y: band-averaged spectrum,
             freq: band-averaged frequency vector,
             lb: lower confidence limit
             ub: upper confidence limit
             fig:fig, # corresponding to handles for plot
             ax: ax}
            Also plots the power spectrum on log-log scale'''
    
    N = len(y)
    T = dt*N
    
    # 1) remove mean from timeseries
    y -= y.mean(axis=0) # subtract mean from time series
    
    # 2) Taper
    
    # 3/4) FFT & freq vector (get rid of frequencies less than 0)
    Y, freq = centeredFFT(y, dt) # centered FFT of the time series
    Y       = Y[freq>0]
    freq    = freq[freq>0]
    
    # 5) raw one-sided spectral density
    Y = (2*T / N**2) * np.conj(Y) * Y
    
    # 6) band averaging
    Y    = band_avg(Y, M)
    freq = band_avg(freq, M)
    
    # discard imaginary components due to numerical error
    if ~np.iscomplex(y).any():
        Y = np.real(Y)
        
    ###### Confidence limits and plotting #####
    nu = 2*M # DOF
    lb, ub = confid(alpha=alpha,nu=nu)

    # Plot
    if make_plot:
        fig,ax=plt.subplots()
        if logplot:
            ax.loglog(freq, Y)
            ax.errorbar(x=xloc,y=yloc, xerr=None, yerr=yloc*np.array([lb,ub])[:,None],
                   fmt='.',color='orange', capsize=4, label='95% confidence interval')
        else:
            ax.plot(freq,Y)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'Variance / Hz')
        ax.legend()
    else:
        fig=None
        ax=None
    
    # return results as dictionary
    return {'Y':Y, 'freq':freq, 'lb':lb, 'ub':ub, 'fig':fig, 'ax':ax}