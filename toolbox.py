'''Functions used for ENSO/MJO analysis, for class project in 12.805
Theo Carr, December 2020'''

import xarray as xr
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from numpy.linalg import inv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

##################################################################################################
#####   Functions written for this project ###
##################################################################################################

####### Data analysis functions (e.g. least squares, significance testing, etc.)
def get_autocov(x, shift=False):
    '''Function to get autocovariance of timeseries (wrapper for numpy function)'''
    if shift:
        x0 = np.correlate(x, x, mode='full')[len(x)-1:]
    else:
        x0 = np.correlate(x, x, mode='same')
    return x0 / len(x)

def get_autocorr(x, shift=False):
    '''Get autocorrelation of timeseries (i.e normalize autocovariance by variance)'''
    return get_autocov(x, shift=shift) / np.var(x)

def get_Cxx(x):
    '''Create autocovariance matrix from autocovariance of a time series'''
    return toeplitz(get_autocov(x, shift=True))

def get_confid(x, alpha=.05, n=500):
    '''Get 95% confidence interval for correlation based on autocorrelation of the timeseries
    alpha is confidence level
    n is number of random samples to draw'''
    Cxx = get_Cxx(x)
    eps = np.finfo(np.float32).eps
    L = np.linalg.cholesky(Cxx + eps* np.identity(Cxx.shape[0]))
    x_rand = np.random.randn(L.shape[0], n)
    x_b = L @ x_rand # generate random time series with same autocorrelation
    rho_vals = np.array([np.correlate(x, x_b[:,i]).item() for i in range(n)]) # get covariance for each sample
    rho_vals /= (np.std(x_b, axis=0) * np.std(x) * len(x)) # normalize to get correlation
    lb, ub = np.percentile(rho_vals, 100*np.array([alpha/2, 1-(alpha/2)])) # lower/upper bounds
    return lb, ub

def mc_test_composite(data, n_samples, n_sims, alpha=.05):
    '''Get upper and lower bounds for Monte-Carlo composite'''
    precip_djf = get_djf(data)
    res = np.zeros([n_sims, len(precip_djf.latitude), len(precip_djf.longitude)])
    for i in range(n_sims):
        idx = np.random.randint(len(precip_djf.time),size=n_samples)
        res[i,:] = precip_djf.isel(time=idx).mean(dim='time') # make random selections and take mean
    
    # Get confidence limits
    res = np.percentile(res, 100*np.array([alpha/2, 1-alpha/2]), axis=0)
    lb = res[0,:]
    ub = res[1,:]
    return lb, ub

def get_rho(x1, x2):
    '''Function to get correlation coefficient for 2 timeseries'''
    x = np.stack([x1,x2], axis=1)
    cov = (x.T @ x) / (len(x)-1)
    rho = cov[0,1] / np.prod(np.sqrt(np.diag(cov)))
    return rho
def get_lagged_autocorr(x, lags=np.arange(1,26)):
    '''Get lagged autocorrelation for time series, and reflect about t=0'''
    rho = []
    for lag in lags:
        x1 = x[:-lag]
        x2 = x[lag:]
        rho.append(get_rho(x1,x2))
    rho = list(reversed(rho))+ [1.] + rho # reflect about y-axis
    return rho
def get_lagged_corr(x1, x2, lags=np.arange(-25,26)):
    '''Get lagged correlation for two time series'''
    rho = []
    for lag in lags:
        if lag==0:
            y1 = x1
            y2 = x2
        elif lag<0:
            y1 = x1[-lag:]
            y2 = x2[:lag]
        else:
            y1 = x1[:-lag]
            y2 = x2[lag:]
        rho.append(get_rho(y1,y2))
    return rho

def get_ls_mats(data, lead=14):
    '''Get matrices for least squares fitting, with specified lead time
    data is dictionary of form {'features':DataArray, 'out':DataArray}
        - each of 'features' and 'out' is a DataArray
    lead is an integer specifying number of months between last feature variable 
        and first day of the period to be predicted'''
    E = data['features'].isel(time=slice(None,-lead)).values
    y = data['out'].isel(time=slice(lead,None)).values
    y_per = data['out'].isel(time=slice(None,-lead)).values
    return E, y, y_per

def add_constant(x):
    '''Add a column of ones to matrix, representing constant coefficient for mat. mult.'''
    x = np.append(x, np.ones([x.shape[0],1]), axis=1)
    return x
    
def ls_fit(E,y, x0=None, gamma=.1):
    '''Function solves least-squares to get coefficients for linear projection of E onto Y
    x0 is first guess to use (taper method)
    gamma is weight for the taper
    returns x, a vector of the best-fit coefficients'''
    # Add array of ones to feature vector (to represent constant term in linear model)
    E = add_constant(E)
    if x0 is not None: # use tapered method
        S = gamma * np.identity(E.shape[1])
        coef = inv(E.T @ E + inv(S)) @ (E.T @ y + inv(S) @ x0)
    else: # don't use taper
        coef = inv(E.T @ E) @ E.T @ y
    return coef

def ls_eval(x, E):
    '''Function multiplies coefficients (x) by E matrix'''
    E = add_constant(E)
    return E @ x

def mse(y, y0):
    '''Compute mean square error
    Return a) MSE for each grid point
           b) total MSE'''
    by_loc = np.mean((y-y0)**2, axis=0)
    total  = np.mean(by_loc)
    return by_loc, total

def corr(x,y,axis=0):
    '''Get correlation between two arrays of arbitrary shape. Get correlation along specified axis'''
    eps = np.finfo(np.double).eps # machine epsilon (to avoid division by zero)
    cov = np.mean((x-np.mean(x,axis=axis)) * (y-np.mean(y,axis=axis)), axis=axis)
    return cov / (np.std(x,axis=axis)*np.std(y,axis=axis) + eps)

def get_sig(corr_):
    '''Get significant correlations by comparing to 95% confidence threshold'''
    return (corr_< rho_sig_test[:,0]) | (corr_ > rho_sig_test[:,1])

######################### Miscellaneous ######################
def prep(x):
    '''Function removes trend and seasonal cycle from xarray variable'''
    y = x - get_trend_fast(x)
    y = y.groupby('time.month') - y.groupby('time.month').mean()
    return y

def prep_mjo_data(x):
    '''Function to pre-process data associated with the MJO
    Specifically: 1. remove linear relationship with ENSO
                  2. remove 120 day running mean
                  3. average over latitudes, and normalize by global variance'''
    print( 'Removing linear relationship with ENSO...')
    x_mjo = x - get_proj_xr(oni_daily, x) # remove relationship with ENSO
    print('Removing 120-day running mean...')
    x_mjo = x_mjo.isel(time=slice(119,None)) - \
        x_mjo.rolling(time=120).mean().isel(time=slice(119,None)) # 120-day mean
    x_mjo = x_mjo.mean(dim='latitude')
    x_mjo = x_mjo / x_mjo.std()
    return x_mjo

def get_mjo_phase(x, y):
    '''Function to get the phase of the MJO, based on first two PCs from Wheeler/Hendon index'''
    mag = np.sqrt(x**2 + y**2)
    if mag < 1: # only classify if magnitude is greater than 1
        return -1
    else:
        if x>=0:
            if y>=0:
                return 5. if np.abs(x)>=np.abs(y) else 6
            else:
                return 4 if np.abs(x)>=np.abs(y) else 3
        else:
            if y>=0:
                return 8 if np.abs(x)>=np.abs(y) else 7
            else:
                return 1 if np.abs(x)>=np.abs(y) else 2
            return
        return
    return

def get_mjo_phase_(data):
    '''Helper function: takes single argument, allowing us to pass it to np.apply_along_axis'''
    x,y = data
    return get_mjo_phase(x,y)

def get_per(data, lead=14):
    '''Get persistence prediction given specified lead time'''
    return data['out'].isel(time=slice(None,-lead)).values

def get_season_bools(E):
    djf = (E.month==12) | (E.month==1) | (E.month==2)
    mam =  (E.month==3) | (E.month==4) | (E.month==5)
    jja = (E.month==6) | (E.month==7) | (E.month==8)
    son = (E.month==9) | (E.month==10) | (E.month==11)
    return {'djf':djf, 'mam':mam, 'jja':jja, 'son':son}

def get_djf(data):
    '''Get DJF months from data array'''
    d = data.time
    d = d[(d.dt.month == 12) | (d.dt.month == 1) | (d.dt.month == 2)]
    return data.sel(time=d)

##################################################################################################
##### Functions borrowed from previous projects: functions for plotting and efficient pre-processing ###
##################################################################################################

def plot_setup(plot_range = [-125.25,-66,22.5,50], figsize = (7,5), central_lon=0):
#     Function sets up plotting environment for continental US
#     Returns fig, ax

    # Set up figure for plotting
    fig = plt.figure(figsize = figsize)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_lon))
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    ax.add_feature(states_provinces, edgecolor='black')
    ax.coastlines()
    ax.set_extent(plot_range, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS)
    ax.title.set_fontsize(30)  
    return fig, ax


def multis_to_datetime(multis):
#     Function to convert multi-index of year/month to Pandas datetime index
    multi_to_datetime = lambda multi : datetime(multi[0],multi[1],1)
    return(pd.Index([multi_to_datetime(multi) for multi in multis]))

def unstack_month_and_year(data):
    '''Function 'unstacks' month and year in a dataframe with 'time' dimension
     The 'time' dimension is separated into a month and a year dimension
     This increases the number of dimensions by 1'''
    year = data.time.dt.year
    month = data.time.dt.month
    new_idx = pd.MultiIndex.from_arrays([year,month],names=('year','month'))
    data = data.assign_coords({'time':new_idx}).unstack('time')
    return data

def stack_month_and_year(data):
    '''Function takes datarray that has year and month as separate dimensions, and stacks them to form time dimension
    Args: data is a dataarray with month and year as separate dimensions
    Returns: datarray with one fewer dimension than input (year and month stacked to form time)'''
    data = data.stack(time=['year','month']) #### Make time a coordinate (and a datetime index)
    data =  data.assign_coords({'time':multis_to_datetime(data.time.values)}).transpose('time',...)
    idx = [] # get indices of non-NaN data (i.e., valid timeslices)
    for t in data.time:
        idx.append(~np.isnan(data.sel(time=t)).values.all())
    return data.isel(time=idx) # return data without NaNs

def get_trend_fast(data):
    '''Function performs fast detrend of the data'''
    shape = data.shape
    d = data.values.reshape(shape[0],-1)
    trend = np.zeros(d.shape, dtype=d.dtype)
    allnan_idx  = np.where(np.isnan(d).all(axis=0))[0]
    somenan_idx = np.where(np.isnan(d).any(axis=0))[0]
    somenan_idx = [i for i in somenan_idx if i not in allnan_idx]
    nonan_idx = [i for i in np.arange(d.shape[1]) if (i not in somenan_idx) and (i not in allnan_idx)]

    trend[:, nonan_idx]   = get_coefs_fast(d[:,nonan_idx])
    if len(somenan_idx) > 0:
        trend[:, somenan_idx] = res2 = np.apply_along_axis(handle_bad_series, arr=d[:,somenan_idx], axis=0)
    trend[:, allnan_idx]  = np.nan
    return trend.reshape(shape)

def get_coefs_fast(Y):
    '''Function vectorizes detrend (using einsum)'''
    n = Y.shape[0]
    x = np.stack([np.arange(n), np.ones(n, dtype=Y.dtype)],     axis=1)
    X = np.stack([x for _ in range(Y.shape[1])], axis=2)
    
    res0 = np.einsum('abn,bcn->acn', np.swapaxes(X,1,0), X)
    a = res0[0,0,:]
    b = res0[0,1,:]
    c = res0[1,0,:]
    d = res0[1,1,:]

    res1 = (1/(a*d - b*c)) * np.array([[d,-b],[-c,a]])
    res2 = np.einsum('abn,bn->an', np.swapaxes(X,1,0), Y)
    coefs = np.einsum('abn,bn->an',res1,res2)
    return np.einsum('abc,bc->ac', X, coefs)

def remove_season(data, standardize=True, mean=None, std=None, dims=None):
    '''Function to remove seasonality from data.
    Args: data is xr.array with 'time' dimension.
          standardize is boolean specifying whether to standardize the data
          mean is None or an array specifying mean to remove
          std is None or an array specifying std to divide by 
          dims are dimensions to obtain mean and std along
    Returns de-seasonalized data with same shape as input'''
    
    data = utils.unstack_month_and_year(data)
    
    if mean is None:
        if dims is None:
            dims = ['year']
        mean = data.mean(dim=dims, skipna=True) # mean
        std = replace_zero_with_eps(data.std(dim=dims, skipna=True)) # std (fill zeros with machine epsilon)
        std = data.std(dim=dims, skipna=True)
    if standardize:
        data = (data - mean) / std # avoid dividing by zero
    else:
        data = data - mean
        
    data = utils.stack_month_and_year(data)

    return data, mean, std