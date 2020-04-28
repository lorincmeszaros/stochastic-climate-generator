# -*- coding: utf-8 -*-
"""
@author: Lorinc Meszaros
Affiliation: TU Delft and Deltares, Delft, The Netherlands

Pre-processing for Gibbs sampler to:
    1. Extract seasonal shape
    2. Produce time shifts for the new scenarios
"""

#==============================================================================
#STEP 0 - Import data
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel
from scipy.signal import argrelextrema
from scipy import stats
from scipy.stats import rankdata
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import random
from scipy import interpolate
#==============================================================================
#Define functions
def seasonal_mean(x, freq):
    """
    Return means for each period in x. freq is an int that gives the
    number of periods per cycle. E.g., 12 for monthly. NaNs are ignored
    in the mean.
    """
    return np.array([np.nanmean(x[i::freq], axis=0) for i in range(freq)])

def seasonal_component(x, freq):
    """
    Tiles seasonal means (periodical avereages) into a time seris as long as the original.
    """
    nobs=len(x)
    period_averages=seasonal_mean(x, freq)
    period_averages -= np.mean(period_averages, axis=0)
    return np.tile(period_averages.T, nobs // freq + 1).T[:nobs]
#==============================================================================
#Load and initialize data
data=np.load(r'tensor_daily_mean_5D.npy')
#Replace Nan with 0
data[np.where(np.isnan(data) == True)]=0
#Reshape
data=np.append(data[:,:,:,:,0],data[:,:,:,:,1],axis=3)
#Data view 3D
data_slice=data[:,3,0,:]

#CalendarYear
calendarYear=365.00
#==============================================================================
#Initialize empty vectors
#cosine_yearly_fitted=np.zeros((400, 90,np.shape(data)[3]))
x_data_ideal_matrix=np.zeros((420, 90,np.shape(data)[3]))
x_data_ideal_cont_matrix=np.zeros((420, 90,np.shape(data)[3]))
x_data_slice_matrix=np.zeros((420, 90,np.shape(data)[3]))
y_data_slice_matrix=np.zeros((420, 90,np.shape(data)[3]))
y_data_slice_smooth_matrix=np.zeros((420, 90,np.shape(data)[3]))
y_data_slice_smooth_365_nearest_matrix=np.zeros((420, 90,np.shape(data)[3]))
x_data_ideal_1D_matrix=np.zeros((np.shape(data)[0], np.shape(data)[3]))
y_data_365_nearest_matrix=np.zeros((np.shape(data)[0], np.shape(data)[3]))

deviation_matrix=np.zeros((90,np.shape(data)[3]))
line_intercept=np.zeros((1,np.shape(data)[3]))
line_slope=np.zeros((1,np.shape(data)[3]))

residual_pattern_sqr_matrix=np.zeros((int(calendarYear), np.shape(data)[3]))
#==============================================================================
#Zero to Nan
x_data_slice_matrix[x_data_slice_matrix == 0] = np.nan
x_data_ideal_matrix[x_data_ideal_matrix == 0] = np.nan
x_data_ideal_cont_matrix[x_data_ideal_cont_matrix == 0] = np.nan
y_data_slice_matrix[y_data_slice_matrix == 0] = np.nan
y_data_slice_smooth_matrix[y_data_slice_smooth_matrix == 0] = np.nan
y_data_slice_smooth_365_nearest_matrix[y_data_slice_smooth_365_nearest_matrix == 0] = np.nan
x_data_ideal_1D_matrix[x_data_ideal_1D_matrix == 0] = np.nan
y_data_365_nearest_matrix[y_data_365_nearest_matrix == 0] = np.nan

residual_pattern_sqr_matrix[residual_pattern_sqr_matrix == 0] = np.nan
#==============================================================================
#Choose time interval by the number of timesteps
datalimit1=0
datalimit2=32872
#Initialize empty matrices
y_data_detrended_matrix=np.zeros((datalimit2, np.shape(data)[3]))
trend=np.zeros((datalimit2,np.shape(data)[3]))
residual=np.zeros((datalimit2,np.shape(data)[3]))
#Plot param
plt.rcParams['figure.figsize'] = (10,5)
#Initialize x and y
x_data=np.arange(0,datalimit2)/calendarYear
y_data_all=data[datalimit1:datalimit2,0,0,:]    
#Choose scenarios
scenarios= [0,1,2,4,5,6,7,9] #range(np.shape(data)[3])

for i in scenarios:
        y_data=data[datalimit1:datalimit2,0,0,i]    
        
        #==============================================================================       
        #STEP0 - Identify, trend, seasonality, residual
        
        result = seasonal_decompose(y_data, freq=365, model='additive')

        #Fit lineat trend
        #Get parameters: alpha and beta
        
        #Fit curve with lmfit        
        line_mod=LinearModel(prefix='line_')        
        pars_line = line_mod.guess(y_data, x=x_data)
        result_line_model=line_mod.fit(y_data, pars_line, x=x_data)
        print(result_line_model.fit_report())
        
        line_intercept[:,i]=result_line_model.params['line_intercept']._val
        line_slope[:,i]=result_line_model.params['line_slope']._val
        
        trend[:,i]=result_line_model.best_fit
        #==============================================================================
        #STEP 2
        #Remove trend       
        y_data_detrended=y_data-result_line_model.best_fit
        y_data_detrended_matrix[:,i]=y_data_detrended
        
        y_data_detrend=seasonal_component(y_data_detrended, int(calendarYear)) #seasonal component               
        #==============================================================================       
        #Smooth LOWESS
        lowess = sm.nonparametric.lowess
        data_smooth_lowess=lowess(y_data_detrended, x_data, frac=1./500, it=0, delta = 0.01, is_sorted=True, missing='drop', return_sorted=False)
                
        # for local minima
        #add reflective boundary
        #data_smooth_reflective = np.pad(data_smooth, 0, mode='reflect')
        local_min_location=np.array(argrelextrema(data_smooth_lowess, np.less, order=300, mode='wrap'))
        #local_min_location = (np.insert(local_min_location, 90,(len(x_data)-1))).reshape((1,-1))
        local_min=data_smooth_lowess[local_min_location]
                
        #distance between minima
        dist_minima=np.transpose(np.diff(local_min_location))
        
        #Plot deviations from calendar year (histogram)
        deviation=(calendarYear-dist_minima)/calendarYear
        deviation_matrix[:len(deviation),i]=deviation.flatten(order='F')
                   
        #==============================================================================
        #STEP 3
        #Chop years and Fit cosine curve
        #Get parameters: amplitude
                
        for j in range(np.shape(local_min_location)[1]-1):
            y_data_slice=y_data_detrended[np.int(local_min_location[:,j]):np.int(local_min_location[:,j+1])]
            x_data_slice=np.arange(0,len(y_data_slice))/calendarYear 
            x_data_slice_365=np.arange(0,365)/calendarYear
            #Remove time change from data
            x_data_ideal=np.linspace(0,calendarYear,len(y_data_slice))/calendarYear
            if j==0:
                x_data_ideal_cont=np.linspace((j*calendarYear),(j+1)*calendarYear,len(y_data_slice))/calendarYear
            else:
                x_data_ideal_cont=np.linspace((j*calendarYear)+1,(j+1)*calendarYear,len(y_data_slice))/calendarYear
            y_data_slice_smooth=lowess(y_data_slice, x_data_ideal, frac=1./10, it=0, is_sorted=True, missing='drop', return_sorted=False) 
            
            #Interpolate to regular step - smooth data    
            f = interpolate.interp1d(x_data_ideal, y_data_slice_smooth,kind='nearest', fill_value="extrapolate")
            y_data_slice_smooth_365_nearest= f(x_data_slice_365)   # use interpolation function returned by `interp1d`
            
            x_data_slice_matrix[:len(x_data[np.int(local_min_location[:,j]):np.int(local_min_location[:,j+1])]),j,i]=x_data_slice
            x_data_ideal_matrix[:len(x_data[np.int(local_min_location[:,j]):np.int(local_min_location[:,j+1])]),j,i]=x_data_ideal
            x_data_ideal_cont_matrix[:len(x_data[np.int(local_min_location[:,j]):np.int(local_min_location[:,j+1])]),j,i]=x_data_ideal_cont
            y_data_slice_matrix[:len(y_data_detrend[np.int(local_min_location[:,j]):np.int(local_min_location[:,j+1])]),j,i]=y_data_slice
            y_data_slice_smooth_matrix[:len(y_data_detrend[np.int(local_min_location[:,j]):np.int(local_min_location[:,j+1])]),j,i]=y_data_slice_smooth
            y_data_slice_smooth_365_nearest_matrix[:len(y_data_slice_smooth_365_nearest),j,i]=y_data_slice_smooth_365_nearest
       
        #Plot fitted all sin curve (lmfit)
        plt.figure(figsize=(12, 5))
        labels=['Data points']
        plt.plot(x_data_ideal_matrix[:,:,i], y_data_slice_matrix[:,:,i], 'bo', alpha = 0.1)
        plt.plot(x_data_ideal_matrix[:,:,i], y_data_slice_smooth_matrix[:,:,i], 'k-')
        plt.xlabel('Number of years')
        plt.ylabel('Radiation')
        plt.legend(labels)
        plt.title("Scenario: " + str(i))
        plt.show()
        
        #x_data_ideal_1D
        x_data_ideal_1D = x_data_ideal_cont_matrix[:,:,i].flatten(order='F')
        x_data_ideal_1D = x_data_ideal_1D[~np.isnan(x_data_ideal_1D)]
        
        #Interpolate to regular step      
        f = interpolate.interp1d(x_data_ideal_1D, y_data[:len(x_data_ideal_1D)],kind='nearest', fill_value="extrapolate")
        y_data_365_nearest= f(x_data[:len(x_data_ideal_1D)])   # use interpolation function returned by `interp1d`    
                
        #Save in matrix
        x_data_ideal_1D_matrix[:len(x_data_ideal_1D),i]=x_data_ideal_1D
        y_data_365_nearest_matrix[:len(y_data_365_nearest),i]=y_data_365_nearest
        
        print(i)
        #==============================================================================

#==============================================================================      
#Flatten arrays with all years into 1D vector
x_data_ideal_flattened=np.zeros((np.shape(x_data_ideal_matrix)[0], np.shape(x_data_ideal_matrix)[1]*np.shape(x_data_ideal_matrix)[2]))
y_data_slice_flattened=np.zeros((np.shape(y_data_slice_matrix)[0], np.shape(y_data_slice_matrix)[1]*np.shape(y_data_slice_matrix)[2]))        
y_data_slice_smooth_flattened=np.zeros((np.shape(y_data_slice_smooth_matrix)[0], np.shape(y_data_slice_smooth_matrix)[1]*np.shape(y_data_slice_smooth_matrix)[2]))        
y_data_slice_smooth_365_nearest_flattened=np.zeros((np.shape(y_data_slice_smooth_matrix)[0], np.shape(y_data_slice_smooth_matrix)[1]*np.shape(y_data_slice_smooth_matrix)[2]))        
for k in range(np.shape(y_data_slice_matrix)[0]):        
    x_data_ideal_flattened[k,:]=x_data_ideal_matrix[k,:,:].flatten(order='F')
    y_data_slice_flattened[k,:]=y_data_slice_matrix[k,:,:].flatten(order='F')
    y_data_slice_smooth_flattened[k,:]=y_data_slice_smooth_matrix[k,:,:].flatten(order='F')
    y_data_slice_smooth_365_nearest_flattened[k,:]=y_data_slice_smooth_365_nearest_matrix[k,:,:].flatten(order='F')


#Average smooth curve
av_loess_scenario=np.nanmean(y_data_slice_smooth_365_nearest_matrix, axis=1)
av_loess_scenario=av_loess_scenario[:365,:]

#Save av_loess_scenario
np.save(r'av_loess_scenario.npy', av_loess_scenario)

#Average smoothed
av_loess=np.nanmean(y_data_slice_smooth_365_nearest_flattened, axis=1)
av_loess=av_loess[~np.isnan(av_loess)]

#==============================================================================
#All parameters
deviation_flattened=deviation_matrix.flatten(order='F')
line_intercept_flattened=line_intercept.flatten(order='F')
line_slope_flattened=line_slope.flatten(order='F')
#Remove Nans
deviation_flattened= deviation_flattened[~np.isnan(deviation_flattened)] 

#Remove zeros
#amplitude_loess_flattened= amplitude_loess_flattened[np.nonzero(amplitude_loess_flattened)] 
deviation_flattened= deviation_flattened[np.nonzero(deviation_flattened)] 
line_intercept_flattened=line_intercept_flattened[np.nonzero(line_intercept_flattened)] 
line_slope_flattened=line_slope_flattened[np.nonzero(line_slope_flattened)] 

#Analyse parameter change
#Parameter statistics
#deviation
deviation_mean=np.mean(deviation_flattened)
deviation_std=np.std(deviation_flattened)
print('deviation - mean:',np.round(deviation_mean,decimals=4), 'std:',np.round(deviation_std, decimals=4))

#Deviation per scenario
deviation_mean_sc=np.mean(deviation_matrix, axis=0)
deviation_std_sc=np.std(deviation_matrix, axis=0)

#Remove zeros
deviation_mean_sc= deviation_mean_sc[np.nonzero(deviation_mean_sc)] 
deviation_std_sc= deviation_std_sc[np.nonzero(deviation_std_sc)] 

#Line intercept
line_intercept_mean=np.mean(line_intercept_flattened)
line_intercept_std=np.std(line_intercept_flattened)
print('line_intercept - mean:',np.round(line_intercept_mean,decimals=2), 'std:',np.round(line_intercept_std, decimals=2))

#Line slope
line_slope_mean=np.mean(line_slope_flattened)
line_slope_std=np.std(line_slope_flattened)
print('line_slope - mean:',np.round(line_slope_mean,decimals=2), 'std:',np.round(line_slope_std, decimals=2))

#==============================================================================
#Combine LOESS curves
new_loess=[]
def new_loess_func():
    for l in np.arange(85):
        if l==0:
            new_loess = lowess(av_loess[:365], np.squeeze(np.linspace(0, dist_minima[l], num=len(av_loess[:365]))/calendarYear), frac=1./10, it=0, is_sorted=True, missing='drop', return_sorted=False) 
        else:
            new_loess = np.append(new_loess, (lowess(av_loess[:365], np.squeeze(np.linspace(0, dist_minima[l], num=len(av_loess[:365]))/calendarYear), frac=1./10, it=0, is_sorted=True, missing='drop', return_sorted=False)), axis=0)
    return new_loess

long_loess=new_loess_func()

#Mean y_data_detrended
y_data_detrended_mean=np.mean(y_data_detrended_matrix, axis=1)

#residuals
resdiuals_longloess=y_data_detrended_mean[:len(long_loess)]-long_loess[:]
#==============================================================================
#Add trend and model residual

for r in scenarios:
    model_trend_loess=long_loess+trend[:len(long_loess),r]
       
    #residuals
    #residuals_modeled=y_data[:len(long_cosine)]-model_trend_residual
    residuals_modeled_loess=y_data_365_nearest_matrix[:len(long_loess),r]-model_trend_loess
    std_residuals_modeled_loess=np.nanstd(residuals_modeled_loess)
   
    #Take variance of residuals
    residuals_modeled_loess_var=np.square(residuals_modeled_loess)  
    residuals_modeled_loess_var_av=seasonal_mean(residuals_modeled_loess_var, 365)
    
    #Pattern in average residual shape
    residual_pattern=lowess(residuals_modeled_loess_var_av, np.arange(len(residuals_modeled_loess_var_av)), frac=1./10, it=0, is_sorted=True, missing='drop', return_sorted=False)    
    residual_pattern_sqr=np.sqrt(residual_pattern)
    residual_pattern_sqr_matrix[:len(residual_pattern_sqr),r]=residual_pattern_sqr

#Save residual_pattern_sqr_matrix
np.save(r'av_residual_pattern_sqr_scenario.npy', residual_pattern_sqr_matrix)

#Average residual shape
av_residual_pattern_sqr=np.nanmean(residual_pattern_sqr_matrix, axis=1)

#Tile them together
residual_pattern_sqr_long=np.tile(residual_pattern_sqr,(len(residuals_modeled_loess)/len(residual_pattern_sqr)))

#Choose number of scenarios
nscen=120
sigma_resid=np.zeros((nscen,len(residual_pattern_sqr_long)))
for i in range(nscen):
    sigma_resid[i,:]=residual_pattern_sqr_long*[random.normalvariate(0, 1) for _ in range(len(residual_pattern_sqr_long))]

#==============================================================================
#Derive distribution parameters
parameter_array2=np.stack((line_intercept_flattened, line_slope_flattened), axis=1)

mean_parameters2=np.mean(parameter_array2, axis=0)
cov_parameters2=np.cov(parameter_array2, rowvar=False)
corrcoef_parameters2=np.corrcoef(parameter_array2, rowvar=False)
print('Mean alpha' ,np.around(mean_parameters2[0], 2), 'Mean Beta',np.around(mean_parameters2[1], 3))
print(np.around(cov_parameters2, 2))
print(np.around(corrcoef_parameters2, 2))

#Sample new parameters from multivariate normal distribution
sample_size=(90,nscen)
sample_parameters2=np.random.multivariate_normal(mean_parameters2, cov_parameters2,nscen)

#==============================================================================
#LHS sample - deviation
sample_deviation=np.random.normal(deviation_mean, deviation_std,sample_size)

#Hypercube sampling
# Find the ranks of each column
p = np.shape(sample_deviation)[1];
sample_deviation_lhs = np.zeros(np.shape(sample_deviation))
Rank = np.transpose(np.zeros(np.shape(sample_deviation)))
for i in range(p):
    Rank[i,:] = rankdata(sample_deviation[:,i]) # Equivalent to tiedrank - in-built Matlab function

#Get gridded or smoothed-out values on the unit interval
sample_deviation_lhs = np.transpose(Rank) - np.random.uniform(0,1, size=np.shape(sample_deviation_lhs)) #smoothed values
#x_lhs = np.transpose(Rank) - 0.5; #gridded values at interval center values

sample_deviation_lhs = sample_deviation_lhs / len(sample_deviation)

#Transform the marginals to desired distribution (Normal and Inverse-Gamma)
#maintaining the ranks (and therefore rank correlations) from the original random sample
sample_deviation_lhs = stats.norm.ppf(q=sample_deviation_lhs, loc=deviation_mean, scale=deviation_std) 

#dist_min_new=(calendarYear+sample_parameters[:,:,1]*calendarYear)
dist_min_new=(calendarYear+sample_deviation[:,:]*calendarYear)

#Derive distribution parameters after sampling [for verification]
deviation_mean_after=np.mean(sample_deviation[:,:], axis=0)
deviation_std_after=np.mean(sample_deviation[:,:], axis=0)
#==============================================================================
#LHS sample - deviation per scenario
sample_size_sc=(90,nscen/len(scenarios))

sample_deviation_sc=np.zeros((sample_size))
for i in np.arange(len(scenarios)):
    sample_deviation_sc[:,(i*nscen/len(scenarios)):((i+1)*nscen/len(scenarios))]=np.random.normal(deviation_mean_sc[i], deviation_std_sc[i],sample_size_sc)
#==============================================================================
#Construct new scenarios
#Time change
tau_tk_matrix = np.zeros((np.where(x_data==np.float(np.shape(sample_deviation_sc)[0]-1))[0].item(),np.shape(sample_deviation_sc)[1]))
for z in np.arange(np.shape(sample_deviation_sc)[1]):
#m_i = i - d_i
    m_i = (np.arange(np.shape(sample_deviation_sc)[0])) - sample_deviation_sc[:,z]
    m_i[0] = 0
    
    #Tau_tk 
    tau_tk = np.zeros(np.where(x_data==np.float(np.shape(m_i)[0]-1))[0]) 
    count = 0
    for t in range(np.shape(m_i)[0]-1):
        tk = x_data[(x_data >= t) & (x_data < t+1)]
        for t_k in tk:
            tau_tk[count] = m_i[t] + (m_i[t+1] - m_i[t]) * (t_k - t)
            count = count + 1
    tau_tk_matrix[:,z]=tau_tk

np.save(r'tau_tk_matrix.npy', tau_tk_matrix)
print("Tau_tk array saved")
#==============================================================================