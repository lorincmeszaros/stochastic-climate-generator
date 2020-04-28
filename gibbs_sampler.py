# -*- coding: utf-8 -*-
"""

@author: Lorinc Meszaros
Affiliation: TU Delft and Deltares, Delft, The Netherlands
    
Gibbs sampler

The general approach to deriving an update for a variable is

1. Write down the posterior conditional density in log-form
2. Throw away all terms that don’t depend on the current sampling variable
3. Pretend this is the density for your variable of interest and all other variables are fixed. What distribution does the log-density remind you of?
4. That’s your conditional sampling density!
"""
#%%------------------------------------------------------------------------------
#Import required packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
import scipy as sc
from scipy.stats import invgamma
from scipy.stats import gamma
from scipy.stats import rankdata
from scipy import interpolate
import random
import time
#%%------------------------------------------------------------------------------
#Set figure param
plt.rcParams['figure.figsize'] = (10, 5)

#Derive Gibbs updates
#Updates for Theta
def sample_theta(y, x, scen, AjV, sigma_E_2, sigma_alpha_2, sigma_beta_2, sigma_AjS_2, mu_theta):
    N = len(y)
    assert len(x) == N
    y_p = y.reshape(-1,1)
    sigma_theta = np.zeros((2+J,2+J))
    #Update sigma_theta covariance matrix
    sigma_theta[0,0] = sigma_alpha_2
    sigma_theta[1,1] = sigma_beta_2
    for s in range(J):
        sigma_theta[(2+s),(2+s)] = sigma_AjS_2[s]
    
    phi_S_loop = np.zeros((N,J))
    for i in range(J):
        phi_S_loop[(i)*(N/J):(i+1)*(N/J), i] = phi_S[:,scen]    
            
    x_p = np.concatenate((np.repeat(1,N).reshape(-1,1),x.reshape(-1,1),phi_S_loop),axis=1)
    
    AjV_diag = np.zeros((N))
    for k in range(1,J+1):
        AjV_diag[((k-1)*(N/J)):((k-1)*(N/J)+(N/J))] = np.repeat(AjV[0,k-1],N/J)
    Sigma= np.diagflat([AjV_diag * phi_V_diag[:,scen]])*sigma_E_2  #symmetric square matrix
    
    linalg_sol = sc.linalg.solve_triangular(Sigma, x_p) #overwrite_b=True, check_finite=False
    P = np.dot(np.transpose(x_p),linalg_sol) + np.linalg.inv(sigma_theta)

    linalg_sol_y = sc.linalg.solve_triangular(Sigma, y_p) #overwrite_b=True, check_finite=False
    V = np.dot(np.transpose(x_p),linalg_sol_y) + np.dot(np.linalg.inv(sigma_theta),mu_theta) #- np.dot(np.transpose(x),linalg_mean_sol * sigma_E_2**-1)  

    cov = np.linalg.inv(P)
    mean = np.dot(cov,V)

    return np.random.multivariate_normal(np.squeeze(mean), cov)

#Update for sigma_E_2

def sample_sigma_E_2(y, x, scen, alpha, AjS, AjV, beta, alpha_E, beta_E):
    N = len(y)
    AjV_diag = np.zeros((N))
    AjS_diag = np.zeros((N))
    for k in range(1,J+1):
        AjV_diag[((k-1)*(N/J)):((k-1)*(N/J)+(N/J))] = np.repeat(AjV[0,k-1],N/J)
        AjS_diag[((k-1)*(N/J)):((k-1)*(N/J)+(N/J))] = np.repeat(AjS[0,k-1],N/J)
    Sigma= np.diagflat([AjV_diag * phi_V_diag[:,scen]])  #symmetric square matrix
    alpha_new = alpha_E + (N / 2.0)
    mu_k = alpha + x*beta + AjS_diag*phi_S_diag[:,scen]
    linalg_sig_sol = sc.linalg.solve_triangular((Sigma), (y - mu_k)) #np.dot(np.linalg.inv(Sigma), (y - mu_k))
    resid =  0.5 * np.dot(np.transpose(y - mu_k),linalg_sig_sol) 
    beta_new = beta_E + resid
    return invgamma.rvs(alpha_new, scale=beta_new)

#Update for AjV
    
def sample_AjV(y, x, J, scen, alpha, beta, AjS, sigma_E_2, a_AjV, b_AjV):
    N = len(y)
    AjS_diag = np.zeros((N))
    AjV_sample = np.zeros((1,J))
    A_sum = np.zeros((1,(N/J)))
    for k in range(1,J+1):
        AjS_diag[((k-1)*(N/J)):((k-1)*(N/J)+(N/J))] = np.repeat(AjS[0,k-1],N/J)    
    for j in range(1,J+1):
        for tk in range((((j-1)*N)/J),((j*N)/J)): 
            mu_k = alpha + x[tk]*beta + AjS_diag[tk] * phi_S[tk-(j-1)*(N/J),scen]
            A_sum[:,(tk-(j-1)*(N/J))]=np.square(y[tk]- mu_k)*(sigma_E_2 * phi_V[tk-(j-1)*(N/J),scen])**-1
        a_new = a_AjV + ((N / (2*J)))
        resid =  0.5 * (np.sum(A_sum))
        b_new = b_AjV[j-1] + resid
        AjV_sample[:,j-1] = invgamma.rvs(a_new, scale=b_new)
    return AjV_sample

#Update for sigma_alpha_2

def sample_sigma_alpha_2(scenarios, alpha, delta1_alpha, delta2_alpha):
    L = len(scenarios)
    delta1 = delta1_alpha + (L / 2.0)
    resid =  0.5 * np.sum(np.square(alpha)) 
    delta2 = delta2_alpha + resid
    return invgamma.rvs(delta1, scale=delta2)

#Update for sigma_beta_2

def sample_sigma_beta_2(scenarios, beta, lambda1_beta, lambda2_beta):
    L = len(scenarios)
    lambda1 = lambda1_beta + (L / 2.0)
    resid =  0.5 * np.sum(np.square(beta)) 
    lambda2 = lambda2_beta + resid
    return invgamma.rvs(lambda1, scale=lambda2)

#Update for sigma_AjS_2

def sample_sigma_AjS_2(scenarios, AjS, omega1_AjS, omega2_AjS):
    sigma_AjS_2_new = np.zeros((J))
    omega1 = omega1_AjS + (J / 2.0)
    for j in range(J):
        resid =  0.5 * np.sum(np.square(AjS[j,:])) 
        omega2 = omega2_AjS + resid
        sigma_AjS_2_new[j] = invgamma.rvs(omega1, scale=omega2)
    return sigma_AjS_2_new

#Update for beta_E

def sample_beta_E(scenarios, sigma_E_2, alpha_E, lambda_sigma_E_2):
    L = len(scenarios)
    gamma1 = L * alpha_E + 1
    resid =  np.sum(sigma_E_2**-1) 
    gamma2 = lambda_sigma_E_2 + resid
    return gamma.rvs(gamma1, scale=gamma2)

#Update for b_AjV

def sample_b_AjV(scenarios, AjV, a_AjV, kappa_b_AjV):
    b_AjV_new = np.zeros((J))
    kappa1 = J * a_AjV + 1
    for j in range(J):
        resid =  np.sum(AjV[j,:]**-1)
        kappa2 = kappa_b_AjV + resid
        b_AjV_new[j] = gamma.rvs(kappa1, scale=kappa2**-1)
    return b_AjV_new

#%%------------------------------------------------------------------------------
#Write the Gibbs sampler

def gibbs(y_data_365_nearest_matrix, x, iters, scenarios, init, hypers):
    sigma_E_2_init = init["sigma_E_2"]
    AjV_init = np.asarray(init["AjV"], dtype=np.float64).reshape(1,J)
    AjV_true=np.asarray(init["AjV"], dtype=np.float64).reshape(1,J)
    sigma_alpha_2_init = init["sigma_alpha_2"]
    sigma_beta_2_init = init["sigma_beta_2"]
    sigma_AjS_2_init = init["sigma_AjS_2"]
    beta_E_init = init["beta_E"]
    b_AjV_init = init["b_AjV"]
    
    trace = np.zeros((iters,(6+4*J), len(scenarios))) ## trace to store values of beta, sigma_E_2
    print('No of iterations:')
    print(" ")
    for it in range(iters):
        print('Scenarios:')
        print(" ")
        if it == 0:      
            count=0
            for scen in scenarios:
                y=y_data_365_nearest_matrix[datalimit1:datalimit2, scen]
                y=y[~np.isnan(y)]
                assert len(y) == len(x)
                start = time.time()
                theta = sample_theta(y, x, scen, AjV_init, sigma_E_2_init, sigma_alpha_2_init, sigma_beta_2_init, sigma_AjS_2_init, hypers["mu_theta"])
                print("theta" + "_sc" + str(count) + "_it" + str(it))
                alpha = theta[0]
                beta = theta[1]
                AjS = np.transpose(theta[2:].reshape(-1,1))
                sigma_E_2 = sample_sigma_E_2(y, x, scen, alpha, AjS, AjV_true, beta, hypers["alpha_E"], beta_E_init)
                print("sigma_E_2" + "_sc" + str(count) + "_it" + str(it))
                AjV = sample_AjV(y, x, J, scen, alpha, beta, AjS, sigma_E_2, hypers["a_AjV"], b_AjV_init)
                print("AjV" + "_sc" + str(count) + "_it" + str(it))
                trace[it, :, count] = np.concatenate((np.array([theta[0], theta[1], sigma_E_2]).reshape(1,3), AjS, AjV, np.array([sigma_alpha_2_init, sigma_beta_2_init, beta_E_init]).reshape(1,3), sigma_AjS_2_init.reshape(1,J), b_AjV_init.reshape(1,J)), axis=1)
                print("sc_" + str(count))
                end = time.time()
                print("it_time: " + str(end - start))
                count=count+1
        else:
            count=0
            for scen in scenarios:
                y=y_data_365_nearest_matrix[datalimit1:datalimit2, scen]
                y=y[~np.isnan(y)]
                assert len(y) == len(x)
                start = time.time()
                sigma_alpha_2 = sample_sigma_alpha_2(scenarios, trace[it-1,0,:], hypers["delta1_alpha"], hypers["delta2_alpha"])
                print("sigma_alpha_2" + "_sc" + str(count) + "_it" + str(it))
                sigma_beta_2 = sample_sigma_beta_2(scenarios, trace[it-1,1,:], hypers["lambda1_beta"], hypers["lambda2_beta"])
                print("sigma_beta_2" + "_sc" + str(count) + "_it" + str(it))
                sigma_AjS_2 = sample_sigma_AjS_2(scenarios, trace[it-1,np.arange(3,(3+J)).tolist(),:], hypers["omega1_AjS"], hypers["omega2_AjS"])
                print("sigma_AjS_2" + "_sc" + str(count) + "_it" + str(it))
                beta_E = sample_beta_E(scenarios, trace[it-1,2,:], hypers["alpha_E"], hypers["lambda_sigma_E_2"])
                print("beta_E" + "_sc" + str(count) + "_it" + str(it))
                b_AjV = sample_b_AjV(scenarios, trace[it-1,np.arange(3+J,(3+J+J)).tolist(),:], hypers["a_AjV"], hypers["kappa_b_AjV"])
                print("b_AjV" + "_sc" + str(count) + "_it" + str(it))
                theta = sample_theta(y, x, scen, np.transpose(trace[it-1,np.arange(3+J,(3+J+J)).tolist(),count].reshape(-1,1)), trace[it-1,2,count], sigma_alpha_2, sigma_beta_2, sigma_AjS_2, hypers["mu_theta"])
                print("theta" + "_sc" + str(count) + "_it" + str(it))
                alpha = theta[0]
                beta = theta[1]
                AjS = np.transpose(theta[2:].reshape(-1,1))
                sigma_E_2 = sample_sigma_E_2(y, x, scen, alpha, AjS, AjV_true, beta, hypers["alpha_E"], beta_E)
                print("sigma_E_2" + "_sc" + str(count) + "_it" + str(it))
                AjV = sample_AjV(y, x, J, scen, alpha, beta, AjS, sigma_E_2, hypers["a_AjV"], b_AjV)
                print("AjV" + "_sc" + str(count) + "_it" + str(it))
                trace[it, :, count] = np.concatenate((np.array([theta[0], theta[1], sigma_E_2]).reshape(1,3), AjS, AjV, np.array([sigma_alpha_2, sigma_beta_2, beta_E]).reshape(1,3), sigma_AjS_2.reshape(1,J), b_AjV.reshape(1,J)), axis=1)
                print("sc_" + str(count))
                end = time.time()
                print("it_time: " + str(end - start)) 
                print(" ")
                count=count+1
        print(" ")
        print("Successfully completed for this iteration: " + str(it))
        print(" ")
    print("Successfully completed for all iterations")
    return trace

print("Update steps defined")
#%%-----------------------------------------------------------------------------
#Load data, pre-processed shapes, and results of the time shift function
y_data_all=np.load(r'y_data_all.npy')
av_loess = np.load(r'av_loess_scenario.npy')
av_residual_pattern_sqr = np.load(r'av_residual_pattern_sqr_scenario.npy')
x_data_ideal_1D_matrix = np.load(r'x_data_ideal_1D_matrix.npy')
y_data_365_nearest_matrix=np.load(r'y_data_365_nearest_matrix.npy')

#Remove zero columns from x_data_ideal_1D_matrix
x_data_ideal_1D_matrix = x_data_ideal_1D_matrix[:,~np.all(np.isnan(x_data_ideal_1D_matrix), axis=0)]

#CalendarYear
calendarYear=365.00
#Define data range in number of timesteps
datalimit1=0
datalimit2=29200

#Choose the number of scenarios
scenarios= [0,1,2,4,5,6,7,9]
sce=0
scen_names=['CNRM 4.5', 'ICHEC 4.5', 'IPSL 4.5', 'MPI 4.5', 'CNRM 8.5', 'ICHEC 8.5', 'IPSL 8.5', 'MPI 8.5']
#0-CNRM 4.5, 1-ICHEC 4.5, 2-IPSL 4.5, 3-MOHC-HadGEM 4.5 (not included), 4-MPI 4.5
#5-CNRM 8.5, 6-ICHEC 8.5, 7-IPSL 8.5, 8-MOHC-HadGEM 8.5 (not included), 9-MPI 8.5

#Stations
#stations= ['Marsdiep Noord','Doove Balg West',
#                'Vliestroom','Doove Balg Oost',
#                'Blauwe Slenk Oost','Harlingen Voorhaven','Dantziggat',
#                'Zoutkamperlaag Zeegat','Zoutkamperlaag',
#                'Harlingen Havenmond West']

#Initialize x and y
x=np.arange(0,datalimit2)/calendarYear
y=y_data_365_nearest_matrix[datalimit1:datalimit2, sce]
#Remove NaN
y=y[~np.isnan(y)]

#%%-----------------------------------------------------------------------------
#Number of timesteps and years
N=len(x)
J=N/int(calendarYear)

#Seasonal pattern
phi_S=av_loess
phi_S=np.nan_to_num(phi_S)
phi_S=phi_S-np.nanmin(phi_S, axis=0)  
phi_S[:,scenarios]=phi_S[:,scenarios]+0.01 
phi_S[phi_S==0]=np.nan
phi_S_diag = np.zeros((N,np.shape(phi_S)[1]))
for r in scenarios:
    phi_S_diag[:,r]=np.tile(phi_S[:,r],J) 
    
#Residual pattern    
phi_V=av_residual_pattern_sqr

phi_V_diag = np.zeros((N,np.shape(phi_V)[1]))
for r in scenarios:
    phi_V_diag[:,r]=np.tile(phi_V[:,r],J) 
        
AjV_true=np.repeat(0.5,J)

#Supply initial parameter estimates and hyper parameters
#Place N(0,1) priors on beta, and inv-gamma(2,1) on sigma_E_2

## specify initial values
init = {"sigma_E_2": 125.0,
        "AjV": AjV_true,
        "sigma_alpha_2": 1,
        "sigma_beta_2": 1,
        "sigma_AjS_2": np.repeat(1,J),
        "beta_E": 1,
        "b_AjV": np.repeat(1,J)}

## specify hyper parameters
hypers = {"mu_theta": np.transpose(np.repeat(0.0,2+J)).reshape(-1,1),
         "alpha_E": 2,
         "lambda_sigma_E_2": 0.01,
         "a_AjV": 2,
         "kappa_b_AjV": 0.01,
         "delta1_alpha": 2,
         "delta2_alpha": 1,
         "lambda1_beta": 2,
         "lambda2_beta": 1,
         "omega1_AjS": 2,
         "omega2_AjS": 1}

#%%----------------------------------------------------------------------------
#Test algorithm  
iters = 1005 #Choose number of iterations

#Run the Gibbs sampler 
trace = gibbs(y_data_365_nearest_matrix, x, iters, scenarios, init, hypers)

#%%----------------------------------------------------------------------------
#Forward simulate

trace_burnt = trace[5:iters,:] #Ignore burn in period

#Choose Number of scenarios per original scenario
n_newscen=15

#Hypercube sampling
# Find the ranks
x_lhs = np.zeros(n_newscen)
Rank = np.transpose(np.zeros(n_newscen))
Rank = rankdata(range(0, n_newscen)) # Equivalent to tiedrank

#Get gridded or smoothed-out values on the unit interval
#x_lhs = np.transpose(Rank) - np.random.uniform(0,1, size=np.shape(x_lhs)) #smoothed values
x_lhs = np.transpose(Rank) - 0.5; #gridded values at interval center values
#Scale between 0-1
x_lhs = x_lhs / n_newscen
#SCale back to original
sample_no_lhs = np.array(x_lhs * len(trace_burnt), dtype=int).tolist() 

#Random sampling
sample_no = random.sample(range(0, len(trace_burnt)), n_newscen)


df = pd.DataFrame(np.transpose(np.vstack((sample_no, sample_no, sample_no_lhs, sample_no_lhs))), columns = ['x1', 'y1', 'x2', 'y2'])

#Generate new scenarios
tau_tk=np.load(r'tau_tk_matrix.npy') #Add Tau(tk) from pre-processing
y_new_nearest_no365=np.zeros((N,len(trace_burnt),np.shape(trace_burnt)[2]))
y_tau_tk_new=np.zeros((N,n_newscen,np.shape(trace_burnt)[2]))

for sce in np.arange(np.shape(trace_burnt)[2]): 
    count=1
    for s in sample_no_lhs:
        alpha_new = trace_burnt[s,0, sce]
        beta_new= trace_burnt[s,1, sce]
        sigma_E_2_new= trace_burnt[s,2, sce]
        AjS_new=np.asarray(trace_burnt[s,3:(3+J), sce], dtype=np.float64).reshape(1,J)
        AjV_new=np.asarray(trace_burnt[s,(3+J):(3+J+J), sce], dtype=np.float64).reshape(1,J)
       
        AjS_diag = np.zeros((N))
        AjV_diag = np.zeros((N))
        for k in range(1,J+1):
            AjS_diag[((k-1)*(N/J)):((k-1)*(N/J)+(N/J))] = np.repeat(AjS_new[0,k-1],N/J)
            AjV_diag[((k-1)*(N/J)):((k-1)*(N/J)+(N/J))] = np.repeat(AjV_new[0,k-1],N/J)
        Sigma_S_new= np.diagflat([AjS_diag]) * phi_S_diag[:,scenarios[sce]] #symmetric square matrix
        Sigma_new= np.diagflat([AjV_diag]) * phi_V_diag[:,scenarios[sce]] #symmetric square matrix
        
        y_new=np.zeros((N,1))
        for i in range(N):
            y_new[i] = alpha_new + x[i] * beta_new + Sigma_S_new[i,i] + np.random.normal(0, np.sqrt(Sigma_new[i,i] * sigma_E_2_new))
               
        #Transform back
        x_data_ideal_1D=x_data_ideal_1D_matrix[datalimit1:datalimit2, sce]
        x_data_ideal_1D=x_data_ideal_1D[~np.isnan(x_data_ideal_1D)]
        
        #Interpolate to regular step
        f = interpolate.interp1d(x, y_new[:,0], kind='nearest', fill_value="extrapolate")
        y_new_nearest_no365[:,s,sce]= f(x_data_ideal_1D[datalimit1:datalimit2])   # use interpolation function returned by `interp1d` to follow original spacing  
        y_tau_tk_new[:,count-1,sce]= f(tau_tk[datalimit1:datalimit2,sce*len(sample_no_lhs)+count-1])   # use interpolation function returned by `interp1d` to follow generted spacing
        print('New scneario no ' + str(count) + ' for original scenario no ' + str(sce))    
        count=count+1

np.save(r'y_tau_tk_new.npy', y_tau_tk_new)
print("Array saved")
#%%----------------------------------------------------------------------------
#Post-process for Delwaq simulation
y_tau_tk_new[y_tau_tk_new < 0] = 0 #lower limit is 0 (negative value has no physical meaning)
y_new_nearest_no365[y_new_nearest_no365 < 0] = 0 #lower limit is 0 (negative value has no physical meaning)

#save array
np.save(r'y_tau_tk_new.npy', y_tau_tk_new)

#Remove zero columns
y_new_nearest_no365_sce=np.zeros((N,n_newscen,np.shape(trace_burnt)[2]))
for sce in np.arange(np.shape(trace_burnt)[2]): 
    temp=y_new_nearest_no365[:, :, sce]
    idx = np.argwhere(np.all(temp[..., :] == 0, axis=0))    
    y_new_nearest_no365_sce[:,:,sce] = np.delete(temp, idx, axis=1)

#Flattened generated data
y_tau_tk_new = y_tau_tk_new.reshape(datalimit2,n_newscen*len(scenarios))[:,:]
y_new_nearest_no365_sce = y_new_nearest_no365_sce.reshape(datalimit2,n_newscen*len(scenarios))[:,:]

#save array
np.save(r'y_tau_tk_new_flat.npy', y_tau_tk_new)
np.save(r'y_new_nearest_no365_sce_flat.npy', y_new_nearest_no365_sce)
print("Array saved")

#load array (if previous steps interrupted but outputs saved continue from here)
y_tau_tk_new=np.load(r'y_tau_tk_new.npy')
y_tau_tk_new_flat=np.load(r'y_tau_tk_new_flat.npy')
y_new_nearest_no365_sce_flat=np.load(r'y_new_nearest_no365_sce_flat.npy')

#Plot
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
plt.rcParams['figure.figsize'] = (10,10)
#Plot 1
axes[0].plot(tau_tk[datalimit1:datalimit2,:1], y_tau_tk_new_flat[:,:1], 'o', alpha=0.6)
axes[0].plot(x[datalimit1:datalimit2], y_data_all[datalimit1:datalimit2,scenarios[0]], 'ko', label='Baseline scenario', alpha=0.8)
axes[0].set_title('No of generated scenarios per baseline scenario: n = ' + str(1))
axes[0].legend()
axes[0].set_ylabel('Radiation [W m-2]')

#Plot 2
axes[1].plot(tau_tk[datalimit1:datalimit2,:5], y_tau_tk_new_flat[:,:5], 'o', alpha=0.6)
axes[1].plot(x[datalimit1:datalimit2], y_data_all[datalimit1:datalimit2,scenarios[0]], 'ko', label='Baseline scenario', alpha=0.8)
axes[1].set_title('No of generated scenarios per baseline scenario: n = ' + str(5))
axes[1].legend()
axes[1].set_ylabel('Radiation [W m-2]')

#Plot 3
axes[2].plot(tau_tk[datalimit1:datalimit2,:15], y_tau_tk_new_flat[:,:15], 'o', alpha=0.6)
axes[2].plot(x[datalimit1:datalimit2], y_data_all[datalimit1:datalimit2,scenarios[0]], 'ko', label='Baseline scenario', alpha=0.8)
axes[2].set_title('No of generated scenarios per baseline scenario: n = ' + str(15))
axes[2].legend()
axes[2].set_ylabel('Radiation [W m-2]')

#Generic
plt.xlim(10,12)
plt.xlabel('Years')
plt.savefig(r'generated_scenarios.pdf', format='pdf', bbox_inches='tight')
plt.show()
#%%----------------------------------------------------------------------------
#Compare statistics
#Mean
baseline_mean=np.mean(y_data_all[datalimit1:datalimit2,scenarios])
generated_mean=np.mean(y_tau_tk_new_flat[:,:])
#Std
baseline_std=np.std(y_data_all[datalimit1:datalimit2,scenarios])
generated_std=np.std(y_tau_tk_new_flat[:,:])
print('Mean_baseline: '+ str(baseline_mean) + '; ' + 'Mean_generated: '+ str(generated_mean))
print('Std_baseline: '+ str(baseline_std) + '; ' + 'Std_generated: '+ str(generated_std))

#R-squared
from sklearn.metrics import r2_score
#Each scenario
r2=np.zeros((15))
for r in np.arange(15):
    r2[r]=r2_score(y_data_all[datalimit1:datalimit2,scenarios[0]], y_tau_tk_new_flat[:,r])
    
#Average
r2_av=r2_score(np.mean(y_data_all[datalimit1:datalimit2,scenarios],axis=1), np.mean(y_tau_tk_new_flat[:,:],axis=1))
#%%----------------------------------------------------------------------------

