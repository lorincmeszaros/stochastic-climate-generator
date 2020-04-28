# -*- coding: utf-8 -*-
"""
@author: Lorinc Meszaros
Affiliation: TU Delft and Deltares, Delft, The Netherlands

Preparing daily/monthly/yearly averages
"""

#==============================================================================
#Import
import scipy.io
import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#==============================================================================
#Load CORDEX meteo data set
csv_mainpath=r'/CORDEX/TS'
filenames_all = os.listdir(csv_mainpath)
data_name= ['rsds','tas','uas','vas','clt','hurs','ps'] 
stations= ['Marsdiep Noord','Doove Balg West',
                'Vliestroom','Doove Balg Oost',
                'Blauwe Slenk Oost','Harlingen Voorhaven','Dantziggat',
                'Zoutkamperlaag Zeegat','Zoutkamperlaag',
                'Harlingen Havenmond West']
            
#======SELECT=========
driving_model_id=['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR'] #'CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR'
experiment_id=['rcp45','rcp85']; # 'rcp45','rcp85'; 'historical'
#=====================
#Output path
output_path_yearly=r'tensor_yearly_mean_5D.npy'
output_path_monthly=r'tensor_monthly_mean_5D.npy'
output_path_daily=r'tensor_daily_mean_5D.npy'
#=====================
#Choose Time window
time_start=2006
time_end=2096
#=====================
yearly_mean=[]
monthly_mean=[]
daily_mean=[]
tensor_yearly_mean_5D=np.zeros((91,7,10,5,2)) #timestep X variable X station X driving model X experiment
tensor_monthly_mean_5D=np.zeros((1092,7,10,5,2))
tensor_daily_mean_5D=np.zeros((33238,7,10,5,2))

print ('Extracting starts')
print ('Progress (%): 0')
c=0
for experiment in experiment_id:
    for driving_model in driving_model_id:
        for station in stations:
            i=0
            path=[]
            for dname in data_name:        
                path.append(os.path.join(csv_mainpath, dname + '_' + driving_model + '_' + experiment + '_' + station + '.csv'))
        
            #X1       
            X1 = pd.read_csv(path[0],header=None,usecols=[0,1]) #2009 1 March-1 Nov
            X1 = X1.dropna()
              
            year = []; month = []; day = []; hour = []
            for i in range(len(X1)):
               year.append(X1.loc[i,0][0:4])
               month.append(X1.loc[i,0][5:7])
               day.append(X1.loc[i,0][8:10])
               hour.append(X1.loc[i,0][11:13])
               
            #Year from string to int
            year=map(int, year)
            
            X1['year'] = year; X1['month'] = month; X1['day'] = day; X1['hour'] = hour; X1['value'] = X1.loc[:,1];
            X1=X1.drop(X1.columns[[0,1]], axis=1)
            #Use radiation during the day
            X1 = X1.loc[~(X1.loc[:,'value']==0)]
            #Set time window
            X1 = X1.loc[(X1.loc[:,'year'] >= time_start) & (X1.loc[:,'year'] <= time_end)]
                        
            yearly_mean_rad = X1.groupby(['year'])['value'].mean()
            monthly_mean_rad = X1.groupby(['year','month'])['value'].mean()
            daily_mean_rad = X1.groupby(['year','month','day'])['value'].mean()
            print ('X1')
            
            #X2    
            X2 = pd.read_csv(path[1],header=None,usecols=[0,1])
            X2 = X2.dropna()
            
            year = []; month = []; day = []; hour = []
            for i in range(len(X2)):
               year.append(X2.loc[i,0][0:4])
               month.append(X2.loc[i,0][5:7])
               day.append(X2.loc[i,0][8:10])
               hour.append(X2.loc[i,0][11:13])
                           
            #Year from string to int
            year=map(int, year)
            
            X2['year'] = year; X2['month'] = month; X2['day'] = day; X2['hour'] = hour; X2['value'] = X2.loc[:,1];
            X2=X2.drop(X2.columns[[0,1]], axis=1)
            #Set time window
            X2 = X2.loc[(X2.loc[:,'year'] >= time_start) & (X2.loc[:,'year'] <= time_end)]
            
            yearly_mean_temp = X2.groupby(['year'])['value'].mean()
            monthly_mean_temp = X2.groupby(['year','month'])['value'].mean()
            daily_mean_temp = X2.groupby(['year','month','day'])['value'].mean()
            print ('X2')
            
            #X3    
            X3 = pd.read_csv(path[2],header=None,usecols=[0,1])
            X3 = X3.dropna()
            
            year = []; month = []; day = []; hour = []
            for i in range(len(X3)):
               year.append(X3.loc[i,0][0:4])
               month.append(X3.loc[i,0][5:7])
               day.append(X3.loc[i,0][8:10])
               hour.append(X3.loc[i,0][11:13])
                           
            #Year from string to int
            year=map(int, year)
            
            X3['year'] = year; X3['month'] = month; X3['day'] = day; X3['hour'] = hour; X3['value'] = X3.loc[:,1];
            X3=X3.drop(X3.columns[[0,1]], axis=1)    
            #Set time window
            X3 = X3.loc[(X3.loc[:,'year'] >= time_start) & (X3.loc[:,'year'] <= time_end)]
            
            yearly_mean_uwind = X3.groupby(['year'])['value'].mean()
            monthly_mean_uwind = X3.groupby(['year','month'])['value'].mean()
            daily_mean_uwind = X3.groupby(['year','month','day'])['value'].mean()
            print ('X3')
            
            #X4
            X4 = pd.read_csv(path[3],header=None,usecols=[0,1])
            X4 = X4.dropna()
            
            year = []; month = []; day = []; hour = []
            for i in range(len(X4)):
               year.append(X4.loc[i,0][0:4])
               month.append(X4.loc[i,0][5:7])
               day.append(X4.loc[i,0][8:10])
               hour.append(X4.loc[i,0][11:13])
                           
            #Year from string to int
            year=map(int, year)
            
            X4['year'] = year; X4['month'] = month; X4['day'] = day; X4['hour'] = hour; X4['value'] = X4.loc[:,1];
            X4=X4.drop(X4.columns[[0,1]], axis=1)
            #Set time window
            X4 = X4.loc[(X4.loc[:,'year'] >= time_start) & (X4.loc[:,'year'] <= time_end)]
            
            yearly_mean_vwind = X4.groupby(['year'])['value'].mean()
            monthly_mean_vwind = X4.groupby(['year','month'])['value'].mean()
            daily_mean_vwind = X4.groupby(['year','month','day'])['value'].mean()
            print ('X4')
            
            #X5
            X5 = pd.read_csv(path[4],header=None,usecols=[0,1])
            X5 = X5.dropna()
            
            year = []; month = []; day = []; hour = []
            for i in range(len(X5)):
               year.append(X5.loc[i,0][0:4])
               month.append(X5.loc[i,0][5:7])
               day.append(X5.loc[i,0][8:10])
               hour.append(X5.loc[i,0][11:13])
                           
            #Year from string to int
            year=map(int, year)
            
            X5['year'] = year; X5['month'] = month; X5['day'] = day; X5['hour'] = hour; X5['value'] = X5.loc[:,1];
            X5=X5.drop(X5.columns[[0,1]], axis=1)   
            #Set time window
            X5 = X5.loc[(X5.loc[:,'year'] >= time_start) & (X5.loc[:,'year'] <= time_end)]
            
            yearly_mean_clt = X5.groupby(['year'])['value'].mean()
            monthly_mean_clt = X5.groupby(['year','month'])['value'].mean()
            daily_mean_clt = X5.groupby(['year','month','day'])['value'].mean()
            print ('X5')
            
            #X6
            X6 = pd.read_csv(path[5],header=None,usecols=[0,1])
            X6 = X6.dropna()
            
            year = []; month = []; day = []; hour = []
            for i in range(len(X6)):
               year.append(X6.loc[i,0][0:4])
               month.append(X6.loc[i,0][5:7])
               day.append(X6.loc[i,0][8:10])
               hour.append(X6.loc[i,0][11:13])
                           
            #Year from string to int
            year=map(int, year)
            
            X6['year'] = year; X6['month'] = month; X6['day'] = day; X6['hour'] = hour; X6['value'] = X6.loc[:,1];
            X6=X6.drop(X6.columns[[0,1]], axis=1)  
            #Set time window
            X6 = X6.loc[(X6.loc[:,'year'] >= time_start) & (X6.loc[:,'year'] <= time_end)]
            
            yearly_mean_hurs = X6.groupby(['year'])['value'].mean()
            monthly_mean_hurs = X6.groupby(['year','month'])['value'].mean()
            daily_mean_hurs = X6.groupby(['year','month','day'])['value'].mean()
            print ('X6')
            
            #X7
            X7 = pd.read_csv(path[6],header=None,usecols=[0,1])
            X7 = X7.dropna()
            
            year = []; month = []; day = []; hour = []
            for i in range(len(X7)):
               year.append(X7.loc[i,0][0:4])
               month.append(X7.loc[i,0][5:7])
               day.append(X7.loc[i,0][8:10])
               hour.append(X7.loc[i,0][11:13])
                           
            #Year from string to int
            year=map(int, year)
            
            X7['year'] = year; X7['month'] = month; X7['day'] = day; X7['hour'] = hour; X7['value'] = X7.loc[:,1];
            X7=X7.drop(X7.columns[[0,1]], axis=1)    
            #Set time window
            X7 = X7.loc[(X7.loc[:,'year'] >= time_start) & (X7.loc[:,'year'] <= time_end)]
            
            yearly_mean_press = X7.groupby(['year'])['value'].mean()
            monthly_mean_press = X7.groupby(['year','month'])['value'].mean()
            daily_mean_press = X7.groupby(['year','month','day'])['value'].mean()
            print ('X7')
            print ('----')
            
            del year,month,day,hour
        
        #------------ 
            yearly_mean=pd.concat([yearly_mean_rad, yearly_mean_temp, yearly_mean_uwind, yearly_mean_vwind, yearly_mean_clt, yearly_mean_hurs, yearly_mean_press], axis=1)
            monthly_mean=pd.concat([monthly_mean_rad, monthly_mean_temp, monthly_mean_uwind, monthly_mean_vwind, monthly_mean_clt, monthly_mean_hurs, monthly_mean_press], axis=1)
            daily_mean=pd.concat([daily_mean_rad, daily_mean_temp, daily_mean_uwind, daily_mean_vwind, daily_mean_clt, daily_mean_hurs, daily_mean_press], axis=1)
            #Append tensors
            tensor_yearly_mean_5D[:len(yearly_mean),:,stations.index(station),driving_model_id.index(driving_model),experiment_id.index(experiment)]=yearly_mean.values
            tensor_monthly_mean_5D[:len(monthly_mean),:,stations.index(station),driving_model_id.index(driving_model),experiment_id.index(experiment)]=monthly_mean.values
            tensor_daily_mean_5D[:len(daily_mean),:,stations.index(station),driving_model_id.index(driving_model),experiment_id.index(experiment)]=daily_mean.values
                    
            del path,X1,X2,X3,X4,X5,X6,X7, yearly_mean, monthly_mean, daily_mean
            c=c+1
            print ('Progress (%): ', (c)*100/(len(stations)*len(driving_model_id)*len(experiment_id)))
    
#Save results
np.save(output_path_yearly, tensor_yearly_mean_5D)
np.save(output_path_monthly, tensor_monthly_mean_5D)   
np.save(output_path_daily, tensor_daily_mean_5D)               

#Investigate a 3D slice of the 5D array
tensor_daily_mean_3D=tensor_daily_mean_5D[:,:,:,0,0]
