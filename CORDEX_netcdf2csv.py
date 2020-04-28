# -*- coding: utf-8 -*-
"""
Read and extracting data in NetCDF 2D array Grids Files
For product: Cordex EUR-11

Note: Check information of nc files, because it relates to #Root_name

Items need to be addressed:
    1 - path_folder, data_name
    2 - site_name, site_name_ful, lati, loni in Lookup_WZsite.xlsx   
    5 - out_path

Function: Using numpy and KD-trees with netCDF data, by @author: unidata
URL: r'https://github.com/Unidata/unidata-python-workshop/blob/master/notebooks/netCDF/netcdf-by-coordinates.ipynb'
"""
#%%
import numpy as np
import pandas as pd
import netCDF4
from math import pi
from numpy import cos,sin
from scipy.spatial import cKDTree
import glob
import os
###===================================================================
## 1- Monitoring stations [Manual Definition]
Lookup_WZsite = pd.read_excel(r'Lookup_WZsite.xlsx')
site_name = Lookup_WZsite['site_name'][0:11]
site_name_ful= Lookup_WZsite['site_name_ful'][0:11]
lati= Lookup_WZsite['lati'][0:11]
loni= Lookup_WZsite['loni'][0:11]      
location = zip(lati, loni)
print(zip(site_name,location)) - double check locations
    
#[Finish manual definition]

###===================================================================
## 2- Function: Looking up array indices using KD-Tree 
def kdtree_fast (latvar, lonvar, lat0, lon0):
    rad_factor= pi/180.0 #for trignometry, need angles in radians
    # Read Lat,Long from file to numpy arrays
    latvals= latvar[:]*rad_factor
    lonvals= lonvar[:]*rad_factor
    ny,nx = latvals.shape
    clat,clon = cos(latvals),cos(lonvals)
    slat,slon = sin(latvals),sin(lonvals)
    # Build kd-tree from big arrays of 3D coordinates
    triples = list(zip(np.ravel(clat*clon), np.ravel(clat*slon), np.ravel(slat)))
    kdt = cKDTree(triples)
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    clat0,clon0 = cos(lat0_rad),cos(lon0_rad)
    slat0,slon0 = sin(lat0_rad),sin(lon0_rad)
    dist_sq_min, minindex_1d = kdt.query([clat0*clon0, clat0*slon0, slat0])
    iy_min, ix_min = np.unravel_index(minindex_1d, latvals.shape)
    return iy_min,ix_min
###===================================================================
#%%
## 3 - READ NC FILES
# [Manual define]=====================================================
path_folder= r'\CORDEX\ncf'
data_name= [#'pr',
            'rsds',
            'tas',
            'uas',
            'vas',
            'clt',
            'hurs',
            'ps']

c=0
for dname in data_name:
    data_path= os.path.join(path_folder, dname)
    ncf= glob.glob(data_path + "_IPSL-IPSL-CM5A-MR_*.nc")    
    #_CNRM-CERFACS-CNRM-CM5_
    #_ICHEC-EC-EARTH_
    #_IPSL-IPSL-CM5A-MR_
    #_MOHC-HadGEM2-ES_
    #_MPI-M-MPI-ESM-LR_
    
    ###===================================================================
    ## 4 - Import data from files
    for nc_i in ncf:
        nc_i= netCDF4.Dataset(nc_i,'r')
        latvar= nc_i.variables['lat']
        lonvar= nc_i.variables['lon']
        
        lat0= latvar[0]
        lon0= lonvar[0]
        
        # Time variables
        time_var= nc_i.variables['time']
        time_range= nc_i.variables['time'][:]
        time_date= netCDF4.num2date(time_var[:],time_var.units)
        print ("Reading file ", nc_i.driving_experiment)
        print ('Extracting variable:', dname)
        print ('Start date - End date:', time_date[0],time_date[-1])
        print (' ')
       
    ###===================================================================
        ## 5 - EXTRACTING DATA 
        Root_name= dname+'_'+nc_i.driving_model_id+'_'+nc_i.experiment_id
        out_path= r'\CORDEX\TS'
        
        for i in range(len(site_name)):
            out_name= (Root_name+'_'+ site_name_ful[i]+'.csv') 
            iy,ix = kdtree_fast(latvar, lonvar, location[i][0], location[i][1])
            #print ('Exact Location lat-lon:', location[i])
            #print ('Closest lat-lon:', latvar[iy,ix], lonvar[iy,ix])
            #print ('Array indices [iy,ix]=', iy, ix)          
    
            data_i= nc_i.variables[dname][:,iy,ix]
            data_ts= pd.Series(data_i, index= time_date, name= dname )
            print ('Writing location ',site_name_ful[i])
            print ('NC file extracting (%) ', (i+1)*100/len(site_name ))        
    ###===================================================================
            ## 6 - PRINTING to *.CSV
            data_ts.to_csv(os.path.join(out_path,out_name))        
    
    #Print overall progress
    c=c+1
    print ('Progress (%): ', (c)*100/len(data_name))          
    ###===================================================================
