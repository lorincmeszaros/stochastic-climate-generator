# stochastic-climate-generator
Python implementation of a hierarchical (multi-level) Gibbs sampler to generate radaition scenarios from existing Euro-CORDEX climate scenarios

Use the scripts in the following order:
  0. Get CORDEX scenarios in netCDF format from http://data.dta.cnr.it/ecopotential/wadden_sea/ (for Wadden Sea) or https://www.euro-cordex.net/
  1. CORDEX_netcdf2csv.py --> Usage: Export relevant data at the required locations as .csv file from the netCDF format
  2. CORDEX_averaging.py --> Usage: Create a multi dimensional numpy array with all variables and scenarios at the required locations and apply daily/monthly/yearly averaging. 
  3. preprocess.py --> Usage: Load the daily avaraged CORDEX dataset and (1) extract seasonal shape, (2) generate time shift for new scnearios 
  4. gibbs_sampler.py --> Usage: Using the preprocessed seasonal shape and time shifts (1) run the hierarchical Gibbs sampler algortihm and produce parameter posteriors, (2) forward simulate and generate new scenarios 
