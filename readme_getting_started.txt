Two lysimeter data sets were analysed: a set at Hawkesbay, North Island (NI_...), and a set at the Canterbury plains at the south Island (SI_..). 
The lysimeter time series analyses of these data sets can be found in the .py files:
timeseries_Nisland.py
timeseries_Sisland.py

These codes have are similar except for the fact that they analyse different data sets (e.g. NI and SI). The purpose of these scripts is twofold: 1. perform a statistical timeseries analysis that compares model and lysimeter recharge, percipitation and evapotransipiration ensemble members. 2. filter/excluded the worst performing ensemble members based on the outcomes of the statistical analysis.

For this the code contains the following steps: 
1. import packages  
2. import modelled and lysimeter field csv data. 
3. pre-fromating. (calculating montly aggregates, ..)
4. looping paralell over each lysimeter site, 
	Withing loop
	4a make correct date indexing
	4b matching lengths of time series
	4c match same indexing across model and lysimeter data (always put first day of the month as aggregate)
	4d calculate statistics for P R and ET: ensemble mean, ensemble spread (minimum - maximum value present at each time step), kge (and coefficients), rmse
	4e apply double indexing with first index lysimeter location, second index is ensemble member
out: double indexed dataframe with 

5 loop through use double indexed dataframe to build time series plots 
6 apply a similar loop to plot a bar graph with kge values. 
7 Apply filtering. determine at each location which model recharge members score kge>0.4. 
8 Count for each member how many times kge>0.4 is true across the locations. Only keep members that have a count of 3 or larger
	(Note: the script "timeseries_Sisland_old.py" is a the same. Only, the filtering condition is different. Members are excluded the a cross-site average kge score of kge<0.4)
9 redo steps 4 to 6 with the filtered ensemble set.





////Spatial analysis in GOOGLE EARTH ENGINE (GEE)///
model script with exports:

To obtain the model csv data of all possible P and ET input combination the scribt above was used. The model was used to run all combinations, whereafter the statistical outpus, such as a timeseries of maps, 
was exported as asset. 

The resulting (timeseries of) maps were imported in a separate script. This script models calculates the ensemble-RMSE, ensemble-spread, time averaged ensemble mean of outputted recharge, and the inputted percipitation and Evapotranspiration. The results are then mapped. More over maps are made of the difference between pre and post filtered RMSE and mean values. This to indicated the effect of filtering. 

link statitics script:


Link APP:

https://hijisvanadel.users.earthengine.app/view/gnsrecharge1 

The App was GEE-web app was integrated in an .exe file that can be run from your desktop. The App can be updated in the GEE code editor. The python based interface draws the web-app into the interface based its correspoding webadress.
(can also be found in the APP editor in GEE)
The python script for this GEE-App interface is:

GNSRechargeApp_master.py

Note: python 3.9 was used for all the python scripts above. 

 



