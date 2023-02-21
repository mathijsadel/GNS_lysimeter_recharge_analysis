import pandas as pd
import glob
import hydroeval as he
import sys
import numpy
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.colors import to_hex
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date
from cycler import cycler
import time
print(mean([0.67,0.19,0.56,0.52,0.42]))

numpy.set_printoptions(threshold=sys.maxsize)


simulations = [5.3, 4.2, 5.7, 2.3]
evaluations = [4.7, 4.3, 5.5, 2.7]

nse = he.evaluator(he.nse, simulations, evaluations)

kge, r, alpha, beta = he.evaluator(he.kge, simulations, evaluations)

import itertools



columns = ['date','R']
columnslys = ['date','P','R','ET']

path = 'C:/Users/mathijs/Desktop/DATASETS/modeltimeseries_2ndanalysis/*'
all_files = glob.glob(path + "/*.csv")
pathlys = 'C:/Users/mathijs/Desktop/DATASETS/lysimeterdata_NI/*'
all_fileslys = glob.glob(path +"/*.csv")

print('b1',all_files)

#nse = he.evaluator(he.nse, simulations, evaluations)

#kge, r, alpha, beta = he.evaluator(he.kge, simulations, evaluations)



#LYSOMETER DATA IMPORTS
custom_date_parser = lambda x: datetime.strptime(x, "%d/MM/YYYY")
dfly_ashc = pd.read_csv('C:/Users/mathijs/Desktop/DATASETS/lysimeterdata_NI/1ashclys.csv',header = None, names=columnslys, parse_dates=['date'],infer_datetime_format=True).set_index(['date'])
dfly_bripa = pd.read_csv('C:/Users/mathijs/Desktop/DATASETS/lysimeterdata_NI/2bripalys.csv',  header = None, names=columnslys, parse_dates=['date'],infer_datetime_format=True).set_index(['date'])
dfly_fern = pd.read_csv('C:/Users/mathijs/Desktop/DATASETS/lysimeterdata_NI/3fernlys.csv',  header = None, names=columnslys, parse_dates=['date'],infer_datetime_format=True).set_index(['date'])
dfly_kaha = pd.read_csv('C:/Users/mathijs/Desktop/DATASETS/lysimeterdata_NI/4kahalys.csv', header = None, names=columnslys, parse_dates=['date'],infer_datetime_format=True).set_index(['date'])
dfly_mara = pd.read_csv('C:/Users/mathijs/Desktop/DATASETS/lysimeterdata_NI/5maralys.csv', header = None, names=columnslys, parse_dates=['date'],infer_datetime_format=True).set_index(['date'])
lysdata = [dfly_ashc,dfly_bripa,dfly_fern,dfly_kaha, dfly_mara]

print('b11',lysdata[0].to_string(),lysdata[1].to_string(),lysdata[2].to_string(),lysdata[3].to_string(), 'b111')











# #preview characteristics of lysimeter data
# #experiment with moving average sinosoidal ET_daily (from daily P-R), sinosoidal effect
# dfly_kaha.rolling(365).mean().plot(title='moving average 365 day window')
# dfly_kaha.rolling(270).mean().plot(title='moving average 270 day window')
# dfly_kaha.rolling(180).mean().plot(title='moving average 180 day window')
# dfly_kaha.rolling(90).mean().plot(title='moving average 90 day window')
# dfly_kaha.rolling(46).mean().plot(title='moving average 46 day window')






#dfly_airp,dfly_horo,dfly_linc,dfly_winch = [pd.read_csv(all_fileslys[0],header = None, names=columnslys)],[pd.read_csv(all_fileslys[1],header = None, names=columnslys)],[pd.read_csv(all_fileslys[2],header = None, names=columnslys)],[pd.read_csv(all_fileslys[3],header = None, names=columnslys)]   #output [dataframe] [dataframe] [dataframe] [dataframe]

#dfly_airp[0]['date'] = pd.to_datetime(dfly_airp[0]['date'], format='%d/%m/%Y');

print('b122',dfly_bripa.resample("M").sum(),'b12',dfly_bripa.to_string())


#import lysimeter csv & aggregate to monthly
for dfl in range(len(lysdata)):
    lysdata[dfl] = lysdata[dfl].resample("M").sum()

columnsET, columnsP, columnsR, columnsswd, = ['date','PML','GLDAS','MOD16','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM','GPM3H','ERA5','VCSN','GLDAS','CHIRPS','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']
siteList = ['ashcott', 'bridge pa', 'fern hill', 'kaharoa','maraekakaho']
print('monthly agregates lysimeter data:', lysdata[3].to_string(),'Lyssddaaaaattttaa')


# #quick plot for all lys data per R, P, ET
# print('b001',lysdata[0].join(lysdata[1]).join(lysdata[2]).join(lysdata[3]).join(lysdata[4]).join(lysdata[5]))
lysdata[3].plot()
lysdata[2].plot()

#select data rows based on date index (from)
lysdata[0]=lysdata[0].loc['1-08-2013':'1-1-2015']; lysdata[1]=lysdata[1].loc['1-01-2013':'1-1-2015'];lysdata[2]=lysdata[2].loc['1-1-2013':'1-1-2015'];lysdata[3]=lysdata[3].loc['1-1-2013':'1-1-2015'];lysdata[4]=lysdata[4].loc['1-1-2013':'1-1-2015']
# dfly_horo[]
# dfly_linc[]
# dfly_winch[]

print(lysdata,'--> lysdata' )

#initialize
dfmET_all=[]; dfmP_all=[]; dfmR_all=[]; df_mswd_all=[];

lyRlist = []
lyPlist = []
lyETlist= []

dfstatsET, dfstatsP, dfstatsR= [pd.DataFrame(columns=['location','member','kge', 'r', 'alpha', 'beta'])]*3

print('links',all_files[1::4], siteList[0:5],lysdata)
for (mET,mP,mR,mswd,lsite,lysd) in zip(all_files[0::4],all_files[1::4],all_files[2::4],all_files[3::4],siteList[0:5],lysdata):                           #for each lysimeter location... chirst, hororata, lincoln, winchmore
    dfmET, dfmP,dfmR,df_mswd = [pd.read_csv(mET,header = None, names=columnsET, parse_dates=['date'])],[pd.read_csv(mP,header = None, names=columnsP, parse_dates=['date'])],[pd.read_csv(mR,header = None, names=columnsR, parse_dates=['date'])],[pd.read_csv(mswd,header = None, names=columnsswd, parse_dates=['date'])]   #output [dataframe] [dataframe] [dataframe] [dataframe]
    print('misc3', dfmR)
    #lysidata
    lysR = lysd['R']; lysP = lysd['P']; lysET = lysd['ET'];
    Llys = len(lysd['R'])
    #getindex in model data of first lys date, get first day first month of aggregated month of lysimeter series, last en first dates are the same for R,ET,P cause same dataframe
    fdatelysd = lysd.first_valid_index().replace(day=1)
    ldatelysd = lysd.last_valid_index().replace(day=1)


    #get same length lys vs model # starting and ending with same date.    Using booleans
    dfmET[0]=dfmET[0][(dfmET[0]['date']>=fdatelysd) & (dfmET[0]['date']<=ldatelysd)];
    dfmP[0]=dfmP[0][(dfmP[0]['date']>=fdatelysd) & (dfmP[0]['date']<=ldatelysd)];
    dfmR[0]=dfmR[0][(dfmR[0]['date']>=fdatelysd) & (dfmR[0]['date']<=ldatelysd)];
    df_mswd[0]=df_mswd[0][(df_mswd[0]['date']>=fdatelysd) & (df_mswd[0]['date']<=ldatelysd)]

    # same indexing for monthly aggregates
    print('b1001', dfmET[0])
    lysd['ET'].index = dfmET[0].index
    lysd['P'].index = dfmET[0].index
    lysd['R'].index = dfmET[0].index
    #print(lysd['R'].index)


    dfmET[0]['ens mean'], dfmP[0]['ens mean'], dfmR[0]['ens mean'], df_mswd[0]['ens mean'] = dfmET[0].mean(axis=1), dfmP[0].mean(axis=1),dfmR[0].mean(axis=1),df_mswd[0].mean(axis=1)  #ensemble mean per timestep                                                          #ensemble averages per time step
    dfmET[0]['ens spread'], dfmP[0]['ens spread'], dfmR[0]['ens spread'], df_mswd[0]['ens spread'] = dfmET[0].max(axis=1).subtract(dfmET[0].min(axis=1)), dfmP[0].max(axis=1).subtract(dfmP[0].min(axis=1)), dfmR[0].max(axis=1).subtract(dfmR[0].min(axis=1)),df_mswd[0].max(axis=1).subtract(df_mswd[0].min(axis=1))

    #__R__
    #calc stats
    kge, r, alpha, beta = he.evaluator(he.kge, dfmR[0].iloc[:, 1:19], lysd['R']);                           #calc stats of Recharge, location x
    rmse = he.evaluator(he.rmse, dfmR[0].iloc[:, 1:19], lysd['R']);                                         #out: 18 rmse values rmse between lys and each of the model simulations
    membermean_rmse = [mean(rmse)]*18                                                                                   #out: list 18 copies av rmse(lys vs ensmember)
    rmse_ensmean = he.evaluator(he.rmse, dfmR[0]['ens mean'], lysd['R']); rmse_ensmean=[float(rmse_ensmean)]*18
    tav_spread = [mean(dfmR[0]['ens spread'])]*18
    # print('!!!!!!', membermean_spread)                                                                                                                   #stats into df

    #create multi-index, stats to df
    member=columnsR[1:19];                                                               #list like [1,2,3,1,2,3,1,2,3]
    lsites=[lsite]*len(columnsR[1:19]);                                                  #list with copies [1,1,1,2,2,2,3,3,3]
    dflocationi=pd.DataFrame({'member':member,'location':lsites,'kge': kge, 'r': r, 'alpha': alpha, 'beta': beta,'rmse_ly_ensmember':rmse,'mean_rmse(ly_ensmember)':membermean_rmse,'rmse(ly_ensmean) (mm/day)':rmse_ensmean,'tav_spread(maxt-mint) (mm/day)':tav_spread}).sort_values(by="kge",ascending=False)   #make dataframe (for appending), set indices to model combinames, sort based on kge
    dfstatsR=pd.concat([dfstatsR,dflocationi])



    #__P__
    #calc stats
    kge, r, alpha, beta = he.evaluator(he.kge, dfmP[0].iloc[:, 1:7],
    lysd['P']);  # calc stats of Recharge, location x
    rmse = he.evaluator(he.rmse, dfmP[0].iloc[:, 1:7], lysd['P']);  # out: 18 rmse values rmse between lys and each of the model simulations
    membermean_rmse = [mean(rmse)] * 6  # out: list 18 copies av rmse(lys vs ensmember)
    rmse_ensmean = he.evaluator(he.rmse, dfmP[0]['ens mean'], lysd['P']);
    rmse_ensmean = [float(rmse_ensmean)] * 6
    tav_spread = [mean(dfmP[0]['ens spread'])] * 6
                                                                                                                    #stats into df

    # create multi-index, stats to df
    member=columnsP[1:7];                                                                 #list like [1,2,3,1,2,3,1,2,3]
    lsites=[lsite]*len(columnsP[1:7]);                                                    #list with copies [1,1,1,2,2,2,3,3,3]
    dflocationi = pd.DataFrame({'member':member,'location':lsites,'kge': kge, 'r': r, 'alpha': alpha, 'beta': beta, 'rmse_ly_ensmember': rmse,'mean_rmse(ly_ensmember)': membermean_rmse, 'rmse(ly_ensmean) (mm/day)': rmse_ensmean,'tav_spread(maxt-mint) (mm/day)': tav_spread}).sort_values(by="kge", ascending=False)  # make dataframe (for appending), set indices to model combinames, sort based on kge
    dfstatsP = pd.concat([dfstatsP, dflocationi])
    print(dfstatsP.to_string(), "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    #__ET__
    #calc stats
    kge, r, alpha, beta = he.evaluator(he.kge, dfmET[0].iloc[:, 1:4],
    lysd['ET']);  # calc stats of Recharge, location x
    rmse = he.evaluator(he.rmse, dfmET[0].iloc[:, 1:4], lysd['ET']);  # out: 18 rmse values rmse between lys and each of the model simulations
    membermean_rmse = [mean(rmse)] * 3  # out: list 18 copies av rmse(lys vs ensmember)
    rmse_ensmean = he.evaluator(he.rmse, dfmET[0]['ens mean'], lysd['ET']);
    rmse_ensmean = [float(rmse_ensmean)] * 3
    tav_spread = [mean(dfmET[0]['ens spread'])] * 3
    # print('!!!!!!', membermean_spread)                                                                                                                   #stats into df

    # create multi-index, stats to df
    member=columnsET[1:4]; #print(member);                                                                  #list like [1,2,3,1,2,3,1,2,3]
    lsites=[lsite]*len(columnsET[1:4]); #print(lsites)                                                      #list with copies [1,1,1,2,2,2,3,3,3]
    dflocationi = pd.DataFrame({'member':member,'location':lsites,'kge': kge, 'r': r, 'alpha': alpha, 'beta': beta, 'rmse_ly_ensmember': rmse,'mean_rmse(ly_ensmember)': membermean_rmse, 'rmse(ly_ensmean) (mm/day)': rmse_ensmean,'tav_spread(maxt-mint) (mm/day)': tav_spread}).sort_values(by="kge", ascending=False)  # make dataframe (for appending), set indices to model combinames, sort based on kge
    dfstatsET = pd.concat([dfstatsET, dflocationi])
    print('ET!!!!!!!!',dfstatsET.to_string(),'P',dfstatsP.to_string(),'R', dfstatsR.to_string())


    #add lys data at location 1/4 to dataframe eg dfmET (dfmET contains data for location i)

    dfmET[0]["lysimeter"] = lysd['ET']
    dfmP[0]["lysimeter"] = lysd['P']
    dfmR[0]["lysimeter"] = lysd['R']
    #print(lysd['P'],lysd['R'],dfmET[0]["lysimeter"],dfmP[0]["lysimeter"],dfmR[0]["lysimeter"])
    # print(lysd['P'], 'lyssssssssssssssssssssssssssssssss')
     #prepa
    dfmET_all += dfmET
    dfmP_all += dfmP
    dfmR_all += dfmR
    df_mswd_all += df_mswd

    #for later use in plotting:
    lyRlist += [lysd['R']]
    lyPlist += [lysd['P']]
    lyETlist +=[lysd['ET']]


print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',dfmR[0])
#print(lyRlist, 't he a waosme ness')
dfstatsET=dfstatsET.set_index(['location','member'])
dfstatsP=dfstatsP.set_index(['location','member'])
dfstatsR=dfstatsR.set_index(['location','member'])



#print(dfstatsET.to_string(),dfstatsP.to_string(),dfstatsR.to_string(),)

# ax = dfstatsET.unstack(level=0).plot(kind='line', subplots=True, rot=0, figsize=(9, 7), layout=(4,8))
# plt.tight_layout()
# print(dfstatsET.to_string())
# print(dfstatsP.to_string())
# print(dfstatsR.to_string())

# print(dfmR_all)

# active_count_filename = glob.glob('Active_Count_*.csv')[0]
# df1 = pd.read_csv(active_count_filename)
# print(dfstats,'dfstatsszzzzzzzzzzzzzz')


#stats bar plots ET,P,R


#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',dfmP[(dfmP['date']>datetime.date(2008,1,1)) & (dfmP['date']<datetime.date(2008,3,1))])
#plt=pd.DataFrame.plot(dfmET[0]['ens spread'])
#plt.show()

#print(columnsR[1:19])                                                                                                  #make dataframe with values

# dfstats['kge'],dfstats['r'],dfstats['alpha'],dfstats['beta'] = kge, r, alpha, beta                                      #stats into df
# dfstats = dfstats.set_index([columnsR[1:19]]).sort_values(by="kge",ascending=False)                                     #sort


# custom_cycler = (cycler(color=['c', 'm', 'y', 'k']) +
#                  cycler(lw=[1, 2, 3, 4]))
# fig, axs = plt.subplots(2, 2, constrained_layout=True,figsize=(13, 9))
# print(fig,'lolookkkoo')
#
#
# for irc,idf in zip([[0,0],[0,1],[1,0],[1,1]],[0,1,2,3]):      #row collums right?  axs [[axes axes][axes axes]],  [0,1] row column
#         print(irc[0],irc[1],idf)
#         axs[irc[0]][irc[1]].plot(dfmET_all[idf][['date']],dfmET_all[idf][['PML','GLDAS','MOD16']],label=['lijntje','j','fjj'])
#         axs[irc[0]][irc[1]].set_title('subplot 1')
#         axs[irc[0]][irc[1]].set_xlabel('lead time days')
#         axs[irc[0]][irc[1]].set_ylabel('et mm/d')
#         #axs[irc[0]][irc[1]].set_prop_cycle(['red', 'black', 'yellow'])
#         axs[irc[0]][irc[1]].legend(loc='best')
# fig.suptitle('Temporal analysis ET', fontsize=16)
# plt.show()
# print('lollolollllooll',dfmET_all[0].iloc[:,1:4])#a'])
#
# print(axs)
#
#
#
# #print(dfmET_all[0][['PML','GLDAS','MOD16']],'formatttt?')
# #plt.figure(); a=dfmR_all[1][['GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS']].plot(); plt.legend(loc='best');
#
# #plt.show()
# #print(dfmET)
#
# #make, and redistribute colour palet for the ensemble members
colors = [#[0, 82, 246, 255],
          #[0, 196, 196, 255],
          #[0, 137, 83, 255],
          [1, 233, 11, 255],
          #[234, 255, 31, 255],
          [255, 176, 0, 255],
          [247, 19, 0, 255],
          #[193, 0, 76, 255],
          #[255, 0, 255, 255]
        ]

cmp = LinearSegmentedColormap.from_list('', np.array(colors) / 255, 256)

print('b15.5',dfmR_all,'b15.55',dfmR_all[1].to_string(),dfmR_all[2].to_string())

for dfs,strdf, cols,stats,lyR,lyP,lyET in zip([dfmET_all,dfmP_all,dfmR_all],['ET','P','R'],[columnsET, columnsP, columnsR],[dfstatsET,dfstatsP,dfstatsR],lyRlist,lyPlist,lyETlist):  #for each ET, P, R #lyRlist list containing 4 lists R at each site

    fig, axs = plt.subplots(2,2, sharex=True,figsize=(13, 9), sharey=True)
    #print(axs)
    axs=axs.flatten()
    fig.suptitle('Temporal analysis {0}'.format(strdf), fontsize=16)

    #for colr, mem in zip()
    clmlist=list(dfs[0])[1:-5]
    #print(clmlist,len(cols),'heremate')
    colrlist = (cmp(np.linspace(0, 1, len(clmlist))) * 255).astype(np.uint8)                                                                                                   #zip loops over shortest so define enough colors
    colrlist = [to_hex(cmp(v)) for v in np.linspace(0, 1, len(clmlist))]

    for loca in range(4):                                                                                                                                                       #for each location
        print('b16',axs[loca])
        for cm, cr in zip(clmlist,colrlist):                                                                                                                                    #for each line

            print(dfs[loca][['date']],'b17')
            #print(cm)
            print(dfs[loca][[cm]],'b18',loca,cm)
            axs[loca].plot(dfs[loca][['date']],dfs[loca][[cm]],label=cm+' KGE: {0}'.format(round(stats.loc[siteList[loca],cm]['kge'],2)), color=cr)                                         # chist airp, EThor, linc, winchmore
            axs[loca].plot(dfs[loca][['date']], dfs[loca][['lysimeter']], label=cm + 'lysimeter', color='black')
            axs[loca].legend(loc='best')
            axs[loca].set_xlabel('lead time (months)')
            axs[loca].set_ylabel('{} (mm/month)'.format(strdf))
            axs[loca].title.set_text('{0}'.format(siteList[loca]))
            #print('loll')

#for col in kr:
    #ind1 = iplot * len(year_str_list) + 1
    #ind2 = iplot * len(year_str_list) + len(year_str_list) + 1
    #ax1.plot(dfmET_all[0][['date']],dfmET_all[0]['PML']) #label=scenario_str, color=my_plot_colours[iplot], marker='o', mfc=my_plot_colours[iplot])  #






#plot again, text with stats kge, membermean rmse,mena spread mean
for dfs,strdf, cols,stats in zip([dfmET_all,dfmP_all,dfmR_all],['ET','P','R'],[columnsET, columnsP, columnsR],[dfstatsET,dfstatsP,dfstatsR]):

    fig, axs = plt.subplots(2,2, sharex=True,figsize=(17, 9), sharey=True)
    #print(axs)
    axs=axs.flatten()
    fig.suptitle('Temporal analysis {0}'.format(strdf), fontsize=16)

    #for colr, mem in zip()
    clmlist=list(dfs[0])[1:-5]
    #print(clmlist,len(cols),'heremate')
    colrlist = (cmp(np.linspace(0, 1, len(clmlist))) * 255).astype(np.uint8)                                            #zip loops over shortest so define enough colors
    colrlist = [to_hex(cmp(v)) for v in np.linspace(0, 1, len(clmlist))]
    print('b244', clmlist)
    for loca in range(4):
        #print(axs[loca])
        for cm, cr in zip(clmlist,colrlist):
            #print(cm)

            axs[loca].plot(dfs[loca][['date']],dfs[loca][[cm]],label=cm+' KGE: {0}'.format(round(stats.loc[siteList[loca],cm]['kge'],2)), color=cr)                                         # chist airp, EThor, linc, winchmore
            axs[loca].plot(dfs[loca][['date']], dfs[loca][['lysimeter']], label=cm + 'lysimeter', color='black')
            #axs[loca].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0))
            axs[loca].set_xlabel('lead time months')
            axs[loca].set_ylabel('{} (mm/month)'.format(strdf))
            axs[loca].title.set_text('{0}'.format(siteList[loca]))
            # print(round(mean(stats.loc[siteList[loca],:]['kge']),2))
            a=round(mean(stats.loc[siteList[loca],:]['kge']),2); b=round(mean(stats.loc[siteList[loca],:]['mean_rmse(ly_ensmember)']),2); c=round(mean(stats.loc[siteList[loca],:]['tav_spread(maxt-mint) (mm/day)']),2);
            boxstring = '\n'.join((
                r'$\mathrm{KGE}=%.2f$' % (a, ),
                r'$\overline{RMSE} = %.2f$ (mm/month)' % (b, ),
                r'$\overline{spread} = %.2f$ (mm/month)' % (c, )))
            props = dict(boxstyle='square', facecolor='white', alpha=0.5);
            axs[loca].text(0.05, 0.95, boxstring, transform=axs[loca].transAxes, fontsize=10,
                    verticalalignment='top', bbox=props);
            #print('loll')




#bar plots stats
for df,strdf in zip([dfstatsET,dfstatsP,dfstatsR],['ET','P','R']):
    figbar, axbar = plt.subplots(1,5, sharex=False,figsize=(90, 5),sharey=False)
    axbar = axbar.flatten()
    figbar.suptitle('Temporal analysis {0}'.format(strdf), fontsize=16)
    df[df.columns[0:4]].loc['ashcott'].plot(ax=axbar[0], kind='bar'); axbar[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),ncol=4, fancybox=True, shadow=True); axbar[0].title.set_text('Ashcott')
    df[df.columns[0:4]].loc['bridge pa'].plot(ax=axbar[1], kind='bar',legend=0);#axbar[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),ncol=4, fancybox=True, shadow=True);
    axbar[1].title.set_text('Bridge Pa')
    df[df.columns[0:4]].loc['fern hill'].plot(ax=axbar[2], kind='bar',legend=0);#axbar[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),ncol=4, fancybox=True, shadow=True);
    axbar[2].title.set_text('Fern Hill')
    df[df.columns[0:4]].loc['kaharoa'].plot(ax=axbar[3], kind='bar',legend=0);#axbar[3].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),ncol=4, fancybox=True, shadow=True);
    axbar[3].title.set_text('Kaharoa')
    df[df.columns[0:4]].loc['maraekakaho'].plot(ax=axbar[4], kind='bar',legend=0);#axbar[4].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),ncol=4, fancybox=True, shadow=True);
    axbar[4].title.set_text('Maraekakaho')
    figbar.subplots_adjust(bottom=0.25)
    # df[df.columns[0:4]].loc['maraekakaho'].plot(ax=axbar[0], kind='bar'); axbar[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
    #       ncol=4, fancybox=True, shadow=True); axbar[0].title.set_text('maraekakaho')















###############################calculate averages return list higher then 0.4 kge#################################
##################################################################################################################
MkgeP_permember = dfstatsP.groupby(level=['member']).mean().sort_values(by="kge",ascending=False)
MkgeET_permember = dfstatsET.groupby(level=['member']).mean().sort_values(by="kge",ascending=False)
MkgeR_permember = dfstatsR.groupby(level=['member']).mean().sort_values(by="kge",ascending=False)

print('b109','best members across lysimeters','P___',MkgeP_permember,'ET___',MkgeET_permember,'R___', MkgeR_permember)#,MkgeET_permember,MkgeR_permember)#.groupby(level=['location']).sum())

#best performing lysimeters?
MkgeP_perlocation = dfstatsP.groupby(level=['location']).mean().sort_values(by="kge",ascending=False)
MkgeET_perlocation = dfstatsET.groupby(level=['location']).mean().sort_values(by="kge",ascending=False)
MkgeR_perlocation = dfstatsR.groupby(level=['location']).mean().sort_values(by="kge",ascending=False)
print('b110','best lysimeters:','P___',MkgeP_perlocation,'ET___',MkgeET_perlocation,'R___', MkgeR_perlocation)



#group count across sites, how many times a member has kge>0.4 , then groupby member and sum count, sort
dfstatsP['KGElt04?'] = np.where(dfstatsP['kge']>=0.4,1,0); count04P = dfstatsP.groupby(level=['member']).sum().sort_values(by="KGElt04?",ascending=False);
dfstatsET['KGElt04?'] = np.where(dfstatsET['kge']>=0.4,1,0); count04ET = dfstatsET.groupby(level=['member']).sum().sort_values(by="KGElt04?",ascending=False);
dfstatsR['KGElt04?'] = np.where(dfstatsR['kge']>=0.4,1,0); count04R = dfstatsR.groupby(level=['member']).sum().sort_values(by="KGElt04?",ascending=False);
#list members that had exceeded 0.4 kge more then twice
Ptopcount_gt04=count04P['KGElt04?'].loc[count04P['KGElt04?']>=2].index.tolist()
ETtopcount_gt04=count04ET['KGElt04?'].loc[count04ET['KGElt04?']>=2].index.tolist()
Rtopcount_gt04=count04R['KGElt04?'].loc[count04R['KGElt04?']>=2].index.tolist()

print('b210 top counts exceeding 0.4 kge','P__',Ptopcount_gt04,'ET__',ETtopcount_gt04,'R__',Rtopcount_gt04)
print('b213',dfstatsET, 'still all members are present')






###################filtered plots kge >0.4 ##############################
#########################################################################
#########################################################################
#########################################################################

#copy past of for loop to recalculate spread and rmse filtered:

#columnsET, columnsP, columnsR, columnsswd, = ['date','PML','GLDAS','MOD16','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM','GPM3H','ERA5','VCSN','GLDAS','CHIRPS','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']
#updated for this north island analysis (I left the list above with all the members just for comparison), this list is later used for plotting and so defined outside the for loop that calculates the stats
columnsET, columnsP, columnsR, columnsswd, = ['date','PML','GLDAS','MOD16','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GLDAS', 'VCSN', 'GPM3H','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM3H_MOD16', 'VCSN_PML', 'ERA5_MOD16', 'ERA5_PML', 'GPM_PML', 'GPM3H_GLDAS', 'ERA5_GLDAS','lysimeter','ens mean','ens spread','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']


#initialize
dfmET_all=[]; dfmP_all=[]; dfmR_all=[]; df_mswd_all=[];

lyRlist = []
lyPlist = []
lyETlist= []

dfstatsET, dfstatsP, dfstatsR= [pd.DataFrame(columns=['location','member','kge', 'r', 'alpha', 'beta'])]*3

print('links',all_files[0::4], 'take every fourth link from list starting at 0')
print('b4001','lysdata has right lenghts and includes all !5! series:',lysdata)


for (mET,mP,mR,mswd,lsite,lysd) in zip(all_files[0::4],all_files[1::4],all_files[2::4],all_files[3::4],siteList[0:5],lysdata):                           #for each lysimeter location... chirst, hororata, lincoln, winchmore
    #import using old column titles, so all gets imported
    columnsET, columnsP, columnsR, columnsswd, = ['date','PML','GLDAS','MOD16','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM','GPM3H','ERA5','VCSN','GLDAS','CHIRPS','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']
    dfmET, dfmP,dfmR,df_mswd = [pd.read_csv(mET,header = None, names=columnsET, parse_dates=['date'])],[pd.read_csv(mP,header = None, names=columnsP, parse_dates=['date'])],[pd.read_csv(mR,header = None, names=columnsR, parse_dates=['date'])],[pd.read_csv(mswd,header = None, names=columnsswd, parse_dates=['date'])]   #output [dataframe] [dataframe] [dataframe] [dataframe]

    print('b4002 this is site:',lsite)
    #filter imports kge>0.4 (select take only those member colums based on >=2 count of exceedence of kge>0.4 across sites variable)
    dfmP[0]= dfmP[0][['date']+Ptopcount_gt04+['lysimeter','ens mean','ens spread','RMSElys_mean']]
    dfmR[0] = dfmR[0][['date']+Rtopcount_gt04+['lysimeter','ens mean','ens spread','RMSElys_mean']]     #columns are in random order!
    dfmET[0] = dfmET[0][['date']+['PML', 'GLDAS']+['lysimeter','ens mean','ens spread','RMSElys_mean']]         #IMPORTANT NOTE: here i force pml and gldas to be included in the filtered ensemble. Otherwise non would be included
    #dfmET[0] = dfmET[0][['date']+ETtopcount_gt04+['lysimeter','ens mean','ens spread','RMSElys_mean']]

    #update to new column titles
    columnsET, columnsP, columnsR, columnsswd, = ['date','PML','GLDAS','MOD16','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GLDAS', 'VCSN', 'GPM3H','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM3H_MOD16', 'VCSN_PML', 'ERA5_MOD16', 'ERA5_PML', 'GPM_PML', 'GPM3H_GLDAS', 'ERA5_GLDAS','lysimeter','ens mean','ens spread','RMSElys_mean'], ['date', 'GPM_PML', 'GPM_GLDAS', 'GPM_MOD16', 'GPM3H_PML','GPM3H_GLDAS', 'GPM3H_MOD16', 'ERA5_PML', 'ERA5_GLDAS','ERA5_MOD16', 'VCSN_PML', 'VCSN_GLDAS', 'VCSN_MOD16', 'GLDAS_PML','GLDAS_GLDAS', 'GLDAS_MOD16', 'CHIRPS_PML', 'CHIRPS_GLDAS','CHIRPS_MOD16', 'lysimeter', 'ens mean', 'ens spread','RMSElys_mean']



    #lysidata
    lysR = lysd['R']; lysP = lysd['P']; lysET = lysd['ET'];
    Llys = len(lysd['R'])
    #getindex in model data of first lys date, get first day first month of aggregated month of lysimeter series, last en first dates are the same for R,ET,P cause same dataframe
    fdatelysd = lysd.first_valid_index().replace(day=1)
    ldatelysd = lysd.last_valid_index().replace(day=1)


    #get same length lys vs model # starting and ending with same date.    Using booleans
    dfmET[0]=dfmET[0][(dfmET[0]['date']>=fdatelysd) & (dfmET[0]['date']<=ldatelysd)];
    dfmP[0]=dfmP[0][(dfmP[0]['date']>=fdatelysd) & (dfmP[0]['date']<=ldatelysd)];
    dfmR[0]=dfmR[0][(dfmR[0]['date']>=fdatelysd) & (dfmR[0]['date']<=ldatelysd)];
    df_mswd[0]=df_mswd[0][(df_mswd[0]['date']>=fdatelysd) & (df_mswd[0]['date']<=ldatelysd)]

    # same indexing for monthly aggregates
    print('b1001', dfmET[0])
    lysd['ET'].index = dfmET[0].index
    lysd['P'].index = dfmET[0].index
    lysd['R'].index = dfmET[0].index
    #print(lysd['R'].index)



    dfmET[0]['ens mean'], dfmP[0]['ens mean'], dfmR[0]['ens mean'], df_mswd[0]['ens mean'] = dfmET[0].mean(axis=1), dfmP[0].mean(axis=1),dfmR[0].mean(axis=1),df_mswd[0].mean(axis=1)  #ensemble mean per timestep                                                          #ensemble averages per time step
    dfmET[0]['ens spread'], dfmP[0]['ens spread'], dfmR[0]['ens spread'], df_mswd[0]['ens spread'] = dfmET[0].max(axis=1).subtract(dfmET[0].min(axis=1)), dfmP[0].max(axis=1).subtract(dfmP[0].min(axis=1)), dfmR[0].max(axis=1).subtract(dfmR[0].min(axis=1)),df_mswd[0].max(axis=1).subtract(df_mswd[0].min(axis=1))   #rowwise calculate diff max min

    #__R__
    #calc stats
    print(dfmR[0].to_string(),'dfmRr')
    kge, r, alpha, beta = he.evaluator(he.kge, dfmR[0].loc[:,Rtopcount_gt04], lysd['R']);                  #select topcount columns and calc statistics         #calc stats of Recharge, location x, !!! dfmR zelfde column volgorde als columnsR
    rmse = he.evaluator(he.rmse, dfmR[0].loc[:, Rtopcount_gt04], lysd['R']);                                         #out: 18 rmse values rmse between lys and each of the model simulations
    membermean_rmse = [mean(rmse)]*len(Rtopcount_gt04)                                                                                   #out: list 18 copies av rmse(lys vs ensmember)
    rmse_ensmean = he.evaluator(he.rmse, dfmR[0]['ens mean'], lysd['R']); rmse_ensmean=[float(rmse_ensmean)]*len(Rtopcount_gt04)
    tav_spread = [mean(dfmR[0]['ens spread'])]*len(Rtopcount_gt04)
    print('kge here',kge, tav_spread)                                                                                                                   #stats into df

    #create multi-index, stats to df
    member=Rtopcount_gt04;                                                               #member list like [1,2,3,1,2,3,1,2,3]
    lsites=[lsite]*len(Rtopcount_gt04);                                                 #site list with copies [1,1,1,2,2,2,3,3,3], these two lists are put 'next' to each other for double indexing
    dflocationi=pd.DataFrame({'member':member,'location':lsites,'kge': kge, 'r': r, 'alpha': alpha, 'beta': beta,'rmse_ly_ensmember':rmse,'mean_rmse(ly_ensmember)':membermean_rmse,'rmse(ly_ensmean) (mm/day)':rmse_ensmean,'tav_spread(maxt-mint) (mm/day)':tav_spread}).sort_values(by="kge",ascending=False)   #make dataframe (for appending), set indices to model combinames, sort based on kge
    dfstatsR=pd.concat([dfstatsR,dflocationi])
    print('xxxxxxxx',dfstatsR.to_string())

    #__P__
    #calc stats
    kge, r, alpha, beta = he.evaluator(he.kge, dfmP[0].loc[:,Ptopcount_gt04],   #take kge of 5 P series
    lysd['P']);  # calc stats of Recharge, location x
    rmse = he.evaluator(he.rmse, dfmP[0].loc[:, Ptopcount_gt04], lysd['P']);  # out: 18 rmse values rmse between lys and each of the model simulations
    membermean_rmse = [mean(rmse)] * len(Ptopcount_gt04)                              # out: list 18 copies av rmse(lys vs ensmember),  the '4' is to make a list of copies to make multi-index stats dataframe
    rmse_ensmean = he.evaluator(he.rmse, dfmP[0]['ens mean'], lysd['P']);
    rmse_ensmean = [float(rmse_ensmean)] * len(Ptopcount_gt04)
    tav_spread = [mean(dfmP[0]['ens spread'])] * len(Ptopcount_gt04)
                                                                                                                   #stats into df

    # create multi-index, stats to df
    member=Ptopcount_gt04;                                                                 #list like [1,2,3,1,2,3,1,2,3]
    lsites=[lsite]*len(Ptopcount_gt04);                                                    #list with copies [1,1,1,2,2,2,3,3,3]
    dflocationi = pd.DataFrame({'member':member,'location':lsites,'kge': kge, 'r': r, 'alpha': alpha, 'beta': beta, 'rmse_ly_ensmember': rmse,'mean_rmse(ly_ensmember)': membermean_rmse, 'rmse(ly_ensmean) (mm/day)': rmse_ensmean,'tav_spread(maxt-mint) (mm/day)': tav_spread}).sort_values(by="kge", ascending=False)  # make dataframe (for appending), set indices to model combinames, sort based on kge
    dfstatsP = pd.concat([dfstatsP, dflocationi])
    print('xxxxxxxx',dfstatsP.to_string())

    #__ET__
    #calc stats
    kge, r, alpha, beta = he.evaluator(he.kge, dfmET[0].iloc[:, 1:3],                      #note: index 1:3 is used instead of .loc(... ETtopcount_gt04 ) as I force to included at least 2 members
    lysd['ET']);  # calc stats of Recharge, location x
    rmse = he.evaluator(he.rmse, dfmET[0].iloc[:, 1:3], lysd['ET']);  # out: 18 rmse values rmse between lys and each of the model simulations
    membermean_rmse = [mean(rmse)] * 2  # out: list 18 copies av rmse(lys vs ensmember)
    rmse_ensmean = he.evaluator(he.rmse, dfmET[0]['ens mean'], lysd['ET']);
    rmse_ensmean = [float(rmse_ensmean)] * 2
    tav_spread = [mean(dfmET[0]['ens spread'])] * 2
    # print('!!!!!!', membermean_spread)                                                                                                                   #stats into df

    # create multi-index, stats to df
    member=columnsET[1:3]; #print(member);                                                                  #list like [1,2,3,1,2,3,1,2,3]
    lsites=[lsite]*len(columnsET[1:3]); #print(lsites)                                                      #list with copies [1,1,1,2,2,2,3,3,3]
    dflocationi = pd.DataFrame({'member':member,'location':lsites,'kge': kge, 'r': r, 'alpha': alpha, 'beta': beta, 'rmse_ly_ensmember': rmse,'mean_rmse(ly_ensmember)': membermean_rmse, 'rmse(ly_ensmean) (mm/day)': rmse_ensmean,'tav_spread(maxt-mint) (mm/day)': tav_spread}).sort_values(by="kge", ascending=False)  # make dataframe (for appending), set indices to model combinames, sort based on kge
    dfstatsET = pd.concat([dfstatsET, dflocationi])
    #print('ET!!!!!!!!',dfstatsET.to_string(),'P',dfstatsP.to_string(),'R', dfstatsR.to_string())


    #add lys data at location 1/4 to dataframe eg dfmET (dfmET contains data for location i)

    dfmET[0]["lysimeter"] = lysd['ET']
    dfmP[0]["lysimeter"] = lysd['P']
    dfmR[0]["lysimeter"] = lysd['R']
    #print(lysd['P'],lysd['R'],dfmET[0]["lysimeter"],dfmP[0]["lysimeter"],dfmR[0]["lysimeter"])
    # print(lysd['P'], 'lyssssssssssssssssssssssssssssssss')
     #prepa
    dfmET_all += dfmET
    dfmP_all += dfmP
    dfmR_all += dfmR
    df_mswd_all += df_mswd

    #for later use in plotting:
    lyRlist += [lysd['R']]
    lyPlist += [lysd['P']]
    lyETlist +=[lysd['ET']]


#print(lyRlist, 't he a waosme ness')
dfstatsET=dfstatsET.set_index(['location','member'])
dfstatsP=dfstatsP.set_index(['location','member'])
dfstatsR=dfstatsR.set_index(['location','member'])

print(dfstatsR,'b123', '... same members as: topcount variable?',Rtopcount_gt04)



print('b4003, look here for peace of mind',dfstatsP.to_string())
print('b4004, dfmET_all',dfmET_all)









##PLOTTING filter stats for >0.4


#INCLUDE ONLY SERIES THAT ARE FILTERED FOR                              (making amount of columns from 'lysimeter' onwards the same for R,P,ET for indexing inloop)
dfmR_all[0]=dfmR_all[0][['date']+Rtopcount_gt04+['lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmR_all[1]=dfmR_all[1][['date']+Rtopcount_gt04+['lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmR_all[2]=dfmR_all[2][['date']+Rtopcount_gt04+['lysimeter','ens mean','ens spread','RMSElys_mean']];dfmR_all[3]=dfmR_all[3][['date']+Rtopcount_gt04+['lysimeter','ens mean','ens spread','RMSElys_mean']]
dfmP_all[0]=dfmP_all[0][['date']+Ptopcount_gt04+['lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmP_all[1]=dfmP_all[1][['date']+Ptopcount_gt04+['lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmP_all[2]=dfmP_all[2][['date']+Ptopcount_gt04+['lysimeter','ens mean','ens spread','RMSElys_mean']];dfmP_all[3]=dfmP_all[3][['date']+Ptopcount_gt04+['lysimeter','ens mean','ens spread','RMSElys_mean']]
dfmET_all[0]=dfmET_all[0][['date','PML','GLDAS','lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmET_all[1]=dfmET_all[1][['date','PML','GLDAS','lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmET_all[2]=dfmET_all[2][['date','PML','GLDAS','lysimeter','ens mean','ens spread','RMSElys_mean']];dfmET_all[3]=dfmET_all[3][['date','PML','GLDAS','lysimeter','ens mean','ens spread','RMSElys_mean']]



# dfstatsET = dfstatsET[dfstatsET.index.isin(['PML', 'MOD16'], level=1)]
# dfstatsP = dfstatsP[dfstatsP.index.isin(['VCSN', 'GPM3H','GPM','ERA5'], level=1)]

#dfstatsR = dfstatsR.loc[(slice(None),['VCSN_PML','ERA5_GLDAS','VCSN_MOD16','CHIRPS_MOD16','GPM_GLDAS'])]#[dfstatsR.index.isin(['VCSN_PML','ERA5_GLDAS','VCSN_MOD16','GPM_GLDAS','CHIRPS_MOD16'], level=1)]


#PLOT AGAIN NOW ONLY SOME LINES
for dfs,strdf, cols,stats in zip([dfmET_all,dfmP_all,dfmR_all],['ET','P','R'],[columnsET, columnsP, columnsR],[dfstatsET,dfstatsP,dfstatsR]):

    fig, axs = plt.subplots(3,2, sharex=False,figsize=(102, 54), sharey=False)                                            # define to-be-filled spaces for sub-plots
    plt.subplots_adjust(wspace=0.7)

    #print(axs)
    axs=axs.flatten()

    fig.suptitle('Temporal analysis {0}'.format(strdf), fontsize=16)

    #for colr, mem in zip()
    clmlist=list(dfs[0])[1:-4]
    print(clmlist,'b222')
    #print(clmlist,len(cols),'heremate')
    colrlist = (cmp(np.linspace(0, 1, len(clmlist))) * 255).astype(np.uint8)                                            #zip loops over shortest so define enough colors
    colrlist = [to_hex(cmp(v)) for v in np.linspace(0, 1, len(clmlist))]
    print('collist',clmlist)
    for loca in range(5):
        #print(axs[loca])
        lp = True
        for cm, cr in zip(clmlist,colrlist):
            print('ffhiero',dfs[loca][[cm]],'next')
            if lp == True:                                                                                              #adding legend with kge values
                axs[loca].plot(dfs[loca][['date']], dfs[loca][['lysimeter']], label='lysimeter', color='black',zorder=1)
                lp=False
            axs[loca].plot(dfs[loca][['date']],dfs[loca][[cm]],label=cm+' KGE: {0}'.format(round(stats.loc[siteList[loca],cm]['kge'],2)), color=cr)                                         # chist airp, EThor, linc, winchmore
            axs[loca].plot(dfs[loca][['date']], dfs[loca][['lysimeter']], color='black', zorder=2)
            #axs[loca].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0))
            axs[loca].set_xlabel('lead time months')
            axs[loca].set_ylabel('{} (mm/month)'.format(strdf))
            axs[loca].title.set_text('{0}'.format(siteList[loca]))
            # print(round(mean(stats.loc[siteList[loca],:]['kge']),2))
            a=round(mean(stats.loc[siteList[loca],:]['kge']),2); b=round(mean(stats.loc[siteList[loca],:]['mean_rmse(ly_ensmember)']),2); c=round(mean(stats.loc[siteList[loca],:]['tav_spread(maxt-mint) (mm/day)']),2);

            boxstring = '\n'.join((
                r'$\overline{\mathrm{KGE}}=%.2f$' % (a, ),
                r'$\overline{RMSE} = %.2f$ (mm/month)' % (b, ),
                r'$\overline{spread} = %.2f$ (mm/month)' % (c, )))
            props = dict(boxstyle='square', facecolor='white', alpha=0.5);
            axs[loca].text(0.05, 0.95, boxstring, transform=axs[loca].transAxes, fontsize=9,
                    verticalalignment='top', bbox=props);
            print('lp',lp)

            #put legend in top right corner of each subplot:
            axs[loca].legend(loc=1)#bbox_to_anchor=(0.7, 1.0))
























# columns = [[1,2,3],[1,4,5],[6,4,2]]
# colors = ['green', 'pink', 'blue']
# labels = ['foo', 'bar', 'baz']
#
# fig, ax = plt.subplots(2, 1,sharex=True)
# for i, column in enumerate(columns):
#     ax[0].plot(column, color=colors[i], label=labels[i])
#     ax[1].plot(column, color=colors[i], label=labels[i])
# ax[0].legend(loc='lower center', bbox_to_anchor=(0.82, 1),
#           fancybox=True, shadow=True, ncol=3)
# plt.show()




#print(dfmET[0].iloc[:,1:4])
#print(kge,r,alpha,beta)
# plot1=dfmET_all.iloc[:,1:].plot()
# plot2=dfmP_all.iloc[:,1:].plot()
# plot2=dfmR_all.iloc[:,1:].plot()
# plot2=df_mswd_all.iloc[:,1:].plot()

#df.rename(columns={"a":"b"}, inplace = True)

#print(dfstatsP.loc[dfstatsP.index[4]],'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',dfstatsP.index[4])





#print(dfmET_all[0].to_string(),dfmP_all[0].to_string(),dfmR_all[0].to_string())


plt.show()












