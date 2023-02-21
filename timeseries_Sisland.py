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
from cycler import cycler
import time


numpy.set_printoptions(threshold=sys.maxsize)


simulations = [5.3, 4.2, 5.7, 2.3]
evaluations = [4.7, 4.3, 5.5, 2.7]

nse = he.evaluator(he.nse, simulations, evaluations)

kge, r, alpha, beta = he.evaluator(he.kge, simulations, evaluations)

import itertools



columns = ['date','R']
columnslys = ['date','P','R','ET']

path = 'C:/Users/mathijs/Desktop/DATASETS/modeltimeseries/*'
all_files = glob.glob(path + "/*.csv")
pathlys = 'C:/Users/mathijs/Desktop/DATASETS/lysimeterdata/*'
all_fileslys = glob.glob(path +"/*.csv")

print(all_files)

#nse = he.evaluator(he.nse, simulations, evaluations)

#kge, r, alpha, beta = he.evaluator(he.kge, simulations, evaluations)



#LYSOMETER DATA IMPORTS
custom_date_parser = lambda x: datetime.strptime(x, "%d/MM/YYYY")
dfly_airp = pd.read_csv('C:/Users/mathijs/Desktop/DATASETS/lysimeterdata/1airportlys.csv',header = None, names=columnslys, parse_dates=['date'],infer_datetime_format=True).set_index(['date'])
dfly_horo = pd.read_csv('C:/Users/mathijs/Desktop/DATASETS/lysimeterdata/2hororatalys.csv',  header = None, names=columnslys, parse_dates=['date']).set_index(['date'])
dfly_linc = pd.read_csv('C:/Users/mathijs/Desktop/DATASETS/lysimeterdata/3lincolnlys.csv',  header = None, names=columnslys, parse_dates=['date'],infer_datetime_format=True).set_index(['date'])
dfly_winch = pd.read_csv('C:/Users/mathijs/Desktop/DATASETS/lysimeterdata/4winchmorelys.csv', header = None, names=columnslys, parse_dates=['date'],infer_datetime_format=True).set_index(['date'])
lysdata = [dfly_airp,dfly_horo,dfly_linc,dfly_winch]

print('woawowwie looooooooookkkkk',lysdata[0].to_string(), 'ennnnnnnnnnddd')
#dfly_airp,dfly_horo,dfly_linc,dfly_winch = [pd.read_csv(all_fileslys[0],header = None, names=columnslys)],[pd.read_csv(all_fileslys[1],header = None, names=columnslys)],[pd.read_csv(all_fileslys[2],header = None, names=columnslys)],[pd.read_csv(all_fileslys[3],header = None, names=columnslys)]   #output [dataframe] [dataframe] [dataframe] [dataframe]

#dfly_airp[0]['date'] = pd.to_datetime(dfly_airp[0]['date'], format='%d/%m/%Y');




#import lysimeter csv & aggregate to monthly
for dfl in range(len(lysdata)):
    lysdata[dfl] = lysdata[dfl].resample("M").sum()

columnsET, columnsP, columnsR, columnsswd, = ['date','PML','GLDAS','MOD16','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM','GPM3H','ERA5','VCSN','GLDAS','CHIRPS','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']
siteList = ['christ airport', 'hororata', 'lincoln', 'winchmore']
print(lysdata,'Lyssddaaaaattttaa')

lysdata[3].plot()
print(lysdata[3].to_string())

#select data rows based on date index (from)
lysdata[0]=lysdata[0].loc['1-1-2003':'1-1-2008']; lysdata[1]=lysdata[1].loc['1-1-2003':'1-1-2008'];lysdata[2]=lysdata[2].loc['1-1-2003':'1-1-2008'];lysdata[3]=lysdata[3].loc['1-1-2003':'1-1-2008']
# dfly_horo[]
# dfly_linc[]
# dfly_winch[]



#initialize
dfmET_all=[]; dfmP_all=[]; dfmR_all=[]; df_mswd_all=[];

lyRlist = []
lyPlist = []
lyETlist= []

dfstatsET, dfstatsP, dfstatsR= [pd.DataFrame(columns=['location','member','kge', 'r', 'alpha', 'beta'])]*3

print('links',all_files[0::4])
for (mET,mP,mR,mswd,lsite,lysd) in zip(all_files[0::4],all_files[1::4],all_files[2::4],all_files[3::4],siteList[0:4],lysdata):                           #for each lysimeter location... chirst, hororata, lincoln, winchmore
    dfmET, dfmP,dfmR,df_mswd = [pd.read_csv(mET,header = None, names=columnsET, parse_dates=['date'])],[pd.read_csv(mP,header = None, names=columnsP, parse_dates=['date'])],[pd.read_csv(mR,header = None, names=columnsR, parse_dates=['date'])],[pd.read_csv(mswd,header = None, names=columnsswd, parse_dates=['date'])]   #output [dataframe] [dataframe] [dataframe] [dataframe]
    print('misc3', dfmR)
    #lysidata
    lysR = lysd['R']; lysP = lysd['P']; lysR = lysd['ET']; Llys = len(lysd['R'])

    #get same length lys vs model
    dfmET[0]=dfmET[0].iloc[:Llys,:]; dfmP[0]=dfmP[0].iloc[:Llys,:]; dfmR[0]=dfmR[0].iloc[:Llys,:]; df_mswd[0]=df_mswd[0].iloc[:Llys,:]
    # same indexing for monthly aggregates
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

#print(dfmR_all)

# active_count_filename = glob.glob('Active_Count_*.csv')[0]
# df1 = pd.read_csv(active_count_filename)
# print(dfstats,'dfstatsszzzzzzzzzzzzzz')


#stats bar plots ET,P,R



# plt=pd.DataFrame.plot(dfmET[0]['ens spread'])
# plt.show()

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



#print(dfmET_all[0][['PML','GLDAS','MOD16']],'formatttt?')
#plt.figure(); a=dfmR_all[1][['GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS']].plot(); plt.legend(loc='best');

#plt.show()
#print(dfmET)

#make, and redistribute colour palet for the ensemble members
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

print(dfmR_all[2].to_string(),'hereallllllllllllllllllllllllllll')

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
        print(axs[loca])
        for cm, cr in zip(clmlist,colrlist):                                                                                                                                    #for each line
            #print(cm)

            axs[loca].plot(dfs[0][['date']],dfs[loca][[cm]],label=cm+' KGE: {0}'.format(round(stats.loc[siteList[loca],cm]['kge'],2)), color=cr)                                         # chist airp, EThor, linc, winchmore
            axs[loca].plot(dfs[0][['date']], dfs[loca][['lysimeter']], label=cm + 'lysimeter', color='black')
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

    for loca in range(4):
        #print(axs[loca])
        for cm, cr in zip(clmlist,colrlist):
            #print(cm)

            axs[loca].plot(dfs[0][['date']],dfs[loca][[cm]],label=cm+' KGE: {0}'.format(round(stats.loc[siteList[loca],cm]['kge'],2)), color=cr)                                         # chist airp, EThor, linc, winchmore
            axs[loca].plot(dfs[0][['date']], dfs[loca][['lysimeter']], label=cm + 'lysimeter', color='black')
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
    figbar, axbar = plt.subplots(1,4, sharex=False,figsize=(30, 5),sharey=False)
    axbar = axbar.flatten()
    figbar.suptitle('Temporal analysis {0}'.format(strdf), fontsize=16)
    df[df.columns[0:4]].loc['christ airport'].plot(ax=axbar[0], kind='bar'); axbar[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
          ncol=4, fancybox=True, shadow=True); axbar[0].title.set_text('Christchurch Airport')
    df[df.columns[0:4]].loc['hororata'].plot(ax=axbar[1], kind='bar');axbar[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
          ncol=4, fancybox=True, shadow=True); axbar[1].title.set_text('Hororata')
    df[df.columns[0:4]].loc['lincoln'].plot(ax=axbar[2], kind='bar',legend=1);axbar[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
          ncol=4, fancybox=True, shadow=True); axbar[2].title.set_text('Lincoln')
    df[df.columns[0:4]].loc['winchmore'].plot(ax=axbar[3], kind='bar',legend=1);axbar[3].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
          ncol=4, fancybox=True, shadow=True); axbar[3].title.set_text('Winchmore')
    figbar.subplots_adjust(bottom=0.25)






###################filtered plots kge >0.4 ##############################
#########################################################################
#########################################################################
#########################################################################

#copy past of for loop to recalculate spread and rmse filtered:

columnsET, columnsP, columnsR, columnsswd, = ['date','PML','GLDAS','MOD16','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM','GPM3H','ERA5','VCSN','GLDAS','CHIRPS','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']


#initialize
dfmET_all=[]; dfmP_all=[]; dfmR_all=[]; df_mswd_all=[];

lyRlist = []
lyPlist = []
lyETlist= []

dfstatsET, dfstatsP, dfstatsR= [pd.DataFrame(columns=['location','member','kge', 'r', 'alpha', 'beta'])]*3

print('links',all_files[0::4])




for (mET,mP,mR,mswd,lsite,lysd) in zip(all_files[0::4],all_files[1::4],all_files[2::4],all_files[3::4],siteList[0:4],lysdata):                           #for each lysimeter location... chirst, hororata, lincoln, winchmore
    #import using old column titles
    columnsET, columnsP, columnsR, columnsswd, = ['date','PML','GLDAS','MOD16','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM','GPM3H','ERA5','VCSN','GLDAS','CHIRPS','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean'],['date','GPM_PML','GPM_GLDAS','GPM_MOD16','GPM3H_PML','GPM3H_GLDAS','GPM3H_MOD16','ERA5_PML','ERA5_GLDAS','ERA5_MOD16','VCSN_PML','VCSN_GLDAS','VCSN_MOD16','GLDAS_PML','GLDAS_GLDAS','GLDAS_MOD16','CHIRPS_PML','CHIRPS_GLDAS','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']
    dfmET, dfmP,dfmR,df_mswd = [pd.read_csv(mET,header = None, names=columnsET, parse_dates=['date'])],[pd.read_csv(mP,header = None, names=columnsP, parse_dates=['date'])],[pd.read_csv(mR,header = None, names=columnsR, parse_dates=['date'])],[pd.read_csv(mswd,header = None, names=columnsswd, parse_dates=['date'])]   #output [dataframe] [dataframe] [dataframe] [dataframe]


    #filter imports kge>0.4 (select take only those columns)
    dfmET[0] = dfmET[0][['date','PML','MOD16','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean']]
    dfmP[0]= dfmP[0][['date','GPM','GPM3H','ERA5','VCSN','lysimeter','ens mean','ens spread','ens rmse','RMSElys_mean']]
    dfmR[0] = dfmR[0][['date','GPM_GLDAS','ERA5_GLDAS','VCSN_PML','VCSN_MOD16','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']]     #columns are in random order!

    #update to new column titles
    columnsET, columnsP, columnsR, columnsswd, = ['date', 'PML', 'MOD16', 'lysimeter', 'ens mean', 'ens spread','ens rmse', 'RMSElys_mean'], ['date', 'GPM', 'GPM3H', 'ERA5', 'VCSN','lysimeter', 'ens mean', 'ens spread','ens rmse', 'RMSElys_mean'], ['date','GPM_GLDAS','ERA5_GLDAS','VCSN_PML','VCSN_MOD16','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean'], ['date', 'GPM_PML', 'GPM_GLDAS', 'GPM_MOD16', 'GPM3H_PML','GPM3H_GLDAS', 'GPM3H_MOD16', 'ERA5_PML', 'ERA5_GLDAS','ERA5_MOD16', 'VCSN_PML', 'VCSN_GLDAS', 'VCSN_MOD16', 'GLDAS_PML','GLDAS_GLDAS', 'GLDAS_MOD16', 'CHIRPS_PML', 'CHIRPS_GLDAS','CHIRPS_MOD16', 'lysimeter', 'ens mean', 'ens spread','RMSElys_mean']

    print('yeapp',dfmR[0].to_string())
    #dfmET, dfmP, dfmR, df_mswd =

    #lysidata
    lysR = lysd['R']; lysP = lysd['P']; lysR = lysd['ET']; Llys = len(lysd['R'])

    #get same length lys vs model
    dfmET[0]=dfmET[0].iloc[:Llys,:]; dfmP[0]=dfmP[0].iloc[:Llys,:]; dfmR[0]=dfmR[0].iloc[:Llys,:]; df_mswd[0]=df_mswd[0].iloc[:Llys,:]
    # same indexing for monthly aggregates
    lysd['ET'].index = dfmET[0].index
    lysd['P'].index = dfmET[0].index
    lysd['R'].index = dfmET[0].index
    #print(lysd['R'].index)


    dfmET[0]['ens mean'], dfmP[0]['ens mean'], dfmR[0]['ens mean'], df_mswd[0]['ens mean'] = dfmET[0].mean(axis=1), dfmP[0].mean(axis=1),dfmR[0].mean(axis=1),df_mswd[0].mean(axis=1)  #ensemble mean per timestep                                                          #ensemble averages per time step
    dfmET[0]['ens spread'], dfmP[0]['ens spread'], dfmR[0]['ens spread'], df_mswd[0]['ens spread'] = dfmET[0].max(axis=1).subtract(dfmET[0].min(axis=1)), dfmP[0].max(axis=1).subtract(dfmP[0].min(axis=1)), dfmR[0].max(axis=1).subtract(dfmR[0].min(axis=1)),df_mswd[0].max(axis=1).subtract(df_mswd[0].min(axis=1))   #rowwise calculate diff max min

    #__R__
    #calc stats
    print(dfmR[0].to_string(),'dfmRr')
    kge, r, alpha, beta = he.evaluator(he.kge, dfmR[0].iloc[:, 1:6], lysd['R']);                           #calc stats of Recharge, location x, !!! dfmR zelfde column volgorde als columnsR
    rmse = he.evaluator(he.rmse, dfmR[0].iloc[:, 1:6], lysd['R']);                                         #out: 18 rmse values rmse between lys and each of the model simulations
    membermean_rmse = [mean(rmse)]*5                                                                                   #out: list 18 copies av rmse(lys vs ensmember)
    rmse_ensmean = he.evaluator(he.rmse, dfmR[0]['ens mean'], lysd['R']); rmse_ensmean=[float(rmse_ensmean)]*5
    tav_spread = [mean(dfmR[0]['ens spread'])]*5
    print('kge here',kge, tav_spread)                                                                                                                   #stats into df

    #create multi-index, stats to df
    member=columnsR[1:6];                                                               #list like [1,2,3,1,2,3,1,2,3]
    lsites=[lsite]*len(columnsR[1:6]);                                                  #list with copies [1,1,1,2,2,2,3,3,3]
    dflocationi=pd.DataFrame({'member':member,'location':lsites,'kge': kge, 'r': r, 'alpha': alpha, 'beta': beta,'rmse_ly_ensmember':rmse,'mean_rmse(ly_ensmember)':membermean_rmse,'rmse(ly_ensmean) (mm/day)':rmse_ensmean,'tav_spread(maxt-mint) (mm/day)':tav_spread}).sort_values(by="kge",ascending=False)   #make dataframe (for appending), set indices to model combinames, sort based on kge
    dfstatsR=pd.concat([dfstatsR,dflocationi])
    print('xxxxxxxx',dfstatsR.to_string())

    #__P__
    #calc stats
    kge, r, alpha, beta = he.evaluator(he.kge, dfmP[0].iloc[:, 1:5],   #take kge of 5 P series
    lysd['P']);  # calc stats of Recharge, location x
    rmse = he.evaluator(he.rmse, dfmP[0].iloc[:, 1:5], lysd['P']);  # out: 18 rmse values rmse between lys and each of the model simulations
    membermean_rmse = [mean(rmse)] * 4                              # out: list 18 copies av rmse(lys vs ensmember),  the '4' is to make a list of copies to make multi-index stats dataframe
    rmse_ensmean = he.evaluator(he.rmse, dfmP[0]['ens mean'], lysd['P']);
    rmse_ensmean = [float(rmse_ensmean)] * 4
    tav_spread = [mean(dfmP[0]['ens spread'])] * 4
                                                                                                                   #stats into df

    # create multi-index, stats to df
    member=columnsP[1:5];                                                                 #list like [1,2,3,1,2,3,1,2,3]
    lsites=[lsite]*len(columnsP[1:5]);                                                    #list with copies [1,1,1,2,2,2,3,3,3]
    dflocationi = pd.DataFrame({'member':member,'location':lsites,'kge': kge, 'r': r, 'alpha': alpha, 'beta': beta, 'rmse_ly_ensmember': rmse,'mean_rmse(ly_ensmember)': membermean_rmse, 'rmse(ly_ensmean) (mm/day)': rmse_ensmean,'tav_spread(maxt-mint) (mm/day)': tav_spread}).sort_values(by="kge", ascending=False)  # make dataframe (for appending), set indices to model combinames, sort based on kge
    dfstatsP = pd.concat([dfstatsP, dflocationi])
    print('xxxxxxxx',dfstatsP.to_string())

    #__ET__
    #calc stats
    kge, r, alpha, beta = he.evaluator(he.kge, dfmET[0].iloc[:, 1:3],
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





print('look here for peace',dfstatsP.to_string())

##PLOTTING filter stats for >0.4


#INCLUDE ONLY SERIES THAT ARE FILTERED FOR
dfmR_all[0]=dfmR_all[0][['date','GPM_GLDAS','ERA5_GLDAS','VCSN_PML','VCSN_MOD16','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmR_all[1]=dfmR_all[1][['date','GPM_GLDAS','ERA5_GLDAS','VCSN_PML','VCSN_MOD16','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmR_all[2]=dfmR_all[2][['date','GPM_GLDAS','ERA5_GLDAS','VCSN_PML','VCSN_MOD16','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']];dfmR_all[3]=dfmR_all[3][['date','GPM_GLDAS','ERA5_GLDAS','VCSN_PML','VCSN_MOD16','CHIRPS_MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']]
dfmP_all[0]=dfmP_all[0][['date','GPM','GPM3H','ERA5','VCSN','lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmP_all[1]=dfmP_all[1][['date','GPM','GPM3H','ERA5','VCSN','lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmP_all[2]=dfmP_all[2][['date','GPM','GPM3H','ERA5','VCSN','lysimeter','ens mean','ens spread','RMSElys_mean']];dfmP_all[3]=dfmP_all[3][['date','GPM','GPM3H','ERA5','VCSN','lysimeter','ens mean','ens spread','RMSElys_mean']]
dfmET_all[0]=dfmET_all[0][['date','PML','MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmET_all[1]=dfmET_all[1][['date','PML','MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']]; dfmET_all[2]=dfmET_all[2][['date','PML','MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']];dfmET_all[3]=dfmET_all[3][['date','PML','MOD16','lysimeter','ens mean','ens spread','RMSElys_mean']]



# dfstatsET = dfstatsET[dfstatsET.index.isin(['PML', 'MOD16'], level=1)]
# dfstatsP = dfstatsP[dfstatsP.index.isin(['VCSN', 'GPM3H','GPM','ERA5'], level=1)]

#dfstatsR = dfstatsR.loc[(slice(None),['VCSN_PML','ERA5_GLDAS','VCSN_MOD16','CHIRPS_MOD16','GPM_GLDAS'])]#[dfstatsR.index.isin(['VCSN_PML','ERA5_GLDAS','VCSN_MOD16','GPM_GLDAS','CHIRPS_MOD16'], level=1)]

print('wrong',dfmR_all)
#PLOT AGAIN NOW ONLY SOME LINES
for dfs,strdf, cols,stats in zip([dfmET_all,dfmP_all,dfmR_all],['ET','P','R'],[columnsET, columnsP, columnsR],[dfstatsET,dfstatsP,dfstatsR]):

    fig, axs = plt.subplots(2,2, sharex=False,figsize=(17, 9), sharey=False)
    plt.subplots_adjust(wspace=0.4)

    #print(axs)
    axs=axs.flatten()

    fig.suptitle('Temporal analysis {0}'.format(strdf), fontsize=16)

    #for colr, mem in zip()
    clmlist=list(dfs[0])[1:-4]
    print(clmlist)
    #print(clmlist,len(cols),'heremate')
    colrlist = (cmp(np.linspace(0, 1, len(clmlist))) * 255).astype(np.uint8)                                            #zip loops over shortest so define enough colors
    colrlist = [to_hex(cmp(v)) for v in np.linspace(0, 1, len(clmlist))]
    print('collist',clmlist)
    for loca in range(4):
        #print(axs[loca])
        lp = True
        for cm, cr in zip(clmlist,colrlist):

            if lp == True:                                                                                              #adding legend with kge values
                axs[loca].plot(dfs[0][['date']], dfs[loca][['lysimeter']], label='lysimeter', color='black',zorder=1)
                lp=False
            axs[loca].plot(dfs[0][['date']],dfs[loca][[cm]],label=cm+' KGE: {0}'.format(round(stats.loc[siteList[loca],cm]['kge'],2)), color=cr)                                         # chist airp, EThor, linc, winchmore
            axs[loca].plot(dfs[0][['date']], dfs[loca][['lysimeter']], color='black', zorder=2)
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


            axs[loca].legend(bbox_to_anchor=(1, 1.05))








plt.show()


















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















