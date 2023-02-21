# QUICK FOERIER TRANSFORM OF VARIENCE AND SPREAD, IS THERE ANY SEASONALITY?
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = pd.read_csv('C:/Users/mathijs/Desktop/DATASETS/FIGURES/fourrier/4kahalyss.csv', parse_dates=['date'],infer_datetime_format=True).set_index(['date'])

data = data['ET'].astype(float).values
print(data)

N = data.shape[0] #number of elements
t = np.linspace(0, 300, N)
#t=np.arange(N)
s = data

fft = abs(np.fft.fft(s))
fftfreq = np.fft.fftfreq(len(s))

T = t[1] - t[0]
print(T)

f = np.linspace(0, 1 / T, N)
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.plot(fftfreq, np.absolute(fft))
#plt.xlim(0,1)

#plt.show()












#########################different way
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import tukey

from numpy.fft import fft, fftshift
import pandas as pd



gtrend = pd.read_csv('C:/Users/mathijs/Desktop/DATASETS/FIGURES/fourrier/4kahalyss.csv', parse_dates=['date'],infer_datetime_format=True).set_index(['date'])#pd.read_csv('multiTimeline.csv',index_col=0,skiprows=2)
gtrend['date']=gtrend.index
gtrend['date']=gtrend["date"].astype("datetime64[M]")
gtrend.index=gtrend['date']; gtrend=gtrend.drop(columns=['date'])
gtrend=gtrend.rolling(6).mean()
gtrend=gtrend
print(gtrend.dtypes)
gtrendd=gtrend.to_string()
print(gtrendd)
# for index, row in gtrend.iterrows():
#     gtrend.loc[index,'date']= gtrend.loc[index,'date'].replace(day=1)
#     gtrend.index= gtrend.index.replace(day=1)
# print(gtrend)




a_gtrend_orig = gtrend['ET']
t_gtrend_orig = np.linspace( 0, len(a_gtrend_orig)/12, len(a_gtrend_orig), endpoint=False )

a_gtrend_windowed = (a_gtrend_orig-np.median( a_gtrend_orig ))*tukey( len(a_gtrend_orig) )

plt.subplot( 2, 1, 1 )
plt.plot( t_gtrend_orig, a_gtrend_orig, label='raw data'  )
plt.plot( t_gtrend_orig, a_gtrend_windowed, label='windowed data' )
plt.xlabel( 'years' )
plt.legend()

a_gtrend_psd = abs(rfft( a_gtrend_orig ))
a_gtrend_psdtukey = abs(rfft( a_gtrend_windowed ) )

# Notice that we assert the delta-time here,
# It would be better to get it from the data.
a_gtrend_freqs = rfftfreq( len(a_gtrend_orig), d = 1./12. )

# For the PSD graph, we skip the first two points, this brings us more into a useful scale
# those points represent the baseline (or mean), and are usually not relevant to the analysis
plt.subplot( 2, 1, 2 )
plt.plot( a_gtrend_freqs[1:], a_gtrend_psd[1:], label='psd(power spectral density) raw data' )
plt.plot( a_gtrend_freqs[1:], a_gtrend_psdtukey[1:], label='windowed psd' )
plt.xlabel( 'frequency ($yr^{-1}$)' )
plt.legend()

plt.tight_layout()
plt.show()

#https://stackoverflow.com/questions/52690632/analyzing-seasonality-of-google-trend-time-series-using-fft

gtrend.plot()
plt.show()