import glob
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from astroML.time_series import lomb_scargle, lomb_scargle_bootstrap

#returns a list of absolute paths for all the data files
def listTablePath():
	path='/net/xrb/MACHO/'
	return glob.glob(path+'F_**/*.*.*')
	
def npTable(target):
	#load into numpy array. if the side of pier is west return 1, else retun 0
	#dont bother reading in column 4, I dont know what the data mean, weird format
	t=np.loadtxt(target, skiprows=10, converters = {2: lambda s: True if s=='West' else False}, 
		usecols=[i for i in range(0,36) if i !=4])

	#mask bad values
	tt=ma.masked_values(t,-99.0)
	ttt=ma.masked_values(tt,999)
	return ttt

############################################################################################################################
#input: file name for a macho data ascii file                                                                              #
############################################################################################################################
#action:read in macho data ascii file to panda data frame                                                                  #
#based on information in the ascii file, compute the calibrated R and V magnitudes                                         #
#the procedure is outlined in detail by Alcock et al 1999.                                                                 #
#V = Vraw + a0 + (a1 + 0.022*Xtemplate)*(Vraw - Rraw) + 2.5*log10(ExpTime)                                                 #
#R = Rraw + b0 + (b1 + 0.004*Xtemplate)*(Vraw - Rraw) + 2.5*log10(ExpTime)                                                 #
#a0, b0, a1, b1 are zeropoint and color-correction coefficents. a0 and b0 depend on which amplifier on detector you're on  #
#a1 and b1 are determined emperically in the above paper                                                                   #
############################################################################################################################
#output: pandas data frame, including observation date, calibrated rmags, rErr, calibrated bmags, berror
def pdTable(target):
	#these are the column headers for the data
	#the headers in the txt file have a '!' at the start of the line, so recognizing them automatically is hard
	names=['Date','Obsid','Pier','Exposure','Checklist','Airmass','rMag','rErr',
		'rDS','rTF','rCP','rX2','rMP','rCR','rA','rXpix','rYpix','rSky','rFWHM',
		'rTobs','r','bMag','bErr','bDS','bTF','bCP','bX2','bMP','bCR','bA','bXpix',
		'bYpix','bSky','bFWHM','bTobs','b']
	
    #read in the data as a pandas table, store as variable 't'
	t=pd.read_table(target,skiprows=10,header=None,sep="\s+",names=names,
		na_values=[-99,999,9.999],comment='#',error_bad_lines=False,index_col=False)

	#grab template observation ID's. This will tell you the obsID of the template observation
    #we need the template observation so we can determine the template airmass to calibrate data
    #the template obs id is sometimes listed as 0 for rows which have bad or missing data. filter those out
	bTobsID=t['bTobs'][t['bTobs'] > 0].values[0]
	rTobsID=t['rTobs'][t['rTobs'] > 0].values[0]

	#grab the air mass for the template observation
	bTair=t[t['Obsid']==bTobsID]['Airmass'].values[0]
	rTair=t[t['Obsid']==rTobsID]['Airmass'].values[0]

	#grab the amplifier number for the observation. Divide by two and take floor
    #there are two amplifiers per ccd loral. a0 and b0 depend on which loral you are on 
    #These will be arrays
	bAmp=np.floor(t['bA']/2).values
	rAmp=np.floor(t['rA']/2).values

	#definitions for constants used in magnitude calculation
    #the keys are CCD numbers, the values are a0 or b0 for the CCD
	aOneDict={0:-.2059, 1:-1.876, 2:-2.065, 3:-2.059}
	bOneDict={4:0.1784, 5:0.1785, 6: 0.1868, 7:0.1784}

    #calculate a0 and b0 for every observation, store as numpy array. 
	aOne=np.asarray([aOneDict[i] if i > 0 else np.nan for i in rAmp ])
	bOne=np.asarray([bOneDict[i] if i > 0 else np.nan for i in bAmp ])

    #calculate a1 and b1. these are scalar values
	aZero= 18.410 - 0.279*rTair
	bZero= 18.087 - 0.222*bTair

	#calculate the calibrated V and R magnitudes for the object
	V=t['bMag'].values + aZero + (aOne + 0.022*rTair)*(t['bMag'] - t['rMag']) + 2.5 * np.log10(t['Exposure'])
	R=t['rMag'].values + bZero + (bOne + 0.004*bTair)*(t['bMag'] - t['rMag']) + 2.5 * np.log10(t['Exposure'])

	#return a pandas data frame and remove any rows that have NaNs 
	return pd.DataFrame({'Date':t['Date'].values, 'R':R.values, 'Rerr':t['rErr'], 'V':V.values, 'Verr':t['bErr'].values}).dropna(how='any')

############################################################################################################################
#input: pandas data frame including observation dates, calibrated V and R magnitudes and their instrumental errors         #
############################################################################################################################
#action: compute the lomb scargle periodogram of the time series photometry of input                                       #
#create plot of periodogram and display                                                                                    #
#largely stolen from Jake's code in astroML                                                                                #
############################################################################################################################
#output: returns pandas data frame, containing linear sample of 100000 pts between 0 and 10, and power spectra of values   #
############################################################################################################################
def powerSpectra(t):
    #period is an np array of 100000 points linearly spaced between 0 and 10
    #we will be looking for periods between 0 and 10 days. adjust if need be
    period = np.linspace(.1, 10, 100000)
    
    #omega is the angular frequency of the period
    omega = 2 * np.pi / period
	
    #compute lobm-scargle ps
    PS=lomb_scargle(t['Date'].values,t['R'].values,t['Rerr'].values, omega, generalized=True)
    
    PSwindow=lomb_scargle(t['Date'].values, np.ones_like(t['R'].values), 1 , omega, generalized=False, subtract_mean=False)
    
    #D = lomb_scargle_bootstrap(t['Date'].values,t['R'].values,t['Rerr'].values, omega, generalized=True, N_bootstraps=1000, random_state=0)
    #sig1, sig5 = np.percentile(D, [99, 95])
    #print str(sig5)+" "+str(sig1)
    
    #create quick plot of power spectra
    fig=plt.figure(figsize=(16,8))
    ax1=fig.add_subplot(211)
    ax1.plot(period, PS, '-', c='black', lw=1, zorder=1)
    ax1.set_ylabel('data PSD')
    ax1.set_ylim(0,1)
    #ax1.plot([period[0], period[-1]], [sig1, sig1], ':', c='black')
    #ax1.plot([period[0], period[-1]], [sig5, sig5], ':', c='black')
    ax2=fig.add_subplot(212)
    ax2.plot(period, PSwindow, '-', c='black', lw=1, zorder=1)
    ax2.set_ylabel('window PSD')
    
    
    plt.show()
    return pd.DataFrame({"PS":PS, "omega":omega, "period":period})

############################################################################################################################
#input: pandas data frame including observation dates, calibrated V and R magnitudes and their instrumental errors
#keyword: m, the number of terms to allow in multiver periodogram. default value is 5
############################################################################################################################
#action: compute the multiterm periodogram of the time series photometry of input                                          #
#create plot of periodogram and display                                                                                    #
#largely stolen from Jake's code in astroML                                                                                #
#This function is almost identical to powerSpectra() above.
############################################################################################################################
#output: returns pandas data frame, containing linear sample of 100000 pts between 0 and 10, and power spectra of values   #
############################################################################################################################
def powerSpectraM(t, omega0, factor, m=5, width=.03):
    #omega is the angular frequency of the period
    omega = np.linspace(omega0 - width, omega0 + width, 1000)
	
    #compute lobm-scargle ps
    PS=multiterm_periodogram(t['Date'].values,t['R'].values,t['Rerr'].values, omega / factor ,m)
    
    
    #create quick plot of power spectra
    fig=plt.figure(figsize=(16,8))
    ax1=fig.add_subplot(111)
    ax1.plot(omega / factor, PS, '-', c='black', lw=1, zorder=1)
    #ax1.plot([period[0], period[-1]], [sig1, sig1], ':', c='black')
    #ax1.plot([period[0], period[-1]], [sig5, sig5], ':', c='black')
    
    plt.show()
    return pd.DataFrame({"PS":PS, "omega":omega/ factor})

#string length algorithm
#input: pandas data frame with date, r magnitude, r error, v magnitude, v error columns
#action: for a range of potential periods, create folded light curves and evaluate the smoothness
#of each light curve using the string length method.
#output: pandas data frame with period and string length columns.
def stringLength(t, pstart=.1, pend=10, plength=10000):
    #get the time and r magnitude values as numpy arrays from the input data frame
    time=t['Date'].values
    mag=t['R'].values
    m=len(mag)
    
    #create a numpy array that has candidate periods
    period = np.linspace(pstart,pend,plength)
    
    #lengthList is a list that will hold the string lenght for each trial period.
    #it starts off empty, but we will populate it element by element
    lengthList=[]
    
    #for every trial period, calculate the phase of each date
    #order each (phase, magnitude) point in ascending order by phase
    #calculate the distance between the m-1 and mth point
    #add up all the distances and append this value to length list
    
    for i in period:
        phase=(time%i)/i
        order=np.argsort(phase)
        u=np.column_stack((phase[order], mag[order]))
        lengthList.append((((u[0:m-1]-u[1:m])**2).sum()))
    
    return pd.DataFrame({"String Length": np.array(lengthList),
                         "Period":period})


