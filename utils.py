# utility function meant specifically for the 6OHDA project
# written by Dana Zemel
# last edited 10/10/2018 (most code originally written nov. 2017)

import numpy as np
import warnings
from scipy.signal import butter, filtfilt, lfilter
from scipy import interpolate
from scipy import signal
from chronux import *
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import pickle
from IO import *
from numpy import matlib


def getPowerInBand(f,Sxx,minf, maxf):
    # get the mean power in a frequency band:
    # input: 
    #     f - frequencies of spectogram
    #     Sxx - power output of spectogram
    #     fmin - lower lim of froqency band
    #     fmax - upper lim of frequency band
    
    # TODO: add sense check on variables
    ind = indices(f, lambda x: x >= minf and x<=maxf)
    return np.sum(Sxx[ind,:],0) 
    

def indices(a, func):
    # matlab equvalent of find from: 
    #   https://stackoverflow.com/questions/5957470/matlab-style-find-function-in-python
    return [i for (i, val) in enumerate(a) if func(val)]

def getPowerSpec(lfpDict):
    # This function takes in a dict with lfp data (that was returned from getData())
    # and returns the average power spectra
    # Inputs: 
    #   lfpDict - a dictionary with lfp data as returned from getData()
    # Outputs:
    #   M - mean power spectra
    #   Std - standard diviation of power spectra
    #   f - frequency list

    data = []
    for j in lfpDict:
        lfp = lfpDict[j]['lfp']['lfp']
        f, t, Sxx = signal.spectrogram(lfp[0,:],lfpDict[j]['lfp']['FS'],window=('hamming'),nperseg=140,noverlap =120,nfft=1200)

        Power = np.sum(Sxx,1)
        totPower = np.sum(Power)
        if totPower == 0:
            totPower=1
        #beta = np.mean(getPowerInBand(f,Sxx,13,20)/np.sum(Sxx,axis = 0)
        data.append(Power/totPower)

    data = np.array(data)
    M = np.mean(data,axis=0)
    Std = np.std(data, axis = 0)
    return M, Std, f

def FindOffset(speed, th=5,hi=40,hiWin=20,thWin=10,shift=-2):
    # This function identifies data offset based on analogy to Mike Romanos procedure for movment onset.
    # Inputs:
    #   speed - the linear speed at cm/s
    #   th - intial threshold (default 5cm/s)
    #   hi - high thershould (default 40cm/s)
    #   thWin - # consecutive data point below th
    #   hiWin - # window to check for data points abouve hi
    #   shift - # data point to shift left when detrmining onset
    # outputs:
    #   sOffset - booleen array with True at every mvmt onset time point
    
    # sense checks: 
    #--------------
    if not isinstance(speed,np.ndarray):
        raise ValueError('Speed must be numpy array')
    
    isVec, speed = checkIfVector(speed,shape='N')
    
    if not isVec:
        raise ValueError('Speed must be vector')
    
    parDfaults = {'th':5,'hi':40,'hiWin':20,'thWin':10,'shift':-2}
    for par in parDfaults.keys():
        try: 
            exec("%s = int(%s)" % (par,par))
        except ValueError as er:
            print(par+" must be an integer, value changed to default")
            exec("%s = %s" % (par,parDfaults[par]))
    # actual function:
    #-----------------
    sOffset = np.full(speed.shape, False) # container
    sTh = (speed > th)                  # find where speed croses th
    
    for s in range(11,len(sTh)-hiWin):    #TODO: Find the proper way to speed this up 
        if not sTh[s]:
            continue
        if np.sum(sTh[s+1:s+thWin+1])>0:   #TODO: make numData points dynamic
            continue
        if np.sum(speed[s-hiWin:s]<hi)<1:
            continue
        sOffset[s-shift] = True
    return sOffset

def smooth(data,wLen=10,win = 'flat', factor=1):
    # This function convolve data with window. When windoe is flat this is the 
    # moving average. This code is mainly taken from: 
    # from http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    # with minor alterations.
    # inputs:
    #    x = the data vecotor (must be a numeric nd-array vector!)
    #    WLen - window lenth in data point (int, must be smaller the len(x))
    #    win - window type (optional, must be in list of numpy win options)
    
    #   sense checks:
    #------------------
        
    isVec, data = checkIfVector(data,shape='N')
    if not (isinstance(data,np.ndarray) and isVec and data.dtype in ['float32','float64','int']):
         raise ValueError('Data must be numeric nd-array vector')

    try: 
        wLen = int(wLen)
    except ValueError as er:
        warnings.warn("window params must be integers, default values used")
        wLen=10
    
    if wLen > data.size:
        raise ValueError('window length must be shorter then the data')
    
    if wLen <3:
        return data
    
    if not win in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")    
    
    #   actual function:
    #------------------
    
    s=np.r_[np.flip(data[wLen-1:0:-1],0),data,np.flip(data[-2:-wLen-1:-1],0)]
    if win == 'flat': #moving average
        w=np.ones(wLen)*factor
    else:
        w=eval('np.'+win+'(wLen)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(np.floor(wLen/2)):-int(np.floor(wLen/2))+1]

def removeMAMin(x,maWlen,wlen):
    xSmooth = smooth(x,wLen=maWlen)
    minList = np.zeros_like(x)
    for i in range(0,xSmooth.size):
        sStart = max(0,i-wlen)
        sStop = min(xSmooth.size,i+wlen+1)
        minList[i] = min(xSmooth[sStart:sStop])
    return x-minList

def getCaOnset(trace,th,th2,wlenMA=10,wlenDetrend=500,oldPeaks=None):
    # This function take a Ca trace and retuen the Ca even onset point. This 
    # function is based on Mike Romano's procedure.
    # Inputs:
    #   trace - the Ca trace. Numeric numpy vector
    #   th - thershold on the Z-score (numeric)
    #   th2 - how many standard deviations for secound pass over traces (int)
    #   wlenM - window length for moving Average (int)
    #   wlenDetrend - window length for detrending (int)
    # Output:
    #   caOnset - boolean np vectore same size as trace with even onset marked as True
    
    # Sense Checks:
    #-------------------
    
    #TODO: fill this in...
    if oldPeaks==None:
        oldPeaks = np.zeros_like(trace,dtype=bool)
    #  Actual function:
    #-------------------------
    traceFilt = removeMAMin(trace,wlenMA,wlenDetrend)
    peakList = ((traceFilt-np.mean(traceFilt[~oldPeaks]))/np.std(traceFilt[~oldPeaks]))>th
    
    while not (np.sum(peakList) == 0 or np.sum(peakList) == np.sum(oldPeaks)):
        oldPeaks = np.logical_or(peakList,oldPeaks)
        peakList = ((traceFilt-np.mean(traceFilt[~oldPeaks]))/np.std(traceFilt[~oldPeaks]))>th
    
    mu = np.mean(traceFilt[~peakList])
    std = np.std(traceFilt[~peakList])
    
    peakDiff = np.append(np.asarray(peakList[1:],dtype=int)-np.asarray(peakList[:-1],dtype=int),0)
    peakStart = np.asarray([i for i, x in enumerate(peakDiff) if x == 1])
    peakEnd =   np.asarray([i for i, x in enumerate(peakDiff) if x == -1])
    
    # prevent erroring out if provided bad trace with no event or bad thersholds:
    if np.sum(peakStart.shape) == 0:
        caPeaks = np.zeros_like(trace,dtype=bool)
        diffCaPeaks= np.append(np.asarray(caPeaks[1:],dtype=int)-np.asarray(caPeaks[:-1],dtype=int),0)
        caOnset = diffCaPeaks == 1
        return caOnset,caPeaks

    peakEnd = peakEnd[[i > peakStart[0] for i in peakEnd ]]
    
    # prevent erroring out if provided bad trace with no event or bad thersholds:
    if np.sum(peakEnd.shape) == 0:
        caPeaks = np.zeros_like(trace,dtype=bool)
        diffCaPeaks= np.append(np.asarray(caPeaks[1:],dtype=int)-np.asarray(caPeaks[:-1],dtype=int),0)
        caOnset = diffCaPeaks == 1
        return caOnset,caPeaks
    
    peakStart = peakStart[[i < peakEnd[-1] for i in peakStart ]]
    
    caPeaks = np.zeros_like(trace,dtype=bool)
    for peak in range(0,peakStart.size):
        if np.max(traceFilt[peakStart[peak]:peakEnd[peak]])>mu+th2*std:
            caPeaks[peakStart[peak]:peakEnd[peak]] = 1
            
    diffCaPeaks= np.append(np.asarray(caPeaks[1:],dtype=int)-np.asarray(caPeaks[:-1],dtype=int),0)
    caOnset = diffCaPeaks == 1
    
    return caOnset,caPeaks

def alignToOnset(data,onset, winPre =10, winPost=20 ):
    #TODO: test case where data isn't vector..
    
    # this function allignes data based on even onset
    #Input:
    #   data - the data to be alligned (must be np array), assume axis=0 is time axis
    #   onset - a bool ndarray/list that is True on even onset
    #   winPre - how many data points before even onset
    #   winPost - how many data points post event onset
    #Output:
    #   Aligned - a window size X dataDim X numEvents array containing the aligned data
    
    # sense checks: 
    #--------------
    if not all(isinstance(i,np.ndarray) for i in [data, onset]):
        raise ValueError('onset and data must be numpy arrays')
    
    isVec, onset = checkIfVector(onset,shape='N')
    isVec2, data = checkIfVector(data,shape='(N,1)') 
    
    if not isVec or not onset.dtype =='bool':
        raise ValueError('Onset must be a bool vector')
    
    if data.shape[0]< len(onset):
        warnings.warn("data is shorter than onset. Trancuating onset vector")
        onset = onset[0:data.shape[0]]
    
       #if data is vector, make sure no extra dim 
    
    try: 
        winPre = int(winPre)
    except ValueError as er:
        warnings.warn("window params must be integers, default values used")
        winPre=10
    try: 
        winPost = int(winPost)
    except ValueError as er:
        warnings.warn("window params must be integers, default values used")
        winPost=20
   
    # actual function:
    #-----------------
    #TODO: test case where aligned is not a vector...
    aligned = np.empty((winPre+winPost,data.shape[1], np.sum(onset[winPre:-winPost])))
    j =0 # index for num true values
    for s in range(winPre,len(onset)-winPost):
        if onset[s]:
            aligned[:,:,j] = data[s-winPre:s+winPost]
            j = j+1
    
    aligned = np.squeeze(aligned)
    return aligned    

def checkIfVector(v,shape='N'):
    # this function make sure v is a vector and returns v with the desired shape
    # input: 
    #   v - potential vector
    #   shape - the final shape to be returned (options:['(1,N)','(N,1)','N'])
    if v.ndim>2:
        return False, v
    elif v.ndim==2 and not (v.shape[1] == 1 or v.shape[0] == 1):
        return False, v
    else:
        if not shape in ['(1,N)','(N,1)','N']:
            shape == 'N'
        N = np.max(v.shape)
        v.shape = eval(shape)
        return True, v



def FindMvmtOnset(speed, th=5,hi=40,hiWin=20,thWin=10,shift=2):
    # This function identifies movment onset based on Mike Romanos procedure.
    # all default settings chosen to match Mike's methodology from "Unique contributions
    # of parvalbumin and cholinergic interneurons in organizing striatal networks
    # during voluntary movement"
    # Inputs:
    #   speed - the linear speed at cm/s
    #   th - intial threshold (default 5cm/s)
    #   hi - high thershould (default 40cm/s)
    #   thWin - # consecutive data point below th
    #   hiWin - # window to check for data points abouve hi
    #   shift - # data point to shift left when detrmining onset
    # outputs:
    #   sOnset - booleen array with True at every mvmt onset time point
    
    # sense checks: 
    #--------------
    if not isinstance(speed,np.ndarray):
        raise ValueError('Speed must be numpy array')
    
    isVec, speed = checkIfVector(speed,shape='N')
    
    if not isVec:
        raise ValueError('Speed must be vector')
    
    parDfaults = {'th':5,'hi':40,'hiWin':20,'thWin':10,'shift':2}
    for par in parDfaults.keys():
        try: 
            exec("%s = int(%s)" % (par,par))
        except ValueError as er:
            print(par+" must be an integer, value changed to default")
            exec("%s = %s" % (par,parDfaults[par]))
    # actual function:
    #-----------------
    sOnset = np.full(speed.shape, False) # container
    smoothSpeed  = smooth(speed,20)
    sTh = (smoothSpeed >= th)                  # find where speed croses th
    
    for s in range(thWin+1,len(sTh)-hiWin):    #TODO: Find the proper way to speed this up 
        if not sTh[s]:
            continue
        if np.sum(sTh[s-thWin:s])>0:   #TODO: make numData points dynamic
            continue
        if np.sum(smoothSpeed[s:s+hiWin]>hi)<1:
            continue
        sOnset[s-shift] = True
    return sOnset

def FindMvmtOnset2(speed, th_weak=4,th_strong=2 ,hi=8,hiWin=20,thWin=20,shift=2):
    # This function identifies movment onset based on Mike Romanos procedure.
    # all default settings chosen to match Mike's methodology from "Unique contributions
    # of parvalbumin and cholinergic interneurons in organizing striatal networks
    # during voluntary movement"
    # Inputs:
    #   speed - the linear speed at cm/s
    #   th_weak - stationary period my not exceed this value more than once (default 4cm/s)
    #   th_strong - onset points must be below this value (default 2cm/s)
    #   hi - high thershould (default 40cm/s)
    #   thWin - # consecutive data point below th
    #   hiWin - # window to check for data points abouve hi
    #   shift - # data point to shift left when detrmining onset
    # outputs:
    #   sOnset - booleen array with True at every mvmt onset time point
    
    # sense checks: 
    #--------------
    if not isinstance(speed,np.ndarray):
        raise ValueError('Speed must be numpy array')
    
    isVec, speed = checkIfVector(speed,shape='N')
    
    if not isVec:
        raise ValueError('Speed must be vector')
    
    parDfaults = {'hi':40,'hiWin':20,'thWin':10,'shift':2}
    for par in parDfaults.keys():
        try: 
            exec("%s = int(%s)" % (par,par))
        except ValueError as er:
            print(par+" must be an integer, value changed to default")
            exec("%s = %s" % (par,parDfaults[par]))
    # actual function:
    #-----------------
    sOnset = np.full(speed.shape, False) # container
    smoothSpeed  = smooth(speed,20)
    sTh = (smoothSpeed >= th_strong)                  # find where speed croses th
    
    for s in range(thWin+1,len(sTh)-hiWin):    #TODO: Find the proper way to speed this up 
        if not sTh[s]:
            continue
        if np.sum(smoothSpeed[s-thWin:s]>th_weak)>1:   #TODO: make numData points dynamic
            continue
        if np.sum(smoothSpeed[s:s+hiWin]>hi)<1:
            continue
        sOnset[s-shift] = True
        sTh[s:s+hiWin]=0
    return sOnset

def formatCaOnset(traces,th=3,th2=4.5,wlenMA=10,wlenDetrend=500,oldPeaks=None):
    # this function format the ca onset to one array..
    csOnset = np.zeros_like(traces,dtype=bool)
    csEvent = np.zeros_like(traces,dtype=bool)
    for l in range(0,traces.shape[1]):
        y,x = getCaOnset(traces[:,l],th,th2,wlenMA=10,wlenDetrend=500,oldPeaks=None)
        csOnset[:,l]=y
        csEvent[:,l]=x

    return csOnset, csEvent 

# from scipy cookbook... bandpass butter filters:
def butter_bandpass(lowcut, highcut, fs, order=5):
   nyq = 0.5 * fs
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band')
   return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = filtfilt(b, a, data)
  return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def HighSpeedPeriods(ax,speed,dt,th, dataPoints, Color):
    Sdata = {}
    #find movment onset:
    lspeed =  smooth(speed, dataPoints/2)
    hiSpeed = lspeed >= th
#    hiSpeed = hiSpeed.T
    d = np.diff(1* hiSpeed)
    segments = {'start':[],'end':[]}
    if hiSpeed[0] ==1:
        tStart = 0
    else:
        tStart = None
        
    for l in range(0,len(d)):
        if d[l] == 1:
            tStart = l
        if d[l] == -1 and l-tStart > dataPoints:
            segments['start'].append(tStart)
            segments['end'].append(l)
            tStart = None
            
    if tStart is not None and np.sum(hiSpeed[tStart:]) > dataPoints:
        segments['start'].append(tStart)
        segments['end'].append(len(hiSpeed)-1)
    Sdata['highSpeed'] = segments
    # plot speed vd. onset:
    t = np.linspace(0,len(speed)*dt,len(speed))
    ax.plot(t,speed,color='black')
#    ax.plot(t,lspeed)
    for l in range(0, len(segments['start'])):
        ax.axvspan(t[segments['start'][l]], t[segments['end'][l]], color= Color, alpha=0.5)
    return Sdata

def LowSpeedPeriods(ax,speed,dt,th, dataPoints, Color):
    Sdata = {}
    #find movment onset:
    lspeed =  smooth(speed, dataPoints/2)
    loSpeed = lspeed <= th
#    hiSpeed = hiSpeed.T
    d = np.diff(1* loSpeed)
    segments = {'start':[],'end':[]}
    if loSpeed[0] ==1:
        tStart = 0
    else:
        tStart = None
        
    for l in range(0,len(d)):
        if d[l] == 1:
            tStart = l
        if d[l] == -1 and l-tStart > dataPoints:
            segments['start'].append(tStart)
            segments['end'].append(l)
            tStart = None
            
    if tStart is not None and np.sum(loSpeed[tStart:]) > dataPoints:
        segments['start'].append(tStart)
        segments['end'].append(len(loSpeed)-1)
    Sdata['lowSpeed'] = segments
    # plot speed vd. onset:
    t = np.linspace(0,len(speed)*dt,len(speed))
    ax.plot(t,speed,color='black')
#    ax.plot(t,lspeed)
    for l in range(0, len(segments['start'])):
        ax.axvspan(t[segments['start'][l]], t[segments['end'][l]], color= Color, alpha=0.5)
    return Sdata


def calcLfpFeatuers(lfp,Fs,Wfreq = None):
    # This function calculate common features for rythem analysis
    #Input:
    #   lfp - the lfp data
    #   FS - sampling freqency of lfp
    #   Wfreq - (OPTIONAL) list of frequencies to retuen
    #Output:
    #   fe - a dict with all features..
    
    f, t, Sxx = signal.spectrogram(lfp,Fs,window=('hamming'),nperseg=140,noverlap =120,nfft=1200)
    Pxx = 10*np.log10(np.abs(Sxx)) 
    Pxx[np.isinf(Pxx)] = 0
    
    fe = {}
    if Wfreq is None:
        Wfreq = {'delta':{'low':1,'high':4},
                 'theta':{'low':5,'high':8},
                 'lowBeta':{'low':9,'high':14},
                 'Beta2':{'low':15,'high':30},
                 'Beta':{'low':13,'high':20},
                 'Beta2a':{'low':15,'high':23},
                 'Beta2b':{'low':23,'high':30},
                 'Gamma':{'low':40,'high':100}}
    
    for freq in Wfreq.keys():
        p = getPowerInBand(f,Sxx,Wfreq[freq]['low'], Wfreq[freq]['high'])
        a = interpolate.interp1d(t,p,fill_value='extrapolate')
        fe[freq] = a
    
    fe['sx'] = interpolate.interp1d(t,np.sum(Sxx,0),fill_value='extrapolate')
    fe['lfp'] =lfp
    fe['Sxx'] = Sxx
    fe['sxx_f'] = f
    fe['sxx_t']= t
    fe['Pxx'] = Pxx
    
    return fe           

def charSpikes(event_time,whole_trace,std_threshold,pre_window,Fs):
    x_axis = np.arange(0,np.max(whole_trace.shape))/Fs
    event_idx = np.full_like(event_time,np.nan)
    event_amp =  np.full((event_time.shape[0],1),np.nan)
    pre_event_threshold = np.full((event_time.shape[0],1),np.nan)
    for spike_time_idx,eTime in enumerate(event_time):
        start_idx = np.argmin(np.abs(x_axis-eTime[0]))
        end_idx = np.argmin(np.abs(x_axis-eTime[1]))
        
        current_trace = whole_trace[end_idx:]
        d_current_trace = current_trace[1:]-current_trace[:-1]
        extra_end_idx = [i for (i, val) in enumerate(d_current_trace) if val<=0]
        if len(extra_end_idx)>2:
            extra_end_idx = extra_end_idx[:2]
            
        try:
            end_idx = end_idx+extra_end_idx[-1]
        except:
            print('skipped iteration')
            continue
           
        current_trace = whole_trace[start_idx:end_idx+1]
        ref_idx = start_idx
        max_amp = np.nanmax(current_trace)
        max_idx = np.nanargmax(current_trace)
        end_idx = ref_idx+max_idx
        if max_idx>0:
            min_amp = np.nanmin(current_trace[:max_idx+1])
            min_idx = np.nanargmin(current_trace[:max_idx+1])
        else:
            min_amp = current_trace[0]
            min_idx = 0
        start_idx = ref_idx+min_idx
        
        pre_start_idx = np.max([0,int(start_idx-pre_window*Fs)])
        pre_event_threshold[spike_time_idx] = std_threshold*np.nanstd(whole_trace[pre_start_idx:start_idx+1])
#        print(np.nanstd(whole_trace[pre_start_idx:start_idx+1]),max_amp,min_amp,)
        event_amp[spike_time_idx] = max_amp-min_amp
        event_time[spike_time_idx,0] = x_axis[start_idx]
        event_time[spike_time_idx,1] = x_axis[end_idx]
        event_idx[spike_time_idx,0] = start_idx
        event_idx[spike_time_idx,1] = end_idx
        if  event_amp[spike_time_idx]>pre_event_threshold[spike_time_idx]:
            whole_trace[start_idx:int(end_idx+3)] = np.nan

    return event_amp, event_time ,event_idx,pre_event_threshold


def markEventFall(event_idx,whole_trace,max_fall):
    event_fall = np.full_like(event_idx,np.nan)
    for spike_time_idx,eIdx in enumerate(event_idx):
        end_idx = eIdx[1]
        event_fall[spike_time_idx,0] = end_idx+1 
        eWidth = end_idx - eIdx[0]
        if spike_time_idx+1 <event_idx.shape[0]:
            next_idx = event_idx[spike_time_idx+1,0]
        else: 
            next_idx = whole_trace.shape[0]-1
        next_idx = int(min(next_idx,end_idx+1+max_fall*eWidth))
        if next_idx>end_idx+2:
            event_fall[spike_time_idx,1] = np.argmin(np.abs(whole_trace[int(end_idx+2):next_idx]-np.min(whole_trace[int(event_idx[spike_time_idx,0]):int(event_idx[spike_time_idx,1]+1)])))+end_idx+1
        else:
            event_fall[spike_time_idx,1] = end_idx+1
        
    return event_fall

def caSpikeFinder(dff,Fs=20,tapers = [2,3],std_threshold = 7,window_size = 1,pre_window = 20,max_fall = 4):
    x_axis = np.arange(0,dff.shape[1])/Fs
    caOnset = np.full_like(dff,np.nan)
    caFall = np.full_like(dff,np.nan)

    for roi in range(0,dff.shape[0]):
#        print('ROI ',str(roi),'/',str(dff.shape[0]))
        whole_trace = np.copy(dff[roi,:])
        whole_trace = butter_bandpass_filter(whole_trace,0.02,9.9,20)
        S,t,f,_= mtspecgramc(whole_trace,[window_size, 1/Fs],tapers=tapers,Fs=Fs)
        normalized_S = S-np.mean(S,axis=0)
        normalized_S = np.divide(normalized_S,np.std(S,axis=0,ddof=1))

        #func = lambda x: (x >= 0) and (x<=fpass[-1])
        f_idx = [i for (i, val) in enumerate(f) if (val>=0 and val <=2)]

        power = np.mean(normalized_S[:,f_idx],axis=1)
        d_power= power[1:]-power[:-1]

        scaleMedian = 3*1.4826*np.median(np.abs(d_power-np.median(d_power))) 
        up_power_idx_list = [i for (i, val) in enumerate(d_power) if val>scaleMedian]
     
        if len(up_power_idx_list) == 0:
            continue

        up_power_idx_list = [up_power_idx_list[0]]+[val for (i, val) in enumerate(up_power_idx_list) if val-up_power_idx_list[i-1]>1]
        up_power_idx_list = [val for (i, val) in enumerate(up_power_idx_list) if d_power[val]>np.mean(d_power)]

        down_power_idx_list =[d_power.size for (i, val) in enumerate(up_power_idx_list)] 

        for idx,up_power_idx in enumerate(up_power_idx_list):
            current_d_power = d_power[up_power_idx:]
            try:
                down_power_idx_list[idx] = up_power_idx+ np.min([i for (i,val) in enumerate(current_d_power) if val<=0])
            except:
                down_power_idx_list[idx] = up_power_idx

        keepLokking = True
        passNum = 1
        results = {}
        std_threshold2 = std_threshold
        pre_window2 = pre_window

        event_time = np.asarray([x_axis[up_power_idx_list] , x_axis[down_power_idx_list]]).T

        while keepLokking:               
            event_amp, event_time ,event_idx,pre_event_threshold = charSpikes(event_time,whole_trace,std_threshold2,pre_window2,Fs)

            pre_event_threshold = np.delete(pre_event_threshold,np.nonzero(np.isnan(event_amp))[0])
            event_time = np.delete(event_time,np.nonzero(np.isnan(event_amp))[0], axis=0)
            event_idx = np.delete(event_idx,np.nonzero(np.isnan(event_amp))[0], axis=0)            
            event_amp = np.delete(event_amp,np.nonzero(np.isnan(event_amp))[0], axis=0)

            pre_event_threshold = np.delete(pre_event_threshold,np.where(event_idx[:,1]-event_idx[:,0]<3),axis=0)
            event_time = np.delete(event_time,np.where(event_idx[:,1]-event_idx[:,0]<3),axis=0)  
            event_amp = np.delete(event_amp,np.where(event_idx[:,1]-event_idx[:,0]<3),axis=0)
            event_idx = np.delete(event_idx,np.where(event_idx[:,1]-event_idx[:,0]<3),axis=0) 

            nextPass = event_time[event_amp[:,0]<pre_event_threshold,:]

            event_time = np.delete(event_time,np.where(event_amp[:,0]<pre_event_threshold),axis=0)
            event_idx = np.delete(event_idx,np.where(event_amp[:,0]<pre_event_threshold),axis=0)   
            event_amp = np.delete(event_amp,np.where(event_amp[:,0]<pre_event_threshold),axis=0)

            whole_trace2 = np.copy(dff[roi,:])
            event_fall =  markEventFall(event_idx,whole_trace2,max_fall)

            if len(event_amp) >0:
                results['pass_'+str(passNum)] = {'event_time':event_time,
                        'event_idx':event_idx,'event_amp':event_amp,
                        'pre_event_threshold':pre_event_threshold, 
                        'event_fall':event_fall}

                if passNum < 3:
                    std_threshold2 = std_threshold2*0.9
                    pre_window2 = pre_window2*1.75

                passNum = passNum+1
                event_time = nextPass

                for Sidx,val in enumerate(event_idx):
                    whole_trace[int(val[0]):int(event_fall[Sidx,1])] = np.nan
            else:
                keepLokking = False


        for passes in results.keys(): 
            for st, en in results[passes]['event_idx']:
                caOnset[roi,int(st):int(en)+1] = 1
            for st, en in results[passes]['event_fall']:
                caFall[roi,int(st):int(en)+1] = 1

    return caOnset,caFall

def windowed_view(arr, window, overlap):
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)

def getAlignedSpeed(cellType,cre = None, mice = None, period = None, day=None,savePath = '/home/dana_z/ssd_2TB/6OHDA/speed2ca/'):
    # function that take in the classification and return the appropreate data:
    #Inputs:
    #   cellType - return MSN or CRE if both pass ['MNS','CRE']
    #   mice - (Optional) list of mice from to include. Default: None - will load data for all mice
    #   period - (Optional) either 'Pre' or 'Post'. difault: None - return full length of data from picked sessions
    #   day - (Optional) lambda function with logic for picking days. Default: None - ignore day attr when picking data
    #           NOTE: day will be ignored if period is specified
    #   cre - (Optional) which cre mouse is it. options:None (default), "PV", "CHI"
    #                   must have trace included in dataType list to be taken into account
    #   WinPre - (Optional) length of pre window in secounds (default 2)
    #   WinPost - (Optional) length of post window in secounds (default 2)
    #Output:
    #   data - the requested data. format: {mice_session:{dataType:data}}
    
    
    dFile = 'FinalData_6OHDA_H.h5'
    # double check parameters inputs are valid:    
    df = pd.read_csv(savePath+'sessions')
    
    if period == None and day != None and isinstance(day,type(lambda c:None)):
        df['keep'] = df.apply(lambda row: day(row.day), axis=1)
        df = df[(df.keep==True)]
    
    if period in ['Healthy','Day 1-4','Day 5-12','Day 13-20','One Month']:
        df = df[(df.period==period)]
       
    if cre in ['PV','CHI','NA']:
        df = df[(df.cre==cre)]

    if not isinstance(cellType,list):
        cellType = [cellType]
        
    cellType = list(set(cellType).intersection(set(['MSN','CRE'])))
    if len(cellType) == 0:
        raise ValueError('Not a valid cellType value. cellType must be in ["MSN","CRE"]')
    
    # traverse the hdf5 file:
    if mice == None:
        mice = getMiceList(dFile) 
    elif not isinstance(mice,list):
        mice = [mice]
    
    if not isinstance(mice[0],str):
        for m in range(0,len(mice)):
            mice[m] = str(mice[m])
    df = df[(df.mouse.isin(mice))]
    # start extracting the data:   
    
    # alllocate memory:
    nNeurons = 0;
    if 'MSN' in cellType:
        nNeurons = nNeurons + int(df.numMsn.sum()) - int(df.numred.sum())
    if 'CRE' in cellType:
        nNeurons = nNeurons + int(df.numred.sum())
    
#    print(df, nNeurons)
    dResult = np.empty([80,nNeurons],dtype=float)
    
    ind = 0
    for sess in df.sess.unique():
        if 'MSN' in cellType:
            try:
                tempD = pickle.load(open(savePath+'MSN/'+sess,'rb'))
            except:
                print('ignored ',sess)
                continue
#            tempD = np.squeeze(tempD)
#            print(tempD.shape,ind,ind+tempD.shape[1])
            dResult[:,ind:ind+tempD.shape[1]] = tempD   
            ind = ind+tempD.shape[1]
        # for every Cre neuron:
        if 'CRE' in cellType:
            try:
                tempD = pickle.load(open(savePath+'CRE/'+sess,'rb'))
            except:
                continue
#            tempD = np.squeeze(tempD)
            dResult[:,ind:ind+tempD.shape[1]] = tempD   
            ind = ind+tempD.shape[1]
        
    return dResult[:,:ind],df

def getDrugFromSess(sess):

    if sess[-1] == 'A':
        return 'Amph'
    elif sess[-1] == 'L':
        return 'L-Dopa'
    elif sess[-1] == 'S':
        return 'Saline'
    else:
        return'None'

def getAlignedLFP(cellType,cre = None, mice = None, period = None, day=None, drug=None,drugPeriod='Pre'):
    # function that take in the classification and return the appropreate data:
    #Inputs:
    #   cellType - return MSN or CRE if both pass ['MNS','CRE']
    #   mice - (Optional) list of mice from to include. Default: None - will load data for all mice
    #   period - (Optional) either 'Pre' or 'Post'. difault: None - return full length of data from picked sessions
    #   day - (Optional) lambda function with logic for picking days. Default: None - ignore day attr when picking data
    #           NOTE: day will be ignored if period is specified
    #   cre - (Optional) which cre mouse is it. options:None (default), "PV", "CHI"
    #                   must have trace included in dataType list to be taken into account
    #   WinPre - (Optional) length of pre window in secounds (default 2)
    #   WinPost - (Optional) length of post window in secounds (default 2)
    #Output:
    #   data - the requested data. format: {mice_session:{dataType:data}}
    
    
    dFile = 'FinalData_6OHDA_H.h5'
    # double check parameters inputs are valid:
    if drugPeriod=='Post':
        savePath = '/home/dana_z/HD1/lfp2ca_notNormalize/'#'/home/dana_z/HD1/lfpAligned2Ca/Post/'
    else:
        savePath = '/home/dana_z/HD1/lfp2ca_notNormalize/'#'/home/dana_z/HD1/lfpAligned2Ca/Pre/'

    df = pd.read_csv(savePath+'sessions')
    
    if period == None and day != None and isinstance(day,type(lambda c:None)):
        df['keep'] = df.apply(lambda row: day(row.day), axis=1)
        df = df[(df.keep==True)]
    
    if period in ['Healthy','Day 1-4','Day 5-12','Day 13-20','One Month']:
        df = df[(df.period==period)]
       
    if cre in ['PV','CHI','NA']:
        df = df[(df.cre==cre)]
    
    if drug in ['Amph','L-Dopa','Saline','None']:
        df = df[(df.drug==drug)]
    

    if not isinstance(cellType,list):
        cellType = [cellType]
        
    cellType = list(set(cellType).intersection(set(['MSN','CRE'])))
    if len(cellType) == 0:
        raise ValueError('Not a valid cellType value. cellType must be in ["MSN","CRE"]')
    
    # traverse the hdf5 file:
    if mice == None:
        mice = getMiceList(dFile) 
    elif not isinstance(mice,list):
        mice = [mice]
    
    if not isinstance(mice[0],str):
        for m in range(0,len(mice)):
            mice[m] = str(mice[m])
    df = df[(df.mouse.isin(mice))]
    # start extracting the data:   
    
    # alllocate memory:
    nNeurons = 0;
    if 'MSN' in cellType:
        nNeurons = nNeurons + int(df.numMsn.sum()) - int(df.numred.sum())
    if 'CRE' in cellType:
        nNeurons = nNeurons + int(df.numred.sum())
    
    dResult = np.empty([12206,87,nNeurons],dtype=float)
    
    ind = 0
    for sess in df.sess.unique():
        if 'MSN' in cellType:
            tempD = pickle.load(open(savePath+'MSN/'+sess,'rb'))
            tempD[tempD==9999] = np.nan
            tempD[tempD==-9999] = np.nan
            dResult[:,:,ind:ind+tempD.shape[2]] = tempD   
            ind = ind+tempD.shape[2]
        # for every Cre neuron:
        if 'CRE' in cellType:
            try:
                tempD = pickle.load(open(savePath+'CRE/'+sess,'rb'))
            except:
                continue
            tempD[tempD==9999] = np.nan
            tempD[tempD==-9999] = np.nan
            dResult[:,:,ind:ind+tempD.shape[2]] = tempD   
            ind = ind+tempD.shape[2]
        
    return dResult[:,:,:ind],df


def getRotPeriods(ax,rot,dt,th,lth,dataPoints,Color = {'hiAC':'mediumseagreen','hiC':'limegreen','lo':'tomato'},plt=True):
    Sdata = {}
    #find movment onset:
    lrot =  smooth(rot, dataPoints/2)
    hiC_rot = lrot >= th
    hiAC_rot = lrot <= -th
    lo_rot = (lrot <= lth) & (lrot >= -lth)
#    hiRot = hiRot.T
    dhiC = np.diff(1* hiC_rot)
    dhiAC = np.diff(1* hiAC_rot)
    dlo = np.diff(1* lo_rot)
    segments = {'hiAC':{'start':[],'end':[]},'hiC':{'start':[],'end':[]},'lo':{'start':[],'end':[]}}
    for cond in segments.keys():
        hiRot = eval(cond+'_rot')
        d = eval('d'+cond)
        
        if hiRot[0] ==1:
            tStart = 0
        else:
            tStart = None

        for l in range(0,len(d)):
            if d[l] == 1:
                tStart = l
            if d[l] == -1 and l-tStart > dataPoints:
                segments[cond]['start'].append(tStart)
                segments[cond]['end'].append(l)
                tStart = None

        if tStart is not None and np.sum(hiRot[tStart:]) > dataPoints:
            segments[cond]['start'].append(tStart)
            segments[cond]['end'].append(len(hiRot)-1)
        Sdata[cond] = segments[cond]
        # plot rot vd. onset:
        t = np.linspace(0,len(rot)*dt,len(rot))
        ax.plot(t,rot,color='black')

    #   Fix that to draw all thress periods in diff colors
        if plt:
            for l in range(0, len(segments[cond]['start'])):
                ax.axvspan(t[segments[cond]['start'][l]], t[segments[cond]['end'][l]], color= Color[cond], alpha=0.5)
    return Sdata

def periodCalcOld(day):
    if day== 0:
        return 'Healthy'
    elif day<5:
        return 'Day 1-4'
    elif day<13:
        return 'Day 5-12'
    elif day<21:
        return 'Day 13-20'
    else:
        return 'One Month'

def periodCalc(day):
    if day== 0:
        return 'Healthy'
    elif day<13:
        return 'Day 1-13'
    else:
        return 'Day 14-35'

def calcLFPAlign2Mvmt(savePath,mvmt='speed',onsetName='mvmtOnset',norAxis=1):
    # function used to calculte LFP aligned to mvmt onset, or rotation onset 6/15/2020
    Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']
    miceList = getMiceList(Files[0]) 
    f = h5py.File('Spectograms.hdf5','r') #LFP coeffs
    
    # constents for analysis:
    WinPre = 2 #s
    WinPost = 2 #s
    df = pd.DataFrame(columns=['mouse','sess','day','period','cre'])
    
    for m in miceList:
        data =  getData(Files[0],[mvmt,'lfp'],period ='Pre', mice=m)
        cre = getCreType(Files[1],m)
        for sess in tqdm(data.keys()): 
            if sess[5] == 'B':
                day = 0
            else:
                day = int(re.findall(r'\d+',sess[5:])[0])
    
             # get data
            speedOnset = getOnsetOrPeriod(m,sess,'Pre',onsetName)
            if np.sum(speedOnset)==0:
                print('no onset in sess: ',sess)
                continue
    #        rotOnset = getOnsetOrPeriod(m,sess,'Pre','rotOnset')
            
            coeff = np.abs(f[m][sess]['Pre']['coeff'].value)
            lfpOutliers = removeLFPOutliers(data[sess]['lfp']['lfp'], sess)
            try:
                coeff[:,(lfpOutliers[:,0]==1)] = np.nan
                if norAxis == 1:
                    coeff = coeff.T/np.nansum(coeff,axis=norAxis) # So that axis[0] is the time axis + normalize power in frequency per sesion
                else: 
                    coeff = coeff/np.nansum(coeff,axis=norAxis)
                    coeff = coeff.T
            except:
                print(sess)
                continue
            
            # add session to df, so can be retrived
    
            dtS = float(1/data[sess][mvmt]['Fs'])
            dtL = float(1/data[sess]['lfp']['FS'])
            ts = np.arange(0, np.max(speedOnset.shape)) * dtS 
            tl = np.arange(0, np.max(data[sess]['lfp']['lfp'].shape)) * dtL
    
            tPlot = np.linspace(-WinPre,WinPost,int((WinPre+WinPost)/dtL))      
            
            # convert onset to LFP time: 
            onsets = np.full_like(tl,False)
            for si in ts[speedOnset]:
                ti = np.argmin(np.abs(tl-si))
                onsets[ti] = True
            onsets = onsets.astype(bool)
            # for every Cre neuron:
            al = alignToOnset(coeff,onsets, winPost =WinPost/dtL, winPre =WinPre/dtL)
    
            if al.ndim <3:
                try:
                    al = np.reshape(al,(al.shape[0],al.shape[1],1))
                except:
                    print('no onset, when there should be in sess= ',sess)
                    continue
           
            if 'aligned' in locals():
                aligned = np.concatenate((aligned,al), axis = 2)
            else:
                aligned = al
                
            aligned = np.nan_to_num(aligned,nan=9999.0)
            pickle.dump( aligned, open( savePath+sess, "wb" ), protocol=4 )
            df= df.append({'mouse':m,'sess':sess,'day':day,'period': periodCalc(day),'cre':cre},ignore_index=True)
            del aligned
    df.to_csv(savePath+'sessions')

def getAlignedLFP_mvmt(savePath,cre = None, mice = None, period = None, day=None, drug=None,drugPeriod='Pre'):
    # function that take in the classification and return the appropreate data:
    #Inputs:
    #   cellType - return MSN or CRE if both pass ['MNS','CRE']
    #   mice - (Optional) list of mice from to include. Default: None - will load data for all mice
    #   period - (Optional) either 'Pre' or 'Post'. difault: None - return full length of data from picked sessions
    #   day - (Optional) lambda function with logic for picking days. Default: None - ignore day attr when picking data
    #           NOTE: day will be ignored if period is specified
    #   cre - (Optional) which cre mouse is it. options:None (default), "PV", "CHI"
    #                   must have trace included in dataType list to be taken into account
    #   WinPre - (Optional) length of pre window in secounds (default 2)
    #   WinPost - (Optional) length of post window in secounds (default 2)
    #Output:
    #   data - the requested data. format: {mice_session:{dataType:data}}
    
    
    dFile = 'FinalData_6OHDA_H.h5'
    # double check parameters inputs are valid:

    df = pd.read_csv(savePath+'sessions')
    
    if period == None and day != None and isinstance(day,type(lambda c:None)):
        df['keep'] = df.apply(lambda row: day(row.day), axis=1)
        df = df[(df.keep==True)]
    
    if period in df.period.unique():
        df = df[(df.period==period)]
       
    if cre in ['PV','CHI','NA']:
        df = df[(df.cre==cre)]
    
    if drug in ['Amph','L-Dopa','Saline','None']:
        df = df[(df.drug==drug)]
     
   
    # traverse the hdf5 file:
    if mice == None:
        mice = getMiceList(dFile) 
    elif not isinstance(mice,list):
        mice = [mice]
    
    if not isinstance(mice[0],str):
        for m in range(0,len(mice)):
            mice[m] = str(mice[m])
    df = df[(df.mouse.isin(mice))]
    # start extracting the data:   
    
    # alllocate memory:
   
    dResult = np.empty([12206,87,len(df)*50],dtype=float)
    
    ind = 0
    for sess in df.sess.unique():
            tempD = pickle.load(open(savePath+sess,'rb'))
            tempD[tempD==9999.0] = np.nan
            tempD[tempD==-9999.0] = np.nan
#            print(ind, dResult[:,:,ind:ind+tempD.shape[2]].shape,tempD.shape,ind+tempD.shape[2],len(df))
            dResult[:,:,ind:ind+tempD.shape[2]] = tempD   
            ind = ind+tempD.shape[2]
        
    return dResult[:,:,:ind],df

def circShiftRoll(shifts,data):
    return  np.array([np.roll(row, x) for row,x in zip(data, shifts)])

def perNeuronPerEvent(aDff,dff,onset,niter,PostS=40,preS=40):
    if np.min(dff.shape) ==1:
        aDff = np.reshape(aDff,(aDff.shape[0],1,aDff.shape[1]))
    if np.sum(onset) ==1:
        aDff = np.reshape(aDff,(aDff.shape[0],aDff.shape[1],1))
        
    nT,nN,nE = aDff.shape
    # if no event in session -> move on...
    if nE == 0:
        return None,None,None
    if onset.ndim ==1:
        onset = np.reshape(onset,(1,onset.shape[0]))
    
        
    Th = np.empty((nN,niter*nE))
#    print(nE, Th.shape,dff.shape,np.sum(onset))
#    print(nE, nN,nT)
    # calculate thersholds:
    for itr in range(0,niter):
        shifts = np.random.randint(size=1,low=0,high=onset.shape[1])
        Ons = circShiftRoll(shifts,onset)
        sdff = alignToOnset(dff.T, Ons, winPost=PostS,winPre=preS)
#         if sdff.size ==0 or sdff.ndim <2:
#             shifts = np.random.randint(size=1,low=0,high=onset.shape[1])
#             Ons = circShiftRoll(shifts,onset)
#             sdff = alignToOnset(dff.T, Ons, winPost=PostS,winPre=preS)            
#         if nN ==1:
#             sdff = np.reshape(sdff,(sdff.shape[0],1,sdff.shape[1]))
#         if nE ==1: 
#             sdff = np.reshape(sdff,(sdff.shape[0],sdff.shape[1],1))

            
        # when 1 or movement events -> could end up with no event 2 secounds in -> empty sdff
        while isinstance(sdff,int) or sdff.ndim <3:
            shifts = np.random.randint(size=1,low=0,high=onset.shape[1])
            Ons = circShiftRoll(shifts,onset)
            sdff = alignToOnset(dff.T, Ons, winPost=PostS,winPre=preS)
            try:
                if sdff.size ==0:
                    sdff = 0    
                    continue
                if nN ==1:
                    sdff = np.reshape(sdff,(sdff.shape[0],1,sdff.shape[1]))
                if nE ==1: 
                    sdff = np.reshape(sdff,(sdff.shape[0],sdff.shape[1],1))
            except: 
                sdff = 1;



        # if sdff contain negative value, shift entire session up
        mins = np.min(sdff,axis=0)
        sdff[:,mins<0] =sdff[:,mins<0]+np.abs(mins[mins<0])
        # calculate the means
#        print(sdff.shape)
        muPre = np.mean(sdff[:int(nT/2),:,:],axis=0)
        muPost = np.mean(sdff[int(nT/2):,:,:],axis=0)
        ra = muPost/muPre
        while ra.shape[1]<nE: # incase lose event cause too close to begining of time axis
             ra = np.concatenate((ra,np.mean(ra,axis=1,keepdims=True)),axis=1)    
        if ra.shape[1]>nE:
            ra = ra[:,:nE]
        Th[:,itr*nE:(itr+1)*nE] = ra

    Th = np.percentile(Th,95,axis=1)

    Th = (matlib.repmat(Th,nE,1).T)

    mins = np.min(aDff,axis=0)
    aDff[:,mins<0] =aDff[:,mins<0]+np.abs(mins[mins<0])

    prePoints = aDff[:int(nT/2),:,:]
    postPoints = aDff[int(nT/2):,:,:]

    muPre = np.mean(prePoints,axis=0)
    muPost = np.mean(postPoints,axis=0)
    results = muPost/muPre    
#    print(nE,nN,Th.shape,results.shape)
    return (results>Th),results,Th[:,0]

def getAlignedLFP2(cellType,cre = None, mice = None, period = None, day=None, drug=None,drugPeriod='Pre'):
    # function that take in the classification and return the appropreate data:
    #Inputs:
    #   cellType - return MSN or CRE if both pass ['MNS','CRE']
    #   mice - (Optional) list of mice from to include. Default: None - will load data for all mice
    #   period - (Optional) either 'Pre' or 'Post'. difault: None - return full length of data from picked sessions
    #   day - (Optional) lambda function with logic for picking days. Default: None - ignore day attr when picking data
    #           NOTE: day will be ignored if period is specified
    #   cre - (Optional) which cre mouse is it. options:None (default), "PV", "CHI"
    #                   must have trace included in dataType list to be taken into account
    #   WinPre - (Optional) length of pre window in secounds (default 2)
    #   WinPost - (Optional) length of post window in secounds (default 2)
    #Output:
    #   data - the requested data. format: {mice_session:{dataType:data}}
    
    
    dFile = 'FinalData_6OHDA_H.h5'
    # double check parameters inputs are valid:
    if drugPeriod=='Post':
        savePath = '/home/dana_z/HD1/lfp2ca_notNormalize/'#'/home/dana_z/HD1/lfpAligned2Ca/Post/'
    else:
        savePath = '/home/dana_z/HD1/lfp2ca_notNormalize/'#'/home/dana_z/HD1/lfpAligned2Ca/Pre/'

    df = pd.read_csv(savePath+'sessions')
    
    if period == None and day != None and isinstance(day,type(lambda c:None)):
        df['keep'] = df.apply(lambda row: day(row.day), axis=1)
        df = df[(df.keep==True)]
    
    if period in ['Healthy','Day 1-4','Day 5-12','Day 13-20','One Month']:
        df = df[(df.period==period)]
       
    if cre in ['PV','CHI','NA']:
        df = df[(df.cre==cre)]
    
    if drug in ['Amph','L-Dopa','Saline','None']:
        df = df[(df.drug==drug)]
    

    if not isinstance(cellType,list):
        cellType = [cellType]
        
    cellType = list(set(cellType).intersection(set(['MSN','CRE'])))
    if len(cellType) == 0:
        raise ValueError('Not a valid cellType value. cellType must be in ["MSN","CRE"]')
    
    # traverse the hdf5 file:
    if mice == None:
        mice = getMiceList(dFile) 
    elif not isinstance(mice,list):
        mice = [mice]
    
    if not isinstance(mice[0],str):
        for m in range(0,len(mice)):
            mice[m] = str(mice[m])
    df = df[(df.mouse.isin(mice))]
    # start extracting the data:   
    
    # alllocate memory:
#    nNeurons = 0;
#    if 'MSN' in cellType:
#        nNeurons = nNeurons + int(df.numMsn.sum()) - int(df.numred.sum())
#    if 'CRE' in cellType:
#        nNeurons = nNeurons + int(df.numred.sum())
    nSess = len(df.sess.unique())
    dResult = np.empty([12206,87,nSess],dtype=float)
    
    ind = 0
    for sess in df.sess.unique():
        if 'MSN' in cellType:
            tempD = pickle.load(open(savePath+'MSN/'+sess,'rb'))
            tempD[tempD==9999] = np.nan
            tempD[tempD==-9999] = np.nan
            tempD = np.nanmean(tempD,axis=2)
            dResult[:,:,ind] = tempD   
            ind = ind+1
        # for every Cre neuron:
        if 'CRE' in cellType:
            try:
                tempD = pickle.load(open(savePath+'CRE/'+sess,'rb'))
            except:
                continue
            tempD[tempD==9999] = np.nan
            tempD[tempD==-9999] = np.nan
            tempD = np.nanmean(tempD,axis=2)
            dResult[:,:,ind] = tempD   
            ind = ind+1
        
    return dResult[:,:,:ind],df