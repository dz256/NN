# This file contains data loading functions taylord specifically for the 6OHDA experiment.
# created by Dana Zemel 
# last edited: 10/9/2018

import os
import numpy as np
import sys
import h5py
import warnings


def get_lfp(hf,m,s,period,red):
    # return the lfp from a given mouse m, and session s in file hf for a given period:
    # meant to be used by getData ONLY - use at your own risk
    
    #print(m+'/'+s+'/ePhys/lfp')
    if (m+'/'+s+'/ePhys/lfp' in hf):
        if (period == None):
            return {'lfp':hf[m][s]['ePhys']['lfp'][:],'FS':hf[m][s]['ePhys']['lfp'].attrs['FS']}
        elif ( m+'/'+s+'/traces/dff' not in hf):
            Fs = hf[m][s]['ePhys']['lfp'].attrs['FS']
            dt = 1/Fs
            lfp = hf[m][s]['ePhys']['lfp'][:]
            if lfp.shape[0] < lfp.shape[1]:
                lfp = lfp.T
            t = np.linspace(0,dt*lfp.shape[0],lfp.shape[0])
            if period == 'Post':
                tStart = 15*60
                tEnd = np.max(t)
                if tEnd <= tStart:
                    return {'lfp':'NO data for period','FS':None}
            if period=='Pre': 
                tStart = 5
                tEnd = 10*60
            return {'lfp':lfp[(t>=tStart)&(t<=tEnd),:],'FS':Fs}
                
        elif (period == 'Post' and hf[m][s]['traces']['dff'].attrs[period] == b'null'):
            return {'lfp':'NO data for period','FS':None}
        else:
            num_points = max(hf[m][s]['traces']['dff'].shape)
            dt = hf[m][s]['traces']['dff'].attrs['dt']
            t = np.linspace(0,dt*num_points,num_points)
            t = t[slice( *map(int, hf[m][s]['traces']['dff'].attrs[period][2:].decode("utf-8").split(':') ) )]
            tStart = np.min(t)
            tEnd = np.max(t)
            Fs = hf[m][s]['ePhys']['lfp'].attrs['FS']
            dt = 1/Fs
            lfp = hf[m][s]['ePhys']['lfp'][:]
            if lfp.shape[0] < lfp.shape[1]:
                lfp = lfp.T
            t = np.linspace(0,dt*lfp.shape[0],lfp.shape[0])
            lfp = lfp[(t>=tStart)&(t<=tEnd)]
            return {'lfp':np.reshape(lfp,(lfp.shape[0],1)),'FS':Fs}
            
    else:
        return {'lfp':'NO lfp data for session','FS':None}

def get_mvmt(hf,m,s,period,red):
    # return the mvmt from a given mouse m, and session s in file hf for a given period:
    # meant to be used by getData ONLY - use at your own risk
    
    if m+'/'+s+'/mvmt' in hf:
        if period == None:
            return {'speed':hf[m][s]['mvmt']['speed'][:],
                    'phi':hf[m][s]['mvmt']['phi'][:],
                    'rot':hf[m][s]['mvmt']['rotation'][:],
                    'Fs':1/hf[m][s]['mvmt'].attrs['dt']}
        else:
            #get the pre period time range:
            if not m+'/'+s+'/traces/dff' in hf:
                dt = hf[m][s]['mvmt'].attrs['dt']
                speed = hf[m][s]['mvmt']['speed'][:]
                phi = hf[m][s]['mvmt']['phi'][:]
                rot = hf[m][s]['mvmt']['rotation'][:]
                
                num_points = speed.size
                t = np.linspace(0,dt*num_points,num_points)
                if period == 'Post':
                    tStart = 15*60
                    tEnd = np.max(t)
                    if tEnd <= tStart:
                        return {'mvmt':'NO data for period','Fs':None}
                else: 
                    tStart = 5
                    tEnd = 10*60
                return {'speed':speed[:,(t>=tStart)&(t<=tEnd)],
                        'phi':phi[:,(t>=tStart)&(t<=tEnd)],
                        'rot':rot[:,(t>=tStart)&(t<=tEnd)],
                        'Fs':1/dt}            
            elif (period == 'Post' and hf[m][s]['traces']['dff'].attrs[period] == b'null'):
                return {'mvmt':'NO data for period','Fs':None}
            else:
                num_points = max(hf[m][s]['traces']['dff'].shape)
                dt = hf[m][s]['traces']['dff'].attrs['dt']
                t = np.linspace(0,dt*num_points,num_points)
                t = t[slice( *map(int, hf[m][s]['traces']['dff'].attrs[period][2:].decode("utf-8").split(':') ) )]
                tStart = np.min(t)
                tEnd = np.max(t)
                
                dt = hf[m][s]['mvmt'].attrs['dt']
                speed = hf[m][s]['mvmt']['speed'][:]
                speed = speed.T[(t>=tStart)&(t<=tEnd)]
                phi = hf[m][s]['mvmt']['phi'][:]
                phi = phi.T[(t>=tStart)&(t<=tEnd)]
                rot = hf[m][s]['mvmt']['rotation'][:]
                rot = rot.T[(t>=tStart)&(t<=tEnd)]
                t = np.linspace(0,dt*speed.size,speed.size)
                return {'speed':np.reshape(speed,(1,speed.shape[0])),
                        'phi':np.reshape(phi,(1,phi.shape[0])),
                        'rot':np.reshape(rot,(1,rot.shape[0])),
                        'Fs':1/dt}            
    else:
        return {'mvmt':'NO mvmt data for session','Fs':None}
    
def get_speed(hf,m,s,period,red):
    # return the speed from a given mouse m, and session s in file hf for a given period:
    # meant to be used by getData ONLY - use at your own risk
    
    if m+'/'+s+'/mvmt' in hf:
        if period == None:
            return {'speed':hf[m][s]['mvmt']['speed'][:],
                    'Fs':1/hf[m][s]['mvmt'].attrs['dt']}
        else:
            #get the pre period time range:
            if not m+'/'+s+'/traces/dff' in hf:
                dt = hf[m][s]['mvmt'].attrs['dt']
                speed = hf[m][s]['mvmt']['speed'][:]
                
                num_points = speed.size
                t = np.linspace(0,dt*num_points,num_points)
                if period == 'Post':
                    tStart = 15*60
                    tEnd = np.max(t)
                    if tEnd <= tStart:
                        return {'speed':'NO data for period','Fs':None}
                else: 
                    tStart = 5
                    tEnd = 10*60
                return {'speed':speed[:,(t>=tStart)&(t<=tEnd)],
                        'Fs':1/dt}            
            elif (period == 'Post' and hf[m][s]['traces']['dff'].attrs[period] == b'null'):
               return {'speed':'NO data for period','Fs':None}

            else:
                num_points = max(hf[m][s]['traces']['dff'].shape)
                dt = hf[m][s]['traces']['dff'].attrs['dt']
                t = np.linspace(0,dt*num_points,num_points)
                t = t[slice( *map(int, hf[m][s]['traces']['dff'].attrs[period][2:].decode("utf-8").split(':') ) )]
                tStart = np.min(t)
                tEnd = np.max(t)
                dt = hf[m][s]['mvmt'].attrs['dt']
                speed = hf[m][s]['mvmt']['speed'][:]
                t = np.linspace(0,dt*speed.size,speed.size)
                speed = speed.T[(t>=tStart)&(t<=tEnd)]
                return {'speed':np.reshape(speed,(1,speed.shape[0])),
                        'Fs':1/dt}            
    else:
        return {'speed':'NO speed data for session','Fs':None}

    
def get_phi(hf,m,s,period,red):
    # return the phi (direction) from a given mouse m, and session s in file hf for a given period:
    # meant to be used by getData ONLY - use at your own risk
    
    if m+'/'+s+'/mvmt' in hf:
        if period == None:
            return {'phi':hf[m][s]['mvmt']['phi'][:],
                    'Fs':1/hf[m][s]['mvmt'].attrs['dt']}
        else:
            #get the pre period time range:
            if not m+'/'+s+'/traces/dff' in hf:
                dt = hf[m][s]['mvmt'].attrs['dt']
                phi = hf[m][s]['mvmt']['phi'][:]
                
                num_points = phi.size
                t = np.linspace(0,dt*num_points,num_points)
                if period == 'Post':
                    tStart = 15*60
                    tEnd = np.max(t)
                    if tEnd <= tStart:
                        return {'phi':'NO data for period','Fs':None}
                else: 
                    tStart = 5
                    tEnd = 10*60
                return {'phi':phi[:,(t>=tStart)&(t<=tEnd)],
                        'Fs':1/dt}            
            elif (period == 'Post' and hf[m][s]['traces']['dff'].attrs[period] == b'null'):
                return {'phi':'NO data for period','Fs':None}
            else:
                num_points = max(hf[m][s]['traces']['dff'].shape)
                dt = hf[m][s]['traces']['dff'].attrs['dt']
                t = np.linspace(0,dt*num_points,num_points)
                t = t[slice( *map(int, hf[m][s]['traces']['dff'].attrs[period][2:].decode("utf-8").split(':') ) )]
                tStart = np.min(t)
                tEnd = np.max(t)
                
                dt = hf[m][s]['mvmt'].attrs['dt']
                phi = hf[m][s]['mvmt']['phi'][:]
                t = np.linspace(0,dt*phi.size,phi.size)
                phi = phi.T[(t>=tStart)&(t<=tEnd)]
                return {'phi':np.reshape(phi,(1,phi.shape[0])),
                        'Fs':1/dt}            
    else:
        return {'phi':'NO phi data for session','Fs':None}

    
def get_rot(hf,m,s,period,red):
    # return the rotation from a given mouse m, and session s in file hf for a given period:
    # meant to be used by getData ONLY - use at your own risk
    
    if m+'/'+s+'/mvmt' in hf:
        if period == None:
            return {'rot':hf[m][s]['mvmt']['rotation'][:],
                    'Fs':1/hf[m][s]['mvmt'].attrs['dt']}
        else:
            #get the pre period time range:
            if not m+'/'+s+'/traces/dff' in hf:
                dt = hf[m][s]['mvmt'].attrs['dt']
                rot = hf[m][s]['mvmt']['rotation'][:]
                
                num_points = rot.size
                t = np.linspace(0,dt*num_points,num_points)
                if period == 'Post':
                    tStart = 15*60
                    tEnd = np.max(t)
                    if tEnd <= tStart:
                       return {'rot':'NO data for period','Fs':None}
                else: 
                    tStart = 5
                    tEnd = 10*60
                return {'rot':rot[:,(t>=tStart)&(t<=tEnd)],
                        'Fs':1/dt}           
            elif (period == 'Post' and hf[m][s]['traces']['dff'].attrs[period] == b'null'):
                return {'rot':'NO data for period','Fs':None}

            else:
                num_points = max(hf[m][s]['traces']['dff'].shape)
                dt = hf[m][s]['traces']['dff'].attrs['dt']
                t = np.linspace(0,dt*num_points,num_points)
                t = t[slice( *map(int, hf[m][s]['traces']['dff'].attrs[period][2:].decode("utf-8").split(':') ) )]
                tStart = np.min(t)
                tEnd = np.max(t)
                
                dt = hf[m][s]['mvmt'].attrs['dt']
                rot = hf[m][s]['mvmt']['rotation'][:]
                t = np.linspace(0,dt*rot.size,rot.size)
                rot = rot.T[(t>=tStart)&(t<=tEnd)]
                return {'rot':np.reshape(rot,(1,rot.shape[0])),
                        'Fs':1/dt}            
    else:
        return {'rot':'NO rot data for session','Fs':None}

    
def get_trace(hf,m,s,period,red):
        # return the mvmt from a given mouse m, and session s in file hf for a given period:
    # meant to be used by getData ONLY - use at your own risk
    
    if m+'/'+s+'/traces/dff' in hf:
        if period == None:
            if red == None:
                return {'dff':hf[m][s]['traces']['dff'][:],
                        'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                        'numred':int(hf[m][s]['traces']['dff'].attrs['numRed'])}
            else:
                numred = int(hf[m][s]['traces']['dff'].attrs['numRed'])
                print(numred)
                if red:
                    return {'dff':hf[m][s]['traces']['dff'][0:numred,:][:],
                            'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                            'numred':numred}
                else:
                    return {'dff':hf[m][s]['traces']['dff'][numred:,:][:],
                            'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                            'numred':numred} 
                
        else:
            #get the pre period time range:
            if (period == 'Post' and hf[m][s]['traces']['dff'].attrs[period] == b'null'):
                return {'dff':'NO data for period','FS':None,'numred':0}
            else:
                numred = hf[m][s]['traces']['dff'].attrs['numRed']
                dff = hf[m][s]['traces']['dff'][:]
                #dff = dff.T
                if dff.shape[1] ==1:
                    dff = dff.T
                dff = dff[:,slice( *map(int, hf[m][s]['traces']['dff'].attrs[period][2:].decode("utf-8").split(':') ) )]
                
                if red == None:
                    return {'dff':dff,
                            'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                            'numred':numred}
                else:
                    numred = int(hf[m][s]['traces']['dff'].attrs['numRed'])
                    if red:
                        return {'dff':dff[0:numred,:],
                                'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                                'numred':numred}
                    else:
                        return {'dff':dff[numred:,:],
                                'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                                'numred':numred} 
    else:
        return {'dff':'NO trace data for session','FS':None,'numred':0}
    
def get_rawTrace(hf,m,s,period,red):
        # return the mvmt from a given mouse m, and session s in file hf for a given period:
    # meant to be used by getData ONLY - use at your own risk
    
    if m+'/'+s+'/traces/rawTraces' in hf:
        if period == None:
            if red == None:
                return {'f':hf[m][s]['traces']['dff'][:],
                        'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                        'numred':int(hf[m][s]['traces']['dff'].attrs['numRed'])}
            else:
                numred = int(hf[m][s]['traces']['dff'].attrs['numRed'])
                print(numred)
                if red:
                    return {'f':hf[m][s]['traces']['rawTraces'][0:numred,:][:],
                            'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                            'numred':numred}
                else:
                    return {'f':hf[m][s]['traces']['rawTraces'][numred:,:][:],
                            'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                            'numred':numred} 
                
        else:
            #get the pre period time range:
            if (period == 'Post' and hf[m][s]['traces']['dff'].attrs[period] == b'null'):
                return {'f':'NO data for period','FS':None,'numred':0}
            else:
                numred = hf[m][s]['traces']['dff'].attrs['numRed']
                f = hf[m][s]['traces']['rawTraces'][:]
                #dff = dff.T
                f = f[:,slice( *map(int, hf[m][s]['traces']['dff'].attrs[period][2:].decode("utf-8").split(':') ) )]
                
                if red == None:
                    return {'f':f,
                            'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                            'numred':numred}
                else:
                    numred = int(hf[m][s]['traces']['dff'].attrs['numRed'])
                    if red:
                        return {'f':f[0:numred,:],
                                'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                                'numred':numred}
                    else:
                        return {'f':f[numred:,:],
                                'FS':1/hf[m][s]['traces']['dff'].attrs['dt'],
                                'numred':numred} 
    else:
        return {'f':'NO trace data for session','FS':None,'numred':0}


def getData(fileName,dataType, mice = None,drug = None,period = None,day=None,cre = None, red=None):
    # function that take in the classification and return the appropreate data:
    #Inputs:
    #   fileName = string specify the exact location of the data hdf5 file
    #   dataType - a list that can include: lfp,mvmt,speed,phi,rot,and/or trace
    #   mice - (Optional) list of mice from to include. Default: None - will load data for all mice
    #   drug - (Optional) list of drugs to include. Default: None - ignore drug attr when picking data
    #   period - (Optional) either 'Pre' or 'Post'. difault: None - return full length of data from picked sessions
    #   day - (Optional) lambda function with logic for picking days. Default: None - ignore day attr when picking data
    #   cre - (Optional) which cre mouse is it. options:None (default), "PV", "CHI"
    #   red - (Optional) None - get all cells, True - get only red cells, False - get only MSN
    #                   must have trace included in dataType list to be taken into account
    #Output:
    #   data - the requested data. format: {mice_session:{dataType:data}}
    
    
    #make sure file exsits:
    if not os.path.isfile(fileName):
        raise ValueError('the file does not exsits')
    
    # intialize data
    data = {}
    
    # double check parameters inputs are valid:
    if drug not in [None,b'Amphetamin',b'L-Dopa',b'Saline',b'NA']:
        drug = None
        warnings.warn('Invalid input. valid options: b\'Amphetamin\',b\'L-Dopa\',b\'Saline\', or b\'NA\'')
    
    if day != None and not isinstance(day,type(lambda c:None)):
        day = None
        warnings.warn('Invalid input. day must be a lambda function')
    
    if period not in [None,'Pre','Post']:
        period = None
        warnings.warn('Invalid input. Period must be either "Pre" or "post".')
    
    if cre not in [None,'PV','CHI','NA']:
        cre = None
        warnings.warn('Invalid input. Cre must be either "PV" or "CHI".')
    
    if red not in [None,True,False]:
        red = None
        warnings.warn('Invalid input. red must be bool value.')
        
    if not isinstance(dataType,list):
        dataType = [dataType]
        
    dataType = list(set(dataType).intersection(set(['lfp','mvmt','speed','phi','rot','trace','rawTrace'])))
    if len(dataType) == 0:
        raise ValueError('Not a valid data type. dataType must be in [\'lfp\',\'mvmt\',\'speed\',\'phi\',\'rot\',\'trace\',\'rawTrace\']')

    
    # traverse the hdf5 file:
    with h5py.File(fileName,'r') as hf:
        # intialize the mice list:
        if mice == None:
            mice = list(hf.keys())
        elif not isinstance(mice,list):
            mice = [mice]
        
        if not isinstance(mice[0],str):
            for m in range(0,len(mice)):
                mice[m] = str(mice[m])
            
        # start extracting the data:
        for m in mice:
            if cre == None or hf[m].attrs['type'].decode("utf-8") == cre:
                sess = list(hf[m].keys())
                for s in sess:
                    if drug == None or drug == hf[m][s].attrs['drug']:
                        if day == None or day(hf[m][s].attrs['day']):
                            # this session match creteria - get data:
                            for t in dataType:
                                if m+'_'+s not in data.keys():
                                    data[m+'_'+s]= {t:globals()['get_'+t](hf,m,s,period,red)}
                                else:
                                    data[m+'_'+s][t] = globals()['get_'+t](hf,m,s,period,red)
    
    # assuming that user wants only sessions that has all requested data:
    if 'trace' in dataType and red == True:
        #clean sessions that has no red cells from output:
        print('cleaning up trace data')
        sL = list(data.keys())
        for s in sL:
            if data[s]['trace']['numred'] ==0:
                print('I deleted session: '+s)
                del data[s]
    
    if 'rawTrace' in dataType and red == True:
        #clean sessions that has no red cells from output:
        print('cleaning up raw trace data')
        sL = list(data.keys())
        for s in sL:
            if data[s]['rawTrace']['numred'] ==0:
                print('I deleted session: '+s)
                del data[s]
   
    if 'trace' in dataType:
        #clean sessions that has no red cells from output:
        print('cleaning up trace data')
        sL = list(data.keys())
        for s in sL:
            if data[s]['trace']['FS'] ==None:
                print('I deleted session: '+s)
                del data[s]
    
    if 'rawTrace' in dataType:
        #clean sessions that has no red cells from output:
        print('cleaning up raw trace data')
        sL = list(data.keys())
        for s in sL:
            if data[s]['rawTrace']['FS'] ==None:
                print('I deleted session: '+s)
                del data[s]

    
    if 'lfp' in dataType: 
        #clean sessions that has no lfp from output:
        print('cleaning up lfp data')
        sL = list(data.keys())
        for s in sL:
            if data[s]['lfp']['FS'] ==None:
                print('I deleted session: '+s)
                del data[s]

    if 'mvmt' in dataType:
        #clean sessions that has no mvmt from output:
        print('cleaning up mvmt data')
        sL = list(data.keys())
        for s in sL:
            if data[s]['mvmt']['Fs'] ==None:
                print('I deleted session: '+s)
                del data[s]
    if 'speed' in dataType:
        #clean sessions that has no mvmt from output:
        print('cleaning up speed data')
        sL = list(data.keys())
        for s in sL:
            if data[s]['speed']['Fs'] ==None:
                print('I deleted session: '+s)
                del data[s]
    if 'rot' in dataType:
        #clean sessions that has no mvmt from output:
        print('cleaning up rot data')
        sL = list(data.keys())
        for s in sL:
            if data[s]['rot']['Fs'] ==None:
                print('I deleted session: '+s)
                del data[s]
    if 'phi' in dataType:
        #clean sessions that has no mvmt from output:
        print('cleaning up phi data')
        sL = list(data.keys())
        for s in sL:
            if data[s]['phi']['Fs'] ==None:
                print('I deleted session: '+s)
                del data[s]
    
    return data
    
    
# functions to get quick info about a hdf5 struct: 
def getMiceList(fileName):
    # takes in a file name and return all the mice that exsits in file
    with h5py.File(fileName,'r') as hf:
        return list(hf.keys())

def getSessionList(fileName):
    sess = []
    mList = getMiceList(fileName)
    # takes in a file name and return all the mice that exsits in file
    with h5py.File(fileName,'r') as hf:
        for m in mList:
                sess +=[m+'_'+x for x in list(hf[m].keys())]
        return sess
    
def getSkipList(fileName,m,s):
    # takes in a file name and return all the mice that exsits in file
    with h5py.File(fileName,'r') as hf:
        return hf[m][s]['traces']['skipList'][:]
    
def getPdata(fileName,s):
    # takes in a file name and return all the mice that exsits in file
    with h5py.File(fileName,'r') as hf:
        return {'Hi_mvmt_start':hf[s]['hiLo_start'][:],
                'Hi_mvmt_stop':hf[s]['hiLo_end'][:],
                'EPS':hf[s]['EPS'][:],
                'caOnset':hf[s]['caOnset'][:]}
    
def getNumRed(fileName,m,s):
    with h5py.File(fileName,'r') as hf:
        if m+'/'+s+'/traces/dff' in hf:
            return int(hf[m][s]['traces']['dff'].attrs['numRed'])
        else:
            return -1

def getCreType(fileName,m):
    # takes in a file name and return all the mice that exsits in file
    with h5py.File(fileName,'r') as hf:
        return hf[m].attrs['type'].decode("utf-8")

def getOnsetOrPeriod(m,s,period,OPtype,fileName='OnsetsAndPeriods.hdf5'):
    # takes in a file name and return all the mice that exsits in file
    with h5py.File(fileName,'r') as hf:
        if  m+'/'+s+'/'+period+'/'+ OPtype not in hf:
            print(m+'/'+s+'/'+period+'/'+ OPtype +' NOT in FILE')
            return []        
        else: 
            return hf[m][s][period][OPtype][:]
                
def removeLFPOutliers(lfp, suffix):
    outlier = np.full_like(lfp,False)
    # special cases...
    if suffix ==  '1236_day30A':
        outlier[77683:923286] = True
    if suffix ==  '1236_day35L':
        outlier[69850:517514] = True
    
    # regular outliers:
    outlier = outlier + (np.abs(lfp)>0.01)
    if np.sum(outlier)>0:
        print('found ',np.sum(outlier),' outlier points')
    return outlier
                