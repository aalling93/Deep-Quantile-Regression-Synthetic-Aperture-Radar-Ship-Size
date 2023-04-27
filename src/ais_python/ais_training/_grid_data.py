#####################
#
# gridded data is used in, e.g., classification problems.
# When the objective is, for instance, to predict the next relative region of AIS
#
#####################


import numpy as np


def scale_grid_seq(samples,scaler):
    '''
    '''
    samples_scaled = []
    for i in range(len(samples)):
        samples_scaled.append(scale_data(samples[i],ais_scaler))
        
    samples_scaled = np.array(samples_scaled)
    return samples_scaled



def scale_data_grid(org_data,scaler):
    '''
    SCALING FOR CLASSI
        To prepare the data for the Deep learning methods, it need to be scaled to a standard scale.
        We are here using the standard scaler in the sklearn library, in which the data is scaled to a mean value of 0.
        
        
    '''
    #print(org_data.shape)
    data =copy.deepcopy(org_data)
    if (len(data[0].shape))==3:
        for i in range((data.shape[0])):
            for j in range(data[i].shape[-2]):               
                data[i][:,j,:]=scaler.transform(data[i][:,j,:])
    
    return data


def y_class(change_coord,resolution = 0.01):
    '''
    
    '''    
    y_class= -1
    
    #print(resolution*5)
    
    ########### GRID FOR LOWER CLASSES ##################
    
    if (-resolution <= change_coord[0] <= resolution)  and (-resolution <= change_coord[1] <= resolution*1):
        #print('class 0')
        y_class = 0
    if (resolution*1 <= change_coord[0] <= resolution*3)  and (resolution <= change_coord[1] <= resolution*3):
        #print('class 1')
        y_class =1
    if (-resolution*1 <= change_coord[0] <= resolution*1)  and (resolution*1 <= change_coord[1] <= resolution*3):
        #print('class 2')
        y_class = 2
    if (-resolution*3 <= change_coord[0] <= -resolution*1) and (resolution*1 <= change_coord[1] <= resolution*3):
        #print('class 3')
        y_class = 3
    if (-resolution*3 <= change_coord[0] <= -resolution*1)   and (-resolution*1 <= change_coord[1] <= resolution*1):
        #print('class 4')
        y_class = 4
    if (-resolution*3 <= change_coord[0] <= -resolution*1)   and (-resolution*3 <= change_coord[1] <= -resolution*1):
        #print('class 5')
        y_class = 5
    if (-resolution <= change_coord[0] <= resolution*1)  and (-resolution*3 <= change_coord[1] <= -resolution*1):
        ##print('class 6')
        y_class = 6
    if (resolution <= change_coord[0] <= resolution*3)  and (-resolution*3 <= change_coord[1] <= -resolution*1):
        #print('class 7')
        y_class = 7
    if (resolution <= change_coord[0] <= resolution*3)  and (-resolution <= change_coord[1] <= resolution*1):
        #print('class 8')
        y_class = 8
    
        
        
    if (resolution*3 <= change_coord[0] <= resolution*5)  and (resolution*3 <= change_coord[0] <= resolution*5):
        #print('class 9')
        y_class = 9
    if (resolution <= change_coord[0] <= resolution*3)  and (resolution*3 <= change_coord[0] <= resolution*5):
        #print('class 10')  
        y_class = 10
    if (-resolution <= change_coord[0] <= resolution) and (resolution*3 <= change_coord[0] <= resolution*5):
        #print('class 11')
        y_class = 11
    if (-resolution*3 <= change_coord[0] <= -resolution)   and (resolution*3 <= change_coord[0] <= resolution*5):
        #print('class 12')
        y_class = 12
    if (-resolution*5 <= change_coord[0] <= -resolution*3)  and (resolution*3 <= change_coord[0] <= resolution*5):
        #print('class 13')
        y_class = 13
    if (-resolution*5 <= change_coord[0] <= -resolution*3)  and (resolution <= change_coord[1] <= resolution*3):
        #print('class 14')
        y_class = 14
    if (-resolution*5 <= change_coord[0] <= -resolution*3)  and (-resolution <= change_coord[1] <= resolution*1):
        #print('class 15')
        y_class = 15
    if (-resolution*5 <= change_coord[0] <= -resolution*3)  and (-resolution*3 <= change_coord[1] <= -resolution*1):
        #print('class 16')
        y_class = 16
    if (-resolution*5 <= change_coord[0] <= -resolution*3)  and (-resolution*5 <= change_coord[1] <= -resolution*3):
        #print('class 17')
        y_class = 17
    if (-resolution*3 <= change_coord[0] <= -resolution)   and (-resolution*5 <= change_coord[1] <= -resolution*3):
        #print('class 18')
        y_class = 18
    if (-resolution <= change_coord[0] <= resolution)  and (-resolution*5 <= change_coord[1] <= -resolution*3):
        #print('class 19')
        y_class = 19
    if (resolution <= change_coord[0] <= resolution*3)  and (-resolution*5 <= change_coord[1] <= -resolution*3):
        #print('class 20')    
        y_class = 20
    if (resolution*3 <= change_coord[0] <= resolution*5)  and (-resolution*5 <= change_coord[1] <= -resolution*3):
        #print('class 21')
        y_class = 21
    if (resolution*3 <= change_coord[0] <= resolution*5)   and (-resolution*3 <= change_coord[1] <= -resolution*1):
        #print('class 22')
        y_class = 22
    if (resolution*3 <= change_coord[0] <= resolution*5)   and (-resolution <= change_coord[1] <= resolution*1):
        #print('class 23')
        y_class = 23
    if (resolution*3 <= change_coord[0] <= resolution*5)   and (resolution <= change_coord[1] <= resolution*3):
        #print('class 24')
        y_class = 24
        
        
    ########### GRID FOR MIDDLE CLASSES ##################
    if (resolution*5 <= change_coord[0] <= resolution*10)   and (resolution*5 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 25
    if (resolution <= change_coord[0] <= resolution*5)   and (resolution*5 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 26
    if (-resolution*5 <= change_coord[0] <= 0)   and (resolution*5 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 27
    if (-resolution*10 <= change_coord[0] <= -resolution*5)   and (resolution*5 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 28
    if (-resolution*10 <= change_coord[0] <= -resolution*5)  and (0 <= change_coord[1] <= resolution*5):
        #print('class 24')
        y_class = 29
    if (-resolution*10 <= change_coord[0] <= -resolution*5)   and (-resolution*5 <= change_coord[1] <= 0):
        #print('class 24')
        y_class = 30
    if (-resolution*10 <= change_coord[0] <= -resolution*5)   and (-resolution*10 <= change_coord[1] <= -resolution*5):
        #print('class 24')
        y_class = 31
    if (-resolution*5 <= change_coord[0] <= 0)   and (-resolution*10 <= change_coord[1] <= -resolution*5):
        #print('class 24')
        y_class = 32
    if (0 <= change_coord[0] <= resolution*5)   and (-resolution*10 <= change_coord[1] <= -resolution*5):
        #print('class 24')
        y_class = 33
    if (resolution*5 <= change_coord[0] <= resolution*10)   and (-resolution*10 <= change_coord[1] <= -resolution*5):
        #print('class 24')
        y_class = 34
    if (resolution*5 <= change_coord[0] <= resolution*10)   and (-resolution*5 <= change_coord[1] <= 0):
        #print('class 24')
        y_class = 35
    if (resolution*5 <= change_coord[0] <= resolution*10)   and (0 <= change_coord[1] <= resolution*5):
        #print('class 24')
        y_class = 36
        
        
    ########### GRID FOR advanced CLASSES ##################
    if (resolution*10 <= change_coord[0] <= resolution*20)   and (resolution*10 <= change_coord[1] <= resolution*20):
        #print('class 24')
        y_class = 37
    if (0 <= change_coord[0] <= resolution*10)   and (resolution*10 <= change_coord[1] <= resolution*20):
        #print('class 24')
        y_class = 38
    if (-resolution*10 <= change_coord[0] <= 0)   and (resolution*10 <= change_coord[1] <= resolution*20):
        #print('class 24')
        y_class = 39
    if (-resolution*20 <= change_coord[0] <= -resolution*10)   and (resolution*10 <= change_coord[1] <= resolution*20):
        #print('class 24')
        y_class = 40
    if (-resolution*20 <= change_coord[0] <= -resolution*10)   and (0 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 41
    if (-resolution*20 <= change_coord[0] <= -resolution*10)  and (-resolution*10 <= change_coord[1] <= 0):
        #print('class 24')
        y_class = 42
    if (-resolution*20 <= change_coord[0] <= -resolution*10)   and (-resolution*20 <= change_coord[1] <= -resolution*10):
        #print('class 24')
        y_class = 43
    if (-resolution*10 <= change_coord[0] <= 0)   and (-resolution*20 <= change_coord[1] <= -resolution*10):
        #print('class 24')
        y_class = 44
    if (0 <= change_coord[0] <= resolution*10)   and (-resolution*20 <= change_coord[1] <= -resolution*10):
        #print('class 24')
        y_class = 45
    if (resolution*10 <= change_coord[0] <= resolution*20)   and (-resolution*20 <= change_coord[1] <= -resolution*10):
        #print('class 24')
        y_class = 46
    if (resolution*10 <= change_coord[0] <= resolution*20)   and (-resolution*10 <= change_coord[1] <= 0):
        #print('class 24')
        y_class = 47
    if (resolution*10 <= change_coord[0] <= resolution*20)   and (0 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 48

    ########### GRID FOR greater CLASSES ##################
        
    y_class = y_class+1   
    return y_class