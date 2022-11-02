#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:20:08 2022

@author: marcorax93

This is a synthetic benchmark for the feedback.

The stimuli is composed by 5x15 neurons representing 3 different 5x5 tiles 
with 3 possible sub-patterns or characters, "v", "x" and "/" 
The 3 characters are combined together to compose two distinct input patterns
or words:
    
Pattern 1: "vxv" and Pattern 2 "v/v"

If the feedback works correctly, the network will learn only the sub-patterns "x" 
and "/" as "v" doesn't help to classify the two words
"""


import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import expit

from Libs.HOTSLib import fb_surfaces, all_surfaces


simulation_time = 1e4#us

#High frequency state for neuron and low frequency state (for background noise)
high_freq = 1e-2
low_freq = 1e-5

def PoissonSpikeGen(neuron_freq, activity_duration):
    """ Small function generating a Poisson spike train. It calculates 
    Function parameters:
                neuron_freq: (float) The neuron expected frequency (how many
                                      spikes per time step, if neuron_freq=1
                                      and actvivity_duration=5 expect an average
                                      of 5 spikes)
                activity_duration: (int) maximum timestamp in the resulting 
                                   timestamps array, unitless
    Result:    
                spike_train: (1D array) containing timestamps of generated events
                 
                                
    """
    spike_train = []
    #First spike
    new_spike = round(random.expovariate(neuron_freq))
    while new_spike<activity_duration:
        spike_train.append(new_spike)
        new_spike = round(spike_train[-1]+random.expovariate(neuron_freq))
        
    return np.array(spike_train)



#%% Create the stimulation patterns

n_files = 1000

def word_generator(word, sim_time, high_f, low_f=0):
    """
    Function used to generate words of the synthetic benchmark.
    It sets a small lattice of 5x5 tiles with poisson neurons firing at 
    high_f frequency composing a pattern defined by a character.
    The background can have silent neurons or firing with a baseline frequency 
    (low_f)
    Arguments:
        word (string): word composed of the available characters: v,x,/
        sim_time (int): number of microseconds used for the simulation
        high_f (float): firing frequency 1/us of the active neurons in the 
                        lattice (that compose a character)
        low_f (float): firing frequency 1/us of the in-active neurons in the 
                        lattice (background neurons/pixels)
                        
    Returns:
        events (list): list containing 4 arrays for each attribute of the 
                       generated events [x,y,p,t] x,y arrays for the position
                       of the event in the tile, p for the index of the tile
                       and t for the timestamp of the event.
                        
    """
        
    ####SUB PATTERNS:
        
    #1 "v":
        
    #  *...*
    #  *...* 
    #  .*.*.
    #  .*.*.
    #  ..*..
    
    sub_pattern_map_1 = np.array([[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,1,0,1,0],
                                  [0,0,1,0,0]],dtype=bool)
    
    #2 "/":
        
    #  ....*
    #  ...*. 
    #  ..*..
    #  .*...
    #  *....
    
    sub_pattern_map_2 = np.array([[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],
                                  [1,0,0,0,0]],dtype=bool)
    
    #3 "x":
        
    #  *...*
    #  .*.*. 
    #  ..*..
    #  .*.*.
    #  *...*
    
    sub_pattern_map_3 = np.array([[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],
                                  [1,0,0,0,1]],dtype=bool)
    
    #3 "t":
        
    #  *****
    #  ..*.. 
    #  ..*..
    #  ..*..
    #  ..*..
    
    sub_pattern_map_4 = np.array([[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],
                                  [0,0,1,0,0]],dtype=bool)
    
    #3 "y":
        
    #  *...*
    #  .*.*. 
    #  ..*..
    #  ..*..
    #  ..*..
    
    sub_pattern_map_5 = np.array([[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],
                                  [0,0,1,0,0]],dtype=bool)
    
    dictionary = {"v": sub_pattern_map_1, '/': sub_pattern_map_2,
                  "x": sub_pattern_map_3, "t": sub_pattern_map_4, 
                  "y": sub_pattern_map_5}
    
    events = [np.array(0,dtype=int) for i in range(4)]
    
    for char_i,char in enumerate(word):
        char_map = dictionary[char]
        for y in range(5):
            for x in range(5):
                if char_map[y,x]:
                    pixel_ts = PoissonSpikeGen(high_f, sim_time)
                elif low_f!=0:
                    pixel_ts = PoissonSpikeGen(low_f, sim_time)
                else:
                    pixel_ts=[]
                n_events = len(pixel_ts)
                pixel_ts = np.asarray(pixel_ts, dtype=int)
                pixel_x = x*np.ones(n_events,dtype=int)
                pixel_y = y*np.ones(n_events,dtype=int)
                pixel_p = char_i*np.ones(n_events,dtype=int)
                events[0] = np.append(events[0], pixel_x)
                events[1] = np.append(events[1], pixel_y)
                events[2] = np.append(events[2], pixel_p)
                events[3] = np.append(events[3], pixel_ts)
    
    #sorting events by timestamp
    sorted_args = np.argsort(events[3])
    events = [events[i][sorted_args] for i in range(4)]
    
    return events

data_events = []
data_labels = []
for i_file in range(n_files//2):
    events_p1 = word_generator(word = "v/v", sim_time=simulation_time,
                               high_f = high_freq, low_f = low_freq)
    data_events.append(events_p1)
    data_labels.append(0)
    # labels_p1 = 0*np.ones(len(events_p1),dtype=int)
    
    events_p2 = word_generator(word = "vxv", sim_time=simulation_time,
                               high_f = high_freq, low_f = low_freq)
    data_events.append(events_p2)
    data_labels.append(1)
    # labels_p2 = 1*np.ones(len(events_p2),dtype=int)
    
    # events_p3 = word_generator(word = "vtv", sim_time=simulation_time,
    #                             high_f = high_freq, low_f = low_freq)
    # data_events.append(events_p3)
    # data_labels.append(2)
    # labels_p3 = 2*np.ones(len(events_p3),dtype=int)


#%% Generate the timesurfaces
res_x = 5
res_y = 5
tau = 5000
n_pol = 3

data_surf=[]

characters_ts = [[], [], []]
for i_file in range(n_files):
    
    events = data_events[i_file]
    surfs = all_surfaces(events, res_x, res_y, tau, n_pol)
        
    
    #keep only the surface of the character for which the event as being generated
    surfs_cut = surfs[np.arange(len(surfs)),events[2],:,:]
    data_surf.append(surfs_cut)
    for i,event in enumerate(events[2]):
        if event == 0 or event == 2:
            characters_ts[0].append(surfs_cut[i])
        if event == 1:
            characters_ts[data_labels[i_file]+1].append(surfs_cut[i])
            
        

#pattern plot to check all is correct
surf_i = 100
concat_surf = np.reshape(surfs, [len(surfs),15,5])
plt.figure()
plt.imshow(concat_surf[surf_i])
concat_all_surfs = np.concatenate(data_surf)

#%% Calculations on average charcters ts (to get ideal centroids and thresholds)
# and also a low dimensional plot
average_character_ts = [np.mean(characters_ts[ch],0) for ch in range(len(characters_ts))]
std_character_max_radio= [np.max(np.sqrt(np.sum((average_character_ts[ch]-characters_ts[ch])**2,axis=(1,2)))) for ch in range(len(characters_ts))]

#Triangle edges
AB = np.sqrt(np.sum((average_character_ts[0]-average_character_ts[1])**2))
AC = np.sqrt(np.sum((average_character_ts[0]-average_character_ts[2])**2))
BC = np.sqrt(np.sum((average_character_ts[1]-average_character_ts[2])**2))

A = (0,0)
B = (AB,0)

#I can find C as one of the intersections of the two circles
#from https://math.stackexchange.com/questions/256100/how-can-i-find-the-points-at-which-two-circles-intersect
Cx = ((AC**2)-(BC**2)+(AB**2))/(2*AB)
Cy = np.sqrt(AC**2-Cx**2)
C=(Cx,Cy)


# plt.plot(A[0],A[1],'.')
radioV = plt.Circle(A, std_character_max_radio[0], edgecolor='r', fill=False,
                    linewidth = 3, alpha=1)
centerV = plt.Circle(A, 0.1, color='r', alpha=1)
radioSlash = plt.Circle(B, std_character_max_radio[1], edgecolor='#00ffff', 
                        fill=False, linewidth = 3, alpha=1)
centerSlash = plt.Circle(B, 0.1, color='#00ffff', alpha=1)
radioX = plt.Circle(C, std_character_max_radio[2], edgecolor ='#000033',
                    fill=False, linewidth = 3, alpha=1)
centerX = plt.Circle(C, 0.1, color='#000033', alpha=1)



fig_ths, ax_ths = plt.subplots()
ax_ths.add_patch(radioV)
ax_ths.add_patch(radioSlash)
ax_ths.add_artist(radioX)


ax_ths.add_patch(centerV)
ax_ths.add_patch(centerSlash)
ax_ths.add_artist(centerX)



#Use adjustable='box-forced' to make the plot area square-shaped as well.
ax_ths.set_aspect('equal', adjustable='datalim')
ax_ths.plot()   #Causes an autoscale update.
fig_ths.show()

#%% Kmeans clustering
n_k_clusters=2
surf_x = 5
surf_y = 5
kmeans = KMeans(n_clusters=n_k_clusters)
kmeans.fit(np.reshape(concat_all_surfs, [len(concat_all_surfs),5*5]))
k_centroids= kmeans.cluster_centers_

#plot centroids
fig, axs = plt.subplots(n_k_clusters)
kmeans_weights_0 = np.random.rand(surf_x, surf_y, n_k_clusters)

for pol_i in range(n_k_clusters):
    axs[pol_i].imshow(np.reshape(k_centroids[pol_i], [5,5]))
    kmeans_weights_0[:,:,pol_i]=np.reshape(k_centroids[pol_i], [5,5])

#%% Learning last layer response Kmeans

surf_x = 5
surf_y = 5
n_clusters = 2

n_words = 2

kmeans_weights_1 = np.random.rand(n_clusters, n_pol, n_words) #classifier

lrate = 0.003

time_context_1 = np.zeros([n_clusters, n_pol],dtype=int)

tau_1 = 50
        
pause_pressed=False    
for epoch in range(3):   
    for word_i in range(len(data_surf)):
        n_events = len(data_surf[word_i])
        word_surf = data_surf[word_i]
        progress=0
        rel_accuracy = 0

        #event mask used to avoid exponentialdecay calculation forpixel 
        # that didnot generate an event yet
        mask_start_1 = np.zeros([n_clusters, n_pol],dtype=int)
        mask_start_fb = np.zeros([n_words],dtype=int)
    
        for ts_i in range(n_events):
            
            label = data_labels[word_i]
            
            rec_distances_0=np.sum((word_surf[ts_i,:,:,None]-kmeans_weights_0[:,:,:])**2,axis=(0,1))
            
            rec_closest_0=np.argmin(rec_distances_0,axis=0)
            
                
            ref_pol = data_events[word_i][2][ts_i]
            ref_ts =  data_events[word_i][3][ts_i]
            time_context_1[rec_closest_0,ref_pol] = ref_ts
            mask_start_1[rec_closest_0,ref_pol]=1
            
            ts_lay_1 = np.exp((time_context_1-ref_ts)*mask_start_1/tau_1)*mask_start_1                
                        
            
            rec_distances_1=np.sum((ts_lay_1[:,:,None]-kmeans_weights_1[:,:,:])**2,axis=(0,1))
            rec_closest_1=np.argmin(rec_distances_1,axis=0)
                            
            progress+=1/n_events
            if rec_closest_1==label:
                result = "Correct"
                rel_accuracy += 1/n_events

            else:
                result = "Wrong"
                
            print("Epoch "+str(word_i)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy))
            print("Prediction: "+result+str(label))

            #supervised
            elem_distances_1 = (ts_lay_1[:,:]-kmeans_weights_1[:,:,label])
            kmeans_weights_1[:,:,label]+=lrate*elem_distances_1[:]
            

                        
                    
                        



#%% clustering after LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


event_labels = np.zeros(len(concat_all_surfs),dtype=int)
event_i=0
for i_file in range(n_files):
    n_events = len(data_events[i_file][0])
    label=data_labels[i_file]
    event_labels[event_i:event_i+n_events] = label*np.ones([n_events],dtype=int)
    event_i+=n_events
    
clf = LinearDiscriminantAnalysis()
clf.fit(np.reshape(concat_all_surfs, [len(concat_all_surfs),5*5]), event_labels)
clf.score(np.reshape(concat_all_surfs, [len(concat_all_surfs),5*5]), event_labels)

n_k_clusters=2
kmeans = KMeans(n_clusters=n_k_clusters)
kmeans.fit(clf.transform(np.reshape(concat_all_surfs, [len(concat_all_surfs),5*5])))
k_centroids= kmeans.cluster_centers_

#obtain the LDA features

tr=clf.coef_

patterns = np.zeros([n_k_clusters,5*5]) 

for i_centroid in range(n_k_clusters):
    patterns[i_centroid] = tr*k_centroids[i_centroid]

#plot centroids
fig, axs = plt.subplots(n_k_clusters)
for pol_i in range(n_k_clusters):
    axs[pol_i].imshow(np.reshape(patterns[pol_i], [5,5]))    


#%% New Learning rule find the thresholds

from pynput import keyboard

surf_x = 5
surf_y = 5
n_clusters = 2

n_words = 2

weights_0 = np.random.rand(surf_x, surf_y, n_clusters)
weights_0[:,:,0] = average_character_ts[1]
weights_0[:,:,1] = average_character_ts[2]


weights_1 = 0.5*np.ones([n_clusters, n_pol, n_words]) #classifier
weights_1[0,1,0]=1
weights_1[1,1,1]=1

# th_0 = np.zeros(n_clusters)+6
th_0 = np.zeros(n_clusters)+5
# th_0 = np.zeros(n_clusters)
th_0[0]=0.1


circle_th0 = plt.Circle(B, th_0[0], color='#00ffff', alpha=0.2)
circle_th1 = plt.Circle(C, th_0[1], color='#000033', alpha=0.2)

ax_ths.add_patch(circle_th0)
ax_ths.add_artist(circle_th1)

fig_ths.show()

y_som_old=0
dt_y_som=0

# lrate_non_boost = 0.004 
lrate_non_boost = 0.004 
# lrate_boost = 1

# lrate_boost = 0.003
lrate_boost = 0.01


lrate=lrate_boost

n_all_events = len(concat_all_surfs)


# Leaky Integral of S overtime at layer 1
ISs_1 = 0
Act_0 = np.zeros([n_words])
ISs_tau_1 = 100 #integration time constant 
ISs_1_history = np.zeros([n_all_events])
ISsderiv_1_history = np.zeros([n_all_events])
ISsdoublederiv_1_history = np.zeros([n_all_events])

Act_0_history = np.zeros([n_words,n_all_events])



#initialize weights 0 to surfaces:

# for cluster_i in range(n_clusters[0]):
#     label=np.random.randint(0,9)
#     recording=np.random.randint(0,len(train_surfs_0[label]))
#     surface_i=np.random.randint(0,len(train_surfs_0[label][recording]))
#     weights_0[:,:,cluster_i]=train_surfs_0[label][recording][surface_i]
s_event = 0
r_event = 0 
def on_press(key):
    global pause_pressed
    global lrate
    global lrate_non_boost
    global th_0
    global i_event
    global s_event
    global r_event
    print('{0} pressed'.format(
        key))
    if key.char == ('p'):
        pause_pressed=True
    if key.char == ('s'):
        # lrate=lrate_non_boost
        th_0 = np.zeros(n_clusters)+10
        s_event = i_event
    if key.char == ('r'):
        # lrate=lrate_non_boost
        th_0[0] = 2.3817399
        th_0[1] = 3.063397
        r_event = i_event



time_context_1 = np.zeros([n_clusters, n_pol],dtype=int)
time_context_fb = np.zeros([n_words],dtype=int)

tau_1 = 5
        
pause_pressed=False    
with keyboard.Listener(on_press=on_press) as listener:
    for epoch in range(3):   
        i_event=0
        for word_i in range(len(data_surf)):
            n_events = len(data_surf[word_i])
            word_surf = data_surf[word_i]
            progress=0
            rel_accuracy = 0
    
            #event mask used to avoid exponentialdecay calculation forpixel 
            # that didnot generate an event yet
            mask_start_1 = np.zeros([n_clusters, n_pol],dtype=int)
            mask_start_fb = np.zeros([n_words],dtype=int)
            y_som_old=0
            y_som=0
            y_som_dt=0
            Act_0[:]=0
            ISs_1=0
            iss_double_deriv_old=0

            for ts_i in range(n_events):
                
                
                label = data_labels[word_i]
                
                rec_distances_0=np.sum((word_surf[ts_i,:,:,None]-weights_0[:,:,:])**2,axis=(0,1))
                
                # rec_closest_0=np.argmin(rec_distances_0,axis=0)
                #new_fb
                rec_closest_0=np.argmin(rec_distances_0-th_0,axis=0)
                rec_closest_0_one_hot = np.zeros([n_clusters])
                rec_closest_0_one_hot[rec_closest_0]=1
                if (rec_distances_0[rec_closest_0]-th_0[rec_closest_0])<0:
    
    
                    
                    ref_pol = data_events[word_i][2][ts_i]
                    ref_ts =  data_events[word_i][3][ts_i]
                    time_context_1[rec_closest_0,ref_pol] = ref_ts
                    mask_start_1[rec_closest_0,ref_pol]=1
                    
                    ts_lay_1 = np.exp((time_context_1-ref_ts)*mask_start_1/tau_1)*mask_start_1                
                                
                    
                    rec_distances_1=np.sum((ts_lay_1[:,:,None]-weights_1[:,:,:])**2,axis=(0,1))
                    rec_closest_1=np.argmin(rec_distances_1,axis=0)
                    
    
                    time_context_fb[rec_closest_1] = ref_ts
                    mask_start_fb[rec_closest_1]=1
                    
                    train_surfs_1_recording_fb = np.exp((time_context_fb-ref_ts)*mask_start_fb/tau_1)*mask_start_fb                
                                
                
    
        
                   
                    norm = n_words-1
                    #supervised
                    y_som=(train_surfs_1_recording_fb[label]-np.sum((train_surfs_1_recording_fb[np.arange(n_words)!=label]/norm),axis=0)) #normalized by activation
                    
                    #unsupervised
                    # y_som=(train_surfs_1_recording_fb[label]-np.sum((train_surfs_1_recording_fb[np.arange(n_words)!=rec_closest_1]/norm),axis=0)) #normalized by activation
    
                    
                    dt_y_som = y_som - y_som_old
                    y_som_old = y_som
                    
                    # y_som_dt[1:] = (y_som[1:]-y_som[:-1])/((timestamps[1:]+1-timestamps[:-1])*0.001)
                    y_corr=y_som*(y_som>0)*(train_surfs_1_recording_fb[label]==1)
                    # np.random.shuffle(y_corr)# Test feedback modulation hypothesis with null class
                    
                    rec_closest_1_one_hot = np.zeros([n_words])
                    rec_closest_1_one_hot[rec_closest_1]=1
                    class_rate=np.sum(rec_closest_1_one_hot,axis=0)
                        
                    progress+=1/n_events
                    if rec_closest_1==label:
                        result = "Correct"
                        rel_accuracy += 1/n_events
    
                    else:
                        result = "Wrong"
                        
                    print("Epoch "+str(word_i)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy))
                    print("Prediction: "+result+str(label))
                    
                    ISsderiv_1_history[i_event] = y_som - ISs_1/5
                    iss_double_deriv = ISsderiv_1_history[i_event] -  iss_double_deriv_old
                    iss_double_deriv_old = iss_double_deriv
                    ISs_1 -= ISs_1/5
                    Act_0 -= Act_0/ISs_tau_1                
                    ISs_1 += y_som
                    Act_0[rec_closest_0] += 1
                    ISs_1_history[i_event] = ISs_1
                    Act_0_history[:,i_event] = Act_0
                        
                    #supervised
                    elem_distances_1 = (ts_lay_1[:,:]-weights_1[:,:,label])
                    # weights_1[:,:,label]+=lrate*elem_distances_1[:]
                    
                    #unsupervised
                    # elem_distances_1 = (ts_lay_1[:,:]-weights_1[:,:,rec_closest_1])
                    # weights_1[:,:,rec_closest_1]+=lrate*elem_distances_1[:]              


                    for i_cluster in range(n_clusters):
                        if i_cluster==rec_closest_0:
                            th_0[rec_closest_0] += 0.0001*y_som*np.exp(-np.abs((rec_distances_0[rec_closest_0]-th_0[rec_closest_0]))/0.5)
                        else:
                            th_0[i_cluster] -= 0.0001*y_som*np.exp(-np.abs((rec_distances_0[i_cluster]-th_0[i_cluster]))/0.5)


                    # th_0[rec_closest_0] += 0.1*expit(100*dt_y_som)*(dt_y_som>0p) - 0.1*expit(-100*dt_y_som)*(dt_y_som<0) 
                    # th_0[rec_closest_0] -= dt_y_som
    
    
    
                    # th_0[rec_closest_0] -= 0.32*expit(np.abs(y_som))*(y_som!=0)
    
                    # th_0[rec_closest_0] +=  0.001*th_0[rec_closest_0]*(y_som<=0) - 0.4*expit(y_som)*(y_som>0)
    
                    
                    # y_corr=1*(y_som==0)
        
                    # y_som_rect=y_som*(y_som>0)
                    # y_corr=y_som_rect*(y_som_rect>np.mean(y_som))
                    # y_anticorr = y_som*(y_som<0)
                    # y_anticorr = -1*(y_som<0)
        
                    
        
                    print("Y-som: "+str(y_som)+" dt Y-som: "+str(dt_y_som)+" Closest_center: "+str(rec_closest_0))
                    print(th_0)
                    # print("Y-som: "+str(y_som)+"   pY-corr: "+str(y_corr))
    
                    
                    
                    elem_distances_0 = (word_surf[ts_i,:,:,None]-weights_0[:,:,:])
                    # Keep only the distances for winners
                    elem_distances_0=elem_distances_0[:,:,:]*rec_closest_0_one_hot[None,None,:]
                    # y_corr[y_corr>1] = 1
                    #TODO the way I am normalizng the effectp of the feedback kinda makes all number learn the same (the ones with less average feedback learn the same as the ones with more)
                    #I should make sure to learn more from wrong examples than right ones.
                    # weights_0[:,:,:]+=lrate*(y_som*elem_distances_0[:])#/norm_factor
                    # y_som = np.abs(y_som)`
                    # weights_0[:,:,:]+=lrate*(y_som*(y_som>0)*elem_distances_0[:])#/norm_factor
                    # weights_0[:,:,:]+=lrate*(y_som*elem_distances_0[:])#/norm_factor
                    # weights_0[:,:,:]+=lrate*(dt_y_som*elem_distances_0[:])#/norm_factor
                    # weights_0[:,:,:]+=lrate*(dt_y_som*(dt_y_som>0)*elem_distances_0[:])#/norm_factor
                    
                # alpha = 0.99999
                # th_0[rec_closest_0] =  alpha*th_0[rec_closest_0] + (1-alpha)*rec_distances_0[rec_closest_0]

    
                # else:
                #     th_0 += 0.000003
    
    
                #NO FEEDBACK
                # weights_0[:,:,:]+=lrate*elem_distances_0[:]
    
                # th_0[np.arange(n_clusters)!=rec_closest_0] += 0.001*th_0[np.arange(n_clusters)!=rec_closest_0]
    
                
                if pause_pressed == True:    
                    circle_th0.radius = th_0[0]
                    circle_th1.radius = th_0[1]
                    fig_ths.show()
                    plt.pause(5)
                    pause_pressed=False
                
                i_event+=1
                        
                        
                    
                        

    # listener.join()
    
# fb_selected_weights_0 = weights_0
# fb_selected_weights_1 = weights_1

#%%
plt.figure()
plt.plot(Act_0_history[0])
plt.plot(Act_0_history[1])
plt.plot(ISs_1_history, alpha=0.7)
plt.axvline(s_event)
plt.axvline(r_event)



#%% New Learning rule (at the time of the proposal) two layers differential

from pynput import keyboard

surf_x = 5
surf_y = 5
n_clusters = 2

n_words = 2

weights_0 = np.random.rand(surf_x, surf_y, n_clusters)
weights_1 = np.random.rand(n_clusters, n_pol, n_words) #classifier

th_0 = np.zeros(n_clusters)+10




fig, axs = plt.subplots(n_clusters)
fig.suptitle("New L Features")

n_all_events = len(concat_all_surfs)



#initialize weights 0 to surfaces:

# for cluster_i in range(n_clusters[0]):
#     label=np.random.randint(0,9)
#     recording=np.random.randint(0,len(train_surfs_0[label]))
#     surface_i=np.random.randint(0,len(train_surfs_0[label][recording]))
#     weights_0[:,:,cluster_i]=train_surfs_0[label][recording][surface_i]

def on_press(key):
    global pause_pressed
    global lrate
    global lrate_non_boost
    global th_0
    print('{0} pressed'.format(
        key))
    if key.char == ('p'):
        pause_pressed=True
    if key.char == ('s'):
        lrate=lrate_non_boost
        # th_0 = np.zeros(n_clusters)+4



time_context_1 = np.zeros([n_clusters, n_pol],dtype=int)
time_context_fb = np.zeros([n_words],dtype=int)

tau_1 = 5
tau_1_fb = 5


lrate_1 =  0.001
lrate_0 = lrate_1

lrate_th_1 = 2*lrate_1
        
pause_pressed=False    
with keyboard.Listener(on_press=on_press) as listener:
    
    for epoch in range(3):   
        for word_i in range(len(data_surf)):
            n_events = len(data_surf[word_i])
            word_surf = data_surf[word_i]
            computed_events = 0
            event_accuracy = 0
    
            #event mask used to avoid exponentialdecay calculation forpixel 
            # that didnot generate an event yet
            mask_start_1 = np.zeros([n_clusters, n_pol],dtype=int)
            mask_start_fb = np.zeros([n_words],dtype=int)
            y_som_old=0
            y_som=0
            y_som_dt=0        
            for ts_i in range(n_events):
                
                label = data_labels[word_i]
                
                rec_distances_0=np.sum((word_surf[ts_i,:,:,None]-weights_0[:,:,:])**2,axis=(0,1))

                rec_closest_0=np.argmin(rec_distances_0-th_0,axis=0)
                rec_closest_0_one_hot = np.zeros([n_clusters])
                rec_closest_0_one_hot[rec_closest_0]=1
                
                if (rec_distances_0[rec_closest_0]-th_0[rec_closest_0])<0:
                    
                    ref_pol = data_events[word_i][2][ts_i]
                    ref_ts =  data_events[word_i][3][ts_i]
                    time_context_1[rec_closest_0,ref_pol] = ref_ts
                    mask_start_1[rec_closest_0,ref_pol]=1
                    
                    ts_lay_1 = np.exp((time_context_1-ref_ts)*mask_start_1/tau_1)*mask_start_1                
                                
                    
                    rec_distances_1=np.sum((ts_lay_1[:,:,None]-weights_1[:,:,:])**2,axis=(0,1))
                    rec_closest_1=np.argmin(rec_distances_1,axis=0)
                    
    
                    time_context_fb[rec_closest_1] = ref_ts
                    mask_start_fb[rec_closest_1]=1
                    
                    train_surfs_1_recording_fb = np.exp((time_context_fb-ref_ts)*mask_start_fb/tau_1_fb)*mask_start_fb                                 
                    norm = n_words-1
                    
                    #supervised
                    y_som=(train_surfs_1_recording_fb[label]-np.sum((train_surfs_1_recording_fb[np.arange(n_words)!=label]/norm),axis=0)) #normalized by activation
                    
                    #unsupervised
                    # y_som=(train_surfs_1_recording_fb[rec_closest_1]-np.sum((train_surfs_1_recording_fb[np.arange(n_words)!=rec_closest_1]/norm),axis=0)) #normalized by activation
    
                    
                    dt_y_som = y_som - y_som_old
                    y_som_old = y_som
                    
                    
                    rec_closest_1_one_hot = np.zeros([n_words])
                    rec_closest_1_one_hot[rec_closest_1]=1
                    class_rate=np.sum(rec_closest_1_one_hot,axis=0)
                        
                    #Layer 1
                            
                    #supervised
                    elem_distances_1 = (ts_lay_1[:,:]-weights_1[:,:,label])
                    weights_1[:,:,label]+=lrate_1*elem_distances_1[:]
                    
                    #unsupervised
                    # elem_distances_1 = (ts_lay_1[:,:]-weights_1[:,:,rec_closest_1])
                    # weights_1[:,:,rec_closest_1]+=lrate_1*elem_distances_1[:]              


                    #Layer 0
                    
                    #weights
                    elem_distances_0 = (word_surf[ts_i,:,:,None]-weights_0[:,:,:])
                    # Keep only the distances for winners
                    elem_distances_0=elem_distances_0[:,:,:]*rec_closest_0_one_hot[None,None,:]

                    weights_0[:,:,:]+=lrate_0*(dt_y_som*elem_distances_0[:]) + 0.01*lrate_0*(y_som*elem_distances_0[:])#/norm_factor

                
                    #threshold
                    for i_cluster in range(n_clusters):
                        if i_cluster==rec_closest_0:
                            th_0[rec_closest_0] += lrate_th_1*y_som*np.exp(-np.abs((rec_distances_0[rec_closest_0]-th_0[rec_closest_0]))/0.5)
                        elif ((rec_distances_0[i_cluster]-th_0[i_cluster])<0):
                            th_0[i_cluster] -= lrate_th_1*y_som*np.exp(-np.abs((rec_distances_0[i_cluster]-th_0[i_cluster]))/0.5)


                    computed_events += 1
                    if rec_closest_1==label:
                        result = "Correct"
                        event_accuracy += 1
    
                    else:
                        result = "Wrong"
                    
                    progress = computed_events/n_events
                    rel_accuracy = event_accuracy/computed_events
                        
                    print("Epoch "+str(word_i)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy))
                    print("Prediction: "+result+str(label))
                    print("Y-som: "+str(y_som)+" dt Y-som: "+str(dt_y_som)+" Closest_center: "+str(rec_closest_0))
                    print(th_0)
    
                if pause_pressed == True:    
                    if n_clusters>1:
                        for feat in range(n_clusters):
                            axs[feat].imshow(weights_0[:,:,feat] )
                            plt.draw()
                    elif n_clusters==1:
                        axs.imshow(weights_0[:,:,feat] )
                        plt.draw()
                    plt.pause(5)
                    pause_pressed=False
                    
                        
                        
                    
                        

    
fb_selected_weights_0 = weights_0
fb_selected_weights_1 = weights_1

#%% New Learning rule (at the time of the proposal) two layers differential (PROBABILTY RYAD)

#TODO learning rate drops too fast, (higher one would help to reach a perfec x and a perfect slash)
#Outside boundary calculation might create a problem (since to calculate sparsity you have to propagate the spike)
# Reducing thresholds refine ts (i don't fully understand why) but also creates situation in which the net does not produce any event

from pynput import keyboard

surf_x = 5
surf_y = 5
n_clusters = 2
ts_dim_0 = surf_x*surf_y

n_words = 2

weights_0 = np.random.rand(surf_x, surf_y, n_clusters)
weights_1 = np.random.rand(n_clusters, n_pol, n_words) #classifier
# th_0 = np.zeros(n_clusters)+6
th_0 = np.zeros(n_clusters)+30
# th_0 = np.zeros(n_clusters)
# th_0[0]=2

y_som_old=0
dt_y_som=0

# lrate_non_boost = 0.003 
lrate_non_boost = 0.004 
# lrate_boost = 1

lrate_boost = 0.004
# lrate_boost = 0.01


lrate=lrate_boost


fig, axs = plt.subplots(n_clusters)
fig.suptitle("New L Features")

n_all_events = len(concat_all_surfs)



#initialize weights 0 to surfaces:

# for cluster_i in range(n_clusters[0]):
#     label=np.random.randint(0,9)
#     recording=np.random.randint(0,len(train_surfs_0[label]))
#     surface_i=np.random.randint(0,len(train_surfs_0[label][recording]))
#     weights_0[:,:,cluster_i]=train_surfs_0[label][recording][surface_i]

def on_press(key):
    global pause_pressed
    global lrate
    global lrate_non_boost
    global th_0
    print('{0} pressed'.format(
        key))
    if key.char == ('p'):
        pause_pressed=True
    if key.char == ('s'):
        lrate=lrate_non_boost
        # th_0 = np.zeros(n_clusters)+4



time_context_1 = np.zeros([n_clusters, n_pol],dtype=int)
time_context_fb = np.zeros([n_words],dtype=int)

tau_1 = 5
        
pause_pressed=False    
with keyboard.Listener(on_press=on_press) as listener:
    
    for epoch in range(3):   
        for word_i in range(len(data_surf)):
            n_events = len(data_surf[word_i])
            word_surf = data_surf[word_i]
            progress=0
            rel_accuracy = 0
    
            #event mask used to avoid exponentialdecay calculation forpixel 
            # that didnot generate an event yet
            mask_start_1 = np.zeros([n_clusters, n_pol],dtype=int)
            mask_start_fb = np.zeros([n_words],dtype=int)
            y_som_old=0
            y_som=0
            y_som_dt=0        
            for ts_i in range(n_events):
                
                label = data_labels[word_i]
                
                rec_distances_0=np.sum((word_surf[ts_i,:,:,None]-weights_0[:,:,:])**2,axis=(0,1))
                rec_prob_0 = (1/np.sqrt(((2*np.pi))*th_0))*np.exp(-(1/(2*th_0))*rec_distances_0)

                
                
                
                #new_fb
                rec_closest_0=np.argmax(rec_prob_0,axis=0)
                rec_closest_0_one_hot = np.zeros([n_clusters])
                rec_closest_0_one_hot[rec_closest_0]=1
                
                
                
                if (rec_prob_0[rec_closest_0])>0.1:
    
    
                    
                    ref_pol = data_events[word_i][2][ts_i]
                    ref_ts =  data_events[word_i][3][ts_i]
                    time_context_1[rec_closest_0,ref_pol] = ref_ts
                    mask_start_1[rec_closest_0,ref_pol]=1
                    
                    ts_lay_1 = np.exp((time_context_1-ref_ts)*mask_start_1/tau_1)*mask_start_1                
                                
                    
                    rec_distances_1=np.sum((ts_lay_1[:,:,None]-weights_1[:,:,:])**2,axis=(0,1))
                    rec_closest_1=np.argmin(rec_distances_1,axis=0)
                    
    
                    time_context_fb[rec_closest_1] = ref_ts
                    mask_start_fb[rec_closest_1]=1
                    
                    train_surfs_1_recording_fb = np.exp((time_context_fb-ref_ts)*mask_start_fb/tau_1)*mask_start_fb                
                                
                
    
        
                   
                    norm = n_words-1
                    #supervised
                    y_som=(train_surfs_1_recording_fb[label]-np.sum((train_surfs_1_recording_fb[np.arange(n_words)!=label]/norm),axis=0)) #normalized by activation
                    
                    #unsupervised
                    # y_som=(train_surfs_1_recording_fb[label]-np.sum((train_surfs_1_recording_fb[np.arange(n_words)!=rec_closest_1]/norm),axis=0)) #normalized by activation
    
                    
                    dt_y_som = y_som - y_som_old
                    y_som_old = y_som
                    
                    # y_som_dt[1:] = (y_som[1:]-y_som[:-1])/((timestamps[1:]+1-timestamps[:-1])*0.001)
                    y_corr=y_som*(y_som>0)*(train_surfs_1_recording_fb[label]==1)
                    # np.random.shuffle(y_corr)# Test feedback modulation hypothesis with null class
                    
                    rec_closest_1_one_hot = np.zeros([n_words])
                    rec_closest_1_one_hot[rec_closest_1]=1
                    class_rate=np.sum(rec_closest_1_one_hot,axis=0)
                        
                    progress+=1/n_events
                    if rec_closest_1==label:
                        result = "Correct"
                        rel_accuracy += 1/n_events
    
                    else:
                        result = "Wrong"
                        
                    print("Epoch "+str(word_i)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy))
                    print("Prediction: "+result+str(label))
    
                    #supervised
                    elem_distances_1 = (ts_lay_1[:,:]-weights_1[:,:,label])
                    weights_1[:,:,label]+=lrate*elem_distances_1[:]
                    
                    #unsupervised
                    # elem_distances_1 = (ts_lay_1[:,:]-weights_1[:,:,rec_closest_1])
                    # weights_1[:,:,rec_closest_1]+=lrate*elem_distances_1[:]              
                
                    
                    #### YOU ARE IN THE RIGHT DIRECTION, CONTINUE HERE AND DECIDE WHAT TO DO WHEN THE EVENT IS DROPPED
                    #### MAYBE ADD A T CHaracter so you can do VTV as a new word
                    #new fb
                    # th_0[rec_closest_0] -= 0.32*expit(y_som)*(y_som>0)
                    # th_0[rec_closest_0] -= 0.90*expit(dt_y_som)*(dt_y_som>0)
    
                    # th_0[rec_closest_0] += 0.01*dt_y_som * (dt_y_som<0)
                    # th_0[rec_closest_0] += 0.01*dt_y_som 


                    # th_0[rec_closest_0] += 0.1*expit(100*dt_y_som)*(dt_y_som>0p) - 0.1*expit(-100*dt_y_som)*(dt_y_som<0) 
                    # th_0[rec_closest_0] -= dt_y_som
    
    
    
                    # th_0[rec_closest_0] -= 0.32*expit(np.abs(y_som))*(y_som!=0)
    
                    # th_0[rec_closest_0] +=  0.001*th_0[rec_closest_0]*(y_som<=0) - 0.4*expit(y_som)*(y_som>0)
    
                    
                    # y_corr=1*(y_som==0)
        
                    # y_som_rect=y_som*(y_som>0)
                    # y_corr=y_som_rect*(y_som_rect>np.mean(y_som))
                    # y_anticorr = y_som*(y_som<0)
                    # y_anticorr = -1*(y_som<0)
        
                    print("Y-som: "+str(y_som)+" dt Y-som: "+str(dt_y_som)+" Closest_center: "+str(rec_closest_0))
                    print(th_0)
                    # print("Y-som: "+str(y_som)+"   pY-corr: "+str(y_corr))
    
                    
                    
                    elem_distances_0 = (word_surf[ts_i,:,:,None]-weights_0[:,:,:])
                    # Keep only the distances for winners
                    elem_distances_0=elem_distances_0[:,:,:]*rec_closest_0_one_hot[None,None,:]
                    # y_corr[y_corr>1] = 1
                    #TODO the way I am normalizng the effectp of the feedback kinda makes all number learn the same (the ones with less average feedback learn the same as the ones with more)
                    #I should make sure to learn more from wrong examples than right ones.
                    # weights_0[:,:,:]+=lrate*(y_som*elem_distances_0[:])#/norm_factor
                    # y_som = np.abs(y_som)`
                    # weights_0[:,:,:]+=lrate*(y_som*(y_som>0)*elem_distances_0[:])#/norm_factor
                    # weights_0[:,:,:]+=lrate*(y_som*elem_distances_0[:])#/norm_factor
                    weights_0[:,:,:]+=lrate*(dt_y_som*elem_distances_0[:])#/norm_factor
                    # weights_0[:,:,:]+=lrate*(dt_y_som*(dt_y_som>0)*elem_distances_0[:])#/norm_factor
                    
                alpha = 0.99999
                th_0[rec_closest_0] =  alpha*th_0[rec_closest_0] + (1-alpha)*rec_distances_0[rec_closest_0]

    
                # else:
                #     th_0 += 0.000003
    
    
                #NO FEEDBACK
                # weights_0[:,:,:]+=lrate*elem_distances_0[:]
    
                # th_0[np.arange(n_clusters)!=rec_closest_0] += 0.001*th_0[np.arange(n_clusters)!=rec_closest_0]
    
    
                if pause_pressed == True:    
                    if n_clusters>1:
                        for feat in range(n_clusters):
                            axs[feat].imshow(weights_0[:,:,feat] )
                            plt.draw()
                    elif n_clusters==1:
                        axs.imshow(weights_0[:,:,feat] )
                        plt.draw()
                    plt.pause(5)
                    pause_pressed=False
                    
                        
                        
                    
                        

    listener.join()
    
fb_selected_weights_0 = weights_0
fb_selected_weights_1 = weights_1



#%% Plot feedback centroids

#plot centroids
fig, axs = plt.subplots(n_clusters)
for pol_i in range(n_clusters):
    if n_clusters>1:
        axs[pol_i].imshow(np.reshape(weights_0[:,:,pol_i], [5,5]))
    elif n_clusters==1:
        axs.imshow(np.reshape(weights_0[:,:,pol_i], [5,5]))
        
#%% TEST SET GENERATION

n_files_test = 500

data_events_test = []
data_labels_test = []
for i_file in range(n_files_test):
    events_p1 = word_generator(word = "v/v", sim_time=simulation_time,
                               high_f = high_freq, low_f = low_freq)
    data_events_test.append(events_p1)
    data_labels_test.append(0)
    # labels_p1 = 0*np.ones(len(events_p1),dtype=int)
    
    events_p2 = word_generator(word = "vxv", sim_time=simulation_time,
                               high_f = high_freq, low_f = low_freq)
    data_events_test.append(events_p2)
    data_labels_test.append(1)
    # labels_p2 = 1*np.ones(len(events_p2),dtype=int)
    
    # events_p3 = word_generator(word = "vtv", sim_time=simulation_time,
    #                             high_f = high_freq, low_f = low_freq)
    # data_events_test.append(events_p3)
    # data_labels_test.append(2)
    # labels_p3 = 2*np.ones(len(events_p3),dtype=int)


#%% Generate the timesurfaces
res_x = 5
res_y = 5
tau = 5000
n_pol = 3

data_surf_test=[]
for i_file in range(n_files_test):
    
    events = data_events_test[i_file]
    surfs = all_surfaces(events, res_x, res_y, tau, n_pol)
        
    
    #keep only the surface of the character for which the event as being generated
    surfs_cut = surfs[np.arange(len(surfs)),events[2],:,:]
    data_surf_test.append(surfs_cut)

#pattern plot to check all is correct
surf_i = 100
concat_surf = np.reshape(surfs, [len(surfs),15,5])
plt.figure()
plt.imshow(concat_surf[surf_i])

#%% Separability test for v/v vxv words Feedback
surf_x = 5
surf_y = 5
n_clusters = 2

n_words = 2

# th_0 = np.zeros(n_clusters)+10

#to keep count of the 
progress_history = np.zeros([n_files_test])  

rel_accuracy_history = np.zeros([n_files_test])  


time_context_1 = np.zeros([n_clusters, n_pol],dtype=int)

tau_1 = 50
        
accuracy=0
for word_i in range(n_files_test):
    n_events = len(data_surf_test[word_i])
    word_surf = data_surf_test[word_i]
    progress=0
    rel_accuracy = 0

    #event mask used to avoid exponentialdecay calculation forpixel 
    # that didnot generate an event yet
    mask_start_1 = np.zeros([n_clusters, n_pol],dtype=int)
    

    for ts_i in range(n_events):
        
        label = data_labels_test[word_i]
        
        rec_distances_0=np.sum((word_surf[ts_i,:,:,None]-fb_selected_weights_0[:,:,:])**2,axis=(0,1))
        
        rec_closest_0=np.argmin(rec_distances_0-th_0,axis=0)
        
        if (rec_distances_0[rec_closest_0]-th_0[rec_closest_0])<0:
            
            ref_pol = data_events_test[word_i][2][ts_i]
            ref_ts =  data_events_test[word_i][3][ts_i]
            time_context_1[rec_closest_0,ref_pol] = ref_ts
            mask_start_1[rec_closest_0,ref_pol]=1
            
            ts_lay_1 = np.exp((time_context_1-ref_ts)*mask_start_1/tau_1)*mask_start_1                
                                    
            rec_distances_1=np.sum((ts_lay_1[:,:,None]-fb_selected_weights_1[:,:,:])**2,axis=(0,1))
            rec_closest_1=np.argmin(rec_distances_1,axis=0)
            
                              
            progress+=1/n_events
            if rec_closest_1==label:
                result = "Correct"
                rel_accuracy += 1/n_events

            else:
                result = "Wrong"
                
            print("Epoch "+str(word_i)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy*100))
            print("Prediction: "+result+str(label))
        
    if progress!=0:
        if (rel_accuracy/progress)>0.5:
            accuracy += 1
            progress_history[word_i] = progress
            rel_accuracy_history[word_i] = rel_accuracy

#%% Separability test for v/v vxv words Kmeans
surf_x = 5
surf_y = 5
n_clusters = 2

n_words = 2

#to keep count of the 
Kmeans_progress_history = np.zeros([n_files_test])  

Kmeans_rel_accuracy_history = np.zeros([n_files_test])  


time_context_1 = np.zeros([n_clusters, n_pol],dtype=int)

tau_1 = 50
        
Kmeans_accuracy=0
for word_i in range(n_files_test):
    n_events = len(data_surf_test[word_i])
    word_surf = data_surf_test[word_i]
    progress=0
    rel_accuracy = 0

    #event mask used to avoid exponentialdecay calculation forpixel 
    # that didnot generate an event yet
    mask_start_1 = np.zeros([n_clusters, n_pol],dtype=int)
    

    for ts_i in range(n_events):
        
        label = data_labels_test[word_i]
        
        rec_distances_0=np.sum((word_surf[ts_i,:,:,None]-kmeans_weights_0[:,:,:])**2,axis=(0,1))
        
        rec_closest_0=np.argmin(rec_distances_0,axis=0)
        
            
        ref_pol = data_events_test[word_i][2][ts_i]
        ref_ts =  data_events_test[word_i][3][ts_i]
        time_context_1[rec_closest_0,ref_pol] = ref_ts
        mask_start_1[rec_closest_0,ref_pol]=1
        
        ts_lay_1 = np.exp((time_context_1-ref_ts)*mask_start_1/tau_1)*mask_start_1                
                                
        rec_distances_1=np.sum((ts_lay_1[:,:,None]-kmeans_weights_1[:,:,:])**2,axis=(0,1))
        rec_closest_1=np.argmin(rec_distances_1,axis=0)
        
                          
        progress+=1/n_events
        if rec_closest_1==label:
            result = "Correct"
            rel_accuracy += 1/n_events

        else:
            result = "Wrong"
            
        print("Epoch "+str(word_i)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy*100))
        print("Prediction: "+result+str(label))
        
    if progress!=0:
        if rel_accuracy>0.5:
            Kmeans_accuracy += 1
        Kmeans_progress_history[word_i] = progress
        Kmeans_rel_accuracy_history[word_i] = rel_accuracy


#%% Compare accuracies 

n_events = [len(data_surf_test[word_i]) for word_i in range(n_files_test) ]

plt.figure()
plt.title("Event Accuracy")
plt.plot(Kmeans_rel_accuracy_history, label="Kmeans")
plt.plot(np.nan_to_num(rel_accuracy_history/progress_history), label="Feedback")
plt.legend()
plt.ylabel("Event Accuracy (correct events/total events)")
plt.xlabel("Recording index")

plt.figure()
plt.title("N events per recording")
plt.plot(n_events, label="Kmeans")
plt.plot(n_events*progress_history, label="Feedback")
plt.ylabel("# Total events)")
plt.xlabel("Recording index")
plt.legend()



