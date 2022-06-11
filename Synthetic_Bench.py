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


simulation_time = 1e5#us

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
    
    dictionary = {"v": sub_pattern_map_1, '/': sub_pattern_map_2,
                  "x": sub_pattern_map_3}
    
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
        

events_p1 = word_generator(word = "v/v", sim_time=simulation_time,
                           high_f = high_freq, low_f = low_freq)

labels_p1 = 0*np.ones(len(events_p1),dtype=int)

events_p2 = word_generator(word = "vxv", sim_time=simulation_time,
                           high_f = high_freq, low_f = low_freq)

labels_p2 = 1*np.ones(len(events_p2),dtype=int)


#%% Generate the timesurfaces
res_x = 5
res_y = 5
tau = 5000
n_pol = 3

surfs_p1 = all_surfaces(events_p1, res_x, res_y, tau, n_pol)
surfs_p2 = all_surfaces(events_p2, res_x, res_y, tau, n_pol)

#pattern plot to check all is correct
surf_i = 100
concat_surf = np.reshape(surfs_p1, [len(surfs_p1),15,5])
plt.figure()
plt.imshow(concat_surf[surf_i])


#keep only the surface of the character for which the event as being generated
surfs_p1 = surfs_p1[np.arange(len(surfs_p1)),events_p1[2],:,:]
surfs_p2 = surfs_p2[np.arange(len(surfs_p2)),events_p2[2],:,:]

data_surf = [surfs_p1,surfs_p2]
data_labels = [0,1]
data_events = [events_p1, events_p2]

concat_all_surfs = np.concatenate(data_surf)


#%% Kmeans clustering
n_k_clusters=2
kmeans = KMeans(n_clusters=n_k_clusters)
kmeans.fit(np.reshape(concat_all_surfs, [len(concat_all_surfs),5*5]))
k_centroids= kmeans.cluster_centers_

#plot centroids
fig, axs = plt.subplots(n_k_clusters)
for pol_i in range(n_k_clusters):
    axs[pol_i].imshow(np.reshape(k_centroids[pol_i], [5,5]))
    
#%% New Learning rule (under work) two layers
from pynput import keyboard

surf_x = 5
surf_y = 5
n_clusters = 2

weights_0 = np.random.rand(surf_x, surf_y, n_clusters)
weights_1 = np.random.rand(n_clusters, n_pol, 2) #classifier
th_0 = np.zeros(n_clusters)+0.0001


lrate_non_boost = 0.0000001
# lrate_boost = 1

lrate_boost = 0.0007

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
    print('{0} pressed'.format(
        key))
    if key.char == ('p'):
        pause_pressed=True
    if key.char == ('s'):
        lrate=lrate_non_boost


time_context_1 = np.zeros([n_clusters, n_pol],dtype=int)
time_context_fb = np.zeros([2],dtype=int)

tau_1 = 50
        
pause_pressed=False    
with keyboard.Listener(on_press=on_press) as listener:
    for epoch in range(30):
        
        for word_i in range(len(data_surf)):
            n_events = len(data_surf[word_i])
            word_surf = data_surf[word_i]
            progress=0
            rel_accuracy = 0

            #event mask used to avoid exponentialdecay calculation forpixel 
            # that didnot generate an event yet
            mask_start_1 = np.zeros([n_clusters, n_pol],dtype=int)
            mask_start_fb = np.zeros([2],dtype=int)
            for ts_i in range(n_events):
                
                label = data_labels[word_i]
                
                rec_distances_0=np.sum((word_surf[ts_i,:,:,None]-weights_0[:,:,:])**2,axis=(0,1))
                
                # rec_closest_0=np.argmin(rec_distances_0,axis=0)
                #new_fb
                rec_closest_0=np.argmin(rec_distances_0-th_0,axis=0)
                rec_closest_0_one_hot = np.zeros([n_clusters])
                rec_closest_0_one_hot[rec_closest_0]=1
                
                if min(rec_distances_0-th_0)<0:


                    
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
                                
                
                    elem_distances_1 = (ts_lay_1[:,:]-weights_1[:,:,label])
                    weights_1[:,:,label]+=lrate*elem_distances_1[:]
        
                   
                    norm = 2-1
                    y_som=(train_surfs_1_recording_fb[label]-np.sum((train_surfs_1_recording_fb[np.arange(2)!=label]/norm),axis=0)) #normalized by activation

                    # y_som=train_surfs_1_recording_fb[:,label]-1
                    # y_som_dt = np.zeros(len(y_som))
                    # y_som_dt[1:] = (y_som[1:]-y_som[:-1])/((timestamps[1:]+1-timestamps[:-1])*0.001)
                    y_corr=y_som*(y_som>0)*(train_surfs_1_recording_fb[label]==1)
                    # np.random.shuffle(y_corr)# Test feedback modulation hypothesis with null class
                    
                    rec_closest_1_one_hot = np.zeros([2])
                    rec_closest_1_one_hot[rec_closest_1]=1
                    class_rate=np.sum(rec_closest_1_one_hot,axis=0)
                        
                    progress+=1/n_events
                    if rec_closest_1==label:
                        result = "Correct"
                        rel_accuracy += 1/n_events
    
                    else:
                        result = "Wrong"
                        
                    print("Epoch "+str(epoch)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy))
                    print("Prediction: "+result+str(label))
                        
                else:
                    y_som=0
                    
                #new fb
                th_0 += 0.001*th_0 
                th_0[rec_closest_0] -= 0.32*expit(y_som)*(y_som>0)
                # th_0[rec_closest_0] -= 0.32*expit(np.abs(y_som))*(y_som!=0)

                # th_0[rec_closest_0] +=  0.001*th_0[rec_closest_0]*(y_som<=0) - 0.4*expit(y_som)*(y_som>0)

                
                # y_corr=1*(y_som==0)
    
                # y_som_rect=y_som*(y_som>0)
                # y_corr=y_som_rect*(y_som_rect>np.mean(y_som))
                # y_anticorr = y_som*(y_som<0)
                # y_anticorr = -1*(y_som<0)
    
                print("Y-som: "+str(y_som)+" Closest_center: "+str(rec_closest_0))
                # print("Y-som: "+str(y_som)+"   Y-corr: "+str(y_corr))

                
                
                elem_distances_0 = (word_surf[ts_i,:,:,None]-weights_0[:,:,:])
                # Keep only the distances for winners
                elem_distances_0=elem_distances_0[:,:,:]*rec_closest_0_one_hot[None,None,:]
                # y_corr[y_corr>1] = 1
                #TODO the way I am normalizng the effectp of the feedback kinda makes all number learn the same (the ones with less average feedback learn the same as the ones with more)
                #I should make sure to learn more from wrong examples than right ones.
                # weights_0[:,:,:]+=lrate*(y_som*elem_distances_0[:])#/norm_factor
                # y_som = np.abs(y_som)`
                # weights_0[:,:,:]+=lrate*(y_som*(y_som>0)*elem_distances_0[:])#/norm_factor
                weights_0[:,:,:]+=lrate*(y_som*elem_distances_0[:])#/norm_factor


                #NO FEEDBACK
                # weights_0[:,:,:]+=lrate*elem_distances_0[:]

    
    
                if pause_pressed == True:     
                    for feat in range(n_clusters):
                        axs[feat].imshow(weights_0[:,:,feat] )
                        plt.draw()
                    plt.pause(5)
                    pause_pressed=False
                
                    
                    
                
                    

    listener.join()

#%% Plot feedback centroids

#plot centroids
fig, axs = plt.subplots(n_clusters)
for pol_i in range(n_clusters):
    axs[pol_i].imshow(np.reshape(weights_0[:,:,pol_i], [5,5]))