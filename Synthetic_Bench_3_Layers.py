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
        word (string): word composed of the available characters: v,x,/,t,y
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
    
    events_p1 = word_generator(word = "ytyv/v", sim_time=simulation_time,
                               high_f = high_freq, low_f = low_freq)
    
    events_p2 = word_generator(word = "ytyvxv", sim_time=simulation_time,
                               high_f = high_freq, low_f = low_freq)
    
    data_events.append(events_p1)
    data_labels.append(0)
    
    
    data_events.append(events_p2)
    data_labels.append(1)


#%% Generate the timesurfaces
res_x = 5
res_y = 5
tau = 5000
n_pol = 6
    
data_surf=[]

characters_ts = [[], [], [], [], []]
for i_file in range(n_files):
    
    events = data_events[i_file]
    surfs = all_surfaces(events, res_x, res_y, tau, n_pol)
        
    
    #keep only the surface of the character for which the event as being generated
    surfs_cut = surfs[np.arange(len(surfs)),events[2],:,:]
    data_surf.append(surfs_cut)
    for i,event in enumerate(events[2]):
        if event == 0 or event == 2:#y
            characters_ts[0].append(surfs_cut[i])
        if event == 1:#t
            characters_ts[1].append(surfs_cut[i])
        if event == 3 or event==5:#v
            characters_ts[2].append(surfs_cut[i])
        if event == 4:#\x
            characters_ts[3+data_labels[i_file]].append(surfs_cut[i])
            
        

#pattern plot to check all is correct
surf_i = 100
concat_surf = np.reshape(surfs, [len(surfs),5*6,5])
plt.figure()
plt.imshow(concat_surf[surf_i])
concat_all_surfs = np.concatenate(data_surf)


#%% Kmeans clustering
n_k_clusters=5
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





#%% New Learning rule (at the time of the proposal) two layers differential
##TODO second hidden get stuck so lower only learns the two available features VXV (V,X)
from pynput import keyboard

surf_x = 5
surf_y = 5
n_clusters_0 = 5 
n_clusters_1 = 6

#Each word is 3 characters long and sentences have no spaces
#Between words
n_words = 2
n_sentences = 2
word_length = 3 

weights_0 = np.random.rand(surf_x, surf_y, n_clusters_0)
weights_1 = np.random.rand(n_clusters_0, word_length, n_clusters_1) 
weights_2 = np.random.rand(n_clusters_1, n_words, n_sentences) #classifier

th_0 = np.zeros(n_clusters_0)+10
th_1 = np.zeros(n_clusters_1)+10



fig, axs = plt.subplots(n_clusters_0)
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
    global print_lay
    
    print('{0} pressed'.format(
        key))
    if key.char == ('p'):
        pause_pressed=True
    if key.char == ('1'):
        print_lay=1
    if key.char == ('2'):
        print_lay=2



time_context_1 = np.zeros([n_clusters_0, n_pol],dtype=int)
time_context_fb_1 = np.zeros([n_clusters_1],dtype=int)

time_context_2 = np.zeros([n_clusters_1, n_words],dtype=int)
time_context_fb_2 = np.zeros([n_sentences],dtype=int)

tau_1 = 5
tau_2 = 5

lrate_2 = 0.0005
lrate_1 = 1.5*lrate_2
lrate_0 = 1.5*lrate_1

###TODO THE TH0 AFFECTS COMPUTATION, IF TO FAST CLASSES GET'S CUT OUT 
###seems that more units does not fix that, weirdly enough. too slow and other
## characters creeps in.
lrate_th_1 = 5*lrate_1
lrate_th_0 = 5*lrate_0


pause_pressed=False  
print_lay=2  
with keyboard.Listener(on_press=on_press) as listener:
    
    for epoch in range(3):   
        for sentence_i in range(len(data_surf)):
            n_events = len(data_surf[sentence_i])
            sentence_surfs = data_surf[sentence_i]
            computed_events = 0
            event_accuracy = 0
    
            #event mask used to avoid exponential decay calculation for pixel 
            # that did not generate an event yet
            mask_start_1 = np.zeros([n_clusters_0, n_pol],dtype=int)
            mask_start_fb_1 = np.zeros([n_clusters_1],dtype=int)
            
            mask_start_2 = np.zeros([n_clusters_1, n_words],dtype=int)
            mask_start_fb_2 = np.zeros([n_sentences],dtype=int)
            
            y_som_0=0
            y_som_old_0=0
            dt_y_som_0=0
           
            y_som_1=0
            y_som_old_1=0
            dt_y_som_1=0   
            
            for ts_i in range(n_events):
                
                label = data_labels[sentence_i]
                ref_ch_pos = data_events[sentence_i][2][ts_i]
                ref_word_pos = ref_ch_pos//word_length
                ref_ts =  data_events[sentence_i][3][ts_i]
                
                rec_distances_0=np.sum((sentence_surfs[ts_i,:,:,None]-weights_0[:,:,:])**2,axis=(0,1))

                
                # Closest center with threshold computation
                # rec_closest_0=np.argmin(rec_distances_0-th_0,axis=0)
                rec_closest_0=np.argmin(rec_distances_0,axis=0)


                
                # Layer 1 check
                if (rec_distances_0[rec_closest_0]-th_0[rec_closest_0])<0:
                       
                    time_context_1[rec_closest_0,ref_ch_pos] = ref_ts
                    mask_start_1[rec_closest_0,ref_ch_pos]=1
                    
                    # Extracting the single word ts
                    beg_ch_index = ref_word_pos*word_length
                    end_ch_index = (ref_word_pos+1)*word_length
                    ts_lay_1 = np.exp((time_context_1[:,beg_ch_index:end_ch_index]\
                                       -ref_ts)*mask_start_1[:,beg_ch_index:end_ch_index]/\
                                          tau_1)*mask_start_1[:,beg_ch_index:end_ch_index]    

                    
                    rec_distances_1=np.sum((ts_lay_1[:,:,None]-weights_1[:,:,:])**2,axis=(0,1))

                    
                    # rec_closest_1=np.argmin(rec_distances_1-th_1,axis=0)
                    rec_closest_1=np.argmin(rec_distances_1,axis=0)

                    
                    # Layer 2 check
                    if (rec_distances_1[rec_closest_1]-th_1[rec_closest_1])<0:

                        time_context_2[rec_closest_1,ref_word_pos] = ref_ts
                        mask_start_2[rec_closest_1,ref_word_pos]=1
                        
                        ts_lay_2 = np.exp((time_context_2-ref_ts)*mask_start_2/\
                                              tau_2)*mask_start_2                                 
                        
                        rec_distances_2=np.sum((ts_lay_2[:,:,None]-weights_2[:,:,:])**2,axis=(0,1))

                        
                        rec_closest_2=np.argmin(rec_distances_2,axis=0)
                        
                        
                        ##FEEDBACK CALCULATIONS
                        
                        #Layer 2
                        time_context_fb_2[rec_closest_2] = ref_ts
                        mask_start_fb_2[rec_closest_2]=1    
                        ts_fb_lay_2 = np.exp((time_context_fb_2-ref_ts)*mask_start_fb_2/tau_2)*mask_start_fb_2                                                        
                        norm = n_sentences-1
                        
                        #supervised
                        y_som_1=(ts_fb_lay_2[label]-np.sum((ts_fb_lay_2[np.arange(n_sentences)!=label]/norm),axis=0)) #normalized by activation
                        
                        #unsupervised
                        # y_som_1=(ts_fb_lay_2[rec_closest_2]-np.sum((ts_fb_lay_2[np.arange(n_sentences)!=rec_closest_2]/norm),axis=0)) #normalized by activation
        
        
                        dt_y_som_1 = y_som_1 - y_som_old_1
                        y_som_old_1 = y_som_1
                        
                        #Layer 1
                        time_context_fb_1[rec_closest_1] = ref_ts
                        mask_start_fb_1[rec_closest_1]=1    
                        ts_fb_lay_1 = np.exp((time_context_fb_1-ref_ts)*mask_start_fb_1/tau_1)*mask_start_fb_1                                                        
                        norm = n_clusters_1-1
                        
                        y_som_0=(ts_fb_lay_1[rec_closest_1]-np.sum((ts_fb_lay_1[np.arange(n_clusters_1)!=rec_closest_1]/norm),axis=0)) #normalized by activation
                        
                        y_som_0  = np.sign(y_som_1)*np.abs(y_som_0)
            
                        dt_y_som_0 = y_som_0 - y_som_old_0
                        y_som_old_0 = y_som_0
                        
                        
                        ##WEIGHTS AND TH UPDATE

                        #Layer 2
                        #supervised
                        elem_distances_2 = (ts_lay_2[:,:]-weights_2[:,:,label])
                        weights_2[:,:,label]+=lrate_2*elem_distances_2[:]
                        
                        #unsupervised       
                        # elem_distances_2 = (ts_lay_2[:,:]-weights_2[:,:,rec_closest_1])
                        # weights_2[:,:,rec_closest_1]+=lrate_2*elem_distances_2[:]

                        #Layer 1
                        #weights
                        elem_distances_1 = (ts_lay_1[:,:]-weights_1[:,:,rec_closest_1])
                        # rec_closest_1_one_hot = np.zeros([n_clusters_1])
                        # rec_closest_1_one_hot[rec_closest_1]=1
                        
                        # keep only the distances for winners
                        # elem_distances_1=elem_distances_1[:,:,:]*rec_closest_1_one_hot[None,None,:]
                        weights_1[:,:,rec_closest_1]+=lrate_1*(dt_y_som_1*elem_distances_1[:]) + 0.01*lrate_1*(y_som_1*elem_distances_1[:])
                       
                        #treshold
                        for i_cluster in range(n_clusters_1):
                            if i_cluster==rec_closest_1:
                                th_1[rec_closest_1] += lrate_th_1*y_som_1*np.exp(-np.abs((rec_distances_1[rec_closest_1]-th_1[rec_closest_1]))/0.5)
                            elif ((rec_distances_1[i_cluster]-th_1[i_cluster])<0) and (y_som_1>0):
                            # else:
                                th_1[i_cluster] -= lrate_th_1*y_som_1*np.exp(-np.abs((rec_distances_1[i_cluster]-th_1[i_cluster]))/0.5)
        
        
                        #Layer 0
                        #weights
                        elem_distances_0 = (sentence_surfs[ts_i,:,:,None]-weights_0[:,:,:])
                        rec_closest_0_one_hot = np.zeros([n_clusters_0])
                        rec_closest_0_one_hot[rec_closest_0]=1
                        
                        # Keep only the distances for winners
                        elem_distances_0=elem_distances_0[:,:,:]*rec_closest_0_one_hot[None,None,:]
                        weights_0[:,:,:]+= (lrate_0*(dt_y_som_0*elem_distances_0[:]) + 0.01*lrate_0*(y_som_0*elem_distances_0[:]))
                       
                        #treshold
                        for i_cluster in range(n_clusters_0):
                            if i_cluster==rec_closest_0:
                                th_0[rec_closest_0] += lrate_th_0*dt_y_som_0*np.exp(-np.abs((rec_distances_0[rec_closest_0]-th_0[rec_closest_0]))/0.5) + 0.01*lrate_th_0*y_som_0*np.exp(-np.abs((rec_distances_0[rec_closest_0]-th_0[rec_closest_0]))/0.5)
                            elif ((rec_distances_0[i_cluster]-th_0[i_cluster])<0) and (y_som_0>0) and (dt_y_som_0>0):
                            # else:
                                th_0[i_cluster] -= lrate_th_0*dt_y_som_0*np.exp(-np.abs((rec_distances_0[i_cluster]-th_0[i_cluster]))/0.5) + 0.01*lrate_th_0*y_som_0*np.exp(-np.abs((rec_distances_0[i_cluster]-th_0[i_cluster]))/0.5)
                              
                        

                        ## PROGRESS UPDATE
                        rec_closest_2_one_hot = np.zeros([n_sentences])
                        rec_closest_2_one_hot[rec_closest_2]=1
                        class_rate=np.sum(rec_closest_2_one_hot,axis=0)
                            
                        computed_events += 1
                        if rec_closest_2==label:
                            result = "Correct"
                            event_accuracy += 1
        
                        else:
                            result = "Wrong"
                        
                        progress = computed_events/n_events
                        rel_accuracy = event_accuracy/computed_events
                        print("Epoch "+str(epoch)+", Sentence "+str(sentence_i)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy))
                        print("Prediction: "+result+str(label))
                        
                        if print_lay==1:
                            #Layer0
                            print("Y-som: "+str(y_som_0)+" dt Y-som: "+str(dt_y_som_0)+" Closest_center: "+str(rec_closest_0))
                            print(rec_distances_0-th_0)
                            print(th_0)
                        elif print_lay==2:
                            #Layer1
                            print("Y-som: "+str(y_som_1)+" dt Y-som: "+str(dt_y_som_1)+" Closest_center: "+str(rec_closest_1))
                            print(th_1)

                        
    
                if pause_pressed == True:    
                    if n_clusters_0>1:
                        for feat in range(n_clusters_0):
                            axs[feat].imshow(weights_0[:,:,feat])
                            plt.draw()
                    elif n_clusters_0==1:
                        axs.imshow(weights_0[:,:,feat])
                        plt.draw()
                    plt.pause(5)
                    pause_pressed=False
                    
                        
                        
                    
                        

    listener.join()
    
fb_selected_weights_0 = weights_0
fb_selected_weights_1 = weights_1



#%% New Learning rule (dynamical thresholds!) two layers differential
##TODO second hidden get stuck so lower only learns the two available features VXV (V,X)
from pynput import keyboard

surf_x = 5
surf_y = 5
n_clusters_0 = 3 
n_clusters_1 = 6

#Each word is 3 characters long and sentences have no spaces
#Between words
n_words = 2
n_sentences = 2
word_length = 3 

weights_0 = np.random.rand(surf_x, surf_y, n_clusters_0)
weights_1 = np.random.rand(n_clusters_0, word_length, n_clusters_1) 
weights_2 = np.random.rand(n_clusters_1, n_words, n_sentences) #classifier

th_0 = np.zeros(n_clusters_0)+10
th_1 = np.zeros(n_clusters_1)+10


#%% PRETRAINING

n_pre_t_recs = 100
ratio_kmeans = 0.5
total_evs = sum([len(data_surf[sentence_i]) for sentence_i in range(n_pre_t_recs)])

#######KMEANS#####
kmeans = KMeans(n_clusters=n_clusters_0)
kmeans.fit(np.reshape(concat_all_surfs[:total_evs], [len(concat_all_surfs[:total_evs]),5*5]))
k_centroids= kmeans.cluster_centers_

#plot centroids
fig, axs = plt.subplots(n_clusters_0)
kmeans_weights_0 = np.random.rand(surf_x, surf_y, n_clusters_0)

for pol_i in range(n_clusters_0):
    axs[pol_i].imshow(np.reshape(k_centroids[pol_i], [5,5]))
    kmeans_weights_0[:,:,pol_i]=np.reshape(k_centroids[pol_i], [5,5])

weights_0 = ratio_kmeans*kmeans_weights_0 + (1-ratio_kmeans)*weights_0

##################


all_ts_lay_1 = np.zeros([total_evs, n_clusters_0, word_length])

total_ev_i = 0   
for sentence_i in range(n_pre_t_recs):
    n_events = len(data_surf[sentence_i])
    sentence_surfs = data_surf[sentence_i]
    
    
    time_context_1 = np.zeros([n_clusters_0, n_pol],dtype=int)
    mask_start_1 = np.zeros([n_clusters_0, n_pol],dtype=int)  
    time_context_2 = np.zeros([n_clusters_1, n_words],dtype=int)
    mask_start_2 = np.zeros([n_clusters_1, n_words],dtype=int)


    for ts_i in range(n_events):
        
        label = data_labels[sentence_i]
        ref_ch_pos = data_events[sentence_i][2][ts_i]
        ref_word_pos = ref_ch_pos//word_length
        ref_ts =  data_events[sentence_i][3][ts_i]
        
        rec_distances_0=np.sum((sentence_surfs[ts_i,:,:,None]-weights_0[:,:,:])**2,axis=(0,1))

        
        # Closest center with threshold computation
        rec_closest_0=np.argmin(rec_distances_0,axis=0)
        
               
        time_context_1[rec_closest_0,ref_ch_pos] = ref_ts
        mask_start_1[rec_closest_0,ref_ch_pos]=1
        
        # Extracting the single word ts
        beg_ch_index = ref_word_pos*word_length
        end_ch_index = (ref_word_pos+1)*word_length
        all_ts_lay_1[total_ev_i] = np.exp((time_context_1[:,beg_ch_index:end_ch_index]\
                           -ref_ts)*mask_start_1[:,beg_ch_index:end_ch_index]/\
                              tau_1)*mask_start_1[:,beg_ch_index:end_ch_index]    


        
        total_ev_i+=1


#######KMEANS#####
kmeans = KMeans(n_clusters=n_clusters_1)
kmeans.fit(np.reshape(all_ts_lay_1, [len(all_ts_lay_1),n_clusters_0*word_length]))
k_centroids= kmeans.cluster_centers_

kmeans_weights_1 = np.random.rand(n_clusters_0, word_length, n_clusters_1)

for pol_i in range(n_clusters_1):
    kmeans_weights_1[:,:,pol_i]=np.reshape(k_centroids[pol_i], [n_clusters_0,word_length])

weights_1 = ratio_kmeans*kmeans_weights_1 + (1-ratio_kmeans)*weights_1

##################


label_mean_weights = np.random.rand(n_clusters_1, n_words, n_sentences)
max_dist_0 = 0
max_dist_1 = 0

n_sentence_per_label = np.zeros(n_sentences)
for sentence_i in range(n_pre_t_recs):
    n_events = len(data_surf[sentence_i])
    sentence_surfs = data_surf[sentence_i]
    
    
    time_context_1 = np.zeros([n_clusters_0, n_pol],dtype=int)
    mask_start_1 = np.zeros([n_clusters_0, n_pol],dtype=int)  
    time_context_2 = np.zeros([n_clusters_1, n_words],dtype=int)
    mask_start_2 = np.zeros([n_clusters_1, n_words],dtype=int)
    label = data_labels[sentence_i]
    
    n_sentence_per_label[label] += 1
    
    for ts_i in range(n_events):
        
        ref_ch_pos = data_events[sentence_i][2][ts_i]
        ref_word_pos = ref_ch_pos//word_length
        ref_ts =  data_events[sentence_i][3][ts_i]
        
        rec_distances_0=np.sum((sentence_surfs[ts_i,:,:,None]-weights_0[:,:,:])**2,axis=(0,1))

        
        # Closest center with threshold computation
        rec_closest_0=np.argmin(rec_distances_0,axis=0)
        
        if np.max(rec_distances_0,axis=0)>max_dist_0:
            max_dist_0 = np.max(rec_distances_0,axis=0)
        
               
        time_context_1[rec_closest_0,ref_ch_pos] = ref_ts
        mask_start_1[rec_closest_0,ref_ch_pos]=1
        
        # Extracting the single word ts
        beg_ch_index = ref_word_pos*word_length
        end_ch_index = (ref_word_pos+1)*word_length
        ts_lay_1 = np.exp((time_context_1[:,beg_ch_index:end_ch_index]\
                           -ref_ts)*mask_start_1[:,beg_ch_index:end_ch_index]/\
                              tau_1)*mask_start_1[:,beg_ch_index:end_ch_index]    

        
        rec_distances_1=np.sum((ts_lay_1[:,:,None]-weights_1[:,:,:])**2,axis=(0,1))

        
        rec_closest_1=np.argmin(rec_distances_1,axis=0)

        if np.max(rec_distances_1,axis=0)>max_dist_1:
            max_dist_1 = np.max(rec_distances_1,axis=0)

        time_context_2[rec_closest_1,ref_word_pos] = ref_ts
        mask_start_2[rec_closest_1,ref_word_pos]=1
        
        ts_lay_2 = np.exp((time_context_2-ref_ts)*mask_start_2/\
                              tau_2)*mask_start_2   

        label_mean_weights[:,:,label] += ts_lay_2/n_events                  

for label_i in range(n_sentences):
    label_mean_weights[:,:,label_i] = label_mean_weights[:,:,label_i]/n_sentence_per_label[label_i]     
                
                   
weights_2 = ratio_kmeans*label_mean_weights + (1-ratio_kmeans)*weights_2

th_0 = np.zeros(n_clusters_0)+max_dist_0
th_1 = np.zeros(n_clusters_1)+max_dist_1

#%% Training
fig, axs = plt.subplots(n_clusters_0)
fig.suptitle("New L Features")

n_all_events = len(concat_all_surfs)



#initialize weights 0 to surfaces:

# for cluster_i in range(n_clusters[0]):
#     label=np.random.randint(0,9)
#     recording=np.random.randint(0,len(train_surfs_0[label]))
#     surface_i=np.random.randint(0,len(train_surfs_0[label][recording]))lrate_0
#     weights_0[:,:,cluster_i]=train_surfs_0[label][recording][surface_i]

def on_press(key):
    global pause_pressed
    global print_lay
    global slow_learning
    global th_0
    global th_1
    print('{0} pressed'.format(
        key))
    if key.char == ('p'):
        pause_pressed=True
    if key.char == ('1'):
        print_lay=1
    if key.char == ('2'):
        print_lay=2
    if key.char == ('s'):
        slow_learning=True
        th_0 = np.zeros(n_clusters_0)+10
        th_1 = np.zeros(n_clusters_1)+10


time_context_1 = np.zeros([n_clusters_0, n_pol],dtype=int)
time_context_fb_1 = np.zeros([n_clusters_1, n_words],dtype=int)

time_context_2 = np.zeros([n_clusters_1, n_words],dtype=int)
time_context_fb_2 = np.zeros([n_sentences],dtype=int)

tau_1 = 1
tau_2 = 1

lrate_2 = 0.0001
lrate_1 = 0.0001
lrate_0 = 0.0001

lrate_2_pre = 0.0005
lrate_1_pre = 0.0005
lrate_0_pre = 0.0005

###TODO THE TH0 AFFECTS COMPUTATION, IF TO FAST CLASSES GET'S CUT OUT 
###seems that more units does not fix that, weirdly enough. too slow and other
## characters creeps in.lrate_0

lrate_th_1 = 0.0001
lrate_th_0 = 0.0001

th_tau_rel = 0.1


feedback_sep = 0.0001

ratio_sds = 0.01

pause_pressed=False  
slow_learning = True
print_lay=2  
with keyboard.Listener(on_press=on_press) as listener:
    # begin = time.time()
    for epoch in range(1): 
        sentence_accuracy = np.zeros(len(data_surf))
        for sentence_i in range(1):
            n_events = len(data_surf[sentence_i])
            sentence_surfs = data_surf[sentence_i]
            computed_events = 0
            event_accuracy = 0
    
            #event mask used to avoid exponential decay calculation for pixel 
            # that did not generate an event yet
            mask_start_1 = np.zeros([n_clusters_0, n_pol],dtype=int)
            mask_start_fb_1 = np.zeros([n_clusters_1, n_words],dtype=int)
            
            mask_start_2 = np.zeros([n_clusters_1, n_words],dtype=int)
            mask_start_fb_2 = np.zeros([n_sentences],dtype=int)
            
            y_som_0=np.zeros(2)
            y_som_old_0=np.zeros(2)
            dt_y_som_0=np.zeros(2)
           
            y_som_1=0
            y_som_old_1=0
            dt_y_som_1=0   
            
            max_th_0 = 0
            max_th_1 = 0
            
            if not slow_learning:        
                th_0 = np.zeros(n_clusters_0)+10
                th_1 = np.zeros(n_clusters_1)+10
                
            for ts_i in range(1):
                

                label = data_labels[sentence_i]
                ref_ch_pos = data_events[sentence_i][2][ts_i]
                ref_word_pos = ref_ch_pos//word_length
                ref_ts =  data_events[sentence_i][3][ts_i]
                
                rec_distances_0=np.sum((sentence_surfs[ts_i,:,:,None]-weights_0[:,:,:])**2,axis=(0,1))

                
                # Closest center with threshold computation
                rec_closest_0=np.argmin(rec_distances_0-th_0,axis=0)
                # rec_closest_0=np.argmin(rec_distances_0,axis=0)
                
                if not slow_learning:
                    if (rec_distances_0[rec_closest_0] > max_th_0):
                        max_th_0 = rec_distances_0[rec_closest_0]
                
                # Layer 1 check
                if (rec_distances_0[rec_closest_0]-th_0[rec_closest_0])<0:
                       
                    time_context_1[rec_closest_0,ref_ch_pos] = ref_ts
                    mask_start_1[rec_closest_0,ref_ch_pos]=1
                    
                    # Extracting the single word ts
                    beg_ch_index = ref_word_pos*word_length
                    end_ch_index = (ref_word_pos+1)*word_length
                    ts_lay_1 = np.exp((time_context_1[:,beg_ch_index:end_ch_index]\
                                       -ref_ts)*mask_start_1[:,beg_ch_index:end_ch_index]/\
                                          tau_1)*mask_start_1[:,beg_ch_index:end_ch_index]    

                    
                    rec_distances_1=np.sum((ts_lay_1[:,:,None]-weights_1[:,:,:])**2,axis=(0,1))

                    
                    rec_closest_1=np.argmin(rec_distances_1-th_1,axis=0)
                    # rec_closest_1=np.argmin(rec_distances_1,axis=0)

                    if not slow_learning:
                        if (rec_distances_1[rec_closest_1] > max_th_1):
                            max_th_1 = rec_distances_1[rec_closest_0]
                    
                    # Layer 2 check
                    if (rec_distances_1[rec_closest_1]-th_1[rec_closest_1])<0:

                        time_context_2[rec_closest_1,ref_word_pos] = ref_ts
                        mask_start_2[rec_closest_1,ref_word_pos]=1
                        
                        ts_lay_2 = np.exp((time_context_2-ref_ts)*mask_start_2/\
                                              tau_2)*mask_start_2                                 
                        
                        rec_distances_2=np.sum((ts_lay_2[:,:,None]-weights_2[:,:,:])**2,axis=(0,1))

                        
                        rec_closest_2=np.argmin(rec_distances_2,axis=0)
                        
                        
                        ##FEEDBACK CALCULATIONS
                        
                        #Layer 2
                        time_context_fb_2[rec_closest_2] = ref_ts
                        mask_start_fb_2[rec_closest_2]=1    
                        ts_fb_lay_2 = np.exp((time_context_fb_2-ref_ts)*mask_start_fb_2/tau_2)*mask_start_fb_2                                                        
                        norm = n_sentences-1
                        
                        #supervised
                        y_som_1=(ts_fb_lay_2[rec_closest_2]-np.sum((ts_fb_lay_2[np.arange(n_sentences)!=rec_closest_2]/norm),axis=0)) #normalized by activation
                        if label != rec_closest_2:
                            y_som_1=-y_som_1
                        
                        #unsupervised
                        # y_som_1=(ts_fb_lay_2[rec_closest_2]-np.sum((ts_fb_lay_2[np.arange(n_sentences)!=rec_closest_2]/norm),axis=0)) #normalized by activation
        
        
                        # dt_y_som_1 = np.sign(y_som_1)*np.abs(y_som_1 - y_som_old_1)
                        dt_y_som_1 = (y_som_1 - y_som_old_1)

                        y_som_old_1 = y_som_1.copy()
                        
                        #Layer 1
                        time_context_fb_1[rec_closest_1, ref_word_pos] = ref_ts
                        mask_start_fb_1[rec_closest_1, ref_word_pos]=1    
                        ts_fb_lay_1 = np.exp((time_context_fb_1-ref_ts)*mask_start_fb_1/tau_1)*mask_start_fb_1                                                        
                        norm = n_clusters_1-1
                        
                        y_som_0[ref_word_pos] = (ts_fb_lay_1[rec_closest_1, ref_word_pos]-np.sum((ts_fb_lay_1[np.arange(n_clusters_1)!=rec_closest_1, ref_word_pos]/norm),axis=0)) #normalized by activation
                        
                        y_som_0  = np.sign(y_som_1)*np.abs(y_som_0)
            
                        # dt_y_som_0 =  np.sign(y_som_1)*np.abs(y_som_0 - y_som_old_0)
                        dt_y_som_0 =  y_som_0 - y_som_old_0

                        y_som_old_0 = y_som_0.copy()
                        
                        w_y_som_0 = y_som_0[ref_word_pos]
                        w_dt_y_som_0 = dt_y_som_0[ref_word_pos]
                        
                        ##WEIGHTS AND TH UPDATE

                        #Layer 2                       
                        #supervised
                        elem_distances_2 = (ts_lay_2[:,:]-weights_2[:,:,rec_closest_2])
                        if slow_learning:
                            weights_2[:,:,rec_closest_2]+=lrate_2*(dt_y_som_1*elem_distances_2[:]) + ratio_sds*lrate_2*(y_som_1*elem_distances_2[:])
                        else:
                            weights_2[:,:,rec_closest_2]+= lrate_2_pre*(dt_y_som_1*elem_distances_2[:] + 0*ratio_sds*(y_som_1*elem_distances_2[:]))

                            
                            
                        #unsupervised       
                        # elem_distances_2 = (ts_lay_2[:,:]-weights_2[:,:,rec_closest_1])
                        # weights_2[:,:,rec_closest_1]+=lrate_2*elem_distances_2[:]

                        #Layer 1
                        #weights
                        elem_distances_1 = (ts_lay_1[:,:]-weights_1[:,:,rec_closest_1])
                        # rec_closest_1_one_hot = np.zeros([n_clusters_1])
                        # rec_closest_1_one_hot[rec_closest_1]=1
                        
                        # keep only the distances for winners
                        # elem_distances_1=elem_distances_1p[:,:,:]*rec_closest_1_one_hot[None,None,:]
                        if slow_learning: 
                            weights_1[:,:,rec_closest_1]+=lrate_1*(dt_y_som_1*elem_distances_1[:]) + ratio_sds*lrate_1*(y_som_1*elem_distances_1[:])
                        else:
                            weights_1[:,:,rec_closest_1]+= lrate_1_pre*(dt_y_som_1*elem_distances_1[:] + 0*ratio_sds*(y_som_1*elem_distances_1[:]))

                        #treshold
                        if slow_learning:
                            for i_cluster in range(n_clusters_1): # THIS WORKS
                                th_tau = th_tau_rel*np.mean(th_1[i_cluster])
                                # th_tau = 0.5
                                if i_cluster==rec_closest_1:
                                    th_dist = th_1[rec_closest_1]-rec_distances_1[rec_closest_1]
                                    # th_update = np.exp(-th_dist/th_tau)*th_dist
                                    th_update = np.exp(-th_dist/th_tau)*th_1[rec_closest_1]
                                    th_1[rec_closest_1] += lrate_th_1*dt_y_som_1*th_update + ratio_sds*lrate_th_1*y_som_1*th_update
                                elif ((rec_distances_1[i_cluster]-th_1[i_cluster])<0) and (y_som_1>=0) and (dt_y_som_1>=0):
                                # else:
                                    th_dist = th_1[i_cluster]-rec_distances_1[i_cluster]
                                    # th_update = np.exp(-th_dist/th_tau)*th_dist
                                    th_update = np.exp(-th_dist/th_tau)*th_1[i_cluster]
                                    th_1[i_cluster] -= lrate_th_1*dt_y_som_1*th_update + ratio_sds*lrate_th_1*y_som_1*th_update
                                    
                            # for i_cluster in range(n_clusters_1):
                            #     th_tau = th_tau_rel*np.abs((th_1[i_cluster]-rec_distances_1[i_cluster]))
                            #     # th_tau = 0.5
                            #     if i_cluster==rec_closest_1:
                            #         th_1[rec_closest_1] += lrate_th_1*dt_y_som_1*np.exp(-np.abs((rec_distances_1[rec_closest_1]-th_1[rec_closest_1]))/th_tau)+0.01*lrate_th_1*y_som_1*np.exp(-np.abs((rec_distances_1[rec_closest_1]-th_1[rec_closest_1]))/th_tau)
                            #     elif ((rec_distances_1[i_cluster]-th_1[i_cluster])<0) and (y_som_1>=0) and (dt_y_som_1>=0):
                            #     # else:
                            #         th_1[i_cluster] -= 0.01*lrate_th_1*th_1[i_cluster]
                                    
                        if not slow_learning:                            
                            #fast_threshold_control
                            for i_cluster in range(n_clusters_1):
                                if (i_cluster==rec_closest_1) and (y_som_1<0):
                                    th_1[i_cluster] -=  feedback_sep*th_1[i_cluster]
                                # elif (rec_distances_1[i_cluster]-th_1[i_cluster])<0 and (y_som_1<0):             
                                #     th_1[i_cluster] +=  feedback_sep*th_1[i_cluster]                                    
            
                        #Layer 0
                        #weights
                        elem_distances_0 = (sentence_surfs[ts_i,:,:,None]-weights_0[:,:,:])
                        rec_closest_0_one_hot = np.zeros([n_clusters_0])
                        rec_closest_0_one_hot[rec_closest_0]=1
                        
                        # Keep only the distances for winners
                        elem_distances_0=elem_distances_0[:,:,:]*rec_closest_0_one_hot[None,None,:]
                        if slow_learning: 
                            weights_0[:,:,:] += (lrate_0*(w_dt_y_som_0*elem_distances_0[:]) + ratio_sds*lrate_0*(w_y_som_0*elem_distances_0[:]))
                        else: 
                            weights_0[:,:,:] += lrate_0_pre*(w_dt_y_som_0*elem_distances_0[:] + 0*ratio_sds*(w_y_som_0*elem_distances_0[:]))
                        
                        if slow_learning:                            
                            # treshold
                            for i_cluster in range(n_clusters_0):# THIS WORKS
                                th_tau = th_tau_rel*np.mean(th_0[i_cluster])
                                # th_tau = 0.5
                                if i_cluster==rec_closest_0:   
                                    th_dist = th_0[rec_closest_0]-rec_distances_0[rec_closest_0]
                                    # th_update = np.exp(-th_dist/th_tau)*th_dist
                                    th_update = np.exp(-th_dist/th_tau)*th_0[rec_closest_0]
                                    th_0[rec_closest_0] += lrate_th_0*w_dt_y_som_0*th_update + ratio_sds*lrate_th_0*w_y_som_0*th_update
                                    
                                elif ((rec_distances_0[i_cluster]-th_0[i_cluster])<0)  and (w_y_som_0>=0) and (w_dt_y_som_0>=0):
                                # else:
                                    th_dist = th_0[i_cluster]-rec_distances_0[i_cluster]
                                    # th_update = np.exp(-th_dist/th_tau)*th_dist
                                    th_update = np.exp(-th_dist/th_tau)*th_0[i_cluster]
                                    th_0[i_cluster] -= lrate_th_0*w_dt_y_som_0*th_update + ratio_sds*lrate_th_0*w_y_som_0*th_update
 
                            # #treshold
                            # for i_cluster in range(n_clusters_0):
                            #     th_tau = th_tau_rel*np.abs((th_0[i_cluster]-rec_distances_0[i_cluster]))
                            #     # th_tau = 0.5
                            #     if i_cluster==rec_closest_0:                                
                            #         th_0[rec_closest_0] += lrate_th_0*dt_y_som_0*np.exp(-np.abs((rec_distances_0[rec_closest_0]-th_0[rec_closest_0]))/th_tau) + 0.01*lrate_th_0*y_som_0*np.exp(-np.abs((rec_distances_0[rec_closest_0]-th_0[rec_closest_0]))/th_tau)
                            #     elif ((rec_distances_0[i_cluster]-th_0[i_cluster])<0) and (y_som_0>=0) and (dt_y_som_0>=0):
                            #     # else:
                            #         th_0[i_cluster] -= 0.01*lrate_th_0*th_0[i_cluster]
   
        

                        if not slow_learning:                            
                            #fast_threshold_control
                            for i_cluster in range(n_clusters_0):
                                if (i_cluster==rec_closest_0) and (w_y_som_0<0):
                                    th_0[i_cluster] -=  feedback_sep*th_0[i_cluster]
                                # elif (rec_distances_0[i_cluster]-th_0[i_cluster])<0 and (y_som_0<0):             
                                #     th_0[i_cluster] +=  feedback_sep*th_0[i_cluster]

                        #treshold_coeff
                        # for i_cluster in range(n_clusters_0):
                        #     if (i_cluster==rec_closest_0) and (y_som_0<0):
                        #         th_0[i_cluster] +=  0.01*y_som_0*lrate_th_0*(th_0[i_cluster]-rec_distances_0[i_cluster])
                        #     elif (rec_distances_0[i_cluster]-th_0[i_cluster])<0 and (y_som_0>0):                 
                        #         th_0[i_cluster] -= 0.01*y_som_0*lrate_th_0*(th_0[i_cluster]-rec_distances_0[i_cluster])      
            
                        # th_0[rec_closest_0] += (np.abs(y_som_0)-1)*lrate_th_0*(th_0[rec_closest_0]-rec_distances_0[rec_closest_0])
                                                  

                        ## PROGRESS UPDATE
                        rec_closest_2_one_hot = np.zeros([n_sentences])
                        rec_closest_2_one_hot[rec_closest_2]=1
                        class_rate=np.sum(rec_closest_2_one_hot,axis=0)
                            
                        computed_events += 1
                        if rec_closest_2==label:
                            result = "Correct"
                            event_accuracy += 1
        
                        else:
                            result = "Wrong"
                        
                        progress = computed_events/n_events
                        rel_accuracy = event_accuracy/computed_events
                        print("Epoch "+str(epoch)+", Sentence "+str(sentence_i)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy))
                        print("Prediction: "+result+str(label))
                        
                        if print_lay==1:
                            #Layer0
                            print("Y-som: "+str(y_som_0)+" dt Y-som: "+str(dt_y_som_0)+" Closest_center: "+str(rec_closest_0))
                            print(rec_distances_0-th_0)
                            print(th_0)
                        elif print_lay==2:
                            #Layer1
                            print("Y-som: "+str(y_som_1)+" dt Y-som: "+str(dt_y_som_1)+" Closest_center: "+str(rec_closest_1))
                            print(rec_distances_1-th_1)
                            print(th_1)

                        
                
                if pause_pressed == True:    
                    if n_clusters_0>1:
                        for feat in range(n_clusters_0):
                            axs[feat].imshow(weights_0[:,:,feat])
                            plt.draw()
                    elif n_clusters_0==1:
                        axs.imshow(weights_0[:,:,feat])
                        plt.draw()
                    plt.pause(5)
                    pause_pressed=False
            
            
            sentence_accuracy[sentence_i]=rel_accuracy
            
            if sentence_i>3:
                increase_max = 0.05
                accuracy_increase_3 = np.abs((sentence_accuracy[sentence_i] - sentence_accuracy[sentence_i-1])) < increase_max
                accuracy_increase_2 = np.abs((sentence_accuracy[sentence_i-1] - sentence_accuracy[sentence_i-2])) < increase_max
                accuracy_increase_1 = np.abs((sentence_accuracy[sentence_i-2] - sentence_accuracy[sentence_i-3])) < increase_max
                if accuracy_increase_3 and accuracy_increase_2 and accuracy_increase_1 and not slow_learning :
                    slow_learning = True
                    th_0 = np.zeros(n_clusters_0)+max_th_0*1.5
                    th_1 = np.zeros(n_clusters_1)+max_th_1*1.5
                        
                
                   

                        
    # end = time.time()

    listener.join()
 

    
 
fb_selected_weights_0 = weights_0
fb_selected_weights_1 = weights_1



#%% Plot feedback centroids layer 0

#plot centroids
fig, axs = plt.subplots(n_clusters_0)
for pol_i in range(n_clusters_0):
    if n_clusters_0>1:
        # axs[pol_i].imshow(np.reshape(weights_0[:,:,pol_i], [5,5]))
        axs[pol_i].imshow(np.reshape(kmeans_weights_0[:,:,pol_i], [5,5]))

    elif n_clusters_0==1:
        # axs.imshow(np.reshape(weights_0[:,:,pol_i], [5,5]))
        axs.imshow(np.reshape(kmeans_weights_0[:,:,pol_i], [5,5]))



#%% Save weights kmeans weights and thresholds:
 
save_folder = "Results/Synth/"

np.save(save_folder+"centroids_0.npy",weights_0)
np.save(save_folder+"centroids_1.npy",weights_1)
np.save(save_folder+"centroids_2.npy",weights_2)

np.save(save_folder+"th_0.npy",th_0)
np.save(save_folder+"th_1.npy",th_1)

np.save(save_folder+"start_centroids_0.npy",kmeans_weights_0)
np.save(save_folder+"start_centroids_1.npy",kmeans_weights_1)
np.save(save_folder+"start_centroids_2.npy",label_mean_weights)


#%% Load weights kmeans weights and thresholds:
 
save_folder = "Results/Synth/"

weights_0 = np.load(save_folder+"centroids_0.npy")
weights_1 = np.load(save_folder+"centroids_1.npy")
weights_2 = np.load(save_folder+"centroids_2.npy")

th_0 = np.load(save_folder+"th_0.npy")
th_1 = np.load(save_folder+"th_1.npy")

kmeans_weights_0 = np.load(save_folder+"start_centroids_0.npy")
kmeans_weights_1 = np.load(save_folder+"start_centroids_1.npy")
label_mean_weights = np.load(save_folder+"start_centroids_2.npy")

# ##KMEAnS net parameters
# weights_0=kmeans_weights_0
# weights_1=kmeans_weights_1
# weights_2=label_mean_weights

# th_0 = np.zeros(n_clusters_0)+100
# th_1 = np.zeros(n_clusters_1)+100

#%% Testing Network 


time_context_1 = np.zeros([n_clusters_0, n_pol],dtype=int)

time_context_2 = np.zeros([n_clusters_1, n_words],dtype=int)

tau_1 = 1
tau_2 = 1


sentence_accuracy = np.zeros(len(data_surf))
events_processed = np.zeros(len(data_surf))

for sentence_i in range(len(data_surf)):
    n_events = len(data_surf[sentence_i])
    sentence_surfs = data_surf[sentence_i]
    computed_events = 0
    event_accuracy = 0

    #event mask used to avoid exponential decay calculation for pixel 
    # that did not generate an event yet
    mask_start_1 = np.zeros([n_clusters_0, n_pol],dtype=int)
    
    mask_start_2 = np.zeros([n_clusters_1, n_words],dtype=int)
    
    
        
    for ts_i in range(n_events):
        

        label = data_labels[sentence_i]
        ref_ch_pos = data_events[sentence_i][2][ts_i]
        ref_word_pos = ref_ch_pos//word_length
        ref_ts =  data_events[sentence_i][3][ts_i]
        
        rec_distances_0=np.sum((sentence_surfs[ts_i,:,:,None]-weights_0[:,:,:])**2,axis=(0,1))

        
        # Closest center with threshold computation
        rec_closest_0=np.argmin(rec_distances_0-th_0,axis=0)
        # rec_closest_0=np.argmin(rec_distances_0,axis=0)
        

        
        # Layer 1 check
        if (rec_distances_0[rec_closest_0]-th_0[rec_closest_0])<0:
               
            time_context_1[rec_closest_0,ref_ch_pos] = ref_ts
            mask_start_1[rec_closest_0,ref_ch_pos]=1
            
            # Extracting the single word ts
            beg_ch_index = ref_word_pos*word_length
            end_ch_index = (ref_word_pos+1)*word_length
            ts_lay_1 = np.exp((time_context_1[:,beg_ch_index:end_ch_index]\
                               -ref_ts)*mask_start_1[:,beg_ch_index:end_ch_index]/\
                                  tau_1)*mask_start_1[:,beg_ch_index:end_ch_index]    

            
            rec_distances_1=np.sum((ts_lay_1[:,:,None]-weights_1[:,:,:])**2,axis=(0,1))

            
            rec_closest_1=np.argmin(rec_distances_1-th_1,axis=0)
            # rec_closest_1=np.argmin(rec_distances_1,axis=0)

            
            # Layer 2 check
            if (rec_distances_1[rec_closest_1]-th_1[rec_closest_1])<0:

                time_context_2[rec_closest_1,ref_word_pos] = ref_ts
                mask_start_2[rec_closest_1,ref_word_pos]=1
                
                ts_lay_2 = np.exp((time_context_2-ref_ts)*mask_start_2/\
                                      tau_2)*mask_start_2                                 
                
                rec_distances_2=np.sum((ts_lay_2[:,:,None]-weights_2[:,:,:])**2,axis=(0,1))

                
                rec_closest_2=np.argmin(rec_distances_2,axis=0)
                
                
 
                ## PROGRESS UPDATE
                rec_closest_2_one_hot = np.zeros([n_sentences])
                rec_closest_2_one_hot[rec_closest_2]=1
                class_rate=np.sum(rec_closest_2_one_hot,axis=0)
                    
                computed_events += 1
                if rec_closest_2==label:
                    result = "Correct"
                    event_accuracy += 1

                else:
                    result = "Wrong"
                
                progress = computed_events/n_events
                rel_accuracy = event_accuracy/computed_events
                print("Sentence "+str(sentence_i)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy))
                print("Prediction: "+result+str(label))

                
        

    
    sentence_accuracy[sentence_i]=rel_accuracy
    events_processed[sentence_i] = progress


# new_rule_sentence_accuracy = sentence_accuracy
# new_rule_events_processed = events_processed

k_means_sentence_accuracy = sentence_accuracy
k_means_new_rule_events_processed = events_processed




#%% Save results

save_folder = "Results/Synth/"


res = {"New Rule Acc" : new_rule_sentence_accuracy,
       "New Rule Evs" : new_rule_events_processed,
       "Kmeans Acc" : k_means_sentence_accuracy,
       "Kmeans Evs" : k_means_new_rule_events_processed}

np.save(save_folder+"results.npy", res)

#%% Load results

save_folder = "Results/Synth/"

res = np.load(save_folder+"results.npy", allow_pickle=True)

#%% Plot results

plt.figure()

plt.plot(res.any()["New Rule Acc"])
plt.plot(res.any()["New Rule Evs"])
plt.plot(res.any()["Kmeans Acc"])
