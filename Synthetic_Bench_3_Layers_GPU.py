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
import pyopencl as cl

mf = cl.mem_flags


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





#%% Generate the timesurfaces_GPU

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

batch_size = 1000

res_x = 5
res_y = 5
tau = 5000
n_pol = 6

global_space=np.array([batch_size,res_x,res_y])
local_space=np.array([1,res_x,res_y])


n_max_events = max([len(data_events[i][0]) for i in range(batch_size)])

xs_np = np.nan*np.zeros((batch_size,n_max_events),dtype=np.int32)
ys_np = np.nan*np.zeros((batch_size,n_max_events),dtype=np.int32)
ps_np = np.nan*np.zeros((batch_size,n_max_events),dtype=np.int32)
ts_np = np.nan*np.zeros((batch_size,n_max_events),dtype=np.int32)
TS_np = np.nan*np.zeros((batch_size,n_max_events,res_x,res_y),dtype=np.float32)
# TS_np = np.nan*np.zeros((batch_size,n_max_events),dtype=np.float32)


for i in range(batch_size):
    n_events = len(data_events[i][0])
    xs_np[i,:n_events] = data_events[i][0]
    ys_np[i,:n_events] = data_events[i][1]
    ps_np[i,:n_events] = data_events[i][2]
    ts_np[i,:n_events] = data_events[i][3]

xs_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xs_np)
ys_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ys_np)
ps_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ps_np)
ts_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts_np)
res_x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(res_x))
res_y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(res_y))
tau_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(tau))
n_pol_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_pol))
n_max_events_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_max_events))

TS_b = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=TS_np)

f = open('all_surfaces.cl', 'r')
fstr = "".join(f.readlines())
# print(fstr)
program=cl.Program(ctx, fstr).build()

kernel=program.all_surfaces(queue, global_space, local_space, xs_b, ys_b, ps_b,
                     ts_b, res_x_b, res_y_b, tau_b, n_pol_b, TS_b, n_max_events_b)

cl.enqueue_copy(queue, TS_np, TS_b).wait()



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


