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
        

events = word_generator(word = "v/v", sim_time=simulation_time,
                        high_f = high_freq, low_f = low_freq)

#%% Generate the timesurfaces