#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 08:37:36 2022

@author: marcorax93
"""


import numpy as np
from scipy import io
import random 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import expit
from pynput import keyboard
import pyopencl as cl
import time

from Libs.HOTSLib import fb_surfaces, all_surfaces, n_mnist_rearranging,\
                            dataset_resize

from Libs.Dense_Layer.Dense_Layer import Dense_Layer
from Libs.Class_Layer.Class_Layer import Class_Layer
from Libs.Conv_Layer.Conv_Layer import Conv_Layer



#%% Data loading and parameters setting
            
## Data loading
train_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/train_set.mat')['train_set'])
test_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/test_set.mat')['test_set'])
n_recording_labels_train=[len(train_set_orig[label]) for label in range(len(train_set_orig))]
n_recording_labels_test=[len(test_set_orig[label]) for label in range(len(train_set_orig))]

# using a subset of N-MNIST to lower memory usage
num_labels = len(test_set_orig)

# N-MNIST resolution 
res_x = 28
res_y = 28

train_set_orig = dataset_resize(train_set_orig,res_x,res_y)
test_set_orig = dataset_resize(test_set_orig,res_x,res_y)


#Create an index to scramble the labels
train_labels = np.concatenate([label*np.ones(len(train_set_orig[label]),\
                               dtype=int) for label in range(num_labels)])

train_rec_idx = np.concatenate([np.arange(len(train_set_orig[label]),\
                                dtype=int) for label in range(num_labels)])

temp = list(zip(train_labels, train_rec_idx))
random.shuffle(temp)
train_labels, train_rec_idx = zip(*temp)
    
test_labels = np.concatenate([label*np.ones(len(test_set_orig[label]),\
                               dtype=int) for label in range(num_labels)])

test_rec_idx = np.concatenate([np.arange(len(test_set_orig[label]),\
                           dtype=int) for label in range(num_labels)])
    
temp = list(zip(test_labels, test_rec_idx))
random.shuffle(temp)
test_labels, test_rec_idx = zip(*temp)    

    
#%% GPU Initialization
mf = cl.mem_flags
platforms = cl.get_platforms()
platform_i = 0 #Select the platform manually here
devices = platforms[platform_i].get_devices(device_type=cl.device_type.GPU)
print("Max work group size: ",devices[0].max_work_group_size)
ctx = cl.Context(devices=devices)#TODO Check how the context apply to more than one GPU
queue = cl.CommandQueue(ctx)
    
#%% Create the network

# Parameters
batch_size = 32 #too high and it might affect how fast it converges 128 it halts progression
n_labels = 10
n_epochs = np.int32(np.floor(len(train_labels)/batch_size))#I will lose some results


#Dense Layer 1 data and parameters
n_pol_0 = 1
tau_0 = 1e5
n_clusters_0 = 32
# n_clusters_0 = 1
# lrate_0 = 1e-2
# th_lrate_0 = 1e-1
lrate_0 = 5e-4
th_lrate_0 = 5e-3#Check if smaller can help differentiate more clusters
s_gain = 1e-1

th_decay_0=0.5
th_size_0=300
res_x_0 = 28
res_y_0 = 28

Dense0 = Dense_Layer(n_clusters_0, tau_0, res_x_0, res_y_0, n_pol_0, lrate_0,
                     th_size_0, th_lrate_0, th_decay_0, ctx, batch_size,
                     s_gain, debug=True)

#Class Layer 1 data and parameters
tau_1 = 1e4#1e3 actually gave some nice features
tau_1_fb = 1e1
n_clusters_1=10
lrate_1 = 5e-4#If too high  you end up  with a single cluster same as lrate0 works well
res_x_1 = 1
res_y_1 = 1

Class1 = Class_Layer(n_clusters_1, tau_1, res_x_1, res_y_1, n_clusters_0, lrate_1,
                     ctx, batch_size, s_gain, fb_signal=True, fb_tau=tau_1_fb,
                     debug=True)



Dense0.buffers["input_S_bf"] = Class1.buffers["output_S_bf"]
Dense0.buffers["input_dS_bf"] = Class1.buffers["output_dS_bf"]

Dense0.variables["input_S"] = Class1.variables["output_S"]
Dense0.variables["input_dS"] = Class1.variables["output_dS"]

fevskip = np.zeros(batch_size, dtype=np.int32)
bevskip = np.zeros(batch_size, dtype=np.int32)

# fevskip for feed event skip, and bevskip for back event skip, 1=>true 0=>false
fevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip)
bevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip)

#%% Initialize clusters

rec = 0
epoch_i=0
n_batches = 60000//batch_size
# n_batches = 1
batch_i = 0
ev_i = 0

second_fase_i_epoch = 2
# second_fase_i_epoch = 0



f = open('Libs/cl_kernels/next_ev.cl', 'r')
fstr = "".join(f.readlines())
program=cl.Program(ctx, fstr).build(options='-cl-std=CL2.0')

n_epochs=2
for epoch_i in range(n_epochs):
    rec=0
    for batch_i in range(n_batches):     
        n_events_rec=np.zeros(batch_size, dtype=int)
        for i in range(batch_size):
            data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
            n_events_rec[i] = len(data_events[0])
            
        n_max_events = max(n_events_rec)
        
        xs_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ys_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ps_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ts_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        train_batch_labels = np.zeros([batch_size],dtype=np.int32)
        n_events_batch = np.zeros([batch_size],dtype=np.int32)
        
        
        for i in range(batch_size):
            data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
            n_events = len(data_events[0])
            xs_np[i,:n_events] = data_events[0]
            ys_np[i,:n_events] = data_events[1]
            ps_np[i,:n_events] = data_events[2]*0 #removing pol information at layer 1
            ts_np[i,:n_events] = data_events[3]
            train_batch_labels[i] = train_labels[rec+i]
            n_events_batch[i] = n_events
            
        rec+=batch_size 
        processed_ev = np.zeros([batch_size],dtype=np.int32)
        correct_ev = np.zeros([batch_size],dtype=np.int32)
        
        # Network Buffers
        xs_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=xs_np)
        ys_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ys_np)
        ps_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ps_np)
        ts_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts_np)
        ev_i_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.int32(0))
        n_events_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_max_events))
        processed_ev_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=processed_ev)
        correct_ev_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=correct_ev)
        batch_labels_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_batch_labels)
        fevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip)
        bevskip_bf = cl.Buffer(ctxs, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip)
        
        net_buffers = {"xs_bf" : xs_bf, "ys_bf" : ys_bf, "ps_bf" : ps_bf, 
                       "ts_bf" : ts_bf, "ev_i_bf" : ev_i_bf, 
                       "n_events_bf" : n_events_bf, 
                       "processed_ev_bf" : processed_ev_bf, 
                       "correct_ev_bf" : correct_ev_bf, 
                       "batch_labels_bf" : batch_labels_bf, 
                       "n_labels_bf" : Class1.buffers["n_clusters_bf"],
                       "fevskip_bf" : fevskip_bf, "bevskip_bf" : bevskip_bf}
        
        start_exec = time.time()
        for ev_i in range(n_max_events):
            
            Dense0.infer(net_buffers, queue)
            Class1.infer(net_buffers, queue)
            Class1.learn(net_buffers, queue)
            
            if epoch_i<second_fase_i_epoch:
                Dense0.init_learn(net_buffers, queue)
            else:
                Dense0.learn(net_buffers, queue)                                      
                    
            kernel=program.next_ev(queue, np.array([batch_size]), None, ev_i_bf)
        
        end_exec = time.time()
        
        Dense0.batch_update(queue)
        Class1.batch_update(queue)
        
        processed_ev = Class1.variables["processed_ev"]
        correct_ev = Class1.variables["correct_ev"]
        
        avg_processed_ev = np.mean(processed_ev / n_events_batch)
        avg_accuracy = np.mean(correct_ev / processed_ev)
        
        print("TRAIN")
        print("Epoch: "+str(epoch_i)+" of "+str(n_epochs))
        print("Batch: "+str(batch_i)+" of "+str(n_batches))
        print("Processed rec "+str(rec)+" of "+str(len(train_labels))+" Label: "+str(np.unique(train_batch_labels[np.where(correct_ev!=0)])))
        print("Elapsed time is ", (end_exec-start_exec) * 10**3, "ms")
        print("Accuracy is "+str(avg_accuracy)+" of "+str(avg_processed_ev)+" processed events")
        
        Dense0.batch_flush(queue)
        Class1.batch_flush(queue)
        
        if epoch_i<second_fase_i_epoch:
            Dense0.variables["thresholds"][:]=th_size_0
            thresholds=Dense0.variables["thresholds"]
            thresholds_bf=Dense0.buffers["thresholds_bf"]
            cl.enqueue_copy(queue, thresholds_bf, thresholds).wait()
        

#%% copy weights and prepare for phase 2 
    
# Dense0_old_weights = Dense0.variables["centroids"].copy()
# Class1_old_weights = Class1.variables["centroids"].copy()

               
Dense0.variables["thresholds"][:]=230
thresholds=Dense0.variables["thresholds"]
thresholds_bf=Dense0.buffers["thresholds_bf"]
cl.enqueue_copy(queue, thresholds_bf, thresholds).wait()


Dense0.variables["centroids"] = Dense0_old_weights.copy() 
Class1.variables["centroids"] = Class1_old_weights.copy() 

centroids_0_bf=Dense0.buffers["centroids_bf"]
centroids_1_bf=Class1.buffers["centroids_bf"]
cl.enqueue_copy(queue, centroids_0_bf, Dense0_old_weights).wait()
cl.enqueue_copy(queue, centroids_1_bf, Class1_old_weights).wait()

th_lrate_0 = 1e-5
lrate_0 = 1e-4
lrate_1 = 1e-4


Dense0.parameters["th_lrate"]=th_lrate_0
th_lrate_bf=Dense0.buffers["th_lrate_bf"]
cl.enqueue_copy(queue, th_lrate_bf, np.float32(th_lrate_0)).wait()
    
Dense0.parameters["lrate"]=lrate_0
lrate_0_bf=Dense0.buffers["lrate_bf"]
cl.enqueue_copy(queue, lrate_0_bf, np.float32(lrate_0)).wait()

Class1.parameters["lrate"]=lrate_1
lrate_1_bf=Dense0.buffers["lrate_bf"]
cl.enqueue_copy(queue, lrate_1_bf, np.float32(lrate_1)).wait()

s_gain=1e-1 # if s rate is too high, thresholds will grow to infinity

Dense0.parameters["s_gain"]=s_gain
s_gain_bf=Dense0.buffers["s_gain_bf"]
cl.enqueue_copy(queue, s_gain_bf, np.float32(s_gain)).wait()

Class1.parameters["s_gain"]=s_gain
s_gain_bf=Class1.buffers["s_gain_bf"]
cl.enqueue_copy(queue, s_gain_bf, np.float32(s_gain)).wait()


#%% Print
centroids0 = Dense0.variables["centroids"]
centroids1 = Class1.variables["centroids"]

# centroids0 = Dense0.variables["dcentroids"]
# centroids1 = Class1.variables["dcentroids"]

for i in range(n_clusters_0):
    plt.figure()
    plt.title("cluster: "+str(i))
    plt.imshow(centroids0[0,i,:,:,0].transpose())
    
for i in range(n_clusters_1):
    plt.figure()
    plt.title("cluster: "+str(i))
    plt.imshow(centroids1[:,i,0,0,:].transpose())
plt.ylabel("Layer0 cluster#")  

#%% COntrol variables
dcentroids=Dense0.variables["dcentroids"]
dcentroids_bf=Dense0.buffers["dcentroids_bf"]
cl.enqueue_copy(queue, dcentroids, dcentroids_bf).wait()
Dense0.variables["dcentroids"]=dcentroids

distances=Dense0.variables["distances"]
distances_bf=Dense0.buffers["distances_bf"]
cl.enqueue_copy(queue, distances, distances_bf).wait()
Dense0.variables["dcentroids"]=distances

dcentroids=Class1.variables["dcentroids"]
dcentroids_bf=Class1.buffers["dcentroids_bf"]
cl.enqueue_copy(queue, dcentroids, dcentroids_bf).wait()
Class1.variables["dcentroids"]=dcentroids


S0=Dense0.variables["input_S"]
dS0=Dense0.variables["input_dS"]

S0_bf=Dense0.buffers["input_S_bf"]
dS0_bf=Dense0.buffers["input_dS_bf"]

cl.enqueue_copy(queue, S0, S0_bf).wait()
cl.enqueue_copy(queue, dS0, dS0_bf).wait()


S1=Class1.variables["output_S"]
dS1=Class1.variables["output_dS"]

S1_bf=Class1.buffers["output_S_bf"]
dS1_bf=Class1.buffers["output_dS_bf"]

cl.enqueue_copy(queue, S1, S1_bf).wait()
cl.enqueue_copy(queue, dS1, dS1_bf).wait()


#%% Save weights TODO add the ability to salve parameters and number of epochs run

file="Results/NMNIST_dense_results/Dense0_centroids"
np.save(file, Dense0.variables["centroids"])

file="Results/NMNIST_dense_results/Class1_centroids"
np.save(file, Class1.variables["centroids"])

#%% TEST

rec = 0
epoch_i=0
n_batches = 60000//batch_size
# n_batches = 1
batch_i = 0
ev_i = 0

f = open('Libs/cl_kernels/next_ev.cl', 'r')
fstr = "".join(f.readlines())
program=cl.Program(ctx, fstr).build(options='-cl-std=CL2.0')


n_epochs=20
for epoch_i in range(n_epochs):
    rec=0
    for batch_i in range(n_batches):    
        
        n_events_rec=np.zeros(batch_size, dtype=int)
        for i in range(batch_size):
            data_events = test_set_orig[test_labels[rec+i]][test_rec_idx[rec+i]]
            n_events_rec[i] = len(data_events[0])
            
        n_max_events = max(n_events_rec)
    
        xs_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ys_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ps_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ts_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        test_batch_labels = np.zeros([batch_size],dtype=np.int32)
        n_events_batch = np.zeros([batch_size],dtype=np.int32)
    
    
        for i in range(batch_size):
            data_events = test_set_orig[test_labels[rec+i]][test_rec_idx[rec+i]]
            n_events = len(data_events[0])
            xs_np[i,:n_events] = data_events[0]
            ys_np[i,:n_events] = data_events[1]
            ps_np[i,:n_events] = data_events[2]*0 #removing pol information at layer 1
            ts_np[i,:n_events] = data_events[3]
            test_batch_labels[i] = test_labels[rec+i]
            n_events_batch[i] = n_events
            
        rec+=batch_size 
        processed_ev = np.zeros([batch_size],dtype=np.int32)
        correct_ev = np.zeros([batch_size],dtype=np.int32)
        
        # Network Buffers
        xs_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=xs_np)
        ys_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ys_np)
        ps_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ps_np)
        ts_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts_np)
        ev_i_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.int32(0))
        n_events_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_max_events))
        processed_ev_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=processed_ev)
        correct_ev_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=correct_ev)
        batch_labels_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_batch_labels)
        fevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip)
        bevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip)
        
        net_buffers = {"xs_bf" : xs_bf, "ys_bf" : ys_bf, "ps_bf" : ps_bf, 
                       "ts_bf" : ts_bf, "ev_i_bf" : ev_i_bf, 
                       "n_events_bf" : n_events_bf, 
                       "processed_ev_bf" : processed_ev_bf, 
                       "correct_ev_bf" : correct_ev_bf, 
                       "batch_labels_bf" : batch_labels_bf, 
                       "fevskip_bf" : fevskip_bf, "bevskip_bf" : bevskip_bf}
        
        start_exec = time.time()
        for ev_i in range(n_max_events):
            
            Dense0.infer(net_buffers, queue)
            Class1.infer(net_buffers, queue)
            kernel=program.next_ev(queue, np.array([batch_size]), None, ev_i_bf)
        
        end_exec = time.time()
        
        Dense0.batch_update(queue)
        Class1.batch_update(queue)
        
        processed_ev = Class1.variables["processed_ev"]
        correct_ev = Class1.variables["correct_ev"]
        
        avg_processed_ev = np.mean(processed_ev / n_events_batch)
        avg_accuracy = np.mean(correct_ev / processed_ev)
        
        print("TEST")
        print("Epoch: "+str(epoch_i)+" of "+str(n_epochs))
        print("Batch: "+str(batch_i)+" of "+str(n_batches))
        print("Processed rec "+str(rec)+" of "+str(len(train_labels))+" Label: "+str(np.unique(train_batch_labels[np.where(correct_ev!=0)])))
        print("Elapsed time is ", (end_exec-start_exec) * 10**3, "ms")
        print("Accuracy is "+str(avg_accuracy)+" of "+str(avg_processed_ev)+" processed events")
        
        Dense0.batch_flush(queue)
        Class1.batch_flush(queue)
        
#%% Create the network three layers

# Parameters
# batch_size = 32 #too high and it might affect how fast it converges 128 it halts progression
batch_size = 64 #too high and it might affect how fast it converges 128 it halts progression

n_labels = 10
n_epochs = np.int32(np.floor(len(train_labels)/batch_size))#I will lose some results


#Dense Layer 1 data and parameters
n_pol_0 = 1
tau_0 = 1e4
n_clusters_0 = 32
# n_clusters_0 = 1
# lrate_0 = 1e-2
# th_lrate_0 = 1e-1
lrate_0 = 1e-1
# th_lrate_0 = 5e-2#Check if smaller can help differentiate more clusters
th_lrate_0 = 1e-1#Check if smaller can help differentiate more clusters
# s_gain = 1e-1
s_gain = 1e-1

win_l_0 = 9

th_decay_0=0.1
th_size_0=250
# th_size_0=30
res_x_0 = 28
res_y_0 = 28

Conv0 = Conv_Layer(n_clusters_0, tau_0, res_x_0, res_y_0, win_l_0, n_pol_0, lrate_0,
                     th_size_0, th_lrate_0, th_decay_0, ctx, batch_size,
                     s_gain, debug=True)

#Dense Layer 2 data and parameters
tau_1 = 1e1
tau_1_fb = 1e1
# tau_1_fb = 1e4
n_clusters_1 = 16
# n_clusters_1 = 1
# lrate_1 = 1e-2
# th_lrate_1 = 1e-1
lrate_1 = 1e-1#it was -3
th_lrate_1 = 1e-1#Check if smaller can help differentiate more clusters
s_gain = 1e-1

th_decay_1=0.1
th_size_1=40000
# th_size_1=300
res_x_1 = 28
res_y_1 = 28

Dense1 = Dense_Layer(n_clusters_1, tau_1, res_x_1, res_y_1, n_clusters_0, lrate_1,
                     th_size_1, th_lrate_1, th_decay_1, ctx, batch_size,
                     s_gain, fb_signal=True, fb_tau=tau_1_fb, debug=True)

#Class Layer 3 data and parameters
tau_2 = 1e1#1e3 actually gave some nice features
tau_2_fb = 1e1
n_clusters_2=10
lrate_2 = 1e-4#If too high  you end up  with a single cluster same as lrate0 works well
res_x_2 = 1
res_y_2 = 1

Class2 = Class_Layer(n_clusters_2, tau_2, res_x_2, res_y_2, n_clusters_1, lrate_2,
                     ctx, batch_size, s_gain, fb_signal=True, fb_tau=tau_2_fb,
                     debug=True)


Conv0.buffers["input_S_bf"] = Dense1.buffers["output_S_bf"]
Conv0.buffers["input_dS_bf"] = Dense1.buffers["output_dS_bf"]
Dense1.buffers["input_S_bf"] = Class2.buffers["output_S_bf"]
Dense1.buffers["input_dS_bf"] = Class2.buffers["output_dS_bf"]
Dense1.buffers["correct_response_bf"] = Class2.buffers["correct_response_bf"]

Conv0.variables["input_S"] = Dense1.variables["output_S"]
Conv0.variables["input_dS"] = Dense1.variables["output_dS"]
Dense1.variables["input_S"] = Class2.variables["output_S"]
Dense1.variables["input_dS"] = Class2.variables["output_dS"]
Dense1.variables["correct_response"] = Class2.variables["correct_response"]

fevskip = np.zeros(batch_size, dtype=np.int32)
bevskip = np.zeros(batch_size, dtype=np.int32)

# fevskip for feed event skip, and bevskip for back event skip, 1=>true 0=>false
fevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip)
bevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip)

#%% Initialize clusters

rec = 0
epoch_i=0
n_batches = 60000//batch_size
# n_batches = 1
batch_i = 0
ev_i = 0
second_fase_i_epoch = 1
# second_fase_i_epoch = 0



f = open('Libs/cl_kernels/next_ev.cl', 'r')
fstr = "".join(f.readlines())
program=cl.Program(ctx, fstr).build(options='-cl-std=CL2.0')

n_epochs=1
for epoch_i in range(n_epochs):
    rec=0
    for batch_i in range(n_batches):     
    # for batch_i in range(2):     
        n_events_rec=np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
            n_events_rec[i] = len(data_events[0])
            
        n_max_events = max(n_events_rec)
        
        S0_rec = np.zeros([batch_size, n_max_events,28,28,n_clusters_0])
        dS0_rec = np.zeros([batch_size, n_max_events,28,28,n_clusters_0])
        S1_rec = np.zeros([batch_size, n_max_events, n_clusters_1])
        dS1_rec = np.zeros([batch_size, n_max_events,n_clusters_1])
        
        xs_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ys_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ps_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ts_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        train_batch_labels = np.zeros([batch_size],dtype=np.int32)
        n_events_batch = np.zeros([batch_size],dtype=np.int32)
        
        
        for i in range(batch_size):
            data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
            n_events = len(data_events[0])
            xs_np[i,:n_events] = data_events[0]
            ys_np[i,:n_events] = data_events[1]
            ps_np[i,:n_events] = data_events[2]*0 #removing pol information at layer 1
            ts_np[i,:n_events] = data_events[3]
            train_batch_labels[i] = train_labels[rec+i]
            n_events_batch[i] = n_events
            
        rec+=batch_size 
        processed_ev = np.zeros([batch_size],dtype=np.int32)
        correct_ev = np.zeros([batch_size],dtype=np.int32)
        
        # Network Buffers
        xs_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=xs_np)
        ys_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ys_np)
        ps_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ps_np)
        ts_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts_np)
        ev_i_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.int32(0))
        n_events_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_max_events))
        processed_ev_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=processed_ev)
        correct_ev_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=correct_ev)
        batch_labels_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_batch_labels)
        fevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip)
        bevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip)
        
        net_buffers = {"xs_bf" : xs_bf, "ys_bf" : ys_bf, "ps_bf" : ps_bf, 
                       "ts_bf" : ts_bf, "ev_i_bf" : ev_i_bf, 
                       "n_events_bf" : n_events_bf, 
                       "processed_ev_bf" : processed_ev_bf, 
                       "correct_ev_bf" : correct_ev_bf, 
                       "batch_labels_bf" : batch_labels_bf, 
                       "n_labels_bf" : Class2.buffers["n_clusters_bf"],
                       "fevskip_bf" : fevskip_bf, "bevskip_bf" : bevskip_bf}
        
        start_exec = time.time()
        for ev_i in range(n_max_events):
            
            Conv0.infer(net_buffers, queue)
            Dense1.infer(net_buffers, queue)
            Class2.infer(net_buffers, queue)
            
            if (batch_i%5):
                Class2.learn(net_buffers, queue)
                
                if epoch_i<second_fase_i_epoch:
                    Dense1.init_learn(net_buffers, queue)
                    Conv0.init_learn(net_buffers, queue)
    
                else:
                    Dense1.learn(net_buffers, queue)      
                    Conv0.learn(net_buffers, queue)
                                    
            
            # S0=Dense1.variables["output_S"]
            # dS0=Dense1.variables["output_dS"]

            # S0_bf=Dense1.buffers["output_S_bf"]
            # dS0_bf=Dense1.buffers["output_dS_bf"]

            # cl.enqueue_copy(queue, S0, S0_bf).wait()
            # cl.enqueue_copy(queue, dS0, dS0_bf).wait()
            
            # closest0=Conv0.variables["closest_c"]
            # closest_bf=Conv0.buffers["closest_c_bf"]
            # cl.enqueue_copy(queue, closest0, closest_bf).wait()
            
            # for bt in range(batch_size):
            #     S0_rec[bt, ev_i, xs_np[bt,ev_i],ys_np[bt,ev_i],closest0[bt]] = S0[bt]
            #     dS0_rec[bt, ev_i, xs_np[bt,ev_i],ys_np[bt,ev_i],closest0[bt]] = dS0[bt]

            # # print(S0)


            # S1=Class2.variables["output_S"]
            # dS1=Class2.variables["output_dS"]

            # S1_bf=Class2.buffers["output_S_bf"]
            # dS1_bf=Class2.buffers["output_dS_bf"]

            # cl.enqueue_copy(queue, S1, S1_bf).wait()
            # cl.enqueue_copy(queue, dS1, dS1_bf).wait()
            
            # closest1=Class2.variables["closest_c"]
            # closest_bf=Class2.buffers["closest_c_bf"]
            # cl.enqueue_copy(queue, closest1, closest_bf).wait()
            
            # for bt in range(batch_size):
            #     S1_rec[bt,ev_i,closest1[bt]] = S1[bt]
            #     dS1_rec[bt,ev_i,closest1[bt]] = dS1[bt]
            
            # print(dS1)

            
            # closest0=Conv0.variables["closest_c"]
            # closest_bf=Conv0.buffers["closest_c_bf"]
            # cl.enqueue_copy(queue, closest0, closest_bf).wait()

                            
            # thresholds0=Conv0.variables["thresholds"]
            # thresholds_bf=Conv0.buffers["thresholds_bf"]
            # cl.enqueue_copy(queue, thresholds0, thresholds_bf).wait()

            # thresholds1=Dense1.variables["thresholds"]
            # thresholds_bf=Dense1.buffers["thresholds_bf"]
            # cl.enqueue_copy(queue, thresholds1, thresholds_bf).wait()

            # print(thresholds0)
            
            kernel=program.next_ev(queue, np.array([batch_size]), None, ev_i_bf)
        
        end_exec = time.time()
        
        if (batch_i%5):
            print("LEARNING BATCH")
        else:
            print("VALIDATION BATCH")

        # if (batch_i%5):
        Conv0.batch_update(queue)        
        Dense1.batch_update(queue)
        Class2.batch_update(queue)
        
        processed_ev = Class2.variables["processed_ev"]
        correct_ev = Class2.variables["correct_ev"]
        
        avg_processed_ev = np.mean(processed_ev / n_events_batch)
        avg_accuracy = np.mean(correct_ev / processed_ev)
        
        # print("TRAIN")
        print("Epoch: "+str(epoch_i)+" of "+str(n_epochs))
        print("Batch: "+str(batch_i)+" of "+str(n_batches))
        print("Processed rec "+str(rec)+" of "+str(len(train_labels))+" Label: "+str(np.unique(train_batch_labels[np.where(correct_ev!=0)])))
        print("Elapsed time is ", (end_exec-start_exec) * 10**3, "ms")
        print("Accuracy is "+str(avg_accuracy)+" of "+str(avg_processed_ev)+" processed events")

        Conv0.batch_flush(queue)        
        Dense1.batch_flush(queue)
        Class2.batch_flush(queue)
        
        if epoch_i<second_fase_i_epoch:
            Conv0.variables["thresholds"][:]=th_size_0
            thresholds=Conv0.variables["thresholds"]
            thresholds_bf=Conv0.buffers["thresholds_bf"]
            cl.enqueue_copy(queue, thresholds_bf, thresholds).wait()
            Dense1.variables["thresholds"][:]=th_size_1
            thresholds=Dense1.variables["thresholds"]
            thresholds_bf=Dense1.buffers["thresholds_bf"]
            cl.enqueue_copy(queue, thresholds_bf, thresholds).wait()

#%%
plt.figure()
plt.imshow(S1_rec[0])            
            
#%% COntrol variables


distances=Conv0.variables["distances"]
distances_bf=Conv0.buffers["distances_bf"]
cl.enqueue_copy(queue, distances, distances_bf).wait()

time_surface=Conv0.variables["time_surface"]
time_surface_bf=Conv0.buffers["time_surface_bf"]
cl.enqueue_copy(queue, time_surface, time_surface_bf).wait()

distances=Dense1.variables["distances"]
distances_bf=Dense1.buffers["distances_bf"]
cl.enqueue_copy(queue, distances, distances_bf).wait()

dcentroids0=Conv0.variables["dcentroids"]
dcentroids_bf=Conv0.buffers["dcentroids_bf"]
cl.enqueue_copy(queue, dcentroids0, dcentroids_bf).wait()
Conv0.variables["dcentroids"]=dcentroids0

dcentroids=Class2.variables["dcentroids"]
dcentroids_bf=Class2.buffers["dcentroids_bf"]
cl.enqueue_copy(queue, dcentroids, dcentroids_bf).wait()
Class2.variables["dcentroids"]=dcentroids


closest0=Conv0.variables["closest_c"]
closest_bf=Conv0.buffers["closest_c_bf"]
cl.enqueue_copy(queue, closest0, closest_bf).wait()


closest1=Dense1.variables["closest_c"]
closest_bf=Dense1.buffers["closest_c_bf"]
cl.enqueue_copy(queue, closest1, closest_bf).wait()

closest2=Class2.variables["closest_c"]
closest_bf=Class2.buffers["closest_c_bf"]
cl.enqueue_copy(queue, closest2, closest_bf).wait()

S0=Dense1.variables["output_S"]
dS0=Dense1.variables["output_dS"]

S0_bf=Dense1.buffers["output_S_bf"]
dS0_bf=Dense1.buffers["output_dS_bf"]

cl.enqueue_copy(queue, S0, S0_bf).wait()
cl.enqueue_copy(queue, dS0, dS0_bf).wait()


S1=Class2.variables["output_S"]
dS1=Class2.variables["output_dS"]

S1_bf=Class2.buffers["output_S_bf"]
dS1_bf=Class2.buffers["output_dS_bf"]

cl.enqueue_copy(queue, S1, S1_bf).wait()
cl.enqueue_copy(queue, dS1, dS1_bf).wait()

for i in range(len(time_surface)):
    plt.figure()
    plt.title("ts recording: "+str(i))
    plt.imshow(time_surface[i,:,:,0].transpose())
#%% Print
centroids0 = Conv0.variables["centroids"]
centroids1 = Dense1.variables["centroids"]
centroids2 = Class2.variables["centroids"]


for i in range(n_clusters_0):
    plt.figure()
    plt.title("cluster: "+str(i))
    plt.imshow(centroids0[1,i,:,:,0].transpose())
    
for i in range(n_clusters_1):
    plt.figure()
    plt.title("cluster: "+str(i))
    plt.imshow(np.concatenate(centroids1[0,i,:,:,:].transpose()))
    # plt.imshow(centroids1[0,i,:,:,30].transpose())
    
for i in range(n_clusters_2):
    plt.figure()
    plt.title("cluster: "+str(i))
    plt.imshow(centroids2[:,i,0,0,:].transpose())
plt.ylabel("Layer0 cluster#")  

#%% copy weights and prepare for phase 2 
    
# Conv0_old_weights = Conv0.variables["centroids"].copy()
# Dense1_old_weights = Dense1.variables["centroids"].copy()
# Class2_old_weights = Class2.variables["centroids"].copy()

               
# Dense0.variables["thresholds"][:]=230
# thresholds=Dense0.variables["thresholds"]
# thresholds_bf=Dense0.buffers["thresholds_bf"]
# cl.enqueue_copy(queue, thresholds_bf, thresholds).wait()


Conv0.variables["centroids"] = Conv0_old_weights.copy() 
Dense1.variables["centroids"] = Dense1_old_weights.copy() 
Class2.variables["centroids"] = Class2_old_weights.copy() 


centroids_0_bf=Conv0.buffers["centroids_bf"]
centroids_1_bf=Dense1.buffers["centroids_bf"]
centroids_2_bf=Class2.buffers["centroids_bf"]
cl.enqueue_copy(queue, centroids_0_bf, Conv0_old_weights).wait()
cl.enqueue_copy(queue, centroids_1_bf, Dense1_old_weights).wait()
cl.enqueue_copy(queue, centroids_2_bf, Class2_old_weights).wait()


# th_lrate_0 = 1e-4



# Conv0.parameters["th_lrate"]=th_lrate_0
# th_lrate_bf=Conv0.buffers["th_lrate_bf"]
# cl.enqueue_copy(queue, th_lrate_bf, np.float32(th_lrate_0)).wait()

# Dense1.parameters["th_lrate"]=th_lrate_0
# th_lrate_bf=Dense1.buffers["th_lrate_bf"]
# cl.enqueue_copy(queue, th_lrate_bf, np.float32(th_lrate_0)).wait()
    
#%% SAVE

np.save("Conv0_old_weights_learnt",Conv0_old_weights)
np.save("Dense1_old_weights_learnt",Dense1_old_weights)
np.save("Class2_old_weights_learnt",Class2_old_weights)

#%% LOAD

Conv0_old_weights=np.load("Conv0_old_weights.npy")
Dense1_old_weights=np.load("Dense1_old_weights.npy")
Class2_old_weights=np.load("Class2_old_weights.npy")


#%% Create the network two layers

# Parameters
# batch_size = 4 #too high and it might affect how fast it converges 128 it halts progression
batch_size = 128 #too high and it might affect how fast it converges 128 it halts progression

n_labels = 10
n_epochs = np.int32(np.floor(len(train_labels)/batch_size))#I will lose some results


#Dense Layer 1 data and parameters
n_pol_0 = 1
tau_0 = 1e5
n_clusters_0 = 32
# n_clusters_0 = 1
# lrate_0 = 1e-2
# th_lrate_0 = 1e-1
lrate_0 = 1e-4
# th_lrate_0 = 5e-2#Check if smaller can help differentiate more clusters
th_lrate_0 = 1e-4#Check if smaller can help differentiate more clusters
s_gain = 1e-3
# s_gain = 1e1

win_l_0 = 9

th_decay_0=0.9
# th_size_0=250
th_size_0=20
res_x_0 = 28
res_y_0 = 28

Conv0 = Conv_Layer(n_clusters_0, tau_0, res_x_0, res_y_0, win_l_0, n_pol_0, lrate_0,
                     th_size_0, th_lrate_0, th_decay_0, ctx, batch_size,
                     s_gain, debug=True)


#Class Layer 3 data and parameters
tau_2 = 1e3#1e3 actually gave some nice features
tau_2_fb = 1e1
n_clusters_2=10
lrate_2 = 1e-4#If too high  you end up  with a single cluster same as lrate0 works well
res_x_2 = 28
res_y_2 = 28

Class2 = Class_Layer(n_clusters_2, tau_2, res_x_2, res_y_2, n_clusters_0, lrate_2,
                     ctx, batch_size, s_gain, fb_signal=True, fb_tau=tau_2_fb,
                     debug=True)


Conv0.buffers["input_S_bf"] = Class2.buffers["output_S_bf"]
Conv0.buffers["input_dS_bf"] = Class2.buffers["output_dS_bf"]
# Dense1.buffers["correct_response_bf"] = Class2.buffers["correct_response_bf"]

Conv0.variables["input_S"] = Class2.variables["output_S"]
Conv0.variables["input_dS"] = Class2.variables["output_dS"]

# Dense1.variables["correct_response"] = Class2.variables["correct_response"]

fevskip = np.zeros(batch_size, dtype=np.int32)
bevskip = np.zeros(batch_size, dtype=np.int32)

# fevskip for feed event skip, and bevskip for back event skip, 1=>true 0=>false
fevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip)
bevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip)

#%% Initialize clusters 0

rec = 0
epoch_i=0
n_batches = 60000//batch_size
# n_batches = 1
batch_i = 0
ev_i = 0
# second_fase_i_epoch = 1
second_fase_i_epoch = 0



f = open('Libs/cl_kernels/next_ev.cl', 'r')
fstr = "".join(f.readlines())
program=cl.Program(ctx, fstr).build(options='-cl-std=CL2.0')

n_epochs=1
for epoch_i in range(n_epochs):
    rec=0
    # for batch_i in range(n_batches):     
    for batch_i in range(1):     
        n_events_rec=np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
            n_events_rec[i] = len(data_events[0])
            
        n_max_events = max(n_events_rec)
        
        
        xs_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ys_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ps_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ts_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        train_batch_labels = np.zeros([batch_size],dtype=np.int32)
        n_events_batch = np.zeros([batch_size],dtype=np.int32)
        time_surface = np.zeros([batch_size, n_max_events,
                                 win_l_0, win_l_0, n_pol_0],dtype=np.float32)#TODO add zeropadding

        
        
        for i in range(batch_size):
            data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
            n_events = len(data_events[0])
            xs_np[i,:n_events] = data_events[0]
            ys_np[i,:n_events] = data_events[1]
            ps_np[i,:n_events] = data_events[2]*0 #removing pol information at layer 1
            ts_np[i,:n_events] = data_events[3]
            train_batch_labels[i] = train_labels[rec+i]
            n_events_batch[i] = n_events
            
        rec+=batch_size 

        
        # Network Buffers
        xs_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=xs_np)
        ys_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ys_np)
        ps_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ps_np)
        ts_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts_np)
        ev_i_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.int32(0))
        n_events_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_max_events))

        batch_labels_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_batch_labels)
        fevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip)
        bevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip)
        time_surface_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=time_surface)
        
        net_buffers = {"xs_bf" : xs_bf, "ys_bf" : ys_bf, "ps_bf" : ps_bf, 
                       "ts_bf" : ts_bf, "ev_i_bf" : ev_i_bf, 
                       "n_events_bf" : n_events_bf, 
                       "batch_labels_bf" : batch_labels_bf, 
                       "fevskip_bf" : fevskip_bf, "bevskip_bf" : bevskip_bf}
        
        start_exec = time.time()
        for ev_i in range(n_max_events):
            
            Conv0.init_infer(net_buffers, queue, time_surface_bf)
            kernel=program.next_ev(queue, np.array([batch_size]), None, ev_i_bf)
        
        end_exec = time.time()
        
    
        
        # print("TRAIN")
        print("Epoch: "+str(epoch_i)+" of "+str(n_epochs))
        print("Batch: "+str(batch_i)+" of "+str(n_batches))
        print("Elapsed time is ", (end_exec-start_exec) * 10**3, "ms")


        Conv0.batch_flush(queue)    
        cl.enqueue_copy(queue, time_surface, time_surface_bf).wait()

time_surface = np.reshape(time_surface, [batch_size,n_max_events,win_l_0*win_l_0*n_pol_0])
time_surface_input = np.zeros([sum(n_events_batch), win_l_0*win_l_0*n_pol_0])
count = 0;
for i,n_ev in enumerate(n_events_batch):
    time_surface_input[count:count+n_ev] = time_surface[i,:n_ev]
    count+=n_ev
    
init_kmeans_0 = KMeans(n_clusters=n_clusters_0).fit(time_surface_input)
init_centroids_0 =  np.reshape(init_kmeans_0.cluster_centers_, [n_clusters_0,win_l_0,win_l_0,n_pol_0])

for i in range(n_clusters_0):
    plt.figure()
    plt.imshow(init_centroids_0[i,:,:,0].transpose())

del time_surface_input
del init_kmeans_0

Conv0.variables["centroids"][:] = init_centroids_0
centroids_0_bf=Conv0.buffers["centroids_bf"]
cl.enqueue_copy(queue, centroids_0_bf, Conv0.variables["centroids"]).wait()

#%%
centroid_distance = np.zeros([n_clusters_0,n_clusters_0])
for i in range(n_clusters_0):
    for j in range(n_clusters_0):
        centroid_distance[i,j] = np.sum((init_centroids_0[i]-init_centroids_0[j])**2)
radius=np.max(centroid_distance)/2
#%% Initialize clusters 1 


for epoch_i in range(n_epochs):
    rec=0
    # for batch_i in range(n_batches):     
    for batch_i in range(1):     
        n_events_rec=np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
            n_events_rec[i] = len(data_events[0])
            
        n_max_events = max(n_events_rec)
              
        xs_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ys_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ps_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ts_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        train_batch_labels = np.zeros([batch_size],dtype=np.int32)
        n_events_batch = np.zeros([batch_size],dtype=np.int32)
        time_surface = np.zeros([batch_size, res_x_2, res_y_2, n_clusters_0],dtype=np.float32)#TODO add zeropadding
        
        
        for i in range(batch_size):
            data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
            n_events = len(data_events[0])
            xs_np[i,:n_events] = data_events[0]
            ys_np[i,:n_events] = data_events[1]
            ps_np[i,:n_events] = data_events[2]*0 #removing pol information at layer 1
            ts_np[i,:n_events] = data_events[3]
            train_batch_labels[i] = train_labels[rec+i]
            n_events_batch[i] = n_events
            
        rec+=batch_size 
        
        # Network Buffers
        xs_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=xs_np)
        ys_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ys_np)
        ps_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ps_np)
        ts_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts_np)
        ev_i_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.int32(0))
        n_events_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_max_events))
        batch_labels_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_batch_labels)
        fevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip)
        bevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip)
        # time_surface_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=time_surface)

        
        net_buffers = {"xs_bf" : xs_bf, "ys_bf" : ys_bf, "ps_bf" : ps_bf, 
                       "ts_bf" : ts_bf, "ev_i_bf" : ev_i_bf, 
                       "n_events_bf" : n_events_bf, 
                       "batch_labels_bf" : batch_labels_bf, 
                       "fevskip_bf" : fevskip_bf, "bevskip_bf" : bevskip_bf}
        
        start_exec = time.time()
        for ev_i in range(n_max_events):
            
            Conv0.infer(net_buffers, queue)
            # Class2.init_infer(net_buffers, queue, time_surface_bf)
            Class2.infer(net_buffers, queue)
            cl.enqueue_copy(queue,Class2.variables["time_surface"], Class2.buffers["time_surface_bf"]).wait()
            for rec in range(batch_size):
                if ev_i<n_events_batch[rec]:
                    time_surface[rec] += Class2.variables["time_surface"][rec]/n_events_batch[rec]
                
            
            kernel=program.next_ev(queue, np.array([batch_size]), None, ev_i_bf)
        
        end_exec = time.time()
        

        
        # print("TRAIN")
        print("Epoch: "+str(epoch_i)+" of "+str(n_epochs))
        print("Batch: "+str(batch_i)+" of "+str(n_batches))
        print("Elapsed time is ", (end_exec-start_exec) * 10**3, "ms")

        Conv0.batch_flush(queue)        
        Class2.batch_flush(queue)
        # cl.enqueue_copy(queue, time_surface, time_surface_bf).wait()


init_centroids_2 = np.zeros([n_clusters_2, res_x_2, res_y_2, n_clusters_0])
n_label_recs = [sum(train_batch_labels==label) for label in range(n_labels)]
for i,n_ev in enumerate(n_events_batch):
    init_centroids_2[train_batch_labels[i]] += time_surface[i]/n_label_recs[train_batch_labels[i]]
    

y = np.arange(res_x_2//2,res_x_2*n_clusters_0,res_x_2)
lb = np.arange(0,n_clusters_0)

for i in range(n_clusters_2):
    plt.figure()
    plt.title("cluster: "+str(i))
    plt.imshow(np.concatenate(init_centroids_2[i,:,:,:].transpose()))
    plt.yticks(y, lb, rotation='vertical')



Class2.variables["centroids"][:] = init_centroids_2
centroids_2_bf=Class2.buffers["centroids_bf"]
cl.enqueue_copy(queue, centroids_2_bf, Class2.variables["centroids"]).wait()


#%% Train

rec = 0
epoch_i=0
n_batches = 60000//batch_size
# n_batches = 1
batch_i = 0
ev_i = 0
# second_fase_i_epoch = 1
second_fase_i_epoch = 0



f = open('Libs/cl_kernels/next_ev.cl', 'r')
fstr = "".join(f.readlines())
program=cl.Program(ctx, fstr).build(options='-cl-std=CL2.0')

n_epochs=20
validation_split=0.1
validation_accuracy = np.zeros([n_epochs, int(n_batches*validation_split)+1])
validation_cutoff = np.zeros([n_epochs, int(n_batches*validation_split)+1])

for epoch_i in range(0, n_epochs):
    rec=0
    for batch_i in range(0,n_batches):     
    # for batch_i in range(1):     
        n_events_rec=np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
            n_events_rec[i] = len(data_events[0])
            
        n_max_events = max(n_events_rec)
        
        S0_rec = np.zeros([batch_size,28,28,n_clusters_0])
        dS0_rec = np.zeros([batch_size,28,28,n_clusters_0])
        rec_dist = np.zeros([batch_size, n_max_events, n_clusters_2])
        
        xs_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ys_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ps_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ts_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        train_batch_labels = np.zeros([batch_size],dtype=np.int32)
        n_events_batch = np.zeros([batch_size],dtype=np.int32)
        
        
        for i in range(batch_size):
            data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
            n_events = len(data_events[0])
            xs_np[i,:n_events] = data_events[0]
            ys_np[i,:n_events] = data_events[1]
            ps_np[i,:n_events] = data_events[2]*0 #removing pol information at layer 1
            ts_np[i,:n_events] = data_events[3]
            train_batch_labels[i] = train_labels[rec+i]
            n_events_batch[i] = n_events
            
        rec+=batch_size 
        processed_ev = np.zeros([batch_size],dtype=np.int32)
        correct_ev = np.zeros([batch_size],dtype=np.int32)
        
        # Network Buffers
        xs_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=xs_np)
        ys_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ys_np)
        ps_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ps_np)
        ts_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts_np)
        ev_i_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.int32(0))
        n_events_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_max_events))
        processed_ev_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=processed_ev)
        correct_ev_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=correct_ev)
        batch_labels_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_batch_labels)
        fevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip)
        bevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip)
        
        net_buffers = {"xs_bf" : xs_bf, "ys_bf" : ys_bf, "ps_bf" : ps_bf, 
                       "ts_bf" : ts_bf, "ev_i_bf" : ev_i_bf, 
                       "n_events_bf" : n_events_bf, 
                       "processed_ev_bf" : processed_ev_bf, 
                       "correct_ev_bf" : correct_ev_bf, 
                       "batch_labels_bf" : batch_labels_bf, 
                       "n_labels_bf" : Class2.buffers["n_clusters_bf"],
                       "fevskip_bf" : fevskip_bf, "bevskip_bf" : bevskip_bf}
        
        start_exec = time.time()
        
        for ev_i in range(n_max_events):
            
            Conv0.infer(net_buffers, queue)
            Class2.infer(net_buffers, queue)
            
            if (batch_i%validation_split*100):
                Class2.learn(net_buffers, queue)
                
                if epoch_i<second_fase_i_epoch:
                    Conv0.init_learn(net_buffers, queue)
    
                else:
                    Conv0.learn(net_buffers, queue)
                                    
            


            # S0=Class2.variables["output_S"]
            # dS0=Class2.variables["output_dS"]

            # S0_bf=Class2.buffers["output_S_bf"]
            # dS0_bf=Class2.buffers["output_dS_bf"]

            # cl.enqueue_copy(queue, S0, S0_bf).wait()
            # cl.enqueue_copy(queue, dS0, dS0_bf).wait()
            
            # closest0=Conv0.variables["closest_c"]
            # closest_bf=Conv0.buffers["closest_c_bf"]
            # cl.enqueue_copy(queue, closest0, closest_bf).wait()
            
            # for bt in range(batch_size):
            #     if ts_np[bt,ev_i]!=-1:
            #         S0_rec[bt, xs_np[bt,ev_i],ys_np[bt,ev_i],closest0[bt]] += S0[bt]
            #         dS0_rec[bt, xs_np[bt,ev_i],ys_np[bt,ev_i],closest0[bt]] += dS0[bt]
            

            
            # closest0=Conv0.variables["closest_c"]
            # closest_bf=Conv0.buffers["closest_c_bf"]
            # cl.enqueue_copy(queue, closest0, closest_bf).wait()

                            
            # thresholds0=Conv0.variables["thresholds"]
            # thresholds_bf=Conv0.buffers["thresholds_bf"]
            # cl.enqueue_copy(queue, thresholds0, thresholds_bf).wait()



            # print(thresholds0)
            
            # distances0=Class2.variables["distances"]
            # distances_bf=Class2.buffers["distances_bf"]
            # cl.enqueue_copy(queue, distances0, distances_bf).wait()
            
            # rec_dist[:,ev_i,:] = distances0
            
            kernel=program.next_ev(queue, np.array([batch_size]), None, ev_i_bf)
        
        end_exec = time.time()
        
        if (batch_i%int(validation_split*100)):
            print("LEARNING BATCH")
        else:
            print("VALIDATION BATCH")

        # if (batch_i%5):
        Conv0.batch_update(queue)        
        Class2.batch_update(queue)
        
        processed_ev = Class2.variables["processed_ev"]
        correct_ev = Class2.variables["correct_ev"]
        
        avg_processed_ev = np.mean(processed_ev / n_events_batch)
        avg_accuracy = np.mean(correct_ev / processed_ev)
        label_accuracy = np.sum((correct_ev / processed_ev)>0.5)/batch_size
        
        # print("TRAIN")
        print("Epoch: "+str(epoch_i)+" of "+str(n_epochs))
        print("Batch: "+str(batch_i)+" of "+str(n_batches))
        print("Processed rec "+str(rec)+" of "+str(len(train_labels))+" Label: "+str(np.unique(train_batch_labels[np.where(correct_ev!=0)])))
        print("Elapsed time is ", (end_exec-start_exec) * 10**3, "ms")
        print("Accuracy is "+str(avg_accuracy)+" of "+str(avg_processed_ev)+" processed events")
        print("Label Accuracy is "+str(label_accuracy))

        Conv0.batch_flush(queue)        
        Class2.batch_flush(queue)
        
        if epoch_i<second_fase_i_epoch:
            Conv0.variables["thresholds"][:]=th_size_0
            thresholds=Conv0.variables["thresholds"]
            thresholds_bf=Conv0.buffers["thresholds_bf"]
            cl.enqueue_copy(queue, thresholds_bf, thresholds).wait()
            
        if not (batch_i%int(validation_split*100)):
            validation_accuracy[epoch_i, int(batch_i*validation_split)] = label_accuracy
            validation_cutoff[epoch_i, int(batch_i*validation_split)] = avg_processed_ev



# centroids0_base=Conv0.variables["centroids"].copy()
# centroids2_base=Class2.variables["centroids"].copy()

centroids0_update=Conv0.variables["centroids"].copy()
centroids2_update=Class2.variables["centroids"].copy()

centroids2 = centroids2_update - init_centroids_2
centroids0 = centroids0_update - init_centroids_0
#TODO CONTINUE FROM THIS

#%% S dS plots

y = np.arange(res_x_2//2,res_x_2*n_clusters_0,res_x_2)
lb = np.arange(0,n_clusters_0)

plt.figure()
plt.title("S0")
plt.imshow(np.concatenate(np.mean(S0_rec[:,:,:,:],axis=0).transpose()))
plt.yticks(y, lb, rotation='vertical')

cluster_i = 0
for b_i in range(batch_size):
    plt.figure()
    plt.title("S0")
    plt.imshow(S0_rec[b_i,:,:,cluster_i].transpose())

plt.figure()
plt.title("dS0")
plt.imshow(np.concatenate(np.mean(dS0_rec[:,:,:,:],axis=0).transpose()))
plt.yticks(y, lb, rotation='vertical')

for b_i in range(batch_size):
    plt.figure()
    plt.title("dS0")
    plt.imshow(dS0_rec[b_i,:,:,cluster_i].transpose())
    
    
for b_i in range(batch_size):
    plt.figure()
    plt.title("S0+dS0")
    plt.imshow((dS0_rec[b_i,:,:,cluster_i].transpose())+ s_gain*(dS0_rec[b_i,:,:,cluster_i].transpose()))

plt.figure()
plt.title("dS0")
plt.imshow(np.concatenate(dS0_rec[0,:,:,:].transpose()))
plt.yticks(y, lb, rotation='vertical')


plt.figure()
plt.title("S0+dS0")
tot_dS=np.concatenate(np.mean(dS0_rec[:,:,:,:],axis=0).transpose())
tot_S=s_gain*np.concatenate(np.mean(S0_rec[:,:,:,:],axis=0).transpose())
plt.imshow(tot_dS+tot_S)
plt.yticks(y, lb, rotation='vertical')



#%%
plt.figure()
res = np.sum(dS0_rec[:,:,:,:,1], axis=(0,1)).transpose()
plt.imshow(res)            
            
#%% COntrol variables


distances=Conv0.variables["distances"]
distances_bf=Conv0.buffers["distances_bf"]
cl.enqueue_copy(queue, distances, distances_bf).wait()

thresholds_update=Conv0.variables["thresholds_update"]
thresholds_update_bf=Conv0.buffers["thresholds_update_bf"]
cl.enqueue_copy(queue, thresholds_update, thresholds_update_bf).wait()

time_surface=Conv0.variables["time_surface"]
time_surface_bf=Conv0.buffers["time_surface_bf"]
cl.enqueue_copy(queue, time_surface, time_surface_bf).wait()


dcentroids0=Conv0.variables["dcentroids"]
dcentroids_bf=Conv0.buffers["dcentroids_bf"]
cl.enqueue_copy(queue, dcentroids0, dcentroids_bf).wait()
Conv0.variables["dcentroids"]=dcentroids0

dcentroids=Class2.variables["dcentroids"]
dcentroids_bf=Class2.buffers["dcentroids_bf"]
cl.enqueue_copy(queue, dcentroids, dcentroids_bf).wait()
Class2.variables["dcentroids"]=dcentroids


closest0=Conv0.variables["closest_c"]
closest_bf=Conv0.buffers["closest_c_bf"]
cl.enqueue_copy(queue, closest0, closest_bf).wait()




closest2=Class2.variables["closest_c"]
closest_bf=Class2.buffers["closest_c_bf"]
cl.enqueue_copy(queue, closest2, closest_bf).wait()


S0=Class2.variables["output_S"]
dS0=Class2.variables["output_dS"]

S0_bf=Class2.buffers["output_S_bf"]
dS0_bf=Class2.buffers["output_dS_bf"]

cl.enqueue_copy(queue, S0, S0_bf).wait()
cl.enqueue_copy(queue, dS0, dS0_bf).wait()

for i in range(len(time_surface)):
    plt.figure()
    plt.title("ts recording: "+str(i))
    plt.imshow(time_surface[i,:,:,0].transpose())
#%% Print
centroids0 = Conv0.variables["centroids"] # + centroids0*100
centroids2 = Class2.variables["centroids"] #+ centroids2*100


for i in range(n_clusters_0):
    plt.figure()
    plt.title("cluster: "+str(i))
    plt.imshow(centroids0[1,i,:,:,0].transpose())

y = np.arange(res_x_2//2,res_x_2*n_clusters_0,res_x_2)
lb = np.arange(0,n_clusters_0)

for i in range(n_clusters_2):
    plt.figure()
    plt.title("cluster: "+str(i))
    plt.imshow(np.concatenate(centroids2[0,i,:,:,:].transpose()))
    plt.yticks(y, lb, rotation='vertical')


#%% copy weights and prepare for phase 2 
    
# Conv0_old_weights = Conv0.variables["centroids"].copy()
# Class2_old_weights = Class2.variables["centroids"].copy()

               
# Dense0.variables["thresholds"][:]=230
# thresholds=Dense0.variables["thresholds"]
# thresholds_bf=Dense0.buffers["thresholds_bf"]
# cl.enqueue_copy(queue, thresholds_bf, thresholds).wait()


# Conv0.variables["centroids"] = Conv0_old_weights.copy() 
# Dense1.variables["centroids"] = Dense1_old_weights.copy() 
# Class2.variables["centroids"] = Class2_old_weights.copy() 


# centroids_0_bf=Conv0.buffers["centroids_bf"]
# centroids_1_bf=Dense1.buffers["centroids_bf"]
# centroids_2_bf=Class2.buffers["centroids_bf"]
# cl.enqueue_copy(queue, centroids_0_bf, Conv0_old_weights).wait()
# cl.enqueue_copy(queue, centroids_1_bf, Dense1_old_weights).wait()
# cl.enqueue_copy(queue, centroids_2_bf, Class2_old_weights).wait()


s_gain = 1e-8
# s_gain = 1e-3



Conv0.parameters["s_gain"]=s_gain
s_gain_bf=Conv0.buffers["s_gain_bf"]
cl.enqueue_copy(queue, s_gain_bf, np.float32(s_gain)).wait()

Class2.parameters["s_gain"]=s_gain
s_gain_bf=Class2.buffers["s_gain_bf"]
cl.enqueue_copy(queue, s_gain_bf, np.float32(s_gain)).wait()
    

th_size_0=30

Conv0.variables["thresholds"][:]=th_size_0
thresholds_bf=Conv0.buffers["thresholds_bf"]
cl.enqueue_copy(queue, thresholds_bf, Conv0.variables["thresholds"]).wait()


lrate_0 = 2e-5
# lrate_0 = 2e-4


Conv0.parameters["lrate"]=lrate_0
lrate_bf=Conv0.buffers["lrate_bf"]
cl.enqueue_copy(queue, lrate_bf, np.float32(lrate_0)).wait()


th_lrate_0 = 2e-5
# th_lrate_0 = 2e-4


Conv0.parameters["th_lrate"]=th_lrate_0
th_lrate_bf=Conv0.buffers["th_lrate_bf"]
cl.enqueue_copy(queue, s_gain_bf, np.float32(th_lrate_0)).wait()

# lrate_2 = 1e-6
lrate_2 = 2e-5


Class2.parameters["lrate"]=lrate_2
lrate_bf=Class2.buffers["lrate_bf"]
cl.enqueue_copy(queue, lrate_bf, np.float32(lrate_2)).wait()

#%% SAVE
save_dr = "Results/weights_save_NMNIST/"
np.save(save_dr+"Conv0_old_weights_best_results_th_stable91_tmp3",Conv0.variables["centroids"])
np.save(save_dr+"Conv0_old_th_learnt_best_results_th_stable91_tmp3",Conv0.variables["thresholds"])
np.save(save_dr+"Class2_old_weights_learnt_best_results_th_stable91_tmp3",Class2.variables["centroids"])
np.save(save_dr+"Accuracy_progress_stable91_tmp3",validation_accuracy)
np.save(save_dr+"Cutoff_progress_stable91_tmp3",avg_processed_ev)



#%% LOAD
save_dr = "Results/weights_save_NMNIST/"

Conv0.variables["centroids"]=np.load(save_dr+"Conv0_old_weights_best_results_th_stable91.npy")
Conv0.variables["thresholds"]=np.load(save_dr+"Conv0_old_th_learnt_best_results_th_stable91.npy")
Class2.variables["centroids"]=np.load(save_dr+"Class2_old_weights_learnt_best_results_th_stable91.npy")

centroids_0_bf=Conv0.buffers["centroids_bf"]
thresholds_0_bf=Conv0.buffers["thresholds_bf"]
centroids_2_bf=Class2.buffers["centroids_bf"]
cl.enqueue_copy(queue, centroids_0_bf, Conv0.variables["centroids"]).wait()
cl.enqueue_copy(queue, thresholds_0_bf, Conv0.variables["thresholds"]).wait()
cl.enqueue_copy(queue, centroids_2_bf, Class2.variables["centroids"]).wait()
