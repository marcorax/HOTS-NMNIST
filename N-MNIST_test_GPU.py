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

from Libs.HOTSLib import fb_surfaces, all_surfaces, n_mnist_rearranging,\
                            dataset_resize


#%% Data loading and parameters setting
            
## Data loading
train_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/train_set.mat')['train_set'])
test_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/test_set.mat')['test_set'])
n_recording_labels_train=[len(train_set_orig[label]) for label in range(len(train_set_orig))]
n_recording_labels_test=[len(test_set_orig[label]) for label in range(len(train_set_orig))]

# using a subset of N-MNIST to lower memory usage
files_dataset_train = min(n_recording_labels_train)//10 #10
files_dataset_test = min(n_recording_labels_test)//10 #10 
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

#%% GPU Initialization
mf = cl.mem_flags
platforms = cl.get_platforms()
platform_i = 0 #Select the platform manually here
devices = platforms[platform_i].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=devices)#TODO Check how the context apply to more than one GPU
queue = cl.CommandQueue(ctx)
    
#%% GPU - Train

# Parameters
batch_size = 1024
n_labels = 10
n_epochs = np.int32(np.floor(len(train_labels)/batch_size))#I will lose some results


#Conv Layer 1 data and parameters
surf_x_0 = 11
surf_y_0 = 11
n_pol_0 = 1
tau_0 = 50e3
suf_div_x_0 = surf_x_0//2
suf_div_y_0 = surf_y_0//2
n_clusters_0 = 64
weights_0 = np.random.rand(batch_size, surf_x_0, surf_y_0, n_clusters_0)
time_context_0 = np.zeros([batch_size, res_x, res_y, n_pol_0],dtype=np.int32)#adding extrapixel to zeropadp
mask_0 = np.zeros([batch_size, res_x, res_y, n_pol_0],dtype=np.int32)#adding extrapixel to zeropadp
th_0 = np.zeros([batch_size,n_clusters_0], dtype=np.float32)+60


#Dense Layer 2 data and parameters Classifier
weights_1 = np.zeros([batch_size, n_clusters_0, res_x, res_y, n_labels]) #classifier
time_context_1 = np.zeros([batch_size, n_clusters_0, res_x, res_y],dtype=np.int32)
time_context_fb_1 = np.zeros([batch_size, n_labels], dtype=np.int32)#TODO Check if it is still zero

#Data Buffers allocation
weights_0_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=weights_0)
weights_1_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=weights_1)
surf_x_0_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(surf_x_0))
surf_y_0_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(surf_y_0))
res_x_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(res_x))
res_y_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(res_y))
tau_0_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(tau_0))
n_pol_0_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_pol_0))

#Building the kernel
global_space=np.array([batch_size,surf_x_0,surf_y_0])
local_space=np.array([1,surf_x_0,surf_y_0])
f = open('surf_conv.cl', 'r')
fstr = "".join(f.readlines())
# print(fstr)
program=cl.Program(ctx, fstr).build()

rec = 0
for epoch_i in range(1):
    
    n_events_rec=np.zeros(batch_size, dtype=int)
    for i in range(batch_size):
        data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
        n_events_rec[i] = len(data_events[0])
        
    n_max_events = max(n_events_rec)

    xs_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
    ys_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
    ps_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
    ts_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
    TS_np = -1*np.ones((batch_size,n_max_events,surf_x_0,surf_y_0),dtype=np.float32)    
    
    for i in range(batch_size):
        data_events = train_set_orig[train_labels[rec+i]][train_rec_idx[rec+i]]
        n_events = len(data_events[0])
        xs_np[i,:n_events] = data_events[0]
        ys_np[i,:n_events] = data_events[1]
        ps_np[i,:n_events] = data_events[2]*0 #throwing away events polarities at first layer
        ts_np[i,:n_events] = data_events[3]
        
    rec+=batch_size 
    batch_size=len(train_labels[rec:rec+batch_size])
        
    time_context_0_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=time_context_0)
    mask_0_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=mask_0)

    xs_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xs_np)
    ys_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ys_np)
    ps_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ps_np)
    ts_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts_np)
    n_max_events_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_max_events))
    TS_bf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=TS_np)
    event_idx_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.int32(0))


    for ev_i in range(n_max_events):
        kernel=program.surf_conv(queue, global_space, local_space, xs_bf, ys_bf, ps_bf,
                             ts_bf, res_x_bf, res_y_bf, surf_x_0_bf, surf_y_0_bf,
                             tau_0_bf, n_pol_0_bf, TS_bf, event_idx_bf,
                             n_max_events_bf, time_context_0_bf, mask_0_bf)
        print("Processed ev "+str(ev_i)+" of "+str(n_max_events))


       
    print("Processed rec "+str(rec)+" of "+str(len(train_labels)))
       

cl.enqueue_copy(queue, TS_np, TS_bf).wait()
# ev_out = np.int32(0)
cl.enqueue_copy(queue, mask_0, mask_0_bf).wait()
cl.enqueue_copy(queue, time_context_0, time_context_0_bf).wait()


#%% Learning





#Each word is 3 characters long and sentences have no spaces
#Between words

weights_0 = np.random.rand(surf_x, surf_y, n_clusters_0)
# weights_1 = np.random.rand(n_clusters_0, res_x, res_y, n_labels) #classifier
weights_1 = np.zeros([n_clusters_0, res_x, res_y, n_labels]) #classifier




th_0 = np.zeros(n_clusters_0)+60



# fig, axs = plt.subplots(n_clusters_0)
fig, axs = plt.subplots(8,8)
axs = axs.flatten()
fig.suptitle("New L Features")


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

time_context_0 = np.zeros([res_x+surf_x-1, res_y+surf_x-1],dtype=int)#adding extrapixel to zeropadp
time_context_1 = np.zeros([n_clusters_0, res_x, res_y],dtype=int)
time_context_fb_1 = np.zeros([n_labels],dtype=int)


tau_0 = 50e3
# tau_1 = 1e1#se non va prova questi
tau_1 = 1e2


lrate_1 = 5e-5
lrate_0 = 1.5*lrate_1


lrate_th_0 = 2*lrate_0

exp_decay_1=0.5


pause_pressed=False  
print_lay=2  
with keyboard.Listener(on_press=on_press) as listener:
    
    for epoch in range(3):   
        for data_rec_i, label_i in enumerate(train_labels):
            
            rec_i = train_rec_idx[data_rec_i]
            n_events = len(train_set_orig[label_i][rec_i][0])
                        
            
            computed_events = 0
            event_accuracy = 0
    
            #event mask used to avoid exponential decay calculation for pixel 
            # that did not generate an event yet
            mask_start_0 = np.zeros([res_x+surf_x-1, res_y+surf_x-1],dtype=int)#adding extrapixel to zeropad 
            mask_start_1 = np.zeros([n_clusters_0, res_x, res_y],dtype=int)
            mask_start_fb_1 = np.zeros([n_labels],dtype=int)
            
            y_som_0=0
            y_som_old_0=0
            dt_y_som_0=0
           
            
            for ev_i in range(n_events):
                
                ref_x = train_set_orig[label_i][rec_i][0][ev_i]     
                ref_y = train_set_orig[label_i][rec_i][1][ev_i]     
                ref_ts =  train_set_orig[label_i][rec_i][3][ev_i]
                
                #Create the first time surface
                time_context_0[ref_x+suf_div_x,ref_y+suf_div_y] = ref_ts
                mask_start_0[ref_x+suf_div_x,ref_y+suf_div_y] = 1
                
                beg_ts_x = ref_x#+suf_div_x-suf_div_x
                beg_ts_y = ref_y
                end_ts_x = ref_x+(2*suf_div_x)+1
                end_ts_y = ref_y+(2*suf_div_y)+1

                ts_time_context_0 = time_context_0[beg_ts_x:end_ts_x,\
                                                   beg_ts_y:end_ts_y]
                    
                ts_mask_start_0 = mask_start_0[beg_ts_x:end_ts_x,\
                                                   beg_ts_y:end_ts_y]
                                
                
                ts_lay_0 = np.exp(((ts_time_context_0-ref_ts)/tau_0)*ts_mask_start_0)*ts_mask_start_0
                
                rec_distances_0=np.sum((ts_lay_0[:,:,None]-weights_0)**2,axis=(0,1))

                
                # Closest center with threshold computation
                rec_closest_0=np.argmin(rec_distances_0-th_0,axis=0)
                # rec_closest_0=np.argmin(rec_distances_0,axis=0)


                
                # Layer 1 check
                if (rec_distances_0[rec_closest_0]-th_0[rec_closest_0])<0:
                   
                    time_context_1[rec_closest_0, ref_x, ref_y] = ref_ts
                    mask_start_1[rec_closest_0, ref_x, ref_y] = 1
                    
                    ts_lay_1 = np.exp((time_context_1-ref_ts)*mask_start_1/tau_1)*mask_start_1                                 
                    
                    rec_distances_1=np.sum((ts_lay_1[:,:,:,None]-weights_1)**2,axis=(0,1,2))
                  
                    rec_closest_1=np.argmin(rec_distances_1,axis=0)
                    
                    
                    ##FEEDBACK CALCULATIONS
                    
                    #Layer 1
                    time_context_fb_1[rec_closest_1] = ref_ts
                    mask_start_fb_1[rec_closest_1]=1    
                    ts_fb_lay_1 = np.exp((time_context_fb_1-ref_ts)*mask_start_fb_1/tau_1)*mask_start_fb_1                                                        
                    
                    y_som_0=(ts_fb_lay_1[rec_closest_1]-np.sum((ts_fb_lay_1[np.arange(n_labels)!=rec_closest_1]),axis=0)) #normalized by activation
                    
                    #supervised
                    if rec_closest_1!=label_i:
                        y_som_0=-y_som_0

    
    
                    dt_y_som_0 = y_som_0 - y_som_old_0
                    y_som_old_0 = y_som_0
                    
                    
                    ##WEIGHTS AND TH UPDATE

                    #Layer 1
                    #supervised
                    elem_distances_1 = (ts_lay_1[:,:,:]-weights_1[:,:,:,label_i])
                    weights_1[:,:,:,label_i]+=lrate_1*elem_distances_1[:]
                    
                    #unsupervised       
                    # elem_distances_1 = (ts_lay_1[:,:,:]-weights_1[:,:,:,rec_closest_1])
                    # weights_1[:,:,rec_closest_1]+=lrate_1*elem_distances_1[:]
    
                    #Layer 0
                    #weights
                    elem_distances_0 = (ts_lay_0[:,:]-weights_0[:,:,rec_closest_0])
                    # rec_closest_0_one_hot = np.zeros([n_clusters_0])
                    # rec_closest_0_one_hot[rec_closest_0]=1
                    
                    # Keep only the distances for winners
                    weights_0[:,:,rec_closest_0]+= (lrate_0*(dt_y_som_0*elem_distances_0[:]) + 0.01*lrate_0*(y_som_0*elem_distances_0[:]))

                   
                    #threshold
                    for i_cluster in range(n_clusters_0):
                        if i_cluster==rec_closest_0:
                            th_0[rec_closest_0] += lrate_th_0*dt_y_som_0*np.exp((rec_distances_0[rec_closest_0]-th_0[rec_closest_0])/exp_decay_1) + 0.01*lrate_th_0*y_som_0*np.exp((rec_distances_0[rec_closest_0]-th_0[rec_closest_0])/exp_decay_1)
                        elif ((rec_distances_0[i_cluster]-th_0[i_cluster])<0) and (dt_y_som_0>0) and (y_som_0>0):
                        # elif ((rec_distances_0[i_cluster]-th_0[i_cluster])<0):

                        # else:
                            th_0[i_cluster] -= lrate_th_0*dt_y_som_0*np.exp((rec_distances_0[i_cluster]-th_0[i_cluster])/exp_decay_1) + 0.01*lrate_th_0*y_som_0*np.exp((rec_distances_0[i_cluster]-th_0[i_cluster])/exp_decay_1)
                          
                    

                    ## PROGRESS UPDATE
                    rec_closest_1_one_hot = np.zeros([n_labels])
                    rec_closest_1_one_hot[rec_closest_1]=1
                    class_rate=np.sum(rec_closest_1_one_hot,axis=0)
                        
                    computed_events += 1
                    if rec_closest_1==label_i:
                        result = "Correct"
                        event_accuracy += 1
    
                    else:
                        result = "Wrong"
                    
                    progress = computed_events/n_events
                    rel_accuracy = event_accuracy/computed_events
                    print("Epoch "+str(epoch)+", Recording "+str(data_rec_i)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy))

                    
                    if print_lay==1:
                    #Layer0
                        print(rec_distances_0)
                        print("Prediction: "+result+str(label_i))
                        print("Y-som: "+str(y_som_0)+" dt Y-som: "+str(dt_y_som_0)+" Closest_center: "+str(rec_closest_0))
                        print(th_0)
                    elif print_lay==2:
                        #Layer1
                        print(rec_distances_1)
                        print("Prediction: "+result+str(label_i))
                        print("Y-som: "+str(y_som_0)+" dt Y-som: "+str(dt_y_som_0)+" Closest_center: "+str(rec_closest_1))
                        # print(th_1)

                        
    
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

#%% Plot feedback centroids layer 0

#plot centroids
# fig, axs = plt.subplots(n_clusters_0)
fig, axs = plt.subplots(8,8)
axs=axs.flatten()
for pol_i in range(n_clusters_0):
    if n_clusters_0>1:
        axs[pol_i].imshow(np.reshape(weights_0[:,:,pol_i], [surf_x,surf_y]))
    elif n_clusters_0==1:
        axs.imshow(np.reshape(weights_0[:,:,pol_i], [surf_x,surf_y]))

#%% Testing

tau_1 = 50000



time_context_0 = np.zeros([res_x+surf_x-1, res_y+surf_x-1],dtype=int)#adding extrapixel to zeropad
time_context_1 = np.zeros([n_clusters_0, res_x, res_y],dtype=int)
time_context_fb_1 = np.zeros([n_labels],dtype=int)



accuracies = []
all_progress = []
for data_rec_i, label_i in enumerate(test_labels):
    
    rec_i = test_rec_idx[data_rec_i]
    n_events = len(test_set_orig[label_i][rec_i][0])
                
    
    computed_events = 0
    event_accuracy = 0

    #event mask used to avoid exponential decay calculation for pixel 
    # that did not generate an event yet
    mask_start_0 = np.zeros([res_x+surf_x-1, res_y+surf_x-1],dtype=int)#adding extrapixel to zeropad 
    mask_start_1 = np.zeros([n_clusters_0, res_x, res_y],dtype=int)
    mask_start_fb_1 = np.zeros([n_labels],dtype=int)
    
    y_som_0=0
    y_som_old_0=0
    dt_y_som_0=0
   
    
    for ev_i in range(n_events):
        
        ref_x = test_set_orig[label_i][rec_i][0][ev_i]     
        ref_y = test_set_orig[label_i][rec_i][1][ev_i]     
        ref_ts =  test_set_orig[label_i][rec_i][3][ev_i]
        
        #Create the first time surface
        time_context_0[ref_x+suf_div_x,ref_y+suf_div_y] = ref_ts
        mask_start_0[ref_x+suf_div_x,ref_y+suf_div_y] = 1
        
        beg_ts_x = ref_x#+suf_div_x-suf_div_x
        beg_ts_y = ref_y
        end_ts_x = ref_x+(2*suf_div_x)+1
        end_ts_y = ref_y+(2*suf_div_y)+1

        ts_time_context_0 = time_context_0[beg_ts_x:end_ts_x,\
                                           beg_ts_y:end_ts_y]
            
        ts_mask_start_0 = mask_start_0[beg_ts_x:end_ts_x,\
                                           beg_ts_y:end_ts_y]
                        
        
        ts_lay_0 = np.exp(((ts_time_context_0-ref_ts)/tau_0)*ts_mask_start_0)*ts_mask_start_0
        
        rec_distances_0=np.sum((ts_lay_0[:,:,None]-weights_0)**2,axis=(0,1))

        
        # Closest center with threshold computation
        # rec_closest_0=np.argmin(rec_distances_0-th_0,axis=0)
        rec_closest_0=np.argmin(rec_distances_0,axis=0)


        
        # Layer 1 check
        if (rec_distances_0[rec_closest_0]-th_0[rec_closest_0])<0:
           
            time_context_1[rec_closest_0, ref_x, ref_y] = ref_ts
            mask_start_1[rec_closest_0, ref_x, ref_y] = 1
            
            ts_lay_1 = np.exp((time_context_1-ref_ts)*mask_start_1/tau_1)*mask_start_1                                 
            
            rec_distances_1=np.sum((ts_lay_1[:,:,:,None]-weights_1)**2,axis=(0,1,2))
          
            rec_closest_1=np.argmin(rec_distances_1,axis=0)
            
            
            ##FEEDBACK CALCULATIONS
            
            #Layer 1
            time_context_fb_1[rec_closest_1] = ref_ts
            mask_start_fb_1[rec_closest_1]=1    
            ts_fb_lay_1 = np.exp((time_context_fb_1-ref_ts)*mask_start_fb_1/tau_1)*mask_start_fb_1                                                        
            norm = n_labels-1
            
            #supervised
            y_som_0=(ts_fb_lay_1[label_i]-np.sum((ts_fb_lay_1[np.arange(n_labels)!=label_i]/norm),axis=0)) #normalized by activation
            
            #unsupervised
            # y_som_1=(ts_fb_lay_1[rec_closest_1]-np.sum((ts_fb_lay_1[np.arange(n_labels)!=rec_closest_1]/norm),axis=0)) #normalized by activation


            dt_y_som_0 = y_som_0 - y_som_old_0
            y_som_old_0 = y_som_0
            
            

            ## PROGRESS UPDATE
            rec_closest_1_one_hot = np.zeros([n_labels])
            rec_closest_1_one_hot[rec_closest_1]=1
            class_rate=np.sum(rec_closest_1_one_hot,axis=0)
                
            computed_events += 1
            if rec_closest_1==label_i:
                result = "Correct"
                event_accuracy += 1

            else:
                result = "Wrong"
            
            progress = computed_events/n_events
            rel_accuracy = event_accuracy/computed_events
            print("Epoch "+str(epoch)+", Recording "+str(data_rec_i)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(rel_accuracy))
            print(rec_distances_0)
            print("Prediction: "+result+str(label_i))
            
            # if print_lay==1:
            #Layer0
            print("Y-som: "+str(y_som_0)+" dt Y-som: "+str(dt_y_som_0)+" Closest_center: "+str(rec_closest_0))
            print(th_0)
            # elif print_lay==2:
            #     #Layer1
            #     print("Y-som: "+str(y_som_1)+" dt Y-som: "+str(dt_y_som_1)+" Closest_center: "+str(rec_closest_1))
            #     print(th_1)
            
    
    accuracies.append(rel_accuracy)
    all_progress.append(progress*100)

                

print("Test accuracy is: "+str(sum(((np.array(accuracies)>0.5)/len(accuracies))*100))+"%")
                        
                        
                    
                        

