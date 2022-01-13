#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:28:55 2020

@author: marcorax93
"""

import numpy as np
from joblib import Parallel, delayed 
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
from scipy.spatial.distance import cdist
import time, gc



def surfaces(data_recording, res_x, res_y, surf_dim, tau, n_pol):
    """ 
    This function is used to generate the time surfaces per each recording.

    Arguments :
        
        data_recording: data of a single recording. List of 4 arrays [x, y, p, t]
                        containing spatial coordinates of events (x,y), polarities
                        (p) and timestamps (t).
                        
        res_x, res_y: the x and y pixel resolution of the dataset. 
                        
        surf_dim: the lateral dimension of the squared time context used to
                  build the time surface (it needs to be an odd number to have 
                  a integer number of pixels around the position of the 
                  reference event).
        
        tau : temporal coefficient for the exp decay.
       
        n_pol: number of polarities of the data_recording, if =-1, polarity info
               will be discarded (useful for the first layer usually).
               
    Returns: 
        surfs: array containing all the time surfaces created using the input data.
               -dimension of surf if pol is a int>0 [n_events, n_pol, surf_dim, surf_dim]
               -dimension of surf if pol is -1 [n_events, surf_dim, surf_dim]
        
    """
    
    dl = surf_dim//2 #number of corner extrapixels to zeropadding image plane
    n_events=len(data_recording[3])

    # Allocate some memory
    if n_pol == -1:
        surface = np.zeros([res_y+2*dl, res_x+2*dl], dtype=np.float32) #zeropadding the borders
        surfs = np.zeros([n_events, surf_dim, surf_dim], dtype=np.float32)
        timestamp_table = np.ones([res_y+2*dl, res_x+2*dl])*-1 # (Starting negative values will keep the decay map on FALSE STATES) 

    else:
        surface = np.zeros([n_pol, res_y+2*dl, res_x+2*dl], dtype=np.float32) #zeropadding the borders
        surfs = np.zeros([n_events, n_pol,  surf_dim, surf_dim], dtype=np.float32)
        timestamp_table = np.ones([n_pol, res_y+2*dl, res_x+2*dl])*-1 # (Starting negative values will keep the decay map on FALSE STATES) 

    
    for event in range(n_events):
        new_ts = data_recording[3][event]
        new_x = int(data_recording[0][event]+dl)#offsets to account padding
        new_y = int(data_recording[1][event]+dl)#offsets to account padding
        decay_map = ((new_ts-timestamp_table)>0)*(timestamp_table>0) # map used to identify the regime for each pixel, 1 if decaying
        surface = np.exp(((timestamp_table-new_ts)*decay_map)/tau)*decay_map 
        
        # Update surfs and tables. 
        if n_pol == -1: 
            timestamp_table[new_y,new_x] = new_ts
            surfs[event,:,:] = surface[new_y-dl:new_y+dl+1,
                                             new_x-dl:new_x+dl+1] 
        else:
            new_pol = int(data_recording[2][event])#offsets to account padding                      
            timestamp_table[new_pol,new_y,new_x] = new_ts
            surfs[event, :, :, :] = surface[:, new_y-dl:new_y+dl+1,
                                                  new_x-dl:new_x+dl+1] 
        
    return surfs


def fb_surfaces(data_recording, n_clusters, tau):
    """ 
    This function is used to generate the time surfaces used for feedback per each recording.
    x and y are not used, but in the future it will be adapted for cnn networks

    Arguments :
        
        data_recording: data of a single recording. List of 4 arrays [x, y, p, t]
                        containing spatial coordinates of events (x,y), polarities
                        (p) and timestamps (t).
                        
        n_clusters: The number of clusters in the current layer. It determines the surf length
                        
        
        tau : temporal coefficient for the exp decay.
       
        n_pol: number of polarities of the data_recording, if =-1, polarity info
               will be discarded (useful for the first layer usually).
               
    Returns: 
        surfs: array containing all the time surfaces created using the input data.
               -dimension of surf if pol is a int>0 [n_events, n_pol, surf_dim, surf_dim]
               -dimension of surf if pol is -1 [n_events, surf_dim, surf_dim]
        
    """
    
    n_events=len(data_recording[3])

    # Allocate some memory
    surface = np.zeros([n_clusters], dtype=np.float32) #zeropadding the borders
    surfs = np.zeros([n_events, n_clusters], dtype=np.float32)
    timestamp_table = np.ones([n_clusters])*-1 # (Starting negative values will keep the decay map on FALSE STATES) 

    
    for event in range(n_events):
        new_ts = data_recording[3][event]
        new_cluster = int(data_recording[2][event])
        timestamp_table[new_cluster] = new_ts
        decay_map = ((new_ts-timestamp_table)>=0)*(timestamp_table>0) # map used to identify the regime for each pixel, 1 if decaying
        surface = np.exp(((timestamp_table-new_ts)*decay_map)/tau)*decay_map 
        
        # Update surfs and tables. 
        surfs[event,:] = surface[:]
        
    return surfs

def learn(dataset, surf_dim, res_x, res_y, tau, n_clusters, n_pol,
          num_batches, n_jobs):
    
    """ 
    This function is used to generate a layer of the network and learn the 
    prototypes. 

    Arguments :
        
        dataset: list containing the data_recording for every recording of the 
                 training dataset sorted by label. 
                               
        surf_dim: the lateral dimension of the squared time context used to
                  build the time surface (it needs to be an odd number to have 
                  a integer number of pixels around the position of the 
                  reference event)
                  
        res_x, res_y: the x and y pixel resolution of the dataset. 
        
        tau : temporal coefficient for the exp decay.
           
        num_clusters: number of clusters extracted by the layers using the 
                      minibatch Kmeans. 
        
        n_pol: number of polarities of the data_recording, if =-1, polarity info
               will be discarded (useful for the first layer usually).
               
        num_batches: number of minibatches used by the minibatch kmeans algorithm.
                     (only use it if memory bound)
                     
        n_jobs: Both time surface generation and Kmeans can run on multiple threads.
                It CAN be an higher value than the number of threads, but use less
                if you like multitasking
        
    Returns:
        
        dataset: The input dataset with the output polarities (cluster index) 
                 inferred after training.  
        
        kmeans: output from sklearn MiniBatchKMeans.

    """
       
    
    num_labels = len(dataset)
    
    n_total_events = 0
    max_recording_per_label = max([len(dataset[label]) for label in range(num_labels)])
    n_events_map = np.zeros([num_labels,max_recording_per_label])
    for label in range(num_labels):
        num_recordings_label = len(dataset[label])
        for recording in range(num_recordings_label):
            n_events = len(dataset[label][recording][3])
            n_total_events += n_events
            n_events_map[label,recording] = n_events
    
    
    batch_recording = max_recording_per_label//num_batches
    

    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                          verbose=0,
                          batch_size=100000)
    
    # kmeans = KMeans(n_clusters=n_clusters,
    #                       # random_state=0, 
    #                       verbose=1, max_iter=1000000)
    
    print('Generating Time Surfaces and Clustering')
    start_time = time.time()
    for batch in range(num_batches):
        print("\rProgress: "+str((batch/num_batches)*100)+"%                                  ", end='')
        
        ## TIME SURFACES ##
        if batch == num_batches-1:
            batch_dataset = [dataset[label][batch*batch_recording:] for label in range(num_labels)]
            n_events_batch_map = n_events_map[:,batch*batch_recording:]
            n_batch_events = int(np.sum(n_events_batch_map))
            
        else:
            batch_dataset = [dataset[label][batch*batch_recording:(batch+1)*batch_recording] for label in range(num_labels)]
            n_events_batch_map = n_events_map[:,batch*batch_recording:(batch+1)*batch_recording]
            n_batch_events = int(np.sum(n_events_batch_map))
            

        if n_pol == -1:
            total_surfs = np.zeros([n_batch_events,surf_dim,surf_dim], dtype=np.float16)
        else:
            total_surfs = np.zeros([n_batch_events,n_pol,surf_dim,surf_dim], dtype=np.float16)
            
        for label in range(num_labels):
            num_recordings_label = len(batch_dataset[label])
            events_prev = int(np.sum(n_events_batch_map[:label,:]))
            events_sofar = int(np.sum(n_events_batch_map[:label+1,:]))
            surf_label = Parallel(n_jobs=n_jobs)(delayed(surfaces)(batch_dataset[label][recording], res_x, res_y, surf_dim,
                           tau, n_pol) for recording in range(num_recordings_label))
            
            total_surfs[events_prev:events_sofar] = np.concatenate(surf_label,axis=0)
        
        gc.collect()
        
        
        idx = np.arange(n_batch_events)
        np.random.shuffle(idx)
        ## CLUSTERING ##
        if n_pol == -1:
            total_surfs = total_surfs[idx,:,:]
            total_surfs = total_surfs.reshape([len(total_surfs),surf_dim**2]).astype('float32')
        else:            
            total_surfs = total_surfs[idx,:,:,:]
            total_surfs = total_surfs.reshape([len(total_surfs),n_pol*surf_dim**2]).astype('float32')


        kmeans = kmeans.fit(total_surfs)
        kmeans.distortion_ = (sum(np.min(cdist(total_surfs, kmeans.cluster_centers_,'euclidean'), axis=1)) / total_surfs.shape[0])
        
    print('\rProgress 100%. Completed in: '+ str(time.time()-start_time)+'seconds')     
        
    
    print('Generating Time Surfaces and Infering') 
    start_time = time.time()
    for batch in range(num_batches):
        print("\rProgress: "+str((batch/num_batches)*100)+"%                                  ", end='')
        ## TIME SURFACES ##
        if batch == num_batches-1:
            batch_dataset = [dataset[label][batch*batch_recording:] for label in range(num_labels)]
            n_events_batch_map = n_events_map[:,batch*batch_recording:]
            n_batch_events = int(np.sum(n_events_batch_map))
            
        else:
            batch_dataset = [dataset[label][batch*batch_recording:(batch+1)*batch_recording] for label in range(num_labels)]
            n_events_batch_map = n_events_map[:,batch*batch_recording:(batch+1)*batch_recording]
            n_batch_events = int(np.sum(n_events_batch_map))
            

        if n_pol == -1:
            total_surfs = np.zeros([n_batch_events,surf_dim,surf_dim], dtype=np.float16)
        else:
            total_surfs = np.zeros([n_batch_events,n_pol,surf_dim,surf_dim], dtype=np.float16)
            
        for label in range(num_labels):
            num_recordings_label = len(batch_dataset[label])
            events_prev = int(np.sum(n_events_batch_map[:label,:]))
            events_sofar = int(np.sum(n_events_batch_map[:label+1,:]))
            surf_label = Parallel(n_jobs=n_jobs)(delayed(surfaces)(batch_dataset[label][recording], res_x, res_y, surf_dim,
                           tau, n_pol) for recording in range(num_recordings_label))
            
            total_surfs[events_prev:events_sofar] = np.concatenate(surf_label,axis=0)
        
        gc.collect()
        

        ## INFERING ##
        new_pols = np.zeros(n_batch_events, dtype="uint16")
        if n_pol == -1:
            total_surfs = total_surfs.reshape([len(total_surfs),surf_dim**2]).astype('float32')
        else:            
            total_surfs = total_surfs.reshape([len(total_surfs),n_pol*surf_dim**2]).astype('float32')
        

        new_pols[:] = kmeans.predict(total_surfs)             
        gc.collect()  
        
        # Substiuting pols in the dataset 
        for label in range(num_labels):
            num_recordings_label = len(batch_dataset[label])
            for recording in range(num_recordings_label):
                n_events = len(batch_dataset[label][recording][0])
                events_prev = int(np.sum(n_events_batch_map.flatten()[:label*batch_recording+recording]))
                dataset[label][(batch*batch_recording)+recording][2]=new_pols[events_prev:events_prev+n_events]

            
    print('\rProgress 100%. Completed in: '+ str(time.time()-start_time)+'seconds')   
    
    return dataset, kmeans


def infer(dataset, surf_dim, res_x, res_y, tau, n_pol, kmeans, num_batches,
          n_jobs):
    
    """ 
    This function is used to infer the response of a layer of the network.

    Arguments :
        
        dataset: list containing the data_recording for every recording of the 
                 test dataset sorted by label. 
                               
        surf_dim: the lateral dimension of the squared time context used to
                  build the time surface (it needs to be an odd number to have 
                  a integer number of pixels around the position of the 
                  reference event)
                  
        res_x, res_y: the x and y pixel resolution of the dataset. 
        
        tau : temporal coefficient for the exp decay.
                   
        n_pol: number of polarities of the data_recording, if =-1, polarity info
               will be discarded (useful for the first layer usually).
      
        kmeans: output from sklearn MiniBatchKMeans from learn function.
               
        num_batches: number of minibatches used by the minibatch kmeans algorithm.
                     (only use it if memory bound)
                     
        n_jobs: Both time surface generation and Kmeans can run on multiple threads.
                It CAN be an higher value than the number of threads, but use less
                if you like multitasking
        
    Returns:
        
        dataset: The input dataset with the output polarities (cluster index) 
                 inferred.  
        
    """
    num_labels = len(dataset)
    n_total_events = 0
    max_recording_per_label = max([len(dataset[label]) for label in range(num_labels)])
    n_events_map = np.zeros([num_labels,max_recording_per_label])
    for label in range(num_labels):
        num_recordings_label = len(dataset[label])
        for recording in range(num_recordings_label):
            n_events = len(dataset[label][recording][3])
            n_total_events += n_events
            n_events_map[label,recording] = n_events
    
    
    batch_recording = max_recording_per_label//num_batches
        
    print('Generating Time Surfaces and Infering') 
    start_time = time.time()
    for batch in range(num_batches):
        print("\rProgress: "+str((batch/num_batches)*100)+"%                                  ", end='')
        ## TIME SURFACES ##
        if batch == num_batches-1:
            batch_dataset = [dataset[label][batch*batch_recording:] for label in range(num_labels)]
            n_events_batch_map = n_events_map[:,batch*batch_recording:]
            n_batch_events = int(np.sum(n_events_batch_map))
            
        else:
            batch_dataset = [dataset[label][batch*batch_recording:(batch+1)*batch_recording] for label in range(num_labels)]
            n_events_batch_map = n_events_map[:,batch*batch_recording:(batch+1)*batch_recording]
            n_batch_events = int(np.sum(n_events_batch_map))
            

        if n_pol == -1:
            total_surfs = np.zeros([n_batch_events,surf_dim,surf_dim], dtype=np.float16)
        else:
            total_surfs = np.zeros([n_batch_events,n_pol,surf_dim,surf_dim], dtype=np.float16)
            
        for label in range(num_labels):
            num_recordings_label = len(batch_dataset[label])
            events_prev = int(np.sum(n_events_batch_map[:label,:]))
            events_sofar = int(np.sum(n_events_batch_map[:label+1,:]))
            surf_label = Parallel(n_jobs=n_jobs)(delayed(surfaces)(batch_dataset[label][recording], res_x, res_y, surf_dim,
                            tau, n_pol) for recording in range(num_recordings_label))
            
            total_surfs[events_prev:events_sofar] = np.concatenate(surf_label,axis=0)
        
        gc.collect()
        

        ## INFERING ##
        new_pols = np.zeros(n_batch_events, dtype="uint16")
        if n_pol == -1:
            total_surfs = total_surfs.reshape([len(total_surfs),surf_dim**2]).astype('float32')
        else:            
            total_surfs = total_surfs.reshape([len(total_surfs),n_pol*surf_dim**2]).astype('float32')

        new_pols[:] = kmeans.predict(total_surfs)             
        gc.collect()  
        
        # Sobstiuting pols in the dataset 
        for label in range(num_labels):
            num_recordings_label = len(batch_dataset[label])
            for recording in range(num_recordings_label):
                n_events = len(batch_dataset[label][recording][0])
                events_prev = int(np.sum(n_events_batch_map.flatten()[:label*batch_recording+recording]))
                dataset[label][(batch*batch_recording)+recording][2]=new_pols[events_prev:events_prev+n_events]
    print('\rProgress 100%. Completed in: '+ str(time.time()-start_time)+'seconds')   

    return dataset


def signature_gen(dataset, n_clusters, n_jobs):
    
    """ 
    This function is used generate signatures from the histogram as in the original
    paper and also train a Support Vector Classifier (machine) to compare classifiers.

    Arguments :
        
        dataset: list containing the data_recording for every recording of the 
                 training dataset sorted by label. 
                               
        num_clusters: number of clusters extracted by the layer of the network.
                     
        n_jobs: hist generation can run on multiple threads.
                It CAN be an higher value than the number of threads, but use less
                if you like multitasking
        
    Returns:
        
        signatures: The histogram signatures (a 2d array [labels,num_clusters]) for 
                    training dataset
     
        norm_signatures: The normalized histogram signatures 
                         (a 2d array [labels,num_clusters]) for training dataset
    
        svc: The support vector classifier (sklearn svm) trained on histograms
       
        norm_svc: The support vector classifier (sklearn svm) trained on 
                  normalized histograms

        
    """
    
    def hists_gen(label, n_recordings, pols, n_clusters):
        n_events = len(pols)
        hist = np.array([sum(pols==cluster) for cluster in range(n_clusters)])/n_recordings
        norm_hist = hist/n_events
        
        return hist, norm_hist
    
    n_labels=len(dataset)
    signatures = np.zeros([n_labels, n_clusters])
    norm_signatures = np.zeros([n_labels, n_clusters])
    all_hists=[]
    all_norm_hists=[]
    labels = []
    for label in range(n_labels):
        n_recordings = len(dataset[label])
        hists, norm_hists  =  zip(*Parallel(n_jobs=n_jobs)(delayed(hists_gen)(label,
                                       n_recordings, dataset[label][recording][2],
                                       n_clusters) for recording in range(n_recordings)))
        all_hists += hists
        all_norm_hists += norm_hists
        labels += [label for recording in range(n_recordings)]
        signatures[label,:] = sum(hists)
        norm_signatures[label,:] = sum(norm_hists)
        
 
    svc = svm.SVC(decision_function_shape='ovr', kernel='poly')
    svc.fit(all_hists, labels)
    
    norm_svc = svm.SVC(decision_function_shape='ovr', kernel='poly')
    norm_svc.fit(all_norm_hists, labels)
    
    return signatures, norm_signatures, svc, norm_svc


def histogram_accuracy(dataset, n_clusters, signatures, norm_signatures, n_jobs):
    """ 
    This function is used to test the histogram classifier accuracy, with both
    signatures and normalized signatures obtained with signature_gen.
    
    Arguments :
        
        dataset: list containing the data_recording for every recording of the 
                 training dataset sorted by label. 
                               
        num_clusters: number of clusters extracted by the layer of the network.
        
        signatures: The histogram signatures (a 2d array [labels,num_clusters]) for 
                    training dataset
     
        norm_signatures: The normalized histogram signatures 
                         (a 2d array [labels,num_clusters]) for training dataset
                     
        n_jobs: hist generation can run on multiple threads.
                It CAN be an higher value than the number of threads, but use less
                if you like multitasking
        
    Returns:
        
    
        test_signatures: The histogram signatures (a 2d array [n_toral_recordings,num_clusters])
                         of the test dataset (per each recording)
     
        test_norm_signatures: The normalized histogram signatures 
                              (a 2d array [n_toral_recordings,num_clusters]) of 
                              the test dataset (per each recording)
                              
        euc_accuracy: percent of correctly guessed recordings of test set using 
                      uclidean distance of histograms
                      
        euc_accuracy: percent of correctly guessed recordings of test set using 
                      uclidean distance of normalized histograms

        euc_labels: predicted labels of test set using uclidean distance of
                    histograms

        norm_euc_labels: predicted labels of test set using uclidean distance of
                         normalized histograms
                    
    """
    
    def hists_gen(label, n_recordings, pols, n_clusters, signatures, norm_signatures):
        n_events = len(pols)
        hist = np.array([sum(pols==cluster) for cluster in range(n_clusters)])
        norm_hist = hist/n_events
        euc_label=np.argmin(np.linalg.norm(signatures-hist, axis=1))
        norm_euc_label=np.argmin(np.linalg.norm(norm_signatures-norm_hist, axis=1))
        
        return hist, norm_hist, euc_label, norm_euc_label
        
    
    n_labels=len(dataset)
    n_toral_recordings = sum([len(dataset[label]) for label in range(n_labels)])
    test_signatures = np.zeros([n_toral_recordings, n_clusters])
    test_norm_signatures = np.zeros([n_toral_recordings, n_clusters])
    recording_idx = 0
    euc_labels = np.zeros(n_toral_recordings)
    norm_euc_labels = np.zeros(n_toral_recordings)
    euc_accuracy = 0
    norm_euc_accuracy = 0
    
    for label in range(n_labels):
        n_recordings = len(dataset[label])
        hists, norm_hists, rec_euc_label, rec_norm_euc_label  =  zip(*Parallel(n_jobs=n_jobs)(delayed(hists_gen)(label,
                                                                     n_recordings, dataset[label][recording][2],
                                                                     n_clusters, signatures, norm_signatures) for recording in range(n_recordings)))
        test_signatures[recording_idx:recording_idx+n_recordings,:] = np.asarray(hists)
        test_norm_signatures[recording_idx:recording_idx+n_recordings,:] = np.asarray(norm_hists)
        euc_labels[recording_idx:recording_idx+n_recordings] = np.asarray(rec_euc_label)
        norm_euc_labels[recording_idx:recording_idx+n_recordings] = np.asarray(rec_norm_euc_label)
        
        euc_accuracy += sum(np.asarray(rec_euc_label)==label)*(100/n_toral_recordings)
        norm_euc_accuracy += sum(np.asarray(rec_norm_euc_label)==label)*(100/n_toral_recordings)
        
        recording_idx+=n_recordings
        
            
    return test_signatures, test_norm_signatures, euc_accuracy, norm_euc_accuracy, euc_labels, norm_euc_labels

def recon_rates_svm(svc, norm_svc, test_signatures, test_norm_signatures, test_set):
    
    """ 
    This function is used to test the SVC (support vector classifier) accuracy, 
    with both signatures and normalized signatures obtained with the histogram_accuracy
    (it takes some time to generate them so I only calculate them once).
    
    Arguments :
        
        svc: The support vector classifier (sklearn svm) trained on histograms
        
        norm_svc: The support vector classifier (sklearn svm) trained on 
                  normalized histograms                        
        
        test_signatures: The histogram signatures (a 2d array [n_toral_recordings,num_clusters]) 
                         of the test dataset (per each recording)
     
        test_norm_signatures: The normalized histogram signatures 
                              (a 2d array [n_toral_recordings,num_clusters]) of 
                              the test dataset (per each recording)
                     
        test_set: list containing the data_recording for every recording of the 
                 test dataset sorted by label. 
        
    Returns:
        
                             
        rec_rate_svc: percent of correctly guessed recordings of test set using 
                      the SVC on histograms
                      
        rec_rate_norm_svc: percent of correctly guessed recordings of test set using 
                           the SVC on normalized histograms

                   
    """
    svc_labels = svc.predict(test_signatures)
    norm_svc_labels = norm_svc.predict(test_norm_signatures)
    rec_rate_svc = 0
    rec_rate_norm_svc = 0
    label_idx = 0
    for label in range(len(test_set)):
        for recording in range(len(test_set[label])):
            if svc_labels[label_idx]==label:
                rec_rate_svc += 1/len(svc_labels)*100
            if norm_svc_labels[label_idx]==label:
                rec_rate_norm_svc += 1/len(svc_labels)*100
            label_idx+=1
    return rec_rate_svc, rec_rate_norm_svc


def n_mnist_rearranging(dataset):
    """
    A function used to re-arrange n-mnist to a format more fitting for hots
    calculation.
    
    Arguments:
        
        dataset: original n-mnist dataset
    
    Returns:
        
        rearranged_dataset: list containing the data_recording for every
                            recording of the dataset sorted by label. 
                            Data of a single recording is a list of 4 arrays [x, y, p, t]
                            containing spatial coordinates of events (x,y), 
                            polarities (p) and timestamps (t).   
    """
    rearranged_dataset = []
    n_labels = len(dataset)
    for label in range(n_labels):
        n_recordings = len(dataset[label][0][0])
        dataset_recording = []
        for recording in range(n_recordings):
            x = dataset[label][0][0][recording][0]-1 # pixel index starts from 1 in N-MNIST
            y = dataset[label][0][0][recording][1]-1 # pixel index starts from 1 in N-MNIST
            p = dataset[label][0][0][recording][2]-1 # polarity index starts from 1 in N-MNIST
            ts = dataset[label][0][0][recording][3]
            dataset_recording.append([x,y,p,ts])
        rearranged_dataset.append(dataset_recording)
        
    return rearranged_dataset


def dataset_resize(dataset,res_x,res_y):
    """
    A function used to cut edge pixels of n-mnist and reduce its size
    
    Arguments:
        
        dataset: list containing the data_recording for every
                 recording of the dataset sorted by label. 
                 
        res_x, res_y: the x and y pixel resolution of the dataset. 
    
    Returns:
        
        dataset: list containing the data_recording for every
                 recording of the dataset sorted by label, cut to res_x,res_y. 

    """
    for label in range(len(dataset)):
       for recording in range((len(dataset[label]))):
           idx = dataset[label][recording][0]<res_x
           idy = dataset[label][recording][1]<res_y
           dataset[label][recording][0] = dataset[label][recording][0][idx*idy]
           dataset[label][recording][1] = dataset[label][recording][1][idx*idy]
           dataset[label][recording][2] = dataset[label][recording][2][idx*idy]
           dataset[label][recording][3] = dataset[label][recording][3][idx*idy]
   
    return dataset

def spac_downsample(dataset, ldim):
    """
    A function used to spatially undersample the dataset, it takes x and y 
    coordinates of events and devides them by ldim to the closest integer (floor 
    operation). This allows to increase spatial integration without scaling up 
    the dimensionality of each layer (memrory intensive for clustering)
                                                                       
    Arguments:
        
        dataset: list containing the data_recording for every
                 recording of the dataset sorted by label. 
                 
        ldim: the downsampling factor.
    
    Returns:
        
        dataset: list containing the data_recording for every
                 recording of the dataset sorted by label, scaled by ldim. 

    """
    for label in range(len(dataset)):
        for recording in range((len(dataset[label]))):
            dataset[label][recording][0]=dataset[label][recording][0]//ldim
            dataset[label][recording][1]=dataset[label][recording][1]//ldim
            
    return dataset



