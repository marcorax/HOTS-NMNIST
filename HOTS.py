#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:28:11 2020

@author: marcorax93

Script used for HOTS on N-MNIST,
this version uses batched-kmeans as a clustering algorihm, a subsampling layer
and two different classifiers (normalized histograms distance as in the original paper,
and a support vector machine trained on the histograms)
 
"""

from scipy import io
import numpy as np
import random, gc, pickle

from Libs.HOTSLib import n_mnist_rearranging, learn, infer, signature_gen,\
                 histogram_accuracy, dataset_resize,spac_downsample,recon_rates_svm

#%% Data loading and parameters setting
            
## Data loading
train_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/train_set.mat')['train_set'])
test_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/test_set.mat')['test_set'])
n_recording_labels_train=[len(train_set_orig[label]) for label in range(len(train_set_orig))]
n_recording_labels_test=[len(test_set_orig[label]) for label in range(len(train_set_orig))]

# using a subset of N-MNIST to lower memory usage
files_dataset_train = min(n_recording_labels_train)//10
files_dataset_test = min(n_recording_labels_test)//10
num_labels = len(test_set_orig)

# N-MNIST resolution 
res_x = 28
res_y = 28


# Network parameters
layers = 2
surf_dim = [7,3]#lateral dimension of surfaces
n_clusters = [16,512]
n_jobs = 21  
n_pol = [-1,16]#input polarities of each layer (if -1 polarity is discarded.)
n_batches=[1,1]#batches of data for minibatchkmeans
n_batches_test=[1,1]
u=7 #Spatial downsample factor
n_runs = 5 # run the code multiple times on reshuffled data to better assest performance
seeds = [1,2,3,4,5]




#%%% BENCH HOTS

# HOTS tau for first and second layer.
tau = [5000,92000]

H_kmeansss = [] #save the networks layer for every run
H_res = [] #save the networks layer for every run


for run in range(n_runs):
    run_euc_res = []
    run_norm_res = []
    run_svc_res = []
    run_svc_norm_res = []
    run_kmeansss = []
    #Random data shuffling
    train_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/train_set.mat')['train_set'])
    test_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/test_set.mat')['test_set'])
    train_set_orig = dataset_resize(train_set_orig,res_x,res_y)
    test_set_orig = dataset_resize(test_set_orig,res_x,res_y)
    for label in range(num_labels):
        random.Random(seeds[run]).shuffle(train_set_orig[label])
        random.Random(seeds[run]).shuffle(test_set_orig[label])
        
    train_set = [train_set_orig[label][:files_dataset_train] for label in range(num_labels)]
    test_set = [test_set_orig[label][:files_dataset_test] for label in range(num_labels)]
    
    layer_res_x = res_x
    layer_res_y = res_y

    for layer in range(layers):
        print('##################____LAYER_'+str(layer)+'____###################')
        print('TRAIN SET LEARNING')
        [train_set, kmeans] = learn(train_set, surf_dim[layer], layer_res_x,
                                    layer_res_y, tau[layer], n_clusters[layer],
                                    n_pol[layer], n_batches[layer], n_jobs)
        run_kmeansss.append(kmeans)
        train_set=spac_downsample(train_set,u)
        print('TEST SET INFERING')
        test_set = infer(test_set, surf_dim[layer], layer_res_x, layer_res_y,
                         tau[layer], n_pol[layer], kmeans, n_batches_test[layer],
                         n_jobs)
        
        test_set=spac_downsample(test_set,u)
        layer_res_x=layer_res_x//u
        layer_res_y=layer_res_y//u            
        # gc.collect()
        print('SIGNATURE GENERATION')
        [signatures, norm_signatures, svc, norm_svc] = signature_gen(train_set, n_clusters[layer], n_jobs)
        
        print('TESTING')
        [test_signatures, test_norm_signatures,
          euc_accuracy, norm_euc_accuracy,
          euc_label, norm_euc_label] = histogram_accuracy(test_set, n_clusters[layer], signatures,
                                                          norm_signatures, n_jobs)
        
        run_euc_res.append(euc_accuracy)
        run_norm_res.append(norm_euc_accuracy)
        rec_rate_svc,rec_rate_norm_svc = recon_rates_svm(svc,norm_svc,test_signatures,test_norm_signatures, test_set)
        
        run_svc_res.append(rec_rate_svc)
        run_svc_norm_res.append(rec_rate_norm_svc)
        
        print(run)
        print('Euclidean accuracy: '+str(euc_accuracy)+'%')   
        print('Normalized euclidean accuracy: '+str(norm_euc_accuracy)+'%')       
        print('Svc accuracy: '+str(rec_rate_svc)+'%')   
        print('Normalized Svc accuracy: '+str(rec_rate_norm_svc)+'%')                                               
        gc.collect()

       
    H_kmeansss.append(run_kmeansss)   
    H_res.append(run_svc_res)
        

#%% Save run (Uncomment all code to save)
# filename='Results/HOTS results/test_result_new.pkl'
# with open(filename, 'wb') as f: 
#     pickle.dump([H_kmeansss, H_res], f)


#%% Load previous results 
# filename='Results/HOTS results/test_result_new.pkl'
# with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
#     H_kmeansss, H_res = pickle.load(f)

#%% Results:
#Layer 1 mean:
H1=np.mean(np.array(H_res)[:,0])
#Layer 2 mean:
H2=np.mean(np.array(H_res)[:,1])

#Layer 1 Standard Deviation:
H1_sd=np.std(np.array(H_res)[:,0])
#Layer 2 Standard Deviations:
H2_sd=np.std(np.array(H_res)[:,1])



print("Layer1 HOTS: "+str(H1)+"+-"+str(H1_sd)) #Mean result +- std   
print("Layer2 HOTS: "+str(H2)+"+-"+str(H2_sd)) #Mean result +- std 