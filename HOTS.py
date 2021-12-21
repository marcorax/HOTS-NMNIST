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
import random, gc, pickle, copy
from joblib import Parallel, delayed
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import seaborn as sns


# Plotting settings
sns.set(style="white")
plt.style.use("dark_background")


from Libs.HOTSLib import n_mnist_rearranging, learn, infer, signature_gen,\
                 histogram_accuracy, dataset_resize,spac_downsample,recon_rates_svm,\
                 surfaces, fb_surfaces

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

train_set_orig = dataset_resize(train_set_orig,res_x,res_y)
test_set_orig = dataset_resize(test_set_orig,res_x,res_y)



# Network parameters
layers = 1
surf_dim = [7,3]#lateral dimension of surfaces
n_clusters = [32,512]
n_jobs = 21  
n_pol = [-1,32]#input polarities of each layer (if -1 polarity is discarded.)
n_batches=[1,1]#batches of data for minibatchkmeans
n_batches_test=[1,1]
u=7 #Spatial downsample factor
n_runs = 1 # run the code multiple times on reshuffled data to better assest performance
seeds = [1,2,3,4,5]

# HOTS tau for first and second layer.
tau = [5000,92000]


#%%% BENCH HOTS (Standard Kmeans)



H_kmeansss = [] #save the networks layer for every run
H_eucl_res = [] #save the networks layer for every run
H_eucl_norm_res = [] #save the networks layer for every run
H_svc_res = [] #save the networks layer for every run
H_svc_norm_res = [] #save the networks layer for every run


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
    H_eucl_res.append(run_euc_res)
    H_eucl_norm_res.append(run_norm_res)
    H_svc_res.append(run_svc_res)
    H_svc_norm_res.append(run_svc_norm_res)

    
H_clusters = []
for run in range(n_runs):
    H_clusters_run = []
    for layer in range(layers):        
        H_clusters_run.append(np.moveaxis(np.reshape(H_kmeansss[0][0].cluster_centers_, (n_clusters[0], surf_dim[0], surf_dim[0])), 0, 2))
    H_clusters.append(H_clusters_run)

#%% Save run (Uncomment all code to save)
filename='Results/HOTS results/1Lay1run5000_32.pkl'
with open(filename, 'wb') as f: 
    pickle.dump([H_clusters, H_eucl_res, H_eucl_norm_res, H_svc_res, H_svc_norm_res], f)


#%% Load previous results 
filename='Results/HOTS results/1Lay1run5000_32.pkl'
with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
    H_clusters, H_eucl_res, H_eucl_norm_res, H_svc_res, H_svc_norm_res = pickle.load(f)

#%% Results (best classifier "svc"):
#Layer 1 mean:
H1=np.mean(np.array(H_svc_res)[:,0])
#Layer 2 mean:
H2=np.mean(np.array(H_svc_res)[:,1])

#Layer 1 Standard Deviation:
H1_sd=np.std(np.array(H_svc_res)[:,0])
#Layer 2 Standard Deviations:
H2_sd=np.std(np.array(H_svc_res)[:,1])



print("Layer1 HOTS: "+str(H1)+"+-"+str(H1_sd)) #Mean result +- std   
print("Layer2 HOTS: "+str(H2)+"+-"+str(H2_sd)) #Mean result +- std 


#%% Create the time surfaces of the train_set first layer
layer = 0
train_surfs_0 = []
for label in range(10):
    train_surfs_0_label = Parallel(n_jobs=n_jobs)(delayed(surfaces)(train_set_orig[label][recording], res_x, res_y, surf_dim[layer],
                                tau[layer], n_pol[layer]) for recording in range(files_dataset_train))
    train_surfs_0.append(train_surfs_0_label)
    
    
#%% Create the time surfaces of the test_set first layer
layer = 0
test_surfs_0 = []
for label in range(10):
    test_surfs_0_label = Parallel(n_jobs=n_jobs)(delayed(surfaces)(test_set_orig[label][recording], res_x, res_y, surf_dim[layer],
                                tau[layer], n_pol[layer]) for recording in range(files_dataset_test))
    test_surfs_0.append(test_surfs_0_label)
    
#%% New Learning rule (under work)
weights_0 = np.random.rand(surf_dim[0], surf_dim[0], n_clusters[0])
weights_1 = np.random.rand(n_clusters[0], surf_dim[1], surf_dim[1], 10) #classifier

lrate = 0.05

# label=0
# recording=0
random_rec_pick=np.mgrid[:10, :files_dataset_train].reshape(2,-1).T
np.random.shuffle(random_rec_pick)
rel_accuracy = [ ]
train_net_response_0 = copy.deepcopy(train_set_orig)

for epoch in range(100):
    progress=0
    for label,recording in random_rec_pick:
        rec_distances_0=np.sum((train_surfs_0[label][recording][:,:,:,None]-weights_0[None,:,:,:])**2,axis=(1,2))
        rec_closest_0=np.argmin(rec_distances_0,axis=1)
        rec_closest_0_one_hot = np.zeros([len(rec_closest_0),n_clusters[0]])
        rec_closest_0_one_hot[np.arange(len(rec_closest_0)),rec_closest_0]=1
        u_sampled_x=train_set_orig[label][recording][0]//u
        u_sampled_y=train_set_orig[label][recording][1]//u
        train_surfs_1_recording=surfaces([u_sampled_x, u_sampled_y, rec_closest_0, train_set_orig[label][recording][3]], res_x//u, res_y//u, surf_dim[layer+1],\
                                        tau[layer+1], n_pol[layer+1])
        
        train_net_response_0[label][recording][2] = rec_closest_0
        
        rec_distances_1=np.sum((train_surfs_1_recording[:,:,:,:,None]-weights_1[None,:,:,:,:])**2,axis=(1,2,3))
        rec_closest_1=np.argmin(rec_distances_1,axis=1)
        train_surfs_1_recording_fb=fb_surfaces([u_sampled_x, u_sampled_y, rec_closest_1, train_set_orig[label][recording][3]], 10,\
                                        tau[layer])
    
        elem_distances_1 = (train_surfs_1_recording[:,:,:,:]-weights_1[None,:,:,:,label])
        weights_1[:,:,:,label]+=lrate*np.mean(elem_distances_1[:],axis=0)
       
        
        y_som=train_surfs_1_recording_fb[:,label]-np.sum((train_surfs_1_recording_fb[:,np.arange(10)!=label]/(10-1)),axis=1) #normalized by activation
        y_corr=y_som*(y_som>0)
        elem_distances_0 = (train_surfs_0[label][recording][:,:,:,None]-weights_0[None,:,:,:])
        # Keep only the distances for winners
        elem_distances_0=elem_distances_0[:,:,:,:]*rec_closest_0_one_hot[:,None,None,:]
        weights_0[:,:,:]+=lrate*(np.sum(y_corr[:,None,None,None]*elem_distances_0[:],axis=0)/(np.sum(rec_closest_0_one_hot*y_corr[:,None],axis=0)+1))
        
                

        
    
        rec_closest_1_one_hot = np.zeros([len(rec_closest_1),10])
        rec_closest_1_one_hot[np.arange(len(rec_closest_1)),rec_closest_1]=1
        class_rate=np.sum(rec_closest_1_one_hot,axis=0)
            
        progress+=1/len(random_rec_pick)
        if np.argmax(class_rate)==label:
            result = "Correct"
        else:
            result = "Wrong"
            
        
            
        print("Epoch "+str(epoch)+"  Progress: "+str(progress*100)+"%   Relative Accuracy: "+ str(class_rate[label]-np.max(class_rate[np.arange(10)!=label])))
        print("Prediction: "+result+str(label))

#%% Testing

test_net_response_0 = copy.deepcopy(test_set_orig)


Accuracy=0
for label in range(10):
    for recording in range(files_dataset_test):
        rec_distances_0=np.sum((test_surfs_0[label][recording][:,:,:,None]-weights_0[None,:,:,:])**2,axis=(1,2))
        rec_closest_0=np.argmin(rec_distances_0,axis=1)
        u_sampled_x=test_set_orig[label][recording][0]//u
        u_sampled_y=test_set_orig[label][recording][1]//u
        test_surfs_1_recording=surfaces([u_sampled_x, u_sampled_y, rec_closest_0, test_set_orig[label][recording][3]], res_x//u, res_y//u, surf_dim[layer+1],\
                                        tau[layer+1], n_pol[layer+1])
        test_net_response_0[label][recording][2]=rec_closest_0
        rec_distances_1=np.sum((test_surfs_1_recording[:,:,:,:,None]-weights_1[None,:,:,:,:])**2,axis=(1,2,3))
        rec_closest_1=np.argmin(rec_distances_1,axis=1)
        rec_closest_1_one_hot = np.zeros([len(rec_closest_1),10])
        rec_closest_1_one_hot[np.arange(len(rec_closest_1)),rec_closest_1]=1
        class_rate=np.sum(rec_closest_1_one_hot,axis=0)
        if np.argmax(class_rate)==label:
            Accuracy+=1/(files_dataset_test*10)
            
print("relative accuracy = "+str(Accuracy))

#%% Save new learning rule results (Uncomment all code to save)
filename='Results/New L results/1Lay1run5000_32.pkl'
with open(filename, 'wb') as f: 
    pickle.dump([weights_0, weights_1, lrate, Accuracy], f)


#%% Load previous results 
filename='Results/New L results/1Lay1run5000_32.pkl'
with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
    weights_0, weights_1, lrate, Accuracy = pickle.load(f)

#%% Histogram and SVC on classes
layer=0

print('SIGNATURE GENERATION')
[signatures, norm_signatures, svc, norm_svc] = signature_gen(train_net_response_0, n_clusters[layer], n_jobs)

print('TESTING')
[test_signatures, test_norm_signatures,
  euc_accuracy, norm_euc_accuracy,
  euc_label, norm_euc_label] = histogram_accuracy(test_net_response_0, n_clusters[layer], signatures,
                                                  norm_signatures, n_jobs)


rec_rate_svc,rec_rate_norm_svc = recon_rates_svm(svc,norm_svc,test_signatures,test_norm_signatures, test_net_response_0)


print('Euclidean accuracy: '+str(euc_accuracy)+'%')   
print('Normalized euclidean accuracy: '+str(norm_euc_accuracy)+'%')       
print('Svc accuracy: '+str(rec_rate_svc)+'%')   
print('Normalized Svc accuracy: '+str(rec_rate_norm_svc)+'%')                                               
gc.collect()


#%% t-SNE to look at the evolution of surfaces. 
label = 0
recording=2

surfs=test_surfs_0# Set the surfs you want to plot

# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):

    plt.figure()
    plt.scatter(X[:,0],X[:,1], c=y)    
    if title is not None:
        plt.title(title)
        plt.colorbar()
        
recording_timestamps=test_set_orig[label][recording][3]/1.0
recording_x=test_set_orig[label][recording][0]/1.0
recording_y=test_set_orig[label][recording][1]/1.0

recording_surfaces = surfs[layer][recording]

recording_surfaces_flat = np.reshape(recording_surfaces, [len(recording_surfaces), surf_dim[layer]**2])
#For CNN handling i need to add center of tile
recording_surfaces_flat = np.insert(recording_surfaces_flat,surf_dim[layer]**2,recording_x,axis=1)
recording_surfaces_flat = np.insert(recording_surfaces_flat,surf_dim[layer]**2+1,recording_x,axis=1)


tsne = TSNE(n_components=2, init='pca', random_state=0)
embedd_recording=tsne.fit_transform(recording_surfaces_flat)

plot_embedding(embedd_recording, recording_timestamps, "t-SNE timecoded surfaces total")

l_index=0
up_index=500
plot_embedding(embedd_recording[l_index:up_index], recording_timestamps[l_index:up_index], "t-SNE timecoded surfaces timewindow")

#%% Plot Centroids
New_L_features = weights_0
Kmeans_features = H_clusters[0][0]

nrows = 4
ncols = int(np.ceil(n_clusters[0]/nrows))
fig, axs = plt.subplots(nrows, ncols)

for feat in range(n_clusters[0]):
    axs[(feat//ncols)-1, feat%ncols].imshow(Kmeans_features[:,:,feat] )
fig.suptitle("Kmeans Features")


fig, axs = plt.subplots(nrows, ncols)

for feat in range(n_clusters[0]):
    axs[(feat//ncols)-1, feat%ncols].imshow(New_L_features[:,:,feat] )
fig.suptitle("New L Features")