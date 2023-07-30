# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:07:14 2023

@author: marcorax93
"""

import numpy as np
import pyopencl as cl


from Libs.Dense_Layer.Dense_Layer import Dense_Layer
from Libs.Class_Layer.Class_Layer import Class_Layer
from Libs.Conv_Layer.Conv_Layer import Conv_Layer




def Sup3r_Net():
    def __init__(self, layers=[]):
        
        self.layers = layers
        
    def train(self, pre_train = True):
        
        if pre_train:
            
            self.pre_train()
        
    
    def pre_train(self, X, Y, n_epochs=1, batch_size=128):
        """
        Method used to pretrain the layers of the networks by running kmeans 
        
        """
        
        n_X = len(X)
        n_batches = (n_X//batch_size)+1
        
        for layer_i in range(len(self.layers)):
            for epoch_i in range(n_epochs):
                rec_idx=0
                for batch_i in range(n_batches):     
                    n_events_rec=np.zeros(batch_size, dtype=int)
            
                    for i in range(batch_size):
                        data_events = X[rec_idx+i]
                        n_events_rec[i] = len(data_events[0])
                        
                    n_max_events = max(n_events_rec)
                    net_run_alloc(X, Y, rec_idx, batch_size, n_max_events,
                                  self.layers[layer_i])
                    
                    net_buffers = self.net_buffers
                    #TODO continue here

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

            
    def net_run_alloc(self, X, Y, rec_idx, batch_size, n_max_events, layer):
        """
        Method used to allocate memory on the OCL worker before running 
        a train/test/pre-train method
        """
        mf = cl.mem_flags()
        ctx = self.ctx
        
        xs_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ys_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ps_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        ts_np = -1*np.ones((batch_size,n_max_events),dtype=np.int32)
        batch_labels = np.zeros([batch_size],dtype=np.int32)
        n_events_batch = np.zeros([batch_size],dtype=np.int32)
                
        predicted_ev = -1*np.ones([batch_size, n_max_events],dtype=np.int32)
        
        # fevskip for feed event skip, and bevskip for back event skip, 1=>true 0=>false
        fevskip = np.zeros(batch_size, dtype=np.int32)
        bevskip = np.zeros(batch_size, dtype=np.int32)
        



        
        for i in range(batch_size):
            data_events = X[rec_idx+i]
            n_events = len(data_events[0])
            xs_np[i,:n_events] = data_events[0]
            ys_np[i,:n_events] = data_events[1]
            ps_np[i,:n_events] = data_events[2]
            ts_np[i,:n_events] = data_events[3]
            batch_labels[i] = Y[rec_idx+i]
            n_events_batch[i] = n_events
            
        rec_idx+=batch_size 
        
        # Network Buffers
        xs_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=xs_np)
        ys_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ys_np)
        ps_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ps_np)
        ts_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts_np)
        ev_i_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.int32(0))
        n_events_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_max_events))
        batch_labels_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=batch_labels)
        fevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip)
        bevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip)
        predicted_ev_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=predicted_ev)
        fevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip)
        bevskip_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip)


        
        net_buffers = {"xs_bf" : xs_bf, "ys_bf" : ys_bf, "ps_bf" : ps_bf, 
                       "ts_bf" : ts_bf, "ev_i_bf" : ev_i_bf, 
                       "n_events_bf" : n_events_bf, 
                       "predicted_ev_bf" : predicted_ev_bf,
                       "batch_labels_bf" : batch_labels_bf, 
                       "fevskip_bf" : fevskip_bf, "bevskip_bf" : bevskip_bf}

        self.net_buffers = net_buffers