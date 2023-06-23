#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 08:53:14 2023

@author: marcorax93

Class layer of the neural clustering architecture, derived from Dense Layer

Definition:
    dense_layer(n_clusters, tau, res_x, res_y, n_pol, lrate,
                 th_size, th_lrate, th_decay, ctx, batch_size, debug=False)

Parameters: 
        
    n_clusters : number of clusters for this layer
    
    tau : exponential decay tau for the time surface generation
    
    res_x : horizontal time surface size
    
    res_y : vertical time surface size
    
    n_pol : input polarities 
    
    
    th_size : starting threshold (decision boundary radius) for all clusters
    
    th_lrate : threshold learning rate
    
    th_decay : threshold decay, between 1 and 0
    
    ctx : OpenCL context
    
    batch_size : size of the batch for the opencl execution
    
    debug : boolean variable, if true the program will run additional code
            on host as sanity check.
                
"""

import numpy as np
import pyopencl as cl

mf = cl.mem_flags


class Class_Layer:
    """
    Dense layer of the neural clustering architecture.
    
    Parameters: 
        
        ctx : Open CL context
        
        n_clusters : number of clusters for this layer 
        
        tau : exponential decay tau for the time surface generation
        
        res_x : horizontal time surface size
        
        res_y : vertical time surface size
        
        n_pol : input polarities 
        
        lrate : cluster learning rate        
        
        batch_size : size of the batch for the opencl execution
        
        s_gain : additional learning rate that affects S signals  
        
        fb_signal : If True, the layer calculates the feedback signal S, to instruct the learning of lower layers.

        fb_tau : exponential decay tau for the feedback time surface generation            
                    
        debug : boolean variable, if true the program will run additional code on host as sanity check.
        
        
    """
    
    def __init__(self, n_clusters, tau, res_x, res_y, n_pol, lrate,
                 ctx, batch_size, s_gain, fb_signal=True, fb_tau=None,
                 debug=False):
        
        self.fb_signal = fb_signal
        self.debug = debug
        
        #TODO, once I have a network class, move this there
        # Calculating local sizes for kernel executions
        max_workgroup_size = np.inf
        for dev_i in range(len(ctx.devices)):
            dev_max_workgroup_size = ctx.devices[dev_i].max_work_group_size
            if dev_max_workgroup_size<max_workgroup_size:
                max_workgroup_size = dev_max_workgroup_size
                
        loc_ts_size = res_x*res_y*n_pol # OpenCL local size for kernels operating
                                        # on centroids or time surfaces
        if loc_ts_size>max_workgroup_size:
            loc_ts_size=max_workgroup_size
        self.__loc_ts_size = loc_ts_size
        print(loc_ts_size)

            
        loc_cl_size = n_clusters # OpeCL local size for kernel operating on th
                                 # threshold update
        if loc_cl_size>max_workgroup_size:
            loc_ts_size=max_workgroup_size
        self.__loc_cl_size = loc_cl_size

        
        
        ### Creating human readable parameter dictionary 
        
        param_dict = {"n_clusters" : n_clusters,
                      "tau" : tau,
                      "res_x" : res_x,
                      "res_y" : res_y,
                      "n_pol" : n_pol,
                      "lrate" : lrate,
                      "fb_tau" : fb_tau,
                      "ctx" : ctx,
                      "batch_size" : batch_size,
                      "s_gain" : s_gain}
       
        self.parameters = param_dict
        
        ### Allocating variables 
        
        #bit precision for "weight" calculations (threshold, centroids)
        w_prec = np.float64
        self.__w_prec = w_prec 
        #bit precision for "distance" calculations 
        dist_prec = np.float64
        self.__dist_prec = dist_prec 
        #bit precision for "time surface" calculations 
        ts_prec = np.float32
        self.__ts_prec = ts_prec 
        
        centroids = np.zeros([batch_size, n_clusters, res_x, res_y, n_pol],dtype=w_prec)
        centroids[:] = np.random.rand(n_clusters, res_x, res_y, n_pol)#*0.001
        
        # n_pol_per_cluster = n_pol//n_clusters
        # for i_cluster in range(n_clusters):
        #     centroids[:,i_cluster,:,:,np.arange(0,n_pol_per_cluster)+n_pol_per_cluster*i_cluster]=1
        
        #Used to store the dstep in direction of the new centroid position
        dcentroids = np.zeros([batch_size, n_clusters, res_x, res_y, n_pol],dtype=w_prec)
        centroids_update = np.zeros([batch_size, n_clusters, res_x, res_y, n_pol],dtype=w_prec)

        time_context = np.zeros([batch_size, res_x, res_y, n_pol],dtype=np.int32) #time context matrix        
        time_context_mask = np.zeros([batch_size, res_x, res_y, n_pol],dtype=np.int32)
        
        closest_c = np.zeros([batch_size],dtype=np.int32)
        distances = np.zeros([batch_size,n_clusters],dtype=dist_prec)  
        partial_sum = np.zeros([batch_size,n_clusters,loc_ts_size],dtype=dist_prec) 

        time_surface = np.zeros([batch_size,res_x, res_y, n_pol],dtype=ts_prec)
        
        processed_ev = np.zeros([batch_size],dtype=np.int32)
        correct_ev = np.zeros([batch_size],dtype=np.int32)

        
        if fb_signal:
            output_S = np.zeros([batch_size],dtype=np.float32)
            output_dS = np.zeros([batch_size],dtype=np.float32)
            fb_time_context = np.zeros([batch_size, n_clusters],dtype=np.int32) 
            fb_time_context_mask = np.zeros([batch_size, n_clusters],dtype=np.int32)
            fb_partial_sum = np.zeros([batch_size,n_clusters],dtype=dist_prec) 
            correct_response = np.zeros([batch_size],dtype=np.int32)

        else:
            output_S=None
            output_dS=None
            fb_time_context = None
            fb_time_context_mask = None
            fb_partial_sum = None
            correct_response = None
        
        var_dict = {"centroids" : centroids,
                    "dcentroids" : dcentroids,
                    "centroids_update" : centroids_update,
                    "time_context" : time_context,
                    "time_context_mask" : time_context_mask,
                    "fb_time_context" : fb_time_context,
                    "fb_time_context_mask" : fb_time_context_mask,
                    "closest_c" : closest_c,
                    "distances" : distances, 
                    "partial_sum" : partial_sum,
                    "fb_partial_sum" : fb_partial_sum,
                    "time_surface" : time_surface,
                    "output_S" : output_S,
                    "output_dS" : output_dS,
                    "processed_ev" : processed_ev,
                    "correct_ev" : correct_ev,
                    "correct_response" : correct_response}
        
        self.variables = var_dict
        
        self.buffer_init()
        
        self.program_build()
        
        
        
    def buffer_init(self):
        """
        Method used to copy all the host variables and parameters on the 
        buffers. Useful if any variable is changed outsize kernel execution,
        or to initialize all the buffers.
        """
        
        #TODO move all these parameters into the kernel string
        n_clusters=self.parameters["n_clusters"]
        tau=self.parameters["tau"] 
        fb_tau=self.parameters["fb_tau"] 
        res_x=self.parameters["res_x"] 
        res_y=self.parameters["res_y"] 
        n_pol=self.parameters["n_pol"] 
        lrate=self.parameters["lrate"] 
        s_gain=self.parameters["s_gain"] 
        ctx=self.parameters["ctx"] 

        n_clusters_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_clusters)) 
        tau_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(tau)) 
        res_x_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(res_x)) 
        res_y_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(res_y)) 
        n_pol_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(n_pol)) 
        lrate_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(lrate)) 
        s_gain_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(s_gain)) 

        self.buffers = {"n_clusters_bf" : n_clusters_bf,
                        "tau_bf" : tau_bf,
                        "res_x_bf" : res_x_bf,
                        "res_y_bf" : res_y_bf, 
                        "n_pol_bf" : n_pol_bf,
                        "lrate_bf" : lrate_bf,
                        "s_gain_bf" : s_gain_bf}
        
        centroids  = self.variables["centroids"]
        dcentroids  = self.variables["dcentroids"]
        centroids_update  = self.variables["centroids_update"]
        time_context  = self.variables["time_context"]
        time_context_mask  = self.variables["time_context_mask"]
        closest_c  = self.variables["closest_c"]
        distances  = self.variables["distances"]
        partial_sum  = self.variables["partial_sum"]
        time_surface  = self.variables["time_surface"]
        processed_ev  = self.variables["processed_ev"]
        correct_ev  = self.variables["correct_ev"]
        correct_response = self.variables["correct_response"]

        output_S  = self.variables["output_S"]  
        output_dS  = self.variables["output_dS"]  
        fb_time_context  = self.variables["fb_time_context"]  
        fb_time_context_mask  = self.variables["fb_time_context_mask"]  
        fb_partial_sum  = self.variables["fb_partial_sum"]  

        centroids_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=centroids) 
        dcentroids_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=dcentroids) 
        centroids_update_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=centroids_update) 
        time_context_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=time_context) 
        time_context_mask_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=time_context_mask) 
        closest_c_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=closest_c) 
        distances_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=distances) 
        partial_sum_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=partial_sum) 
        time_surface_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=time_surface) 
        processed_ev_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=processed_ev) 
        correct_ev_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=correct_ev) 

        if self.fb_signal:
            output_S_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=output_S) 
            output_dS_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=output_dS) 
            fb_time_context_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fb_time_context) 
            fb_time_context_mask_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fb_time_context_mask) 
            fb_partial_sum_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fb_partial_sum) 
            fb_tau_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(fb_tau)) 
            correct_response_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=correct_response) 
        else:
            output_S_bf=None
            output_dS_bf=None
            fb_time_context_bf=None
            fb_time_context_mask_bf=None
            fb_partial_sum_bf=None
            fb_tau_bf=None
            correct_response_bf=None
        
        
        self.buffers["centroids_bf"] = centroids_bf
        self.buffers["dcentroids_bf"] = dcentroids_bf
        self.buffers["centroids_update_bf"] = centroids_update_bf
        self.buffers["time_context_bf"] = time_context_bf
        self.buffers["time_context_mask_bf"] = time_context_mask_bf
        self.buffers["closest_c_bf"] = closest_c_bf
        self.buffers["distances_bf"] = distances_bf
        self.buffers["partial_sum_bf"] = partial_sum_bf
        self.buffers["time_surface_bf"] = time_surface_bf
        self.buffers["output_S_bf"] = output_S_bf
        self.buffers["output_dS_bf"] = output_dS_bf
        self.buffers["fb_time_context_bf"] = fb_time_context_bf
        self.buffers["fb_time_context_mask_bf"] = fb_time_context_mask_bf
        self.buffers["fb_partial_sum_bf"] = fb_partial_sum_bf
        self.buffers["fb_tau_bf"] = fb_tau_bf
        self.buffers["processed_ev_bf"] = processed_ev_bf
        self.buffers["correct_ev_bf"] = correct_ev_bf
        self.buffers["correct_response_bf"] = correct_response_bf


                
        
    def batch_flush(self, queue):
        """
        This method is used to reset some variables and buffers after a
        batch run, this is fundamental to avoid contamination between
        consequent batches.
        
        Parameters:
            
            queue : OpenCL queue 
            
        """
        res_x=self.parameters["res_x"] 
        res_y=self.parameters["res_y"] 
        n_pol=self.parameters["n_pol"] 
        batch_size=self.parameters["batch_size"] 
        n_clusters=self.parameters["n_clusters"]
        
        if self.fb_signal:
            output_S = np.zeros([batch_size],dtype=np.float32)
            output_dS = np.zeros([batch_size],dtype=np.float32)
            fb_time_context = np.zeros([batch_size, n_clusters],dtype=np.int32)     
            fb_time_context_mask = np.zeros([batch_size, n_clusters],dtype=np.int32)
            correct_response=np.zeros([batch_size],dtype=np.int32)
            self.variables["output_S"] = output_S
            self.variables["output_dS"] = output_dS
            self.variables["fb_time_context"] = fb_time_context
            self.variables["fb_time_context_mask"] = fb_time_context_mask
            self.variables["correct_response"] = correct_response
        
        
        time_context = np.zeros([batch_size, res_x, res_y, n_pol],dtype=np.int32) #time context matrix        
        time_context_mask = np.zeros([batch_size, res_x, res_y, n_pol],dtype=np.int32)
        processed_ev = np.zeros([batch_size],dtype=np.int32)
        correct_ev = np.zeros([batch_size],dtype=np.int32)
        centroids_update = np.zeros([batch_size, n_clusters, res_x, res_y, n_pol],dtype=self.__w_prec)
        self.variables["time_context"] = time_context
        self.variables["time_context_mask"] = time_context_mask
        self.variables["processed_ev"] = processed_ev
        self.variables["correct_ev"] = correct_ev
        self.variables["centroids_update"] = centroids_update
        
              
        if self.fb_signal:
            output_S_bf = self.buffers["output_S_bf"] 
            output_dS_bf = self.buffers["output_dS_bf"] 
            fb_time_context_bf = self.buffers["fb_time_context_bf"] 
            fb_time_context_mask_bf = self.buffers["fb_time_context_mask_bf"] 
            correct_response_bf = self.buffers["correct_response_bf"]
            cl.enqueue_copy(queue, output_S_bf, output_S) 
            cl.enqueue_copy(queue, output_dS_bf, output_dS) 
            cl.enqueue_copy(queue, fb_time_context_bf, fb_time_context) 
            cl.enqueue_copy(queue, fb_time_context_mask_bf, fb_time_context_mask)
            cl.enqueue_copy(queue, correct_response_bf, correct_response).wait()
            self.buffers["output_S_bf"] = output_S_bf
            self.buffers["output_dS_bf"] = output_dS_bf
            self.buffers["fb_time_context_bf"] = fb_time_context_bf
            self.buffers["fb_time_context_mask_bf"] = fb_time_context_mask_bf
            self.buffers["correct_response_bf"] = correct_response_bf

        
        time_context_bf = self.buffers["time_context_bf"]
        time_context_mask_bf = self.buffers["time_context_mask_bf"]    
        processed_ev_bf = self.buffers["processed_ev_bf"]            
        correct_ev_bf = self.buffers["correct_ev_bf"]     
        centroids_update_bf = self.buffers["centroids_update_bf"]                                   
        cl.enqueue_copy(queue, time_context_bf, time_context) 
        cl.enqueue_copy(queue, time_context_mask_bf, time_context_mask) 
        cl.enqueue_copy(queue, processed_ev_bf, processed_ev)      
        cl.enqueue_copy(queue, correct_ev_bf, correct_ev).wait()                   
        cl.enqueue_copy(queue, centroids_update_bf, centroids_update).wait()                   
        self.buffers["time_context_bf"] = time_context_bf
        self.buffers["time_context_mask_bf"] = time_context_mask_bf
        self.buffers["processed_ev_bf"] = processed_ev_bf
        self.buffers["correct_ev_bf"] = correct_ev_bf
        self.buffers["centroids_update_bf"] = centroids_update_bf

             
    def batch_update(self, queue):
        """
        This method is used to update variables after a batch run.
        
        Parameters:
            
            queue : OpenCL queue                          

        """
        
        centroids=self.variables["centroids"]
        centroids_update=self.variables["centroids_update"]
        processed_ev = self.variables["processed_ev"]
        correct_ev = self.variables["correct_ev"]

        centroids_bf = self.buffers["centroids_bf"]
        centroids_update_bf=self.buffers["centroids_update_bf"]
        processed_ev_bf = self.buffers["processed_ev_bf"]
        correct_ev_bf = self.buffers["correct_ev_bf"]
        
       
        cl.enqueue_copy(queue, centroids, centroids_bf).wait()
        cl.enqueue_copy(queue, centroids_update, centroids_update_bf).wait()
        cl.enqueue_copy(queue, processed_ev, processed_ev_bf).wait()
        cl.enqueue_copy(queue, correct_ev, correct_ev_bf).wait()
    
        centroids[:] += np.mean(centroids_update, axis=0)
        cl.enqueue_copy(queue, centroids_bf, centroids).wait()
        
        self.variables["centroids"]=centroids
        self.variables["processed_ev"]=processed_ev
        self.variables["correct_ev"]=correct_ev

    def batch_update_online(self, queue):
        """
        This method is used to update variables after a batch run.
        
        Parameters:
            
            queue : OpenCL queue                          

        """
        
        centroids=self.variables["centroids"]
        processed_ev = self.variables["processed_ev"]
        correct_ev = self.variables["correct_ev"]

        centroids_bf = self.buffers["centroids_bf"]
        processed_ev_bf = self.buffers["processed_ev_bf"]
        correct_ev_bf = self.buffers["correct_ev_bf"]
       
        cl.enqueue_copy(queue, centroids, centroids_bf).wait()
        cl.enqueue_copy(queue, processed_ev, processed_ev_bf).wait()
        cl.enqueue_copy(queue, correct_ev, correct_ev_bf).wait()
    
        centroids[:] = np.mean(centroids, axis=0)
        cl.enqueue_copy(queue, centroids_bf, centroids).wait()
        
        self.variables["centroids"]=centroids
        self.variables["processed_ev"]=processed_ev
        self.variables["correct_ev"]=correct_ev

        
    def host_copy(self):
        """
        Method used to copy the buffers back to the host.
        Used to check results after kernel execution
        """
        robe = 1
        
    def centroid_buffer_copy(self, queue):
        """
        Method used to copy the host to the buffers.
        Used to update the centroids and thresholds after a batch execution.
        
        Parameters :
            
            queue : OpenCL queue to enque the copy
        
        """
        cl.enqueue_copy(queue, self.variables["centroids"],
                        self.buffers["centroids_bf"])

    def program_build(self):
        """
        Method used to build the program with all the layer kernels.
        """
        
        ctx = self.parameters["ctx"]
        
        dr = "Libs/Class_Layer/infer_cl/"
        
        f = open(dr+"context_update.cl", 'r')
        tcont_update = "".join(f.readlines())
        f = open(dr+"ts_gen.cl", 'r')
        ts_gen = "".join(f.readlines())
        f = open(dr+"partial_dist.cl", 'r')
        partial_dist = "".join(f.readlines())
        f = open(dr+"reduce_dist.cl", 'r')
        reduce_dist = "".join(f.readlines())
        f = open(dr+"infer_end.cl", 'r')
        infer_end = "".join(f.readlines())
        
        programtxt = tcont_update+ts_gen+partial_dist+reduce_dist+infer_end

        
        dr = "Libs/Class_Layer/learn_cl/"
        f = open(dr+"w_update.cl", 'r')
        w_update = "".join(f.readlines())

        
        programtxt = programtxt + w_update 
        
        dr = "Libs/Class_Layer/feedback_cl/"
        f = open(dr+"fb_context_update.cl", 'r')
        fb_context_update = "".join(f.readlines())
        f = open(dr+"fb_ts_gen.cl", 'r')
        fb_ts_gen = "".join(f.readlines())
        f = open(dr+"fb_end.cl", 'r')
        fb_end = "".join(f.readlines())
        
        programtxt = programtxt + fb_context_update + fb_ts_gen + fb_end      
        
        dr = "Libs/Dense_Layer/exp_cl/"
        f = open(dr+"w_reduce.cl", 'r')
        w_reduce = "".join(f.readlines())

        programtxt = programtxt + w_reduce
        
        dr = "Libs/Class_Layer/init_cl/"
        f = open(dr+"init_ts_gen.cl", 'r')
        init_ts_gen = "".join(f.readlines())
        
        dr = "Libs/Class_Layer/init_cl/"
        f = open(dr+"init_infer_end.cl", 'r')
        init_infer_end = "".join(f.readlines())

        dr = "Libs/Class_Layer/init_cl/"
        f = open(dr+"init_w_update.cl", 'r')
        init_w_update = "".join(f.readlines())

        programtxt = programtxt + init_ts_gen + init_infer_end + init_w_update
                
        self.program=cl.Program(ctx, programtxt).build(options='-cl-std=CL2.0')
        
    def infer(self, ext_buffer, queue):
        """
        Method to queue the kernels for infering.
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        self.queue_context_update(ext_buffer, queue)
        self.queue_time_surface_generation(ext_buffer, queue)
        self.queue_partial_distances(ext_buffer, queue)
        self.queue_reduction_distances(ext_buffer, queue)
        self.queue_infer_end(ext_buffer, queue)
        
        
    def learn(self, ext_buffer, queue):
        """
        Method to queue the kernels for learning.
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        if self.fb_signal:
            self.queue_feedback_context_update(ext_buffer, queue)
            self.queue_feedback_time_surface_generation(ext_buffer, queue)
            self.queue_feedback_end(ext_buffer, queue)
            
        self.queue_weight_update(ext_buffer, queue)
        # self.queue_experimental_weight_reduce(ext_buffer, queue)
        
    def init_infer(self, ext_buffer, queue, time_surface_bf):
            """
            Method to queue the kernels for infering.
            
            Parameters:
                
                ext_buffer : dictionary containing the required buffers for kernel
                             execution that could not being generated during the 
                             layer initialization.
                
                queue : OpenCL queue                          
                
            """
            self.queue_context_update(ext_buffer, queue)
            self.queue_init_time_surface_generation(ext_buffer, queue, time_surface_bf)

            
    def init_learn(self, ext_buffer, queue):
        """
        Method to queue the kernels for learning.
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        if self.fb_signal:
            self.queue_feedback_context_update(ext_buffer, queue)
            self.queue_feedback_time_surface_generation(ext_buffer, queue)
            self.queue_feedback_end(ext_buffer, queue)
            
        # self.queue_init_weight_update(ext_buffer, queue)
        self.queue_weight_update(ext_buffer, queue)

        # self.queue_experimental_weight_reduce(ext_buffer, queue) 
        
    def queue_context_update(self, ext_buffer, queue):
        """
        Method to queue the context_update kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        
        xs_bf = ext_buffer["xs_bf"]
        ys_bf = ext_buffer["ys_bf"]
        ps_bf = ext_buffer["ps_bf"]
        ts_bf = ext_buffer["ts_bf"]        
        res_x_bf = self.buffers["res_x_bf"]
        res_y_bf = self.buffers["res_y_bf"]
        n_pol_bf = self.buffers["n_pol_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]      
        time_context_bf = self.buffers["time_context_bf"]
        time_context_mask_bf = self.buffers["time_context_mask_bf"]    
        fevskip_bf = ext_buffer["fevskip_bf"] 
        
        batch_size = self.parameters["batch_size"]        

        global_space = (batch_size,)
        local_space = None
        
        self.program.context_update(queue, global_space, local_space, xs_bf,
                                    ys_bf, ps_bf, ts_bf, res_x_bf, res_y_bf,
                                    n_pol_bf, ev_i_bf, n_events_bf, 
                                    time_context_bf, time_context_mask_bf,
                                    fevskip_bf)
        
    def queue_time_surface_generation(self, ext_buffer, queue):
        """
        Method to queue the ts_gen kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        
        xs_bf = ext_buffer["xs_bf"]
        ys_bf = ext_buffer["ys_bf"]
        ps_bf = ext_buffer["ps_bf"]
        ts_bf = ext_buffer["ts_bf"]        
        res_x_bf = self.buffers["res_x_bf"]
        res_y_bf = self.buffers["res_y_bf"]
        tau_bf = self.buffers["tau_bf"]
        n_pol_bf = self.buffers["n_pol_bf"]
        n_clusters_bf = self.buffers["n_clusters_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]      
        time_context_bf = self.buffers["time_context_bf"]
        time_context_mask_bf = self.buffers["time_context_mask_bf"]       
        time_surface_bf = self.buffers["time_surface_bf"]
        fevskip_bf = ext_buffer["fevskip_bf"] 
        
        batch_size = self.parameters["batch_size"]        

        global_space = (batch_size, self.__loc_ts_size)
        local_space = None
        
        self.program.ts_gen(queue, global_space, local_space, xs_bf,
                            ys_bf, ps_bf, ts_bf, res_x_bf, res_y_bf,
                            tau_bf, n_pol_bf, n_clusters_bf, ev_i_bf,
                            n_events_bf, time_context_bf, time_context_mask_bf,
                            time_surface_bf, fevskip_bf)   
        
    def queue_init_time_surface_generation(self, ext_buffer, queue, time_surface_bf):
        """
        Method to queue the ts_gen kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        
        xs_bf = ext_buffer["xs_bf"]
        ys_bf = ext_buffer["ys_bf"]
        ps_bf = ext_buffer["ps_bf"]
        ts_bf = ext_buffer["ts_bf"]        
        res_x_bf = self.buffers["res_x_bf"]
        res_y_bf = self.buffers["res_y_bf"]
        tau_bf = self.buffers["tau_bf"]
        n_pol_bf = self.buffers["n_pol_bf"]
        n_clusters_bf = self.buffers["n_clusters_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]      
        time_context_bf = self.buffers["time_context_bf"]
        time_context_mask_bf = self.buffers["time_context_mask_bf"]       
        fevskip_bf = ext_buffer["fevskip_bf"] 
        
        batch_size = self.parameters["batch_size"]        

        global_space = (batch_size, self.__loc_ts_size)
        local_space = None
        
        self.program.init_ts_gen(queue, global_space, local_space, xs_bf,
                            ys_bf, ps_bf, ts_bf, res_x_bf, res_y_bf,
                            tau_bf, n_pol_bf, n_clusters_bf, ev_i_bf,
                            n_events_bf, time_context_bf, time_context_mask_bf,
                            time_surface_bf, fevskip_bf)    

    def queue_partial_distances(self, ext_buffer, queue):
        """
        Method to queue the partial_dist kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        
        ts_bf = ext_buffer["ts_bf"]        
        res_x_bf = self.buffers["res_x_bf"]
        res_y_bf = self.buffers["res_y_bf"]
        n_pol_bf = self.buffers["n_pol_bf"]
        n_clusters_bf = self.buffers["n_clusters_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]    
        centroids_bf = self.buffers["centroids_bf"]
        partial_sum_bf = self.buffers["partial_sum_bf"]
        time_surface_bf = self.buffers["time_surface_bf"]
        dcentroids_bf = self.buffers["dcentroids_bf"]
        fevskip_bf = ext_buffer["fevskip_bf"] 
        
        batch_size = self.parameters["batch_size"]        
        n_clusters = self.parameters["n_clusters"]    
        
        global_space = (batch_size, self.__loc_ts_size, n_clusters)
        local_space = (1, self.__loc_ts_size, 1)
        
        self.program.partial_dist(queue, global_space, local_space, ts_bf,
                                  res_x_bf, res_y_bf, n_pol_bf, n_clusters_bf,
                                  ev_i_bf, n_events_bf, centroids_bf, 
                                  partial_sum_bf, time_surface_bf, 
                                  dcentroids_bf, fevskip_bf)   
            
    def queue_reduction_distances(self, ext_buffer, queue):
        """
        Method to queue the reduce_dist kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        
        ts_bf = ext_buffer["ts_bf"]        
        n_clusters_bf = self.buffers["n_clusters_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]    
        partial_sum_bf = self.buffers["partial_sum_bf"]
        distances_bf = self.buffers["distances_bf"]
        fevskip_bf = ext_buffer["fevskip_bf"] 
        
        batch_size = self.parameters["batch_size"]        
        n_clusters = self.parameters["n_clusters"]    
        
        global_space = (batch_size, self.__loc_ts_size, n_clusters)
        local_space = (1, self.__loc_ts_size, 1)
        
        self.program.reduce_dist(queue, global_space, local_space, ts_bf,
                                 n_clusters_bf, ev_i_bf, n_events_bf, 
                                 partial_sum_bf, distances_bf, fevskip_bf)
        
    def queue_infer_end(self, ext_buffer, queue):
        """
        Method to queue the infer_end kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        
        ts_bf = ext_buffer["ts_bf"]        
        n_clusters_bf = self.buffers["n_clusters_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]         
        batch_labels_bf = ext_buffer["batch_labels_bf"]
        distances_bf = self.buffers["distances_bf"]
        closest_c_bf = self.buffers["closest_c_bf"]
        processed_ev_bf = self.buffers["processed_ev_bf"]
        correct_ev_bf = self.buffers["correct_ev_bf"]
        predicted_ev_bf = ext_buffer["predicted_ev_bf"]
        fevskip_bf = ext_buffer["fevskip_bf"] 
        bevskip_bf = ext_buffer["bevskip_bf"] 
        correct_response_bf = self.buffers["correct_response_bf"]

        
        batch_size = self.parameters["batch_size"]        

        global_space = (batch_size,)
        local_space = None
        
        self.program.infer_end(queue, global_space, local_space, ts_bf,
                               n_clusters_bf, ev_i_bf,
                               n_events_bf, batch_labels_bf, distances_bf, 
                               closest_c_bf, processed_ev_bf, correct_ev_bf,
                               predicted_ev_bf,
                               correct_response_bf, fevskip_bf, bevskip_bf)
        
    def queue_init_infer_end(self, ext_buffer, queue):
        """
        Method to queue the init_infer_end kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        
        ts_bf = ext_buffer["ts_bf"]        
        n_clusters_bf = self.buffers["n_clusters_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]         
        batch_labels_bf = ext_buffer["batch_labels_bf"]
        distances_bf = self.buffers["distances_bf"]
        closest_c_bf = self.buffers["closest_c_bf"]
        processed_ev_bf = self.buffers["processed_ev_bf"]
        correct_ev_bf = self.buffers["correct_ev_bf"]
        fevskip_bf = ext_buffer["fevskip_bf"] 
        bevskip_bf = ext_buffer["bevskip_bf"] 

        
        batch_size = self.parameters["batch_size"]        

        global_space = (batch_size,)
        local_space = None
        
        self.program.init_infer_end(queue, global_space, local_space, ts_bf,
                                   n_clusters_bf, ev_i_bf,
                                   n_events_bf, batch_labels_bf, distances_bf, 
                                   closest_c_bf, processed_ev_bf, correct_ev_bf,
                                   fevskip_bf, bevskip_bf)      
        
        
    def queue_weight_update(self, ext_buffer, queue):
        """
        Method to queue the w_update kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        

        ts_bf = ext_buffer["ts_bf"]        
        res_x_bf = self.buffers["res_x_bf"]
        res_y_bf = self.buffers["res_y_bf"]
        n_pol_bf = self.buffers["n_pol_bf"]
        n_clusters_bf = self.buffers["n_clusters_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]      
        centroids_bf = self.buffers["centroids_bf"]
        centroids_update_bf = self.buffers["centroids_update_bf"]
        closest_c_bf = self.buffers["closest_c_bf"]
        lrate_bf = self.buffers["lrate_bf"] 
        S_bf = self.buffers["output_S_bf"] 
        s_gain_bf = self.buffers["s_gain_bf"]
        dS_bf = self.buffers["output_dS_bf"] 
        dcentroids_bf = self.buffers["dcentroids_bf"]
        bevskip_bf = ext_buffer["bevskip_bf"] 
        
        batch_size = self.parameters["batch_size"]        

        global_space = (batch_size, self.__loc_ts_size)
        local_space = (1, self.__loc_ts_size)
        
        self.program.w_update(queue, global_space, local_space, ts_bf,
                              res_x_bf, res_y_bf, n_pol_bf, n_clusters_bf,
                              ev_i_bf, n_events_bf, centroids_update_bf, closest_c_bf,
                              lrate_bf, S_bf, s_gain_bf, dS_bf, dcentroids_bf, bevskip_bf)

    def queue_init_weight_update(self, ext_buffer, queue):
        """
        Method to queue the w_update kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        

        ts_bf = ext_buffer["ts_bf"]        
        res_x_bf = self.buffers["res_x_bf"]
        res_y_bf = self.buffers["res_y_bf"]
        n_pol_bf = self.buffers["n_pol_bf"]
        n_clusters_bf = self.buffers["n_clusters_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]      
        centroids_bf = self.buffers["centroids_bf"]
        closest_c_bf = self.buffers["closest_c_bf"]
        lrate_bf = self.buffers["lrate_bf"] 
        S_bf = self.buffers["output_S_bf"] 
        dS_bf = self.buffers["output_dS_bf"] 
        dcentroids_bf = self.buffers["dcentroids_bf"]
        bevskip_bf = ext_buffer["bevskip_bf"] 
        
        batch_size = self.parameters["batch_size"]        

        global_space = (batch_size, self.__loc_ts_size)
        local_space = (1, self.__loc_ts_size)
        
        self.program.init_w_update(queue, global_space, local_space, ts_bf,
                              res_x_bf, res_y_bf, n_pol_bf, n_clusters_bf,
                              ev_i_bf, n_events_bf, centroids_bf, closest_c_bf,
                              lrate_bf, S_bf, dS_bf, dcentroids_bf, bevskip_bf)        
        
    def queue_feedback_context_update(self, ext_buffer, queue):
        """
        Method to queue the fb_context_update kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        

        ts_bf = ext_buffer["ts_bf"]        
        n_clusters_bf = self.buffers["n_clusters_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]      
        fb_time_context_bf = self.buffers["fb_time_context_bf"]
        fb_time_context_mask_bf = self.buffers["fb_time_context_mask_bf"] 
        closest_c_bf = self.buffers["closest_c_bf"]
        bevskip_bf = ext_buffer["bevskip_bf"] 
        
        batch_size = self.parameters["batch_size"]        

        global_space = (batch_size,)
        local_space = None
        
        self.program.fb_context_update(queue, global_space, local_space, ts_bf,
                                       n_clusters_bf, ev_i_bf, n_events_bf,
                                       fb_time_context_bf, 
                                       fb_time_context_mask_bf, closest_c_bf,
                                       bevskip_bf)     
        
    def queue_feedback_time_surface_generation(self, ext_buffer, queue):
        """
        Method to queue the fb_ts_gen kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        

        ts_bf = ext_buffer["ts_bf"]  
        fb_tau_bf = self.buffers["fb_tau_bf"]
        n_clusters_bf = self.buffers["n_clusters_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]      
        fb_time_context_bf = self.buffers["fb_time_context_bf"]
        fb_time_context_mask_bf = self.buffers["fb_time_context_mask_bf"] 
        fb_partial_sum_bf = self.buffers["fb_partial_sum_bf"]
        closest_c_bf = self.buffers["closest_c_bf"]
        bevskip_bf = ext_buffer["bevskip_bf"] 
        
        batch_size = self.parameters["batch_size"]        

        global_space = (batch_size, self.__loc_cl_size)
        local_space = (1, self.__loc_cl_size)
        
        self.program.fb_ts_gen(queue, global_space, local_space, ts_bf, 
                               fb_tau_bf, n_clusters_bf, ev_i_bf, n_events_bf,
                               fb_time_context_bf, fb_time_context_mask_bf,
                               fb_partial_sum_bf, closest_c_bf, bevskip_bf)
        
        
    def queue_feedback_end(self, ext_buffer, queue):
        """
        Method to queue the fb_end kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        

        ts_bf = ext_buffer["ts_bf"]  
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]      
        fb_partial_sum_bf = self.buffers["fb_partial_sum_bf"]
        S_bf = self.buffers["output_S_bf"] 
        dS_bf = self.buffers["output_dS_bf"] 
        bevskip_bf = ext_buffer["bevskip_bf"] 
        correct_response_bf = self.buffers["correct_response_bf"]

        
        batch_size = self.parameters["batch_size"]        


        global_space = (batch_size, self.__loc_cl_size)
        local_space = (1, self.__loc_cl_size)
        
        self.program.fb_end(queue, global_space, local_space, ts_bf, ev_i_bf,
                            n_events_bf, fb_partial_sum_bf, S_bf, dS_bf,
                            correct_response_bf, bevskip_bf)

    def queue_experimental_weight_reduce(self, ext_buffer, queue):
        """
        Method to queue the w_reduce kernel
        
        Parameters:
            
            ext_buffer : dictionary containing the required buffers for kernel
                         execution that could not being generated during the 
                         layer initialization.
            
            queue : OpenCL queue                          
            
        """
        

        ts_bf = ext_buffer["ts_bf"]        
        res_x_bf = self.buffers["res_x_bf"]
        res_y_bf = self.buffers["res_y_bf"]
        n_pol_bf = self.buffers["n_pol_bf"]
        n_clusters_bf = self.buffers["n_clusters_bf"]
        ev_i_bf = ext_buffer["ev_i_bf"] 
        n_events_bf = ext_buffer["n_events_bf"]      
        centroids_bf = self.buffers["centroids_bf"]
        bevskip_bf = ext_buffer["bevskip_bf"] 
        
        batch_size = self.parameters["batch_size"]        

        global_space = (batch_size, self.__loc_ts_size*self.__loc_cl_size)
        local_space = (batch_size, 1)
        
        self.program.w_reduce(queue, global_space, local_space, ts_bf,
                              res_x_bf, res_y_bf, n_pol_bf, n_clusters_bf,
                              ev_i_bf, n_events_bf, centroids_bf, bevskip_bf)