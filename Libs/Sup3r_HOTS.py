# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:07:14 2023

@author: marcorax93
"""

import numpy as np
import pyopencl as cl
from sklearn.cluster import KMeans
import time, os

from Libs.Dense_Layer.Dense_Layer import Dense_Layer
from Libs.Class_Layer.Class_Layer import Class_Layer
from Libs.Conv_Layer.Conv_Layer import Conv_Layer


class Sup3r_Net:
    def __init__(self, ctx, queue, debug=False):
        """
        This method initializes a Sup3r Net.
        This code runs on OpenCL, so it requires a context and a queue.
        At the present moment batch_size is required for memory initialization,
        in the future a .compile method will be used to compile the network on
        OpenCL before running.

        """

        self.layers = []
        self.ctx = ctx
        self.cl_queue = queue
        self.debug = debug

        f = open("Libs/cl_kernels/next_ev.cl", "r")
        fstr = "".join(f.readlines())
        program = cl.Program(ctx, fstr).build(options="-cl-std=CL2.0")
        self.__next_ev = program.next_ev

        self.val_results = {
            "Label Accuracy": [],
            "Event Accuracy": [],
            "Cutoff": [],
        }
        self.test_results = {
            "Label Accuracy": [],
            "Event Accuracy": [],
            "Cutoff": [],
        }

    def pre_train(
        self,
        dataset,
        data_idx,
        alpha=0.5,
        quant_regression=False,
        save_folder=None,
    ):
        """
        Method used to pretrain the layers of the networks by running kmeans
        on the first batch of data

        Parameters:

            dataset : input dataset

            data_idx : array of incidices of the data array, can be shuffled

            alpha : additive noise to each feature to avoid local minima caused
                    by pre training. feat = alpha*A + (1-alpha)*B where A is
                    a random matrix of size feat, generated with a uniform
                    distribution between 0 and 1, and B is the pre trained
                    feature.

            quant_regression : bool, if True, the network is used for quantized
                               regression, meaning that last layer indices now
                               represents quantized values of a regression
                               problem.

            save_folder : string, if provided pre_train will check if
                          pretrained centroids are present and load them up.
                          If no folder is found with this name, pre_train will
                          generate new cenrtoids.


        """

        mf = cl.mem_flags
        ctx = self.ctx

        n_layers = len(self.layers)
        batch_size = self.batch_size

        if not quant_regression:
            batch_labels = np.array(
                [dataset[idx][1] for idx in data_idx[:batch_size]]
            )

        run_clustering = True
        if save_folder != None:
            if os.path.exists(save_folder):
                run_clustering = False

        for layer_i in range(n_layers):
            if run_clustering:

                if self.layers[layer_i].layer_type != "SpacePool":

                    n_pol = self.layers[layer_i].parameters["n_pol"]
                    n_clusters = self.layers[layer_i].parameters["n_clusters"]
                    n_events_batch = np.zeros(batch_size, dtype=int)

                    for i in range(batch_size):
                        data_i = data_idx[i]
                        n_events_batch[i] = len(dataset[data_i][0])

                    n_max_events = max(n_events_batch)
                    ts_np = self.net_run_alloc(
                        dataset, data_idx, 0, batch_size, n_max_events
                    )

                    batch_size, ts_x, ts_y, n_pol = np.shape(
                        self.layers[layer_i].variables["time_surface"]
                    )

                    time_surfaces = np.zeros(
                        [sum(n_events_batch), ts_x, ts_y, n_pol],
                        dtype=np.float32,
                    )

                    batch_events_label = np.zeros(sum(n_events_batch))

                    print("Lay: " + str(layer_i))
                    net_buffers = self.net_buffers

                    # start_exec = time.time()
                    total_ts = 0
                    for ev_i in range(n_max_events):

                        # Run Inference
                        for layer_j in range(layer_i):
                            self.layers[layer_j].infer(
                                net_buffers, self.cl_queue
                            )

                        self.layers[layer_i].infer(net_buffers, self.cl_queue)
                        evskip = np.zeros(batch_size, dtype=np.int32)
                        if self.layers[layer_i].layer_type != "Classifier":
                            cl.enqueue_copy(
                                self.cl_queue,
                                evskip,
                                net_buffers["fevskip_bf"],
                            )
                        else:
                            cl.enqueue_copy(
                                self.cl_queue,
                                evskip,
                                net_buffers["bevskip_bf"],
                            )

                        cl.enqueue_copy(
                            self.cl_queue,
                            self.layers[layer_i].variables["time_surface"],
                            self.layers[layer_i].buffers["time_surface_bf"],
                        ).wait()

                        for file_i in range(batch_size):
                            if (ts_np[file_i, ev_i] != -1) and (
                                evskip[file_i] == 0
                            ):
                                time_surfaces[total_ts] = self.layers[
                                    layer_i
                                ].variables["time_surface"][file_i]
                                if not quant_regression:
                                    batch_events_label[
                                        total_ts
                                    ] = batch_labels[file_i]
                                total_ts += 1

                        self.__next_ev(
                            self.cl_queue,
                            np.array([batch_size]),
                            None,
                            net_buffers["ev_i_bf"],
                        )

                    time_surfaces = time_surfaces[:total_ts]
                    batch_events_label = batch_events_label[:total_ts]

                    # Average nets update
                    for layer_j in range(n_layers):
                        self.layers[layer_j].batch_update(self.cl_queue)

                    # Flush layer data
                    for layer_j in range(n_layers):
                        self.layers[layer_j].batch_flush(self.cl_queue)

                    if self.layers[layer_i].layer_type != "Classifier":

                        time_surfaces = np.reshape(
                            time_surfaces,
                            [sum(n_events_batch), ts_x * ts_y * n_pol],
                        )

                        init_kmeans = KMeans(
                            n_clusters=n_clusters, n_init="auto"
                        ).fit(time_surfaces)

                        init_centroids = np.reshape(
                            init_kmeans.cluster_centers_,
                            [n_clusters, ts_x, ts_y, n_pol],
                        )

                    else:

                        n_classes = self.layers[1].parameters["n_clusters"]
                        init_centroids = np.zeros(
                            [n_clusters, ts_x, ts_y, n_pol]
                        )
                        for class_i in range(n_classes):
                            label_idx = batch_events_label == class_i
                            if sum(label_idx):
                                init_centroids[class_i] = np.mean(
                                    time_surfaces[label_idx], axis=0
                                )
                            else:
                                print(
                                    "WATCHOUT! The first pretrain batch misses one or more classes, try increasing the batch size"
                                )

                    if save_folder != None:

                        # Check if the save_folder exists, if not, create it !
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)

                        # Save the centroid
                        np.save(
                            save_folder
                            + self.layers[layer_i].layer_type
                            + "_"
                            + str(layer_i)
                            + "_weights",
                            init_centroids,
                        )

                        # Save the centroid paramaters
                        Layers_Params = self.layers[layer_i].parameters
                        Layers_Params.pop("ctx")
                        np.save(
                            save_folder
                            + self.layers[layer_i].layer_type
                            + "_"
                            + str(layer_i)
                            + "_params",
                            Layers_Params,
                        )

            else:

                # Load previous centroids
                init_centroids = np.load(
                    save_folder
                    + self.layers[layer_i].layer_type
                    + "_"
                    + str(layer_i)
                    + "_weights.npy"
                )

            centr_max = np.max(init_centroids)

            # weighted sum of random centroid +
            # the pre_trained new centroid
            # centroid =  alpha*centroid + (1-alpha)*(pre_centroid)
            self.layers[layer_i].variables["centroids"][:] = (
                alpha
            ) * self.layers[layer_i].variables["centroids"][0] * centr_max + (
                1 - alpha
            ) * init_centroids

            centroids_bf = self.layers[layer_i].buffers["centroids_bf"]
            cl.enqueue_copy(
                self.cl_queue,
                centroids_bf,
                self.layers[layer_i].variables["centroids"],
            ).wait()

    def train(
        self,
        dataset,
        validation_split=0.1,
        n_epochs=1,
        pre_train=True,
        pre_train_alpha=0.5,
        pre_save_folder=None,
        quant_regression=False,
        shuffle=True,
    ):
        """
        Method used to train the layers of the networks
        """

        batch_size = self.batch_size
        n_layers = len(self.layers)
        n_files = len(dataset)
        data_idx = np.arange(n_files)
        if shuffle:
            np.random.shuffle(data_idx)
        n_batches = n_files // batch_size  # Look if there is
        # a way to not discard the last files for incomplete batches

        validation_accuracy = np.zeros(
            [n_epochs, int(n_batches * validation_split) + 1]
        )
        validation_label_accuracy = np.zeros(
            [n_epochs, int(n_batches * validation_split) + 1]
        )
        validation_cutoff = np.zeros(
            [n_epochs, int(n_batches * validation_split) + 1]
        )

        if pre_train:
            self.pre_train(
                dataset,
                data_idx,
                pre_train_alpha,
                quant_regression,
                pre_save_folder,
            )

        for epoch_i in range(n_epochs):

            if shuffle:
                np.random.shuffle(data_idx)

            rec_idx = 0  # TODO Double check if Rec_IDX is used correctly for
            for batch_i in range(n_batches):
                n_events_batch = np.zeros(batch_size, dtype=int)

                for i in range(batch_size):
                    data_i = data_idx[rec_idx + i]
                    n_events_batch[i] = len(dataset[data_i][0])

                n_max_events = max(n_events_batch)
                ts_np = self.net_run_alloc(
                    dataset, data_idx, rec_idx, batch_size, n_max_events
                )

                net_buffers = self.net_buffers

                start_exec = time.time()

                for ev_i in range(n_max_events):

                    # Run Inference
                    for layer_i in range(n_layers):
                        self.layers[layer_i].infer(net_buffers, self.cl_queue)

                    if batch_i % validation_split * 100:
                        for layer_i in range(n_layers - 1, -1, -1):
                            self.layers[layer_i].learn(
                                net_buffers, self.cl_queue
                            )

                    self.__next_ev(
                        self.cl_queue,
                        np.array([batch_size]),
                        None,
                        net_buffers["ev_i_bf"],
                    )

                # Average nets update
                for layer_i in range(n_layers):
                    self.layers[layer_i].batch_update(self.cl_queue)

                # TODO move predicted ev to classifier layer rather than buffer
                predicted_ev = -1 * np.ones(
                    [batch_size, n_max_events], dtype=np.int32
                )

                cl.enqueue_copy(
                    self.cl_queue, predicted_ev, net_buffers["predicted_ev_bf"]
                ).wait()

                processed_ev = self.layers[-1].variables["processed_ev"]
                correct_ev = self.layers[-1].variables["correct_ev"]

                avg_processed_ev = np.mean(processed_ev / n_events_batch)
                avg_accuracy = np.mean(correct_ev / processed_ev)

                end_exec = time.time()
                print("Epoch: " + str(epoch_i) + " of " + str(n_epochs))
                print("Batch: " + str(batch_i) + " of " + str(n_batches))
                print(
                    "Processed recording "
                    + str(rec_idx)
                    + " of "
                    + str(n_files)
                )
                print(
                    "Elapsed time is ", (end_exec - start_exec) * 10**3, "ms"
                )
                print(
                    "Validation Accuracy is "
                    + str(avg_accuracy)
                    + " of "
                    + str(avg_processed_ev)
                    + " processed events"
                )

                if not quant_regression:
                    label_accuracy = 0
                    for rec_i in range(batch_size):
                        idx, counts = np.unique(
                            predicted_ev[rec_i], return_counts=True
                        )
                        if -1 in idx:
                            where_minus_one = np.where([idx == -1])[0][0]
                            idx = np.delete(idx, where_minus_one)
                            counts = np.delete(counts, where_minus_one)

                        predicted_label = idx[np.argmax(counts)]
                        label_accuracy += (
                            predicted_label
                            == dataset[data_idx[rec_idx + rec_i]][1]
                        )

                    label_accuracy = label_accuracy / batch_size
                    print("Label Accuracy is " + str(label_accuracy))

                # Flush layer data after extracting accuracy
                for layer_i in range(n_layers):
                    self.layers[layer_i].batch_flush(self.cl_queue)

                rec_idx += batch_size

                if not (batch_i % int(validation_split * 100)):
                    validation_accuracy[
                        epoch_i, int(batch_i * validation_split)
                    ] = avg_accuracy
                    validation_label_accuracy[
                        epoch_i, int(batch_i * validation_split)
                    ] = label_accuracy
                    validation_cutoff[
                        epoch_i, int(batch_i * validation_split)
                    ] = avg_processed_ev

                self.val_results = {
                    "Label Accuracy": validation_label_accuracy,
                    "Event Accuracy": validation_accuracy,
                    "Cutoff": validation_cutoff,
                }

    def net_run_alloc(
        self, dataset, data_idx, rec_idx, batch_size, n_max_events
    ):
        """
        Method used to allocate memory on the OCL worker before running
        a train/test/pre-train method
        """
        mf = cl.mem_flags
        ctx = self.ctx

        xs_np = -1 * np.ones((batch_size, n_max_events), dtype=np.int32)
        ys_np = -1 * np.ones((batch_size, n_max_events), dtype=np.int32)
        ps_np = -1 * np.ones((batch_size, n_max_events), dtype=np.int32)
        ts_np = -1 * np.ones((batch_size, n_max_events), dtype=np.int32)
        batch_labels = np.zeros([batch_size], dtype=np.int32)
        n_events_batch = np.zeros([batch_size], dtype=np.int32)

        processed_ev = np.zeros([batch_size], dtype=np.int32)
        correct_ev = np.zeros([batch_size], dtype=np.int32)
        predicted_ev = -1 * np.ones([batch_size, n_max_events], dtype=np.int32)

        # fevskip for feed event skip, and bevskip for back event skip,
        # 1=>true 0=>false
        fevskip = np.zeros(batch_size, dtype=np.int32)
        bevskip = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            data_i = data_idx[rec_idx + i]
            X, Y = dataset[data_i]
            n_events = len(X)
            xs_np[i, :n_events] = np.int32(X["x"])
            ys_np[i, :n_events] = np.int32(X["y"])
            ps_np[i, :n_events] = np.int32(X["p"])
            ts_np[i, :n_events] = np.int32(X["t"])
            # TODO MAKE IT INTO ACTUAL EVENTS
            batch_labels[i] = np.int32(Y)
            n_events_batch[i] = n_events

        # Network Buffers
        xs_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=xs_np)
        ys_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ys_np)
        ps_bf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ps_np)
        ts_bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts_np)
        ev_i_bf = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.int32(0)
        )
        n_events_bf = cl.Buffer(
            ctx,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.int32(n_max_events),
        )
        batch_labels_bf = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=batch_labels
        )
        fevskip_bf = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip
        )
        bevskip_bf = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip
        )
        processed_ev_bf = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=processed_ev
        )
        correct_ev_bf = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=correct_ev
        )
        predicted_ev_bf = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=predicted_ev
        )
        fevskip_bf = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fevskip
        )
        bevskip_bf = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bevskip
        )

        net_buffers = {
            "xs_bf": xs_bf,
            "ys_bf": ys_bf,
            "ps_bf": ps_bf,
            "ts_bf": ts_bf,
            "ev_i_bf": ev_i_bf,
            "n_events_bf": n_events_bf,
            "processed_ev_bf": processed_ev_bf,
            "correct_ev_bf": correct_ev_bf,
            "predicted_ev_bf": predicted_ev_bf,
            "batch_labels_bf": batch_labels_bf,
            "n_labels_bf": self.layers[-1].buffers["n_clusters_bf"],
            "fevskip_bf": fevskip_bf,
            "bevskip_bf": bevskip_bf,
        }

        self.net_buffers = net_buffers

        return ts_np

    def add_Conv(
        self,
        n_clusters,
        tau,
        res_x,
        res_y,
        win_l,
        n_pol,
        th_size,
        th_decay,
        fb_signal=False,
        fb_tau=None,
    ):
        """
        Method used to add a conv layer to the net, stacking on previous layers

        Parameters:

            n_clusters : number of clusters for this layer

            tau : exponential decay tau for the time surface generation

            res_x : horizontal time surface size

            res_y : vertical time surface size

            win_l : lateral size of the convolutional layer, it has to be an odd number

            n_pol : input polarities

            th_size : starting threshold (decision boundary radius) for all clusters

            th_decay : threshold decay, between 1 and 0.

            fb_signal : If True, the layer calculates the feedback signal S, to instruct the learning of lower layers.

            fb_tau : exponential decay tau for the feedback time surface generation


        """

        self.layers.append(
            Conv_Layer(
                n_clusters,
                tau,
                res_x,
                res_y,
                win_l,
                n_pol,
                self.lrate,
                th_size,
                self.th_lrate,
                th_decay,
                self.ctx,
                self.batch_size,
                self.s_gain,
                fb_signal,
                fb_tau,
                self.debug,
            )
        )

        if self.layers and fb_signal:

            self.layers[-2].buffers["input_S_bf"] = self.layers[-1].buffers[
                "output_S_bf"
            ]
            self.layers[-2].buffers["input_dS_bf"] = self.layers[-1].buffers[
                "output_dS_bf"
            ]

            self.layers[-2].variables["input_S"] = self.layers[-1].variables[
                "output_S"
            ]
            self.layers[-2].variables["input_dS"] = self.layers[-1].variables[
                "output_dS"
            ]

    def add_Dense(
        self,
        n_clusters,
        tau,
        res_x,
        res_y,
        n_pol,
        th_size,
        th_decay,
        fb_signal=False,
        fb_tau=None,
    ):
        """
        Method used to add a dense layer to the net,
        stacking on previous layers

        Parameters:

            n_clusters : number of clusters for this layer

            tau : exponential decay tau for the time surface generation

            res_x : horizontal time surface size

            res_y : vertical time surface size

            n_pol : input polarities

            th_size : starting threshold (decision boundary radius) for all clusters

            th_decay : threshold decay, between 1 and 0.

            fb_signal : If True, the layer calculates the feedback signal S, to instruct the learning of lower layers.

            fb_tau : exponential decay tau for the feedback time surface generation


        """

        self.layers.append(
            Dense_Layer(
                n_clusters,
                tau,
                res_x,
                res_y,
                n_pol,
                self.lrate,
                th_size,
                self.th_lrate,
                th_decay,
                self.ctx,
                self.batch_size,
                self.s_gain,
                fb_signal,
                fb_tau,
                self.debug,
            )
        )

        if self.layers and fb_signal:

            self.layers[-2].buffers["input_S_bf"] = self.layers[-1].buffers[
                "output_S_bf"
            ]
            self.layers[-2].buffers["input_dS_bf"] = self.layers[-1].buffers[
                "output_dS_bf"
            ]

            self.layers[-2].variables["input_S"] = self.layers[-1].variables[
                "output_S"
            ]
            self.layers[-2].variables["input_dS"] = self.layers[-1].variables[
                "output_dS"
            ]

    def add_Class(
        self,
        n_clusters,
        tau,
        res_x,
        res_y,
        n_pol,
        fb_signal=False,
        fb_tau=None,
    ):
        """
        Method used to add a classification layer to the net,
        stacking on previous layers

        Parameters:

            n_clusters : number of clusters for this layer

            tau : exponential decay tau for the time surface generation

            res_x : horizontal time surface size

            res_y : vertical time surface size

            n_pol : input polarities

            fb_signal : If True, the layer calculates the feedback signal S, to instruct the learning of lower layers.

            fb_tau : exponential decay tau for the feedback time surface generation


        """

        self.layers.append(
            Class_Layer(
                n_clusters,
                tau,
                res_x,
                res_y,
                n_pol,
                self.lrate,
                self.ctx,
                self.batch_size,
                self.s_gain,
                fb_signal,
                fb_tau,
                self.debug,
            )
        )

        if self.layers and fb_signal:

            self.layers[-2].buffers["input_S_bf"] = self.layers[-1].buffers[
                "output_S_bf"
            ]
            self.layers[-2].buffers["input_dS_bf"] = self.layers[-1].buffers[
                "output_dS_bf"
            ]

            self.layers[-2].variables["input_S"] = self.layers[-1].variables[
                "output_S"
            ]
            self.layers[-2].variables["input_dS"] = self.layers[-1].variables[
                "output_dS"
            ]

        def save_net(save_folder):
            """ """
            # TODO this has to become the whole net save
            n_layers = len(self.layers)

            for layer_i in range(n_layers):
                # Check if the save_folder exists, if not, create it !
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                # Save the centroid
                np.save(
                    save_folder
                    + self.layers[layer_i].layer_type
                    + "_"
                    + str(layer_i)
                    + "_weights",
                    self.layers[layer_i].variables["centroids"],
                )

                # Save the centroid paramaters
                Layers_Params = self.layers[layer_i].parameters
                Layers_Params.pop("ctx")
                np.save(
                    save_folder
                    + self.layers[layer_i].layer_type
                    + "_"
                    + str(layer_i)
                    + "_params",
                    Layers_Params,
                )

        def load_net(save_folder):
            """ """

        def load_net_legacy(save_folder):
            """ """

    def set_optimizer_param(self, lrate, th_lrate, s_gain, batch_size=32):

        # TODO UPDATE TO alpha,beta, gamma delta
        # TODO Recompile the network everytime this is called.
        self.lrate = lrate
        self.th_lrate = th_lrate
        self.s_gain = s_gain
        self.batch_size = batch_size
