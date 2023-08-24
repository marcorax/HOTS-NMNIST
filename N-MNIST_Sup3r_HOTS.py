# -*- coding: utf-8 -*-

"""
Created on Mon Aug  7 16:35:54 2023

@author: marcorax93
"""

from Libs.Sup3r_HOTS import Sup3r_Net
import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt
import pyopencl as cl

# %% Import Tonic dataset

sensor_size = tonic.datasets.NMNIST.sensor_size
cropped_size = (28, 28)
data_trans = tonic.transforms.Compose(
    [
        tonic.transforms.MergePolarities(),
        tonic.transforms.CenterCrop(
            sensor_size=sensor_size, size=cropped_size
        ),
    ]
)

nmnist_train = tonic.datasets.NMNIST(
    save_to="Datasets/", stabilize=True, train=True, transform=data_trans
)

# %% Check if the frames are correct

events, target = nmnist_train[9000]


frame_transform = transforms.ToFrame(
    sensor_size=(*cropped_size, 2), n_time_bins=3
)

frames = frame_transform(events)


def plot_frames(frames):
    fig, axes = plt.subplots(1, len(frames))
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[1] - frame[0])
        axis.axis("off")
    plt.tight_layout()


plot_frames(frames)

# %% GPU Initialization

# TODO, MOVE IT TO NET INITIALIZATION
mf = cl.mem_flags
platforms = cl.get_platforms()
platform_i = 0  # Select the platform manually here
devices = platforms[platform_i].get_devices(device_type=cl.device_type.GPU)
print("Max work group size: ", devices[0].max_work_group_size)
ctx = cl.Context(devices=devices)  # TODO Check how the context apply to more
# than one GPU
queue = cl.CommandQueue(ctx)

# %% Net intialization and training

net = Sup3r_Net(ctx, queue)
net.set_optimizer_param(lrate=1e-3, th_lrate=1e-4, s_gain=5e-4, batch_size=128)
pre_train_folder = "Results/pre_train_test_save/"

net.add_Conv(
    n_clusters=32,
    tau=1e5,
    res_x=28,
    res_y=28,
    win_l=9,
    n_pol=1,
    th_size=50,
    th_decay=0.9,
)
net.add_Class(
    n_clusters=10,
    tau=1e3,
    res_x=28,
    res_y=28,
    n_pol=32,
    fb_signal=True,
    fb_tau=1e3,
)

net.train(
    dataset=nmnist_train,
    n_epochs=20,
    pre_train=True,
    pre_train_alpha=0.3,
    pre_save_folder=pre_train_folder,
)
