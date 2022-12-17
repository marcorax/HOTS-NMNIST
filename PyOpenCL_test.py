#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:20:53 2022

@author: marcorax93
"""


#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel

mf = cl.mem_flags


platforms = cl.get_platforms()
# extensions_nvidia = platforms[0].extensions
# extensions_CPU = platforms[0].extensions

# extensions_nvidia.__contains__('cl_khr_icd')
# extensions_CPU.__contains__('cl_khr_icd')

# devices_nvidia = platforms[0].get_devices()
# devices_CPU = platforms[1].get_devices()

# global_mem_nvidia = devices_nvidia[0].global_mem_size
# global_mem_CPU = devices_CPU[0].global_mem_size

# max_work_nvidia = devices_nvidia[0].max_work_group_size
# max_work_CPU = devices_CPU[0].max_work_group_size


devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
first_gpu  = devices[0] 
print("Total compute units: "+str(first_gpu.max_compute_units))

n = 10000000
a_np = np.random.randn(n).astype(np.float32)
b_np = np.random.randn(n).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
k1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(2))
k2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(3))
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)


# lin_comb = ElementwiseKernel(ctx,
#     "float k1, float *a_g, float k2, float *b_g, float *res_g",
#     "res_g[i] = k1 * a_g[i] + k2 * b_g[i]",
#     "lin_comb")

f = open('lin_comb.cl', 'r')
fstr = "".join(f.readlines())
# print(fstr)
program=cl.Program(ctx, fstr).build()

res_np=np.empty_like(a_np)
program.lin_comb(queue, np.array([n,1]), None, k1_g, a_g, k2_g, b_g, res_g)
cl.enqueue_copy(queue, res_np, res_g)

# Check result
print(res_np - (2 * a_np + 3 * b_np))
print(np.linalg.norm(res_np - (2 * a_np + 3 * b_np)))