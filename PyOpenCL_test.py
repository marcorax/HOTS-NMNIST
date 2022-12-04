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


platforms = cl.get_platforms()
extensions_nvidia = platforms[0].extensions
extensions_CPU = platforms[0].extensions

extensions_nvidia.__contains__('cl_khr_icd')
extensions_CPU.__contains__('cl_khr_icd')

devices_nvidia = platforms[0].get_devices()
devices_CPU = platforms[1].get_devices()

global_mem_nvidia = devices_nvidia[0].global_mem_size
global_mem_CPU = devices_CPU[0].global_mem_size

max_work_nvidia = devices_nvidia[0].max_work_group_size
max_work_CPU = devices_CPU[0].max_work_group_size




n = 10
a_np = np.random.randn(n).astype(np.float32)
b_np = np.random.randn(n).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

a_g = cl.array.to_device(queue, a_np)
b_g = cl.array.to_device(queue, b_np)

# lin_comb = ElementwiseKernel(ctx,
#     "float k1, float *a_g, float k2, float *b_g, float *res_g",
#     "res_g[i] = k1 * a_g[i] + k2 * b_g[i]",
#     "lin_comb")

f = open('lin_comb.cl', 'r')
fstr = "".join(f.readlines())
print(fstr)
program=cl.Program(ctx, fstr).build()

res_g = cl.array.empty_like(a_g)
program.lin_comb(2, a_g, 3, b_g, res_g)

# Check on GPU with PyOpenCL Array:
print((res_g - (2 * a_g + 3 * b_g)).get())

# Check on CPU with Numpy:
res_np = res_g.get()
print(res_np - (2 * a_np + 3 * b_np))
print(np.linalg.norm(res_np - (2 * a_np + 3 * b_np)))