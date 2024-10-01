#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  sim_settings.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.02.2024

import numpy as np

#double_precision = True
#double_precision = False
#precision = '64'
precision = '32'
#precision = '16'

#test = True
test = False

#problem = 'syn_sine'
problem = 'kin40k'
#problem = 'keggu'
#problem = 'year'

oversimdir = './simout/'
simdir = oversimdir+problem+'/'
overfigsdir = './figsout/'
figsdir = overfigsdir+problem+'/'
datdir = 'pickles'

#Ms = np.ceil(np.linspace(1,500,num=10)).astype(int)
#Ms = np.ceil(np.linspace(50,1000,num=10)).astype(int)
Ms = np.ceil(np.linspace(25,750,num=10)).astype(int)
#reps = 100
reps = 10
#K = 10
#K = 1
K = 5
#K = 'M'

## Optimization params
#lr = 1e-3
lr = 1e-2

max_iters = 4000
mb_size = 256 #mb_size = 128
#max_iters = 400
#mb_size = 256 #mb_size = 128

#max_iters = 1000
#mb_size = 2048
#verbose = True

if test:
    for i in range(20):
        print("Test!")
    Ms = np.arange(6,10,2)
    reps = 2
    max_iters = 100

methods = ['torch_hetero','torch_vanil']
#methods = ['torch_vanil','torch_hetero']
