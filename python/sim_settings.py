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
#problem = 'year'
#problem = 'keggu'

oversimdir = './simout/'
simdir = oversimdir+problem+'/'
overfigsdir = './figsout/'
figsdir = overfigsdir+problem+'/'
datdir = 'pickles'

#Ms = np.ceil(np.linspace(50,500,num=10)).astype(int)
#Ms = np.ceil(np.linspace(100,750,num=30)).astype(int)
#Ms = np.ceil(np.linspace(1,100,num=10)).astype(int)
Ms = np.ceil(np.linspace(1,500,num=100)).astype(int)
#reps = 100
reps = 10

## Optimization params
#lr = 1e-3
lr = 1e-2

max_iters = 4000
mb_size = 256 #mb_size = 128

#max_iters = 1000
#mb_size = 2048
#verbose = True

#init_style = 'runif'
#init_style = 'vanil'
#init_style = 'mean'
#init_style = 'samp_rand'
#init_style = 'samp_orth'
init_style = 'samp_inv'

## M2 specific
#get_D = lambda M: int(np.ceil(np.sqrt(M)))
#get_D = lambda M: 100
#get_D = lambda M: 4
#get_D = lambda M: 2
#get_D = lambda M: 10

if test:
    for i in range(20):
        print("Test!")
    Ms = np.arange(6,10,2)
    reps = 2
    max_iters = 100

methods = ['torch_rkhs_1','torch_rkhs_2','torch_rkhs_3','torch_rkhs_4','torch_rkhs_5','torch_rkhs_6']

