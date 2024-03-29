#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  sim_settings.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.02.2024

import numpy as np

#test = True
test = False

#problem = 'syn_sine'
#problem = 'kin40k'
problem = 'year'
#problem = 'keggu'

oversimdir = './simout/'
simdir = oversimdir+problem+'/'
overfigsdir = './figsout/'
figsdir = overfigsdir+problem+'/'
datdir = 'pickles'

#Ms = np.arange(40,321,40)
Ms = np.arange(40,201,20)
reps = 5

## Optimization params
lr = 1e-3
max_iters = 30000 
mb_size = 256
verbose = True

## M2 specific
#get_D = lambda M: int(np.ceil(np.sqrt(M)))
get_D = lambda M: 5
#get_D = lambda M: 2

if test:
    for i in range(20):
        print("Test!")
    Ms = np.arange(6,10,2)
    reps = 2
    max_iters = 100

methods = ['hens','m2']
#methods = ['hens','four','m2']

colors = {'hens':'blue','four':'green','m2':'orange'}
