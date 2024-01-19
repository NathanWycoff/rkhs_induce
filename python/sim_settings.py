#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  sim_settings.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.02.2024

import numpy as np

test = False

#problem = 'syn_sine'
problem = 'kin40k'

oversimdir = './simout/'
simdir = oversimdir+problem+'/'
overfigsdir = './figsout/'
figsdir = overfigsdir+problem+'/'
datdir = 'pickles'

#Ms = np.arange(40,321,40)
Ms = np.arange(40,201,20)
#reps = 30
#reps = 2
reps = 5
#reps = 8
#reps = 15

## Optimization params
#lr = 1e-2
lr = 1e-3
max_iters = 30000 

#get_D = lambda M: int(np.ceil(np.sqrt(M)))
get_D = lambda M: 5
#get_D = lambda M: 2

mb_size = 256
#mb_size = 16

verbose = True

if test:
    Ms = np.arange(6,10,2)
    max_iters = 100
    reps = 2

