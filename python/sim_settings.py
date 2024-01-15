#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  sim_settings.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.02.2024

import numpy as np

#problem = 'syn_sine'
problem = 'kin40k'

oversimdir = './simout/'
simdir = oversimdir+problem+'/'
overfigsdir = './figsout/'
figsdir = overfigsdir+problem+'/'
datdir = 'pickles'

#Ms =[5, 10, 15, 20]
#Ms = np.power(2, np.arange(4,10))
#Ms = np.arange(5,80,20)
#Ms = np.arange(5,80,5)

#Ms = np.arange(40,360,40)
#reps = 30

#Ms = np.array([5,10])
Ms = np.arange(40,360,40)
reps = 2

## Optimization params
lr = 1e-2
max_iters = 2000

#get_D = lambda M: int(np.ceil(np.sqrt(M)))
get_D = lambda M: 5
#get_D = lambda M: 2

mb_size = 256
#mb_size = 16

verbose = True

