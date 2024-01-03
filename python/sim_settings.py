#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  sim_settings.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.02.2024

import numpy as np

simdir = './simout/'
figsdir = './figsout/'

#Ms =[5, 10, 15, 20]
#Ms = np.power(2, np.arange(4,10))
Ms = np.arange(5,80,5)
reps = 30
#reps = 2

## Optimization params
ls = 'backtracking'
#pc = 'id'
pc = 'exp_ada'
max_iters = 250

#get_D = lambda M: int(np.ceil(np.sqrt(M)))
get_D = lambda M: 5

verbose = False