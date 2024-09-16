#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  sim_settings.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.02.2024

import numpy as np

#double_precision = True
#double_precision = False
#precision = '64'
#precision = '32'
precision = '16'

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

#Ms = np.arange(40,321,40)
#Ms = np.arange(40,201,20)
#if problem=='syn_sine':
#    #Ms = np.arange(1,100,20)
#    Ms = np.ceil(np.logspace(0,np.log10(200),num=10)).astype(int)
#else:
    #Ms = np.arange(40,201,20)
#Ms = np.ceil(np.logspace(0,np.log10(50),num=10)).astype(int)
#Ms = np.ceil(np.logspace(0,np.log10(500),num=20)).astype(int)
#Ms = np.ceil(np.logspace(0,np.log10(500),num=10)).astype(int)
#Ms = np.ceil(np.linspace(1,500,num=10)).astype(int)
Ms = np.ceil(np.linspace(50,500,num=10)).astype(int)
#Ms = np.ceil(np.linspace(50,200,num=10)).astype(int)
#reps = 100
reps = 10

## Optimization params
#lr = 1e-3
max_iters = 4000
#max_iters = 500
lr = 1e-2
mb_size = 256
#mb_size = 128
#mb_size = 64
#mb_size = 32
verbose = True

#init_style = 'runif'
#init_style = 'vanil'
init_style = 'mean'
#init_style = 'samp_rand'

## M2 specific
#get_D = lambda M: int(np.ceil(np.sqrt(M)))
#get_D = lambda M: 5
#get_D = lambda M: 1
#get_D = lambda M: 4
#get_D = lambda M: 10
#get_D = lambda M: 8
#get_D = lambda M: 4
#get_D = lambda M: 1
get_D = lambda M: 2
#get_D = lambda M: 10
#get_D = lambda M: 20
#TODO:

if test:
    for i in range(20):
        print("Test!")
    Ms = np.arange(6,10,2)
    reps = 2
    max_iters = 100

methods = ['torch_vanil','torch_rkhs']
#methods = ['torch_rkhs']
#methods = ['torch_vanil']

#colors = {'hens':'blue','four':'green','m2':'orange'}
colors = {'torch_vanil':'blue','torch_rkhs':'green'}
