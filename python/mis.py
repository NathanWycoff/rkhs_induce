#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  mis.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.02.2024

import os
import shutil

exec(open("python/sim_settings.py").read())

if os.path.exists(simdir):
    shutil.rmtree(simdir)
os.makedirs(simdir)
if os.path.exists(figsdir):
    shutil.rmtree(figsdir)
os.makedirs(figsdir)

argsf = ''
for M in Ms:
    for r in range(reps):
        argsf += str(M) + ' ' + str(r) + '\n'

with open('sim_args.txt','w') as f:
    f.write(argsf)
