#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  hen_plot.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.02.2024

import pandas as pd
import os
import matplotlib.pyplot as plt

exec(open("python/sim_settings.py").read())

dfs = []
for f in os.listdir(simdir):
    dfs.append(pd.read_csv(simdir+'/'+f))
df = pd.concat(dfs)
df = df.iloc[:,1:]
