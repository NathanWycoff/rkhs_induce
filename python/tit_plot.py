#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  tit_plot.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.02.2024

import pandas as pd
import os
import matplotlib.pyplot as plt

exec(open("python/sim_settings.py").read())

dfs = []
for f in os.listdir(simdir):
    dfs.append(pd.read_csv(simdir+'/'+f))
df = pd.concat(dfs)
df = df.iloc[:,1:]

df['MSE'] = np.log10(df['MSE'])

rdf = df.groupby(['Method','M']).median()


fig = plt.figure()

plt.subplot(1,2,1)
plt.plot(rdf.loc['tit','MSE'], label = 'tit')
plt.plot(rdf.loc['m2','MSE'], label = 'm2')
plt.ylabel("logMSE")
plt.legend()

plt.subplot(1,2,2)
plt.plot(rdf.loc['tit','Time'], label = 'tit')
plt.plot(rdf.loc['m2','Time'], label = 'm2')
plt.ylabel("Time")
plt.legend()

plt.tight_layout()
plt.savefig("tit_marg.pdf")
plt.close()

fig = plt.figure()
plt.scatter(rdf.loc['tit','MSE'], rdf.loc['tit','Time'], label = 'tit')
plt.scatter(rdf.loc['m2','MSE'], rdf.loc['m2','Time'], label = 'm2')
plt.xlabel("logMSE")
plt.ylabel("Time (s)")
plt.legend()
plt.savefig("tit_paretto.pdf")
plt.close()

