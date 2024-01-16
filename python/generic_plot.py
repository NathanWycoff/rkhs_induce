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

df['MSE'] = np.log10(df['MSE'])

rdf = df.groupby(['Method','M']).median()


fig = plt.figure()

plt.subplot(1,2,1)
plt.plot(rdf.loc['hen','MSE'], label = 'hen')
plt.plot(rdf.loc['m2','MSE'], label = 'm2')
plt.ylabel("logMSE")
plt.legend()

plt.subplot(1,2,2)
plt.plot(rdf.loc['hen','Time'], label = 'hen')
plt.plot(rdf.loc['m2','Time'], label = 'm2')
plt.ylabel("Time")
plt.legend()

plt.subplot(1,2,3)
plt.plot(rdf.loc['hen','TPI'], label = 'hen')
plt.plot(rdf.loc['m2','TPI'], label = 'm2')
plt.ylabel("TPI")
plt.legend()

plt.tight_layout()
plt.savefig("hen_marg.pdf")
plt.close()

fig = plt.figure()
plt.scatter(rdf.loc['hen','MSE'], rdf.loc['hen','Time'], label = 'hen', color = 'blue')
plt.scatter(df.loc['hen','MSE'], rdf.loc['hen','Time'], label = 'hen', color = 'blue', alpha = 0.5)
plt.scatter(rdf.loc['m2','MSE'], rdf.loc['m2','Time'], label = 'm2', color = 'orange')
plt.scatter(df.loc['m2','MSE'], rdf.loc['hen','Time'], label = 'hen', color = 'orange', alpha = 0.5)
plt.xlabel("logMSE")
plt.ylabel("Time (s)")
plt.legend()
plt.savefig("hen_paretto.pdf")
plt.close()


fig = plt.figure()
plt.scatter(rdf.loc['hen','MSE'], rdf.loc['hen','TPI'], label = 'hen')
plt.scatter(rdf.loc['m2','MSE'], rdf.loc['m2','TPI'], label = 'm2')
plt.xlabel("logMSE")
plt.ylabel("TPI (s)")
plt.legend()
plt.savefig("hen_paretto_tpi.pdf")
plt.close()

