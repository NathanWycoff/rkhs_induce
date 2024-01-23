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


fig = plt.figure(figsize=[len(methods)*3,3])

plt.subplot(1,3,1)
for meth in methods:
    plt.plot(rdf.loc[meth,'MSE'], label = meth)
#plt.plot(rdf.loc['hen','MSE'], label = 'hen')
#plt.plot(rdf.loc['m2','MSE'], label = 'm2')
plt.ylabel("logMSE")
plt.legend()

plt.subplot(1,3,2)
for meth in methods:
    plt.plot(rdf.loc[meth,'Time'], label = meth)
plt.ylabel("Time")
plt.legend()

plt.subplot(1,3,3)
for meth in methods:
    plt.plot(rdf.loc[meth,'TPI'], label = meth)
plt.ylabel("TPI")
plt.legend()

plt.tight_layout()
plt.savefig("hen_marg.png")
plt.close()

alpha = 0.2
for tt in ['Time','TPI']:
    fname = 'hen_paretto_'+tt+'.png'
    fig = plt.figure()
    for meth in methods:
        plt.scatter(rdf.loc[meth,'MSE'], rdf.loc[meth,tt], label = meth, color = colors[meth])
        plt.plot(rdf.loc[meth,'MSE'], rdf.loc[meth,tt], color = colors[meth])
        mse = df.loc[df['Method']==meth,'MSE']
        time = df.loc[df['Method']==meth,tt]
        plt.scatter(mse, time, color = colors[meth], alpha = alpha)

    plt.xlabel("logMSE")
    plt.ylabel(tt+" (s)")
    plt.legend()
    plt.savefig(fname)
    plt.close()

