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

fig = plt.figure(figsize=[len(methods)*4,3])

plt.subplot(1,4,1)
for meth in methods:
    plt.plot(rdf.loc[meth,'MSE'], label = meth)
plt.ylabel("logMSE")
plt.legend()

plt.subplot(1,4,2)
for meth in methods:
    plt.plot(rdf.loc[meth,'NLL'], label = meth)
plt.ylabel("NLL")
plt.legend()

plt.subplot(1,4,3)
for meth in methods:
    plt.plot(rdf.loc[meth,'Time'], label = meth)
plt.ylabel("Time")
plt.legend()

plt.subplot(1,4,4)
for meth in methods:
    plt.plot(rdf.loc[meth,'TPI'], label = meth)
plt.ylabel("TPI")
plt.legend()

plt.tight_layout()
plt.savefig(problem+"_marg.png")
plt.close()

from matplotlib import colormaps as cm
#colors = dict([(m,cm['spring'](i/(len(methods)-1))) for i,m in enumerate(methods)])
colors = dict([(m,cm['tab20'](i/(len(methods)-1))) for i,m in enumerate(methods)])

alpha = 0.2
for tt1 in ['Time','TPI']:
    for tt2 in ['MSE','NLL']:
        fname = problem+'_'+tt1+'_'+tt2+'.png'
        fig = plt.figure()
        for meth in methods:
            plt.scatter(rdf.loc[meth,tt2], rdf.loc[meth,tt1], label = meth, color = colors[meth])
            plt.plot(rdf.loc[meth,tt2], rdf.loc[meth,tt1], color = colors[meth])
            mse = df.loc[df['Method']==meth,tt2]
            time = df.loc[df['Method']==meth,tt1]
            plt.scatter(mse, time, color = colors[meth], alpha = alpha)
        plt.yscale('log')

        if tt2=='MSE':
            plt.xlabel("logMSE")
        elif tt2=='NLL':
            plt.xlabel("NLL")
        else:
            raise Exception()
        plt.ylabel(tt1+" (s)")
        plt.legend()
        plt.savefig(fname)
        plt.close()

