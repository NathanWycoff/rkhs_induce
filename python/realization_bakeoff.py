#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  nystrom.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.15.2023

import numpy as np
import matplotlib.pyplot as plt

N = 100
M = 10
P = 1

ell = 1e-1
tau2 = 1e-6

np.random.seed(123)

x = np.linspace(0,1,num=N)
kern = lambda x,y,ell: np.exp(-np.square(x-y)/ell)
def K(x1, x2, ell):
    ret = np.zeros([len(x1),len(x2)])
    for i in range(len(x1)):
        for j in range(len(x2)):
            ret[i,j] = kern(x1[i], x2[j], ell)
    return ret

def K_nys(x, z, ell):
    #Kxx = K(x,x,ell)
    Kxz = K(x,z,ell)
    Kzz = K(z,z,ell)
    ret = Kxz @ np.linalg.inv(Kzz + tau2*np.eye(len(z))) @ Kxz.T
    return ret

def K_new(x, A, ell):
    Kxx = K(x,x,ell)
    Kxz = Kxx @ A.T
    Kzz = A @ Kxx @ A.T
    ret = Kxz @ np.linalg.inv(Kzz + tau2*np.eye(len(z))) @ Kxz.T # TODO: Verify when we add diagonal noise.
    return ret


# Exact
Kxx = K(x, x, ell) + tau2*np.eye(N)

reps = 100
nes = np.repeat(0.,reps)
ne5s = np.repeat(0.,reps)
res = np.repeat(0.,reps)
for rep in range(reps):
    # Nystrom random
    z = np.random.uniform(size=M)
    Kxx2 = K_nys(x,z,ell) + tau2 * np.eye(N) # TODO: Verify when we add diagonal noise.

    # Nystorm subset
    z = x[np.random.choice(N,M,replace=False)]
    Kxx25 = K_nys(x,z,ell) + tau2 * np.eye(N) # TODO: Verify when we add diagonal noise.

    # RKHS Nystrom
    #A = np.random.uniform(size=[M,N])
    A = np.random.normal(size=[M,N])
    Kxx3 = K_new(x,A,ell) + tau2 * np.eye(N)

    L = np.linalg.cholesky(Kxx)
    L2 = np.linalg.cholesky(Kxx2)
    L25 = np.linalg.cholesky(Kxx25)
    L3 = np.linalg.cholesky(Kxx3)

    noise = np.random.normal(size=N)
    y = L @ noise
    y2 = L2 @ noise
    y25 = L25 @ noise
    y3 = L3 @ noise

    fig = plt.figure()
    plt.plot(x,y, label = 'Full')
    plt.plot(x,y2, label = 'Nystrom Random')
    plt.plot(x,y25, label = 'Nystrom Subset')
    plt.plot(x,y3, label = 'RKHS Nystrom')
    plt.legend()
    plt.savefig("temp.pdf")
    plt.close()

    nes[rep] = np.sum(np.square(y-y2))
    ne5s[rep] = np.sum(np.square(y-y25))
    res[rep] = np.sum(np.square(y-y3))

fig = plt.figure()
plt.boxplot([np.log10(d) for d in [nes,ne5s,res]])
plt.savefig('realization_accuracy.pdf')
plt.close()
