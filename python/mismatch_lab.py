#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  mismatch_lab.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.23.2024

import numpy as np

P = 2

#A_0 = np.random.normal(size=[P,P])
#A_0 = A_0.T @ A_0
A_0 = np.diag(np.abs(np.random.normal(size=P)))
A_0i = np.linalg.inv(A_0)

A_1 = np.random.normal(size=[P,P])
A_1 = A_0.T @ A_0
A_1i = np.linalg.inv(A_1)

A_2 = np.random.normal(size=[P,P])
A_2 = A_0.T @ A_0
A_2i = np.linalg.inv(A_2)

DELTA = A_1i+A_2i-A_0i
ev = np.linalg.eigh(DELTA)[0]
delta_pd = np.all(ev>0)

if delta_pd:
    x1 = np.random.normal(size=P)
    x2 = np.random.normal(size=P)
    x3 = np.random.normal(size=P)

    # k(.,x1;A0), k(.,x2;A1), k(.,x3;A2)
    S = np.zeros([3,3])

    const = 1/np.sqrt(np.linalg.det(A_1)) * 1/np.sqrt(np.linalg.det(A_2)) * np.sqrt(np.linalg.det(A_0)) * 1/np.sqrt(np.linalg.det(A_1i+A_2i-A_0i))

    S[0,0] = np.exp(-0.5*(x1-x1).T @ A_0 @ (x1-x1))
    S[0,1] = S[1,0] = np.exp(-0.5*(x1-x2).T @ A_1 @ (x1-x2))
    S[0,2] = S[2,0] = np.exp(-0.5*(x1-x3).T @ A_2 @ (x1-x3))

    S[1,1] = const 
    S[1,2] = S[2,1] = const * np.exp(-0.5*(x2-x3).T @ np.linalg.inv(DELTA) @ (x2-x3))

    S[2,2] = const

    gram_pd = np.all(np.linalg.eigh(S)[0]>0)

    print(delta_pd)
    print(gram_pd)
else:
    print("delta not pd")
