#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  form_kin40.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.11.2024

import numpy as np
import pandas as pd
import os
import shutil
import pickle

exec(open("python/sim_settings.py").read())

test_X_url = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_test_data.asc"
test_y_url = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_test_labels.asc"
train_X_url = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_train_data.asc"
train_y_url = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_train_labels.asc"

test_X = np.array(pd.read_csv(test_X_url, sep = "\s+", header = None))
test_y = np.array(pd.read_csv(test_y_url, sep = "\s+", header = None)).flatten()
train_X = np.array(pd.read_csv(train_X_url, sep = "\s+", header = None))
train_y = np.array(pd.read_csv(train_y_url, sep = "\s+", header = None)).flatten()

if not os.path.exists(datdir):
    os.makedirs(datdir)

probname = 'kin40k'

with open(datdir+'/'+probname+'.pkl','wb') as f:
    pickle.dump([train_X, train_y, test_X, test_y], f)
