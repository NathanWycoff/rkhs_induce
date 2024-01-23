#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  form_kin40.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.11.2024

import numpy as np
import pandas as pd
import os
import shutil
import pickle

exec(open("python/sim_settings.py").read())

staging_dir = 'staging/'

if not os.path.exists(staging_dir):
    os.makedirs(staging_dir)

## kin40k
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

## UCI protein
import requests, zipfile, io

#zip_file_url = 'https://archive.ics.uci.edu/static/public/154/protein+data.zip'
zip_file_url = 'https://archive.ics.uci.edu/static/public/221/kegg+metabolic+reaction+network+undirected.zip'
r = requests.get(zip_file_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(staging_dir)

datstr = 'Reaction Network (Undirected).data'
df = pd.read_csv(staging_dir+datstr, header = None)
# Fill missing with zero in this benchmark.
df = df.iloc[:,1:].apply(lambda x: pd.to_numeric(x, errors = 'coerce'), axis = 0).fillna(0)

y = np.log10(np.array(df.iloc[:,0]).astype(float))
X = np.array(df.iloc[:,1:]).astype(float)

#import matplotlib.pyplot as plt
#fig = plt.figure()
#plt.hist(y)
#plt.savefig("hist_kegg.png")
#plt.close()

probname = 'keggu'

with open(datdir+'/'+probname+'.pkl','wb') as f:
    pickle.dump([X,y], f)

## UCI year
import requests, zipfile, io

zip_file_url = 'https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip'
r = requests.get(zip_file_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(staging_dir)

datstr = 'YearPredictionMSD.txt'
df = pd.read_csv(staging_dir+datstr, header = None)

y = np.array(df.iloc[:,0]).astype(float)
X = np.array(df.iloc[:,1:]).astype(float)

#import matplotlib.pyplot as plt
#fig = plt.figure()
#plt.hist(y)
#plt.savefig("hist_year.png")
#plt.close()

probname = 'year'

with open(datdir+'/'+probname+'.pkl','wb') as f:
    pickle.dump([X,y], f)

