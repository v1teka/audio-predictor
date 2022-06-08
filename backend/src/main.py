#!/usr/bin/env python3
# coding: utf-8

import time
import pandas as pd
import numpy as np

import utils
import plot
import read_h5 as read
import preprocessing as pp
import neural_net as nn
import kmeans as km


args = utils.arg_parser()

# Создать Dataframe из h5 файлов
t_start = time.time()
df = read.h5_to_df('./data/MillionSongSubset', args.size, args.initialize)        
t_extract = time.time()
print('\nGot', len(df.index), 'songs in',
      round((t_extract-t_start), 2), 'seconds.')


# Создать папку для файлов модели
path = utils.setup_model_dir()

# Векторизация треков
print('Pre-processing extracted song data...')

df = pp.preprocess_columns(df)

df = pp.convert_byte_data(df)


# Shuffle a few times
for i in range(5):
    df = df.iloc[np.random.permutation(len(df))]
df = df.fillna(0)
# Transform into NumPy matrix, normalized by column
X, y, y_map = pp.vectorize(df, 'target', path)
t_preproc = time.time()
print('Cleaned and processed', len(df.index), 'rows in',
      round((t_preproc - t_extract), 2), 'seconds.')

# Обучение
print('Training neural network...')
print('[', X.shape[1], '] x [', np.unique(y).size, ']')
model_simple = nn.deep_nn(pp.scaler(X, 'robust', path), y, 'std', path)
# nn.deep_nn(X, y)
t_nn = time.time()
print('Neural network trained in', round((t_nn - t_preproc), 2), 'seconds.')

print('Evaluating model and saving class probabilities...')
predDF = pd.DataFrame.from_records(
    model_simple.predict(pp.scaler(X, 'robust')))
predDF.to_pickle(path + '/model_prob.pkl')

# K-means кластеризация
clusters = 18
print('Applying k-Means classifier with', clusters, 'clusters...')
kmX = km.kmeans(pp.scaler(X, 'robust', path), clusters)
print('Complete.')

# send classified data through neural network
print('Training neural network...')
print('[', kmX.shape[1], '] x [', np.unique(y).size, ']')
model_classified = nn.deep_nn(kmX, y, 'hyb', path)
t_km = time.time()
print('Hybrid k-Means neural network trained in',
      round((t_km - t_nn), 2), 'seconds.')


# График
plot.plot_nn_training(path, 'loss')
plot.plot_nn_training(path, 'acc')
