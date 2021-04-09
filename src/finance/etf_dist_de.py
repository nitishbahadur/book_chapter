#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00

import numpy as np
import pandas as pd
import os
import sys
import csv
import json

from datetime import datetime

from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras import regularizers
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf
from keras.regularizers import Regularizer

import scipy.sparse
from keras.models import load_model

from scipy.spatial.distance import squareform, pdist

def get_etf_data(etf_ticker):
    df_etf_ret = pd.read_csv(r'../data/etf_de/input/{}_returns.csv'.format(etf_ticker))
    df_etf_ret['Date'] = pd.to_datetime(df_etf_ret['Date'], format='%Y-%m-%d')
    df_etf_ret.set_index(df_etf_ret['Date'], inplace=True)
    df_etf_ret.drop(columns=['Date'], inplace=True)    
    X = df_etf_ret.values
    
    split_index = int(len(df_etf_ret)*.80) # 80% is training
    
    X = df_etf_ret.values
    X = X.astype('float32')
    X = X / np.max(np.abs(X))

    x_train = X[:split_index,:]
    x_test = X[split_index:,:]

    return x_train, x_test, df_etf_ret, split_index

def build_lite_ae_model(l1_reg, input_dim, encoding_dim):
    layer1_dropout = 0.3
    layer2_dropout = 0.3
    input_img = Input(shape=(input_dim,))
    encoded = Dense(50, activation='sigmoid')(input_img)
    encoded = Dropout(layer1_dropout)(encoded)
    encoded = Dense(40, activation='sigmoid')(encoded)
    encoded = Dropout(layer2_dropout)(encoded)

    z_layer_input = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)
    encoded = Dense(encoding_dim, activation='sigmoid')(z_layer_input)
    encoded_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)
    
    decoded = Dense(40, activation='sigmoid')(encoded)
    decoded = Dense(50, activation='sigmoid')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # create autoencoder
    autoencoder = Model(input_img, decoded)

    # create encoder
    encoder = Model(input_img, encoded)

    # create decoder model
    encoded_input = Input(shape=(encoding_dim,))
    deco = autoencoder.layers[-3](encoded_input)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    decoder = Model(encoded_input, deco)    

    autoencoder.compile(optimizer='adadelta', loss=mse_l1_loss(encoded_norm, l1_reg)) 
    return encoder, decoder, autoencoder

# -------------------------------------------------------------------------------
# The skinny AE model
# -------------------------------------------------------------------------------
def build_skinny_ae_model(l1_reg, input_dim, encoding_dim):
    input_img = Input(shape=(input_dim,))

#     z_layer_input = Lambda(lambda  x: K.l2_normalize(x,axis=1))(input_img)
    encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
    encoded_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)

    # create encoder model
    encoder = Model(input_img, encoded)
    
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # create autoencoder model
    autoencoder = Model(input_img, decoded)

    # create decoder model
    encoded_input = Input(shape=(encoding_dim,))
    deco = autoencoder.layers[-1](encoded_input)
    decoder = Model(encoded_input, deco)    
                
    autoencoder.compile(optimizer='adadelta', loss=mse_l1_loss(encoded_norm, l1_reg))
    return encoder, decoder, autoencoder


def build_ae_model(l1_reg, input_dim, encoding_dim):
    input_img = Input(shape=(input_dim,))
    encoded = Dense(50, activation='relu')(input_img)
    encoded = Dense(40, activation='relu')(encoded)
    
    z_layer_input = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)
    encoded = Dense(encoding_dim, activation='relu')(z_layer_input)
    encoded_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)

    # create encoder model
    encoder = Model(input_img, encoded)
    
    # decoder
    decoded = Dense(40, activation='relu')(encoded)
    decoded = Dense(50, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # create autoencoder model
    autoencoder = Model(input_img, decoded)

    # create decoder model
    encoded_input = Input(shape=(encoding_dim,))
    deco = autoencoder.layers[-3](encoded_input)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)    
    decoder = Model(encoded_input, deco)    
                
    autoencoder.compile(optimizer='adadelta', loss=mse_l1_loss(encoded_norm, l1_reg))
    return encoder, decoder, autoencoder

# the loss function
def mse_l1_loss(encoded_layer, lambda_):    
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) + lambda_ * K.sum(K.abs(encoded_layer)))
    return loss 

def save_output(x, encoder, date_str, etf_ticker, encoding_dim, l1_reg):
    x_encoded = encoder.predict(x)
    z_row_sorted = sort_by_row(x_encoded)
    mu = np.mean(z_row_sorted, axis=0)
    print("1% count_gt_threshold(z_row_sorted, threshold): {}".format(count_gt_threshold(mu, 0.01)))

    mu1 = np.mean(x_encoded, axis=0)
    print("1% count_gt_threshold(z, threshold): {}".format(count_gt_threshold(mu1, 0.01)))

    x_encoded_filename = r"../data/etf_de/output/x_{}_encoded_{}_{}_{}"
    np.save(x_encoded_filename.format(date_str, encoding_dim, l1_reg, etf_ticker), x_encoded)


def get_data_by_date(df_etf_ret, x_test, split_index, date_str):
    test_width = 60
    test_max_rows = len(x_test) - test_width + 1
    for i in range(0, test_max_rows):
        df_ = df_etf_ret.iloc[i+split_index:i+test_width+split_index,:].copy()
        dt = df_.index[-1]
        if dt.strftime('%Y%m%d') == date_str:
            dist = squareform(pdist(df_.values)) 
            x_train = dist.astype('float32')
            x_train = x_train / np.max(x_train)    
            return x_train

def count_gt_threshold(z, threshold):
    tot = sum(z)
    z_pct = [(i/tot) for i in sorted(z, reverse=True)]
    z_gt_theta = [i for i in z_pct if i >= threshold]
    return len(z_gt_theta)

def sort_by_row(z):
    z_sorted = None
    for i in np.arange(z.shape[0]):
        z_s = sorted(z[i,:], reverse=True)
        if z_sorted is None:
            z_sorted = z_s
        else:
            z_sorted = np.vstack((z_sorted,z_s))
    return z_sorted

# the program main
if __name__ == '__main__':      
    etf_ticker = sys.argv[1]
    date_str = sys.argv[2]
    etf_train, etf_test, df_etf_ret, split_index = get_etf_data(etf_ticker)
    x_train = get_data_by_date(df_etf_ret, etf_test, split_index, date_str)

    l1_reg = 5e-5
    input_dim = x_train.shape[1]    
    encoding_dim = 30
    epochs = 30000
    batch_size = 1

    print("Running RELU/SIGMOID AE with the following parameters : ")
    print("etf_ticker : {} date_str: {}".format(etf_ticker, date_str))
    print("x_train dimension : ({} x {})".format(x_train.shape[0], x_train.shape[1]))
    print("encoding_dim dimension : {}".format(encoding_dim))
    print("epochs : {} batch_size : {}".format(epochs, batch_size))

    # encoder, decoder, autoencoder = build_skinny_ae_model(l1_reg, input_dim, encoding_dim)
    encoder, decoder, autoencoder = build_ae_model(l1_reg, input_dim, encoding_dim)    

    print("Running encoding_dim: {} l1_reg: {}".format(encoding_dim, l1_reg))
    
    history = autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False)
    
    save_output(x_train, encoder, date_str, etf_ticker, encoding_dim, l1_reg)
