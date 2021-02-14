import numpy as np
import pandas as pd
import os
import sys
import csv
import json

from datetime import datetime

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import regularizers
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf
from keras.regularizers import Regularizer

import scipy.sparse
from keras.models import load_model

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


def build_l1_ae_model(l1_reg, input_dim, layer1_dim, encoding_dim):
    input_img = Input(shape=(input_dim,))
    encoded = Dense(layer1_dim, activation='relu')(input_img)

    z_layer_input = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)
    encoded = Dense(encoding_dim, activation='sigmoid')(z_layer_input)
    encoded_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)

    # create encoder model
    encoder = Model(input_img, encoded)
    
    decoded = Dense(layer1_dim, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='tanh')(decoded)

    # create autoencoder model
    autoencoder = Model(input_img, decoded)

    # create decoder model
    encoded_input = Input(shape=(encoding_dim,))
    deco = autoencoder.layers[-2](encoded_input)
    deco = autoencoder.layers[-1](deco)
    decoder = Model(encoded_input, deco)    
                
    autoencoder.compile(optimizer='adadelta', loss=mse_l1_loss(encoded_norm, l1_reg))
    return encoder, decoder, autoencoder


def mse_l1_loss(encoded_layer, lambda_):    
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) + lambda_ * K.sum(K.abs(encoded_layer)))
    return loss 

def save_model(encoding_dim, l1_reg, autoencoder, encoder, decoder, etf_ticker):
    autoencoder_model_path = r"../data/etf_de/output/autoencoder_l1_reg_{}_{}_{}.h5".format(encoding_dim, l1_reg, etf_ticker)
    encoder_model_path = r"../data/etf_de/output/encoder_l1_reg_{}_{}_{}.h5".format(encoding_dim, l1_reg, etf_ticker)
    decoder_model_path = r"../data/etf_de/output/decoder_l1_reg_{}_{}_{}.h5".format(encoding_dim, l1_reg, etf_ticker)

    autoencoder.save(autoencoder_model_path)
    print("autoencoder saved!!!")

    encoder.save(encoder_model_path) 
    print("encoder saved!!!")

    decoder.save(decoder_model_path) 
    print("decoder saved!!!")

def save_history(encoding_dim, l1_reg, history, etf_ticker):
    history_filename = r"../data/etf_de/output/history_l1_{}_{}_{}".format(encoding_dim, l1_reg, etf_ticker)
    with open(history_filename, 'w') as f:
        json.dump(history.history, f)

def save_output(x, autoencoder, encoder, decoder, input_type, etf_ticker):
    print("{} Original : ".format(input_type))
    print(x)

    print("{} Predicted : ".format(input_type))
    x_predicted = autoencoder.predict(x)
    print(x_predicted)

    print("{} Original->Encoded->Decoded(Reconsturcted) : ".format(input_type))
    x_encoded = encoder.predict(x)
    x_reconstructed = decoder.predict(x_encoded)
    print(x_reconstructed)

    print("{} Encoded : ".format(input_type))
    print(x_encoded)    

    x_filename = r"../data/etf_de/output/x_{}_{}_{}_{}"
    np.save(x_filename.format(input_type, encoding_dim, l1_reg, etf_ticker), x)

    x_encoded_filename = r"../data/etf_de/output/x_{}_encoded_{}_{}_{}"
    np.save(x_encoded_filename.format(input_type, encoding_dim, l1_reg, etf_ticker), x_encoded)

    x_predicted_filename = r"../data/etf_de/output/x_{}_predicted_{}_{}_{}"
    np.save(x_predicted_filename.format(input_type, encoding_dim, l1_reg, etf_ticker), x_predicted)

# the program main
if __name__ == '__main__':      
    l1_reg = float(sys.argv[1])
    encoding_dim = int(sys.argv[2])
    epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    etf_ticker = sys.argv[5]
    layer1_dim = int(sys.argv[6])

    x_train, x_test, df_etf_ret, split_index = get_etf_data(etf_ticker)

    print("Running standard AE with the following parameters : ")
    print("x_train dimension : ({} x {})".format(x_train.shape[0], x_train.shape[1]))
    print("x_test dimension : ({} x {})".format(x_test.shape[0], x_test.shape[1]))
    print("encoding_dim dimension : {}".format(encoding_dim))
    print("epochs : {} batch_size : {}".format(epochs, batch_size))
    print("etf_ticker : {}".format(etf_ticker))
    print("layer1_dim : {}".format(layer1_dim))

    input_dim = x_train.shape[1]
    encoder, decoder, autoencoder = build_l1_ae_model(l1_reg, input_dim, layer1_dim, encoding_dim)

    print("Running encoding_dim: {} l1_reg: {}".format(encoding_dim, l1_reg))

    history = autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, verbose=2)
    
    save_model(encoding_dim, l1_reg, autoencoder, encoder, decoder, etf_ticker)

    save_output(x_train, autoencoder, encoder, decoder, 'train', etf_ticker)
    save_output(x_test, autoencoder, encoder, decoder, 'test', etf_ticker)

    save_history(encoding_dim, l1_reg, history, etf_ticker)