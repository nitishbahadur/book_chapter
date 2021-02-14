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

base_input_folder = r'../data/synthetic_polynomial/input'
base_output_folder = r'../data/synthetic_polynomial/output'
input_filename = r'Y_{}.csv'

def get_input_filename(std_dev):
    filename = input_filename.format(std_dev)
    filepath = os.path.join(base_input_folder, filename)
    return filepath

def get_synthetic_data(std_dev):
    filepath = get_input_filename(std_dev)
    print(f"get_synthetic_data is loading {filepath}")
    df = pd.read_csv(filepath, header=None)
    X = df.values
    X = X.astype('float32')
    X = X / (np.max(X) - np.min(X))
    x_train = X[0:3000,:]
    x_test = X[3000:,:]
    return x_train, x_test


def build_lite_ae_model(l1_reg, encoding_dim, layer1_dropout, layer2_dropout, layer3_dropout):
    input_img = Input(shape=(784,))
    encoded = Dense(392, activation='relu')(input_img)
    encoded = Dropout(layer1_dropout)(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dropout(layer2_dropout)(encoded)

    # an additional layer with 30% dropout added to encoder
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dropout(layer3_dropout)(encoded)

    z_layer_input = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)
    encoded = Dense(encoding_dim, activation='sigmoid')(z_layer_input)
    encoded_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)
    
    # an additional layer to match the encoder is added to decoder
    decoded = Dense(64, activation='relu')(encoded)

    # was in the original ICMLA AEDE
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(392, activation='relu')(decoded)
    decoded = Dense(784, activation='tanh')(decoded)

    # create autoencoder
    autoencoder = Model(input_img, decoded)

    # create encoder
    encoder = Model(input_img, encoded)

    # create decoder model
    encoded_input = Input(shape=(encoding_dim,))
    deco = autoencoder.layers[-4](encoded_input) # new code to match additional 64 layer encoder/decoder
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    decoder = Model(encoded_input, deco)    

    # autoencoder.compile(optimizer='adadelta', loss='mse') 
    autoencoder.compile(optimizer='adadelta', loss=mse_regularized_loss(encoded_norm, l1_reg)) 
    return encoder, decoder, autoencoder


def mse_regularized_loss(encoded_layer, lambda_):    
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) + lambda_ * K.sum(K.abs(encoded_layer)))
    return loss 

def get_output_model_path(output_file_type, encoding_dim, l1_reg):
    filename = r'{}_l1_reg_{}_{}.h5'.format(output_file_type, encoding_dim, l1_reg)
    filepath = os.path.join(base_output_folder, filename)
    return filepath

def save_model(encoding_dim, l1_reg, autoencoder, encoder, decoder):
    autoencoder_model_path = get_output_model_path("autoencoder", encoding_dim, l1_reg)
    encoder_model_path = get_output_model_path("encoder", encoding_dim, l1_reg)
    decoder_model_path = get_output_model_path("decoder", encoding_dim, l1_reg)

    autoencoder.save(autoencoder_model_path)
    print("autoencoder saved!!!")

    encoder.save(encoder_model_path) 
    print("encoder saved!!!")

    decoder.save(decoder_model_path) 
    print("decoder saved!!!")

def save_history(encoding_dim, l1_reg, history):
    history_filename = get_output_model_path("history", encoding_dim, l1_reg)
    with open(history_filename, 'w') as f:
        json.dump(history.history, f)

def save_intermediate_training(x, encoder, decoder, epoch):
    input_type = 'train'
    x_encoded = encoder.predict(x)
    x_reconstructed = decoder.predict(x_encoded)

    x_encoded_filename = r"../data/synthetic_polynomial/output/x_{}_{}_encoded_{}_{}"
    np.save(x_encoded_filename.format(input_type, epoch, encoding_dim, l1_reg), x_encoded)

    x_reconstructed_filename = r"../data/synthetic_polynomial/output/x_{}_{}_reconstructed_{}_{}"
    np.save(x_reconstructed_filename.format(input_type, epoch, encoding_dim, l1_reg), x_reconstructed)


def save_output(x, autoencoder, encoder, decoder, layer1_dropout, layer2_dropout, layer3_dropout, input_type):
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

    x_filename = r"../data/synthetic_polynomial/output/x_{}_{}_{}_{}_{}_{}"
    np.save(x_filename.format(input_type, encoding_dim, l1_reg, layer1_dropout, layer2_dropout, layer3_dropout), x)

    x_encoded_filename = r"../data/synthetic_polynomial/output/x_{}_encoded_{}_{}_{}_{}_{}"
    np.save(x_encoded_filename.format(input_type, encoding_dim, l1_reg, layer1_dropout, layer2_dropout, layer3_dropout), x_encoded)

    x_predicted_filename = r"../data/synthetic_polynomial/output/x_{}_predicted_{}_{}_{}_{}_{}"
    np.save(x_predicted_filename.format(input_type, encoding_dim, l1_reg, layer1_dropout, layer2_dropout, layer3_dropout), x_predicted)

class SaveIntermediateTrainingOutput(Callback):
    def __init__(self, x, encoder, decoder):
        super(Callback, self).__init__()
        self.x = x
        self.encoder = encoder
        self.decoder = decoder
        self.counter = 1

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 100 == 0:
            print("File counter: {}".format(self.counter*(epoch+1)))
            save_intermediate_training(self.x, self.encoder, self.decoder, self.counter*(epoch+1))
            self.counter = self.counter + 1


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
    l1_reg = float(sys.argv[1])
    encoding_dim = int(sys.argv[2])
    num_epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    layer1_dropout = float(sys.argv[5])
    layer2_dropout = float(sys.argv[6])
    layer3_dropout = float(sys.argv[7])
    std_dev = float(sys.argv[8])

    x_train, x_test = get_synthetic_data(std_dev)

    print("Running standard AE with the following parameters : ")
    print("x_train dimension : ({} x {})".format(x_train.shape[0], x_train.shape[1]))
    print("encoding_dim dimension : {}".format(encoding_dim))
    print("epochs : {} batch_size : {}".format(num_epochs*100, batch_size))
    print("layer1_dropout : {} layer2_dropout : {} layer3_dropout : {}".format(layer1_dropout, layer2_dropout, layer3_dropout))
    print("std_dev : {}".format(std_dev))

    encoder, decoder, autoencoder = build_lite_ae_model(l1_reg, encoding_dim, layer1_dropout, layer2_dropout, layer3_dropout)

    print("Running encoding_dim: {} l1_reg: {}".format(encoding_dim, l1_reg))

    for i in range(1, num_epochs):
        # history = autoencoder.fit(x_train, x_train, epochs=100, batch_size=batch_size, verbose=2, callbacks=callbacks)
        history = autoencoder.fit(x_train, x_train, epochs=100, batch_size=batch_size, verbose=2)
        z = encoder.predict(x_test) # use x_test
        z_row_sorted = sort_by_row(z)
        z_mu = np.mean(z_row_sorted, axis=0)
        gte_sorted = count_gt_threshold(z_mu, 0.01)
        
        z_mu_1 = sorted(np.mean(z, axis=0), reverse=True)
        gte_dim = count_gt_threshold(z_mu_1, 0.01)
        loss = history.history['loss'][-1]
        print("AE,{},{},{},{},{}".format(std_dev, i*100, loss, gte_sorted, gte_dim))

    save_output(x_train, autoencoder, encoder, decoder, layer1_dropout, layer2_dropout, layer3_dropout, 'train')
    save_output(x_test, autoencoder, encoder, decoder, layer1_dropout, layer2_dropout, layer3_dropout, 'test')