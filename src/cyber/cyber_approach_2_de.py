#!/bin/env python3
#SBATCH -N 1 # nodes requested
#SBATCH -n 1 # tasks requested
#SBATCH -c 4 # cores requested
#SBATCH --gres=gpu:1 # nos GPUS
#SBATCH -o outfile # sebd stdout to outfile
#SBATCH -e errfile # sebd stderr to errfile
#SBATCH -t 24:00:00 # time requested in hour:minute:second

import argparse
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
from keras.models import save_model
from keras.models import load_model

import dataset.Dataset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def get_data(input_dataset):
    print(f"Loading {input_dataset}")
    ds = dataset.Dataset.Dataset.load(input_dataset)
    x_train = ds.training_data.iloc[:,1:-1]
    x_train = x_train.values.astype('float32')
    x_test = ds.test_data.iloc[:,1:-1]
    x_test = x_test.values.astype('float32')
    return x_train, x_test, ds.test_data.iloc[:,:-1]

def build_lite_ae_model(l1_reg, encoding_dim, layer1_dropout, layer2_dropout, layer3_dropout):
    input_img = Input(shape=(74,))
    encoded = Dense(60, activation='relu')(input_img)
    encoded = Dropout(layer1_dropout)(encoded)
    encoded = Dense(45, activation='relu')(encoded)
    encoded = Dropout(layer2_dropout)(encoded)

    # an additional layer with 30% dropout added to encoder
    encoded = Dense(35, activation='relu')(encoded)
    encoded = Dropout(layer3_dropout)(encoded)

    z_layer_input = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)
    encoded = Dense(encoding_dim, activation='sigmoid')(z_layer_input)
    encoded_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)
    
    # an additional layer to match the encoder is added to decoder
    decoded = Dense(35, activation='relu')(encoded)

    # was in the original ICMLA AEDE
    decoded = Dense(45, activation='relu')(decoded)
    decoded = Dense(60, activation='relu')(decoded)
    decoded = Dense(74, activation='sigmoid')(decoded)

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

    x_encoded_filename = r"{}/x_{}_{}_encoded_{}_{}"
    np.save(x_encoded_filename.format(base_output_folder, input_type, epoch, encoding_dim, l1_reg), x_encoded)

    x_reconstructed_filename = r"{}/x_{}_{}_reconstructed_{}_{}"
    np.save(x_reconstructed_filename.format(base_output_folder, input_type, epoch, encoding_dim, l1_reg), x_reconstructed)


def save_output(x, autoencoder, encoder, decoder, layer1_dropout, layer2_dropout, layer3_dropout, input_type):
    x_predicted = autoencoder.predict(x)

    x_encoded = encoder.predict(x)
    x_reconstructed = decoder.predict(x_encoded)

    print(x_encoded)    

    x_filename = r"{}/x_{}_{}_{}_{}_{}_{}"
    np.save(x_filename.format(base_output_folder, input_type, encoding_dim, l1_reg, layer1_dropout, layer2_dropout, layer3_dropout), x)

    x_encoded_filename = r"{}/x_{}_encoded_{}_{}_{}_{}_{}"
    np.save(x_encoded_filename.format(base_output_folder, input_type, encoding_dim, l1_reg, layer1_dropout, layer2_dropout, layer3_dropout), x_encoded)

    x_predicted_filename = r"{}/x_{}_predicted_{}_{}_{}_{}_{}"
    np.save(x_predicted_filename.format(base_output_folder, input_type, encoding_dim, l1_reg, layer1_dropout, layer2_dropout, layer3_dropout), x_predicted)

class SaveIntermediateTrainingOutput(Callback):
    def __init__(self, x, encoder, decoder):
        super(Callback, self).__init__()
        self.x = x
        self.encoder = encoder
        self.decoder = decoder
        self.counter = 1

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 50 == 0:
            print("File counter: {}".format(self.counter*(epoch+1)))
            save_intermediate_training(self.x, self.encoder, self.decoder, self.counter*(epoch+1))
            self.counter = self.counter + 1

def count_gt_threshold(z, threshold):
    # Some of our smaller data slices have only a single row...
    if type(z) == np.float32:
        z = [z]

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
    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', required=True, help='Specify time period on which to assess dimensionality (1min, 5min, 10min, 15min)')
    parser.add_argument('--encoding_dim', required=False, type=int, default=25, help='Dimension used in the final layer of the encoder.')
    parser.add_argument('--l1_reg', required=False, type=float, default=0.01, help='Lambda used for L1 regularization of the encoder hidden layer weights.')
    parser.add_argument('--num_epochs', required=True, type=int, help='Number of epochs to execute for training.')
    parser.add_argument('--batch_size', required=False, type=int, default=32, help='Batch size used during training.')
    parser.add_argument('--layer1_dropout', required=False, type=float, default=0.01, help='Drop out used in layer 1 of encoder.')
    parser.add_argument('--layer2_dropout', required=False, type=float, default=0.01, help='Drop out used in layer 2 of encoder.')
    parser.add_argument('--layer3_dropout', required=False, type=float, default=0.01, help='Drop out used in layer 3 of encoder.')
    parser.add_argument('--dataset', required=True, help='Full path to the dataset to read in.')
    parser.add_argument('--output_dir', required=True, help='Path to where any output should be saved.')
    parser.add_argument('--plot_title', required=True, help='Title to be used for the generated plot.')
    opt = parser.parse_args()
    print(opt)

    l1_reg = opt.l1_reg
    encoding_dim = opt.encoding_dim
    num_epochs = opt.num_epochs
    batch_size = opt.batch_size
    layer1_dropout = opt.layer1_dropout
    layer2_dropout = opt.layer2_dropout
    layer3_dropout = opt.layer3_dropout 
    input_dataset = opt.dataset
    base_output_folder = opt.output_dir
    tp = opt.tp
    plot_title = opt.plot_title

    # Hackity Hack Hack
    # https://www.mmbyte.com/article/9604.html
    tf.compat.v1.disable_eager_execution()

    x_train, x_test, df = get_data(input_dataset)

    timestamp_col = 'timestamp'
    df[timestamp_col]= pd.to_datetime(df[timestamp_col])
    df = df.set_index(df[timestamp_col])
    df = df.sort_index()
    df.drop(columns=[timestamp_col], axis=1, inplace=True)
    min_date = df.index.min()
    max_date = df.index.max() + pd.Timedelta(minutes=1)
    row_range = pd.date_range(start=min_date, end=max_date, freq=tp)

    print("Running standard AE with the following parameters : ")
    print("x_train dimension : ({} x {})".format(x_train.shape[0], x_train.shape[1]))
    print("encoding_dim dimension : {}".format(encoding_dim))
    print("epochs : {} batch_size : {}".format(num_epochs*50, batch_size))
    print("layer1_dropout : {} layer2_dropout : {} layer3_dropout : {}".format(layer1_dropout, layer2_dropout, layer3_dropout))
    print(f'input_dataset : {input_dataset}')
    print(f'base_output_folder : {base_output_folder}')
    print(f'plot_title : "{plot_title}"')

    # If output folder doesn't exist, create it
    if not os.path.exists(base_output_folder):
        os.makedirs(base_output_folder, exist_ok=True)

    encoder, decoder, autoencoder = build_lite_ae_model(l1_reg, encoding_dim, layer1_dropout, layer2_dropout, layer3_dropout)

    print("Running encoding_dim: {} l1_reg: {}".format(encoding_dim, l1_reg))

    loss_metrics = dict()
    loss_metrics['epoch'] = []
    loss_metrics['loss'] = []
    loss_metrics['val_loss'] = []
    for i in range(1, num_epochs):
        history = autoencoder.fit(
            x_train, 
            x_train, 
            epochs=10, 
            batch_size=batch_size, 
            verbose=2, 
            shuffle=True, 
            validation_data=(x_test, x_test)
        )

        epoch = i * 10

        loss = history.history['loss'][-1]
        print("AE,{},{}".format(epoch, loss))
        loss_metrics['epoch'].append(epoch)
        loss_metrics['loss'].append(loss)
        loss_metrics['val_loss'].append(history.history['val_loss'][-1])

        # Only perform this validation every 50 epochs
        if epoch % 50 == 0:
            ###
            # Traverse test data in desired time intervals to 
            # determine dimensionality over time
            ###
            interval_starts = list()
            interval_ends = list()
            gte_sorted_dimensions = list()
            gte_dimensions = list()
            for j in range(len(row_range) - 1):
                interval_start = row_range[j]
                interval_end = row_range[j + 1]
                current_df = df.loc[interval_start:interval_end]

                if current_df.empty:
                    print(f'No data for range {interval_start}:{interval_end}')
                    interval_starts.append(interval_start)
                    interval_ends.append(interval_end)
                    gte_sorted_dimensions.append(0)
                    gte_dimensions.append(0)
                    continue

                # Structure for use in prediction
                x_test_current = current_df.values.astype('float32')

                z = encoder.predict(x_test_current) # use x_test
                z_row_sorted = sort_by_row(z)
                z_mu = np.mean(z_row_sorted, axis=0)
                gte_sorted = count_gt_threshold(z_mu, 0.01)
                
                z_mu_1 = sorted(np.mean(z, axis=0), reverse=True)
                gte_dim = count_gt_threshold(z_mu_1, 0.01)

                interval_starts.append(interval_start)
                interval_ends.append(interval_end)
                gte_sorted_dimensions.append(gte_sorted)
                gte_dimensions.append(gte_dim)
            
            output_df = pd.DataFrame({'interval_start':interval_starts, 'interval_end':interval_ends, 'gte_sorted_dimensions':gte_sorted_dimensions, 'gte_dimensions':gte_dimensions})
            output_df.to_csv(f'{base_output_folder}/{epoch}_ae_{tp}_dimensions.csv', index=False)
            loss_df = pd.DataFrame(loss_metrics)
            loss_df.to_csv(f'{base_output_folder}/{epoch}_ae_{tp}_loss.csv', index=False)

            # Plot the dimensions over time
            fig, ax = plt.subplots()
            plt.plot(output_df.interval_start, output_df.gte_sorted_dimensions)
            plt.text(s=f'{plot_title}', x=0.5, y=0.94, fontsize=15, ha='center', transform=fig.transFigure)
            plt.text(s=f'Dimensionality estimated using {tp} intervals', x=0.5, y=0.88, ha='center', fontsize=10, transform=fig.transFigure)
            plt.subplots_adjust(top=0.85, wspace=0.2)
            plt.xlabel('Time of Day')
            plt.ylabel('Dimension Estimate')
            ax.xaxis_date()
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            fmt = mdates.DateFormatter('%H:%M')
            plt.gca().xaxis.set_major_formatter(fmt)
            ax.tick_params(axis='x', which='major', labelsize=6)
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.savefig(f'{base_output_folder}/{epoch}_ae_{tp}_dimensions.png')
            plt.close()

            # Save off model
            autoencoder.save(f'{base_output_folder}/{epoch}_autoencoder')
            encoder.save(f'{base_output_folder}/{epoch}_encoder')
            decoder.save(f'{base_output_folder}/{epoch}_decoder')

    save_output(x_train, autoencoder, encoder, decoder, layer1_dropout, layer2_dropout, layer3_dropout, 'train')
    save_output(x_test, autoencoder, encoder, decoder, layer1_dropout, layer2_dropout, layer3_dropout, 'test')
    autoencoder.save(f'{base_output_folder}/final_autoencoder')
    encoder.save(f'{base_output_folder}/final_encoder')
    decoder.save(f'{base_output_folder}/final_decoder')
