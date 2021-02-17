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

def mse_regularized_loss(encoded_layer, lambda_):    
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) + lambda_ * K.sum(K.abs(encoded_layer)))
    return loss 

def load_ae_model(base_model_path):
    encoder = load_model(f'{base_model_path}encoder', compile=False)
    decoder = load_model(f'{base_model_path}decoder', compile=False)
    autoencoder = load_model(f'{base_model_path}autoencoder', compile=False)
    return encoder, decoder, autoencoder

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

def create_scree_plot(svp, output_dir):
    plt.figure()
    sv_norm = svp / np.max(svp)
    df = pd.DataFrame({'dimension':np.arange(1, len(svp) + 1), 'svp':svp, 'normalized_sv':sv_norm})
    df.to_csv(f'{output_dir}/ae_scree_data.csv', index=False)
    plt.plot(df.dimension, df.normalized_sv, 'bo-')
    plt.xlabel('Intrinsic Dimension')
    plt.ylabel('Normalized Singular Values')
    plt.savefig(f'{output_dir}/ae_scree_plot.png')
    plt.close()

# the program main
if __name__ == '__main__':      
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Full path to the dataset to read in.')
    parser.add_argument('--output_dir', required=True, help='Path to where any output should be saved.')
    parser.add_argument('--base_model_path', required=False, default=None, help='Base path to model to load if using a pre-fit model.')
    parser.add_argument('--plot_title', required=True, help='Title to be used for the generated plot.')
    parser.add_argument('--tp', required=True, help='Specify time period on which to assess dimensionality (1min, 5min, 10min, 15min)')
    parser.add_argument('--svp_threshold', required=False, type=float, default=0.01, help='Determines the threshold to use when evaluating the dimensionality using SVP.')
    parser.add_argument('--scree', required=False, default=False, action='store_true', help='If supplied a scree plot based on test data will be generated.')
    parser.add_argument('--threshold_metrics', required=False, default=False, action='store_true', help='If supplied gather dimensionality for various svp threshold values.')
    opt = parser.parse_args()
    print(opt)

    input_dataset = opt.dataset
    base_output_folder = opt.output_dir
    base_model_path = opt.base_model_path
    plot_title = opt.plot_title
    tp = opt.tp
    generate_scree = opt.scree

    # Hackity Hack Hack
    # https://www.mmbyte.com/article/9604.html
    tf.compat.v1.disable_eager_execution()

    x_train, x_test, df = get_data(input_dataset)

    # Load models from disk
    encoder, decoder, autoender = load_ae_model(base_model_path)

    timestamp_col = 'timestamp'
    minutes_to_add = int(tp.rstrip('min'))
    df[timestamp_col]= pd.to_datetime(df[timestamp_col])
    df = df.set_index(df[timestamp_col])
    df = df.sort_index()
    df.drop(columns=[timestamp_col], axis=1, inplace=True)
    min_date = df.index.min()
    max_date = df.index.max() + pd.Timedelta(minutes=minutes_to_add)
    row_range = pd.date_range(start=min_date, end=max_date, freq=tp)

    print("Running standard AE with the following parameters : ")
    print("x_test dimension : ({} x {})".format(x_test.shape[0], x_test.shape[1]))
    print(f'input_dataset : {input_dataset}')
    print(f'base_output_folder : {base_output_folder}')
    print(f'plot_title : "{plot_title}"')
    print(f'base_model_path : "{base_model_path}"')
    print(f'svp_threshold : "{opt.svp_threshold}"')

    # If output folder doesn't exist, create it
    if not os.path.exists(base_output_folder):
        os.makedirs(base_output_folder, exist_ok=True)

    if generate_scree:
        # Perform prediction on entire test set
        data = df.values.astype('float32')
        z = encoder.predict(data)
        z_row_sorted = sort_by_row(z)
        z_mu = np.mean(z_row_sorted, axis=0)

        # Create scree plot using SVP
        create_scree_plot(z_mu, base_output_folder)

        # Exit script
        sys.exit()

    if opt.threshold_metrics:
        threshold_values = [
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]

        estimated_dimensionality = []
        for threshold in threshold_values:
            # Perform prediction on test set
            data = df.values.astype('float32')
            z = encoder.predict(data)
            z_row_sorted = sort_by_row(z)
            z_mu = np.mean(z_row_sorted, axis=0)
            gte_sorted = count_gt_threshold(z_mu, threshold)

            # Collect dimensions
            estimated_dimensionality.append(gte_sorted)

        # Save data and plot
        threshold_metrics = pd.DataFrame({'threshold':threshold_values, 'dimension':estimated_dimensionality})
        threshold_metrics.to_csv(f'{base_output_folder}/ae_threshold_metrics.csv')

        plt.figure()
        plt.bar(np.arange(len(threshold_metrics.threshold)), height=threshold_metrics.dimension)
        plt.xticks(np.arange(len(threshold_metrics.threshold)), labels=threshold_metrics.threshold.apply(lambda x : '{0:.3f}'.format(x)), fontsize=4)
        plt.xlabel('SVP Thresholds')
        plt.ylabel('Estimated Dimensionality')
        plt.savefig(f'{base_output_folder}/ae_threshold.png')
        plt.close()

        # Exit script
        sys.exit()

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
        gte_sorted = count_gt_threshold(z_mu, opt.svp_threshold)
        
        z_mu_1 = sorted(np.mean(z, axis=0), reverse=True)
        gte_dim = count_gt_threshold(z_mu_1, opt.svp_threshold)

        interval_starts.append(interval_start)
        interval_ends.append(interval_end)
        gte_sorted_dimensions.append(gte_sorted)
        gte_dimensions.append(gte_dim)
    
    output_df = pd.DataFrame({'interval_start':interval_starts, 'interval_end':interval_ends, 'gte_sorted_dimensions':gte_sorted_dimensions, 'gte_dimensions':gte_dimensions})
    output_df.to_csv(f'{base_output_folder}/ae_{tp}_dimensions.csv', index=False)

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
    plt.ylim(0, 25)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f'{base_output_folder}/ae_{tp}_dimensions.png')
    plt.close()
