#!/usr/bin/env python

import argparse
import dataset.Dataset
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sklearn.decomposition
import sklearn.metrics

###
# Helper functions
###

def count_gt_threshold(z, threshold):
    # Some of our smaller data slices have only a single row...
    if type(z) == np.float32 or type(z) == np.float64:
        z = [z]

    tot = sum(z)
    z_pct = []
    for i in sorted(z, reverse=True):
        if tot != 0:
            z_pct.append(i/tot)
        else:
            z_pct.append(0.0)
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

def create_scree_plot(pca, data, output_path):
    plt.figure()
    sv_norm = pca.singular_values_ / np.max(pca.singular_values_)
    df = pd.DataFrame({'dimension':np.arange(1,data.shape[1] + 1), 'normalized_sv':sv_norm})
    df.to_csv(f'{opt.output_dir}/pca_scree_data.csv', index=False)
    plt.plot(df.dimension, df.normalized_sv, 'bo-')
    plt.xlabel('Intrinsic Dimension')
    plt.ylabel('Normalized Singular Values')
    plt.savefig(f'{output_path}')
    plt.close()

###
# Generate plot of PCA dimensions over time using data from different size
# time periods.
###

# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tp', required=True, help='Specify time period on which to assess dimensionality (1min, 5min, 10min, 15min)')
parser.add_argument('--dataset', required=True, help='Full path to the dataset to read in.')
parser.add_argument('--dstype', required=True, help='Indicates what dataset type to use [training, test, validation].')
parser.add_argument('--train', required=False, action='store_true', default=False, help='Indicates to fit PCA on training data.')
parser.add_argument('--test', required=False, action='store_true', default=False, help='Indicates to load a pre-fit model if no training has occurred.')
parser.add_argument('--model_path', required=False, default=None, help='Path to model to load if using a pre-fit model.')
parser.add_argument('--output_dir', required=True, help='Path to where any output should be saved.')
parser.add_argument('--plot_title', required=True, help='Title to be used for the generated plot.')
parser.add_argument('--ylimit', required=False, default=25, help='Y limit used for produced plot.')
parser.add_argument('--k', required=False, default=None, help='If present, will provide mse metrics for PCA reconstruction from 0-k')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir, exist_ok=True)

# Load dataset
ds = dataset.Dataset.Dataset.load(opt.dataset)
training_df = ds.training_data
if opt.dstype == 'training':
    df = ds.training_data
elif opt.dstype == 'validation':
    df = ds.validation_data
else:
    df = ds.test_data

training_df.drop(columns=['label', 'timestamp'], axis=1, inplace=True)
training_df = training_df.sample(frac=1)
df.drop(columns=['label'], axis=1, inplace=True)

# Determine ranges based on date
# Timestamp column is dropped in this process and used  as an index
timestamp_col = 'timestamp'

minutes_to_add = int(opt.tp.rstrip('min'))

df[timestamp_col]= pd.to_datetime(df[timestamp_col])
df = df.set_index(df[timestamp_col])
df = df.sort_index()
df.drop(columns=[timestamp_col], axis=1, inplace=True)
min_date = df.index.min()
max_date = df.index.max() + pd.Timedelta(minutes=minutes_to_add)
row_range = pd.date_range(start=min_date, end=max_date, freq=opt.tp)

###
# Fit PCA on training data
###

if opt.train:
    # NOTE:  sklearn subtracts mean for us so no need to do this
    #        manually
    pca = sklearn.decomposition.PCA()
    pca.fit(training_df)
    with open(f'{opt.output_dir}/pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

    # Generate scree plot of training data
    create_scree_plot(pca, training_df, f'{opt.output_dir}/training_pca_scree_plot.png')

# We run through ranges of k and produce the reconstruction error
# for each value from 0-k
if opt.k:
    mse_metrics = []
    for k in range(0, int(opt.k) + 1):
        print(f'Calculating PCA loss using {k} dimensions...')
        pca = sklearn.decomposition.PCA(n_components=k)
        pca.fit(training_df)

        # Reconstruct validation data and obtain loss metric
        compressed = pca.transform(df)
        reconstructed = pca.inverse_transform(compressed)

        # Calculate loss
        mse = sklearn.metrics.mean_squared_error(df, reconstructed)
        mse_metrics.append(mse)

    # Store metrics and plot them
    mse_metrics = pd.DataFrame({'k':range(0, int(opt.k) + 1), 'mse':mse_metrics})
    mse_metrics['mse_norm'] = mse_metrics.mse / mse_metrics.mse.max()
    mse_metrics.to_csv(f'{opt.output_dir}/k_mse_metrics.csv', index=False)

    plt.bar(mse_metrics.k, mse_metrics.mse)
    plt.savefig(f'{opt.output_dir}/k_mse_metrics.png')
    plt.close()

    plt.bar(mse_metrics.k, mse_metrics.mse_norm)
    plt.savefig(f'{opt.output_dir}/k_mse_metrics_norm.png')
    plt.close()

if opt.test:
    if opt.model_path:
        print(f'Loading model from {opt.model_path}')
        with open(opt.model_path, 'rb') as f:
            pca = pickle.load(f)

if opt.test:
    # Loop through dataset in blocks of time period
    interval_starts = list()
    interval_ends = list()
    gte_dimensions = list()
    gte_sorted_dimensions = list()
    for i in range(len(row_range) - 1):
        interval_start = row_range[i]
        interval_end = row_range[i + 1]
        current_df = df.loc[interval_start:interval_end]

        if current_df.empty:
            print(f'No data for range {interval_start}:{interval_end}')
            interval_starts.append(interval_start)
            interval_ends.append(interval_end)
            gte_sorted_dimensions.append(0)
            gte_dimensions.append(0)
            continue

        # Transform data to encoded form
        z = pca.transform(current_df)

        z_row_sorted = sort_by_row(z)
        z_mu = np.mean(z_row_sorted, axis=0)
        gte_sorted = count_gt_threshold(z_mu, 0.01)
        
        z_mu_1 = sorted(np.mean(z, axis=0), reverse=True)
        gte_dim = count_gt_threshold(z_mu_1, 0.01)
        
        # Add to data output for saving
        interval_starts.append(interval_start)
        interval_ends.append(interval_end)
        gte_sorted_dimensions.append(gte_sorted)
        gte_dimensions.append(gte_dim)

    # Save dimension data
    output_df = pd.DataFrame({'interval_start':interval_starts, 'interval_end':interval_ends, 'gte_sorted_dimensions':gte_sorted_dimensions, 'gte_dimensions':gte_dimensions})
    output_df.to_csv(f'{opt.output_dir}/pca_{opt.tp}_dimensions.csv', index=False)

    # Plot the dimensions over time
    params = {'text.usetex':True}
    plt.rcParams.update(params)

    fig, ax = plt.subplots()
    plt.plot(output_df.interval_start, output_df.gte_sorted_dimensions)
    plt.text(s=f'{opt.plot_title}', x=0.5, y=0.94, fontsize=15, ha='center', transform=fig.transFigure)
    plt.text(s=f'Dimensionality estimated using {opt.tp} intervals', x=0.5, y=0.88, ha='center', fontsize=10, transform=fig.transFigure)
    plt.subplots_adjust(top=0.85, wspace=0.2)
    plt.xlabel('Time of Day')
    plt.ylabel('Dimension Estimate')
    ax.xaxis_date()

    # Workaround
    locator = mdates.MinuteLocator(interval=30)
    locator.MAXTICKS = 100000

    ax.xaxis.set_major_locator(locator)
    fmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(fmt)
    ax.tick_params(axis='x', which='major', labelsize=6)
    plt.ylim(0, int(opt.ylimit))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f'{opt.output_dir}/pca_{opt.tp}_dimensions.png')
