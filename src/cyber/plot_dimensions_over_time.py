#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def plot_dimensions_over_time_heatmap(fig, ax, ae_df, pca_df, attack_intervals=[]):
    # Convert to dates
    ae_df.interval_start = pd.to_datetime(ae_df.interval_start)
    ae_df.interval_end = pd.to_datetime(ae_df.interval_end)
    pca_df.interval_start = pd.to_datetime(pca_df.interval_start)

    # Normalize so that PCA and AE on same scale
    ae_df.gte_dimensions = ae_df.gte_dimensions / 25
    pca_df.gte_dimensions = pca_df.gte_dimensions / 75

    # Set up our plot limits
    xlims = mdates.date2num(ae_df.interval_start)
    xlimends = mdates.date2num(ae_df.interval_end)
    ylims = [0, 2.0]

    # Plot using imshow
    plt.imshow(
        [ae_df.gte_dimensions, pca_df.gte_dimensions],
        cmap='Blues',
        vmin=0.0,
        vmax=0.5,
        extent=[xlims[0], xlimends[-1] , ylims[0], ylims[1]],
        aspect='auto',
    )

    # Ensure plots are aligned at the end
    ax.set_xlim(xlims[0], xlimends[-2])

    # Make a line to split the two sections
    ax.axhline(y=1.0, color='white', linewidth='1.5')

    plt.yticks([0.5, 1.5], labels=['PCA', 'AE'])
    ax.xaxis_date()
    fmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(fmt)
    ax.tick_params(axis='x', which='major', labelsize=6)

    # Add in guidelines for each interval checked
    for interval_end in xlimends:
        ax.axvline(interval_end, color='gray', linestyle='--', linewidth='0.5', alpha=0.3)

    # Add in attack intervals
    for attack_interval in attack_intervals:
        attack_start = attack_interval[0]
        attack_end = attack_interval[1]
        ax.axvline(mdates.datestr2num(attack_start), color='black', linestyle='--', linewidth='1.5')
        ax.axvline(mdates.datestr2num(attack_end), color='black', linestyle='--', linewidth='1.5')
        
    fig.autofmt_xdate()
    plt.tight_layout()

def plot_dimensions_over_time(fig, ax, ae_df, pca_df, attack_intervals=[]):
    # Convert to dates
    ae_df.interval_start = pd.to_datetime(ae_df.interval_start)
    pca_df.interval_start = pd.to_datetime(pca_df.interval_start)

    plt.plot(ae_df.interval_start, ae_df.gte_dimensions)
    plt.plot(pca_df.interval_start, pca_df.gte_dimensions)
    
    plt.ylim(0, 75)
    plt.legend(['AE', 'PCA'])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    fmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(fmt)
    ax.tick_params(axis='x', which='major', labelsize=6)

    # Add in attack intervals
    for attack_interval in attack_intervals:
        attack_start = attack_interval[0]
        attack_end = attack_interval[1]
        ax.axvspan(*mdates.datestr2num([attack_start, attack_end]), color='red', alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()

###
# Dimensions Over Time
# - Both AE and PCA combined
###

results_dir = './output/20210214_results'
pca_results_dir = './output/20210206_results'
output_dir = './output/20210214_results'

fig, axarr = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6,6))
plt.sca(axarr[0])

# Monday - Load data
ae_1min = pd.read_csv(f'{results_dir}/monday/ae_1min_dimensions.csv')
pca_1min = pd.read_csv(f'{pca_results_dir}/monday/pca_1min_dimensions.csv')
plot_dimensions_over_time(
    fig,
    axarr[0],
    ae_1min, 
    pca_1min, 
    [],
)

plt.sca(axarr[1])
ae_5min = pd.read_csv(f'{results_dir}/monday/ae_5min_dimensions.csv')
pca_5min = pd.read_csv(f'{pca_results_dir}/monday/pca_5min_dimensions.csv')
plot_dimensions_over_time(
    fig,
    axarr[1],
    ae_5min, 
    pca_5min,
    [],
)

plt.sca(axarr[2])
ae_10min = pd.read_csv(f'{results_dir}/monday/ae_10min_dimensions.csv')
pca_10min = pd.read_csv(f'{pca_results_dir}/monday/pca_10min_dimensions.csv')
plot_dimensions_over_time(
    fig,
    axarr[2],
    ae_10min,
    pca_10min,
    [],
)

fig.text(0.5, 0.01, 'Time of Day', ha='center', fontsize=8)
fig.text(0.01, 0.5, 'Estimated Dimension', va='center', rotation='vertical', fontsize=8)
fig.tight_layout()
plt.savefig(f'{output_dir}/monday_dimensions_over_time.png')
plt.close()

# Friday - Load data
fig, axarr = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6,6))
plt.sca(axarr[0])

ae_1min = pd.read_csv(f'{results_dir}/friday/ae_1min_dimensions.csv')
pca_1min = pd.read_csv(f'{pca_results_dir}/friday/pca_1min_dimensions.csv')
plot_dimensions_over_time(
    fig,
    axarr[0],
    ae_1min,
    pca_1min,
    [('2017-07-07 15:56:00', '2017-07-07 16:16:00')],
)

ae_5min = pd.read_csv(f'{results_dir}/friday/ae_5min_dimensions.csv')
pca_5min = pd.read_csv(f'{pca_results_dir}/friday/pca_5min_dimensions.csv')
plt.sca(axarr[1])
plot_dimensions_over_time(
    fig,
    axarr[1],
    ae_5min,
    pca_5min,
    [('2017-07-07 15:56:00', '2017-07-07 16:16:00')],
)

ae_10min = pd.read_csv(f'{results_dir}/friday/ae_10min_dimensions.csv')
pca_10min = pd.read_csv(f'{pca_results_dir}/friday/pca_10min_dimensions.csv')
plt.sca(axarr[2])
plot_dimensions_over_time(
    fig,
    axarr[2],
    ae_10min,
    pca_10min,
    [('2017-07-07 15:56:00', '2017-07-07 16:16:00')],
)

fig.text(0.5, 0.01, 'Time of Day', ha='center', fontsize=8)
fig.text(0.01, 0.5, 'Estimated Dimension', va='center', rotation='vertical', fontsize=8)
fig.tight_layout()
plt.savefig(f'{output_dir}/friday_dimensions_over_time.png')
plt.close()

###
# Dimensions over time heatmap
###
fig, axarr = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6,6))
plt.sca(axarr[0])

# Monday - Load data
ae_1min = pd.read_csv(f'{results_dir}/monday/ae_1min_dimensions.csv')
pca_1min = pd.read_csv(f'{pca_results_dir}/monday/pca_1min_dimensions.csv')
plot_dimensions_over_time_heatmap(
    fig,
    axarr[0],
    ae_1min, 
    pca_1min, 
    [],
)

plt.sca(axarr[1])
ae_5min = pd.read_csv(f'{results_dir}/monday/ae_5min_dimensions.csv')
pca_5min = pd.read_csv(f'{pca_results_dir}/monday/pca_5min_dimensions.csv')
plot_dimensions_over_time_heatmap(
    fig,
    axarr[1],
    ae_5min, 
    pca_5min,
    [],
)

plt.sca(axarr[2])
ae_10min = pd.read_csv(f'{results_dir}/monday/ae_10min_dimensions.csv')
pca_10min = pd.read_csv(f'{pca_results_dir}/monday/pca_10min_dimensions.csv')
plot_dimensions_over_time_heatmap(
    fig,
    axarr[2],
    ae_10min,
    pca_10min,
    [],
)

fig.text(0.5, 0.01, 'Time of Day', ha='center', fontsize=8)
fig.tight_layout()
plt.savefig(f'{output_dir}/monday_dimensions_heatmap.png')
plt.close()

# Friday

fig, axarr = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6,6))
plt.sca(axarr[0])

ae_1min = pd.read_csv(f'{results_dir}/friday/ae_1min_dimensions.csv')
pca_1min = pd.read_csv(f'{pca_results_dir}/friday/pca_1min_dimensions.csv')
plot_dimensions_over_time_heatmap(
    fig,
    axarr[0],
    ae_1min, 
    pca_1min, 
    [('2017-07-07 15:56:00', '2017-07-07 16:16:00')],
)

plt.sca(axarr[1])
ae_5min = pd.read_csv(f'{results_dir}/friday/ae_5min_dimensions.csv')
pca_5min = pd.read_csv(f'{pca_results_dir}/friday/pca_5min_dimensions.csv')
plot_dimensions_over_time_heatmap(
    fig,
    axarr[1],
    ae_5min, 
    pca_5min,
    [('2017-07-07 15:56:00', '2017-07-07 16:16:00')],
)

plt.sca(axarr[2])
ae_10min = pd.read_csv(f'{results_dir}/friday/ae_10min_dimensions.csv')
pca_10min = pd.read_csv(f'{pca_results_dir}/friday/pca_10min_dimensions.csv')
plot_dimensions_over_time_heatmap(
    fig,
    axarr[2],
    ae_10min,
    pca_10min,
    [('2017-07-07 15:56:00', '2017-07-07 16:16:00')],
)

fig.text(0.5, 0.01, 'Time of Day', ha='center', fontsize=8)
fig.tight_layout()
plt.savefig(f'{output_dir}/friday_dimensions_heatmap.png')
plt.close()
