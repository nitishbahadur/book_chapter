#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

###
# Update plot data as needed
###

plot_dict = {
    'legend':[
        r'AE $\lambda = 1e-04$',
        r'AE $\lambda = 2e-04$',
        r'AE $\lambda = 1e-03$',
        r'AE $\lambda = 1.5e-03$',
        r'AE $\lambda = 1e-02$',
        #r'PCA',
        ],
    'scree_data_files':[
        './output/20210206_ae_l1/ae_0.0001/ae_scree_data.csv',
        './output/20210206_ae_l1/ae_0.0002/ae_scree_data.csv',
        './output/20210206_ae_l1/ae_0.001/ae_scree_data.csv',
        './output/20210206_ae_l1/ae_0.0015/ae_scree_data.csv',
        './output/20210206_ae_l1/ae_0.01/ae_scree_data.csv',
        #'./output/20210206_pca_de/pca_scree_data.csv',
    ],
    'output_path':'./output/20210206_results/combined_scree_plot_ae_only.png',
}

###
# Combined scree plot
###
plt.figure()

# Load and plot data
for data_file in plot_dict['scree_data_files']:
    df = pd.read_csv(data_file)
    plt.plot(df.dimension, df.normalized_sv) 

plt.xlabel('Intrinsic Dimension')
plt.ylabel('SV / SVP')
plt.legend(plot_dict['legend'])
plt.xlim((0,25))
plt.savefig(plot_dict['output_path'])
plt.close()
