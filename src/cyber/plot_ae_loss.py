#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

plot_dict = {
    'data_files':[
        './output/20210206_ae_l1/ae_0.0001/4000_ae_1min_loss.csv',
        './output/20210206_ae_l1/ae_0.0002/4000_ae_1min_loss.csv',
        './output/20210206_ae_l1/ae_0.001/4000_ae_1min_loss.csv',
        './output/20210206_ae_l1/ae_0.0015/4000_ae_1min_loss.csv',
        './output/20210206_ae_l1/ae_0.01/4000_ae_1min_loss.csv',
    ],
    'legend':[
        r'$\lambda = 1e-04$',
        r'$\lambda = 2e-04$',
        r'$\lambda = 1e-03$',
        r'$\lambda = 1.5e-03$',
        r'$\lambda = 1e-02$',
    ],
    'file_path':'./output/20210206_results/l1_comparison_test.png',

}

###
# Loss for different l1 values
###

plt.figure()

for data_file in plot_dict['data_files']:
    df = pd.read_csv(data_file)
    plt.plot(df.epoch, df.val_loss)

plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.xlim(0,1000)
plt.legend(plot_dict['legend'])
plt.savefig(plot_dict['file_path'])
plt.close()
