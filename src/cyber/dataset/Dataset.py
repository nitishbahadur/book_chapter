#!/usr/bin/env python

import argparse
import copy
import inquirer
import ipaddress
import math
import numpy as np
import os
import pandas as pd
from pandas.api.types import CategoricalDtype
import ipdb
import pickle
import re
from scipy.io import arff
from sklearn import preprocessing
import sklearn.model_selection
import sys
import torch.utils.data
import dataset.DatasetProcessors

###
# What is a dataset?
# - Contains a base source of data
# - Put into a standard format to enable processing
# - A number of transformations performed on the dataset
# - Able to tell us how many fields it has (with and without the label)
###

class Dataset():
    def __init__(self, name='dataset', dataset_source=None, encoding='ISO-8859-1', sep=',', processors=None):
        self.name = name
        self.processors = processors
        self.normalized_columns = []
        self.dataset_source = dataset_source
        self.original_data = self.load_source_data(self.dataset_source, encoding=encoding, sep=sep)
        self.training_means = None
        self.training_stds = None
        self.scaler = None

        # Is this a reference or do I need a deep copy here?
        # For current usage probably doesn't matter
        self.processed_data = copy.deepcopy(self.original_data)

        # Only generated if the client calls the split method
        self.training_data = pd.DataFrame()
        self.validation_data = pd.DataFrame()
        self.test_data = pd.DataFrame()

    def load_source_data(self, dataset_source, encoding='ISO-8859-1', sep=','):
        '''
        dataset_source:  The path to a valid dataset to load.
                         Expected to be a csv file for now.
        Loads the data found in file [dataset_source] into a Pandas DF.
        '''
        if dataset_source.endswith('.arff'):
            tmp_data = arff.loadarff(dataset_source)
            df = pd.DataFrame(tmp_data[0])
        else:
            df = pd.read_csv(dataset_source, encoding=encoding, sep=sep, skip_blank_lines=True)
        return df

    def add_source_data(self, dataset_source, encoding='ISO-8859-1', sep=','):
        new_df = self.load_source_data(dataset_source)
        self.original_data = pd.concat([self.original_data, new_df], ignore_index=True)
        self.processed_data = copy.deepcopy(self.original_data)

    def perform_processing(self):
        '''
        Takes the original datasource file and generates processed version
        using the processors contained in the dataset.
        Processors are handled in their original order.
        '''
        for processor in self.processors:
            self.processed_data = processor.process(self.processed_data)

        # Make the label column last
        self.processed_data = dataset.DatasetProcessors.MakeLabelColumnLast().process(self.processed_data)
        return self.processed_data

    def append_processor(self, processor=None, process=True):
        '''
        Appends a processor to the end of the processing list
        processor:  The processor to append.
        process:  True if the processor should be applied.
                  After its application the label column remains last.
        '''
        if processor:
            self.processors.append(processor)

            if process:
                if not self.processed_data.empty:
                    self.processed_data = processor.process(self.processed_data)
                    self.processed_data = dataset.DatasetProcessors.MakeLabelColumnLast().process(self.processed_data)

                if not self.test_data.empty:
                    self.test_data = processor.process(self.test_data)
                    self.test_data = dataset.DatasetProcessors.MakeLabelColumnLast().process(self.test_data)

                if not self.validation_data.empty:
                    self.validation_data = processor.process(self.validation_data)
                    self.validation_data = dataset.DatasetProcessors.MakeLabelColumnLast().process(self.validation_data)

                if not self.training_data.empty:
                    self.training_data = processor.process(self.training_data)
                    self.training_data = dataset.DatasetProcessors.MakeLabelColumnLast().process(self.training_data)

        return self.processed_data

    def print_dataset_info(self):
        '''
        Returns the following statistics regarding the dataset object:
        - Name
        - dataset_source
        - Processed number of features
        '''
        print('Dataset Name:  ' + self.name)
        print('Original Source:  ' + self.dataset_source)
        print('Number of Features:  ' + str(len(self.processed_data.columns) - 1))

    def number_of_features(self):
        return len(self.processed_data.columns) - 1

    def number_of_rows(self):
        return len(self.processed_data)

    def split(self, split_point_1=0.15, split_point_2=0.3, shuffle=True, random_state=42):
        if shuffle:
            self.processed_data = self.processed_data.sample(frac=1, random_state=random_state)

        if split_point_2 == 0.0:
            splits = [int(split_point_1 * self.number_of_rows())]
            self.test_data , self.training_data = np.split(self.processed_data, splits)
            self.validation_data = self.test_data
        else:
            splits = [int(split_point_1 * self.number_of_rows()), int(split_point_2 * self.number_of_rows())]
            self.test_data , self.validation_data, self.training_data = np.split(self.processed_data, splits)

        return self.test_data, self.validation_data, self.training_data

    # Expects 3 date intervals; Assumes 23 hour clock
    def split_by_dates(self, target_col, training_interval, validation_interval, testing_interval):
        self.training_data = self.processed_data.loc[
            ((self.processed_data[target_col] >= training_interval[0]) & 
            (self.processed_data[target_col] <= training_interval[1])),:
        ].copy()

        self.validation_data = self.processed_data.loc[
            ((self.processed_data[target_col] >= validation_interval[0]) & 
            (self.processed_data[target_col] <= validation_interval[1])),:
        ].copy()

        self.test_data = self.processed_data.loc[
            ((self.processed_data[target_col] >= testing_interval[0]) & 
            (self.processed_data[target_col] <= testing_interval[1])),:
        ].copy()

        return self.test_data, self.validation_data, self.training_data

    def normalize_columns(self, target_cols=None, scaler=None):
        self.normalized_columns = target_cols
        if self.training_data.empty:
            print('WARNING:  No training data.  Skipped normalization')
            return self.test_data, self.validation_data, self.training_data

        # Do not create a new scaler if one was passed in
        if not scaler:
            scaler = sklearn.preprocessing.StandardScaler().fit(self.training_data[target_cols])

        # Store the scaler incase we have
        # disjoint datasets (two different dataset objects) with
        # one used as training and the other as test/validation
        # In this case, we should use the scaler from the training
        # data for normalization
        self.scaler = scaler

        self.training_data[target_cols] = self.scaler.transform(self.training_data[target_cols])
        self.validation_data[target_cols] = self.scaler.transform(self.validation_data[target_cols])
        self.test_data[target_cols] = self.scaler.transform(self.test_data[target_cols])

        return self.test_data, self.validation_data, self.training_data

    def normalize_columns_min_max(self, target_cols=None, scaler=None):
        self.normalized_columns = target_cols
        if self.training_data.empty:
            print('WARNING:  No training data.  Skipped normalization')
            return self.test_data, self.validation_data, self.training_data

        # Do not create a new scaler if one was passed in
        if not scaler:
            scaler = sklearn.preprocessing.MinMaxScaler().fit(self.training_data[target_cols])

        # Store the scaler incase we have
        # disjoint datasets (two different dataset objects) with
        # one used as training and the other as test/validation
        # In this case, we should use the scaler from the training
        # data for normalization
        self.scaler = scaler

        self.training_data[target_cols] = self.scaler.transform(self.training_data[target_cols])
        self.validation_data[target_cols] = self.scaler.transform(self.validation_data[target_cols])
        self.test_data[target_cols] = self.scaler.transform(self.test_data[target_cols])

        return self.test_data, self.validation_data, self.training_data

    def _znorm(self, df, means, stds):
        # If all values are zero we will get a divide by zero
        # So we just set it to 0.0 in this case
        df = (df - means) / stds
        df.replace(np.NaN, 0.0, inplace=True)
        return df

    def to_csv(self, output_dir='.'):
        os.makedirs(output_dir, exist_ok=True)

        if not self.test_data.empty:
            self.test_data.to_csv(output_dir + '/test_' + self.name + '.csv', index=False)

        if not self.validation_data.empty:
            self.validation_data.to_csv(output_dir + '/validation_' + self.name + '.csv', index=False)

        if not self.training_data.empty:
            self.training_data.to_csv(output_dir + '/train_' + self.name + '.csv', index=False)

        if self.test_data.empty and self.validation_data.empty and self.training_data.empty:
            if not self.processed_data.empty:
                self.processed_data.to_csv(output_dir + '/' + self.name + '.csv', index=False)

    def reprocess(self):
        # Clear out existing data except for original
        self.processed_data = copy.deepcopy(self.original_data)
        self.training_data = pd.DataFrame()
        self.validation_data = pd.DataFrame()
        self.test_data = pd.DataFrame()

        # Process the data again
        self.perform_processing()

    def reindex(self):
        self.training_data.index = np.arange(len(self.training_data))
        self.validation_data.index = np.arange(len(self.validation_data))
        self.test_data.index = np.arange(len(self.test_data))

    @staticmethod
    def load(dataset_path=None):
        if dataset_path:
            if os.path.exists(dataset_path):
                # Load dataset 
                with open(dataset_path, 'rb') as f:
                    ds = pickle.load(f)
                ds.reindex()
                return ds
