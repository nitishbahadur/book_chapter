#!/usr/bin/env python

import argparse
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
from sklearn import preprocessing
import sklearn.model_selection
import sys
import torch.utils.data

class DatasetProcessor():
    '''
    Base class for any processors that will work on a dataset.
    In the future maybe this could also return some information about the change made:
        - Changed columns
        - Renames
        - Drops
    This may have some benefit to inspect how a dataset was altered in preprocessing.
    '''
    def __init__(self):
        pass

    def process(self, df):
        '''
        Perform processing on a pandas dataframe and then returns the updated dataframe.
        This default method just returns the original dataframe.
        '''
        return df

class MakeNiceColumnNames(DatasetProcessor):
    def __init__(self):
        super().__init__()

    def process(self, df):
        # Strip any leading spaces
        df.columns = df.columns.str.strip()

        # Make columns lower case
        df.columns = [col.lower() for col in df.columns]

        # Replace other contiguous spaces with an underscore
        df.columns = [re.sub('\s+', '_', col) for col in df.columns]

        # Special characters replaced by underscore maybe
        df.columns = [re.sub(r'\.+', '_', col) for col in df.columns]
        df.columns = [re.sub(r'/+', '_', col) for col in df.columns]
        df.columns = [re.sub(r'\\+', '_', col) for col in df.columns]
        df.columns = [re.sub(r':', '_', col) for col in df.columns]
        df.columns = [re.sub(r'-', '_', col) for col in df.columns]

        return df

class DropColumns(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        if self.target_cols:
            df = df.drop(columns=self.target_cols, axis=1)
        return df

class DropIPv6Addresses(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        for col in self.target_cols:
            # IPv6 addresses have colons in their format
            indices_to_drop = df[df[col].str.contains(':')].index
            df.drop(index=indices_to_drop, inplace=True)
        return df

class CleanBadValues(DatasetProcessor):
    def __init__(self):
        super().__init__()

    def process(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        return df

class ReplaceMissingValuesWithMean(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        print(df.columns)
        for col in self.target_cols:
            mean = df[(df[col] != '-') & (df[col] != np.nan)][col].astype(float).mean()
            df[col] = df[col].apply(lambda x: mean if x == '-' or x == np.nan else x)
            df[col] = df[col].astype(float)
        return df

class StripStringColumns(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        df[self.target_cols] = df[self.target_cols].str.strip()

class OneHotEncodeColumns(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        if self.target_cols:
            df = pd.get_dummies(df, columns=self.target_cols)
        return df

class MakeLabelColumnLast(DatasetProcessor):
    def __init__(self, target_cols=['label']):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        if self.target_cols:
            df = df[[c for c in df.columns if c not in self.target_cols] + self.target_cols]
        return df

class IpAddressToByteColumns(DatasetProcessor):
    def __init__(self, prefix='', target_col=None):
        super().__init__()
        self.target_col = target_col
        self.prefix = prefix

    def process(self, df):
        new_cols = [self.prefix + '_byte1', self.prefix + '_byte2', self.prefix + '_byte3', self.prefix + '_byte4']
        df[new_cols] = df[self.target_col].str.split( '.', expand=True)
        return df

class FilterPortsOver1024(DatasetProcessor):
    def __init__(self, target_col):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df.loc[ df[self.target_col] >= 1024, self.target_col] = 1024
        return df

class PrepareAttackBenignLabel(DatasetProcessor):
    def __init__(self, target_col='label'):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: 0 if x == 'BENIGN' else 1)
        return df

class PrepareAttackBenignLabelCtu13(DatasetProcessor):
    def __init__(self, target_col='label'):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: 1 if x.rfind('Botnet') != -1 else 0)
        return df

class PrepareAttackBenignLabelIot23(DatasetProcessor):
    def __init__(self, target_col='label'):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: 1 if x == 'Malicious' else 0)
        return df

class PrepareAttackBenignLabelUgr16(DatasetProcessor):
    def __init__(self, target_col='label'):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(lambda x: 0 if x == 'background' else 1)
        return df

class BinaryEncodeColumns(DatasetProcessor):
    def __init__(self, target_cols=None, new_cols=None, padding=0, wrapper_fn=None):
        super().__init__()
        self.target_cols=target_cols
        self.new_cols=new_cols
        self.padding = padding
        self.wrapper_fn = wrapper_fn

    def get_binary_string(self, bits, padding=0):
        binary_string = bin(int(bits)).replace('0b', '').zfill(padding)
        formatted_string = ''
        for bit in binary_string[:-1]:
            formatted_string += bit + ' ' 
        formatted_string += binary_string[-1]
        return formatted_string

    def process(self, df):
        if len(self.target_cols) != len(self.new_cols):
            print('WARNING:  target_cols and new_cols have different lengths.')
            return df

        for i in range(len(self.target_cols)):
            if self.wrapper_fn is not None:
                df[self.new_cols[i]] = df[self.target_cols[i]].map(lambda x: self.get_binary_string(self.wrapper_fn(x), padding=self.padding))
            else:
                df[self.new_cols[i]] = df[self.target_cols[i]].map(lambda x: self.get_binary_string(x, padding=self.padding))
        return df

class StringToColumns(DatasetProcessor):
    def __init__(self, target_cols=None, prefix=None):
        super().__init__()
        self.target_cols = target_cols
        self.prefix = prefix

    def get_field_names(self, prefix=None, how_many=None):
        if not prefix or not how_many:
            return []
        return [ prefix + str(x) for x in range(how_many) ]
        
    def process(self, df):
        for i in range(len(self.target_cols)):
            sample_col = df[self.target_cols[i]][0].replace(' ', '')
            how_many_columns = len(sample_col)
            cols = self.get_field_names(self.prefix[i], how_many_columns)
            df[cols] = df[self.target_cols[i]].str.split(' ', expand=True)
            df[cols] = df[cols].astype(int)
        return df

class AddNetworkClassColumns(DatasetProcessor):
    def __init__(self, target_cols=None, new_cols=None):
        super().__init__()
        self.target_cols = target_cols
        self.new_cols = new_cols

    def process(self, df):
        if len(self.target_cols) != len(self.new_cols):
            print('WARNING:  Number of target_cols and new_cols does not match.')
            return df

        for i in range(len(self.target_cols)):
            tmp_bytes_df = pd.DataFrame()
            tmp_bytes_df[['byte1', 'byte2', 'byte3', 'byte4']] = df[self.target_cols[i]].str.split('.', expand=True)
            tmp_bytes_df['class'] = tmp_bytes_df['byte1'].apply(
                lambda x: 1 if int(x) <= 127 else 2 if int(x) <= 191 else 3)
            df[self.new_cols[i]] = tmp_bytes_df['class']
        return df

class PerformIpNormalization(DatasetProcessor):
    def __init__(self, target_cols=None, prefix=None):
        super().__init__()
        self.target_cols = target_cols
        self.prefix = prefix

    def process(self, df):
        for i in range(len(self.target_cols)):
            tmp_df = pd.DataFrame()
            tmp_df[['byte1', 'byte2', 'byte3', 'byte4']] = df[self.target_cols[i]].str.split('.', expand=True)
            tmp_df[['byte1', 'byte2', 'byte3', 'byte4']] = \
                    tmp_df[['byte1', 'byte2', 'byte3', 'byte4']].astype(int).astype(CategoricalDtype(np.arange(256)))

            tmp_df[self.prefix[i] + 'network_byte1'] = tmp_df['byte1']

            tmp_df[self.prefix[i] + 'network_byte2'] = tmp_df['byte2']
            tmp_df.loc[tmp_df['byte1'].astype(int) <= 127, self.prefix[i] + 'network_byte2'] = 0

            tmp_df[self.prefix[i] + 'network_byte3'] = tmp_df['byte3']
            tmp_df.loc[tmp_df['byte1'].astype(int) <= 191, self.prefix[i] + 'network_byte3'] = 0

            tmp_df[self.prefix[i] + 'network_byte4'] = 0

            tmp_df[self.prefix[i] + 'host_byte1'] = 0

            tmp_df[self.prefix[i] + 'host_byte2'] = tmp_df['byte2']
            tmp_df.loc[tmp_df['byte1'].astype(int) > 127, self.prefix[i] + 'host_byte2'] = 0

            tmp_df[self.prefix[i] + 'host_byte3'] = tmp_df['byte3']
            tmp_df.loc[tmp_df['byte1'].astype(int) > 191, self.prefix[i] + 'host_byte3'] = 0

            tmp_df[self.prefix[i] + 'host_byte4'] = tmp_df['byte4']

            df[[
                self.prefix[i] + 'network_byte1',
                self.prefix[i] + 'network_byte2',
                self.prefix[i] + 'network_byte3',
                self.prefix[i] + 'network_byte4',
                self.prefix[i] + 'host_byte1',
                self.prefix[i] + 'host_byte2',
                self.prefix[i] + 'host_byte3',
                self.prefix[i] + 'host_byte4']] = tmp_df[[
                    self.prefix[i] + 'network_byte1',
                    self.prefix[i] + 'network_byte2',
                    self.prefix[i] + 'network_byte3',
                    self.prefix[i] + 'network_byte4',
                    self.prefix[i] + 'host_byte1',
                    self.prefix[i] + 'host_byte2',
                    self.prefix[i] + 'host_byte3',
                    self.prefix[i] + 'host_byte4']].astype(int).astype(CategoricalDtype(np.arange(256)))

        return df

class PerformPortNormalization(DatasetProcessor):
    def __init__(self, target_cols=None, prefix=None):
        super().__init__()
        self.target_cols = target_cols
        self.prefix = prefix

    def process(self, df):
        for i in range(len(self.target_cols)):
            col = self.target_cols[i]
            prefix = self.prefix[i]

            df[col] = df[col].apply(lambda x: bin(int(x))[2:].zfill(16))

            df[prefix + 'port_byte1'] = df[col].str[:8]
            df[prefix + 'port_byte1'] = df[prefix + 'port_byte1'].apply(lambda x: int(x, 2))
            df[prefix + 'port_byte1'] = df[prefix + 'port_byte1'].astype(CategoricalDtype(np.arange(256)))

            df[prefix + 'port_byte2'] = df[col].str[8:]
            df[prefix + 'port_byte2'] = df[prefix + 'port_byte2'].apply(lambda x: int(x, 2))
            df[prefix + 'port_byte2'] = df[prefix + 'port_byte2'].astype(CategoricalDtype(np.arange(256)))
        return df

class FilterPortsWellKnownOrNot(DatasetProcessor):
    def __init__(self, target_col=None):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        # Order is important here
        df.loc[ df[self.target_col] < 1024, self.target_col] = 0
        df.loc[ df[self.target_col] >= 1024, self.target_col] = 1
        return df

class FilterColumnsUsingRegex(DatasetProcessor):
    def __init__(self, regex='*'):
        super().__init__()
        self.regex = regex

    def process(self, df):
        df = df.filter(regex=self.regex)
        return df

class OnlyKeepTheseColumns(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols
    
    def process(self, df):
        if self.target_cols:
            df = df[self.target_cols]
        return df

class ReorderAndFilterColumnsUsingSubstring(DatasetProcessor):
    def __init__(self, substrings=None):
        super().__init__()
        self.substrings = substrings 

    def process(self, df):
        col_order = []
        for s in self.substrings:
            col_order = col_order + [col for col in df.columns if s in col]
        df = df[col_order]
        return df

class DropDuplicates(DatasetProcessor):
    def __init__(self):
        super().__init__()

    def process(self, df):
        df = df.drop_duplicates()
        return df

class IpAddressInternalOrExternal(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        # Not completely accurate as really based on netowrk class but should approximate it for now
        for col in self.target_cols:
            df[col] = df[col].apply(lambda x: 'internal' if int(x.split('.')[0]) >= 192 else 'external')
        return df

class MarkCommsTypeUsingPorts(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols
        self.port_map = {
            0.0:'system_defined',
            21.0:'ftp',
            22.0:'ssh',
            42.0:'dns',
            53.0:'dns',
            135.0:'dns',
            5353.0:'dns',
            80.0:'http',
            88.0:'kerberos',
            123.0:'ntp',
            137.0:'samba',
            138.0:'samba',
            139.0:'samba',
            389.0:'ldap',
            3268.0:'ldap',
            443.0:'https',
            445.0:'ms-smb',
            465.0:'smtps',
            'registered':'registered',
            'user_application':'user_application',
            'nan':'other',
            np.NaN:'other',
        }

    def map_general_ports(self, val):
        if type(val) == str and val.rfind('0x') != -1:
            val = int(val, base=16)
        else:
            val = int(val)

        if val in self.port_map.keys():
            return val 

        if val >= 1024 and val <= 49151:
            return 'registered'

        if val >= 49152:
            return 'user_application'

        # If none of our cases are met, just return the same value
        return val

    def process(self, df):
        for col in self.target_cols:
            df[col] = df[col].apply(lambda x: self.map_general_ports(x))
            df[col] = df[col].map(self.port_map)
            df[col] = df[col].astype(str)
        return df

class SortByDateColumn(DatasetProcessor):
    def __init__(self, target_col=None):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = pd.to_datetime(df[self.target_col])
        df = df.sort_values(by=self.target_col)
        return df

class AddTimeOfDayToDateColumn(DatasetProcessor):
    def __init__(self, target_col=None):
        super().__init__()
        self.target_col = target_col

    def process(self, df):
        df[self.target_col] = df[self.target_col].apply(
            lambda x:  x + ' PM' 
            if ' 12:' in x or
                ' 01:' in x or ' 1:' in x or
                ' 02:' in x or ' 2:' in x or
                ' 03:' in x or ' 3:' in x or
                ' 04:' in x or  ' 4:'in x or
                ' 05:' in x or ' 5:' in x
            else x + ' AM'
            )
        return df

class FormatDateTime(DatasetProcessor):
    def __init__(self, target_col=None, original_format=None, target_format=None):
        super().__init__()
        self.target_col = target_col
        self.original_format = original_format
        self.target_format = target_format

    def process(self, df):
        df[self.target_col] = pd.to_datetime(df[self.target_col], format=self.original_format)
        return df

class SampleWithStratify(DatasetProcessor):
    def __init__(self, num_samples=50000):
        super().__init__()
        self.num_samples = num_samples

    def process(self, df):
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(int(np.rint(self.num_samples *  len(x)/ len(df))))).sample(frac=1).reset_index(drop=True)
        return df

class DropRowsWithNegativeValues(DatasetProcessor):
    def __init__(self, target_cols=None):
        super().__init__()
        self.target_cols = target_cols

    def process(self, df):
        df = df[(df[self.target_cols] >= 0).all(1)]
        return df
