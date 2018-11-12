#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function

import numpy as np
import chainer
import pandas as pd
from sklearn.model_selection import train_test_split
from chainer.datasets import tuple_dataset
import scipy.io as scio
import pandas


# ============== read data =================
def get_columns(source_array, column_list):
    result = []
    for row in source_array:
        a_row = []
        for column in column_list:
            a_row.append(row[column])
        result.append(a_row)
    return result


def read_mat(data_set):
    data = scio.loadmat('./datasets/' + data_set + '_o.mat')
    orgt = data['orgt']
    return orgt


def read_maxwell():
    data = pd.read_csv('./datasets/maxwell.txt', header=None)
    data = data.values[1:, 1:]
    data = np.log(np.asarray(data).astype('float32'))
    x_data, y_data = np.split(data, (25,), axis=1)
    return x_data, y_data


def read_opens():
    data = pd.read_csv('./datasets/opens.txt', header=None)
    data = data.values[:, 1:]
    data = np.log(np.asarray(data).astype('float32'))
    x_data, y_data = np.split(data, (7,), axis=1)
    return x_data, y_data


def read_miyazaki94():
    data = pd.read_csv('./datasets/miyazaki94.txt', header=None)
    data = data.values[:, 1:]
    x_data, y_data = np.split(data, (7,), axis=1)
    return x_data, y_data


def read_albrecht():
    data_set = 'albrecht'
    orgt = read_mat(data_set)
    x_data, y_data = np.split(orgt, (7,), axis=1)
    return x_data, y_data


def read_china(columns=17):
    data = pd.read_csv('./datasets/china.txt', header=None)
    x_data, y_data = np.split(data.values, (18,), axis=1)
    x_data = x_data[:, 1:]
    x_data = x_data[:, -columns:]
    return x_data, y_data


def read_kemerer(columns=6):
    data = pd.read_csv('./datasets/kemerer.txt', header=None)
    x_data, y_data = np.split(data.values, (7,), axis=1)
    x_data = x_data[:, 1:]
    x_data = x_data[:, -columns:]
    return x_data, y_data


def read_kitchenham(columns=[8]):
    data = pd.read_csv('./datasets/kitchenham.txt', header=None)
    x = get_columns(data.values, columns)
    y = get_columns(data.values, [5])
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


def read_cocnas():
    data = pd.read_csv('./datasets/cocnas.txt', header=None)
    data = data.values[1:, 1:]
    # log
    data = np.log(np.asarray(data).astype('float32'))
    x_data, y_data = np.split(data, (16,), axis=1)
    return x_data, y_data


# ============== end of read data =================
def get_splited_train_and_test_no_validataion(dataset='generate', train_size=0.7):
    in_size = 1
    # x_data, y_data
    if dataset == 'kemerer':
        x_data, y_data = read_kemerer(6)
        in_size = 6
    elif dataset == 'kitchenham':
        x_data, y_data = read_kitchenham([1, 4, 6, 8])
        in_size = 4
    elif dataset == 'cocnas':
        x_data, y_data = read_cocnas()
        in_size = 16
    elif dataset == 'china':
        x_data, y_data = read_china()
        in_size = 17
    elif dataset == 'maxwell':
        x_data, y_data = read_maxwell()
        in_size = 25
    elif dataset == 'albrecht':
        x_data, y_data = read_albrecht()
        in_size = 7
    elif dataset == 'opens':
        x_data, y_data = read_opens()
        in_size = 7
    elif dataset == 'miyazaki94':
        x_data, y_data = read_miyazaki94()
        in_size = 7
    else:
        print('The data set is not exists！！！！')
        return

    print(dataset, ' length = ', len(y_data), ' ndim = ', in_size)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=0, train_size=train_size)
    return x_train, x_test, y_train, y_test, in_size


def get_train_and_test_and_scale(dataset='china', median=100,  train_size=0.8):
    x_train, x_test, y_train, y_test, in_size = get_splited_train_and_test_no_validataion(dataset, train_size)
    train = tuple_dataset.TupleDataset(x_train.astype('float32'), y_train.astype('float32'))
    test = tuple_dataset.TupleDataset(x_test.astype('float32'), y_test.astype('float32'))

    return train, test, in_size, x_train.astype('float32'), y_train.astype('float32'), x_test.astype(
        'float32'), y_test.astype('float32')


def get_splited_train_and_test(dataset='china', train_size=0.5, validation_size=0):
    in_size = 1
    # x_data, y_data
    if dataset == 'kemerer':
        x_data, y_data = read_kemerer(6)
        in_size = 6
    elif dataset == 'kitchenham':
        x_data, y_data = read_kitchenham([1, 4, 6, 8])
        in_size = 4
    elif dataset == 'cocnas':
        x_data, y_data = read_cocnas()
        in_size = 16
    elif dataset == 'china':
        x_data, y_data = read_china()
        in_size = 17
    elif dataset == 'maxwell':
        x_data, y_data = read_maxwell()
        in_size = 25
    elif dataset == 'albrecht':
        x_data, y_data = read_albrecht()
        in_size = 7
    elif dataset == 'opens':
        x_data, y_data = read_opens()
        in_size = 7
    elif dataset == 'miyazaki94':
        x_data, y_data = read_miyazaki94()
        in_size = 7
    else:
        print('The data set is not exists！！！！')
        return

    print(dataset, ' length = ', len(y_data), ' ndim = ', in_size)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=0, train_size=train_size)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, random_state=0,
                                                                  train_size=(1-validation_size))
    return x_train, x_validation, x_test, y_train, y_validation, y_test, in_size


def get_train_and_test(dataset='kemerer', train_size=0.5, validation_size=0):
    x_train, x_validation, x_test, y_train, y_validation, y_test, in_size = \
        get_splited_train_and_test(dataset, train_size, validation_size)
    # tuple
    train = tuple_dataset.TupleDataset(x_train.astype('float32'), y_train.astype('float32'))
    validation = tuple_dataset.TupleDataset(x_validation.astype('float32'), y_validation.astype('float32'))
    test = tuple_dataset.TupleDataset(x_test.astype('float32'), y_test.astype('float32'))
    return train, validation, test, in_size, \
           x_train.astype('float32'), y_train.astype('float32'), \
           x_validation.astype('float32'), y_validation.astype('float32'), \
           x_test.astype('float32'), y_test.astype('float32')


def get_train_and_test_2_dim(dataset='kemerer', train_size=0.5, validation_size=0):
    x_train, x_validation, x_test, y_train, y_validation, y_test, in_size = \
        get_splited_train_and_test(dataset, train_size, validation_size)
    train = tuple_dataset.TupleDataset(tuple_dataset.TupleDataset(x_train.astype('float32')), y_train.astype('float32'))
    validation = tuple_dataset.TupleDataset(tuple_dataset.TupleDataset(x_validation.astype('float32')), y_validation.astype('float32'))
    test = tuple_dataset.TupleDataset(tuple_dataset.TupleDataset(x_test.astype('float32')), y_test.astype('float32'))
    return train, validation, test, in_size, \
           x_train.astype('float32').reshape((len(x_train), 1, len(x_train[0]))), y_train.astype('float32'), \
           x_validation.astype('float32').reshape((len(x_validation), 1, len(x_validation[0]))), y_validation.astype('float32'), \
           x_test.astype('float32').reshape((len(x_test), 1, len(x_test[0]))), y_test.astype('float32')


def get_train_and_test_no_validation(dataset='kemerer', train_size=0.7):
    x_train, x_test, y_train, y_test, in_size = get_splited_train_and_test_no_validataion(dataset, train_size)
    # tuple
    train = tuple_dataset.TupleDataset(x_train.astype('float32'), y_train.astype('float32'))
    test = tuple_dataset.TupleDataset(x_test.astype('float32'), y_test.astype('float32'))
    return train, test, in_size, x_train.astype('float32'), y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')


def get_train_and_test_2_dim_no_validation(dataset='generate', train_size=0.8):
    x_train, x_test, y_train, y_test, in_size = get_splited_train_and_test_no_validataion(dataset, train_size)

    train = tuple_dataset.TupleDataset(tuple_dataset.TupleDataset(x_train.astype('float32')), y_train.astype('float32'))
    test = tuple_dataset.TupleDataset(tuple_dataset.TupleDataset(x_test.astype('float32')), y_test.astype('float32'))
    return train, test, in_size, x_train.astype('float32').reshape((len(x_train), 1, len(x_train[0]))), \
           y_train.astype('float32'), x_test.astype('float32').reshape((len(x_test), 1, len(x_test[0]))), \
           y_test.astype('float32')


if __name__ == "__main__":
    x_data, y_data = read_miyazaki94()
    print('X.shape', x_data.shape)
    print('Y.shape', y_data.shape)
    print(x_data)
    print(y_data)

