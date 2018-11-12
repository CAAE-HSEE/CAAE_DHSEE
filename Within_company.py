#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L
import read_data_validation
import criteria
import math
from chainer import serializers, initializers


class ConvolutionalModel(chainer.Chain):
    def __init__(self, in_size, n_units=15):
        super(ConvolutionalModel, self).__init__()
        with self.init_scope():
            initial_w = initializers.One(dtype=np.float32)
            self.conv = L.ConvolutionND(ndim=1, in_channels=1, out_channels=n_units, ksize=in_size, stride=1, pad=0,
                                        initialW=initial_w)
            self.fc = L.Linear(n_units, 1)

    def __call__(self, x):
        h1 = F.leaky_relu(self.conv(x))
        h2 = self.fc(h1)
        return h2


def main(epoch=3000, batchsize=6, dataset='kitchenham', train_size=0.5,
         validation_size=0.5, validation_patience_original=1000):

    # Read the data
    data_train, data_validation, data_test, data_in_size, \
    data_x_train, data_y_train, \
    data_x_validation, data_y_validation, \
    data_x_test, data_y_test = read_data_validation.get_train_and_test_2_dim(dataset, train_size, validation_size)

    # Set up a neural network to train
    model = ConvolutionalModel(data_in_size)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(data_train, batchsize)

    loss_all = []

    def loss_fun(x, y):
        y_hat = model(x)
        y_hat = y_hat.reshape((len(y_hat), 1))
        loss = F.mean_absolute_error(y_hat, y)
        loss_all.append(loss.data)
        return loss

    def test_pred(data_y_test):
        y_test_predict = model(data_x_test).data
        if dataset in ['cocnas', 'maxwell', 'opens']:
            y_test_predict = np.power(math.e, y_test_predict)
            data_y_test = np.power(math.e, data_y_test)
        test_pred25 = criteria.pred25(data_y_test, y_test_predict)
        print('test_pred25 = ', test_pred25)
        return test_pred25

    def test_lsd(data_y_test):
        y_test_predict = model(data_x_test).data
        if dataset in ['cocnas', 'maxwell', 'opens']:
            y_test_predict = np.power(math.e, y_test_predict)
            data_y_test = np.power(math.e, data_y_test)
        test_lsd = criteria.lsd(data_y_test, y_test_predict)
        print('test_lsd = ', test_lsd)
        return test_lsd

    def best_mdmre(data_y_test):
        y_test_predict = model(data_x_test).data
        if dataset in ['cocnas', 'maxwell', 'opens']:
            y_test_predict = np.power(math.e, y_test_predict)
            data_y_test = np.power(math.e, data_y_test)
        test_mdre = criteria.mdmre(data_y_test, y_test_predict)
        print('test_mdre = ', test_mdre)
        return test_mdre

    def validation_pred(data_y_validation):
        y_validation_predict = model(data_x_validation).data
        if dataset in ['cocnas', 'maxwell', 'opens']:
            y_validation_predict = np.power(math.e, y_validation_predict)
            data_y_validation = np.power(math.e, data_y_validation)
        validation_pred25 = criteria.pred25(data_y_validation, y_validation_predict)
        print('validation_pred25 = ', validation_pred25)
        return validation_pred25

    print("Training...")
    chainer.using_config('train', True)
    validation_frequency = 1
    validation_patience = validation_patience_original
    best_validation_pred25 = 0
    bset_test_pred25 = 0
    best_mdre = 0
    best_lsd = 0

    while train_iter.epoch < epoch:
        # get batch
        batch = train_iter.next()

        # train and update
        x_array, t_array = convert.concat_examples(batch)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        optimizer.update(loss_fun, x, t)

        # validation
        validation_patience -= 1
        if train_iter.epoch % validation_frequency == 0:
            # compute pred25
            validation_pred25 = validation_pred(data_y_validation)
            if validation_pred25 >= best_validation_pred25:
                best_validation_pred25 = validation_pred25
                # test on the test dataset
                test_pred25 = test_pred(data_y_test)
                if test_pred25 > bset_test_pred25:
                    bset_test_pred25 = test_pred25
                    best_mdre = best_mdmre(data_y_test)
                    best_lsd = test_lsd(data_y_test)
                    # save model
                    if True:
                        serializers.save_npz('./models/single_' + dataset + '.model', model)
                validation_patience = validation_patience_original
        if validation_patience == 0:
            break

    chainer.using_config('train', False)
    print("--------------- Train finished ----------------------\n\n\n")

    # Test
    y_predict_data = model(data_x_test).data
    if dataset in ['cocnas', 'maxwell', 'opens']:
        print('power')
        y_predict_data = np.power(math.e, y_predict_data)
        data_y_test = np.power(math.e, data_y_test)

    print('====================================================')
    print('Test criteria of', dataset)

    print('Test_MdMRE', best_mdre)
    print('Test_pred25', bset_test_pred25)
    print('Test_LSD', best_lsd)

    plt.clf()
    t = range(len(y_predict_data))
    plt.plot(t, np.squeeze(data_y_test, axis=1))
    plt.plot(t, np.squeeze(y_predict_data, axis=1))
    plt.title(u'Test on '+dataset+ ' Pred25=' + str(bset_test_pred25))
    plt.show()

if __name__ == '__main__':
    main(epoch=20000, batchsize=50, dataset='china',
         train_size=0.7, validation_size=0.5, validation_patience_original=200)