#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.dataset import convert
import read_data_validation
from chainer import initializers, serializers
from matplotlib import pylab as plt
import criteria
import pandas
import math


class EncoderRegressionModel(chainer.Chain):
    def __init__(self, in_size=[1, 2], encoder_n_units=10, regression_n_units=10, common_out_size=10):
        super(EncoderRegressionModel, self).__init__()
        with self.init_scope():
            initial_w = initializers.HeNormal()

            # Encoder 1
            self.encoder11 = L.Linear(in_size[0], encoder_n_units, initialW=initial_w)
            self.encoder12 = L.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.encoder13 = L.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.encoder14 = L.Linear(encoder_n_units, common_out_size, initialW=initial_w)

            # Encoder 2
            self.encoder21 = L.Linear(in_size[1], encoder_n_units, initialW=initial_w)
            self.encoder22 = L.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.encoder23 = L.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.encoder24 = L.Linear(encoder_n_units, common_out_size, initialW=initial_w)

            # Decoder 1
            self.decoder11 = L.Linear(common_out_size, encoder_n_units, initialW=initial_w)
            self.decoder12 = L.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.decoder13 = L.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.decoder14 = L.Linear(encoder_n_units, in_size[0], initialW=initial_w)

            # Regression 2
            self.regression21 = L.Linear(common_out_size, regression_n_units, initialW=initial_w)
            self.regression22 = L.Linear(regression_n_units, regression_n_units, initialW=initial_w)
            self.regression23 = L.Linear(regression_n_units, regression_n_units, initialW=initial_w)
            self.regression24 = L.Linear(regression_n_units, 1, initialW=initial_w)

    def encoder1_forward(self, x):
        # Encoder1
        h_encoder_11 = F.leaky_relu(self.encoder11(x))
        h_encoder_12 = F.leaky_relu(self.encoder12(h_encoder_11))
        h_encoder_13 = F.leaky_relu(self.encoder13(h_encoder_12))
        encoder_1_output = F.leaky_relu(self.encoder14(h_encoder_13))
        return encoder_1_output

    def encoder2_forward(self, x):
        # Encoder2
        h_encoder_21 = F.leaky_relu(self.encoder21(x))
        h_encoder_22 = F.leaky_relu(self.encoder22(h_encoder_21))
        h_encoder_23 = F.leaky_relu(self.encoder23(h_encoder_22))
        encoder_2_output = F.leaky_relu(self.encoder24(h_encoder_23))
        return encoder_2_output

    def decoder_forward(self, x):
        # Regression 1
        h_decoder_11 = F.leaky_relu(self.decoder11(x))
        h_decoder_12 = F.leaky_relu(self.decoder12(h_decoder_11))
        h_decoder_13 = F.leaky_relu(self.decoder13(h_decoder_12))
        decoder_1_output = F.leaky_relu(self.decoder14(h_decoder_13))
        return decoder_1_output

    def regression2_forward(self, x):
        # Regression 2
        h_regression_21 = F.leaky_relu(self.regression21(x))
        h_regression_22 = F.leaky_relu(self.regression22(h_regression_21))
        h_regression_23 = F.leaky_relu(self.regression23(h_regression_22))
        regression_2_output = F.leaky_relu(self.regression24(h_regression_23))
        return regression_2_output

    def forward_1(self, x1):
        encoder_1_output = self.encoder1_forward(x1)
        regression_1_output = self.decoder_forward(encoder_1_output)
        return regression_1_output

    def forward_2(self, x2):
        encoder_2_output = self.encoder2_forward(x2)
        regression_2_output = self.regression2_forward(encoder_2_output)
        return regression_2_output

    def __call__(self, x1, x2):
        encoder_1_output = self.encoder1_forward(x1)
        encoder_2_output = self.encoder2_forward(x2)

        decoder_1_output = self.decoder_forward(encoder_1_output)
        regression_2_output = self.regression2_forward(encoder_2_output)

        return encoder_1_output, encoder_2_output, decoder_1_output, regression_2_output


class Discriminator(chainer.Chain):
    def __init__(self, in_size, n_units=10):
        super(Discriminator, self).__init__()
        with self.init_scope():
            initial_w = initializers.HeNormal()
            self.l1 = L.Linear(in_size, n_units, initialW=initial_w)  # n_in -> n_units
            self.l2 = L.Linear(n_units, n_units, initialW=initial_w)
            self.l3 = L.Linear(n_units, 1)

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        y = F.sigmoid(self.l3(h2))
        return y


def main(encoder_n_units=32, common_size=16, regression_n_units=32, discriminator_n_units=32, batch_size=[6, 20],
         epoch=3000, data_set_name=['china', 'kitchenham'], validation_patience_original=1000, train_size=0.7,
         save_code=False, show_image=True):

    print("Reading data...")
    train = []
    validation = []
    test = []
    in_size = []
    x_train = []
    y_train = []
    x_validation = []
    y_validation = []
    x_test = []
    y_test = []

    for i in range(len(data_set_name)):
        a_name = data_set_name[i]

        a_train, a_validation, a_test, a_in_size, a_x_train, a_y_train, a_x_validation, a_y_validation, a_x_test, a_y_test = \
            read_data_validation.get_train_and_test(dataset=a_name, train_size=train_size, validation_size=0.5)

        train.append(a_train)
        validation.append(a_validation)
        test.append(a_test)
        in_size.append(a_in_size)
        x_train.append(a_x_train)
        y_train.append(a_y_train)
        x_validation.append(x_validation)
        y_validation.append(y_validation)
        x_test.append(a_x_test)
        y_test.append(a_y_test)

    # Prepare the train iter.
    train_iter = []
    for i in range(len(data_set_name)):
        a_train_iter = chainer.iterators.SerialIterator(train[i], batch_size[i])
        train_iter.append(a_train_iter)

    # Build model
    print("Building model...")
    model = EncoderRegressionModel(in_size=in_size, encoder_n_units=encoder_n_units,
                                   regression_n_units=regression_n_units,
                                   common_out_size=common_size)
    model_optimizer = chainer.optimizers.Adam()
    model_optimizer.setup(model)
    # Build Discriminator
    discriminator = Discriminator(common_size, discriminator_n_units)
    discriminator_optimizer = chainer.optimizers.SGD(lr=0.001)
    discriminator_optimizer.setup(discriminator)

    def test_pred(data_index):
        x_array, t_array = convert.concat_examples(test[data_index])
        x = chainer.Variable(x_array)
        y_test_predict = model.forward_2(x).data
        if data_set_name[data_index] in ['cocnas', 'maxwell', 'opens']:
            y_test_predict = np.power(math.e, y_test_predict)
            t_array = np.power(math.e, t_array)
        test_pred25 = criteria.pred25(t_array, y_test_predict)
        print('test_pred25 = ', test_pred25)
        return test_pred25

    def validation_pred(data_index):
        x_array, t_array = convert.concat_examples(validation[data_index])
        x = chainer.Variable(x_array)
        y_validation_predict = model.forward_2(x).data
        if data_set_name[data_index] in ['cocnas', 'maxwell', 'opens']:
            y_validation_predict = np.power(math.e, y_validation_predict)
            t_array = np.power(math.e, t_array)
        validation_pred25 = criteria.pred25(t_array, y_validation_predict)
        print('validation_pred25 = ', validation_pred25)
        return validation_pred25

    # train
    def discriminator_loss_fun(x1, x2, y1, y2):
        y1_hat = discriminator(x1)
        # print("Dis on data_set 1")
        # print(y1_hat.data[:5])
        y1_hat = y1_hat.reshape(len(y1_hat))
        loss1 = F.sigmoid_cross_entropy(y1_hat, y1)
        y2_hat = discriminator(x2)
        # print("Dis on data_set 2")
        # print(y2_hat.data[:5])
        y2_hat = y2_hat.reshape(len(y2_hat))
        loss2 = F.sigmoid_cross_entropy(y2_hat, y2)
        loss = loss1 + loss2
        # print("Discriminator loss = ", loss.data)
        dis_loss.append(loss.data)
        return loss

    # Loss function
    def loss_fun(x1, x2, y1, y2, label1, label2):
        encoder_1_output, encoder_2_output, decoder_1_output, regression_2_output = model(x1, x2)
        regression_2_output = regression_2_output.reshape((len(regression_2_output), 1))
        decoder_1_loss = F.mean_absolute_error(decoder_1_output, x1)
        regression_2_loss = F.mean_absolute_error(regression_2_output, y2)

        # Generator loss
        y1_hat = discriminator(encoder_1_output)
        y1_hat = y1_hat.reshape(len(y1_hat))
        encoder_1_loss = F.sigmoid_cross_entropy(y1_hat, label1)
        y2_hat = discriminator(encoder_2_output)
        y2_hat = y2_hat.reshape(len(y2_hat))
        encoder_2_loss = F.sigmoid_cross_entropy(y2_hat, label2)

        loss = decoder_1_loss + regression_2_loss * 10 + encoder_1_loss + encoder_2_loss
        loss_all.append(loss.data)
        # print("Generator loss = ", loss.data)
        return loss

    print("Training...")
    chainer.using_config('train', True)
    running = True
    # Storage the loss
    loss_all = []
    dis_loss = []
    validation_frequency = 1
    validation_patience = validation_patience_original
    best_validation_pred25 = 0
    bset_test_pred25 = 0

    # running
    while running:
        running_count = 0
        for i in range(len(data_set_name)):
            if train_iter[i].epoch < epoch:
                running_count += 1
            if running_count == 0:
                running = False

        # get batch
        batch1 = train_iter[0].next()
        x_array, t_array = convert.concat_examples(batch1)
        input_x1 = chainer.Variable(x_array)
        input_y1 = chainer.Variable(t_array)

        batch2 = train_iter[1].next()
        x_array, t_array = convert.concat_examples(batch2)
        input_x2 = chainer.Variable(x_array)
        input_y2 = chainer.Variable(t_array)

        # Train Discriminator on the real data
        encoder_1_output, encoder_2_output, regression_1_output, regression_2_output = model(input_x1, input_x2)
        zeros = np.zeros(len(encoder_1_output), dtype=np.int32)
        ones = np.ones(len(encoder_2_output), dtype=np.int32)
        discriminator_optimizer.update(discriminator_loss_fun, encoder_1_output, encoder_2_output, zeros, ones)

        # Train Generator
        zeros = np.zeros(len(encoder_2_output), dtype=np.int32)
        ones = np.ones(len(encoder_1_output), dtype=np.int32)
        model_optimizer.update(loss_fun, input_x1, input_x2, input_y1, input_y2, ones, zeros)

        # validation
        validation_patience -= 1
        if train_iter[1].epoch % validation_frequency == 0:
            # compute pred25
            validation_pred25 = validation_pred(1)
            if validation_pred25 >= best_validation_pred25:
                best_validation_pred25 = validation_pred25
                # test on the test dataset
                test_pred25 = test_pred(1)
                if test_pred25 > bset_test_pred25:
                    bset_test_pred25 = test_pred25
                    # save model
                    if True:
                        serializers.save_npz('./models/multi_' + data_set_name[1] + '.model', model)
                validation_patience = validation_patience_original
        if validation_patience == 0:
            break

    chainer.using_config('train', False)
    print("--------------- Train finished ----------------\n\n")

    # Save Code, including the train, validation and test.
    if save_code is True:
        data_index = 0
        x_array, t_array = convert.concat_examples(train[data_index])
        x = chainer.Variable(x_array)
        code_train = model.encoder1_forward(x)
        data1_train_code = code_train.data

        x_array, t_array = convert.concat_examples(validation[data_index])
        x = chainer.Variable(x_array)
        code_validation = model.encoder1_forward(x)
        data1_validation_code = code_validation.data

        x_array, t_array = convert.concat_examples(test[data_index])
        x = chainer.Variable(x_array)
        code_test = model.encoder1_forward(x)
        data1_test_code = code_test.data

        data_index = 1
        x_array, t_array = convert.concat_examples(train[data_index])
        x = chainer.Variable(x_array)
        code_train = model.encoder2_forward(x)
        data2_train_code = code_train.data

        x_array, t_array = convert.concat_examples(validation[data_index])
        x = chainer.Variable(x_array)
        code_validation = model.encoder2_forward(x)
        data2_validation_code = code_validation.data

        x_array, t_array = convert.concat_examples(test[data_index])
        x = chainer.Variable(x_array)
        code_test = model.encoder2_forward(x)
        data2_test_code = code_test.data

        code = np.vstack((data1_train_code, data1_validation_code, data1_test_code, data2_train_code,
                          data2_validation_code, data2_test_code))
        print("Code.shape is ", code.shape)
        gen_data = pandas.DataFrame(code)
        gen_data.to_csv('./data/prevModel.csv')

    print('\n\nCriteria Test------------------')

    # Show the loss and the Test result.
    if show_image:
        # Loss
        plt.clf()
        plt.subplot(211)
        data_index = 1
        x_array, t_array = convert.concat_examples(test[data_index])
        x = chainer.Variable(x_array)
        y_predict_data = model.forward_2(x)
        t = range(len(y_predict_data))
        plt.plot(t, np.squeeze(t_array, axis=1))
        plt.plot(t, np.squeeze(y_predict_data.data, axis=1))

        # criteria
        pred25 = criteria.pred25(t_array, y_predict_data.data)
        print("Test on ", data_set_name[data_index], " Pred25 = ",pred25)
        plt.title(u'Test on ' + data_set_name[data_index] + ' Pred25=' + str(pred25))

        plt.subplot(223)
        t = range(len(loss_all))
        plt.plot(t, np.squeeze(loss_all, axis=1))
        plt.title(u'Model loss')

        plt.subplot(224)
        t = range(len(dis_loss))
        plt.plot(t, np.squeeze(dis_loss, axis=1))
        plt.title(u'Discriminator loss')
        plt.show()


if __name__ == '__main__':
    main(encoder_n_units=32, common_size=16, regression_n_units=32, discriminator_n_units=32, batch_size=[15, 15],
         epoch=5000, data_set_name=['cocnas', 'opens'], train_size=0.7, save_code=True, show_image=True,
         validation_patience_original=50)