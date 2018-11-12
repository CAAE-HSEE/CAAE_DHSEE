#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from chainer import Variable
import chainer.functions as F
import math


def mre(effort, effort_hat):
    mre_value = np.fabs(np.fabs(effort.reshape(-1, 1).astype(np.float32) - effort_hat.reshape(-1, 1).astype(np.float32))/effort)
    # print('MRE:', mre_value)
    return mre_value


def mre_gpu(effort, effort_hat):
    effort = Variable(effort)
    effort_hat = Variable(effort_hat)
    mre_value = F.absolute(effort - effort_hat) / effort
    return mre_value


def mmre(effort, effort_hat):
    mre_value = mre(effort, effort_hat)
    mmre_value = 100.0 * sum(mre_value) / len(mre_value)
    print('MMRE(mean Error):', mmre_value.astype('int'))
    return mmre_value


def mdmre(effort, effort_hat):
    mre_value = mre(effort, effort_hat)
    mdmre_value = 100 * np.median(mre_value)
    print('MdMRE:', mdmre_value)
    return mdmre_value


def predl_gpu(effort, effort_hat, len, l=25):
    mre_value = mre_gpu(effort, effort_hat)
    data_len = len
    percent = l / 100.0
    # print(percent)
    count = 0
    for a_mre in mre_value.data:
        if a_mre <= percent:
            count += 1
    predl_value = 100.0 / data_len * count
    return predl_value


def rsd(effort, effort_hat, afp):
    cha = effort - effort_hat
    cha = cha.astype('float32')
    divide = cha/afp
    square = np.square(divide)
    sum = np.sum(square)
    divideN = sum / (len(effort_hat)-1)
    sqrt = math.sqrt(divideN)
    rsd1 = sqrt
    # print(rsd1)
    rsd = math.sqrt(np.sum(np.square(np.divide(effort-effort_hat, afp.astype('float32'))))/(len(effort)-1))
    print('RSD:', rsd)
    return rsd


def lsd(effort, effort_hat):
    e = np.log(effort) - np.log(effort_hat)
    s2 = np.var(e)
    """
    cha = e - (-1/2 * s2)
    square = np.square(cha)
    sum = np.sum(square)
    divide = sum / (len(e)-1)
    sqrt = math.sqrt(divide)
    """
    lsd_value = math.sqrt(np.sum(np.square(e - (-1/2 * s2)))/(len(e)-1))
    print('LSD:', lsd_value)
    return lsd_value


def predl(effort, effort_hat, l=25):
    mre_value = mre(effort, effort_hat)
    data_len = len(effort)
    percent = l / 100.0
    count = 0
    for a_mre in mre_value:
        if a_mre <= percent:
            count += 1
    predl_value = 100.0/data_len*count
    return predl_value
	
def mae(effort, effort_hat):
    mae_value = np.mean(np.fabs(effort.reshape(-1, 1).astype(np.float32) - effort_hat.reshape(-1, 1).astype(np.float32)))
    return mae_value

def pred25(effort, effort_hat):
    pred25_value = predl(effort, effort_hat, l=25)
    # print('Pred25:', pred25_value)
    return pred25_value


def pred25_gpu(effort, effort_hat, len):
    pred25_value = predl_gpu(effort, effort_hat, len, l=25)
    # print('Pred25:', pred25_value)
    return pred25_value


if __name__ == "__main__":
    effort = np.asarray([23, 56, 32, 79])
    effort_hat = np.asarray([25, 48, 45, 90])
    afp = np.asarray([100, 100, 100, 100])
    lsd = lsd(effort, effort_hat)
    print('mre', mre(effort, effort_hat))
    print('mdmre', mdmre(effort, effort_hat))
    # print('pred25', pred25(effort, effort_hat))
    print('lsd', lsd)
    print('mae', mae(effort, effort_hat))
