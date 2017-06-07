# -*- coding: utf8 -*-

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import csv
from sklearn.preprocessing import normalize


def load_data(path,shuffle=False):
    mnist = input_data.read_data_sets(path, one_hot=True)
    trainX = mnist.train.images     # ndarray
    trainY = mnist.train.labels
    trainY = trainY.astype('float32')
    valX = mnist.validation.images
    valY = mnist.validation.labels
    valY = valY.astype('float32')
    testX = mnist.test.images
    testY = mnist.test.labels
    testY = testY.astype('float32')
    if shuffle:
        r = np.random.permutation(len(trainY))
        trainX = trainX[r,:]
        trainY = trainY[r,:]

        r = np.random.permutation(len(valY))
        valX = valX[r,:]
        valY = valY[r,:]

        r = np.random.permutation(len(testY))
        testX = testX[r,:]
        testY = testY[r,:]

    return trainX, trainY, valX, valY, testX, testY


def load_data_batch(path,start=0,batch_size=None,shuffle=False):
    if batch_size == None:
        X = np.load("./" + path).astype(np.float32)[:, :, :, None]
    else:
        X = np.load("./" + path).astype(np.float32)[start:start+batch_size, :]
    X = X/255
    if len(X.shape)>2:
        X = np.reshape(X,[X.shape[0],X.shape[1]*X.shape[2]])
    if shuffle:
        r = np.random.permutation(X.shape[0])
        X = X[r,:]
    trainX = X

    return trainX


def load_goods_data(train_ratio=0.9, shuffle=True, use_cat=True):
    """
    读取商品数据， npy或csv格式。舍弃前3列id信息。划分为train和val数据集。
    :param train_ratio: 根据train_ratio划分为train和val数据集,float, default 0.9
    :param shuffle:     是否重排,bool, default true
    :param use_cat:     是否使用商品分类信息作为特征，bool，default TRUE，（有监督验证特征提取效果时，此项为False）
    :return: list of ndarrys [trainX,valX,trainidx,validx,trainY,valY]
    """
    # X=np.loadtxt(open("./data/goods_vectors_new.csv","rb"),delimiter=",",skiprows=0)
    X = np.load('./data/goods_vectors_20170605.npy').astype(np.float32)
    if use_cat:
        X = X[:,3:]     # 前3列是id，后15列是one-hot的分类信息
        Y = X[:, -15:]
    else:
        Y = X[:, -15:]
        X = X[:, 3:-15]
    nan_loc = np.argwhere(np.isnan(X))
    if len(nan_loc)>0:
        X = np.delete(X,nan_loc[0],axis=0)

    X = normalize(X,axis=1)
    if shuffle:
        idx = np.random.permutation(X.shape[0])
        X = X[idx,:]
        Y = Y[idx,:]
    else :
        idx = range(len(X))
    trainX = X[:np.round(train_ratio * X.shape[0])]
    trainidx = idx[:np.round(train_ratio * X.shape[0])]
    trainY = Y[:np.round(train_ratio * X.shape[0])]
    valX = X[np.round(train_ratio * X.shape[0]):]
    validx = idx[np.round(train_ratio * X.shape[0]):]
    valY = Y[np.round(train_ratio * X.shape[0]):]
    return trainX,valX,trainidx,validx,trainY,valY


def load_goods_data_libsvm():
    """
    libsvm转npy
    :return: 
    """
    X = np.zeros((5449, 602), dtype=np.float32)
    i = 0
    maxid = 0
    for line in open("./data/goods_vectors_20170605.libsvm","r"):
        data = line.split(" ")
        for fea in data[1:]:
            id, value = fea.split(":")
            id = int(id)
            X[i][id] = float(value)
        if id > maxid: maxid = id
        i += 1
    np.save('./data/goods_vectors_20170605.npy',X)



def load_users_data(train_ratio = 0.9):
    """
    读取用户数据， npy格式。划分为train和val数据集。
    :param train_ratio: 根据train_ratio划分为train和val数据集,float, default 0.9
    :return: list of ndarrys [trainX,valX]
    """
    data = np.load('./data/users_data.npy')
    sparse = np.sum(np.sign(data))/(data.shape[0]*data.shape[1])
    print (sparse)
    trainX = data[:np.round(train_ratio * data.shape[0])]
    valX = data[np.round(train_ratio * data.shape[0]):]
    return trainX, valX

#load_goods_data_libsvm()
