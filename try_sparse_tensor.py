# -*- coding: utf8 -*-
import tensorflow as tf
import numpy as np
import sys
import time


def readR(filename):
    indices = []
    values = []
    f = open(filename)
    line = f.readline()
    while True:
        try:
            content = line.split('\t')
            u,i,r = int(content[0]),int(content[1]),int(content[2])
            indices.append([u,i])
            values.append(r)
            line = f.readline()
        except:
            break
    return indices,values


def print_size(data, dataname = 'tensor'):
    siz = float(sys.getsizeof(data))
    units = ['', 'K', 'M', 'G', 'T']
    unit_idx = 0
    while siz > 1000:
        siz = siz/1000
        unit_idx += 1
    print("size of "+dataname+": %.2f " % siz + units[unit_idx]+" bytes" )

denseR = np.load('./data/ml-100k/R.npy')
denseR_tf = tf.convert_to_tensor(denseR,dtype=np.float32)
indices, values = readR('./data/ml-100k/u.data')
indices2 = []
values2 = []
# for i in range(len(denseR2)):
#     for j in range(denseR2.shape[1]):
#         if denseR2[i][j]>0:
#             indices2.append([i,j])
#             values2.append(denseR2[i][j])
sparseR_tf = tf.SparseTensor(indices,values,dense_shape=[943,1682])
# sparseR_tf2 = tf.SparseTensor(indices2,values2,dense_shape=[943,1682])
# denseR_tf2 = tf.sparse_tensor_to_dense(sparseR_tf)
k = 50
# denseU = np.concatenate(tuple([denseR]*100),axis=0)[:,:100]
denseU = np.ones([80000, k],dtype=np.float32)
denseU_tf = tf.convert_to_tensor(denseU,dtype=tf.float32)
# denseI = np.concatenate(tuple([denseR]*10),axis=1)[:100]
denseI = np.ones([k,10000],dtype=np.float32)
denseI_tf = tf.convert_to_tensor(denseI,tf.float32)
tf3 = tf.matmul(denseU_tf,denseI_tf)
tf4 = tf3 + 1.2

sess = tf.Session()
with sess.as_default():
    print_size(denseR_tf.eval(),dataname='dense tensor R')
    print_size(sparseR_tf.eval(), dataname='sparse tensor R')
    print tf3.shape
    t1 = time.time()
    a,b = sess.run([tf3,tf4])
    print_size(a, dataname='mat tensor')
    t2 = time.time()
    print ('matmul ops time', t2-t1)
    print_size(b, dataname='mat tensor')
