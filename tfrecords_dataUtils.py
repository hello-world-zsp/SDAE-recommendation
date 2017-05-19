# -*- coding: utf8 -*-
import tensorflow as tf
import numpy as np

# write tfrecords
def write_tfrecords(path,name):
    filename = path+name+'.npy'
    tfrecords_filename = path+name+'.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    data = np.load(filename)
    data = data.astype(np.uint8)
    data_raw = data.tostring()                  # 要改成string，作为一个object而不是很多行，才能存
    example = tf.train.Example(features=tf.train.Features(feature={     # 数据填入到Example协议内存块(protocol buffer)，
        'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw]))}))
    writer.write(example.SerializeToString())                           # 将协议内存块序列化为一个字符串,写入到TFRecords文件
    writer.close()
    print ('tfrecords saved')

# for i in range(1,6):
#     write_tfrecords(path='./data/ml-100k/',name='R'+str(i)+'_train')
#     write_tfrecords(path='./data/ml-100k/', name='R' + str(i) + '_val')
# write_tfrecords(path='./data/ml-100k/', name='User')
# write_tfrecords(path='./data/ml-100k/', name='Item')
def read_and_decode_tfrecords(tfrecords_filename, nu=None,ni=None):
    filename_queue = tf.train.string_input_producer([tfrecords_filename])  # 根据文件名生成一个队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(                                 # 将Example协议内存块解析为张量
        serialized_example,
        features={
            'data_raw': tf.FixedLenFeature([], tf.string),
        })
    data_rec2_tf = tf.decode_raw(features['data_raw'], tf.uint8)
    if nu is not None:
        data_rec2_tf = tf.reshape(data_rec2_tf,[nu,ni])
    return data_rec2_tf

def data_generator_tfrecords(Rfilename,path,nu,nb,nb_batch,batch_size=None,shuffle = True):
    U = read_and_decode_tfrecords("./" + path + 'User.tfrecords',nu,1)
    I = read_and_decode_tfrecords("./" + path + 'item.tfrecords',nb,1)
    R = read_and_decode_tfrecords(Rfilename,nu,nb)
    # --------------------- 改到这里了！！！！---------------------------------------
    if shuffle:
        ru = np.random.permutation(nu).astype(np.int32)     # 只在第一次读的时候做shuffle
        U = U[ru,:]
        ri = np.random.permutation(nb)
        I = I[ri,:]
    else:
        ru = range(U.shape[0])
        ri = range(I.shape[0])

    if batch_size is None:
        R = R[ru,:]
        R = R[:,ri]
        batch_U = np.concatenate((R, U), axis=1)
        batch_I = np.concatenate((R.T, I), axis=1)                                  # 转置
        yield batch_U, batch_I, R
    else:
        batch = 0
        while batch <= nb_batch:
            batch_U = U[batch*batch_size:(batch+1)*batch_size]
            batch_I = I[batch*batch_size:(batch+1)*batch_size]
            batch_R_u = R[ru,:][batch*batch_size:(batch+1)*batch_size]            # 所选用户对应的评分项
            batch_R_i = R[:,ri][:,batch*batch_size:(batch+1)*batch_size]
            batch_R = batch_R_u[:,ri][:,batch*batch_size:(batch+1)*batch_size]
            batch_U = np.concatenate((batch_R_u,batch_U),axis=1)
            batch_I = np.concatenate((batch_R_i.T,batch_I),axis=1)                  # 转置
            batch += 1
            yield batch_U,batch_I,batch_R