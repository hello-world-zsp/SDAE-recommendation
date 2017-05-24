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

def data_generator_tfrecords(Rfilename,path,Rshape,Usize,Isize,nb_batch,batch_size=None,shuffle = True):
    U = read_and_decode_tfrecords("./" + path + 'User.tfrecords',Rshape[0],Usize)
    I = read_and_decode_tfrecords("./" + path + 'Item.tfrecords',Rshape[1],Isize)
    R = read_and_decode_tfrecords(Rfilename,Rshape[0],Rshape[1])

    if shuffle:
        ru = np.random.permutation(Rshape[0]).astype(np.int32)
        ri = np.random.permutation(Rshape[1]).astype(np.int32)
        U = tf.gather(U,ru)
        I = tf.gather(I,ri)
    else:
        ru = range(Rshape[0])
        ri = range(Rshape[1])

    Ru = tf.gather(R, ru)
    Ri = tf.gather(tf.transpose(R), ri)
    batch = 0
    while batch <= nb_batch:
        batch_U = tf.train.batch([U[:]], batch_size=batch_size, enqueue_many=True)
        batch_I = tf.train.batch([I[:]], batch_size=batch_size, enqueue_many=True)
        batch_R_u = tf.train.batch([Ru[:]], batch_size=batch_size, enqueue_many=True)  # 所选用户对应的评分项
        batch_R_i = tf.train.batch([Ri[:]], batch_size=batch_size, enqueue_many=True)
        batch_R = tf.gather(tf.transpose(batch_R_u),ri)
        batch_R = tf.train.batch([batch_R[:]], batch_size=batch_size, enqueue_many=True)
        batch_R = tf.cast(tf.transpose(batch_R),dtype=tf.float32)
        batch_U = tf.cast(tf.concat((batch_R_u, batch_U), axis=1),dtype=tf.float32)
        batch_I = tf.cast(tf.concat((batch_R_i, batch_I), axis=1),dtype=tf.float32)  # 转置
        batch += 1
        yield batch_U, batch_I, batch_R
