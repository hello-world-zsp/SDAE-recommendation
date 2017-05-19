# -*- coding: utf8 -*-
import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# write tfrecords
def write_tfrecords():
    filename = './data/ml-100k/R.npy'
    tfrecords_filename = 'test.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    data = np.load(filename)
    data = data.astype(np.uint8)
    data_raw = data.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={     # 数据填入到Example协议内存块(protocol buffer)，
        'height': _int64_feature(data.shape[0]),
        'width': _int64_feature(data.shape[1]),
        'data_raw': _bytes_feature(data_raw)}))
    writer.write(example.SerializeToString())                           # 将协议内存块序列化为一个字符串,写入到TFRecords文件
    writer.close()
    print ('tfrecords saved')

# read tfrecords
def read_tfrecord_1(tfrecords_filename):
    example = tf.train.Example()
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for string_record in record_iterator:
        example.ParseFromString(string_record)
        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        data_str = (example.features.feature['data_raw'].bytes_list.value[0])
        data_rec = np.fromstring(data_str,dtype=np.uint8).reshape([height,width])
        # print np.sum(data-data_rec)
    return data_rec

# read tfrecords method 2
def read_and_decode_tfrecords(tfrecords_filename):
    filename_queue = tf.train.string_input_producer([tfrecords_filename])  # 根据文件名生成一个队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(                                 # 将Example协议内存块解析为张量
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'data_raw': tf.FixedLenFeature([], tf.string),
        })
    data_rec2_tf = tf.decode_raw(features['data_raw'], tf.uint8)
    # height = tf.cast(features['height'], tf.int32).eval()
    # width = tf.cast(features['width'], tf.int32).eval()
    data_rec2_tf = tf.reshape(data_rec2_tf,[943,1682])
    return data_rec2_tf

with tf.Session() as sess:
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    data = read_and_decode_tfrecords('test.tfrecords')
    data_batch = tf.train.shuffle_batch([data[:]], batch_size=100, capacity=10000, min_after_dequeue=1000)
    sess.run(tf.initialize_all_variables())
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(data.shape[0]//100):
        data_rec2 = sess.run([data_batch])
        print i
        print data_rec2[0].shape

    # coord.request_stop()
    # coord.join(threads)

print ('done')