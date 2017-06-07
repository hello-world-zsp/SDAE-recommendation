# -*- coding: utf8 -*-
import os
import tensorflow as tf
from tfrecords_dataUtils import *
from tfrecords_DAE import *

"""
一个summary handler,用于记录训练中的loss、metric、权重分布变化等。
"""
class SummaryHandle():
    def __init__(self):
        self.summ_enc_w = []
        self.summ_enc_b = []
        self.summ_dec_b = []
        self.summ_loss = []

    def add_summ(self, e_w, e_b, d_b):
        self.summ_enc_w.append(e_w)
        self.summ_enc_b.append(e_b)
        self.summ_dec_b.append(d_b)


class SDAE(object):
    """
    Stacked Denoising AutoDecoder 堆栈降噪自编码器网络
    本网络通过使用多层有限神经元重建数据（无监督学习）实现对输入数据的特征提取，是DAE层的堆叠。
    Arguments：
    :param sess:            进程，tf.session类
    :param data_shape:      输入训练数据的形状，tuple or list, （记录数 x 特征维度）
    :param val_data_shape:  输入验证数据的形状，tuple or list, （记录数 x 特征维度）
    :param noise:           噪声水平，float, 0-1, default:0
    :param n_nodes:         各层节点数，tuple of ints, default:(256, 128, 64)
    :param learning_rate:   各层初始学习率，tuple of floats, default:(0.1,0.1,0.1)
    :param n_epochs:        各层训练迭代次数，tuple of ints,default:(100,100,100)
    :param is_training:     是否在训练的标志位， bool, default True
    :param data_dir:        数据存储和读取的路径， string, default None
    :param batch_size:      每批训练记录数， int, default: 20
    :param rho:             各层稀疏化水平，每次输出将有（1-rho)的部分接近0， tuple of floats, default(0.05,0.05,0.05)
    :param reg_lambda:      正则化系数，float, default 0.0
    :param sparse_lambda:   稀疏化系数，float, default 1.0
    :param name:            提取特征项的名称，String, default 'default'
    
    Callable functions:
    train：                  逐层训练DAE网络
    """
    def __init__(self, sess, data_shape, val_data_shape, noise=0, n_nodes=(256, 128, 64),
                 learning_rate=(.1, .1, .1), n_epochs=(100, 100, 100), is_training=True,
                 data_dir=None, batch_size=20, rho=(0.05, 0.05, 0.05),
                 reg_lambda=0.0, sparse_lambda=1.0, name='default'):

        self.sess = sess
        self.is_training = is_training
        self.n_nodes = n_nodes              # 各层节点数
        self.n_layers = len(self.n_nodes)   # 层数
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.input_size = data_shape[1]        # 输入特征数
        self.data_shape = data_shape
        self.val_data_shape = val_data_shape

        self.lr = learning_rate
        self.stddev = 0.02                  # 初始化参数用的
        self.noise = noise                  # dropout水平，是tuple
        self.rho = rho                      # 各层稀疏性系数
        self.sparse_lambda = sparse_lambda  # 稀疏loss权重
        self.reg_lambda = reg_lambda        # 正则项权重

        self.checkpoint_dir = 'checkpoint'
        self.result_dir = 'results'
        self.log_dir = 'logs'
        self.data_dir = data_dir
        self.name = name

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

    def train(self):
        """
        由于使用tfrecords数据，不需要预先build，因此build和train合为一个train函数
        
        :return: [用SDAE提取到的训练集特征，用SDAE提取到的验证集特征]，返回特征是后几层特征的拼接。各自shape为（记录数 x 除第一层外各层nodes数目之和）
        """
        # 初始化
        tf.global_variables_initializer().run()
        self.writer = tf.summary.FileWriter('./'+self.log_dir, self.sess.graph)
        # 初始化线程
        coord = tf.train.Coordinator()
        self.hidden_layers =[]
        self.summary_handles = []
        features = []
        val_features = []
        threads = []
        n_input = [self.input_size]+list(self.n_nodes[:-1])
        n_batch = self.data_shape[0] // self.batch_size
        # 初始化数据生成器
        if self.name == 'goods':
            data_generator_train = data_generator_tfrecords(self.data_dir + 'goods_vectors_train.tfrecords',
                                                            self.data_shape, n_batch, batch_size=self.batch_size)
            data_generator_val = data_generator_tfrecords(self.data_dir + 'goods_vectors_val.tfrecords',
                                                          self.val_data_shape, n_batch)
        else:
            data_generator_train = data_generator_tfrecords(self.data_dir + 'user_data_train.tfrecords',
                                                            self.data_shape, n_batch, batch_size=self.batch_size)
            data_generator_val = data_generator_tfrecords(self.data_dir + 'user_data_val.tfrecords',
                                                          self.val_data_shape, n_batch)

        # 逐层训练
        for i in range(self.n_layers):
            # ------------------------ build -----------------------------------
            summary_handle = SummaryHandle()
            layer = DAE(self.sess, n_input[i], self.data_shape, noise=self.noise[i], units=self.n_nodes[i],
                        layer=i, n_epoch=self.n_epochs[i], is_training=self.is_training,
                        batch_size=self.batch_size, learning_rate=self.lr[i],
                        rho=self.rho[i], reg_lambda=self.reg_lambda, sparse_lambda=self.sparse_lambda,
                        summary_handle=summary_handle)
            self.hidden_layers.append(layer)
            self.summary_handles.append(summary_handle)

            # ------------------------ train -----------------------------------
            print("training layer: ",i)
            threads1, threads2 = layer.train(data_generator_train, data_generator_val, coord,
                        summ_writer=self.writer, summ_handle=self.summary_handles[i])
            threads.extend([threads1,threads2])

            # -------------- 保存本层提取到的特征，写入tfrecords文件 --------------
            write_tfrecords(self.data_dir, self.name+'_train_feature_'+str(i), layer.next_x)
            write_tfrecords(self.data_dir, self.name+'_val_feature_' + str(i), layer.next_x_val)
            # 更新数据生成器
            data_generator_train = data_generator_tfrecords(self.data_dir + self.name+'_train_feature_'+str(i)+'.tfrecords',
                                                            layer.next_x.shape, n_batch, batch_size=self.batch_size)
            data_generator_val = data_generator_tfrecords(self.data_dir + self.name+'_val_feature_'+str(i)+'.tfrecords',
                                                          layer.next_x_val.shape, n_batch)
            features.append(layer.next_x)
            val_features.append(layer.next_x_val)

        features = np.concatenate(tuple(features[1:]),axis=1)               # 只保留高层特征，第一层特征丢弃
        val_features = np.concatenate(tuple(val_features[1:]), axis=1)

        # 关闭线程
        coord.request_stop()
        for thread in threads:
            coord.join(thread)
        # 返回训练集和val集每层提取到的特征
        return features, val_features


