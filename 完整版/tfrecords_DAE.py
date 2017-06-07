# -*- coding: utf8 -*-
import tensorflow as tf
from utils import *
import time
import numpy as np

class DAE(object):
    """
    Denoising AutoDecoder 降噪自编码器层
    本网络通过使用有限神经元重建数据（无监督学习）实现对输入数据的特征提取。
    参数：
    :param sess:            进程，tf.session类
    :param input_size:      输入数据维度（列宽）, int,
    :param data_shape:      输入训练数据的形状，tuple or list, （记录数 x 特征维度）
    :param noise:           噪声水平，float, 0-1, default:0
    :param units:           隐层节点数，int, default:20
    :param layer:           层数序号， int, default:20
    :param learning_rate:   初始学习率，float, default:0.01
    :param n_epoch:         训练迭代次数，int,default:100
    :param is_training:     是否在训练的标志位， bool, default True
    :param batch_size:      每批训练记录数， int, default: 20
    :param decay:           学习率衰减程度， float, default: 0.95
    :param rho:             稀疏化水平，每次输出将有（1-rho)的部分接近0， float, default 0.05
    :param reg_lambda:      正则化系数，float, default 0.0
    :param sparse_lambda:   稀疏化系数，float, default 1.0
    :param summary_handle:  summary_handler, SummaryHandle类， default:None
    
    其他properties:
    stddev：                 参数初始化的方差，default: 0.2
    dropout_p：              dropout层保持概率， default: 0.5
    change_lr_epoch：        开始降低学习率的epoch数, default: int(n_epoch*0.3)
    
    callable functions：
    hidden:     一个全连接enc-dec层
    compute：   调用hidden构建网络, 计算loss
    train:      训练
    """
    def __init__(self, sess, input_size, data_shape, noise=0, units=20, layer=0, learning_rate=0.01,
                 n_epoch=100, is_training=True, batch_size=20, decay=0.95,
                 reg_lambda=0.0, rho=0.05, sparse_lambda=1.0, summary_handle=None):

        self.sess = sess
        self.is_training = is_training
        self.units = units                      # 隐层节点数
        self.layer = layer                      # 是第几层
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.input_size = input_size            # 输入特征数
        self.data_shape = data_shape

        self.lr_init = learning_rate
        self.stddev = 0.2                       # 初始化参数用的
        self.noise = noise                      # dropout水平，是数\
        self.reg_lambda = reg_lambda            # 正则化系数,float
        self.dropout_p = 0.5                    # dropout层保持概率
        self.rho = rho                          # 稀疏性系数
        self.sparse_lambda = sparse_lambda
        self.lr_decay = decay
        self.change_lr_epoch = int(n_epoch*0.3) # 开始改变lr的epoch数

        self.summ_handle = summary_handle

    # ------------------------- 隐层 -------------------------------------
    def hidden(self, inputdata, input_size, units, noise, name="default"):
        """
        一个全连接enc-dec层，使用有限个神经元重建数据。
        先用 y = act1(wx + b1)将加过噪声的输入数据x通过有限个神经元（uints）编码为中间特征y，
        再用 x_rec = act2(w'y + b2)将中间特征y解码为重建数据x_rec.
        :param inputdata:       输入数据，2D tensor,(batch_size x input_size)
        :param input_size:      数据数据的特征数（维度）,int
        :param units:           神经元数, int
        :param noise:           噪声水平。float。使用mask噪声，有1-noise个输入数据被污染。
        :param name:            本层名称。string。
        :return:                list of tensors [提取到的特征(batch_size x units)
                                                ， 重建的数据(与inputdata形状相同)]，
        """
        with tf.variable_scope(name):
            # ------------------- 输入数据加噪声 -----------------------------------------
            # mask噪声
            corrupt = tf.layers.dropout(inputdata, rate=noise, training=self.is_training)
            # 加性高斯噪声
            # corrupt = tf.add(input,noise * tf.random_uniform(input.shape))
            # -------------------- encoder ----------------------------------------------
            # 新建可学习参数，可复用
            try:
                ew = tf.get_variable('enc_weights',shape=[input_size, units],
                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=self.stddev),
                                     regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda))  # L2正则化
                eb = tf.get_variable('enc_biases',shape=[1,units],
                                    initializer=tf.constant_initializer(0.0), dtype=tf.float32,
                                    regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda))  # L2正则化
            except ValueError:
                tf.get_variable_scope().reuse_variables()
                ew = tf.get_variable('enc_weights')
                eb = tf.get_variable('enc_biases')
            sew = tf.summary.histogram(name + '/enc_weights', ew)
            seb = tf.summary.histogram(name + '/enc_biases', eb)
            fc1 = tf.add(tf.matmul(corrupt, ew), eb)
            act1 = tf.nn.sigmoid(tf.layers.batch_normalization(fc1))
            character = act1
            self.ew = ew
            self.eb = eb

            # --------------------------- decoder --------------------------------------
            try:
                dw = tf.transpose(ew)
                db = tf.get_variable('dec_biases', shape=[1,input_size],
                                    initializer=tf.constant_initializer(0.0), dtype=tf.float32,
                                    regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda))  # L2正则化
            except ValueError:
                tf.get_variable_scope().reuse_variables()
                db = tf.get_variable('dec_biases')
            sdb = tf.summary.histogram(name + '/dec_biases', db)
            self.summ_handle.add_summ(sew, seb, sdb)
            fc = tf.add(tf.matmul(act1, dw), db)
            out = tf.sigmoid(fc)
        return character, out

    def compute(self, x):
        """
        调用self.hidden()构建网络, 根据hidden()返回的重建数据计算与输入数据的误差，计算loss
        loss = mse + sparse_lambda * sparse_loss + reg_lambda * reg_loss
        :param x:   输入数据，tensor of shape(batch_size x input_size)
        :return:    None
        """
        self.lr = tf.placeholder(tf.float32, name='learning_rate')      # 学习率可调
        self.x = x

        # build a hidden layer
        layer_name = "hidden_layer" + str(self.layer)
        self.character, self.out = self.hidden(self.x, self.input_size, self.units, noise=self.noise,
                                                name=layer_name)

        # 提取需要正则化的参数,获得正则化reg_loss
        reg_losses = tf.losses.get_regularization_losses(layer_name)
        for loss in reg_losses:
            tf.add_to_collection('losses' + layer_name, loss)
        self.reg_losses = tf.get_collection('losses'+layer_name)

        # 计算MSEloss
        mse_loss = mse(self.out, self.x)
        tf.add_to_collection('losses'+layer_name, mse_loss)

        # 计算稀疏项loss
        self.sparse_loss = self.sparse_lambda * sparse_loss(self.rho,self.character)
        tf.add_to_collection('losses' + layer_name, self.sparse_loss)
        self.sparse_coef = tf.reduce_mean(tf.reduce_mean(self.character))

        # 总loss = mse + sparse_lambda * sparse_loss + reg_lambda * reg_loss
        self.loss = tf.add_n(tf.get_collection('losses'+layer_name))

    def train(self, data_generator_train, data_generator_val, coord,summ_writer, summ_handle):
        """
        训练函数，使用data_generator_train分批读取数据进行训练，提取数据的该层特征，同时打印并记录训练进程中的loss和参数分布变化。
        使用data_generator_val产生验证数据，进行验证。
        :param data_generator_train:    训练数据生成器，generator类，内部会调用tf.train.batch不断生成训练数据
        :param data_generator_val:      验证数据生成器，generator类
        :param coord:                   使用的线程
        :param summ_writer:             summary writer
        :param summ_handle:             summary handler
        :return:                        产生训练数据和验证数据所需的队列线程，[threads, threads2],需要外部注销
        """

        # batch_x = data_generator_train.next()   # 产生分批训练数据
        batch_x = next(data_generator_train)    # Python3 产生分批训练数据
        self.compute(batch_x)                   # 构建网络图
        self.train_vals = [var for var in tf.trainable_variables()
                                if str(self.layer) in var.name.split("/")[0]]   # 提取本层被训练变量
        summ_handle.summ_loss.append(tf.summary.scalar('loss'+str(self.layer), self.loss))
        self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9).minimize(self.loss, var_list=self.train_vals)
        tf.initialize_variables(tf.all_variables()).run()                       # 初始化变量

        n_batch = self.data_shape[0] // self.batch_size
        print("num_batch", n_batch)
        counter = 0                             # 写summary用的，记录一共训练了多少个batch
        current_lr = self.lr_init
        begin_time = time.time()                # 记录开始训练的时间
        # 启动tf.train.batch的队列，很重要，没有这一句程序不报错，但是跑着就好像卡住了一样
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        # ------------------------- 训练 ----------------------------------
        for epoch in range(self.n_epoch-1):
            if epoch > self.change_lr_epoch:
                current_lr = current_lr * self.lr_decay             # 调整学习率
            for batch in range(n_batch):
                counter += 1
                _, loss, reg_loss, re_sparse_loss, sparse_coef, out,  character, summ_loss =\
                    self.sess.run([self.optimizer, self.loss,self.reg_losses, self.sparse_loss, self.sparse_coef,
                                   self.out, self.character, summ_handle.summ_loss[0]], feed_dict={self.lr: current_lr})
                summ_writer.add_summary(summ_loss, epoch * n_batch + batch)
                if counter % 50 == 0:
                    # 记录w,b
                    summ_ew, summ_eb, summ_db = self.sess.run([summ_handle.summ_enc_w[0], summ_handle.summ_enc_b[0],
                                                                        summ_handle.summ_dec_b[0]],
                                                                        feed_dict={self.lr:current_lr})
                    summ_writer.add_summary(summ_ew, epoch * n_batch + batch)
                    summ_writer.add_summary(summ_eb, epoch * n_batch + batch)
                    summ_writer.add_summary(summ_db, epoch * n_batch + batch)
            print("epoch ",epoch," train loss: ", loss," reg loss: ",reg_loss, " sparse loss: ",re_sparse_loss,
                      " sparse coef: ", sparse_coef, " time:",str(time.time()-begin_time))

        #----------- 最后一个epoch, 除了训练，还要获取下一层训练的特征图，记录wb的分布 ------
        epoch = self.n_epoch - 1
        characters = []
        outs = []
        for batch in range(n_batch):
            _, loss, out, character, self.ewarray, self.ebarray, summ_loss, summ_ew,summ_eb,summ_db\
                = self.sess.run([self.optimizer, self.loss, self.out,self.character,self.ew,self.eb,
                                summ_handle.summ_loss[0],summ_handle.summ_enc_w[0],summ_handle.summ_enc_b[0],
                                summ_handle.summ_dec_b[0]],
                                feed_dict={self.lr:current_lr})
            summ_writer.add_summary(summ_loss, epoch * n_batch + batch)
            summ_writer.add_summary(summ_ew, epoch * n_batch + batch)
            summ_writer.add_summary(summ_eb, epoch * n_batch + batch)
            summ_writer.add_summary(summ_db, epoch * n_batch + batch)
            characters.append(character)
            outs.append(out)
        print("epoch ", epoch, " train loss: ", loss, " time:", str(time.time() - begin_time))

        # ------------------- validation ------------------------------------------
        val_characters =[]
        # batch_x = data_generator_val.next()
        batch_x = next(data_generator_val)
        self.compute(batch_x)
        threads2 = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        loss, val_character = self.sess.run([self.loss, self.character], feed_dict={self.lr:current_lr})
        val_characters.append(val_character)
        print("val loss: ", loss, " time:", str(time.time() - begin_time))

        # ------------------- 整理本层训练得到的特征和重建数据 -----------------------
        self.next_x = np.concatenate(tuple(characters))
        self.next_x_val = np.concatenate(tuple(val_characters))
        self.rec = np.concatenate(tuple(outs))

        return threads, threads2