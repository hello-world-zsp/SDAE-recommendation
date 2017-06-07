# -*- coding: utf8 -*-

from utils import *
from tfrecords_dataUtils import read_and_decode_tfrecords
import time
import numpy as np
import os
import tensorflow as tf

"""
协同过滤MLP网络，根据输入的用户特征U、商品特征I、用户-商品评分信息R，预测未评分项的评分。
所用算法来自参考文献 [AAAI 2017]A Hybrid Collaborative Filtering Model with Deep Structure for Recommender Systems
网络先训练用户side information的提取(U网络)，再训练商品side information的提取（I网络），得到U和V；
然后根据 R-UV 的误差fine-tune U和I网络。

"""
class MLPrec(object):
    def __init__(self, sess, Rshape, Ushape, Ishape, n_nodes=(20, 20, 20), learning_rate=0.01,
                 n_epoch=100, n_epoch_u=100, n_epoch_i=100, is_training=True, batch_size=20, decay=0.95,
                 reg_lambda=0.0, rho=0.05, sparse_lambda=0.0, alpha=1, beta=1, delta=1, noise=0,
                 data_dir=None):
        """
        
        :param sess:                进程,tf.session类
        :param Rshape:              评分矩阵R的形状，tuple or list, （用户数 x 商品数）
        :param Ushape:              用户数据矩阵U形状，tuple or list， （用户数 x 用户特征维度）
        :param Ishape:              商品数据矩阵I形状，tuple or list， （商品数 x 商品特征维度）
        :param n_nodes:             网络节点数，tuple or list, default (20, 20, 20)
        :param learning_rate:       初始学习率，float, default:0.01
        :param n_epoch:             fine-tune训练迭代次数，int,default:100
        :param n_epoch_u:           U网络训练迭代次数，int,default:100
        :param n_epoch_i:           I网络训练迭代次数，int,default:100
        :param is_training:         是否在训练的标志位， bool, default True
        :param batch_size:          每批训练记录数， int, default: 20
        :param decay:               学习率衰减程度， float, default: 0.95
        :param reg_lambda:          正则化系数，float, default 0.0
        :param rho:                 稀疏化水平，每次输出将有（1-rho)的部分接近0， float, default 0.05
        :param sparse_lambda:       稀疏化系数，float, default 1.0
        :param alpha:               U和I重建误差中，权衡S与X的系数, float, default 1
        :param beta:                总loss中，U网络的重建误差权重, float, default 1
        :param delta:               总loss中，I网络的重建误差权重, float, default 1
        :param noise:               噪声水平，float, 0-1, default:0
        :param data_dir:            数据存储和读取路径，string, default None
        
        其他properties:
        n_layers:                    网络层数，length of n_nodes
        stddev：                     参数初始化的方差，default: 0.2
        dropout_p：                  dropout层保持概率， default: 0.5
        change_lr_epoch：            开始降低学习率的epoch数, default: int(n_epoch*0.3)
        regularizer:                 正则项, L1或L2, default L2
        
        callable functions：
        encoder:                     编码层，一次全连接
        decoder:                     译码层，一次全连接
        compute：                    调用hidden构建网络, 计算loss
        train:                       训练
        """
        self.sess = sess
        self.is_training = is_training
        self.units = n_nodes                    # 隐层节点数
        self.n_layers = len(n_nodes)
        self.n_epoch = n_epoch
        self.n_epoch_u = n_epoch_u
        self.n_epoch_i = n_epoch_i
        self.batch_size = batch_size
        self.Rshape = Rshape
        self.Usize = Ushape[1]
        self.Isize = Ishape[1]

        self.lr_init = learning_rate
        self.stddev = 0.02                      # 初始化参数用的
        self.noise = noise                      # dropout水平，是数
        self.dropout_p = 0.5                    # dropout层保持概率
        self.rho = rho                          # 稀疏性系数
        self.sparse_lambda = sparse_lambda
        self.lr_decay = decay
        self.change_lr_epoch = int(n_epoch*0.3)  # 开始改变lr的epoch数
        self.regularizer = tf.contrib.layers.l2_regularizer  #使用L1或L2正则化

        self.reg_lambda = reg_lambda            # U,V正则化系数,float
        self.lambda_u = self.reg_lambda         # U网络权重正则化系数
        self.lambda_i = self.reg_lambda         # I网络权重正则化系数
        self.alpha = alpha                      # U和I重建误差中，权衡S与X的系数
        self.beta = beta                        # 总loss中，U网络的重建误差权重
        self.delta = delta                      # 总loss重，I网络的重建误差权重

        self.summ_handle = SummaryHandle()
        self.data_dir = data_dir

        self.checkpoint_dir = 'checkpoint'
        self.result_dir = 'results'
        self.log_dir = 'logs'

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)


    # ------------------------- 编码层 -------------------------------------
    def encoder(self, input, units, noise, layerlambda, istraining=True, name="encoder"):
        """
        编码层，一次全连接，对输入的向量x进行加噪声(x_hat)、编码,得到y：
        y = act(w * x_hat + b)
        :param input:           输入数据，2D tensor,(batch_size x input_size)
        :param units:           神经元数, int
        :param noise:           噪声水平，float。使用mask噪声，有1-noise个输入数据被污染
        :param layerlambda:     本层正则化系数，float
        :param istraining:      是否在训练的标志位， bool, default True
        :param name:            本层名称， string， default "encoder"
        :return:                2D tensor, 提取到的特征 (batch_size x units)
                                                
        """
        input_size = int(input.shape[1])
        with tf.variable_scope(name):
            # mask噪声
            corrupt = tf.layers.dropout(input, rate=noise, training=istraining)
            # 新建可学习参数，可复用
            try:
                ew = tf.get_variable('enc_weights', shape=[input_size, units],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=self.stddev),
                                 regularizer=self.regularizer(layerlambda))
                sew = tf.summary.histogram(name + '/enc_weights', ew)
                eb = tf.get_variable('enc_biases',shape=[1,units],
                                initializer=tf.constant_initializer(0.0),dtype=tf.float32,
                                regularizer=self.regularizer(layerlambda))
                seb = tf.summary.histogram(name+'/enc_biases', eb)
                self.ew = ew
                self.eb = eb
                self.summ_handle.add_summ(e_w=sew, e_b=seb)
            except ValueError:
                tf.get_variable_scope().reuse_variables()
                ew = tf.get_variable('enc_weights')
                eb = tf.get_variable('enc_biases')
            fc1 = tf.add(tf.matmul(corrupt, ew), eb)
            act = tf.nn.sigmoid(tf.layers.batch_normalization(fc1))
        return act

    # ------------------------- 译码层 -------------------------------------
    def decoder(self, input, units, layerlambda, istraining=True,name="decoder"):
        """
        译码层，一次全连接，对输入的数据y进行dropout(y_hat)、译码，得到x
        x = act (w * y_hat + b)
        :param input:           输入数据，2D tensor,(batch_size x input_size)
        :param units:           神经元数, int
        :param layerlambda:     本层正则化系数，float
        :param istraining:      是否在训练的标志位， bool, default True
        :param name:            本层名称， string， default "decoder"
        :return:                list of tensors [激活层输出(batch_size x units)
                                                ， fc层输出(batch_size x units)]
        """
        input_size = input.shape[1]
        with tf.variable_scope(name):
            # 新建可学习参数，可复用
            try:
                input = tf.layers.dropout(input, rate=self.noise, training=istraining)
                dw = tf.get_variable('dec_weights', shape=[input_size, units],
                                     initializer=tf.random_normal_initializer(mean=0.0,stddev=self.stddev),
                                     regularizer=self.regularizer(layerlambda))
                sdw = tf.summary.histogram(name + '/dec_weights', dw)
                db = tf.get_variable('dec_biases', shape=[1, units],
                                     initializer=tf.constant_initializer(0.0), dtype=tf.float32,
                                     regularizer=self.regularizer(layerlambda))
                sdb = tf.summary.histogram(name + '/dec_biases', db)
                self.dw = dw
                self.db = db
                self.summ_handle.add_summ(d_w=sdw, d_b=sdb)
            except ValueError:
                tf.get_variable_scope().reuse_variables()
                dw = tf.get_variable('dec_weights')
                db = tf.get_variable('dec_biases')
            fc = tf.add(tf.matmul(input, dw), db)
            out = tf.sigmoid(tf.layers.batch_normalization(fc))
        return out, fc

    def calculate(self, R, u_x, i_x):
        """
        分别堆叠多层encoder和decoder，构建U网络和I网络，通过数据重建，学习高层特征。
        将最高层encoder的编码作为U和V，即R的矩阵分解因子。根据R，U，V和重建误差计算总loss.
        loss = mse(R-UV) + reg_lambda * reg_loss_u_and_i+ beta * u_loss + delta * i_loss
        :param R:           评分矩阵，tensor of shape (batch_size x batch_size)
        :param u_x:         用户矩阵，tensor of shape (batch_size x Usize)
        :param i_x:         商品矩阵，tensor of shape (batch_size x Isize)
        :return:            None
        """
        self.R = R
        self.u_x = u_x
        self.i_x = i_x

        # --------------- 用户网络 ---------------------------------------------------------
        input_size = self.Rshape[1] + self.Usize
        loss_name = 'loss U'
        # ----- encoder -----
        self.U_enc_layers = []
        input_data = self.u_x
        for i in range(self.n_layers):
            layer_name = "U_encoder_layer"+str(i)
            out = self.encoder(input_data, self.units[i], self.noise, self.lambda_u,
                               istraining=self.is_training, name=layer_name)
            self.U_enc_layers.append(out)
            reg_losses = tf.losses.get_regularization_losses(layer_name)
            input_data = tf.concat([out,self.u_x],axis=1)
        self.U = out
        for loss in reg_losses:
            tf.add_to_collection(loss_name, loss)
        # ----- decoder -----
        self.U_dec_layers = []
        input_data = self.U
        dec_nodes = list(self.units[:self.n_layers-1])        # 解码器各层节点数，与编码器对应
        dec_nodes.reverse()
        dec_nodes.append(input_size)
        for i in range(self.n_layers):
            layer_name = "U_decoder_layer" + str(i)
            out, fc = self.decoder(input_data, dec_nodes[i],self.lambda_u,
                                  istraining=self.is_training,name=layer_name)
            self.U_dec_layers.append(out)
            reg_losses = tf.losses.get_regularization_losses(layer_name)
            input_data = tf.concat([out,self.u_x],axis=1)
        self.U_rec = fc
        for loss in reg_losses:
            tf.add_to_collection(loss_name, loss)
        # ----- loss -----
        self.reg_losses_u = tf.get_collection(loss_name)
        self.rec_loss_u = mse_by_part(self.U_rec, self.u_x,self.Rshape[1],self.alpha)
        tf.add_to_collection(loss_name,self.rec_loss_u)
        self.u_loss = tf.add_n(tf.get_collection(loss_name))  # U网络重建误差
        self.reg_losses = list(self.reg_losses_u)
        # ---------------------------------------------------------------------------------

        # --------------- 商品网络 ---------------------------------------------------------
        input_size = self.Rshape[0] + self.Isize
        loss_name = 'loss I'

        # ----- encoder -----
        self.I_enc_layers = []
        input_data = self.i_x
        for i in range(self.n_layers):
            layer_name = "I_encoder_layer" + str(i)
            out = self.encoder(input_data, self.units[i], self.noise, self.lambda_i,
                               istraining=self.is_training, name=layer_name)
            self.I_enc_layers.append(out)
            reg_losses = tf.losses.get_regularization_losses(layer_name)
            input_data = tf.concat([out, self.i_x], axis=1)
        self.V = out
        for loss in reg_losses:
            tf.add_to_collection(loss_name, loss)
        # ----- decoder -----
        self.I_dec_layers = []
        input_data = self.V
        dec_nodes = list(self.units[:self.n_layers - 1])  # 解码器各层节点数，与编码器对应
        dec_nodes.reverse()
        dec_nodes.append(input_size)
        for i in range(self.n_layers):
            layer_name = "I_decoder_layer" + str(i)
            out,fc = self.decoder(input_data, dec_nodes[i], self.lambda_i,
                                  istraining=self.is_training, name=layer_name)
            self.I_dec_layers.append(out)
            reg_losses = tf.losses.get_regularization_losses(layer_name)
            input_data = tf.concat([out, self.i_x], axis=1)
        self.I_rec = fc
        for loss in reg_losses:
            tf.add_to_collection(loss_name, loss)
        # ----- loss -----
        self.reg_losses_i = tf.get_collection(loss_name)
        self.reg_losses.extend(self.reg_losses_i)
        self.rec_loss_i = mse_by_part(self.I_rec, self.i_x, self.Rshape[0], self.alpha)
        tf.add_to_collection(loss_name, self.rec_loss_i)
        self.i_loss = tf.add_n(tf.get_collection(loss_name))  # I网络重建误差
        # ---------------------------------------------------------------------------------

        # ------------------------- 总loss ------------------------------------------------
        self.sparse_loss = self.sparse_lambda * (sparse_loss(self.rho, self.V)+sparse_loss(self.rho, self.U))
        tf.add_to_collection(loss_name, self.sparse_loss)
        self.R_hat = tf.matmul(self.U, tf.transpose(self.V))
        self.rec_loss = mse_mask(self.R,self.R_hat)
        reg_loss_u_and_i = tf.reduce_mean(tf.norm(self.U,axis=1))+tf.reduce_mean(tf.norm(self.V,axis=1))
        self.reg_losses.append(reg_loss_u_and_i)
        self.loss = self.rec_loss + self.reg_lambda * reg_loss_u_and_i+\
                    self.beta * self.u_loss + self.delta * self.i_loss
        self.summ_handle.summ_loss = tf.summary.scalar('total loss',self.loss)
        # 输出预测准确度，和文献比一下
        self.rmse,self.count = rmse_mask(self.R,self.R_hat)
        self.summ_handle.summ_metric = tf.summary.scalar('rmse', self.rmse)
        # ----------------------------------------------------------------------------------
        self.train_vals = tf.trainable_variables()
        self.train_vals_U = ( [var for var in self.train_vals if 'U' in var.name.split("/")[0]])
        self.train_vals_I = ([var for var in self.train_vals if 'I' in var.name.split("/")[0]])

    def train(self, load_data_func):
        """
        训练函数，使用load_data_func分批读取数据进行训练，预测R的未评分项，同时打印并记录训练进程中的loss和参数分布变化。
        :param load_data_func:      训练数据生成器，generator类，内部会调用tf.train.batch不断生成训练数据
        :return:                    list, [train rmse, val rmse]
        """
        self.lr = self.lr_init
        n_batch = self.Rshape[0] // self.batch_size
        Rfilename = "./" + self.data_dir + 'R' + '1_train.tfrecords'
        data_generator = load_data_func(Rfilename, self.data_dir, self.Rshape, self.Usize, self.Isize,
                                        n_batch, batch_size=self.batch_size, shuffle=True)
        batch_u, batch_i, batch_R = next(data_generator)
        self.calculate(batch_R, batch_u, batch_i)

        self.writer = tf.summary.FileWriter('./'+self.log_dir, self.sess.graph)
        self.optimizer = tf.train.RMSPropOptimizer(self.lr,name='optimizer').minimize(self.loss)
        self.Uoptimizer = tf.train.RMSPropOptimizer(self.lr, name='optimizer').minimize(self.u_loss, var_list=self.train_vals_U)
        self.Ioptimizer = tf.train.RMSPropOptimizer(self.lr, name='optimizer').minimize(self.i_loss, var_list=self.train_vals_I)
        tf.global_variables_initializer().run()

        counter = 0
        begin_time = time.time()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
        # --------------------------------- 训练 --------------------------------------------------
        # ------------------------------ 先训练100epoch的U ----------------------------------------
        for epoch in range(self.n_epoch_u):
            if epoch > self.change_lr_epoch:
                self.lr = self.lr * self.lr_decay
            mean_rmse_batch = 0
            for batch in range(n_batch):
                counter += 1
                _, loss, rmse, reg_loss,loss_u,loss_i, summ_loss,summ_rmse =\
                    self.sess.run([self.Uoptimizer,self.loss,self.rmse, self.reg_losses,self.rec_loss_u,self.rec_loss_i,
                                self.summ_handle.summ_loss,self.summ_handle.summ_metric],)
                if counter%50==0:
                # 记录w,b
                    summ_ew, summ_dw, summ_eb, summ_db = self.sess.run([self.summ_handle.summ_enc_w, self.summ_handle.summ_enc_b,
                                                               self.summ_handle.summ_dec_w, self.summ_handle.summ_dec_b],
                                                             )
                    for i in range(self.n_layers):
                        self.writer.add_summary(summ_ew[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_dw[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_eb[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_db[i], epoch * n_batch + batch)
                if epoch == self.n_epoch-1:
                    mean_rmse_batch += rmse
            train_mean_rmse = mean_rmse_batch/n_batch
            if epoch == self.n_epoch-1:
                rmse = train_mean_rmse
            print("epoch ",epoch," train loss: ", loss,"rmse: ", rmse," reg loss: ",reg_loss[-1],
                  " loss u: ",loss_u," loss i: ",loss_i,
                 " time:",str(time.time()-begin_time))
        # ---------------------------------- 再训练100epoch的I --------------------------------------
        for epoch in range(self.n_epoch_i):
            if epoch > self.change_lr_epoch:
                self.lr = self.lr * self.lr_decay
            mean_rmse_batch = 0
            for batch in range(n_batch):
                counter += 1
                _, loss, rmse, reg_loss,loss_u,loss_i, summ_loss,summ_rmse =\
                    self.sess.run([self.Ioptimizer,self.loss,self.rmse,self.reg_losses,self.rec_loss_u,self.rec_loss_i,
                                self.summ_handle.summ_loss,self.summ_handle.summ_metric],
                                  )
                if counter%50==0:
                # 记录w,b
                    summ_ew, summ_dw, summ_eb, summ_db = self.sess.run([self.summ_handle.summ_enc_w, self.summ_handle.summ_enc_b,
                                                               self.summ_handle.summ_dec_w, self.summ_handle.summ_dec_b],
                                                             )
                    for i in range(self.n_layers):
                        self.writer.add_summary(summ_ew[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_dw[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_eb[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_db[i], epoch * n_batch + batch)
                if epoch == self.n_epoch-1:
                    mean_rmse_batch += rmse
            train_mean_rmse = mean_rmse_batch/n_batch
            if epoch == self.n_epoch-1:
                rmse = train_mean_rmse
            print("epoch ",epoch," train loss: ", loss,"rmse: ", rmse," reg loss: ",reg_loss[-1],
                  " loss u: ",loss_u," loss i: ",loss_i,
                 " time:",str(time.time()-begin_time))

        # ---------------------------------- fine-tune --------------------------------------
        for epoch in range(self.n_epoch):
            if epoch > self.change_lr_epoch:
                self.lr = self.lr * self.lr_decay
            mean_rmse_batch = 0
            for batch in range(n_batch):
                counter += 1
                _, loss, rmse, rec_loss, reg_loss,loss_u,loss_i, summ_loss,summ_rmse =\
                    self.sess.run([self.optimizer,self.loss,self.rmse, self.rec_loss,self.reg_losses,self.rec_loss_u,self.rec_loss_i,
                                 self.summ_handle.summ_loss,self.summ_handle.summ_metric],
                                  )
                self.writer.add_summary(summ_loss,epoch * n_batch + batch)
                self.writer.add_summary(summ_rmse, epoch * n_batch + batch)
                if counter%50==0:
                # 记录w,b
                    summ_ew, summ_dw, summ_eb, summ_db = self.sess.run([self.summ_handle.summ_enc_w, self.summ_handle.summ_enc_b,
                                                               self.summ_handle.summ_dec_w, self.summ_handle.summ_dec_b],
                                                             )
                    for i in range(self.n_layers):
                        self.writer.add_summary(summ_ew[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_dw[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_eb[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_db[i], epoch * n_batch + batch)
                if epoch == self.n_epoch-1:
                    mean_rmse_batch += rmse
            train_mean_rmse = mean_rmse_batch/n_batch
            if epoch == self.n_epoch-1:
                rmse = train_mean_rmse
            print("epoch ",epoch," train loss: ", loss,"rmse: ", rmse," rec loss: ", rec_loss," reg loss: ",reg_loss[-1],
                  " loss u: ",loss_u," loss i: ",loss_i,
                 " time:",str(time.time()-begin_time))

        # ----------------------------------- 获取预测R -----------------------------------------------
        self.is_training = False
        Rfilename = "./" + self.data_dir + 'R' + '1_train.tfrecords'
        data_generator = load_data_func(Rfilename, self.data_dir, self.Rshape, self.Usize, self.Isize,
                                        n_batch, shuffle=False)
        batch_u, batch_i, batch_R = next(data_generator)
        self.calculate(batch_R, batch_u, batch_i)

        threads2 = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        R_hat, R = self.sess.run([self.R_hat, self.R])

        np.save(self.data_dir+'Rhat.npy', R_hat)
        coord.request_stop()
        coord.join(threads)
        coord.join(threads2)
        return train_mean_rmse, R, R_hat
