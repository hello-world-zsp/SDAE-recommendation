# -*- coding: utf8 -*-

from utils import *
from dataUtils import save_batch_data
import time
import numpy as np
import os
import tensorflow as tf
from inspect import isgeneratorfunction


class MLPrec(object):
    def __init__(self, sess, Rshape, Ushape, Ishape, n_nodes=(20,20,20), learning_rate=0.01,
                 n_epoch=100, is_training=True, batch_size=20, decay=0.95, save_freq=1,
                 reg_lambda=0.0, rho=0.05, sparse_lambda=0.0, alpha=1,beta=1,delta=1, noise=0,
                 data_dir = None):

        self.sess = sess
        self.is_training = is_training
        self.units = n_nodes              # 隐层节点数
        self.n_layers = len(n_nodes)
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.Rshape = Rshape
        self.Usize = Ushape[1]
        self.Isize = Ishape[1]

        self.lr_init = learning_rate
        self.stddev = 0.02                   # 初始化参数用的
        self.noise = noise                  # dropout水平，是数\
        self.dropout_p = 0.5                # dropout层保持概率
        self.rho = rho                      # 稀疏性系数
        self.sparse_lambda = sparse_lambda
        self.lr_decay = decay
        self.change_lr_epoch = int(n_epoch*0.3)  # 开始改变lr的epoch数
        self.regularizer = tf.contrib.layers.l2_regularizer  #使用L1或L2正则化

        self.reg_lambda = reg_lambda        # U,V正则化系数,float
        self.lambda_u = self.reg_lambda     # U网络权重正则化系数
        self.lambda_i = self.reg_lambda     # I网络权重正则化系数
        self.alpha = alpha                  # U和I重建误差中，权衡S与X的系数
        self.beta = beta                    # 总loss中，U网络的重建误差权重
        self.delta = delta                  # 总loss重，I网络的重建误差权重

        self.summ_handle = SummaryHandle()
        self.save_freq = save_freq          # 特征的总保存次数，每次保存n/save_freq条
        self.save_batch_size = Rshape[0]//save_freq
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
    def encoder(self, input, units, noise, layerlambda, name="encoder"):
        input_size = int(input.shape[1])
        with tf.variable_scope(name):
            # mask噪声
            corrupt = tf.layers.dropout(input,rate= noise,training=self.is_training)
            # 加性高斯噪声
            # corrupt = tf.add(input,noise * tf.random_uniform(input.shape))
            ew = tf.get_variable('enc_weights', shape=[input_size, units],
                                 initializer=tf.random_normal_initializer(mean=0.0,stddev=self.stddev),
                                 regularizer=self.regularizer(layerlambda))
            sew = tf.summary.histogram(name + '/enc_weights', ew)

            eb = tf.get_variable('enc_biases',shape=[1,units],
                                initializer=tf.constant_initializer(0.0),dtype=tf.float32,
                                regularizer=self.regularizer(layerlambda))
            seb = tf.summary.histogram(name+'/enc_biases',eb)
            fc1 = tf.add(tf.matmul(corrupt, ew), eb)
            # fc1 = tf.layers.dropout(fc1,self.dropout_p,training=self.is_training)
            act = tf.nn.sigmoid(tf.layers.batch_normalization(fc1))
            # act = tf.nn.relu(tf.layers.batch_normalization(fc1))
            self.ew = ew
            self.eb = eb
            self.summ_handle.add_summ(e_w=sew, e_b=seb)

        return act

    # ------------------------- 译码层 -------------------------------------
    def decoder(self, input, units, layerlambda, name="encoder"):
        input_size = input.shape[1]
        with tf.variable_scope(name):
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
            fc = tf.add(tf.matmul(input, dw), db)
            # fc = tf.layers.dropout(fc, self.dropout_p, training=self.is_training)
            out = tf.sigmoid(tf.layers.batch_normalization(fc))
            # out = tf.nn.relu(tf.layers.batch_normalization(fc))

        return out,fc

    def build(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.R = tf.placeholder(tf.float32,None,name="Rating_Matrix")
        # --------------- 用户网络 ---------------------------------------------------------
        input_size = self.Rshape[1] + self.Usize
        loss_name = 'loss U'
        self.u_x = tf.placeholder(tf.float32, [None, input_size], name="user_input")
        # ----- encoder -----
        self.U_enc_layers = []
        input_data = self.u_x
        for i in range(self.n_layers):
            layer_name = "U_encoder_layer"+str(i)
            out = self.encoder(input_data, self.units[i], self.noise, self.lambda_u,
                                  name=layer_name)
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
            out,fc = self.decoder(input_data,dec_nodes[i],self.lambda_u,layer_name)
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
        self.i_x = tf.placeholder(tf.float32, [None, input_size], name="item_input")
        # ----- encoder -----
        self.I_enc_layers = []
        input_data = self.i_x
        for i in range(self.n_layers):
            layer_name = "I_encoder_layer" + str(i)
            out = self.encoder(input_data, self.units[i], self.noise, self.lambda_i,
                               name=layer_name)
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
            out,fc = self.decoder(input_data, dec_nodes[i], self.lambda_i, layer_name)
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
        self.rmse = rmse_mask(self.R,self.R_hat)
        self.summ_handle.summ_metric = tf.summary.scalar('rmse', self.rmse)
        # ----------------------------------------------------------------------------------

    def train(self, val_iter, load_data_func):
        # val_iter是进行第几次交叉验证

        self.writer = tf.summary.FileWriter('./'+self.log_dir, self.sess.graph)
        # self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9).minimize(self.loss)
        self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        tf.global_variables_initializer().run()

        counter = 0
        current_lr = self.lr_init
        begin_time = time.time()
        # --------------------------------- 训练 --------------------------------------------------
        for epoch in range(self.n_epoch):
            if epoch > self.change_lr_epoch:
                current_lr = current_lr * self.lr_decay
            if self.batch_size is  None:
                n_batch = 1
            else:
                n_batch = self.Rshape[0]//self.batch_size
            Rfilename = "./" + self.data_dir + 'R' + str(val_iter) + '_train.npy'
            data_generator = load_data_func(Rfilename, self.data_dir, n_batch, batch_size=self.batch_size, shuffle=True)
            for batch in range(n_batch):
                counter += 1
                # 每个batch读一次数据 load_data_func是一个generator,使用next获得它返回的值
                batch_u, batch_i, batch_R = data_generator.next()
                _, loss, rmse, rec_loss, reg_loss,loss_u,loss_i, U, V, summ_loss,summ_rmse =\
                    self.sess.run([self.optimizer,self.loss,self.rmse, self.rec_loss,self.reg_losses,self.rec_loss_u,self.rec_loss_i,
                                self.U, self.V, self.summ_handle.summ_loss,self.summ_handle.summ_metric],
                                  feed_dict={self.u_x: batch_u,self.i_x:batch_i, self.R:batch_R,self.lr:current_lr})
                self.writer.add_summary(summ_loss,epoch * n_batch + batch)
                self.writer.add_summary(summ_rmse, epoch * n_batch + batch)
                if counter%50==0:
                # 记录w,b
                    summ_ew, summ_dw, summ_eb, summ_db = self.sess.run([self.summ_handle.summ_enc_w, self.summ_handle.summ_enc_b,
                                                               self.summ_handle.summ_dec_w, self.summ_handle.summ_dec_b],
                                                              feed_dict={self.u_x: batch_u, self.i_x: batch_i,
                                                                         self.R: batch_R, self.lr: current_lr})
                    for i in range(self.n_layers):
                        self.writer.add_summary(summ_ew[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_dw[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_eb[i], epoch * n_batch + batch)
                        self.writer.add_summary(summ_db[i], epoch * n_batch + batch)
            print("epoch ",epoch," train loss: ", loss,"rmse: ",rmse," rec loss: ", rec_loss," reg loss: ",reg_loss[-1],
                  " loss u: ",loss_u," loss i: ",loss_i,
                  " sparse coef: ",tf.reduce_mean(tf.reduce_mean(U)).eval()+tf.reduce_mean(tf.reduce_mean(V)).eval(),
                 " time:",str(time.time()-begin_time))

        # ----------------------------------- validation -----------------------------------------------
        mean_rmse = 0
        for batch in range(n_batch):
            Rfilename = "./" + self.data_dir + 'R' + str(val_iter) + '_val.npy'
            batch_u, batch_i, batch_R = next(load_data_func(Rfilename, self.data_dir, n_batch, batch_size=self.batch_size))
            loss, rmse = \
                self.sess.run([self.loss, self.rmse],
                              feed_dict={self.u_x: batch_u, self.i_x: batch_i, self.R: batch_R, self.lr: current_lr})
            mean_rmse += rmse
        mean_rmse /= n_batch
        print(" val loss: ", loss, "rmse: ", mean_rmse)
        return rmse




