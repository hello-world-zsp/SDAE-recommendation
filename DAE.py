# -*- coding: utf8 -*-
import tensorflow as tf
from utils import *
from dataUtils import save_batch_data
import time
import numpy as np
from scipy.misc import imsave

class DAE(object):
    def __init__(self, sess, Rshape, Ushape, Ishape, noise=0, u_units=20, i_units=20,layer=0, learning_rate=0.01,
                 n_epoch=100, is_training=True, batch_size=20,decay=0.95, save_freq=1,
                 reg_lambda=0.0, rho=0.05, sparse_lambda=0.0, alpha=1,beta=1,delta=1,
                 summary_handle=None):

        self.sess = sess
        self.is_training = is_training
        self.u_units = u_units              # 用户网络，隐层节点数
        self.i_units = i_units              # 商品网络，隐层节点数
        self.layer = layer                  # 是第几层
        self.n_epoch = n_epoch
        self.n_batch = Rshape[0]*Rshape[1]//(batch_size**2)
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
        self.change_lr_epoch = int(n_epoch*0.3) # 开始改变lr的epoch数
        self.regularizer = tf.contrib.layers.l2_regularizer #使用L1或L2正则化

        self.reg_lambda = reg_lambda        # U,V正则化系数,float
        self.lambda_u = self.reg_lambda           # U网络权重正则化系数
        self.lambda_i = self.reg_lambda            # I网络权重正则化系数
        self.alpha = alpha                  # U和I重建误差中，权衡S与X的系数
        self.beta = beta                    # 总loss中，U网络的重建误差权重
        self.delta = delta                  # 总loss重，I网络的重建误差权重

        self.save_freq = save_freq          # 特征的总保存次数，每次保存n/save_freq条
        self.save_batch_size = Rshape[0]//save_freq
        self.summ_handle = summary_handle
        self.build(self.is_training)

    # ------------------------- 隐层 -------------------------------------
    def hidden(self, input, units, noise, layerlambda,name = "default"):
        input_size = int(input.shape[1])
        with tf.variable_scope(name):
            # mask噪声
            corrupt = tf.layers.dropout(input,rate= noise,training=self.is_training)
            # 加性高斯噪声
            # corrupt = tf.add(input,noise * tf.random_uniform(input.shape))
            ew = tf.get_variable('enc_weights',shape=[input_size, units],
                                 initializer=tf.random_normal_initializer(mean=0.0,stddev=self.stddev),
                                 regularizer=self.regularizer(layerlambda))

            sew = tf.summary.histogram(name + '/enc_weights', ew)

            eb = tf.get_variable('enc_biases',shape=[1,units],
                                initializer=tf.constant_initializer(0.0),dtype=tf.float32,
                                regularizer=self.regularizer(layerlambda))
            seb = tf.summary.histogram(name+'/enc_biases',eb)
            fc1 = tf.add(tf.matmul(corrupt,ew),eb)
            act1 = tf.nn.sigmoid(tf.layers.batch_normalization(fc1))
            character = act1
            self.ew = ew
            self.eb = eb

            dw = tf.transpose(ew)
            db = tf.get_variable('dec_biases',shape=[1,input_size],
                                initializer=tf.constant_initializer(0.0),dtype=tf.float32,
                                regularizer=self.regularizer(layerlambda))
            sdb = tf.summary.histogram(name+'/dec_biases',db)
            self.summ_handle.add_summ(sew, seb,sdb)
            fc = tf.add(tf.matmul(act1,dw),db)

            out = tf.sigmoid(fc)

        return character, out

    def build(self,is_training=True):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.R = tf.placeholder(tf.float32,None,name="Rating_Matrix")
        # --------------- 用户网络 ---------------------------------------------------------
        layer_name = "User_layer" + str(self.layer)
        # self.u_x = tf.placeholder(tf.float32,[self.batch_size, self.Rshape[1]+self.Usize],name="user_input")
        self.u_x = tf.placeholder(tf.float32, [None, self.Rshape[1] + self.Usize], name="user_input")
        self.U, self.u_out = self.hidden(self.u_x,self.u_units,
                                                   self.noise,self.lambda_u, name=layer_name)

        reg_losses = tf.losses.get_regularization_losses(layer_name)
        for loss in reg_losses:
            tf.add_to_collection('losses' + layer_name, loss)
        reg_losses_u = tf.get_collection('losses'+layer_name)
        self.rec_loss_u = mse_by_part(self.u_out, self.u_x,self.Rshape[1],self.alpha)
        tf.add_to_collection('losses'+layer_name,self.rec_loss_u)
        self.sparse_loss = self.sparse_lambda * sparse_loss(self.rho,self.U)
        tf.add_to_collection('losses' + layer_name, self.sparse_loss)
        self.u_loss = tf.add_n(tf.get_collection('losses'+layer_name))  # U网络重建误差
        self.reg_losses = list(reg_losses_u)
        # --------------- 商品网络 ---------------------------------------------------------

        layer_name = "Item_layer" + str(self.layer)
        self.i_x = tf.placeholder(tf.float32,[None,self.Rshape[0]+self.Isize],name="item_input")
        self.V, self.i_out = self.hidden(self.i_x,self.i_units,
                                                    self.noise, self.lambda_i,name=layer_name)

        reg_losses = tf.losses.get_regularization_losses(layer_name)
        for loss in reg_losses:
            tf.add_to_collection('losses' + layer_name, loss)
        self.reg_losses.extend(tf.get_collection('losses'+layer_name))
        self.rec_loss_i = mse_by_part(self.i_out, self.i_x,self.Rshape[0],self.alpha)
        tf.add_to_collection('losses'+layer_name, self.rec_loss_i)
        self.sparse_loss = self.sparse_lambda * sparse_loss(self.rho,self.V)
        tf.add_to_collection('losses' + layer_name, self.sparse_loss)
        self.i_loss = tf.add_n(tf.get_collection('losses'+layer_name))  # I网络重建误差

        # 总loss
        self.R_hat = tf.matmul(self.U,tf.transpose(self.V))
        self.rec_loss = mse_mask(self.R,self.R_hat)
        reg_loss_u_and_i = tf.reduce_sum(tf.norm(self.U,axis=1))+tf.reduce_sum(tf.norm(self.V,axis=1))
        self.reg_losses.append(reg_loss_u_and_i)
        self.loss = self.rec_loss + self.reg_lambda * reg_loss_u_and_i+\
                    self.beta * self.u_loss + self.delta * self.i_loss
        # 输出预测准确度，和文献比一下
        self.rmse = rmse_mask(self.R,self.R_hat)

    def train(self, read_data_path, save_data_path, load_data_batch_func, train_vals, summ_writer, summ_handle):
        temp = set(tf.all_variables())
        self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9).minimize(self.loss,var_list=train_vals)
        # adam中有slot,需要初始化。
        tf.initialize_variables(set(tf.all_variables()) - temp).run()

        counter = 0
        current_lr = self.lr_init
        begin_time = time.time()
        # --------------------------------- 训练 --------------------------------------------------
        for epoch in range(self.n_epoch-1):
            if epoch > self.change_lr_epoch:
                current_lr = current_lr * self.lr_decay
            for batch in range(self.n_batch):
                counter += 1
                # 每个batch读一次数据 想按batch读，就需要每层的特征图都存下来，很占空间
                batch_u, batch_i, batch_R = load_data_batch_func(read_data_path,
                                        batch_size=self.batch_size)

                _, loss, rec_loss, reg_loss,loss_u,loss_i, U, V, summ_loss =\
                    self.sess.run([self.optimizer,self.loss,self.rmse, self.reg_losses,self.rec_loss_u,self.rec_loss_i,
                                self.U, self.V, summ_handle.summ_loss[0]],
                                  feed_dict={self.u_x: batch_u,self.i_x:batch_i, self.R:batch_R,self.lr:current_lr})
                summ_writer.add_summary(summ_loss,epoch * self.n_batch + batch)
                if counter%50==0:
                # 记录w,b
                    summ_ew, summ_eb, summ_db = self.sess.run([summ_handle.summ_enc_w[0], summ_handle.summ_enc_b[0],
                                                                        summ_handle.summ_dec_b[0]],
                                                              feed_dict={self.u_x: batch_u, self.i_x: batch_i,
                                                                         self.R: batch_R, self.lr: current_lr})
                    summ_writer.add_summary(summ_ew, epoch * self.n_batch + batch)
                    summ_writer.add_summary(summ_eb, epoch * self.n_batch + batch)
                    summ_writer.add_summary(summ_db, epoch * self.n_batch + batch)
            print("epoch ",epoch," train loss: ", loss," rec loss: ", rec_loss," reg loss: ",reg_loss,
                  " loss u: ",loss_u," loss i: ",loss_i,
                  " sparse coef: ",tf.reduce_mean(tf.reduce_mean(U)).eval()+tf.reduce_mean(tf.reduce_mean(V)).eval(),
                 " time:",str(time.time()-begin_time))

        #----------------------- 最后一个epoch,除了训练，还要记录wb的分布 --------------------------
        epoch = self.n_epoch - 1
        for batch in range(self.n_batch):
            counter += 1
            batch_u, batch_i, batch_R = load_data_batch_func(read_data_path,
                                                             batch_size=self.batch_size)
            _, loss, self.ewarray, self.ebarray, summ_loss, summ_ew,summ_eb,summ_db\
                = self.sess.run([self.optimizer, self.loss,self.ew,self.eb,
                                summ_handle.summ_loss[0],summ_handle.summ_enc_w[0],summ_handle.summ_enc_b[0],
                                summ_handle.summ_dec_b[0]],
                                feed_dict={self.u_x: batch_u, self.i_x: batch_i, self.R: batch_R, self.lr: current_lr})
            summ_writer.add_summary(summ_loss, epoch * self.n_batch + batch)
            summ_writer.add_summary(summ_ew, epoch * self.n_batch + batch)
            summ_writer.add_summary(summ_eb, epoch * self.n_batch + batch)
            summ_writer.add_summary(summ_db, epoch * self.n_batch + batch)

        # ----------------------- 记录抽取的特征 --------------------------
        Us = []
        Vs = []
        outs =[]
        save_batch_data(save_data_path, inputU = Us, inputV=Vs, is_New=True)
        for batch in range(self.save_freq):
            batch_u, batch_i, batch_R = load_data_batch_func(read_data_path,
                                                             batch_size=self.save_batch_size)
            U, V, Rhat = self.sess.run([self.U, self.V, self.R_hat],
                                feed_dict={self.u_x: batch_u, self.i_x: batch_i, self.R: batch_R, self.lr: current_lr})
            Us.append(U)
            Vs.append(V)
            outs.append(Rhat)
            save_batch_data(save_data_path,inputU=U,inputV=V)
        self.next_U = np.concatenate(tuple(Us))
        self.next_V = np.concatenate(tuple(Vs))
        self.rec = np.concatenate(tuple(outs))
        print("feature saved. time:",str(time.time()-begin_time))
        # -------------------------------------------------------------------



