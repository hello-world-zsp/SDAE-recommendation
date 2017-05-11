# -*- coding: utf8 -*-
import os
import tensorflow as tf
from utils import *
from DAE import *

class SummaryHandle():
    def __init__(self):
        self.summ_enc_w = []
        self.summ_enc_b = []
        self.summ_dec_b = []
        self.summ_loss = []

    def add_summ(self,e_w,e_b,d_b):
        self.summ_enc_w.append(e_w)
        self.summ_enc_b.append(e_b)
        self.summ_dec_b.append(d_b)


class SDAE(object):
    def __init__(self, sess, Rshape,Ushape,Ishape, noise=(0.0,0.0,0.0), n_nodes_u=(180, 42, 10), n_nodes_i=(180, 42, 10),
                 learning_rate=1, n_epochs=100, is_training=True,
                 data_dir = None,batch_size=20,num_show=100,rho=(0.05,0.05,0.05),save_freq = (2,1,1),
                 reg_lambda=(0.0,0.0,0.0),sparse_lambda=(1.0,1.0,1.0),
                 alpha=(1.0,1.0,1.0),beta=(1.0,1.0,1.0),delta=(1.0,1.0,1.0)):

        self.sess = sess
        self.is_training = is_training
        self.n_nodes_u = n_nodes_u             # U网络各层节点数
        self.n_nodes_i = n_nodes_i            # I网络各层节点数
        self.n_layers = len(self.n_nodes_u)   # 层数
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.Rshape = Rshape
        self.Ushape = Ushape
        self.Ishape = Ishape

        self.lr = learning_rate
        self.stddev = 0.02                  # 初始化参数用的
        self.noise = noise                  # dropout水平，是tuple
        self.rho = rho                      # 各层稀疏性系数
        self.sparse_lambda = sparse_lambda  # 稀疏loss权重
        self.reg_lambda = reg_lambda        # 正则项权重
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        self.checkpoint_dir = 'checkpoint'
        self.result_dir = 'results'
        self.log_dir = 'logs'
        self.data_dir = data_dir
        self.save_freq = save_freq          # 特征的总保存次数，每次保存n/save_freq条

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

    def build(self,is_training=True):
        self.hidden_layers =[]
        self.loss = []
        self.summary_handles = []

        for i in range(self.n_layers):
            summary_handle = SummaryHandle()
            layer = DAE(self.sess, self.Rshape,self.Ushape,self.Ishape, noise=self.noise[i],
                        u_units=self.n_nodes_u[i],i_units=self.n_nodes_i[i],
                        layer=i, n_epoch=self.n_epochs[i],is_training=self.is_training,
                        batch_size=self.batch_size,learning_rate=self.lr[i],save_freq=self.save_freq[i],
                        rho=self.rho[i],reg_lambda=self.reg_lambda[i],sparse_lambda=self.sparse_lambda,
                        alpha=self.alpha[i],beta=self.beta[i],delta=self.delta[i],
                        summary_handle = summary_handle)
            self.loss.append(layer.loss)
            self.hidden_layers.append(layer)

            # self.summary_character.append(tf.summary.image(layer.next_x, "character" + str(i)))
            summary_handle.summ_loss.append(tf.summary.scalar('loss'+str(i),layer.loss))
            self.summary_handles.append(summary_handle)

        # ------------------------ 提取各层可训练参数 -----------------------------------
        self.train_vals = tf.trainable_variables()
        self.train_vals_layer =[]
        for i in range(self.n_layers):
            self.train_vals_layer.append ( [var for var in self.train_vals if str(i) in var.name.split("/")[0]])
        # ------------------------------------------------------------------------------


    def train(self,load_data_func):
        tf.global_variables_initializer().run()
        self.writer = tf.summary.FileWriter('./'+self.log_dir, self.sess.graph)

        read_data_path = self.data_dir
        features = []
        imgs = []

        for layer in self.hidden_layers:
            idx = self.hidden_layers.index(layer)
            print("training layer: ",idx )
            save_data_path = self.data_dir + 'character' + str(idx)
            layer.train(read_data_path, save_data_path, load_data_func, self.train_vals_layer[idx],
                        summ_writer=self.writer, summ_handle=self.summary_handles[idx])

            read_data_path = save_data_path
            features.append(layer.next_x)
            # 保存重建图
            # save_image(layer.rec,name = self.result_dir+'/rec'+str(idx)+'.png',n_show = self.n_show)
            # if idx == 0:
            #     img = np.add(layer.ewarray.T,
            #                        np.dot(layer.ebarray.T,np.ones([1,self.input_size])))
            # else:
            #     img = np.add(np.dot(layer.ewarray.T,imgs[idx-1]),
            #                        np.dot(layer.ebarray.T, np.ones([1, self.input_size])))
            # img = tf.sigmoid(img).eval()
            # imgs.append(img)
            # save_image(imgs[idx], name=self.result_dir + '/feature' + str(idx) + '.png', n_show=self.n_nodes[idx])
            # save_image(tf.sigmoid(layer.next_x).eval(), name=self.result_dir + '/character' + str(idx) + '.png', n_show=self.n_show)
        features = np.concatenate(tuple(features[1:]),axis = 1)
        return features


