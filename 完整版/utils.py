# -*- coding: utf8 -*-
import tensorflow as tf
from scipy.misc import imsave
import numpy as np
import os, sys


def solo_to_tuple(val, n=3):
    if type(val) in (list, tuple):
        return val
    else:
        return (val,)*n


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def loss_x_entropy(output, target):
    """Cross entropy loss
    Args:
        output: tensor of net output
        target: tensor of net we are trying to reconstruct
    Returns:
        Scalar tensor of cross entropy
    """
    with tf.name_scope("xentropy_loss"):
        net_output_tf = tf.convert_to_tensor(output, name='input')
        target_tf = tf.convert_to_tensor(target, name='target')
        cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf, name='log_output'), target_tf),
                               tf.multiply(tf.log(1 - net_output_tf),(1 - target_tf)))
        return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                     name='xentropy_mean')


def save_image(input, name,n_show):
    n_per_line = n_show ** 0.5                                              # 每行10张图
    n_lines = n_show//n_per_line
    h = input.shape[1] ** 0.5                                               # 图像大小
    w = h
    img_total = np.zeros([h * n_lines, w * n_per_line])                     # 灰度图
    for i in range(n_show):
        rec = (input[i] + 1) * 127                                          # 将(0,1)变换到（0,254）
        img = np.reshape(rec,[h,w])                                         # 行向量变图像
        row = i // n_lines
        col = i % n_per_line
        img_total[row * h:(row+1) * h,col*w:(col+1)*w] = img
    imsave(name, img_total)


def save_batch_data(name,input_data,is_New=False):
    save_name = os.path.join(sys.path[0], name)
    if not is_New:
        temp = np.load('./'+name).astype(np.float32)
        if temp.shape[0]>0:
            input_data = np.concatenate((temp,input_data), axis=0)

    np.save(save_name,input_data)


def sparse_loss(rho,features):
    # 其实是交叉熵
    rho_hat = tf.reduce_mean(features)
    return rho * tf.log(rho/rho_hat) +(1-rho) * tf.log((1-rho)/(1-rho_hat))


def mse(x, y):
    """mean square error loss
    Args:
        x: tensor of net output
        y: tensor of net we are trying to reconstruct, same shape with x
    Returns:
        Scalar tensor of mse loss
    """
    return tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(x,y),2.0),axis=1))


def mse_by_part(x, y, s_size, alpha):
    """文献中计算loss所需的分块mse，x的前半部分与后半部分重建loss权重不同
    Args:
        x: tensor of net output
        y: tensor of net we are trying to reconstruct, same shape with x
        s_size: int, divide x by s_size into 2 parts
        alpha: float, weights of loss for 1st part
    Returns:
        Scalar tensor of mse loss
    """
    assert (0 <= alpha <= 1), 'alpha must set between 0 and 1.'
    result1 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(x[:,:s_size], y[:, :s_size]), 2.0), axis=1))
    result2 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(x[:,s_size:], y[:, s_size:]), 2.0), axis=1))
    return alpha * result1 + (1-alpha)*result2


def mse_mask(x,y):
    """mean square error loss with mask
    Args:
        x: tensor of net output
        y: tensor of net we are trying to reconstruct, same shape with x
    Returns:
        Scalar tensor of mse loss
    """
    # x=0的地方，不计算mse,用的sign,>0的地方置1，=0的地方置0
    mask = tf.sign(tf.abs(x))
    return tf.reduce_mean(tf.reduce_sum(mask * tf.pow(tf.subtract(x, y), 2.0), axis=1))


def rmse_mask(x, y):
    """root mean square error loss with mask
    Args:
        x: tensor of net output
        y: tensor of net we are trying to reconstruct, same shape with x
    Returns:
        Scalar tensor of rmse loss
    """
    # x=0的地方，不计算mse,用的sign,>0的地方置1，=0的地方置0
    mask = tf.sign(tf.abs(x))
    num = tf.reduce_sum(mask)
    mse = tf.reduce_sum(mask * tf.pow(tf.subtract(x, y), 2.0))
    return tf.sqrt(mse/num),num


def recommand(R,R_hat):
    """
    根据预测评分Rhat和已有评分矩阵R，过滤掉已有评分项，对各用户预测评分排序从大到小
    :param R: 已有评分矩阵R， ndarry
    :param R_hat: 预测评分Rhat, ndarry
    :return: 拍过序的预测评分，2D ndarry
    """
    mask = np.sign(np.abs(R))       # 过滤掉已有的评分
    Rpre = R_hat * mask
    Rpre = -np.sort(-Rpre)
    return Rpre




class SummaryHandle():
    """
    A summary handler
    """
    def __init__(self):
        self.summ_enc_w = []
        self.summ_dec_w = []
        self.summ_enc_b = []
        self.summ_dec_b = []
        self.summ_loss = None

    def add_summ(self,e_w=None,d_w=None,e_b=None,d_b=None):
        if e_w is not None:
            self.summ_enc_w.append(e_w)
        if d_w is not None:
            self.summ_dec_w.append(d_w)
        if e_b is not None:
            self.summ_enc_b.append(e_b)
        if d_b is not None:
            self.summ_dec_b.append(d_b)
