# -*- coding: utf8 -*-
import tensorflow as tf
import sys
import ast
import os

from tfrecords_SDAE import *
from readData import *
import numpy as np

"""
提取商品特征，返回提取的特征，各层特征保存在.data/goods_train_feature_x.tfrecords
"""
# SDAE网络参数
sdae_args_goods = {
        "noise"     : .1,                                # 噪声水平0-1.0
        "n_nodes"   : (512, 256, 128),                   # 各DAE层节点数
        "learning_rate": (.0001, 0.001, 0.001),         # 各层学习率
        "n_epochs"  : (200, 150, 150),                   # 各层迭代次数
        "rho"       :(0.05, 0.02, 0.02),                  # 各层稀疏水平
        "data_dir": './data/',                          # 数据存储和读取的路径
        "batch_size": 50,                                # batch_size
        "reg_lambda":0.0,                                # 正则化系数
        "sparse_lambda":1.0,                             # 稀疏项系数
        "name"      :'goods'                            # 提取特征项的名称
}

def main():
    # ------------------ 读参数、数据 ---------------------
    if len(sys.argv) < 2:
        print("Using defaults:\n".format(sys.argv[0]))
    for arg in sys.argv[1:]:        # argv[0]代表代码本身文件路径，因此要从第二个开始取参数。
        k, v = arg.split('=', 1)
        sdae_args_goods[k] = ast.literal_eval(v)
    for k in ('learning_rate', 'n_epochs', 'n_nodes', 'noise', 'rho'):
        sdae_args_goods[k] = solo_to_tuple(sdae_args_goods[k], n=3)
    print("Stacked DAE arguments: ")
    for k in sorted(sdae_args_goods.keys()):
        print("\t{:15}: {}".format(k, sdae_args_goods[k]))
    if not os.path.isdir(sdae_args_goods["data_dir"]):
        os.makedirs(sdae_args_goods["data_dir"])


    # 获取一下数据信息：有多少条训练数据和验证集数据
    trainX, valX,trainIdx,valIdx,trainY,valY = load_goods_data(train_ratio=0.9, use_cat=True)
    # ----------------- 模型初始化、训练 --------------------
    with tf.Session() as sess:
        print("Initializing...")
        sdae = SDAE(sess, trainX.shape, valX.shape, is_training=True, **sdae_args_goods)
        print("training...")
        features_goods, val_features = sdae.train()
    # ------------------------------------------------------
    np.save(sdae_args_goods["data_dir"] + "goods_feature", features_goods)

if __name__ == '__main__':
    main()
