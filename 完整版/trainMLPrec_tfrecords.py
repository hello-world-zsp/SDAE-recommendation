# -*- coding: utf8 -*-

import sys
import ast
import tensorflow as tf
import time
import utils

from MLPrec_tfrecords import *
from tfrecords_dataUtils import data_generator_tfrecords_CF

"""
协同过滤MLP网络，根据输入的用户特征U、商品特征I、用户-商品评分信息R，预测未评分项的评分。
所用算法来自参考文献 [AAAI 2017]A Hybrid Collaborative Filtering Model with Deep Structure for Recommender Systems
网络先训练用户side information的提取(U网络)，再训练商品side information的提取（I网络），得到U和V；
然后根据 R-UV 的误差fine-tune U和I网络。

定义好mlp_args中的参数,Rshape, Ushape, Ishape。
定义好训练数据生成器data_generator_tfrecords_CF()。
运行python trainMLPrec_tfrecords.py 即可，无需其他参数。
返回训练和验证集的rmse
"""
# MLP网络参数
mlp_args = {
        "noise"     : 0.3,                        # 噪声水平
        "n_nodes"   : (256, 128, 64),             # 网络各层节点数
        "learning_rate": .0001,                  # 学习率
        "n_epoch"  : 200,                        # R迭代次数
        "n_epoch_u"  : 200,                      # U迭代次数
        "n_epoch_i"  : 200,                     # I迭代次数
        "data_dir": 'data/ml-100k/',                     # 数据存储路径
        "batch_size":256,                         # 每批训练数据量
        "rho"       :0.05,                         # 稀疏性水平
        "reg_lambda":0.01,                        # 正则化系数
        "sparse_lambda":0,                        # 稀疏项系数
        "alpha"     :0.5,                          # loss中权重因子，来自文献
        "beta"      :0.5,                          # loss中权重因子，来自文献
        "delta"     :1,                            # loss中权重因子，来自文献
}

def main():
    # ------------------ 读参数、数据 ---------------------
    if len(sys.argv) < 2:
        print("Using defaults:\n".format(sys.argv[0]))
    for arg in sys.argv[1:]:
        k, v = arg.split('=', 1)
        mlp_args[k] = ast.literal_eval(v)
    print("MLP_rec arguments: ")
    for k in sorted(mlp_args.keys()):
        print("\t{:15}: {}".format(k, mlp_args[k]))
    if not os.path.isdir(mlp_args["data_dir"]):
        os.makedirs(mlp_args["data_dir"])
    # 获取各数据统计信息
    Rshape = (943, 1682)
    Ushape = (943, 44)
    Ishape = (1682, 20)

    # ----------------------------------------------------
    begin = time.time()
    with tf.Session() as sess:
        print("Initializing...")
        model = MLPrec(sess,Rshape,Ushape,Ishape,is_training = True,**mlp_args)
        print("training...")
        train_rmse, R, R_hat = model.train(load_data_func=data_generator_tfrecords_CF)
        train_rmse = train_rmse
    print ('train rmse', train_rmse)
    Rrec = utils.recommand(R, R_hat)
    np.save(mlp_args["data_dir"]+'result.npy',Rrec)
    print ('total time: ', time.time()-begin)

if __name__ == '__main__':
    main()
