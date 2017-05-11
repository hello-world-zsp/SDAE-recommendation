# -*- coding: utf8 -*-
import tensorflow as tf
import sys
import ast
import os

from SDAE import *
from dataUtils import getData,read_data_batch

## 都改成flag吧！
flags = tf.app.flags
# flags.DEFINE_string('data_dir', './data/', 'Directory for storing data')
# FLAGS = flags.FLAGS
sdae_args = {
        "noise"     : .1,
        "n_nodes_u"   : (200,100),
        "n_nodes_i"   : (200,100),
        "learning_rate": .001,
        "n_epochs"  : 50,
        "data_dir": 'data/ml-100k/',
        "batch_size": 800,
        "rho"       :0.05,
        "reg_lambda":0.01,
        "sparse_lambda":0,
        "alpha"     :0.2,
        "beta"      :1,
        "delta"     :1,
        "save_freq":(2,1)
}

def main():
    # ------------------ 读参数、数据 ---------------------
    if len(sys.argv) < 2:
        print("Using defaults:\n".format(sys.argv[0]))
    for arg in sys.argv[1:]:        # argv[0]代表代码本身文件路径，因此要从第二个开始取参数。
        k, v = arg.split('=', 1)
        sdae_args[k] = ast.literal_eval(v)

    for k in ('learning_rate', 'n_epochs', 'n_nodes_u', 'n_nodes_i','noise',
              'rho','reg_lambda',"alpha",'beta','delta','save_freq'):
        sdae_args[k] = solo_to_tuple(sdae_args[k],n=2)
    print("Stacked DAE arguments: ")
    for k in sorted(sdae_args.keys()):
        print("\t{:15}: {}".format(k, sdae_args[k]))

    if not os.path.isdir(sdae_args["data_dir"]):
        os.makedirs(sdae_args["data_dir"])
    fileR = sdae_args["data_dir"]+'R.npy'
    fileU = sdae_args["data_dir"] + 'User.npy'
    fileI = sdae_args["data_dir"] + 'Item.npy'
    R,U,I = getData(fileR,fileItem=fileI,fileUser=fileU)
    # ----------------------------------------------------
    # ----------------- 模型初始化 ------------------------
    with tf.Session() as sess:
        print("Initializing...")
        sdae = SDAE(sess,R.shape,U.shape,I.shape,is_training = True,**sdae_args)
        print("build model...")
        sdae.build(is_training = True)
        print("training...")
        features = sdae.train(load_data_func=read_data_batch)

        # mlp = MLPrec(sess, trainX.shape[1], is_training=True, **mlp_args)  # 一个多FC层的 enc-dec结构的网络
        # mlp.build(is_training=True)
        # mlp.train(trainX)


        # mlp = MLP(sess,features.shape[1],is_training = True,**mlp_args)  # 一个多FC层的 enc-dec结构的网络
        # mlp.build(is_training = True)
        # mlp.train(features,trainY[:features.shape[0]],valX,valY)


if __name__ == '__main__':
    main()