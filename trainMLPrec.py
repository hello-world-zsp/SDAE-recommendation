# -*- coding: utf8 -*-
import tensorflow as tf
import sys
import ast
import os

from MLPrec import *
from dataUtils import getData,read_data_batch

## 都改成flag吧！
flags = tf.app.flags
# flags.DEFINE_string('data_dir', './data/', 'Directory for storing data')
# FLAGS = flags.FLAGS
mlp_args = {
        "noise"     : .1,
        "n_nodes"   : (200, 100, 50),
        "learning_rate": .001,
        "n_epoch"  : 400,
        "data_dir": 'data/ml-100k/',
        "batch_size": 300,
        "rho"       :0.05,
        "reg_lambda":0.1,
        "sparse_lambda":0,
        "alpha"     :0.8,
        "beta"      :1,
        "delta"     :1,
        "save_freq":1
}

def main():
    # ------------------ 读参数、数据 ---------------------
    if len(sys.argv) < 2:
        print("Using defaults:\n".format(sys.argv[0]))
    for arg in sys.argv[1:]:        # argv[0]代表代码本身文件路径，因此要从第二个开始取参数。
        k, v = arg.split('=', 1)
        mlp_args[k] = ast.literal_eval(v)

    print("Stacked MLP_rec arguments: ")
    for k in sorted(mlp_args.keys()):
        print("\t{:15}: {}".format(k, mlp_args[k]))

    if not os.path.isdir(mlp_args["data_dir"]):
        os.makedirs(mlp_args["data_dir"])
    fileR = mlp_args["data_dir"]+'R.npy'
    fileU = mlp_args["data_dir"] + 'User.npy'
    fileI = mlp_args["data_dir"] + 'Item.npy'
    R,U,I = getData(fileR,fileItem=fileI,fileUser=fileU)
    # ----------------------------------------------------
    # ----------------- 模型初始化 ------------------------
    with tf.Session() as sess:
        print("Initializing...")
        model = MLPrec(sess,R.shape,U.shape,I.shape,is_training = True,**mlp_args)
        print("build model...")
        model.build()
        print("training...")
        model.train(load_data_func=read_data_batch)


if __name__ == '__main__':
    main()