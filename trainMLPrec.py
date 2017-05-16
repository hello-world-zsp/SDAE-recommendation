# -*- coding: utf8 -*-

import sys
import ast
import os

from MLPrec import *
from dataUtils import data_generator,getData

## 都改成flag吧！
flags = tf.app.flags
# flags.DEFINE_string('data_dir', './data/', 'Directory for storing data')
# FLAGS = flags.FLAGS
mlp_args = {
        "noise"     : 0.0,
        "n_nodes"   : (400, 200, 100, 64),
        "learning_rate": .0001,
        "n_epoch"  : 200,
        "data_dir": 'data/ml-100k/',
        "batch_size": 100,
        "rho"       :0.05,
        "reg_lambda":0.01,
        "sparse_lambda":0,
        "alpha"     :0.2,
        "beta"      :0.8,
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

    print("MLP_rec arguments: ")
    for k in sorted(mlp_args.keys()):
        print("\t{:15}: {}".format(k, mlp_args[k]))

    if not os.path.isdir(mlp_args["data_dir"]):
        os.makedirs(mlp_args["data_dir"])
    fileR = mlp_args["data_dir"] + 'R.npy'
    fileU = mlp_args["data_dir"] + 'User.npy'
    fileI = mlp_args["data_dir"] + 'Item.npy'
    R,U,I = getData(fileR,fileItem=fileI,fileUser=fileU)
    num_vals = 5        # 交叉验证5次
    # ----------------------------------------------------
    # ----------------- 模型初始化 ------------------------
    with tf.Session() as sess:
        print("Initializing...")
        model = MLPrec(sess,R.shape,U.shape,I.shape,is_training = True,**mlp_args)
        print("build model...")
        model.build()
        print("training...")
        val_rmse = 0
        for i in range(1,num_vals+1):
            val_rmse += model.train(i,load_data_func=data_generator)
        print (val_rmse/num_vals)

if __name__ == '__main__':
    main()