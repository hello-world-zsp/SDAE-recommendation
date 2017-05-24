# -*- coding: utf8 -*-

import sys
import ast
import tensorflow as tf
import time

# from MLPrec import *
from MLPrec_tfrecords import *
from dataUtils import data_generator,getData
from tfrecords_dataUtils import data_generator_tfrecords

## 都改成flag吧！
flags = tf.app.flags
# flags.DEFINE_string('data_dir', './data/', 'Directory for storing data')
# FLAGS = flags.FLAGS
mlp_args = {
        "noise"     : 0.0,
        "n_nodes"   : (512,256,128,64),
        "learning_rate": .0001,
        "n_epoch"  : 100,
        "data_dir": 'data/ml-100k/',
        "batch_size": 50,
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
    val_rmse = 0
    begin = time.time()
    for i in range(1, num_vals + 1):
        print ('test ',i)
        tf.reset_default_graph()
        with tf.Session() as sess:
            print("Initializing...")
            model = MLPrec(sess,R.shape,U.shape,I.shape,is_training = True,**mlp_args)
            print("training...")
            val_rmse += model.train(i,load_data_func=data_generator_tfrecords)

    print (val_rmse/num_vals)
    print ('total time: ',time.time()-begin)

if __name__ == '__main__':
    main()