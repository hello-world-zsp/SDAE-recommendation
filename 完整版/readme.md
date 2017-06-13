# 目录
1. 提取用户特征
2. 提取商品特征
3. 获取反馈矩阵R
3. 协同过滤

离线过程文件和所需数据都在 __ml007__ /Zhangsp/gitlab/SDAE-CF 文件夹

-------------
# 提取用户特征 
1. __下载数据__：
    - 翟浩提供的用户数据矩阵在集群上，稀疏矩阵，储存为libsvm格式。
    - __ml002机器__上 /user_profile2/projects/read_libsvm.py，将稀疏矩阵转为稠密矩阵，并存储为npy(user_data.npy). _(这一步因为对spark不熟悉，函数写的很粗糙，for循环嵌套，单机处理，时间很慢，需要再改一下。)_
    - 下载user_data.npy，放到__ml007__ /Zhangsp/gitlab/SDAE-CF/data下。
    - 后续步骤都在__ml007__ /Zhangsp/gitlab/SDAE-CF
2. __转tfrecords文件：__运行tfrecords_dataUtils.py, 取消其中28-30行注释，将npy数据转为tfrecords.得到user_data_train.tfrecords和user_data_val.tfrecords。_注意运行完了再注释上。_
3. __提取特征__：运行SDAE_user.py,提取用户特征。无需参数，得到user_feature.npy文件。
--------
# 提取商品特征
1. __word2vec:__ 使用__ml002__上赵叶宇的generate_vector.py 将商品数据向量化，得到goods_vectors.libsvm
2. __下载:__ 下载goods_vectors.libsvm，放到__ml007__ /Zhangsp/gitlab/SDAE-CF/data下
3. __转tfrecords文件：__运行tfrecords_dataUtils.py, 取消其中25-27行注释，将npy数据转为tfrecords.得到goods_data_train.tfrecords和goods_data_val.tfrecords。_注意运行完了再注释上。_
4. __提取特征__：运行SDAE_goods.py,提取用户特征。无需参数，得到goods_feature.npy文件。

-----------------
# 获取反馈矩阵R
- 没做好。
- R应该是(n_users x n_items)的稠密矩阵。
- R中数据 >= 0, = 0时表示无反馈。
- 可以仿照前例，使用tfrecords_dataUtils.write_tfrecords()将各种格式R矩阵转为tfrecords。

------------------------
# 协同过滤
- 前面得到的user_feature.npy，goods_feature.npy使用tfrecords_dataUtils.write_tfrecords()转为tfrecords文件。
- 运行trainMLPrec_tfrecords.py 进行协同过滤。
    - 该文件需要的数据是前面准备好的类似 user_feature.tfrecords, goods_feature.tfrecords, R_train.tfrecords三份tfrecords文件。
    - 协同过滤的结果为，重建误差 train rmse和各用户的反馈预测。去除用户已经反馈过的项，各行（对应各用户）的预测评分从大到小排序。
    - rmse输出在屏幕上，预测结果保存在./data/Rhat.npy,可供查看。

------------------
# 注
- 之所以采用很多操作，将数据转为tfrecords格式，是因为使用tfrecords格式进行训练可以比使用npy+feed_dict方式训练
    - 速度快17%左右
    - 不用把全部数据load到内存，减少内存占用。这样每个batch都可以使用与内存/显存最大容量相当的数据，提高训练数据量。
- GPU和集群的通讯还没有建立. GPU上没有装pyspark, 各个集群上也没有装tensorflow. 都装好后，可以直接在集群上将数据储存为所需的TFrecords格式，这样可以省去前序很多数据处理和转换的步骤。如此，只需要将各个tfrecords文件准备好，依次运行SDAE_goods.py, SDAE_user.py, trainMLPrec_tfrecords.py即可。
- 各个训练文件虽然运行时不需要提供参数，但是参数都写在了代码的最开头，使用前要检查是否与本批数据相符。尤其是数据的行、列数。
- 出现各种关于tfrecords，queue的错误，或者是程序没报错，但好像卡住了不运行时，要检查提供的tfrecords文件名称、路径、大小是否有误。尤其注意tfrecords_dataUtils.write_tfrecords()与tfrecords_dataUtils.read_and_decode_tfrecords()中写入的数据和读出的数据格式是否对应（默认np.float32）

# 版本
- python2 3均可
- tensorflow 1.0.1
    
