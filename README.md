
# SDAE-recommendation system

-----------------
这是master branch, 另有个tfrecord的branch,里面有个'完整版'文件夹，是整理好的代码和说明。

---------------
# Master branch
- 运行 trainMLPrec.py,模型部分主要在MLPrec.py里面。即trainMLPrec.py->MLPrec.py.
- *tfrecords的是使用tfrecords文件读取数据的相应代码。整体速度比使用placeholder+feed ndarry格式数据的形式快15%。
- SDAE的协同过滤只实现了一层。运行train.py，可以训练，训练完一层后面就会报错。trian.py->SDAE.py->DAE.py的结构。


-----------------------
# MLPrec网络结构：
- 电影评分数据集，ml-100k
- Users Net
    - 4层编码层，__得到U__，再接4层解码层，得到重建值。共8层。
    - 每层输入：上一层的特征+用户特征（side information,年龄、职业），+ 表示拼接。
        - 对第一层，“上一层特征”指该用户对各商品的评分。 
    - 每层都是全连接层。wx+b -> batchnormalization -> sigmoid ->输出。
    - __这样sigmoid的输出是0-1的，decoder最后一层输出不加sigmoid__
    - L2正则化，对各层w和b。
- Items Net和UsersNet结构完全一样，__得到V__
    - 商品特征是电影流派
- 用户年龄特征maxabs_rescale到0-1，用户职业和电影流派是one-hot编码。
- 总loss: mse(R-UV), UsersNet重建误差，ItemsNet重建误差，正则项，||U||,||V||的加权和
    - mse是矩阵各元素误差平方的均值，只针对R有评分的项计算。（mask是sign(abs(x))取到的）
    - __||U||的定义文献没有明说，我是取得U各行（代表各用户）norm的均值。__
- __训练时候，每个batch读batch_size个用户和他们对应的batch_size个评分，用部分的side information和R去训练网络。__
