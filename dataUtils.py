# -*- coding: utf8 -*-
import numpy as np
import os,sys
import math
from sklearn import preprocessing


def getInfo(fileInfo,fileOccup):
    # 获取用户数、电影数、职业列表
    f = open(fileInfo)
    line = f.readline()
    content = line.split(' ')
    num_u = int(content[0])
    line = f.readline()
    content = line.split(' ')
    num_i = int(content[0])

    f = open(fileOccup)
    oc_list = f.readlines()
    oc_list =[oc.split('\n')[0] for oc in oc_list]

    return num_u, num_i, oc_list


# 建立所有评分的矩阵
def readR(filename,Rshape):
    Mrate = np.zeros(Rshape)
    f = open(filename, encoding='utf8')
    line = f.readline()
    while True:
        try:
            content = line.split('\t')
            u,i,r = int(content[0]),int(content[1]),int(content[2])
            Mrate[u][i] = r
            line = f.readline()
        except:
            break
    # Mrate = Mrate/5.0           # 评分归一化
    return Mrate


# 商品信息只用流派(one-hot)
def readItem(filename,ni):
    Mitem = np.zeros((ni,20))   #1个id,19个流派
    f = open(filename, encoding='utf16')
    line = f.readline()
    while line!='':
        # content = re.split('[|,(),\n]',line)
        content = line.split('|')
        L = list()
        L.append(content[0])
        L.extend(content[5:-1])
        L.extend(content[-1].split('\n')[0])
        try:
            L = [int(i)for i in L]
        except:
            print (L[0])
        Mitem[L[0]-1] = np.array(L)
        line = f.readline()
    Mitem = Mitem[:,1:]           # 不用id
    return Mitem


def getData(fileR, fileItem, fileUser):
    R = np.load(fileR)
    mask = np.ma.masked_where(R==0,R)
    # print (R)
    Item = np.load(fileItem)
    # print (Item)
    User = np.load(fileUser)
    # print(User)
    return R,User,Item


def readUser(filename,nu,occup_list ):
    # 获取用户矩阵，各列id,age,gender,occupation，age和occupation是one-hot的
    no = len(occup_list)
    Muser = np.zeros((nu,(3+no)))   # nu行，id,age,gender,occupation列
    f = open(filename)
    line = f.readline()
    while line != '':
        content = line.split('|')
        L = list()
        L.append(content[0])        # id
        L.append(content[1])        # age
        if content[2] == "M":      # gender
            L.append(0)
        else:
            L.append(1)
        one_hot = [content[3].split('\n')[0]==oc for oc in occup_list]
        L.extend(one_hot)
        L = [int(s) for s in L]
        Muser[L[0] - 1] = np.array(L)
        line = f.readline()
    Muser = Muser[:,1:]             # 去掉id列
    Muser[:, 0] = preprocessing.maxabs_scale( Muser[:,0])   # age归一化
    return Muser

def read_data_batch(path,batch_size=None):
    U = np.load("./" + path+'User.npy')
    I = np.load("./" + path+'Item.npy')
    R = np.load("./" + path+'R.npy')
    ru = np.random.permutation(U.shape[0])      # shuffle
    U = U[ru,:]
    batch_U = U[:batch_size]
    ri = np.random.permutation(I.shape[0])
    I = I[ri,:]
    batch_I = I[:batch_size]
    batch_R_u = R[ru,:][:batch_size]            # 所选用户对应的评分项
    batch_R_i = R[:,ri][:,:batch_size]
    batch_R = batch_R_u[:,ri][:,:batch_size]
    batch_U = np.concatenate((batch_R_u,batch_U),axis=1)
    batch_I = np.concatenate((batch_R_i.T,batch_I),axis=1)          # 转置，不知道方向有没有问题

    return batch_U,batch_I,batch_R


def data_generator(Rfilename,path,nb_batch,batch_size=None):
    U = np.load('./' + path + '/User.npy',mmap_mode='r')
    I = np.load("./" + path + 'Item.npy',mmap_mode='r')
    R = np.load(Rfilename,mmap_mode='r')
    ru = np.random.permutation(U.shape[0])      # 只在第一次读的时候做shuffle
    U = U[ru,:]
    ri = np.random.permutation(I.shape[0])
    I = I[ri,:]
    batch = 0
    while batch <= nb_batch:
        batch_U = U[:batch_size]
        batch_I = I[:batch_size]
        batch_R_u = R[ru,:][:batch_size]            # 所选用户对应的评分项
        batch_R_i = R[:,ri][:,:batch_size]
        batch_R = batch_R_u[:,ri][:,:batch_size]
        batch_U = np.concatenate((batch_R_u,batch_U),axis=1)
        batch_I = np.concatenate((batch_R_i.T,batch_I),axis=1)          # 转置，不知道方向有没有问题
        batch += 1
        yield batch_U,batch_I,batch_R

def save_batch_data(save_path, inputU=[], inputV=[], is_New=False):
    save_name = os.path.join(sys.path[0], save_path)
    name = save_path +'U.npy'
    if not is_New:
        temp = np.load('./'+name).astype(np.float32)
        if temp.shape[0]>0:
            inputU = np.concatenate((temp,inputU), axis=0)
    np.save(save_name+'U.npy',inputU)

    name = save_path +'V.npy'
    if not is_New:
        temp = np.load('./'+name).astype(np.float32)
        if temp.shape[0]>0:
            inputV = np.concatenate((temp,inputV), axis=0)
    np.save(save_name+'V.npy',inputV)

path = './data/ml-100k/'
nu,ni,occup_list = getInfo(path+'u.info',path+'u.occupation')
for i in range(1,6):
    R = readR(path+'u'+str(i)+'.base',(nu,ni))
    np.save(path+'R'+str(i)+'_train',R)
    R = readR(path+'u'+str(i)+'.test',(nu,ni))
    np.save(path+'R'+str(i)+'_val',R)
# Item = readItem(path+"u.item",ni)
# np.save(path+'Item',Item)
# User = readUser(path+'u.user',nu,occup_list )
# np.save(path+'User',User)
# getData(path+'R.npy',path+'Item.npy',path+'User.npy')

