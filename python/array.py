# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 20:43
# @File    : array.py
# for i in tqdm(range(10)):
#    time.sleep(random.random())

import numpy as np


#show np.flatten,np.flat,np.ravel,np.reshape,np.resize

# x=np.arange(1,7,1).reshape(3,2)
# print(x)
# print(x.shape)
# print(x.flatten('C'))#默认参数为"C"，即按照行进行重组
# print(x.flatten('F'))#按照列进行重组
# for i in np.arange(0,6,1):
#   print(x.flat[i])   #将原来的二维数组重组为一行


# b = [[1,2,3],[4,5,6],[7,8,9]]
# b=np.array(b)
# print(type(b))
# a=np.arange(1,10,1).reshape(3,3)
#
#
# for i in np.arange(0,3,1):
#   for j in np.arange(0,3,1):
#     if a[i,j]==b[i,j]:
#       print(a[i,j])
#       print(b[i,j])
# print(b)
# print(a)
# zipped = zip(a,b)
# print(list(zipped))# 需要转化为元组的列表
# numpy.stack(arrays, axis=0)
# 沿着新轴连接数组的序列。
# axis参数指定新轴在结果尺寸中的索引。例如，如果axis=0，它将是第一个维度，如果axis=-1，它将是最后一个维度


###############
# y=np.arange(1,7,1).reshape(3,2)
# print(y.flat)                             #表示将y平铺成一行转化为一维数组
# print(y[:,0])                             #表述输出第一列
# print(y[0:2,0])                           #输出第一列的第0个到第1个元素
# print(np.stack((y,y),axis=0))             #axis=0表示两个数组上下堆叠
# print(np.stack((y.flat, y.flat),axis=0))  #axis=0表示两个数组上下堆叠
# print(np.stack((y.flat, y.flat),axis=1))  #axis=1表示两个数组的每一个对应元素组成在一起
# x=np.arange(1,9,1)
# x=x.reshape(4,2)
# print(list(x.flat))
# x=np.stack((x.flat, x.flat), axis=1)
# print(x)
###########
# x=np.linspace(0,100,800)
# x=s2 = 2 * np.sin(0.02 * np.pi * x)+3*np.cos(0.04*np.pi*x)
# t=np.linspace(0,100,800)
# plt.plot(t,x)
# print(np.pi)
# plt.show()

##########
# X, S, S_, H=[1,2],[2,3],[3,4],[4,5]
# models = [X, S, S_, H]
# names = [
#     "Observations (mixed signal)",
#     "True Sources",
#     "ICA recovered signals",
#     "PCA recovered signals",
# ]
# colors = ["red", "steelblue", "orange"]
# for ii, (model, name) in enumerate(zip(models, names),start=1):   ##enumerate是内置枚举函数，zip函数是打包models，names，，
#     print(ii,model,name)                                          # 1代表start=1，意思是列表第一个元素代号为1
#     for sig, color in zip(model, colors):
#         print(sig,color)