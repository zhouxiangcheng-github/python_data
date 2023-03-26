# -*- coding:utf-8 _*-
""" 
@file: practice.py 
@time: 2022/01/12
"""
#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.decomposition import PCA, FastICA

# C = 200
# x = np.arange(C)             #起点为0终点为200，步长为1的数据点
#
# a = np.linspace(-2, 2, 25)

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# arr1=np.array([a, a]).reshape(1,50)
# #print(arr1)
# s1 = np.array([a, a, a, a, a, a, a, a]).reshape(200, )
#
# s2 = 2 * np.sin(0.02 * np.pi * x)
# s3 = np.array(20 * (5 * [2] + 5 * [-2]))
# s4 = [1,1,1]
# s5 = [2,2,2]
# s6 = [2,2,2]
# s7=[[1,1,1],
#     [2,2,2],
#     [3,3,3]]
# s8=[[1,1,1],
#     [2,2,2],
#     [3,3,3]]
# i=np.c_[s7,s8]
# print(i)
# ax1 = plt.subplot(311)
# ax2 = plt.subplot(312)
# ax3 = plt.subplot(313)
# ax1.plot(x,s1)
# ax2.plot(x,s2)
# ax3.plot(x,s3)
# plt.show()

# print(type(a))

# print(arr1)
# newarr = arr.reshape(4, 3)

# print(newarr)
#
# ran=2*np.random.random([3,3])
# s=np.array([s1,s2,s3])
# s8 = np.array([s4,s5,s6])
#print(s)
# print(s8)
# ran1=2*np.random.random([3,3])
# print(ran1)
# mix2=ran1.dot(s8)
# print(mix2)
# mix=ran.dot(s)
# print(mix)
# d=np.dot(ran1,s8)
# print(d)
#
# ica = FastICA(n_components=3)  #源个数为3
# mix2 = mix.T       #矩阵取转置
# print(mix2)
#u = ica.fit_transform(mix)
# #print(u)
# u = u.T
# #print(u)
# print(a)
# print(ran)

# a=np.arange(1,13,1)
# a=a.reshape(3,4)
# print(a)
# # a=np.dot(a,a.T)
# # print(a)
# a[:,0]=a[:,0]*2.0
# print(a)

# x=np.arange(2000)
# np.random.seed(0)
# n_samples = 2000
# time = np.linspace(0, 8, n_samples)
#
# s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
# s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
# s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
#
# print(s1)
#
# plt.subplot(311).plot(x,s1)
# #a1.plot(x,s1)
# plt.subplot(312).plot(x,s2)
# #a2.plot(x,s2)
# plt.subplot(313).plot(x,s3)
#
# plt.show()
# rng = np.random.RandomState(42)  #伪随机数Randomstate生成
# S = rng.standard_t(1.5, size=(20000, 2))
# S[:, 0] *= 2.0


#

#print(x)
# Compute ICA
#ica = FastICA(n_components=3)


######
# data=np.loadtxt("F:\python-project/test_data/test1.txt")
# print(data)
# #print(x)
#
# ica = FastICA(n_components=3)
# S_ = ica.fit_transform(data)  # Reconstruct signals
# np.savetxt(r'F:\python-project\test_data\test3.txt', S_)
#
# print(S_)


######
# X, S, S_, H=[1,2],[2,3],[3,4],[4,5]
# models = [X, S, S_, H]
# names = [
#     "Observations (mixed signal)",
#     "True Sources",
#     "ICA recovered signals",
#     "PCA recovered signals",
# ]
# colors = ["red", "steelblue", "orange"]
# for ii, (model, name) in enumerate(zip(models, names),start=1):
#     print(ii,model,name)
#     for sig, color in zip(model, colors):
#         print(sig,color)


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




# df=pd.DataFrame(pd.read_csv('F:\仪控可靠性\\1\\1.csv',encoding='gb2312',usecols=[9],index_col=None))
# df=pd.read_csv('F:\仪控可靠性\\1\\1.csv',encoding='gb2312',usecols=[9],index_col=None)
# df=pd.DataFrame(pd.read_csv('F:\仪控可靠性\\1\\1.csv',encoding='gb2312',usecols=[9],index_col=None))
# print(df.values.max())

#print(df.drop(axis=0,index=0))


#################################
#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  https://github.com/laszukdawid/PyEMD/issues
#
# Feel free to contact for any information.


# import numpy as np
# import matplotlib.pyplot as plt
#
#
# t=np.linspace(0,1,200)
# s=np.loadtxt(r'F:\python-project/test_data/test4.txt')
#
# plt.plot(t,s[:,0])
# plt.show()


# import pandas as pd
# t=pd.read_table('F:\仪控可靠性\数据\\test\\fft_out 2.txt',header=None)
# x=pd.read_table('F:\仪控可靠性\数据\\test\\fft_out.txt',header=None, index_col=False)
# print(type(t.values))
# max=t.values.max()
# print(max)
# for i in range(len(t.values)):
#  if t.values[i] ==max:
#      print(t.values[i])
#      print(x.values[i])
#      break


# import os
# from tensorflow.python.client import device_lib

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"
#
# if __name__ == "__main__":
#     print(device_lib.list_local_devices())

import pandas as pd
import os
import chardet


# 读取文件编码格式
# def get_encoding(file):
#     # 二进制方式读取，获取字节数据，检测类型
#     with open(file, 'rb') as f:
#         return chardet.detect(f.read())['encoding']
# print(get_encoding('F:\仪控可靠性\数据\处理完的数据\\20_03_27\_20_03_27_00_54_54_ZF65SF-01B屏蔽式电动闸阀可靠性强化试验-7-1基准试验-8.1.4基准冷循环试验-冷态循环试验-243V开阀故障恢复阀A常开_开总15_关总16.csv'))


# df = pd.DataFrame(pd.read_csv('F:\仪控可靠性\数据\确定故障\_20_03_27_00_21_25_ZF65SF-01B屏蔽式电动闸阀可靠性强化试验-7-1基准试验-8.1.4基准冷循环试验-冷态循环试验-正常驱动力冷循环带压差阀A常开_开总12_关总13.csv',encoding='gb2312',usecols=[9, 10]))
# df = pd.DataFrame(pd.read_excel('F:\仪控可靠性\数据\\test\\1.xlsx', usecols=[0]))
# df=df.values.reshape(len(df.values),1)
# t=np.linspace(0,1,len(df)).reshape(len(df),1)
# lie=np.hstack((t,df))
# print(lie.shape)
# np.savetxt('F:\仪控可靠性\数据\\test\\1.txt',lie)
# data = pd.DataFrame(pd.read_table('F:\仪控可靠性\数据\\test\\1.txt',sep=' ' ,usecols=[0,1],header=None))
# #data = np.loadtxt(r'F:\仪控可靠性\数据\\test\\1.txt')
# print(data.shape)
# print(data)
# 输出操作系统特定的路径分隔符，win下为"\\",Linux下为"/"


import os
import re
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

# def deal_one(filepath, save_path):
#     filenames_in = filepath  # 输入文件的文件地址
#     filenames_out = save_path  # 新文件的地址
#     pathDir = os.listdir(filenames_in)  # 读取文件夹路径下的文件列表
#     print(len(pathDir))
#     for allDir in pathDir:
#         child = re.findall(r"(.+?).csv", allDir)  # 正则的方式读取文件名，去扩展名
#         if len(child) >= 0:  # 去掉没用的系统文件
#             newfile = ''
#             needdate = child  #### 这个就是所要的文件名
#         print(allDir)
#         domain1 = os.path.abspath(filenames_in)  # 待处理文件位置
#         info = os.path.join(domain1, allDir)  # 拼接出待处理文件名字
#         print(info)
#         df = pd.DataFrame(pd.read_csv(info, encoding='gb2312', dtype={'A阀电流A相工程值': np.float64}, usecols=['A阀电流A相工程值']))
#         # df = df.values
#         domain2 = os.path.abspath(filenames_out)  # 处理完文件保存地址
#         outfo = os.path.join(domain2, allDir)  # 拼接出新文件名字
#         print(outfo)
#         print(domain2)
#         # df.to_csv(outfo, encoding='gb2312',header=None, index=0)  #去除表头
#         # np.savetxt()
#         #df.to_csv(domain2,encoding='gb2312')
#         df.to_csv(outfo, encoding='gb2312')
#
#         print(info, "处理完")
# filepath = 'F:\仪控可靠性\数据\确定故障'
# savepath = 'F:\仪控可靠性\数据\处理后输出'
# #deal_one(filepath,savepath)
# t = pd.DataFrame(pd.read_csv('F:\仪控可靠性\数据\程序数据_处理前\故障\_06_01_01_00_36_48_阀A常开_开总124_关总93.csv',encoding='gb2312',usecols=[0]))
# #t = pd.to_datetime(t)
# t = t.values
# print(len(t),t.shape)
# for i in range(0,len(t)-1):
#  t[i] = float(t[i])


freq = ['_06_01_01_00_36_48_阀A常开_开总124_关总93.csv', '_06_01_01_09_07_13_阀A常开_开总56_关总28.csv', '_06_01_04_01_30_34_阀A常关_开总2_关总4.csv', '_06_01_04_01_38_38_阀A自开_开总4_关总5_1.csv', '_06_01_04_01_39_16_阀A自关_开总4_关总6.csv', '_19_12_21_16_30_27_阀A常开_开总69_关总37.csv', '_19_12_21_19_03_02_阀A常开_开总72_关总40.csv', '_19_12_21_19_15_07_阀A常开_开总73_关总41.csv', '_19_12_21_19_26_04_阀A常开_开总74_关总42.csv', '_19_12_21_19_38_58_阀A常开_开总74_关总42.csv', '_19_12_21_20_19_28_阀A常开_开总75_关总45.csv', '_19_12_22_15_33_02_阀A常开_开总83_关总52.csv', '_19_12_22_15_40_47_阀A常开_开总84_关总53.csv', '_19_12_22_16_15_50_阀A常开_开总86_关总55.csv', '_19_12_22_16_59_16_阀A常开_开总87_关总56.csv', '_19_12_23_03_33_05__阀B常关_开总45_关总45.csv', '_20_03_27_00_21_25_ZF65SF-01B屏蔽式电动闸阀可靠性强化试验-7-1基准试验-8.1.4基准冷循环试验-冷态循环试验-正常驱动力冷循环带压差阀A常开_开总12_关总13_3000.csv', '_20_04_09_11_39_35_SF65ZF-02C热老化试验-255℃动作老化阀A常关_开总49_关总48_3000.csv', '_20_04_14_16_06_14__SF65ZF-02C冷循环试验_8.3.2高温（环境）动作试验-后冷循环试验-399V阀A常开_开总23_关总24_3.csv', '_20_04_14_18_36_02__SF65ZF-02C冷循环试验_8.3.3闸阀快速温度（环境）循环试验-380V-补充90°冷循环阀A常开_开总27_关总28_3.csv', '_20_04_29_10_32_57_SFZF65-02C环境温度循环+系统温度循环+振动步进_75%75%75%阀A自关_开总158_关总159_3.csv', '_20_04_29_10_57_30_SFZF65-02C环境温度循环+系统温度循环+振动步进+湿度_75%75%75%75%阀A自关_开总160_关总160_3.csv', '_20_04_29_12_53_50_SFZF65-02B环境温度循环+系统温度循环+振动步进+湿度_位置指示器测试阀A常开_开总440_关总508_3.csv']
for i in range(0,len(freq)):
    if i >=16:
        print(i,freq[i],3)
    else:
        print(i,freq[i],1)

print(np.arange(0,1.001,0.001))



fre = 1/3000
print(fre)
t = np.arange(0, 106800 / 1000, 1 / 1000)
t2 = np.linspace(0,106800 / 1000,106800)
print(len(t))
print(len(t2))