#!/F/ZXC/python-project/env python 
# -*- coding:utf-8 _*-
""" 
@file: ICA.py 
@time: 2022/01/11
"""
import numpy as np
import matplotlib.pyplot as plt
import a9mysubplot
from sklearn.decomposition import FastICA

#以下程序调用ICA，输入观察信号，输出为解混合信号
function
#-------------去均值------------
[M,T] = size(X); #获取输入矩阵的行/列数，行数为观测数据的个数，列数为采样点点数
average= mean(X')';  #按行取均值
