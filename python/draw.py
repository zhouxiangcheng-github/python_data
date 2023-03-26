# -*- coding: utf-8 -*-
# @Time    : 2022/3/16 10:56
# @File    : draw.py
# -*- coding:utf-8 _*-

#来源于https://blog.csdn.net/gaotihong/article/details/80983937
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


#简单作图
# x=np.linspace(0,1,100)
# y=2*x
# plt.xlabel("x'")   #坐标轴标签
#
# plt.ylabel("y'")
#
# plt.plot(x,y,color='red',linewidth = 3)
# plt.show()


#figure对象
#在matplotlib中，整个图表为一个figure对象。其实对于每一个弹出的小窗口就是一个Figure对象，如何在一个代码中创建多个Figure对象？
# x=np.linspace(0,1,100)
# y1=x*x
# y2=2*x
# plt.figure(num=1,figsize=(10,5))
# plt.xlabel("x'")   #坐标轴标签
# plt.ylabel("y'")
# plt.plot(x,y1)
# plt.figure(num=2,figsize=(5,5))
# plt.plot(x,y2)
# plt.show()

#在同一张图中显示多条线
# x=np.linspace(0,1,100)
# y1=x**2
# y2=x*2
# l1,=plt.plot(x,y1,label='y1_show',color='red',linewidth=2)
# l2,=plt.plot(x,y2,label='y2_show',color='green',linewidth=2)
#
# plt.legend(loc='upper right')
# plt.show()

#画散点图

# import matplotlib.pyplot as plt
# import numpy as np
#
# n = 1024
# X = np.random.normal(0, 1, n)
# Y = np.random.normal(0, 1, n)
# T = np.arctan2(Y, X)  # for color later on #arctan2(y,x)=arcta(y/x)-Pi
# print(T)
#
# plt.scatter(X, Y, s=75, c=T, alpha=.5)
#
# plt.xlim((-1.5, 1.5))
# plt.xticks([])  # ignore xticks=不显示横坐标
# plt.ylim((-1.5, 1.5))
# # arrange=np.linspace(0,1,1024)
# # plt.yticks(X, arrange)
# plt.yticks([])# ignore yticks
# plt.show()


# data_source = pd.read_excel('F:/南师2020作业/人工智能/datas.xlsx')
# # 函数plot()尝试根据数字绘制出有意义的图形
# print(data_source['datas'])
# plt.plot(data_source['datas'])


# data_source = np.loadtxt('F:\仪控可靠性\数据\A_current.txt')
# # 函数plot()尝试根据数字绘制出有意义的图形
# print(data_source[:,1])
# X=np.linspace(1,144300,144300)
# Y=data_source[:,1]
# plt.plot(data_source[:,0],data_source[:,1])
# plt.show()


#柱状图
import matplotlib.pyplot as plt

name = ['part1_00', 'part1_01', 'part1_02', 'part1_03', 'part2_00', 'part2_01', 'part2_02', 'part2_03', 'part2_04',
        'part2_05', 'part2_06']
num = [79855, 39995, 39997, 1734, 99778, 95630, 99167, 13781, 96339, 57913, 14885]

plt.bar(name, num)
plt.xlabel('name of file')
plt.ylabel("num-of-file")
plt.title("num-of-each-file")
for a, b in zip(name, num):
    plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=5)#（x,y,s)分别代表x轴y轴和数据宽度控制
plt.xticks(fontsize=2)   #字号
plt.show()

filters,epochs = [4,5,6,7,8],[5,6,7,8,9,10]
acc_filters,acc_epochs = [0.9846,0.9851,0.9866,0.987,0.9874],[0.9793,0.9803,0.9824,0.9851,0.9855,0.9867]
show1 = plt.subplot(2,1,1)
plt.xlabel("filters'")   #坐标轴标签
#
plt.ylabel("acc_filters'")
show1.plot(filters,acc_filters)

show2 = plt.subplot(2,1,2)

plt.xlabel("acc_epochs'")   #坐标轴标签
#
plt.ylabel("acc_epochs'")
show2.plot(epochs,acc_epochs)


plt.show()



