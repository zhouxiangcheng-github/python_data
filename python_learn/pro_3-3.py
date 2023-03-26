# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 16:48
# @File    : pro_3-3.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# #data_source = pd.read_excel('F:\\仪控可靠性\\查找出来的数据\\故障\\_20_03_27_00_54_54_ZF65SF-01B屏蔽式电动闸阀可靠性强化试验-7-1基准试验-8.1.4基准冷循环试验-冷态循环试验-243V开阀故障恢复阀A常开_开总15_关总16.xlsx')
# data_source=np.loadtxt("0_03_27_13_12_59_01B屏蔽式电动闸阀可靠性强化试验-7-1基准试验-8.1.4基准试验-冷循环-最小驱动力13.7MPA阀A常开_开总19_关总20.csv")
# # 函数plot()尝试根据数字绘制出有意义的图形
# print(data_source['A_current'])
# plt.figure(figsize=(20,20))
# plt.plot(data_source['A_current'])
# plt.show()


from PyEMD import EEMD, Visualisation
