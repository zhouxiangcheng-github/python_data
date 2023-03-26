# -*- coding:utf-8 _*-
"""
@file: practice15.py
@time: 2022/01/15
"""
from sklearn import svm
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
# data = np.loadtxt(r'F:\量子计算\支撑不足\139VB-N.txt')   #读取文件
# # t = data[:,0]    #读取时间序列
# # data_1=data[:,1]
# # params = {'t':t, 'channel_data':data_1}
# # print(params)
# # clf = svm.SVC()
#
# # print(data)
# #
# # data_a=(np.array(np.arange(1,(len(data[0])-1)*len(data)+1,1)).reshape(len(data),len(data[0])-1))
# #
# #
# # print(data_a)
# # data_a=np.array(data_a, dtype=np.float32)
# # for i in np.arange(1,len(data[0]),1):
# #     data_a[:,i-1] = data[:,i]
# # print(data_a)
#
# from sklearn import svm
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import sklearn
# from sklearn.model_selection import train_test_split
#
#
# # define converts(字典)
# def Iris_label(s):
#     it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
#     return it[s]
#
#
# # 1.读取数据集
# path = 'F:\python-project\iris.data'
# #print(path)
# data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: Iris_label})
# #print(data)
# # converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
# #print(data.shape)
#
# # 2.划分数据与标签
# # x为数据，y为标签:表示从第5列开始切，axis=1表示纵向开始切
# #将data分为一个n维4列的数组X，表示数据，和一个n维1列的数组Y，表示标签
# x, y = np.split(data, indices_or_sections=(4,), axis=1)
# x = x[:, 0:2]      #表示取第1列到第二列=取前两列元素
# train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6,
#                                                                   test_size=0.4)  # sklearn.model_selection.
# #print(train_data[:,0])
# # print(len(x[:,0]))
# # print(len(train_data[:,0]))
# # print(train_data.shape)
#
# # 3.训练svm分类器
# classifier = svm.SVC(C=2, kernel='rbf', gamma=10, probability=True, decision_function_shape='ovr')  # ovr:一对多策略
#
# classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
# #print(train_label.ravel())
# # 4.计算svc分类器的准确率
# print(train_data[0])
# # print(train_data[0].reshape(1,-1))
# print(test_label.reshape(1,60))
# print("测试：",classifier.predict(test_data).reshape(1,60))
# prob=classifier.predict_proba(test_data[0].reshape(1,-1))
# print(prob)
# #print("prob:",np.max(classifier.predict_proba(test_data[0])))
# print("训练集：", classifier.score(train_data, train_label))
# print("测试集：", classifier.score(test_data, test_label))
#
# # 也可直接调用accuracy_score方法计算准确率
# from sklearn.metrics import accuracy_score
#
# tra_label = classifier.predict(train_data)  # 训练集的预测标签
# tes_label = classifier.predict(test_data)  # 测试集的预测标签
# print("训练集：", accuracy_score(train_label, tra_label))
# print("测试集：", accuracy_score(test_label, tes_label))
#
# # 查看决策函数
# print('train_decision_function:\n', classifier.decision_function(train_data))  # (90,3)
# print('predict_result:\n', classifier.predict(train_data))
#
# # 5.绘制图形
# # 确定坐标轴范围
# x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围
# x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围
# x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
# grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# # 指定默认字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# # 设置颜色
# cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
# cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])
#
# grid_hat = classifier.predict(grid_test)  # 预测分类值
# grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
#
# plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
# plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=30, cmap=cm_dark)  # 样本
# plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label[:, 0], s=30, edgecolors='k', zorder=2,
#             cmap=cm_dark)  # 圈中测试集样本点
# plt.xlabel('花萼长度', fontsize=13)
# plt.ylabel('花萼宽度', fontsize=13)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title('鸢尾花SVM二特征分类')
# plt.show()


# -*- coding: utf-8 -*-
# """
# @time: 2022/03/25
# """
# import numpy as np
# import math
# import os
# import re
# import numpy.fft as fft
# import pandas as pd
# # 傅里叶变换
# def data_fft(Time, channel_data):
#     complex_array = fft.fft(channel_data)
#     n = Time.size  # 样本大小
#     freqs = fft.fftfreq(n, Time[1] - Time[0])  # 得到分解波的频率序列
#     pows = np.abs(complex_array)  # 双边谱的幅值
#     pows = pows / n  # 归一化
#     pows = pows[freqs > 0] * 2  # 转换为单边频谱的幅值
#     freqs = freqs[freqs > 0]  # 转换为单边频谱的频率
#     return freqs, pows
# # 特征计算
# def myFeature_Cal(t, channel_data):
#     '''
#     时域特征
#     顺序：
#     Xp,Xpp,Xmv,Xrms,Xsar,Xma,SD,SKE,UR,CRE,BY,IMP,CLE,L,Ck,Xasl
#     Xp（峰值），Xpp（峰峰值）
#     Xmv(均值)，Xrms（有效值）
#     Xsar（方根幅值）,Xma（平均幅值）
#     SD（标准差），SKE（偏度指标）
#     KUR（峭度指标），CRE（峰值指标）
#     BY（波形指标），IMP（脉冲指标）
#     CLE（裕度指标），L（峰值到均方根的距离）
#     Ck（峰态系数）,Xasl（平均信号电平）
#     '''
#     # t和channel_data构成时域信号
#     t = np.array(t)
#     channel_data = np.array(channel_data)
#     n = len(channel_data)
#
#     # Xp（峰值）的计算
#     Xp = max(np.absolute(channel_data))
#
#     # Xpp（峰峰值）的计算
#     Xpp = max(channel_data) - min(channel_data)
#
#     # Xmv(均值)的计算
#     Xmv = np.mean(channel_data)
#
#     # Xrms（有效值）的计算
#     Xrms_fz = 0
#     for i in range(n):
#         Xrms_fz += math.pow(channel_data[i], 2)  # Xrms分子的求解，fz代表“分子”，之后的名称命名同理
#     Xrms = math.sqrt(Xrms_fz / n)
#
#     # Xsar（方根幅值）的计算
#     Xsar_fz = 0
#     for i in range(n):
#         Xsar_fz += math.sqrt(abs(channel_data[i]))
#     Xsar = math.pow(Xsar_fz / n, 2)
#
#     # Xma（平均幅值）的计算
#     Xma_fz = 0
#     for i in range(n):
#         Xma_fz += abs(channel_data[i])
#     Xma = Xma_fz / n
#     # SD（标准差）的计算
#     SD_fz = 0
#     for i in range(n):
#         SD_fz += math.pow(channel_data[i] - Xmv, 2)
#     SD = math.sqrt(SD_fz / (n - 1))
#
#     # SKE（偏度指标）的计算
#     SKE_fz = 0
#     for i in range(n):
#         SKE_fz += math.pow(channel_data[i] - Xmv, 3)
#     SKE = SKE_fz / n / math.pow(Xrms, 3)
#
#     # KUR（峭度指标）的计算
#     KUR_fz = 0
#     for i in range(n):
#         KUR_fz += math.pow(channel_data[i] - Xmv, 4)
#     KUR = KUR_fz / n / math.pow(Xrms, 4)
#
#     # CRE（峰值指标）的计算
#     CRE = Xp / Xrms
#
#     # BY（波形指标）的计算
#     BY = Xrms / Xma
#
#     # IMP（脉冲指标）的计算
#     IMP = Xp / Xma
#
#     # CLE（裕度指标）的计算
#     CLE = Xp / Xsar
#
#     # L（峰值到均方根的距离）的计算
#     L = max(channel_data) - Xrms
#
#     # Ck（峰态系数）的计算
#     Ck_fz = 0
#     for i in range(n):
#         Ck_fz += math.pow((channel_data[i] - Xmv) / SD, 4)
#     Ck = Ck_fz / n - 3
#
#     # Xasl（平均信号电平）的计算
#     Xasl = 20 * np.log10(Xma)
#
#     '''
#     频域特征
#     顺序：
#     MSF,VF,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13
#     FC（和F5一样，保留F5的结果，舍去FC），MSF（均方频率）
#     VF（频率方差），F1（均值频率）
#     F2（标准偏差频率），F3（频谱偏度）
#     F4（频谱峭度），F5（频谱一阶重心）
#     F6（频谱二阶重心），F7（频谱二阶矩）
#     F8，F9
#     F10，F11
#     F12，F13
#     '''
#     # freqs和pows构成频域信号
#     freqs, pows = data_fft(t, channel_data)
#     freqs = np.array(freqs)
#     pows = np.array(pows)
#     N = len(pows)
#
#     # FC（重心频率）的计算
#     FC_fz = 0
#     for i in range(N):
#         FC_fz += freqs[i] * pows[i]
#     FC = FC_fz / sum(pows)
#
#     # MSF（均方频率）的计算
#     MSF_fz = 0
#     for i in range(N):
#         MSF_fz += math.pow(freqs[i], 2) * pows[i]
#     MSF = MSF_fz / sum(pows)
#
#     # VF（频率方差）的计算
#     VF_fz = 0
#     for i in range(N):
#         VF_fz += math.pow(freqs[i] - FC, 2) * pows[i]
#     VF = VF_fz / sum(pows)
#
#     # F1（均值频率）的计算
#     F1 = sum(pows) / N
#
#     # F2（标准偏差频率）的计算
#     F2_fz = 0
#     for i in range(N):
#         F2_fz += abs(pows[i] - F1)
#     F2 = math.sqrt(F2_fz / (N - 1))
#
#     # F3（频谱偏度）的计算
#     F3_fz = 0
#     for i in range(N):
#         F3_fz += (pows[i] - F1)
#     F3 = F3_fz / N / math.pow(F2, 3 / 2)
#
#     # F4（频谱峭度）的计算
#     F4_fz = 0
#     for i in range(N):
#         F4_fz += math.pow((pows[i] - F1), 4)
#     F4 = F4_fz / N / math.pow(F2, 2)
#
#     # F5（频谱一阶重心）的计算
#     F5 = FC
#
#     # F6（频谱二阶重心）的计算
#     F6_fz = 0
#     for i in range(N):
#         F6_fz += math.pow(freqs[i] - F5, 2) * pows[i]
#     F6 = math.sqrt(F6_fz / N)
#
#     # F7（频谱二阶矩）的计算
#     F7 = math.sqrt(MSF)
#
#     # F8的计算
#     F8_fz = 0
#     for i in range(N):
#         F8_fz += math.pow(freqs[i], 4) * pows[i]
#     F8 = math.sqrt(F8_fz / MSF_fz)
#
#     # F9的计算
#     F9 = MSF_fz / math.sqrt(F8_fz * sum(pows))
#
#     # F10的计算
#     F10 = F6 / F5
#
#     # F11（频谱二阶重心）的计算
#     F11_fz = 0
#     for i in range(N):
#         F11_fz += math.pow(freqs[i] - F5, 3) * pows[i]
#     F11 = F11_fz / N / math.pow(F6, 3)
#
#     # 计算F12
#     F12_fz = 0
#     for i in range(N):
#         F12_fz += math.pow(freqs[i] - F5, 4) * pows[i]
#     F12 = F12_fz / N / math.pow(F6, 4)
#
#     # 计算F13
#     F13_fz = 0
#     for i in range(N):
#         F13_fz += math.pow(abs(freqs[i] - F5), 1 / 2) * pows[i]
#     F13 = F13_fz / N / math.pow(F6, 1 / 2)
#
#     """
#     时域特征（16个）：
#     Xp（峰值），Xpp（峰峰值）
#     Xmv(均值)，Xrms（有效值）
#     Xsar（方根幅值）,Xma（平均幅值）
#     SD（标准差），SKE（偏度指标）
#     KUR（峭度指标），CRE（峰值指标）
#     BY（波形指标），IMP（脉冲指标）
#     CLE（裕度指标），L（峰值到均方根的距离）
#     Ck（峰态系数）,Xasl（平均信号电平）
#
#     频域特征（15个）：
#                  ，MSF（均方频率）
#     VF（频率方差），F1（均值频率）
#     F2（标准偏差频率），F3（频谱偏度）
#     F4（频谱峭度），F5（频谱一阶重心）
#     F6（频谱二阶重心），F7（频谱二阶矩）
#     F8，F9
#     F10，F11
#     F12，F13
#     """
#     testFeature = [Xp, Xpp, Xmv, Xrms, Xsar, Xma, SD, SKE, KUR, CRE, BY, IMP, CLE, L, Ck, Xasl,
#                    MSF, VF, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13]
#     return testFeature
#
#
#
# filePaths1 = r"F:\仪控可靠性\数据\程序数据_处理前\故障"
#
# filePaths2 = r"F:\仪控可靠性\数据\程序数据_处理前\正常"
# filePath_alls = [filePaths1, filePaths2]
#
# save_path = r"F:\仪控可靠性\数据\程序数据_处理后\feature.txt"
# lables = [0, 1]  # 用于打标签，标签0为故障，1正常
# num = 0
# all_val = []
# all_val = np.array(all_val).reshape(-1, 32)
# freq = []
# #路径下是文件夹，文件夹下是单个txt源文件
# for filePath_all, lable in zip(filePath_alls, lables):  # 分别读取故障，正常
#     print("路径：", filePath_all)
#     print("标签：", lable)
#     filenames = os.listdir(filePath_all)  # 找出文件夹里面的文件列表，每个文件对应一个数据，.csv格式。以列表格式保存文件名字
#     print(filenames)
#     for file,i in zip(filenames,range(0,len(filenames))):  # 遍历最后一层文件夹里的所有文件，file就是具体的文件
#         child = re.findall(r"(.+?).csv", file)  # 正则的方式读取文件名，去扩展名
#         if len(child) >= 0:  # 去掉没用的系统文件
#             newfile = ''
#             needdate = child  #### 这个就是所要的文件名
#         domain1 = os.path.abspath(filePath_all)  # 待处理文件位置
#         info = os.path.join(domain1, file)  # 拼接出待处理文件名字
#         data = pd.DataFrame(pd.read_csv(info,encoding='gb2312',low_memory=False,dtype={'A阀电流A相工程值':np.float64}, usecols=['A阀电流A相工程值'])).values  # 选取固定列的值生成新表，选取第10列作为新表//有的文件夹选择第11列
#         data = data.reshape(len(data), 1)
#         if (i>=16 and lable==0) or (i>=12 and lable==1):
#             freq=3000
#         else:
#             freq=1000
#         #t = np.linspace(0, 1, len(data)).reshape(len(data), 1)
#         #t = np.arange(0,len(data)/freq,1/freq)
#         t = np.linspace(0,len(data)/freq,len(data))
#         print("通道：", i)
#         print(freq,file,len(t),len(data))
#         channel_data = data[:, 0]  # 读取对应的振动数据，这里用第一个通道
#
#         val = []
#         testFeature = myFeature_Cal(t, channel_data)  # 计算特征
#         testFeature.append(int(lable))
#         #print(testFeature)
#         val = np.array(testFeature).reshape(-1, 32)  # 将val转换为array
#
#         #print("val", val)
#         # 每个通道计算结束，进行训练集堆叠
#         if len(all_val) == 0:
#             all_val = val
#         else:
#             all_val = np.r_[all_val, val]
#         num += 1
#         #print("训练集：", all_val)
#         print("训练集大小：", np.array(all_val).shape)
# #print(all_val)
# print("行数：", num)
# np.savetxt(save_path, all_val)



lie = np.linspace(0,11,12).reshape(3,4)
print(lie)
lie = np.array(lie)[::-1]
print(type(lie),lie)

num = [1,4,8,2,3,7,9,6]
num = np.array(num)
print(num.argsort())