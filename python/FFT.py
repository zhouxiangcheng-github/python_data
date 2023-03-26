# -*- coding: utf-8 -*-
# @Time    : 2022/3/15 17:13
# @File    : FFT.py
# for i in tqdm(range(10)):
#    time.sleep(random.random())
# starttime = datetime.datetime.now()	# 运行时间统计
# endtime = datetime.datetime.now()	# 运行时间统计


import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import pandas as pd

'''
输入：
params为一个字典（C中称为结构体），键为't','channel_data',只需给对应的键提供值
t为时间序列
channel_data要为滤波的通道所对应的数据

输出：
result为一个字典（C中称为结构体），键为'freqs','pows'
freqs为频率
pows为对应的幅值
根据freqs和pows可以绘制频域图
'''


def myFFT(params):
    Time = params['t']  # 时间序列
    channel_data = params['channel_data']  # 对应的振动信号
    complex_array = fft.fft(channel_data)
    n = Time.size  # 样本大小
    freqs = fft.fftfreq(n, Time[1] - Time[0])  # 得到分解波的频率序列
    pows = np.abs(complex_array)  # 双边谱的幅值
    pows = pows / n  # 归一化，不然得出的频谱的幅值是分量的幅值的N/2倍
    pows = pows[freqs > 0] * 2  # 转换为单边频谱的幅值，和上式计算共同得出，效果为[pows>0]*2/N
    freqs = freqs[freqs > 0]  # 转换为单边频谱的频率,只要单边

    # 用于存放结果
    result = {}
    result['freqs'] = freqs
    result['pows'] = pows
    return result


 #data = np.loadtxt(r"F:\仪控可靠性\数据\A_current.txt")
#data = pd.read_excel('F:\仪控可靠性\数据\\test\\1.xlsx',usecols=[0])
 #data = pd.read_excel('F:\仪控可靠性\数据\\test\\2.xlsx',header=None)
 #data = pd.read_excel('F:\仪控可靠性\数据\\test\\3.csv')
#print(data)

#s2= 20**np.sin(10 *np.pi* time)
#s3= 40*np.cos(5 * np.pi*time)
# y1=plt.subplot(4,1,1)
# y2=plt.subplot(4,1,2)
# y3=plt.subplot(4,1,3)
# y1.plot(time,s1)
# y2.plot(time,s2)
# y3.plot(time,s3)


#data=s1  #自己生成pip install tensorflow==2.0
t = np.linspace(0,1,1400)
data = 5* np.sin(2*np.pi*200*t)+10* np.cos(2*np.pi*400*t) +15* np.sin(2*np.pi*600*t)



#t=np.linspace(0,1,len(data))


#channel_data = data.values[:, 0]  # 读取对应的振动数据，这里用第一个通道

channel_data = data
# channel_data=data
params = {'t': t, 'channel_data': channel_data}
result = myFFT(params)
plt.figure(1)
plt.plot(t, channel_data)
plt.figure(2)
plt.plot(result['freqs'], result['pows'])
#np.savetxt(r'C:\Users\Hi ZXC\Desktop\out_data.txt', data)
# for i in np.linspace(0,144300,1):
#     if result['freqs']>5:
#         print(result['freqs'])
pow = result['pows']
fre = result['freqs']
for i in range(len(pow)):
 if pow[i] > 3 :
  print(i,pow[i])
  print(i,fre[i])
#print(result['freqs'])
plt.show()
