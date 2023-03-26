# -*- coding: utf-8 -*-
# @Time    : 2022/3/10 16:34
# @File    : EMD.py
# for i in tqdm(range(10)):
#    time.sleep(random.random())
import numpy as np
from PyEMD import EEMD, Visualisation
import matplotlib.pyplot as plt
import numpy.fft as fft

from tqdm import tqdm
import time, random

for i in tqdm(range(10)):
    time.sleep(random.random())

#傅里叶变换
def data_fft(Time, channel_data):
    complex_array = fft.fft(channel_data)
    n = Time.size    #样本大小
    freqs = fft.fftfreq(n, Time[1] - Time[0])    # 得到分解波的频率序列
    pows = np.abs(complex_array)    #双边谱的幅值
    pows = pows/n    #归一化
    pows = pows[freqs > 0]*2   #转换为单边频谱的幅值
    freqs = freqs[freqs > 0]    #转换为单边频谱的频率
    return freqs, pows

#data=np.loadtxt(r'F:\WeChat Files\WeChat Files\wxid_mkhi8s24fir822\FileStorage\File\2022-03\2.txt')

# data = np.loadtxt(r'F:\仪控可靠性\数据\A_current.txt')   #读取文件
# data=np.loadtxt(r'F:\仪控可靠性\数据\429_10_57.txt')  #

t= np.linspace(0, 1, 1600)
s1= 20*np.sin(2 *np.pi*100* t)
s2= 2**np.sin(2 *np.pi*400* t)
s3= 1*np.cos(2 * np.pi*800*t)
y1=plt.subplot(4,1,1)
y2=plt.subplot(4,1,2)
y3=plt.subplot(4,1,3)
y1.plot(t,s1)
y2.plot(t,s2)
y3.plot(t,s3)


data=s1+s2+s3  #自己生成pip install tensorflow==2.0

y4=plt.subplot(4,1,4)
y4.plot(t,data)
plt.show()

# data=np.loadtxt(r'F:\仪控可靠性\1\2.txt',encoding='gb2312')#ICA数据
# data = np.loadtxt('F:\python-project/test_data/test4.txt')
print(len(data))
#data=np.loadtxt(r'F:\python-project/test_data/test4.txt')
#t = data[:,0]    #读取时间序列
# print(t)
# data = np.loadtxt(r'F:\仪控可靠性\1\emd.txt')
# data = data.T


print(data.shape)
S = data  #读取对应的振动数据，这里用第一个通道


# Extract imfs and residue
# In case of EMD
eemd = EEMD()
eemd.eemd(S, t)
imfs, res = eemd.get_imfs_and_residue()
np.savetxt('F:\仪控可靠性\\1\emd.txt',imfs)
out_data= np.loadtxt('F:\仪控可靠性\\1\emd.txt')
print(out_data.shape)

"""
vis = Visualisation()
#vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
#vis.plot_instant_freq(t, imfs=imfs)
vis.plot_imfs(imfs=imfs[:10], residue=res, t=t, include_residue=True)
vis.plot_instant_freq(t, imfs=imfs[:10])
vis.show()
"""
rownum = len(imfs)+1

plt.figure(figsize=(10,15))     #改变图形大小
plt.subplots_adjust(hspace=0.5, wspace=0.1)     #设置了子图之间的纵、横两方向上的间隙

"""
这两行用于显示中文字体
"""
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


"""
subplot()第一个参数：行数，第二个参数：列数，第三个参数：第几个图
"""
for i in range(len(imfs)):
    freqs, pows = data_fft(t, imfs[i])
    if i==4:
     np.savetxt(r'F:\仪控可靠性\数据\test\fft_out.txt',freqs)
     np.savetxt(r'F:\仪控可靠性\数据\test\fft_out 2.txt', pows)
    plt.subplot(rownum, 2, i*2+1)
    plt.plot(t, imfs[i])     #画时域
    #plt.xticks(())    #不显示坐标刻度
    plt.ylabel("IMF"+str(i+1))
    plt.xlabel("时间/s")
    #plt.ylabel("IMF"+str(i)+"\n\nm/s^2")    #显示纵坐标单位
    plt.subplot(rownum, 2, i*2+2)
    plt.plot(freqs, pows)   #画频域
    plt.xlabel("频率/Hz")
    #plt.xticks(())    #不显示坐标刻度

"""
plt.subplot(rownum, 2, rownum*2-1)
plt.plot(t, res)
plt.xlabel("时间/s")
plt.ylabel("R")
#plt.ylabel("R\n\nm/s^2")    #显示纵坐标单位

freqs, pows = data_fft(t, res)
plt.subplot(rownum, 2, rownum*2)
plt.plot(freqs, pows)
plt.xlabel("频率/Hz")
#plt.ylabel("幅值")
"""
plt.show()