# -*- coding:utf-8 _*-
""" 
@file: 01-13.py 
@time: 2022/01/13
"""
# FastICA_learn by Leo
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 应用方法
'''
函数名称：Jagged
函数功能：锯齿波生成器
输入参数：t              时刻
        period          生成波形的周期
返回参数：jagged_value   与时刻t对应的锯齿波的值
作者：Leo Ma
时间：2020.01.04
'''


def Jagged(t, period=4):
    jagged_value = 0.5 * (t - math.floor(t / period) * period)  # math.floor(x)返回x的下舍整数
    return jagged_value


'''
函数名称：create_data
函数功能：模拟生成原始数据
输入参数：None
返回参数：T          时间变量
        S           源信号
        D           观测到的混合信号
        
作者：Leo Ma
时间：2020.01.04
'''


def create_data():
    # data number
    m = 500
    # 生成时间变量
    T = [0.1 * xi for xi in range(m)]

    # 生成源信号
    S = np.array([[math.sin(xi) for xi in T], [Jagged(xi) for xi in T]], np.float32)
    # 定义混合矩阵
    A = np.array([[0.8, 0.2], [-0.3, -0.7]], np.float32)
    # 生成观测到的混合信号
    D = np.dot(A, S)
    return T, S, D


'''
函数名称：load_wav_data
函数功能：从文件中加载wav数据
输入参数：file_name  要读取的文件名
返回参数：T          时间变量
        S           源信号
        params      wav数据参数
作者：Leo Ma
时间：2020.01.04
'''


def load_wav_data(file_name):
    import wave
    f = wave.open(file_name, 'rb')
    # 获取音频的基本参数
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    # 读取字符串格式的音频
    strData = f.readframes(nframes)
    # 关闭文件
    f.close()
    # 将字符串格式的音频转化为int类型
    waveData = np.fromstring(strData, dtype=np.int16)
    # 将wave幅值归一化
    S = waveData / (max(abs(waveData)))
    T = np.arange(0, nframes) / framerate

    return T, S, params


'''
函数名称：save_wav_data
函数功能：将wav数据保存到文件中
输入参数：file_name  要保存的文件名
          params    保存参数
          S          wav信号
返回参数：None
作者：Leo Ma
时间：2020.01.04
'''


def save_wav_data(file_name, params, S):
    import wave
    import struct
    outwave = wave.open(file_name, 'wb')
    # 设置参数
    outwave.setparams((params))
    # 将信号幅值归一化
    S = S / (max(abs(S)))
    # 逐帧写入文件
    for v in S:
        outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))  # S:16位，-32767~32767，注意不要溢出
    outwave.close()


'''
函数名称：load_create_data
函数功能：从文件中加载wav数据,并对其进行混合
输入参数：None
返回参数：T          时间变量
        S           源信号
        D           观测到的混合信号
        params     wav数据的参数，两端音频参数是一样的
作者：Leo Ma
时间：2020.01.04
'''


def load_create_data():
    # 两段音频截取的一样长
   # T1, S1, params1 = load_wav_data('./voice/sound1.wav')
    T1, S1, params1 = load_wav_data('f:/python-project/wav/1.wav')
    T2, S2, params2 = load_wav_data('f:/python-project/wav/1.wav')

    if np.shape(T1)[0] > np.shape(T2)[0]:
        T = T1
    else:
        T = T2

    # 将大小为(1,m)的S1、S2合并成S，S的大小为(2,m)
    # 其中S1[np.newaxis,:]将(m,)加入一个新轴，变成(1,m)
    # np.vstack((a,b))将a和b两个矩阵在列方向上进行合并
    # 另外,np.hstack((a,b))将a和b两个矩阵在行方向上进行合并
    S = np.vstack((S1[np.newaxis, :], S2[np.newaxis, :]))

    # 定义混合矩阵
    A = np.array([[0.8, 0.2], [-0.3, -0.7]], np.float32)
    # 生成观测到的混合信号
    D = np.dot(A, S)

    return T, S, D, params1


'''
函数名称：show_data
函数功能：画出数据图
输入参数：None
返回参数：None
作者：Leo Ma
时间：2020.01.04
'''


def show_data(T, S):
    plt.plot(T, S[0, :], color='r', marker="*")
    plt.show()
    plt.plot(T, S[1, :], color='b', marker="o")
    plt.show()


# 主函数入口
def main():
    '''第一种创建数据方法：从文件中读取两端语音信号'''

    # 生成数据，T是时间变量，S是源信号，D是观测到的混合信号
    T, S, D, params = load_create_data()
    # 将两段混合信号保存在硬盘上
    save_wav_data('f:/python-project/wav/s1.wav', params, D[0, :])
    save_wav_data('f:/python-project/wav/s2.wav', params, D[1, :])

    '''第二种创建数据方法：用两个波形生成器生成两个信号'''
    # T, S, D = create_data()

    # sklearn 中ICA的使用方法
    ica = FastICA(n_components=2)  # 独立成分为2个
    DT = np.transpose(D)  # 将混合信号矩阵转换为shape=[m,n],这里n=2
    SrT = ica.fit_transform(DT)  # SrT为解混后的2个独立成分，shape=[m,n]
    Sr = np.transpose(SrT)  # 将解混后的信号矩阵转换为shape=[n,m],这里n=2

    print('times:')
    print(ica.n_iter_)  # 算法迭代次数

    # 将两段由FastICA算法重构的信号保存在硬盘上
    save_wav_data('f:/python-project/wav/r1.wav', params, Sr[0, :])
    save_wav_data('f:/python-project/wav/r2.wav', params, Sr[1, :])

    # 画出数据图像
    print("T,D:")
    show_data(T, D)
    print("T,S:")
    show_data(T, S)
    print("T,Sr：")
    show_data(T, Sr)


if __name__ == "__main__":
    main()
