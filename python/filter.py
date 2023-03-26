# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:05
# @File    : filter.py
# for i in tqdm(range(10)):
#    time.sleep(random.random())
# starttime = datetime.datetime.now()	# 运行时间统计
# endtime = datetime.datetime.now()	# 运行时间统计
import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
import numpy.fft as fft
'''
1、限幅滤波法（又称程序判断滤波法）  A、方法：  根据经验判断，确定两次采样允许的最大偏差值（设为A）  每次检测到新值时判断：  如果本次值与上次值之差<=A,则本次值有效  如果本次值与上次值之差>A,则本次值无效,放弃本次值,用上次值代替本次值  B、优点：  能有效克服因偶然因素引起的脉冲干扰  C、缺点  无法抑制那种周期性的干扰  平滑度差  
2、中位值滤波法  A、方法：  连续采样N次（N取奇数）  把N次采样值按大小排列  取中间值为本次有效值  B、优点：  能有效克服因偶然因素引起的波动干扰  对温度、液位的变化缓慢的被测参数有良好的滤波效果  C、缺点：  对流量、速度等快速变化的参数不宜  
3、算术平均滤波法  A、方法：  连续取N个采样值进行算术平均运算  N值较大时：信号平滑度较高，但灵敏度较低  N值较小时：信号平滑度较低，但灵敏度较高  N值的选取：一般流量，N=12；压力：N=4  B、优点：  适用于对一般具有随机干扰的信号进行滤波  这样信号的特点是有一个平均值，信号在某一数值范围附近上下波动  C、缺点：  对于测量速度较慢或要求数据计算速度较快的实时控制不适用  比较浪费RAM  

4、递推平均滤波法（又称滑动平均滤波法）  A、方法：  把连续取N个采样值看成一个队列  队列的长度固定为N  每次采样到一个新数据放入队尾,并扔掉原来队首的一次数据.(先进先出原则)  把队列中的N个数据进行算术平均运算,就可获得新的滤波结果  N值的选取：流量，N=12；压力：N=4；液面，N=4~12；温度，N=1~4  B、优点：  对周期性干扰有良好的抑制作用，平滑度高  适用于高频振荡的系统  C、缺点：  灵敏度低  对偶然出现的脉冲性干扰的抑制作用较差  不易消除由于脉冲干扰所引起的采样值偏差  不适用于脉冲干扰比较严重的场合  比较浪费RAM  

5、中位值平均滤波法（又称防脉冲干扰平均滤波法）  A、方法：  相当于“中位值滤波法”+“算术平均滤波法”  连续采样N个数据，去掉一个最大值和一个最小值  然后计算N-2个数据的算术平均值  N值的选取：3~14  B、优点：  融合了两种滤波法的优点  对于偶然出现的脉冲性干扰，可消除由于脉冲干扰所引起的采样值偏差  C、缺点：  测量速度较慢，和算术平均滤波法一样  比较浪费RAM  

6、限幅平均滤波法  A、方法：  相当于“限幅滤波法”+“递推平均滤波法”  每次采样到的新数据先进行限幅处理，  再送入队列进行递推平均滤波处理  B、优点：  融合了两种滤波法的优点  对于偶然出现的脉冲性干扰，可消除由于脉冲干扰所引起的采样值偏差  C、缺点：  比较浪费RAM  

7、一阶滞后滤波法  A、方法：  取a=0~1  本次滤波结果=（1-a）*本次采样值+a*上次滤波结果  B、优点：  对周期性干扰具有良好的抑制作用  适用于波动频率较高的场合  C、缺点：  相位滞后，灵敏度低  滞后程度取决于a值大小  不能消除滤波频率高于采样频率的1/2的干扰信号  

8、加权递推平均滤波法  A、方法：  是对递推平均滤波法的改进，即不同时刻的数据加以不同的权  通常是，越接近现时刻的数据，权取得越大。  给予新采样值的权系数越大，则灵敏度越高，但信号平滑度越低  B、优点：  适用于有较大纯滞后时间常数的对象  和采样周期较短的系统  C、缺点：  对于纯滞后时间常数较小，采样周期较长，变化缓慢的信号  不能迅速反应系统当前所受干扰的严重程度，滤波效果差 

9、消抖滤波法  A、方法：  设置一个滤波计数器  将每次采样值与当前有效值比较：  如果采样值＝当前有效值，则计数器清零  如果采样值<>当前有效值，则计数器+1，并判断计数器是否>=上限N(溢出)  如果计数器溢出,则将本次值替换当前有效值,并清计数器  B、优点：  对于变化缓慢的被测参数有较好的滤波效果,  可避免在临界值附近控制器的反复开/关跳动或显示器上数值抖动  C、缺点：  对于快速变化的参数不宜  如果在计数器溢出的那一次采样到的值恰好是干扰值,则会将干扰值当作有效值导  入系统  

10、限幅消抖滤波法  A、方法：  相当于“限幅滤波法”+“消抖滤波法”  先限幅,后消抖  B、优点：  继承了“限幅”和“消抖”的优点  改进了“消抖滤波法”中的某些缺陷,避免将干扰值导入系统  C、缺点：  对于快速变化的参数不宜
————————————————
版权声明：本文为CSDN博主「白首太玄经丶」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/kengmila9393/article/details/81455165
'''
'''
算术平均滤波法
方法：根据经验判断，确定两次采样允许的最大偏差值（设为A）
每次检测到新值时判断： 如果本次值与上次值之差<=A,则本次值有效  如果本次值与上次值之差>A,则本次值无效,放弃本次值,用上次值代替本次值
优点：能有效克服因偶然因素引起的脉冲干扰  
缺点：无法抑制那种周期性的干扰 ,平滑度差  

'''
def ArithmeticAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        mean.append(tmp.mean())
    return mean


'''
递推平均滤波法
方法：  把连续取N个采样值看成一个队列  队列的长度固定为N  每次采样到一个新数据放入队尾,并扔掉原来队首的一次数据.(先进先出原则)  把队列中的N个数据进行算术平均运算,就可获得新的滤波结果  
N值的选取：流量，N=12；压力：N=4；液面，N=4~12；温度，N=1~4   
优点：对周期性干扰有良好的抑制作用，平滑度高  适用于高频振荡的系统 
缺点：灵敏度低  对偶然出现的脉冲性干扰的抑制作用较差  不易消除由于脉冲干扰所引起的采样值偏差  不适用于脉冲干扰比较严重的场合  比较浪费RAM  
'''


def SlidingAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    tmpmean = inputs[0].mean()
    mean = []
    for tmp in inputs:
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


'''
中位值平均滤波法
'''


def MedianAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        tmp = np.delete(tmp, np.where(tmp == tmp.max())[0], axis=0)
        tmp = np.delete(tmp, np.where(tmp == tmp.min())[0], axis=0)
        mean.append(tmp.mean())
    return mean


'''
限幅平均滤波法
Amplitude:	限制最大振幅
'''


def AmplitudeLimitingAverage(inputs, per, Amplitude):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    tmpmean = inputs[0].mean()
    tmpnum = inputs[0][0]  # 上一次限幅后结果
    for tmp in inputs:
        for index, newtmp in enumerate(tmp):
            if np.abs(tmpnum - newtmp) > Amplitude:
                tmp[index] = tmpnum
            tmpnum = newtmp
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


'''
一阶滞后滤波法
a:			滞后程度决定因子，0~1
'''


def FirstOrderLag(inputs, a):
    tmpnum = inputs[0]  # 上一次滤波结果
    for index, tmp in enumerate(inputs):
        inputs[index] = (1 - a) * tmp + a * tmpnum
        tmpnum = tmp
    return inputs


'''
加权递推平均滤波法
'''


def WeightBackstepAverage(inputs, per):
    weight = np.array(range(1, np.shape(inputs)[0] + 1))  # 权值列表
    weight = weight / weight.sum()

    for index, tmp in enumerate(inputs):
        inputs[index] = inputs[index] * weight[index]
    return inputs


'''
消抖滤波法
N:			消抖上限
'''


def ShakeOff(inputs, N):
    usenum = inputs[0]  # 有效值
    i = 0  # 标记计数器
    for index, tmp in enumerate(inputs):
        if tmp != usenum:
            i = i + 1
            if i >= N:
                i = 0
                inputs[index] = usenum
    return inputs


'''
限幅消抖滤波法
Amplitude:	限制最大振幅
N:			消抖上限
'''


def AmplitudeLimitingShakeOff(inputs, Amplitude, N):
    # print(inputs)
    tmpnum = inputs[0]
    for index, newtmp in enumerate(inputs):
        if np.abs(tmpnum - newtmp) > Amplitude:
            inputs[index] = tmpnum
        tmpnum = newtmp
    # print(inputs)
    usenum = inputs[0]
    i = 0
    for index2, tmp2 in enumerate(inputs):
        if tmp2 != usenum:
            i = i + 1
            if i >= N:
                i = 0
                inputs[index2] = usenum
    # print(inputs)
    return inputs
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

T = np.arange(0, 0.5, 1 / 4410.0)
T = np.linspace(0,1,69900)
num = signal.chirp(T, f0=10, t1=0.5, f1=1000.0) #表示时间是从0-0.5，频率是从10HZ-1000HZ的一个
num = 100* np.sin(2*np.pi*4*T)+20* np.cos(2*np.pi*400*T) +10* np.sin(2*np.pi*600*T)
num= np.loadtxt(r'F:\仪控可靠性\1\2.txt')
print(len(num))
plt.figure(1)
pl.subplot(2, 1, 1)
pl.plot(num)
result = ArithmeticAverage(num, 30)
#滤掉了高频率的幅值低的杂波
# print(num - result)
pl.subplot(2, 1, 2)
pl.plot(np.linspace(0,1,2330),result)
print(result)
#原函数的傅里叶变换
plt.figure('yuan')
params = {'t': T, 'channel_data': num}
result_fft = myFFT(params)
f0 = plt.subplot(2,1,1)
f0.plot(T,num)
f1 = plt.subplot(2,1,2)
f1.plot(result_fft['freqs'],result_fft['pows'])
#滤波后的傅里叶变换
plt.figure('result')
params={'t':np.linspace(0,1,len(result)),'channel_data':result}
result_fft1 = myFFT(params)
resu=plt.subplot(2,1,1)
resu.plot(result)
resu1 = plt.subplot(2,1,2)
resu1.plot(result_fft1['freqs'],result_fft1['pows'])
pl.show()


#