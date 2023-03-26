
# -*- coding:utf-8 _*-
""" 
@file: 01-13-P.py 
@time: 2022/01/13
"""
#!/F/ZXC/python-project/env python
# -*- coding:utf-8 _*-
""" 
@file: FASTICA12.py 
@time: 2022/01/12
"""
import numpy as np
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA

# Compute ICA
# 改变文件入口text？，更改源个数--n_components
data=np.loadtxt("F:\python-project/test_data/test4.txt")
#np.savetxt(r'F:\python-project/test_data/test5.txt',data[:,0])
color_f = ["red", "steelblue", "orange"]
#enumerate是内置枚举函数，zip函数是打包models，names，，
# 1代表start=1，意思是列表第一个元素代号为1
#observing data/观测数据绘制图
model_f=data
#print(data)
color_f= ["red", "steelblue", "orange"]
for i,(sig, color) in enumerate(zip(model_f.T, color_f),1):
        Sign1=plt.subplot(3,1,i)
        Sign1.plot(sig, color=color)
plt.show()

#运用ICA将观测数据分离成独立信号并绘制图
ica = FastICA(n_components=3)
S_ = ica.fit_transform(data)  # Reconstruct signals
print(S_)
colors = ["red", "steelblue", "orange"]

#enumerate是内置枚举函数，zip函数是打包models，names，，
# 1代表start=1，意思是列表第一个元素代号为1
#zip(model,colors)是把model和colors对应组合
model=S_
colors= ["red", "steelblue", "orange"]
for i,(sig, color) in enumerate(zip(model.T, colors),1):
        Sign1=plt.subplot(3,1,i)
        Sign1.plot(sig, color=color)
print(S_)
np.savetxt(r'F:\python-project\test_data\test2.txt', data)



plt.show()