# -*- coding: utf-8 -*-
# @Time    : 2022/3/12 16:46
# @File    : pandas_data.py
# for i in tqdm(range(10)):
#    time.sleep(random.random())
import numpy as np
import pandas as pd

import os

#可以使用r作为转义符号，同时也可以不使用r，但需要将'\'转为'/',或者将'\'换为'\\'指针对有数字开头的文件名前使用'\\'
path=r'F:\仪控可靠性\1\1.csv'
path1='F:\仪控可靠性\数据\负责的数据\\20_04_13\_20_04_13_10_27_17__SF65ZF-02C振动试验前调试阀A常开_开总95_关总93.csv'
path_out='F:\仪控可靠性\\1\\2.csv'
df=pd.DataFrame(pd.read_csv(path,encoding='gb2312',usecols=[9]))     #使用，提取的数据中，包含了行索引和列索引，在导出的时候如果不需要，可以对index_col=0，即可在导出文件没有行列索引


print(df)
print(df.values[:,0])
print(df.columns)
print(df.axes)

R_TXT = pd.DataFrame(pd.read_table('F:\仪控可靠性\数据\\test\\1.txt',sep=' ' ,usecols=[0,1],header=None)) #pd读取txt文件时要指定分隔符，不然读出的数据当作一列数据保存。
#print(df)
#pd.read_csv参数说明：
#1.filepath_or_buffer为第一个参数，没有default，cann't null，传参数or path：
#2.sep参数是字符型的，代表每行数据内容的分隔符号，默认是逗号，另外常见的还有制表符（\t）、空格等，根据数据的实际情况传值。
#3.header指定第几行是表头，默认会自动推断把第一行作为表头。header=None表示将文件里的第一行也作为数据,若不给header赋值，则默认将第一行作为表头，不作为数据读入。
#4.index_col='年份'：names用来指定列的名称，它是一个类似列表的序列，与数据一一对应。如果文件不包含列名，那么应该设置header=None，列名列表中不允许有重复值。
#5.index_col='年份' 或者 index_col=0：index_col用来指定索引列，可以是行索引的列编号或者列名，如果给定一个序列，则有多个行索引。Pandas不会自动将第一列作为索引，不指定时会自动使用以0开始的自然索引。
#5.index_col=-1:表示不需要前面的序号{0，1，2，3，，，，}
#5.index_col=0:表示不需要前面的序号{0，1，2，3，，，，}
#5.index_col=False:生成的表中含有自带的索引{0，1，2，3，，，}
#6.usecols=[0,4,3]：如果只使用数据的部分列，可以用usecols来指定，这样可以加快加载速度并降低内存消耗。
#7.dtype=np.float64 或者 dtype=np.float64：dtype可以指定各数据列的数据类型。
#8.nrows参数用于指定需要读取的行数，从文件第一行算起，经常用于较大的数据，先取部分进行代码编写。
#df=pd.DataFrame(pd.read_csv('F:\仪控可靠性\\1\\1.csv',dtype={'A阀电流A相工程值':np.float64} ,encoding='gb2312',usecols=['A阀电流A相工程值'],index_col=None))

#domain2 = os.path.abspath(filenames_out)
#print(df.values.max())
#df.to_csv('F:\仪控可靠性\\1\\2.csv',encoding='gb2312')
df.to_excel('F:\仪控可靠性\\1\\1.xlsx',encoding='gb2312',header=None,index=0)       #使用
df.to_csv('F:\仪控可靠性\\1\\3.txt',header=None,index=0)                            #使用,index=0 不输出行索引和列索引
R_TXT.to_csv('F:\仪控可靠性\\1\\2.txt',header=None,index=0)                         #使用
df = pd.DataFrame(pd.read_csv(path1, encoding='gb2312', low_memory=False, dtype={'A阀电流A相工程值': np.float64, 'A阀线电压AB工程值': np.float64}, usecols=['A阀线电压AB工程值', 'A阀电流A相工程值']))
#df = pd.DataFrame(pd.read_csv(path1, encoding='gb2312', low_memory=False, dtype={'A阀电流A相工程值': np.float64, 'A阀线电压AB工程值': np.float64}, usecols=['A阀线电压AB工程值', 'A阀电流A相工程值']))
A_current1=df.values.max(axis=0)   #含有value函数,提取每列的最大值并保存在一个列表中，通过list[i]来遍历每一个值
# print(A_current1)
# print(A_current1[0])
A_current2=df.max(axis=0)          #没有value函数，输出整个列表的时候会把表头输出,如：print(A_current)，若指定输出list的某位置的值不会输出表头：如print(A_current[0])
#print(A_current2)
#print(A_current2[0])
A_current3=df.values.max(axis=1)  #输出每一行的最大值
# print(A_current3)
# print(A_current3[0])
A_current4=df.values              #提取其中的值到列表中
# print(A_current4)
# print(A_current4[0])
