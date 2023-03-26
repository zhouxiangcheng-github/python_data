# -*- coding: utf-8 -*-
# @Time    : 2022/3/11 21:07
# @File    : deal bulk data.py
import os
import re
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import time
import openpyxl



filePaths1 = 'F:\仪控可靠性\数据\负责\\20_03_27'
# filePaths2 = 'F:\仪控可靠性\数据\负责\\20_03_30'
# filePaths3 ='F:\仪控可靠性\数据\负责\\20_03_31'
# filePaths4 = 'F:\仪控可靠性\数据\负责\\20_04_01'
# filePaths5 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_06'
# filePaths6 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_07'
# filePaths7 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_08'
# filePaths8 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_09'
# filePaths9 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_13'
# filePaths10 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_14'
# filePaths11 = 'F:\仪控可靠性\数据\处理完的数据\\20191220'
# filePaths12 = 'F:\仪控可靠性\数据\处理完的数据\\20191226'
# filePath_alls = [filePaths1, filePaths2, filePaths3, filePaths4,
#                  filePaths5, filePaths6, filePaths7, filePaths8,
#                  filePaths9, filePaths10]
filePath_alls = [filePaths1]


save_path1 = 'F:\仪控可靠性\数据\处理\\20_03_27'
# save_path2 = 'F:\仪控可靠性\数据\处理\\20_03_30'
# save_path3 = 'F:\仪控可靠性\数据\处理\\20_03_31'
# save_path4 = 'F:\仪控可靠性\数据\处理\\20_04_01'
# save_path5 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_06'
# save_path6 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_07'
# save_path7 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_08'
# save_path8 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_09'
# save_path9 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_13'
# save_path10 = 'F:\仪控可靠性\数据\处理完的数据\\20_04_14'
#
# save_path_all=[save_path1,save_path2,save_path3,save_path4,
#                save_path5,save_path6,save_path7,save_path8,
#                save_path9.save_path10]
save_path_all=[save_path1]

#处理一个文件
def deal_one(filepath,save_path):
    filenames_in = filepath # 输入文件的文件地址
    filenames_out = save_path  # 新文件的地址
    pathDir = os.listdir(filenames_in)     #读取文件夹路径下的文件列表
    print(len(pathDir))
    for allDir in pathDir:
        child = re.findall(r"(.+?).csv", allDir)  # 正则的方式读取文件名，去扩展名
        if len(child) >= 0:  # 去掉没用的系统文件
            newfile = ''
            needdate = child  #### 这个就是所要的文件名
        domain1 = os.path.abspath(filenames_in)  # 待处理文件位置
        info = os.path.join(domain1, allDir)  # 拼接出待处理文件名字

        df = pd.DataFrame(pd.read_csv(info, encoding='gb2312', dtype={'A阀电流A相工程值': np.float64}, usecols=['A阀电流A相工程值']))
        #df = df.values
        domain2 = os.path.abspath(filenames_out)  # 处理完文件保存地址
        outfo = os.path.join(domain2, allDir)  # 拼接出新文件名字
        # df.to_csv(outfo, encoding='gb2312',header=None, index=0)  #去除表头
        #np.savetxt()
        df.to_csv(outfo, encoding='gb2312')

        print(info, "处理完")

        # df = pd.DataFrame(pd.read_csv(info,encoding='gb2312',low_memory=False,dtype={'A阀电流A相工程值':np.float64,'A阀线电压AB工程值':np.float64}, usecols=['A阀线电压AB工程值', 'A阀电流A相工程值']))  # 选取固定列的值生成新表，选取第10列作为新表//有的文件夹选择第11列
        # #print(df.values)
        # A_current=df.values.max(axis=0)   #写不写axis=0貌似无影响
        # #AB_Voltage=df.values.max(axis=0)
        # print(A_current)
        # if(float(A_current[1])>2.2):            #设立阈值，根据文件具体设置
        #     domain2 = os.path.abspath(filenames_out)  # 处理完文件保存地址
        #     outfo = os.path.join(domain2, allDir)  # 拼接出新文件名字
        #     df.to_csv(outfo, encoding='gb2312')
        #     print(info, "处理完")
        #     print(A_current)
#处理一个子文件,单个文件（excel，txt，csv）的具体地址
def deal_one_one(filepath,save_path):

    print("处理开始")
    df=pd.DataFrame(pd.read_csv(filepath,encoding='gb2312',usecols=['A阀电流A相工程值']))
    print("......")
    df.to_csv(save_path,encoding='gb2312',header=None,index=0)
    print("处理结束")

#创建文件夹
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")

        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


#批量处理数据 对多个文件夹进行遍历，每个文件夹里面含有多个csv文件
#提取每个csv文件里的某一行（列）或多行（列）数据，并对该行（列）数据进行处理
# 将提取出的数据按照相同格式存放到另一个文件夹里
def deal_bulk_data(filePath_alls,save_path_all):

    for filenames_in,filenames_out in zip(filePath_alls,save_path_all):

        pathDir = os.listdir(filenames_in)

        for allDir in pathDir:
            child = re.findall(r"(.+?).csv", allDir)  # 正则的方式读取文件名，去扩展名
            if len(child) >= 0:  # 去掉没用的系统文件
                newfile = ''
                needdate = child  #### 这个就是所要的文件名
            print(filenames_in)
            domain1 = os.path.abspath(filenames_in)  # 待处理文件位置
            info = os.path.join(domain1, allDir)  # 拼接出待处理文件名字
            print(info)
            df = pd.DataFrame(pd.read_csv(info,encoding='gb2312', dtype={'A阀电流A相工程值':np.float64,'A阀线电压AB工程值':np.float64}, usecols=['A阀线电压AB工程值', 'A阀电流A相工程值']))  # 选取固定列的值生成新表，选取第10列作为新表
            A_current=df.values.max(axis=0)
            # print(t)
            if(A_current[0]>1.5):            #设立阈值，根据文件具体设置,A_current[0]表示电流，A_current[1]表示电压
                domain2 = os.path.abspath(filenames_out)  # 处理完文件保存地址
                outfo = os.path.join(domain2, allDir)  # 拼接出新文件名字
                #df.to_csv(outfo, encoding='gb2312',header=None, index=0)  #去除表头
                df.to_csv(outfo, encoding='gb2312')
                print(info, "处理完")
                print(A_current)
def deal_bulk_data1(filePath_alls):#输出每个文件的长度

    for filenames_in in filePath_alls :

        pathDir = os.listdir(filenames_in)

        for allDir in pathDir:
            child = re.findall(r"(.+?).csv", allDir)  # 正则的方式读取文件名，去扩展名
            if len(child) >= 0:  # 去掉没用的系统文件
                newfile = ''
                needdate = child  #### 这个就是所要的文件名
            domain1 = os.path.abspath(filenames_in)  # 待处理文件位置
            info = os.path.join(domain1, allDir)  # 拼接出待处理文件名字

            df = pd.DataFrame(pd.read_csv(info,encoding='gb2312', dtype={'A阀电流A相工程值':np.float64,'A阀线电压AB工程值':np.float64}, usecols=[ 'A阀电流A相工程值']))  # 选取固定列的值生成新表，选取第10列作为新表
            print(len(df)/10,df.values.reshape(1,len(df.values)))
# 处理一个文件夹，并将csv文件输出为同名的txt文件。
def deal_ones(filepath, save_path):
    filenames_in = filepath    # 输入文件的文件地址
    filenames_out = save_path  # 新文件的地址
    pathDir = os.listdir(filenames_in)  # 读取文件夹路径下的文件列表
   # print(pathDir)
   # print(len(pathDir))
    for allDir in pathDir:
        child = re.findall(r"(.+?).csv", allDir)  # 正则的方式读取文件名，去扩展名
        if len(child) >= 0:  # 去掉没用的系统文件
            newfile = ''
            needdate = child  #### 这个就是所要的文件名
        domain1 = os.path.abspath(filenames_in)  # 待处理文件位置
        info = os.path.join(domain1, allDir)  # 拼接出待处理文件名字
        df = pd.DataFrame(pd.read_csv(info, encoding='gb2312', dtype={'A阀电流A相工程值': np.float64}, usecols=['A阀电流A相工程值']))
        # df = df.values
        domain2 = os.path.abspath(filenames_out)  # 处理完文件保存地址
        #print(allDir)
        #print(domain2)
        outfo = os.path.join(domain2, allDir)  # 拼接出新文件名字
        #allDirs = os.path.basename(outfo)      #得到文件名字
        allDirs = os.path.splitext(outfo)[0]
        print(os.path.splitext(outfo)[0],os.path.splitext(outfo)[1])
        newname= allDirs + '.txt'        #拼接加后缀
        outfo1 = os.path.join(domain2,newname) #路径和文件叠加
        # df.to_csv(outfo, encoding='gb2312',header=None, index=0)  #去除表头
        # np.savetxt()


        df = df.values.reshape(len(df.values),1)
        t = np.linspace(0,1,len(df)).reshape(len(df),1)
        df = np.hstack((t,df))
        df = pd.DataFrame(df)
        # print(df)
        df.to_csv(outfo1, encoding='gb2312',index=0,header=None)  #输出纯数据，不带表头序号
        print(info, "处理完")

#处理输入路径下的多个文件夹
#路径下的文件格式为：根路径//文件夹//文件夹//文件(.csv)
#在输出路径下建立相同类型的文件夹和文件，保存为txt文件
def deal_floders(filePaths1,save_path):                    #传入根目录
    filePaths1 = filePaths1                           #把根路径定义为第0层:表示为根路径//文件夹//文件夹//文件(.csv)
    save_path  = save_path
    floder_name_1 = os.listdir(filePaths1)                     #找出根路径下的所有文件夹，表示为第一层文件夹，[处理前_退化', '程序数据_处理前]
    print("floder_name_1:",floder_name_1)

    for floder_name_2 in floder_name_1:                        #分别访问路径下的各个子文件夹
        filename =  filePaths1 + "\\" + floder_name_2          #拼接第一层各个文件夹的路径，理解为第一层文件夹的完整路径
        print("filename",filename)

        fileoutname =save_path + "\\" + floder_name_2[:-1]+'后'
        print("fileoutname:",fileoutname)
        mkdir(fileoutname)


        file_folder_2 = os.listdir(filename)                   #分别读取各个子文件夹下的子文件夹,属于最后一层文件夹
        print("file_folder_2:",file_folder_2)
        #final_file = ""
        for final_folder_3 in file_folder_2:                   #访问最后一层文件夹
            final_folder = filename + "\\"+final_folder_3      #拼接最后一层文件夹目录的各个子文件夹的完整路径
            print("final_files:",final_folder)

            final_folder_out =fileoutname + "\\"+final_folder_3
            print("final_folder_out:", final_folder_out)
            mkdir(final_folder_out)

            files = os.listdir(final_folder)
            all_val = []
            all_val = np.array(all_val).reshape(-1, 32)
            for file in files:                                  #这里访问的是每个最后一层子文件夹下的单个子文件(.csv)
                child = re.findall(r"(.+?).csv", file)          #正则的方式读取文件名，去扩展名
                print("child:",child)
                if len(child) >= 0:  # 去掉没用的系统文件
                    newfile = ''
                    needdate = child  #### 这个就是所要的文件名
                domain1 = os.path.abspath(final_folder)  # 待处理文件位置
                print("待处理文件位置:",domain1)
                info = os.path.join(domain1, file)  # 拼接出待处理文件名字
                print("待处理文件名字:",info)
                data = pd.DataFrame(pd.read_csv(info, encoding='gb2312', low_memory=False, dtype={'A阀电流A相工程值': np.float64},
                                                usecols=['A阀电流A相工程值'])).values  # 选取固定列的值生成新表，选取第10列
                data = data.reshape(len(data), 1)

                domain2 = os.path.abspath(final_folder_out)  # 处理完文件保存地址
                # print(allDir)
                # print(domain2)
                outfo = os.path.join(domain2, file)  # 拼接出新文件名字
                # allDirs = os.path.basename(outfo)      #得到文件名字
                allDirs = os.path.splitext(outfo)[0]
                print(os.path.splitext(outfo)[0], os.path.splitext(outfo)[1])
                #newname = allDirs + '.txt'  # 拼接加后缀txt
                newname = allDirs + '.xlsx' #
                outfo1 = os.path.join(domain2, newname)  # 路径和文件叠加
                print("outfo1:",outfo1)
                data = pd.DataFrame(data,columns=['A阀A相电流'])
                #data.to_csv(outfo1, encoding='gb2312', index=0, header=None)  # 输出纯数据，不带表头序号
                data.to_excel(outfo1, encoding='gb2312', index=0)


# deal_bulk_data(filePath_alls,save_path_all)
# deal_one(filePaths1,save_path2)
#deal_one_one('F:\仪控可靠性\\1\\1.csv','F:\仪控可靠性\\1\\2.txt')
#filepath = [ r"F:\仪控可靠性\数据\程序数据_处理前\故障",r"F:\仪控可靠性\数据\程序数据_处理前\正常"]

# savepath = 'F:\仪控可靠性\数据\处理后输出'
#deal_bulk_data1(filepath)
# deal_one(filepath,savepath)

#创建文件夹
'''
file = 'F:\Data_Code\仪控_处理' #路径
mkdir(file)
'''

#处理根目录下的文件夹，文件夹下仍然是文件夹
deal_floders(r'F:\Data_Code\仪控',r'F:\Data_Code\仪控_处理后')



#deal_bulk_data(filePath_alls,save_path_all)

