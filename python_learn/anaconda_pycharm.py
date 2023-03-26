# -*- coding: utf-8 -*-
#参考：https://www.cnblogs.com/hejer/p/12108775.html


# 复制或克隆环境：
#
# conda create -n 新环境名称–clone 被克隆环境名称
#
# 例如，通过克隆tensorflow2来创建一个称为newtensorflow的副本：
#
# conda create -n newtensorflow–clone tensorflow2

#添加镜像源

# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
# #设置搜索时显示通道地址
# conda config --set show_channel_urls yes


#显示添加的源
# conda config --show channels


#删除指定的源
#conda config --remove channels 源名称或链接