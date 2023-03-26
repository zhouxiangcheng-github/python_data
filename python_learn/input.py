# -*- coding: utf-8 -*-
# @Time    : 2022/3/18 20:00
# @File    : input.py


##输入一个字符列表,并转化为数字列表
str = input("请输入str：").split(" ")          #分隔符为逗号,空格等，在split括号内定义
print(str)
str = [int(str[i]) for i in range(len(str))]  #将字符列表转化为数字列表
print(str)