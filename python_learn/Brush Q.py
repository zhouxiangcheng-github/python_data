# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 15:08
# @File    : Brush Q.py

###
# two_sum
#class two_sum:

class two_sum_class:

    # 简单列表遍历执行
    def two_sum1(self,list,target):
        for i in range(len(list)):
            for j in range(i + 1, len(list)):
                if list[i] + list[j] == target:
                    return i,j
    #使用内部索引
    def two_sum2(self,list,target):
        for i in range(len(list)):
            res = target-list[i]
            if res in list:
                j = list.index(res)
                if j != i :
                    return i,j
                    break
                else:
                    continue
    #使用字典
    def two_sum3(self,list,target):
        d = {}
        for i in range(len(list)):
            a = target-list[i]
            if nums[i] in d:
                return i,d[nums[i]]
            else:
                d[a] = i





target = int(input("输入目标和："))
nums = input("输入：").split(" ")
nums = [int(nums[i]) for i in range(len(nums))]
sum=two_sum_class()
#first,second = sum.two_sum1(nums,target)
#first,second = sum.two_sum2(nums,target)
first,second = sum.two_sum3(nums,target)
print(first,second)

