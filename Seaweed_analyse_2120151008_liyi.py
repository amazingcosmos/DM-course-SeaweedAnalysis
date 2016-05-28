
# coding: utf-8

# # 数据挖掘作业 1 数据探索性分析与预处理
# 
# **姓名：李懿**
# 
# **学号：2120151008**
# 
# **日期：2016.5.28**

# ### 数据分析要求
# 
# **1. 数据可视化和摘要**
# 
# - 数据摘要
# 
# 对标称属性，给出每个可能取值的频数
# 
# 对数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数。
# 
# - 数据的可视化
# 
# 针对数值属性：
# 
# 绘制直方图，如mxPH，用qq图检验其分布是否为正态分布。
# 
# 绘制盒图，对离群值进行识别。
# 
# 对7种海藻，分别绘制其数量与标称变量，如size的条件盒图
# 
# **2. 数据缺失的处理**
# 
# - 分别使用下列四种策略对缺失值进行处理，处理后可视化地对比新旧数据集。
# 
# 1.将缺失部分剔除
# 2.用最高频率值来填补缺失值
# 3.通过属性的相关关系来填补缺失值
# 4.通过数据对象之间的相似性来填补缺失值

# ### 解答内容

#!/usr/bin/env python

import operator
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


# **Step1. 数据处理**
# 
# - 将原始txt文件转换为易于处理的csv文件

# 转换文件格式，生成csv文件
fp_origin = open("./Analysis.txt", 'r')
fp_modified = open("./Analysis.csv", 'w')

line = fp_origin.readline()
while(line):
    temp = line.strip().split()
    temp = ','.join(temp)+'\n'
    fp_modified.write(temp)
    line = fp_origin.readline()
    
fp_origin.close()
fp_modified.close()


# **Step2. 读取数据**
# 
# - 读取csv文件，生成data frame

# 定义两类数据：标称型和数值型
name_category = ["season", "river_size", "river_speed"]
name_value = ["mxPH", "mnO2", "Cl", "NO3", "NH4", "oPO4", "PO4", "Chla", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]
# 存储7种海藻对应的名称
name_seaweed = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]

# 读取数据
data_origin = pd.read_csv("./Analysis.csv", 
                   names = name_category+name_value,
                   na_values = "XXXXXXX")

# 将字符数据转换为category
for item in name_category:
    data_origin[item] = data_origin[item].astype('category')


# **Step 3. 数据摘要**
# 
# - 对标称属性，给出每个可能取值的频数

# 使用value_counts函数统计每个标称属性的取值频数
for item in name_category:
    print item, '的频数为：\n', pd.value_counts(data_origin[item].values), '\n'


# - 对数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数。

# 最大值
data_show = pd.DataFrame(data = data_origin[name_value].max(), columns = ['max'])
# 最小值
data_show['min'] = data_origin[name_value].min()
# 均值
data_show['mean'] = data_origin[name_value].mean()
# 中位数
data_show['median'] = data_origin[name_value].median()
# 四分位数
data_show['quartile'] = data_origin[name_value].describe().loc['25%']
# 缺失值个数
data_show['missing'] = data_origin[name_value].describe().loc['count'].apply(lambda x : 200-x)

print data_show


# **Step 4. 数据可视化 **
# 
# - 针对数值属性：
# 绘制直方图，如mxPH，用qq图检验其分布是否为正态分布。

# 直方图
fig = plt.figure(figsize = (20,11))
i = 1
for item in name_value:
    ax = fig.add_subplot(3, 5, i)
    data_origin[item].plot(kind = 'hist', title = item, ax = ax)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('./image/histogram.jpg')
print 'histogram saved at ./image/histogram.jpg'


# qq图
fig = plt.figure(figsize = (20,12))
i = 1
for item in name_value:
    ax = fig.add_subplot(3, 5, i)
    sm.qqplot(data_origin[item], ax = ax)
    ax.set_title(item)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('./image/qqplot.jpg')
print 'qqplot saved at ./image/qqplot.jpg'

# 从qq图中可以看出，只有mxPH和mnO2两项值符合正态分布，其他值均不符合

# - 绘制盒图，对离群值进行识别。
# 盒图
fig = plt.figure(figsize = (20,12))
i = 1
for item in name_value:
    ax = fig.add_subplot(3, 5, i)
    data_origin[item].plot(kind = 'box')
    i += 1
fig.savefig('./image/boxplot.jpg')
print 'boxplot saved at ./image/boxplot.jpg'


# - 对7种海藻，分别绘制其数量与标称变量，如size的条件盒图
# 条件盒图
fig = plt.figure(figsize = (10, 27))
i = 1
for seaweed in name_seaweed:
    for category in name_category:
        ax = fig.add_subplot(7, 3, i)
        data_origin[[seaweed, category]].boxplot(by = category, ax = ax)
        ax.set_title(seaweed)
        i += 1
plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
fig.savefig('./image/boxplot_condition.jpg')
print 'boxplot_condition saved at ./image/boxplot_condition.jpg'


# **Step 4. 数据缺失的处理**
# 
# 可视化方法：对于**标称属性**，绘制属性的折线图，图中红线是原始数据，蓝线是处理完缺失值之后的数据；**数值属性**：使用直方图，将原始数据和处理后的数据图像进行叠加。图中红色的垂线是原始数据的均值，蓝色的垂线是处理完缺失值之后的均值。

# 4.0 观察数据
# 
# 从绘制的表格上可以看出，缺失值主要集中在Cl、Chla两个属性，第62、199条数据缺失情况比较严重

# 找出含有缺失值的数据条目索引值
nan_list = pd.isnull(data_origin).any(1).nonzero()[0]

# 显示含有缺失值的原始数据条目
# data_origin.iloc[nan_list].style.highlight_null(null_color='red')


# 4.1 将缺失部分剔除
# 
# 使用***dropna()***函数操作。从结果可以看出，由于删除了带有缺失值的整条数据。
# 
# 从标称属性的折线图，可以明显看出处理后的数据量减少；直方图中，蓝色线和红色线不重合，但是十分接近，说明数值属性的均值有改变，但是变化不大。

# 将缺失值对应的数据整条剔除，生成新数据集
data_filtrated = data_origin.dropna()

# 绘制可视化图
fig = plt.figure(figsize = (20,15))

i = 1
# 对标称属性，绘制折线图
for item in name_category:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1

i = 6
# 对数值属性，绘制直方图
for item in name_value:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'filtrated', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('./image/filted_missing_data1.jpg')
print 'filted_missing_data1 saved at ./image/filted_missing_data1.jpg'


# 4.2 用最高频率值来填补缺失值
# 
# 使用***value_counts()***函数统计原始数据中，出现频率最高的值，再用***fillna()***函数将缺失值替换为最高频率值。
# 
# 从折线图看出，处理后标称属性值不变；从直方图可以看出，数值属性的缺失值补全为高频值，均值基本保持不变。


# 建立原始数据的拷贝
data_filtrated = data_origin.copy()
# 对每一列数据，分别进行处理
for item in name_category+name_value:
    # 计算最高频率的值
    most_frequent_value = data_filtrated[item].value_counts().idxmax()
    # 替换缺失值
    data_filtrated[item].fillna(value = most_frequent_value, inplace = True)

# 绘制可视化图
fig = plt.figure(figsize = (20,15))

i = 1
# 对标称属性，绘制折线图
for item in name_category:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1    

i = 6
# 对数值属性，绘制直方图
for item in name_value:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('./image/filted_missing_data2.jpg')
print 'filted_missing_data2 saved at ./image/filted_missing_data2.jpg'


# 4.3 通过属性的相关关系来填补缺失值
# 
# 使用pandas中Series的***interpolate()***函数，对数值属性进行插值计算，并替换缺失值。
# 
# 从直方图中可以看出，处理后的数据，添加了若干个值不同的值，并且均值变化不大。


# 建立原始数据的拷贝
data_filtrated = data_origin.copy()
# 对数值型属性的每一列，进行插值运算
for item in name_value:
    data_filtrated[item].interpolate(inplace = True)

# 绘制可视化图
fig = plt.figure(figsize = (20,15))

i = 1
# 对标称属性，绘制折线图
for item in name_category:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1   
    
i = 6
# 对数值属性，绘制直方图
for item in name_value:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('./image/filted_missing_data3.jpg')
print 'filted_missing_data3 saved at ./image/filted_missing_data3.jpg'


# 4.4 通过数据对象之间的相似性来填补缺失值
# 
# 首先将缺失值设为0，对数据集进行正则化。然后对每两条数据进行差异性计算（分值越高差异性越大）。计算标准为：标称数据不相同记为1分，数值数据差异性分数为数据之间的差值。在处理缺失值时，找到和该条数据对象差异性最小（分数最低）的对象，将最相似的数据条目中对应属性的值替换缺失值。
# 
# 从直方图可以看出，mnO2、Cl、Chla的值发生了改变


# 建立原始数据的拷贝，用于正则化处理
data_norm = data_origin.copy()
# 将数值属性的缺失值替换为0
data_norm[name_value] = data_norm[name_value].fillna(0)
# 对数据进行正则化
data_norm[name_value] = data_norm[name_value].apply(lambda x : (x - np.mean(x)) / (np.max(x) - np.min(x)))

# 构造分数表
score = {}
range_length = len(data_origin)
for i in range(0, range_length):
    score[i] = {}
    for j in range(0, range_length):
        score[i][j] = 0    

# 在处理后的数据中，对每两条数据条目计算差异性得分，分值越高差异性越大
for i in range(0, range_length):
    for j in range(i, range_length):
        for item in name_category:
            if data_norm.iloc[i][item] != data_norm.iloc[j][item]:
                score[i][j] += 1
        for item in name_value:
            temp = abs(data_norm.iloc[i][item] - data_norm.iloc[j][item])
            score[i][j] += temp
        score[j][i] = score[i][j]

# 建立原始数据的拷贝
data_filtrated = data_origin.copy()

# 对有缺失值的条目，用和它相似度最高（得分最低）的数据条目中对应属性的值替换
for index in nan_list:
    best_friend = sorted(score[index].items(), key=operator.itemgetter(1), reverse = False)[1][0]
    for item in name_value:
        if pd.isnull(data_filtrated.iloc[index][item]):
            if pd.isnull(data_origin.iloc[best_friend][item]):
                data_filtrated.ix[index, item] = data_origin[item].value_counts().idxmax()
            else:
                data_filtrated.ix[index, item] = data_origin.iloc[best_friend][item]

# 绘制可视化图
fig = plt.figure(figsize = (20,15))

i = 1
# 对标称属性，绘制折线图
for item in name_category:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1   
    
i = 6
# 对数值属性，绘制直方图
for item in name_value:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('./image/filted_missing_data4.jpg')
print 'filted_missing_data4 saved at ./image/filted_missing_data4.jpg'


# **Step 5.保存预处理后的数据集**
data_filtrated.to_csv('./Analysis_filted.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print 'data after analysis saved at ./Analysis_filted.csv'

