# todo::compute p/f
import numpy as np
from soupsieve.util import lower
# compute median，variances and change rate for all dynamic items
from filter.param import dynamic_list


def init_dict(header):
    dict = {}
    for item in header:
        dict[item] = 0
    return dict


def compute_dynamic_factor(data, header):
    # add head to data
    data.columns = ['name', 'result', 'time']
    # 对于多名称特征数据进行更名
    for index, row in data.iterrows():
        if "Carboxyhemoglobin" in row['name']:
            row['name'] = 'Hemoglobin'
        elif "Methemoglobin" in row['name']:
            row['name'] = 'Hemoglobin'
        elif "HCO3" in row['name']:
            row['name'] = 'bicarbonate'
        elif 'bedside glucose' in row['name']:
            row['name'] = 'glucose'
        # dataframe修改数据需要添加下面这句，否则修改不成功
        data.iloc[index] = row
    # 将处理后的数据进行排序，有利于后续中位数和方差的计算
    data.sort_values(by='name', inplace=True, ascending=True)
    result_list = {}
    # data.to_csv('dynamic.csv', encoding='utf-8')
    i = 0
    # 动态数据中位数，方差，变化率的计算
    while i < len(data):
        item_name = data.iloc[i, 0]
        item_list = []
        item_list.append(data.iloc[i, 1])
        # 汇总每种变量的所有信息
        j = i + 1
        # 取出同一特征item_name的所有数据存储在列表item_list中
        while j < len(data):
            if data.iloc[j, 0] == item_name:
                item_list.append(data.iloc[j, 1])
                j += 1
            else:
                break
        # 当数据只有一条时，其中位数、方差、变化率直接可以得出
        if j - i == 1:
            median = data.iloc[i, 1]
            variance = 0
            change_rate = 0
        # 计算中位数，方差，变化率
        else:
            if len(item_list) % 2 == 0:
                median = (item_list[int(len(item_list) / 2) - 1] + item_list[int(len(item_list) / 2)]) / 2
            else:
                median = item_list[int(len(item_list) / 2)]
            variance = np.var(item_list)
            change_rate = (np.max(item_list) - np.min(item_list)) / np.min(item_list)
        # 数据取小数点后两位
        result_list[item_name + '_median'] = round(median, 2)
        result_list[item_name + '_variances'] = round(variance, 2)
        result_list[item_name + '_changerate'] = round(change_rate, 2)
        i = j
        # 排除已经计算的特征
        if item_name in dynamic_list:
            dynamic_list.remove(item_name)
    # 将未计算或者不存在的特征中位数、方差、变化率置0
    for item in dynamic_list:
        result_list[item + '_median'] = 0
        result_list[item + '_variances'] = 0
        result_list[item + '_changerate'] = 0
    # 更新动态数据数值
    for key, value in result_list.items():
        header[key] = value
    return header
