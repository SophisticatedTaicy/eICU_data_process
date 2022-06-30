# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv
import threading
import time
import datetime

import psycopg2
import pandas as pd
from functools import reduce

from pandas import DataFrame
from tqdm import tqdm

import data_process
import query_sql

# 数据处理流程：
# 根据
import init
import pf_filter
from filter.param import sum_list


class MyThread(threading.Thread):
    def __init__(self, name, data):
        super(MyThread, self).__init__(name=name)
        self.name = name
        self.data = data
        self.is_first = True

    def run(self):
        # print('%s区间数据在运行中-----------' % str(self.name))
        for id in self.data:
            # 住院记录对应的220维数据信息抽取
            header = init.init_dict(header=sum_list)
            # 查询相关住院记录动态数据
            query = query_sql.Query()
            dynamic = query.filter_dynamic(id)
            query = query_sql.Query()
            # 查询相关住院记录静态数据
            query.filter_static(id, header)
            # 计算动态数据的中位数、方差以及变化率
            result_list = query_sql.compute_dynamic(dynamic, header)
            # 最终住院记录数据转换为dataframe
            data = DataFrame([result_list])
            # 数据追加写入文件
            data.to_csv('result.csv', mode='a', header=False, index=False)
            print(str(id) + '写入成功！')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 使用年龄，呼吸衰竭，非心血管衰竭，peep筛选数据
    # peep = find_all_features.Querysql().filter_with_peep()
    # age = find_all_features.Querysql().filter_with_age()
    # respiratory = find_all_features.Querysql().filter_with_respiratory_failure()
    # congestive = find_all_features.Querysql().filter_with_congestive_heart_failure()
    #
    # # 多个dataframe取交集
    # dfs = [peep, age, respiratory, congestive]
    # temp = reduce(lambda x, y: pd.merge(x, y, how='inner'), dfs)

    # 根据p/f筛选

    # print(temp)

    # # 将最终数据排序
    # temp.to_csv('temp.csv', encoding='utf-8')

    # # 找到所有患者的pao2,fio2的值

    # p_fs = find_all_features.filter_w
    # ith_pao2_or_fio2(cursor)

    # 最终结果数据展示表头
    # find_all_features.combine_feature(cursor, 231498)
    # 对氧合指数进行筛选,筛选后的所有id是result_id

    # 多线程抽取数据
    # 最终住院记录id提取
    start = time.time()
    print('开始时间 ： ' + str(datetime.datetime.now()))
    # result = pd.read_csv('result_id.csv', sep=',')
    result = pd.read_csv('result.csv', sep=',')
    df = DataFrame(result)
    df_list = []

    # 数据分割
    for i in range(15):
        df_list.append(df['id'][i * 1000:i * 1000 + 1000])
    df_list.append(df['id'][15000:-1])

    for data in tqdm(df_list):
        name = str(data.head(1).index.tolist()[0]) + '-' + str(data.tail(1).index.tolist()[0])
        # 分线程运行
        MyThread(name=name, data=data).start()
    with open('result.csv', mode='a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sum_list)
        writer.writeheader()
    end = time.time()
    total = (end - start) / 60
    print('结束时间 ： ' + str(datetime.datetime.now()) + '程序运行花费了' + str(time.time() - start) + '秒')
