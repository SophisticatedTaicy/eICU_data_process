# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import threading
import psycopg2
import pandas as pd
from functools import reduce

from pandas import DataFrame

import find_all_features


# 数据处理流程：
# 根据


class MyThread(threading.Thread):
    def __init__(self, name, unitid, cursor):
        super(MyThread, self).__init__(name=name)
        self.name = name
        self.cursor = cursor
        self.unitid = unitid
        self.is_first = True

    def run(self):
        print('%s正在运行中-----------' % str(self.name))
        # 住院记录对应的220维数据信息抽取
        find_all_features.combine_feature(self.cursor, self.unitid)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 获得连接
    conn = psycopg2.connect(database="eicu", user="postgres", password="123456", host="172.16.60.173", port="3307")
    # 获得游标对象
    cursor = conn.cursor()
    # sql语句
    # 查看eicu_crd下面的所有表信息
    # sql = "select * from pg_tables where schemaname='eicu_crd'"
    # 查看具体视图下的具体表信息
    # sql = "select * from information_schema.columns where table_schema = 'eicu_crd'and table_name = 'lab'"
    # 设置当前schema为eicu_crd
    sql = "set search_path to 'eicu_crd'"
    # 执行语句
    cursor.execute(sql)
    # 。。。。。。。。。。 获取数据.
    # 使用年龄，呼吸衰竭，非心血管衰竭，peep筛选数据
    # peep = find_all_features.filter_with_peep(cursor)
    # age = find_all_features.filter_with_age(cursor)
    # respiratory = find_all_features.filter_with_respiratory_failure(cursor)
    # congestive = find_all_features.filter_with_congestive_heart_failure(cursor)

    # # 多个dataframe取交集
    # dfs = [peep, age, respiratory, congestive]
    # temp = reduce(lambda x, y: pd.merge(x, y, how='inner'), dfs)

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
    result = pd.read_csv('result_id.csv', sep=',')
    df = DataFrame(result)
    df_list = []
    # 数据分割
    for i in range(10):
        df_list.append(df[i:i + 1])
    print(df_list.__sizeof__())
    for data in df_list:
        for unitid in data['id']:
            # 分线程运行
            MyThread(name=str(unitid), unitid=unitid, cursor=cursor).start()
