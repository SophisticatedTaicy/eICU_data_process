# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv
import threading
import time
import datetime

import numpy as np
import psycopg2
import pandas as pd
from functools import reduce

from pandas import DataFrame
from tqdm import tqdm

import data_process
import filter.param
import query_sql

# 数据处理流程：
# 根据
import init
import pf_filter
from filter.param import result_header


def concat(data, label):
    origin_ids = data['id']
    modify_ids = label['id']
    print('id is : ' + str(label['id']))
    statuses = label['status']
    details = label['detail']
    new_status = []
    new_detail = []
    for origin_id in origin_ids:
        for modify_id, status, detail in zip(modify_ids, statuses, details):
            if origin_id == modify_id:
                new_status.append(status)
                new_detail.append(detail)
    data.drop(columns='status', inplace=True)
    data['status'] = new_status
    data['detail'] = new_detail
    dataframe = DataFrame(data)
    dataframe.to_csv('data/new_result.csv', mode='w', encoding='utf-8', index=False)
    print('done!')


class MyThread(threading.Thread):
    def __init__(self, name, data):
        super(MyThread, self).__init__(name=name)
        self.name = name
        self.data = data
        self.is_first = True

    def run(self):
        print('%s区间数据在运行中-----------' % str(self.name))
        for item in self.data:
            # 住院记录对应的220维数据信息抽取
            header = init.init_dict(header=result_header)
            id = item[0]
            identification = item[1]
            severity = item[2]
            enrollment = item[3]
            header['id'] = id
            header['severity'] = severity
            # 查询相关住院记录动态数据
            query = query_sql.Query()
            dynamic = query.filter_dynamic(id, identification, enrollment)
            query = query_sql.Query()
            # # 查询相关住院记录静态数据
            query.filter_static(item, header)
            query = query_sql.Query()
            status, detail, unitstay, hospitalstay = query.access_outcome(id, enrollment)
            # print('id : ' + str(header['id']) + ' status : ' + str(status) + ' detail : ' + str(detail))
            header['status'] = status
            header['detail'] = detail
            header['unit'] = unitstay
            header['hospital'] = hospitalstay
            # 计算动态数据的中位数、方差以及变化率
            result_dict = query_sql.compute_dynamic(dynamic, header)
            data = DataFrame([result_dict])
            # 数据追加写入文件
            data.to_csv('result/feature_data.csv', mode='a', encoding='utf-8', header=False, index=False)
            print(str(item) + '写入成功！')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # # 使用年龄，呼吸衰竭，非心血管衰竭，peep筛选数据
    # # # 患者年龄不小于18且需要机械通气
    # start = time.time()
    # print('开始时间 ： ' + str(datetime.datetime.now()))
    # query = query_sql.Query()
    # # 200234
    # age_id = query.filter_with_age()
    # # 需要机械通气
    # query = query_sql.Query()
    # vent_support_id = query.filter_with_vent_support()
    # # print(str(age_id))
    # # # 患者患有呼吸衰竭且不为心血管衰竭
    # query = query_sql.Query()
    # respiratory_id = query.filter_with_respiratory_failure_without_congestive_heart_failure()
    # # print(str(respiratory_id))
    # # # 取满足前述条件的住院记录id
    # ids = [age_id, vent_support_id, respiratory_id]
    # temp = reduce(lambda x, y: pd.merge(x, y, how='inner'), ids)
    # # 将dataframe中重复的内容去除以后转换为list
    # result_list = temp[0].drop_duplicates().values.tolist()
    # result_dataframe = DataFrame(result_list)
    # result_dataframe.to_csv('result/respiratory_filter.csv', mode='w', encoding='utf-8', index=False)
    # # print(str(result_list))
    # print('after filter with age, respiratory failure and vent support : ' + str(len(result_list)))
    # result = []
    # result_list = np.array(pd.read_csv('result/respiratory_filter.csv'))
    # # print(str(result_list[0][0]))
    # for unitid in result_list:
    #     query = query_sql.Query()
    #     # print(str(unitid))
    #     # 患者连续八小时peep不小于5且p/f不大于300
    #     legal = query.filter_with_p_f(unitid[0])
    #     if not legal is None:
    #         result.append(legal)
    # p_f = DataFrame(result)
    # p_f.to_csv('result/p_f_filter.csv', mode='w', encoding='utf-8', index=False)
    # print('after filter with  p/f  : ' + str(len(result)))
    # result = np.array(pd.read_csv('result/p_f_filter.csv'))
    # print('data type ' + str(result.shape))
    # final = []
    # for i in range(0, len(result)):
    #     item = {}
    #     item['id'] = result[i][1]
    #     item['identification'] = result[i][2]
    #     item['severity'] = result[i][0]
    #     id = item['id']
    #     identification = item['identification']
    #     severity = item['severity']
    #     # print('id: ' + str(id) + ' identification : ' + str(identification) + ' severity : ' + str(severity))
    #     query = query_sql.Query()
    #     peep_judge = query.filter_with_peep(id, identification)
    #     if peep_judge:
    #         item['enrollment'] = identification + 1440
    #         final.append(item)
    # print('after filter with peep  : ' + str(len(final)))
    # peep = DataFrame(final)
    # peep.to_csv('result/peep_filter.csv', mode='w', encoding='utf-8', index=False)
    # 多线程抽取数据
    # 最终住院记录id和确诊时间提取
    # result: object = np.array(pd.read_csv('result/peep_filter.csv'))
    # df_list = []
    # # 数据分割
    # for i in range(12):
    #     df_list.append(result[i * 700:i * 700 + 700])
    # df_list.append(result[8400:-1])
    # for data in tqdm(df_list):
    #     name = str(data[0][0]) + '-' + str(data[-1][0])
    #     # 分线程运行
    #     MyThread(name=name, data=data).start()
    # 表头写入
    # Header = filter.param.result_header
    # with open('result/feature_data.csv', mode='a', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=Header)
    #     writer.writeheader()
    # 动态缺省值填充，静态缺失值填充
    result = pd.read_csv('result/feature_data.csv', low_memory=False)
    # fill_invalid_data = query_sql.fill_invalid_data_with_average(result)
    fill_with_0 = query_sql.fill__with_0(result)
