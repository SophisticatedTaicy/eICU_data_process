import pandas as pd
import threading
import numpy as np
import psycopg2
from pandas import DataFrame

import query_sql


class MyThread(threading.Thread):
    def __init__(self, data, thread_name):
        # 注意：一定要显式的调用父类的初始化函数。
        super(MyThread, self).__init__(name=thread_name)
        self.data = data
        self.name = thread_name

    def run(self):
        # 遍历dataframe self.data中包络500个id
        for id in self.data:
            # print(str(id) + '住院记录p/f值计算中。。。。')
            # 用均值填充缺少的值
            query = query_sql.Query()
            data = query.filter_with_p_f(id)
            data = compute_pf(data)
            data = (id, data)
            dataframe = DataFrame(data)
            dataframe.to_csv('p_f.csv', mode='a')


# 判断其是否满足 p/f 大于8小时（480）
def pf_value_filter(id_list_df):
    # 按照时间排序
    id_list_df = id_list_df.sort_values('time', inplace=False, ascending=True)
    time_list = id_list_df['time'].unique()
    # print(time_list)

    pf_value = []
    for current_time in time_list:
        f_in_time = id_list_df[(id_list_df['time'] == current_time) & (id_list_df['labname'] == 'FiO2')]
        p_in_time = id_list_df[(id_list_df['time'] == current_time) & (id_list_df['labname'] == 'paO2')]
        pf_value.append(
            round(p_in_time.loc[p_in_time.index[0], 'value'] / f_in_time.loc[f_in_time.index[0], 'value'], 1))
    # print(pf_value)

    valid_time = 0
    start_time = time_list[0]

    for pf, pf_time in zip(pf_value, time_list):
        # print(str(pf_time) + '--' + str(pf))
        # p/f 小于300
        if pf < 300:
            valid_time = valid_time + (pf_time - start_time)
            start_time = pf_time
            # print(valid_time)
            if valid_time >= 480:
                return True
        else:
            valid_time = 0
            start_time = pf_time
    return False


# 处理一个id对应的lab数据并返回处理后的数据
def fill_with_pf(id):
    # 获取当前id对应的lab检查项
    lab_list = query_sql.Querysql().filter_pf_by_id(id)

    # 当查到的lab记录只有一条时，直接剔除
    if len(lab_list) <= 1:
        return pd.DataFrame()

    # if len(lab_list) <= 1
    lab_list_df = pd.DataFrame(lab_list, columns=['id', 'time', 'name', 'value'])

    # 查询到的记录中没有 FiO2 或 paO2
    if len(lab_list_df.loc[lab_list_df['name'] == 'FiO2']) == 0:
        return pd.DataFrame()
    if len(lab_list_df.loc[lab_list_df['name'] == 'paO2']) == 0:
        return pd.DataFrame()

    # 计算p和f的均值
    fio2_df = lab_list_df.loc[lab_list_df['name'] == 'FiO2']
    fio2_temp_mean = round(fio2_df['value'].mean(), 1)
    pao2_df = lab_list_df.loc[lab_list_df['name'] == 'paO2']
    pao2_temp_mean = round(pao2_df['value'].mean(), 1)

    if fio2_temp_mean == 0 or pao2_temp_mean == 0:
        return pd.DataFrame()

    if fio2_temp_mean == 0 or pao2_temp_mean == 0:
        print('------------------------')
        print(id)

    # 提取时间后去重
    time_list = lab_list_df['time'].unique()

    # 取出id，以便后续新建记录用
    temp_id = lab_list_df.head(1)['id']
    for current_time in time_list:
        f_in_time = lab_list_df[(lab_list_df['time'] == current_time) & (lab_list_df['name'] == 'FiO2')]
        p_in_time = lab_list_df[(lab_list_df['time'] == current_time) & (lab_list_df['name'] == 'paO2')]

        # 某个时间点没有f，用f的均值补充缺少的项
        if len(f_in_time) == 0:
            new_f = {'id': temp_id, 'time': current_time, 'name': 'FiO2', 'value': fio2_temp_mean}
            lab_list_df = lab_list_df.append(pd.DataFrame(new_f))

        # 某个时间点没有p，用p的均值补充缺少的项
        if len(p_in_time) == 0:
            new_p = {'id': temp_id, 'time': current_time, 'name': 'paO2', 'value': pao2_temp_mean}
            lab_list_df = lab_list_df.append(pd.DataFrame(new_p))

        # 某个时间点有f，但值为空，用f的均值补充缺少的值
        if len(f_in_time) > 0:
            f_value = f_in_time.loc[f_in_time.index[0], 'value']
            if f_value is None or np.isnan(f_value) or f_value == 0:
                lab_list_df.loc[f_in_time.index[0], 'value'] = fio2_temp_mean

        # 某个时间点有p，但值为空，用p的均值补充缺少的值
        if len(p_in_time) > 0:
            p_value = p_in_time.loc[p_in_time.index[0], 'value']
            if p_value is None or np.isnan(p_value) or p_value == 0:
                lab_list_df.loc[p_in_time.index[0], 'value'] = pao2_temp_mean

    return lab_list_df


# 计算每次住院记录中的p/f的中位数，方差以及变化率
def compute_pf(data):
    # time name value
    data = sorted(data, key=lambda x: x[0])
    pao2_count = 0
    fio2_count = 0
    pao2 = 0
    fio2 = 0
    for item in data:
        name = item[1]
        value = item[2]
        # print('name : ' + str(name) + ' value : ' + value)
        if 'paO2' in name and value > 1e-6:
            pao2_count += 1
            pao2 += value
        if 'FiO2' in name and value > 1e-6:
            fio2_count += 1
            fio2 += value
    # 只有pao2或者只有fio2时，三项数值置0
    if pao2_count == 0 or fio2_count == 0:
        print('只有一个检查项')
        return None

    # 计算均值
    pao2_mean = pao2 / pao2_count
    fio2_mean = fio2 / fio2_count
    # 用均值填充缺省值
    for i in range(0, len(data)):
        if data[i][1] == 'paO2' and (data[i][2] is None or data[i][2] < 1e-6):
            data[i] = list(data[i])
            data[i][2] = pao2_mean
            data[i] = tuple(data[i])
        if data[i][1] == 'FiO2' and (data[i][2] is None or data[i][2] < 1e-6):
            data[i] = list(data[i])
            data[i][2] = fio2_mean
            data[i] = tuple(data[i])
    # 将空余位置的pao2或fio2值用后位置值替代
    i = 0
    while i < len(data) - 1:
        # 当前项与下一项是同一时刻的pao2和fio2
        if data[i][0] == data[i + 1][0] and data[i][1] != data[i + 1][1]:
            i += 1
        else:
            item_name = data[i][1]
            # 当前时刻pao2或fio2不存在时，使用对应前面项填充
            j = i - 1
            while j >= 0:
                if data[j][1] != item_name:
                    if data[i][0] != data[j][0]:
                        extra_item = list(data[j])
                        extra_item[0] = data[i][0]
                        extra_item = tuple(extra_item)
                        data.insert(i + 1, extra_item)
                    i = i + 1
                    break
                else:
                    j -= 1
        i += 1
    i = 0
    # 计算所有p/f值
    pao2_list = []
    fio2_list = []
    p_f = []
    while i < len(data) - 1:
        item_name = data[i][1]
        item_time = data[i][0]
        item_result = data[i][2]
        next_name = data[i + 1][1]
        next_time = data[i + 1][0]
        if item_name != next_name and item_time == next_time:
            if item_name == 'paO2':
                pao2 = data[i][2]
                pao2_list.append(('paO2', pao2, item_time))
                fio2 = data[i + 1][2]
                if fio2 > 1:
                    fio2 /= 100
                elif fio2 < 1e-6:
                    if fio2_mean > 1:
                        fio2_mean /= 100
                    fio2 = fio2_mean
                fio2_list.append(('FiO2', fio2, next_time))
                p_f.append(('P/F ratio', pao2 / fio2))
            else:
                fio2 = data[i][2]
                if fio2 > 1:
                    fio2 /= 100
                elif fio2 < 1e-6:
                    if fio2_mean > 1:
                        fio2_mean /= 100
                    fio2 = fio2_mean
                fio2_list.append(('FiO2', fio2, item_time))
                pao2 = data[i + 1][2]
                pao2_list.append(('paO2', pao2, next_time))
                p_f.append(('P/F ratio', pao2 / fio2))
            i += 2
        else:
            if item_name == 'paO2':
                pao2 = data[i][2]
                pao2_list.append(('paO2', pao2, item_time))
            else:
                fio2 = data[i][2]
                if fio2 > 1:
                    fio2 /= 100
                if fio2 < 1e-6:
                    if fio2_mean > 1:
                        fio2_mean /= 100
                        fio2 = fio2_mean
                fio2_list.append(('FiO2', fio2, item_time))
            i += 1
    # 将数值排序
    p_f.sort()
    return p_f


if __name__ == '__main__':

    df_list = []
    result = pd.read_csv('data/result_id.csv', sep=',')
    df = DataFrame(result)
    df_list = []

    for i in range(15):
        df_list.append(df['id'][i * 1000:i * 1000 + 1000])
    df_list.append(df['id'][15000:-1])
    for data_slice in df_list:
        name = str(data_slice.head(1).index.tolist()[0]) + '-' + str(data_slice.tail(1).index.tolist()[0])
        MyThread(data_slice, name).start()
