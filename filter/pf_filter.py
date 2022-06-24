import pandas as pd
import threading
import numpy as np
from postgresql import PostgreSQL

class MyThread(threading.Thread):
    def __init__(self, data, thread_name):
        # 注意：一定要显式的调用父类的初始化函数。
        super(MyThread, self).__init__(name=thread_name)
        self.data = data
        self.name = thread_name
        self.is_first = True

    def run(self):
        # 遍历dataframe self.data中包络500个id
        # title_pf = pd.DataFrame()
        # title_pf.to_csv(columns="")
        for idx, row in self.data.iterrows():
            path = "./file/valid_id/" + str(self.name) + ".csv"
            # 用均值填充缺少的值
            result_df = fill_with_pf(row['id'])
            if not result_df.empty:
                try:
                    is_valid = pf_value_filter(result_df)
                    # 将有效的id写入文件
                    if is_valid:
                        valid_id_df = pd.DataFrame([row['id']],columns=['id'])
                        # print(valid_id_df)
                        valid_id_df.to_csv(path, mode='a', header=self.is_first)
                        self.is_first = False
                except RuntimeWarning:
                    print('*--*--*--*--*--*')
                    print(row['id'])
                    continue




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
    lab_list = PostgreSQL().get_list_by_id(id)

    # 当查到的lab记录只有一条时，直接剔除
    if len(lab_list) <= 1:
        return pd.DataFrame()

    # 当查到的lab记录中只有P或只有F时候
    if len(lab_list) <= 1:
        return pd.DataFrame()

    # if len(lab_list) <= 1
    lab_list_df = pd.DataFrame(lab_list, columns=['id', 'time', 'labname', 'value'])

    # 查询到的记录中没有 FiO2 或 paO2
    if len(lab_list_df.loc[lab_list_df['labname'] == 'FiO2']) == 0:
        return pd.DataFrame()
    if len(lab_list_df.loc[lab_list_df['labname'] == 'paO2']) == 0:
        return pd.DataFrame()

    # 计算p和f的均值
    fio2_df = lab_list_df.loc[lab_list_df['labname'] == 'FiO2']
    fio2_temp_mean = round(fio2_df['value'].mean(), 1)
    pao2_df = lab_list_df.loc[lab_list_df['labname'] == 'paO2']
    pao2_temp_mean = round(pao2_df['value'].mean(), 1)

    if fio2_temp_mean==0 or pao2_temp_mean==0:
        return pd.DataFrame()

    if fio2_temp_mean == 0 or pao2_temp_mean == 0:
        print('------------------------')
        print(id)

    # 提取时间后去重
    time_list = lab_list_df['time'].unique()

    # 取出id，以便后续新建记录用
    temp_id = lab_list_df.head(1)['id']
    for current_time in time_list:
        f_in_time = lab_list_df[(lab_list_df['time'] == current_time) & (lab_list_df['labname'] == 'FiO2')]
        p_in_time = lab_list_df[(lab_list_df['time'] == current_time) & (lab_list_df['labname'] == 'paO2')]

        # 某个时间点没有f，用f的均值补充缺少的项
        if len(f_in_time) == 0:
            new_f = {'id': temp_id, 'time': current_time, 'labname': 'FiO2', 'value': fio2_temp_mean}
            lab_list_df = lab_list_df.append(pd.DataFrame(new_f))

        # 某个时间点没有p，用p的均值补充缺少的项
        if len(p_in_time) == 0:
            new_p = {'id': temp_id, 'time': current_time, 'labname': 'paO2', 'value': pao2_temp_mean}
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

    # print(lab_list_df)
    return lab_list_df



if __name__ == '__main__':
    distinct_id_in_lab_df = pd.read_csv('./file/distinct_id_in_lab_df.csv', sep=',')
    df = pd.DataFrame(distinct_id_in_lab_df)
    # print(df.shape)

    df_list = []
    # 切片，每个数据切片有500条数据，一共171个线程
    # for i in range(170):
    #     df_list.append(df[i*500:i*500+500])
    # df_list.append(df[85000:85266])
    # print(len(df_list))

    for i in range(42):
        df_list.append(df[i * 2000:i * 2000 + 2000])
    df_list.append(df[84000:85266])

    # print(len(df_list))
    for data_slice in df_list:
        name = str(data_slice.head(1).index.tolist()[0]) + '-' + str(data_slice.tail(1).index.tolist()[0])
        MyThread(data_slice, name).start()
        # print(name)

    # 测试
    # MyThread(df_list[0], 'test').start()
    # if not fill_with_pf(152801):
    #     print('success')


