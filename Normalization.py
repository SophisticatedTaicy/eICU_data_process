import time
from datetime import datetime
import pandas as pd
from sklearn import preprocessing


def normalization(data):
    # 按照status进行排序，并将数值为2以上的数值变为2，status=0表示快速恢复患者，=1表示长期住院患者，=2表示死亡患者
    for index, row in data.iterrows():
        if row['status'] > 1:
            row['status'] = 2
            data.iloc[index] = row
    # 提取除id和status之外的数据
    feature_data = data.iloc[:, 1:-1]
    print(str(feature_data))
    # 计算原始数据每行和每列的均值和方差，feature_data是多维数据
    # print(str(feature_data.shape))
    scale = preprocessing.StandardScaler().fit(feature_data)
    # 计算出的均值
    mean = scale.mean_
    # 计算出的方差
    std = scale.scale_
    # print('mean : ' + str(mean) + ' std : ' + str(std))
    # 标准化数据
    data_normal = scale.transform(feature_data)
    # data_normal.to_csv('data_normal.csv', mode='w', encoding='utf-8')
    print('data normal : ' + str(data_normal))
    data_normal = pd.DataFrame(data_normal)
    data_normal.to_csv('data_normal.csv', encoding='utf-8')
    # data_normal = preprocessing.scale(item)
    # print(str(data_normal))


if __name__ == '__main__':
    start = time.time()
    print('开始时间 ： ' + str(datetime.now()))
    data_csv = pd.read_csv('data/result.csv', sep=',')
    # 'utf-8' codec can't decode byte 0xb0 in position 113182: invalid start byte 可能文件中含有中文或者其他不支持字符
    norm_data = normalization(data_csv)
    end = time.time()
    total = (end - start) / 60
    print('结束时间 ： ' + str(datetime.now()) + '程序运行花费了' + str(time.time() - start) + '秒')
