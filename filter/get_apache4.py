import pandas as pd
from postgresql import PostgreSQL
import threading
from tqdm import tqdm

class MyThread(threading.Thread):
    def __init__(self, data, thread_name):
        # 注意：一定要显式的调用父类的初始化函数。
        super(MyThread, self).__init__(name=thread_name)
        self.data = data
        self.name = thread_name
        self.is_first = True

    def run(self):
        for id in tqdm(self.data):
            # print(id)
            path = "./file/valid_id/" + str(self.name) + ".csv"
            apache4_list = []
            apache4 = PostgreSQL().get_apache4_by_id(id)
            # apache4_list.append(PostgreSQL().get_apache4_by_id(id))
            if len(apache4) < 1:
                continue
                # apache4_list.append({'id': id, 'apache4': 0})
            if len(apache4) >= 1:
                if apache4[0][0] == -1:
                    continue
                else:
                    apache4_list.append({'id': id, 'apache4': apache4[0][0]})
            apache4_df = pd.DataFrame(apache4_list)
            apache4_df.to_csv('././file/result/apache4.csv',mode='a', columns=['id', 'apache4'], header=False)
            self.is_first = False


if __name__ == '__main__':

    # 经过pf条件后的Id
    valid_id = pd.read_csv('././file/result/result_id.csv')
    # 数据切片
    valid_id_group = []
    for i in range(10):
        valid_id_group.append(valid_id['id'][i * 1500:i * 1500 + 1500])
    valid_id_group.append(valid_id['id'][15000:15407])

    # 多线程
    for data_slice in tqdm(valid_id_group):
        name = str(data_slice.head(1).index.tolist()[0]) + '-' + str(data_slice.tail(1).index.tolist()[0])
        MyThread(data_slice, name).start()
        # print(name)





