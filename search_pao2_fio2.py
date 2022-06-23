import psycopg2
import pandas as pd
from functools import reduce

# 获得连接
import find_all_features

temp_path = './venv/'

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
# 获取数据.
# 使用年龄，呼吸衰竭，非心血管衰竭，peep筛选数据
# peepids = find_all_features.filter_with_peep(cursor)
# peep = find_all_features.list_to_csv(peepids, temp_path, 'peep.csv')
# ageids = find_all_features.filter_with_age(cursor)
# age = find_all_features.list_to_csv(ageids, temp_path, 'age.csv')
# respiratoryids = find_all_features.filter_with_respiratory_failure(cursor)
# respiratory = find_all_features.list_to_csv(respiratoryids, temp_path, 'respiratory.csv')
# congestiveids = find_all_features.filter_with_congestive_heart_failure(cursor)
# congestive = find_all_features.list_to_csv(congestiveids, temp_path, 'congestive.csv')
# # 多个dataframe取交集
# dfs = [peep, age, respiratory, congestive]
# temp = reduce(lambda x, y: pd.merge(x, y, how='inner'), dfs)
# # 将最终数据排序
# temp.to_csv(temp_path + 'temp.csv', encoding='utf-8')
# # 找到所有患者的pao2,fio2的值
# p_fs = find_all_features.filter_with_pao2_or_fio2(cursor)
# p_fs = find_all_features.list_to_csv(p_fs, temp_path, 'pf.csv')
# 找到所有动态数据
dynamic = find_all_features.filter_with_dynamic(cursor, 560047)
find_all_features.compute_dynamic_factor(dynamic, 560047)
print('done')
# 事物提交
conn.commit()


# 关闭数据库连接
conn.close()
