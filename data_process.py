from datetime import datetime

import pandas as pd
from pandas import DataFrame
import time

def filter_death_by_severity(result):
    icu_death = 0
    day28_death = 0
    hospital_death = 0
    icu_death_mild = 0
    icu_death_moderate = 0
    icu_death_severity = 0
    day28_death_mild = 0
    day28_death_moderate = 0
    day28_death_severity = 0
    hospital_death_mild = 0
    hospital_death_moderate = 0
    hospital_death_severity = 0
    death = 0
    sum = len(result)
    for i in range(1, sum):
        status = result.iloc[i]['status']
        severity = result.iloc[i]['P/F ratio_median']
        if status >= 2:
            death = death + 1
            if status == 2:
                icu_death = icu_death + 1
                if severity > 200 and severity <= 300:
                    icu_death_mild = icu_death_mild + 1
                elif severity > 100 and severity <= 200:
                    icu_death_moderate = icu_death_moderate + 1
                elif severity <= 100:
                    icu_death_severity = icu_death_severity + 1
            if status == 2 or status == 3:
                day28_death = day28_death + 1
                if severity > 200 and severity <= 300:
                    day28_death_mild = day28_death_mild + 1
                elif severity > 100 and severity <= 200:
                    day28_death_moderate = day28_death_moderate + 1
                elif severity <= 100:
                    day28_death_severity = day28_death_severity + 1
            if status <= 4:
                hospital_death = hospital_death + 1
                if severity > 200 and severity <= 300:
                    hospital_death_mild = hospital_death_mild + 1
                elif severity > 100 and severity <= 200:
                    hospital_death_moderate = hospital_death_moderate + 1
                elif severity <= 100:
                    hospital_death_severity = hospital_death_severity + 1

    dict = {}
    dict['icu_death'] = icu_death
    dict['icu_death_rate'] = icu_death / sum
    dict['28day_death'] = day28_death
    dict['28day_death_rate'] = day28_death / sum
    dict['hospital_death'] = hospital_death
    dict['hospital_death_rate'] = hospital_death / sum
    dict['icu_death_mild'] = icu_death_mild
    dict['icu_death_moderate'] = icu_death_moderate
    dict['icu_death_severity'] = icu_death_severity
    dict['icu_death_mild_rate'] = icu_death_mild / death
    dict['icu_death_moderate_rate'] = icu_death_moderate / death
    dict['icu_death_severity_rate'] = icu_death_severity / death
    dict['day28_death_mild'] = day28_death_mild
    dict['day28_death_moderate'] = day28_death_moderate
    dict['day28_death_severity'] = day28_death_severity
    dict['day28_death_mild_rate'] = day28_death_mild / death
    dict['day28_death_moderate_rate'] = day28_death_moderate / death
    dict['day28_death_severity_rate'] = day28_death_severity / death
    dict['hospital_death_mild'] = hospital_death_mild
    dict['hospital_death_moderate'] = hospital_death_moderate
    dict['hospital_death_severity'] = hospital_death_severity
    dict['hospital_death_mild_rate'] = hospital_death_mild / death
    dict['hospital_death_moderate_rate'] = hospital_death_moderate / death
    dict['hospital_death_severity_rate'] = hospital_death_severity / death
    return dict

if __name__=='__main__'():
    start = time.time()
    print('开始时间 ： ' + str(datetime.datetime.now()))
    # result = pd.read_csv('result_id.csv', sep=',')
    result = pd.read_csv('result.csv', sep=',')
    dict = filter_death_by_severity(result)
    end = time.time()
    total = (end - start) / 60
    print('结束时间 ： ' + str(datetime.datetime.now()) + '程序运行花费了' + str(time.time() - start) + '秒')
