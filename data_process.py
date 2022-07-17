from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import time

import query_sql


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
        severity = result.iloc[i]['severity']
        # 死亡患者
        if status >= 2:
            death += 1
            if status >= 2 and status <= 3:
                icu_death += 1
                if severity == 3:
                    icu_death_mild += 1
                elif severity == 2:
                    icu_death_moderate += 1
                elif severity == 1:
                    icu_death_severity += 1
            if status >= 2 and status <= 4:
                day28_death += 1
                if severity == 3:
                    day28_death_mild += 1
                elif severity == 2:
                    day28_death_moderate += 1
                elif severity == 1:
                    day28_death_severity += 1
            if status >= 2 and status <= 5:
                hospital_death += 1
                if severity == 3:
                    hospital_death_mild += 1
                elif severity == 2:
                    hospital_death_moderate += 1
                elif severity == 1:
                    hospital_death_severity += 1

    dict = {}
    dict['icu_death'] = icu_death
    dict['icu_death_rate'] = icu_death / sum
    dict['icu_death_mild'] = icu_death_mild
    dict['icu_death_moderate'] = icu_death_moderate
    dict['icu_death_severity'] = icu_death_severity
    dict['icu_death_mild_rate'] = icu_death_mild / death
    dict['icu_death_moderate_rate'] = icu_death_moderate / death
    dict['icu_death_severity_rate'] = icu_death_severity / death
    dict['28day_death'] = day28_death
    dict['28day_death_rate'] = day28_death / sum
    dict['day28_death_mild'] = day28_death_mild
    dict['day28_death_moderate'] = day28_death_moderate
    dict['day28_death_severity'] = day28_death_severity
    dict['day28_death_mild_rate'] = day28_death_mild / death
    dict['day28_death_moderate_rate'] = day28_death_moderate / death
    dict['day28_death_severity_rate'] = day28_death_severity / death
    dict['hospital_death'] = hospital_death
    dict['hospital_death_rate'] = hospital_death / sum
    dict['hospital_death_mild'] = hospital_death_mild
    dict['hospital_death_moderate'] = hospital_death_moderate
    dict['hospital_death_severity'] = hospital_death_severity
    dict['hospital_death_mild_rate'] = hospital_death_mild / death
    dict['hospital_death_moderate_rate'] = hospital_death_moderate / death
    dict['hospital_death_severity_rate'] = hospital_death_severity / death
    mild_rate = [dict['icu_death_mild_rate'], dict['day28_death_mild_rate'], dict['hospital_death_mild_rate']]
    moderate_rate = [dict['icu_death_moderate_rate'], dict['day28_death_moderate_rate'],
                     dict['hospital_death_moderate_rate']]
    severe_rate = [dict['icu_death_severity_rate'], dict['day28_death_severity_rate'],
                   dict['hospital_death_severity_rate']]
    name_list = ['ICU Mortality', '28-day Mortality', 'hospital Mortality']
    label_list = ['Mild', 'Moderate', 'Severe']
    x = list(range(len(name_list)))
    width = 0.3
    # 误差棒属性
    error_kw = {'ecolor': '0.2', 'capsize': 6}
    # 误差大小
    yerr = 0.005
    mild = [dict['icu_death_mild'], dict['day28_death_mild'], dict['hospital_death_mild']]
    moderate = [dict['icu_death_moderate'], dict['day28_death_moderate'],
                dict['hospital_death_moderate']]
    severe = [dict['icu_death_severity'], dict['day28_death_severity'], dict['hospital_death_severity']]
    plt.bar(x, mild_rate, width=width, yerr=yerr, label=label_list[0], error_kw=error_kw, tick_label=name_list,
            fc='r')
    for i, rate, num in zip(x, mild_rate, mild):
        plt.text(i, rate, '%.2f(n=%d)' % (rate, num), ha='center', va='bottom')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, moderate_rate, width=width, yerr=yerr, label=label_list[1], error_kw=error_kw, fc='g')
    for i, rate, num in zip(x, moderate_rate, moderate):
        plt.text(i, rate, '%.2f(n=%d)' % (rate, num), ha='center', va='center')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, severe_rate, width=width, yerr=yerr, label=label_list[2], error_kw=error_kw, fc='b')
    for i, rate, num in zip(x, severe_rate, severe):
        plt.text(i, rate, '%.2f(n=%d)' % (rate, num), ha='center', va='top')
    plt.ylabel('Mortality Rate')
    plt.ylim([-0.05, 0.5])
    plt.legend()
    plt.show()
    return dict


if __name__ == '__main__':
    start = time.time()
    # print('开始时间 ： ' + str(datetime.datetime.now()))
    # result = pd.read_csv('result_id.csv', sep=',')
    # result = pd.read_csv('result/fill_with_average.csv', sep=',')
    # dict = filter_death_by_severity(result)
    # print('analysis : ' + str(dict))
    # end = time.time()
    # total = (end - start) / 60
    result = np.array(pd.read_csv('result/peep_filter.csv', sep=','))
    outcome_list = []
    print('len is : ' + str(len(result)))
    for i in range(0, len(result)):
        result = {}
        id = result[i][0]
        enrollment = result[i][3]
        query = query_sql.Query()
        outcome, detail_outcome = query.access_outcome(id, enrollment)
        result['id'] = id
        result['status'] = outcome
        if 'Expired in ICU' in detail_outcome:
            result['detail'] = 'ICU Death'
        elif 'less than 28 days' in detail_outcome:
            result['detail'] = '28-day Death'
        elif 'Home' not in detail_outcome:
            result['detail'] = 'Hospital Death'
        outcome_list.append([result])
    dataFrame = DataFrame(outcome_list)
    dataFrame.to_csv('result/outcome.csv', mode='w', encoding='utf-8', index=False)
