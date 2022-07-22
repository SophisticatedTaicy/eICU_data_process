from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import time
import seaborn as sns

from scipy import stats

import query_sql


def filter_death_by_severity(result):
    '''
    :param severities: ARDS严重程度，1 重度，2 中度，3 轻度，4未知（）
    :param details: 患者死亡状态，-1 患者活着，0 ICU死亡，1 28天内死亡，2 医院死亡
    :return:患者死亡情况展示
    '''
    icu_death = np.sum(result['detail'] == 0)
    day28_death = np.sum(result['detail'] == 1)
    hospital_death = np.sum(result['detail'] == 2)
    sum = result.shape[0]
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
    for severity, detail in zip(result['severity'], result['detail']):
        if detail == 0:
            if severity == 3:
                icu_death_mild += 1
            elif severity == 2:
                icu_death_moderate += 1
            elif severity == 1:
                icu_death_severity += 1
        if detail <= 1 and detail > -1:
            if severity == 3:
                day28_death_mild += 1
            elif severity == 2:
                day28_death_moderate += 1
            elif severity == 1:
                day28_death_severity += 1
        if detail <= 2 and detail > -1:
            if severity == 3:
                hospital_death_mild += 1
            elif severity == 2:
                hospital_death_moderate += 1
            elif severity == 1:
                hospital_death_severity += 1
        if detail > -1:
            death += 1
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
            fc='g')
    for i, rate, num in zip(x, mild_rate, mild):
        plt.text(i, rate, '%.2f(n=%d)' % (rate, num), ha='center', va='bottom')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, moderate_rate, width=width, yerr=yerr, label=label_list[1], error_kw=error_kw, fc='y')
    for i, rate, num in zip(x, moderate_rate, moderate):
        plt.text(i, rate, '%.2f(n=%d)' % (rate, num), ha='center', va='center')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, severe_rate, width=width, yerr=yerr, label=label_list[2], error_kw=error_kw, fc='r')
    for i, rate, num in zip(x, severe_rate, severe):
        plt.text(i, rate, '%.2f(n=%d)' % (rate, num), ha='center', va='top')
    plt.ylabel('Mortality Rate')
    plt.ylim([-0.05, 0.5])
    plt.legend()
    plt.show()
    return dict


def age_plot(ages):
    '''
    :param ages: 年龄数据
    :return:
    '''
    plt.style.use('ggplot')
    sns.distplot(ages, hist=False, kde=False, fit=stats.norm,
                 fit_kws={'color': 'black', 'linestyle': '-'})
    # 呈现图例
    plt.xlim([10, 100])
    plt.ylim([-0.002, 0.03])
    plt.xlabel('Age at admission')
    plt.ylabel('Estimated density')
    plt.legend()
    # 呈现图形
    plt.show()


def apacheIV_plot(apache):
    '''
        :param ages: 入院Apache分数
        :return:
        '''
    plt.style.use('ggplot')
    sns.distplot(apache, hist=False, kde=False, fit=stats.norm,
                 fit_kws={'color': 'black', 'linestyle': '-'})
    # 呈现图例
    plt.xlim([-5, 205])
    plt.ylim([-0.001, 0.015])
    plt.xlabel('APACHE IV score at admission')
    plt.ylabel('Estimated density')
    plt.legend()
    # 呈现图形
    plt.show()


def stay_boxplot(unit, hospital):
    '''
    :param unit:ICU住院天数
    :param hospital:医院住院天数
    :return:
    '''
    plt.style.use('ggplot')
    plt.boxplot((unit, hospital), labels=('ICU', 'Hospital'))
    # 呈现图例
    plt.ylim([-2, 32])
    plt.xlabel('Stay type')
    plt.ylabel('LOS(days)')
    plt.legend()
    # 呈现图形
    plt.show()


def death_plot(death):
    '''
    :param death:
    :return:
    '''
    plt.style.use('ggplot')
    death = np.array(death)
    print('death ' + str(death))
    ICU_death = np.sum(death == 0)
    day28_death = np.sum((death == 0) | (death == 1))
    Hospital_death = np.sum((death == 0) | (death == 1) | (death == 2))
    death = len(death)
    death_rate = [ICU_death / death, day28_death / death, Hospital_death / death]
    death_sum = [ICU_death, day28_death, Hospital_death]
    x = range(len(death_rate))
    # 误差棒属性
    error_kw = {'ecolor': '0.2', 'capsize': 6}
    # 误差大小
    yerr = 0.005
    name_list = ['ICU Mortality', '28-day Mortality', 'hospital Mortality']
    plt.bar(x, death_rate, width=0.5, yerr=yerr, error_kw=error_kw, tick_label=name_list, fc='g')
    for i, rate, num in zip(x, death_rate, death_sum):
        plt.text(i, rate, '%.2f(n=%d)' % (rate, num), ha='center', va='bottom')
    plt.ylim([-0.05, 0.5])
    plt.ylabel('Mortality rate')
    plt.legend()
    # 呈现图形
    plt.show()


if __name__ == '__main__':
    start = time.time()
    result = pd.read_csv('result/fill_with_average.csv')
    # outcome_list = []
    # severities = result['severity']
    details = np.array(result['detail'])
    ages = np.array(result['age'])
    # age_plot(ages)
    apache = np.array(result['admission_score'])
    # apacheIV_plot(apache)
    unit = np.array(result['unit'])
    hospital = np.array(result['hospital'])
    # stay_boxplot(unit, hospital)
    # death_plot(details)
