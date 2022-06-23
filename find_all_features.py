import csv
import os
import pandas as pd
from numpy.core.defchararray import lower
from pandas.core.frame import DataFrame
import numpy as np
from decimal import Decimal

dynamic_name_list = ['Albumin', 'ALT (SGPT)', 'AST (SGOT)', '-bands', 'Base Excess', '-basos', 'bicarbonate',
                     'total bilirubin', 'BUN', 'calcium', 'Total CO2', 'creatinine', '-eos', 'FiO2', 'glucose',
                     'Hemoglobin', 'PT - INR', 'ionized calcium', 'lactate', 'magnesium', 'paCO2', 'PaO2', 'P/F ratio',
                     'PEEP', 'pH', 'platelets x 1000', 'potassium', 'PTT', 'sodium', 'Temperature', 'WBC x 1000',
                     'Mean airway pressure', 'Plateau Pressure', 'SaO2', 'Tidal Volume (set)', 'cvp', 'etCO2',
                     'padiastolic', 'pamean', 'pasystolic', 'Eyes', 'GCS Total', 'Motor',
                     'Verbal', 'Heart Rate', 'Invasive BP Diastolic', 'Invasive BP Mean',
                     'Invasive BP Systolic', 'Non-Invasive BP Diastolic', 'Non-Invasive BP Mean',
                     'Non-Invasive BP Systolic', 'Respiratory rate', 'PIP']


def searchFile(dictionary, fileType):
    fileList = []
    for root, dirs, files in os.walk(dictionary):
        for file in files:
            if os.path.splitext(file)[1] == fileType:
                print(file)
                file = os.path.join(root, file)
                columns = pd.read_csv(file, low_memory=False, error_bad_lines=False)
                for col in columns:
                    if col not in fileList:
                        fileList.append(col)
    return fileList


# filter data with respiratory failure
def filter_with_respiratory_failure(cursor):
    sql = " select distinct diag.patientunitstayid" \
          " from diagnosis as diag" \
          " where diag.diagnosisstring like '%respiratory failure%'" \
          " union " \
          " select dx.patientunitstayid" \
          " from admissiondx as dx" \
          " where dx.admitdxpath like '%respiratory failure%'"
    cursor.execute(sql)
    list = cursor.fetchall()
    data = DataFrame(list)
    data.to_csv('respiratory_failure.csv', encoding='utf-8')
    return data


# filter data missing congestive heart failure
def filter_with_congestive_heart_failure(cursor):
    sql = " select distinct diag.patientunitstayid" \
          " from diagnosis as diag" \
          " where diag.diagnosisstring not like '%congestive heart failure%'" \
          " union " \
          " select dx.patientunitstayid" \
          " from admissiondx as dx" \
          " where dx.admitdxpath not like '%congestive heart failure%'"
    cursor.execute(sql)
    list = cursor.fetchall()
    data = DataFrame(list)
    # data.to_csv('congestive_heart_failure.csv', encoding='utf-8')
    return data


# filter data with peep
def filter_with_peep(cursor):
    sql = " select distinct l1.patientunitstayid" \
          " from lab as l1" \
          " where position('peep' in lower(l1.labname)) > 0" \
          " and l1.labresult >= 5" \
          " union" \
          " select distinct rc.patientunitstayid" \
          " from respiratorycharting as rc" \
          " where position('peep' in lower(rc.respchartvaluelabel)) > 0" \
          " and rc.respchartvalue >= '5'" \
          " union" \
          " select distinct pe.patientunitstayid" \
          " from physicalexam as pe" \
          " where position('peep' in lower(pe.physicalexamvalue)) > 0" \
          " and pe.physicalexamtext > '5'"
    cursor.execute(sql)
    list = cursor.fetchall()
    data = DataFrame(list)
    # data.to_csv('peep.csv', encoding='utf-8')
    return data


# filter data with age
def filter_with_age(cursor):
    sql = " select distinct patientunitstayid" \
          " from patient as pa" \
          " where pa.age >= '18'"
    cursor.execute(sql)
    list = cursor.fetchall()
    data = DataFrame(list)
    # data.to_csv('age.csv', encoding='utf-8')
    return data


# change list data to csv file
def list_to_csv(list, filename):
    # list转换为DataFrame
    dataframe = DataFrame(list)
    # DataFrame转换为csv
    # dataframe.to_csv(filename, encoding='utf-8')
    return dataframe


# find pao2 and fio2 of all unit stay records
def filter_with_pao2_or_fio2(cursor):
    # 筛选出所有患者记录中的pao2,fio2,并依次按照住院记录id和时间排序
    sql = " select lb.patientunitstayid, lb.labresultoffset as time, lb.labname as name, lb.labresult as result" \
          " from lab as lb" \
          " where (position('pao2' in lower(lb.labname)) > 0" \
          " or position('fio2' in lower(lb.labname)) > 0)" \
          " and lb.labresultoffset > 0" \
          " union" \
          " select rc.patientunitstayid," \
          " rc.respchartoffset                 as time," \
          " rc.respchartvaluelabel             as name," \
          " cast(rc.respchartvalue as numeric) as result" \
          " from respiratorycharting as rc" \
          " where position('fio2' in lower(rc.respchartvaluelabel)) > 0" \
          " and position('%' in rc.respchartvalue) = 0" \
          " and rc.respchartoffset > 0" \
          " union" \
          " select pe.patientunitstayid," \
          " pe.physicalexamoffset                as time," \
          " pe.physicalexamvalue                 as name," \
          " cast(pe.physicalexamtext as numeric) as result" \
          " from physicalexam as pe" \
          " where position('fio2' in lower(pe.physicalexamvalue)) > 0" \
          " and pe.physicalexamoffset > 0" \
          " order by patientunitstayid asc, time asc, name asc;"
    cursor.execute(sql)
    list = cursor.fetchall()
    data = DataFrame(list)
    # data.to_csv('p_f.csv', encoding='utf-8')
    return data


# todo::find out details for ALP,GCS(intub),GCS(unable),Hematocrit and Spo2
# find all dynamic data for each unit stay record
def filter_with_dynamic(cursor, unitid):
    # find dynamic item in lab table
    sql = " select labname as name,  labresult as result,labresultoffset as time" \
          " from lab as lb" \
          " where lb.patientunitstayid=" + str(unitid) + \
          " and labresultoffset > 0" \
          " and (lb.labresult > 0 and (" \
          "    lb.labname = 'Albumin' or" \
          "    lb.labname='ALT (SGPT)'        or" \
          "    lb.labname='AST (SGOT)'        or" \
          "    lb.labname='-bands'         or" \
          "    lb.labname='Base Excess'        or" \
          "    lb.labname='-basos'        or" \
          "    lb.labname='bicarbonate'        or" \
          "    lb.labname='HCO3'        or" \
          "    lb.labname='total bilirubin'        or" \
          "    lb.labname='BUN'      or " \
          "    lb.labname='calcium'        or" \
          "    lb.labname='Total CO2'        or" \
          "    lb.labname='creatinine'       or" \
          "    lb.labname='-eos'        or" \
          "    lb.labname='FiO2'         or" \
          "    lb.labname='glucose'      or " \
          "    lb.labname='bedside glucose'        or" \
          "    lb.labname='Carboxyhemoglobin'        or" \
          "    lb.labname='Methemoglobin'        or" \
          "    lb.labname='ionized calcium'        or" \
          "    lb.labname='lactate'        or" \
          "    lb.labname='magnesium'     or  " \
          "    lb.labname='paCO2'        or" \
          "    lb.labname='PaO2'        or" \
          "    lb.labname='PEEP'        or" \
          "    lb.labname='pH'        or" \
          "    lb.labname='platelets x 1000'       or" \
          "    lb.labname='potassium'          or" \
          "    lb.labname='PTT'         or" \
          "    lb.labname='sodium'        or" \
          "    lb.labname='Temperature'         or" \
          "    lb.labname='WBC x 1000'        " \
          "    ))" \
          " and lb.labresult <=1440;"
    cursor.execute(sql)
    lab_list = cursor.fetchall()

    # find dynamic item in customlab table
    sql = " select cl.labothername as name,  cast(cl.labotherresult as numeric) as result,cl.labotheroffset as time" \
          " from customlab as cl" \
          " where cl.patientunitstayid = " + str(unitid) + \
          " and cl.labotheroffset > 0 " \
          " and cl.labotheroffset <1440 " \
          " and cl.labotherresult > '0'" \
          " and cl.labothername = 'PIP'"
    cursor.execute(sql)
    customlab_list = cursor.fetchall()

    # find dynamic item in nursecharting table
    sql = " select nc.nursingchartcelltypevalname        as name," \
          "        cast(nc.nursingchartvalue as numeric) as result," \
          "        nc.nursingchartoffset as time " \
          " from nursecharting as nc" \
          " where nc.patientunitstayid =" + str(unitid) + \
          "  and nursingchartoffset > 0" \
          "  and nursingchartoffset < 1440" \
          "  and (nc.nursingchartvalue > '0' and (" \
          "   nc.nursingchartcelltypevalname='Eyes'     or" \
          "   nc.nursingchartcelltypevalname='GCS Total'     or" \
          "   nc.nursingchartcelltypevalname='Motor'      or" \
          "   nc.nursingchartcelltypevalname='Verbal'      or" \
          "   nc.nursingchartcelltypevalname='Heart Rate'     or" \
          "   nc.nursingchartcelltypevalname='Non-Invasive BP Diastolic'     or" \
          "   nc.nursingchartcelltypevalname='Non-Invasive BP Mean'     or" \
          "   nc.nursingchartcelltypevalname='Non-Invasive BP Systolic'     or" \
          "   nc.nursingchartcelltypevalname='Invasive BP Diastolic'     or" \
          "   nc.nursingchartcelltypevalname='Invasive BP Mean'     or" \
          "   nc.nursingchartcelltypevalname='Invasive BP Systolic'     or" \
          "   nc.nursingchartcelltypevalname='respiratory rate'    " \
          "    ))"
    cursor.execute(sql)
    nurse_list = cursor.fetchall()

    # find dynamic item in respiratorycharting table
    sql = " select rc.respchartvaluelabel as name, cast(rc.respchartvalue as numeric) as result,rc.respchartoffset as time " \
          " from respiratorycharting as rc" \
          " where rc.patientunitstayid = " + str(unitid) + \
          "  and respchartoffset > 0" \
          "  and respchartoffset < 1440 " \
          "  and (rc.respchartvalue > '0')" \
          "  and (" \
          "   rc.respchartvaluelabel='Mean airway pressure'          or" \
          "   rc.respchartvaluelabel='SaO2'          or" \
          "   rc.respchartvaluelabel='Tidal Volume (set)'          or" \
          "   rc.respchartvaluelabel='Plateau Pressure' )" \
          "order by name asc, result asc;"
    cursor.execute(sql)
    respchart_list = cursor.fetchall()

    # find dynamic item in vitalperiodic table where columns like 'pa%'
    sql = " select case when vp.padiastolic > 0 then vp.padiastolic else 0 end ," \
          " case when vp.pasystolic > 0 then vp.pasystolic else 0 end ," \
          " case when vp.pamean > 0 then vp.pamean else 0 end , vp.observationoffset, " \
          " vp.observationoffset                                        as time" \
          " from vitalperiodic as vp" \
          " where vp.patientunitstayid =  " + str(unitid) + \
          " and vp.observationoffset > 0" \
          " and vp.observationoffset<1440 ;"
    cursor.execute(sql)
    vital = cursor.fetchall()

    vital_list = []
    # format pa% item
    for item in vital:
        if item[0] > 0:
            vital_list.append(['padiastolic', item[0], item[-1]])
        if item[1] > 0:
            vital_list.append(['pasystolic', item[1], item[-1]])
        if item[2] > 0:
            vital_list.append(['pamean', item[2], item[-1]])
    # combine all dynamic item
    result_list = lab_list + customlab_list + nurse_list + respchart_list + vital_list
    data = DataFrame(result_list)
    return data


# todo::compute p/f
# compute median，variances and change rate for all dynamic items
def compute_dynamic_factor(data, header):
    # add head to data
    data.columns = ['name', 'result', 'time']
    p_f = []
    # 对于多名称特征数据进行更名
    for index, row in data.iterrows():
        if "Carboxyhemoglobin" in row['name']:
            row['name'] = 'Hemoglobin'
        elif "Methemoglobin" in row['name']:
            row['name'] = 'Hemoglobin'
        elif "HCO3" in row['name']:
            row['name'] = 'bicarbonate'
        elif 'bedside glucose' in row['name']:
            row['name'] = 'glucose'
        else:
            # 取出所有pao2和fio2数据
            if 'PaO2' in lower(row['name']):
                item = [row['time'], row['name'], row['result']]
                p_f.append(item)
            if 'FiO2' in lower(row['name']):
                item = [row['time'], row['name'], row['result']]
                p_f.append(item)
        # dataframe修改数据需要添加下面这句，否则修改不成功
        data.iloc[index] = row
    # 将处理后的数据进行排序，有利于后续中位数和方差的计算
    data.sort_values(by='name', inplace=True, ascending=True)
    result_list = {}
    # data.to_csv('dynamic.csv', encoding='utf-8')
    i = 0
    # 动态数据中位数，方差，变化率的计算
    while i < len(data):
        item_name = data.iloc[i, 0]
        item_list = []
        item_list.append(data.iloc[i, 1])
        # 汇总每种变量的所有信息
        j = i + 1
        # 取出同一特征item_name的所有数据存储在列表item_list中
        while j < len(data):
            if data.iloc[j, 0] == item_name:
                item_list.append(data.iloc[j, 1])
                j += 1
            else:
                break
        # 当数据只有一条时，其中位数、方差、变化率直接可以得出
        if j - i == 1:
            median = data.iloc[i, 1]
            variance = 0
            change_rate = 0
        # 计算中位数，方差，变化率
        else:
            if len(item_list) % 2 == 0:
                median = (item_list[int(len(item_list) / 2) - 1] + item_list[int(len(item_list) / 2)]) / 2
            else:
                median = item_list[int(len(item_list) / 2)]
            variance = np.var(item_list)
            change_rate = (np.max(item_list) - np.min(item_list)) / np.min(item_list)
        # 数据取小数点后两位
        result_list[item_name + '_median'] = round(median, 2)
        result_list[item_name + '_variances'] = round(variance, 2)
        result_list[item_name + '_changerate'] = round(change_rate, 2)
        i = j
        # 排除已经计算的特征
        if item_name in dynamic_name_list:
            dynamic_name_list.remove(item_name)
    # 将未计算或者不存在的特征中位数、方差、变化率置0
    for item in dynamic_name_list:
        result_list[item + '_median'] = 0
        result_list[item + '_variances'] = 0
        result_list[item + '_changerate'] = 0
    # 更新动态数据数值
    for key, value in result_list.items():
        header[key] = value
    return header


# todo:长期住院和自发恢复待进一步确定
# todo::患者入院来源，性别数据由字符串转换为数值
# 找到住院记录对应的静态数据项， 药物使用（10）+疾病诊断（25）+入院来源+年龄+性别+最终结果（存活，死亡）+BMI+入院评分=41个静态特征
def find_static_data(cursor, unitid, header):
    sql = " select case when position('warfarin' in lower(me.drugname)) > 0 then 1 else 0 end                  as warfarin," \
          " case when position('dobutamine' in lower(infu.drugname)) > 0 then 1  else 0 end             as dobutamine," \
          " case when position('dopamine' in lower(infu.drugname)) > 0 then 1 else 0 end                as Dopamine," \
          " case when position('epinephrine ' in lower(infu.drugname)) > 0 then 1   else 0 end          as epinephrine," \
          " case when position('heparin' in lower(infu.drugname)) > 0 then 1 else 0 end                 as Heparin," \
          " case when position('milrinone' in lower(infu.drugname)) > 0 then 1   else 0 end             as Milrinone," \
          " case when position('norepinephrine' in lower(infu.drugname)) > 0 then 1   else 0 end        as Norepinephrine," \
          " case when position('phenylephrine' in lower(infu.drugname)) > 0 then 1   else 0 end         as phenylephrine," \
          " case when position('vasopressin' in lower(infu.drugname)) > 0 then 1    else 0 end          as vasopressin," \
          " case when position('vasopressor' in lower(tr.treatmentstring)) > 0 then 1  else 0 end       as Vasopressor," \
          " case when position('acute coronary syndrome' in lower(di.diagnosisstring)) > 0 then 1 else 0 end              as Acute_Coronary_Syndrome_diagnosis," \
          " case when position('acute myocardial infarction' in lower(di.diagnosisstring)) > 0 then 1    else 0 end       as Acute_Myocardial_Infarction," \
          " case when position('acute renal failure' in lower(di.diagnosisstring)) > 0 then 1    else 0 end               as Acute_Renal_Failure," \
          " case when position('arrhythmia' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                        as Arrhythmia," \
          " case when position('asthma' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                            as Asthma_Emphysema," \
          " case when position('cancer' in lower(di.diagnosisstring)) > 0 then 1 else 0 end                               as Cancer," \
          " case when position('cardiac arrest' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                    as Cardiac_Arrest," \
          " case when position('shock' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                             as Cardiogenic_Shock," \
          " case when position('cardiovascular' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                    as Cardiovascular_Medical," \
          " case when position('pericardial' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                       as Cardiovascular_Other," \
          " case when position('stroke' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                            as Cerebrovascular_Accident_Stroke," \
          " case when position('chest pain' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                        as Chest_Pain_Unknown_Origin," \
          " case when position('endocrine' in lower(di.diagnosisstring)) > 0 then 1 else 0 end                            as Coma," \
          " case when position('cabg' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                              as Coronary_Artery_Bypass_Graft," \
          " case when position('ketoacidosis' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                      as Diabetic_Ketoacidosis," \
          " case when position('bleeding' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                          as Gastrointestinal_Bleed," \
          " case when position('gi obstruction' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                    as Gastrointestinal_Obstruction," \
          " case when position('neurologic' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                        as Neurologic," \
          " case when position('overdose' in lower(di.diagnosisstring)) > 0 then 1 else 0 end                             as Overdose," \
          " case when position('pneumonia' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                         as Pneumonia," \
          " case when position('respiratory' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                       as Respiratory_Medical_Other," \
          " case when position('sepsis' in lower(di.diagnosisstring)) > 0 then 1 else 0 end                               as Sepsis," \
          " case when position('thoracotomy' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                       as Thoracotomy," \
          " case when position('trauma' in lower(di.diagnosisstring)) > 0 then 1 else 0 end                               as Trauma," \
          " case when position('valve' in lower(di.diagnosisstring)) > 0 then 1    else 0 end                             as Valve_Disease," \
          " pa.hospitaladmitsource                                                                                        as admitsource," \
          " pa.age," \
          " pa.gender," \
          " pa.admissionweight / pow(pa.admissionheight / 100, 2)                                                          as BMI, " \
          " pa.hospitaldischargestatus                                                                                     as status," \
          " aps.apachescore                                                                                                as admission_score" \
          " from patient as pa" \
          "          left join medication as me on me.patientunitstayid = pa.patientunitstayid" \
          "          left join infusiondrug as infu on infu.patientunitstayid = pa.patientunitstayid" \
          "          left join treatment as tr on tr.patientunitstayid = pa.patientunitstayid" \
          "          left join diagnosis as di on di.patientunitstayid = pa.patientunitstayid" \
          "          left join apachepatientresult as aps on aps.patientunitstayid = pa.patientunitstayid" \
          "          left join lab as lb on lb.patientunitstayid = pa.patientunitstayid " \
          " where pa.patientunitstayid = " + str(unitid) + \
          " order by pa.hospitaladmittime24 asc ; "
    cursor.execute(sql)
    list = cursor.fetchone()
    if list is None:
        return;
    # 所有静态数据的特征名
    label = ['warfarin', 'dobutamine', 'Dopamine', 'epinephrine', 'Heparin', 'Milrinone', 'Norepinephrine',
             'phenylephrine',
             'vasopressin', 'Vasopressor', 'Acute_Coronary_Syndrome_diagnosis', 'Acute_Myocardial_Infarction',
             'Acute_Renal_Failure', 'Arrhythmia', 'Asthma_Emphysema', 'Cancer', 'Cardiac_Arrest', 'Cardiogenic_Shock',
             'Cardiovascular_Medical', 'Cardiovascular_Other', 'Cerebrovascular_Accident_Stroke',
             'Chest_Pain_Unknown_Origin',
             'Coma', 'Coronary_Artery_Bypass_Graft', 'Diabetic_Ketoacidosis', 'Gastrointestinal_Bleed',
             'Gastrointestinal_Obstruction',
             'Neurologic', 'Overdose', 'Pneumonia', 'Respiratory_Medical_Other', 'Sepsis', 'Thoracotomy', 'Trauma',
             'Valve_Disease', 'admitsource', 'age', 'gender', 'BMI', 'status', 'admission_score']
    # 将住院记录静态数据信息更新
    for i in range(len(label)):
        header[label[i]] = list[i]
    return header


def combine_feature(cursor, unitid):
    # 初始化特征数据为0，一共200维数据，其中
    # 静态数据维度41维，分别为10种药物使用情况、25种疾病患否情况、患者入院来源、患者年龄、患者性别、患者BMI身体健康素质指数、患者最终结果（存活，死亡）以及患者入院Apache评分，
    # 动态数据199维，包含数据信息统计表中53项动态生物特征的中位数、方差和变化率共159项数据、ALP、GCS(intub)、GCS(unable)、Hematocrit以及SpO2五项尚未找到
    # 先对200维数据初始化为0
    header = {'warfarin': 0, 'dobutamine': 0, 'Dopamine': 0, 'epinephrine': 0, 'Heparin': 0, 'Milrinone': 0,
              'Norepinephrine': 0,
              'phenylephrine': 0, 'vasopressin': 0, 'Vasopressor': 0, 'Acute_Coronary_Syndrome_diagnosis': 0,
              'Acute_Myocardial_Infarction': 0, 'Acute_Renal_Failure': 0, 'Arrhythmia': 0, 'Asthma_Emphysema': 0,
              'Cancer': 0,
              'Cardiac_Arrest': 0, 'Cardiogenic_Shock': 0, 'Cardiovascular_Medical': 0, 'Cardiovascular_Other': 0,
              'Cerebrovascular_Accident_Stroke': 0, 'Chest_Pain_Unknown_Origin': 0, 'Coma': 0,
              'Coronary_Artery_Bypass_Graft': 0,
              'Diabetic_Ketoacidosis': 0, 'Gastrointestinal_Bleed': 0, 'Gastrointestinal_Obstruction': 0,
              'Neurologic': 0,
              'Overdose': 0, 'Pneumonia': 0, 'Respiratory_Medical_Other': 0, 'Sepsis': 0, 'Thoracotomy': 0, 'Trauma': 0,
              'Valve_Disease': 0, 'admitsource': 0, 'age': 0, 'gender': 0, 'BMI': 0, 'status': 0, 'admission_score': 0,
              'Albumin_median': 0, 'ALT (SGPT)_median': 0, 'AST (SGOT)_median': 0, '-bands_median': 0,
              'Base Excess_median': 0, '-basos_median': 0, 'bicarbonate_median': 0, 'total bilirubin_median': 0,
              'BUN_median': 0, 'calcium_median': 0, 'Total CO2_median': 0, 'creatinine_median': 0, '-eos_median': 0,
              'FiO2_median': 0, 'glucose_median': 0, 'Hemoglobin_median': 0, 'PT - INR_median': 0,
              'ionized calcium_median': 0, 'lactate_median': 0, 'magnesium_median': 0, 'paCO2_median': 0,
              'PaO2_median': 0,
              'P/F ratio_median': 0, 'PEEP_median': 0, 'pH_median': 0, 'platelets x 1000_median': 0,
              'potassium_median': 0,
              'PTT_median': 0, 'sodium_median': 0, 'Temperature_median': 0, 'WBC x 1000_median': 0,
              'Mean airway pressure_median': 0, 'Plateau Pressure_median': 0, 'SaO2_median': 0,
              'Tidal Volume (set)_median': 0, 'cvp_median': 0, 'etCO2_median': 0, 'padiastolic_median': 0,
              'pamean_median': 0, 'pasystolic_median': 0, 'Eyes_median': 0, 'GCS Total_median': 0, 'Motor_median': 0,
              'Verbal_median': 0, 'Heart Rate_median': 0, 'Invasive BP Diastolic_median': 0,
              'Invasive BP Mean_median': 0,
              'Invasive BP Systolic_median': 0, 'Non-Invasive BP Diastolic_median': 0, 'Non-Invasive BP Mean_median': 0,
              'Non-Invasive BP Systolic_median': 0, 'Respiratory rate_median': 0, 'PIP_median': 0,
              'Albumin_variances': 0,
              'ALT (SGPT)_variances': 0, 'AST (SGOT)_variances': 0, '-bands_variances': 0, 'Base Excess_variances': 0,
              '-basos_variances': 0, 'bicarbonate_variances': 0, 'total bilirubin_variances': 0, 'BUN_variances': 0,
              'calcium_variances': 0, 'Total CO2_variances': 0, 'creatinine_variances': 0, '-eos_variances': 0,
              'FiO2_variances': 0, 'glucose_variances': 0, 'Hemoglobin_variances': 0, 'PT - INR_variances': 0,
              'ionized calcium_variances': 0, 'lactate_variances': 0, 'magnesium_variances': 0, 'paCO2_variances': 0,
              'PaO2_variances': 0, 'P/F ratio_variances': 0, 'PEEP_variances': 0, 'pH_variances': 0,
              'platelets x 1000_variances': 0, 'potassium_variances': 0, 'PTT_variances': 0, 'sodium_variances': 0,
              'Temperature_variances': 0, 'WBC x 1000_variances': 0, 'Mean airway pressure_variances': 0,
              'Plateau Pressure_variances': 0, 'SaO2_variances': 0, 'Tidal Volume (set)_variances': 0,
              'cvp_variances': 0,
              'etCO2_variances': 0, 'padiastolic_variances': 0, 'pamean_variances': 0, 'pasystolic_variances': 0,
              'Eyes_variances': 0, 'GCS Total_variances': 0, 'Motor_variances': 0, 'Verbal_variances': 0,
              'Heart Rate_variances': 0, 'Invasive BP Diastolic_variances': 0, 'Invasive BP Mean_variances': 0,
              'Invasive BP Systolic_variances': 0, 'Non-Invasive BP Diastolic_variances': 0,
              'Non-Invasive BP Mean_variances': 0, 'Non-Invasive BP Systolic_variances': 0,
              'Respiratory rate_variances': 0,
              'PIP_variances': 0, 'Albumin_changerate': 0, 'ALT (SGPT)_changerate': 0, 'AST (SGOT)_changerate': 0,
              '-bands_changerate': 0, 'Base Excess_changerate': 0, '-basos_changerate': 0, 'bicarbonate_changerate': 0,
              'total bilirubin_changerate': 0, 'BUN_changerate': 0, 'calcium_changerate': 0, 'Total CO2_changerate': 0,
              'creatinine_changerate': 0, '-eos_changerate': 0, 'FiO2_changerate': 0, 'glucose_changerate': 0,
              'Hemoglobin_changerate': 0, 'PT - INR_changerate': 0, 'ionized calcium_changerate': 0,
              'lactate_changerate': 0, 'magnesium_changerate': 0, 'paCO2_changerate': 0, 'PaO2_changerate': 0,
              'P/F ratio_changerate': 0, 'PEEP_changerate': 0, 'pH_changerate': 0, 'platelets x 1000_changerate': 0,
              'potassium_changerate': 0, 'PTT_changerate': 0, 'sodium_changerate': 0, 'Temperature_changerate': 0,
              'WBC x 1000_changerate': 0, 'Mean airway pressure_changerate': 0, 'Plateau Pressure_changerate': 0,
              'SaO2_changerate': 0, 'Tidal Volume (set)_changerate': 0, 'cvp_changerate': 0, 'etCO2_changerate': 0,
              'padiastolic_changerate': 0, 'pamean_changerate': 0, 'pasystolic_changerate': 0, 'Eyes_changerate': 0,
              'GCS Total_changerate': 0, 'Motor_changerate': 0, 'Verbal_changerate': 0, 'Heart Rate_changerate': 0,
              'Invasive BP Diastolic_changerate': 0, 'Invasive BP Mean_changerate': 0,
              'Invasive BP Systolic_changerate': 0,
              'Non-Invasive BP Diastolic_changerate': 0, 'Non-Invasive BP Mean_changerate': 0,
              'Non-Invasive BP Systolic_changerate': 0, 'Respiratory rate_changerate': 0, 'PIP_changerate': 0}
    # 查询相关住院记录静态数据
    header = find_static_data(cursor, unitid, header)
    # 查询相关住院记录动态数据
    dynamic = filter_with_dynamic(cursor, unitid)
    # 计算动态数据的中位数、方差以及变化率
    dynamic_dict = compute_dynamic_factor(dynamic, header)
    # 最终住院记录数据转换为dataframe
    data = DataFrame([dynamic_dict])
    # 数据追加写入文件
    data.to_csv('result.csv', mode='a', header=False)
    print(str(unitid) + '数据写入成功！')
