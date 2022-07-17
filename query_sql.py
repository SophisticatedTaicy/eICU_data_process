import csv
import os
from filecmp import cmp

import pandas as pd
from numpy.core.defchararray import lower
from pandas.core.frame import DataFrame
import numpy as np
from decimal import Decimal
import psycopg2
from psycopg2._psycopg import cursor

import pf_filter
from filter.param import DATABASE, USER, PASSWORD, HOST, PORT, SEARCH_PATH, dynamic_list, result_header


class Query:
    def __init__(self):
        self.database = DATABASE
        self.user = USER
        self.password = PASSWORD
        self.host = HOST
        self.port = PORT
        self.search_path = SEARCH_PATH
        self.connection = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port)

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

    def filter_with_respiratory_failure_without_congestive_heart_failure(self):
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        sql = " select di.patientunitstayid" \
              "      from diagnosis as di" \
              " where di.icd9code like '%518.81%' " \
              "  or di.icd9code like '%518.83%'" \
              "  or di.icd9code like '%518.84%';"
        cursor.execute(sql)
        list = cursor.fetchall()
        data = DataFrame(list)
        # data.to_csv('respiratory_failure.csv', encoding='utf-8')
        cursor.close()
        self.connection.close()
        return data

    # filter data missing congestive heart failure
    def filter_with_congestive_heart_failure(self):
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
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
        cursor.close()
        self.connection.close()
        return data

    # 筛选入院是满足柏林定义或者入院后满足八小时氧合指数满足柏林定义的患者住院记录
    def filter_with_p_f(self, unitid):
        '''
        :param unitid: 患者住院记录id
        :return: ARDS患者住院记录、ARDS确诊时间以及患者ARDS严重程度，1：重度，2：中度，3：轻度
        '''
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        # 先判断患者入院时，是否满足柏林定义
        sql = " select pao2, fio2" \
              " from apacheapsvar" \
              " where fio2 > 0" \
              "   and pao2 > 0" \
              "   and patientunitstayid = " + str(unitid)
        cursor.execute(sql)
        pa_fi = cursor.fetchall()
        result = {}
        if pa_fi:
            if pa_fi[0]:
                pao2 = pa_fi[0][0]
                fio2 = pa_fi[0][1]
                if fio2 > 1:
                    fio2 /= 100
                p_f = pao2 / fio2
                # 患者入院时满足ARDS，严重级别需要按照入院前八小时内最差氧合指数确定
                if p_f <= 300:
                    sql = " select labresultoffset as time,labname as name,labresult as value" \
                          " from lab" \
                          " where patientunitstayid = " + str(unitid) + \
                          " and labname in( 'paO2', 'FiO2')" \
                          " and labresultoffset >= -480" \
                          " and labresultoffset < 0" \
                          " order by time asc"
                    cursor.execute(sql)
                    pa_fi = cursor.fetchall()
                    cursor.close()
                    self.connection.close()
                    flag, identification, severity = access_ards(pa_fi)
                    if severity > 0:
                        result['severity'] = severity
                    else:
                        if p_f < 100:
                            severity = 1
                        elif p_f >= 100 and p_f < 200:
                            severity = 2
                        else:
                            severity = 3
                        result['severity'] = severity
                    result['id'] = unitid
                    result['identification'] = 0
                    # print('患者入院时满足ARDS标准')
                    return result

        sql = " select labresultoffset as time,labname as name,labresult as value" \
              " from lab" \
              " where patientunitstayid = " + str(unitid) + \
              "  and labname in( 'paO2', 'FiO2')" \
              " order by time asc"
        cursor.execute(sql)
        pa_fi = cursor.fetchall()
        cursor.close()
        self.connection.close()
        flag, identification, severity = access_ards(pa_fi)

        if flag == False:
            print('没有满足连续八小时的p/f,此时identification 为： ' + str(identification) + ' severity 为： ' + str(severity))
            return
        else:
            # 先对数值进行大小排序用于计算中位数
            result['id'] = unitid
            result['identification'] = identification
            result['severity'] = severity
        return result

    # filter data with peep 判断在指定确诊时间内peep是否都>=5，
    def filter_with_peep(self, id, identification):
        '''
        :param id: 满足呼吸衰竭且心血管衰竭，年龄不小于18，需要呼吸机支持的患者住院记录id
        :param identification: 患者满足连续八小时
        :return:True/False 当前患者的peep是否满足柏林定义
        '''
        # 只有peep,fio2和pao2数值在多张表中查找,在此处记录peep,fio2,pao2和p/f的动态数值
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        # 先找出每次住院记录中peep的所有值,如果peep没有或者不足八小时则不判断peep连续八小时大于5,直接判断p/f将p/f连续八小时的结束时间作为患者ARDS确诊时间
        sql = "  select labresultoffset as time,labresult as value" \
              "  from lab" \
              "  where patientunitstayid = " + str(id) + \
              "  and labresultoffset<=" + str(identification) + \
              "  and labresultoffset>" + str(identification - 480) + \
              "  and labname like 'PEEP'"
        cursor.execute(sql)
        peep_list = cursor.fetchall()
        # 判断在确诊时间前八小时，peep是否满足柏林定义
        # 将列表按照时间排序
        peep_list = sorted(peep_list, key=lambda x: x[0])
        if not peep_list:
            return True
        i = 0
        while i < len(peep_list):
            if not peep_list[i][1] is None:
                if peep_list[i][1] >= 5:
                    i += 1
                else:
                    print('identification is : ' + str(identification) + ' 此时的peep为： ' + str(peep_list))
                    break
            else:
                i += 1
        if i != len(peep_list):
            return False
        return True

    # filter data with age
    def filter_with_age(self):
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        # 当使用age>='18'时年龄为2，。。，9或者None时会被误选中,需要二次筛选
        sql = "  select patientunitstayid, case when age like '> 89' then 90 when age ~* '[0-9]' then cast(age as Numeric) end as age" \
              "  from patient as pa" \
              "  where pa.age >= '18';"
        cursor.execute(sql)
        list = cursor.fetchall()
        # id,age
        result = []
        for item in list:
            if item[1] >= 18:
                result.append(item[0])
        # print('age ids length : ' + str(len(result)))
        data = DataFrame(result)
        cursor.close()
        self.connection.close()
        return data

    def filter_with_vent_support(self):
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        # 住院记录中有呼吸机开始或者持续，患者吸入氧气浓度>21(正常人正常吸入氧气浓度为21，大于21，说明是使用氧气面罩或者呼吸机，当呼吸流速不大于3时为氧气面罩吸氧)
        sql = "  select patientunitstayid" \
              "  from respiratorycharting" \
              " where respchartvaluelabel like 'RT Vent On/Off'" \
              "    and respchartvalue in ('Continued', 'Start')" \
              "  union" \
              "  distinct" \
              "  select patientunitstayid" \
              "  from respiratorycharting" \
              " where respchartvaluelabel like 'FiO2'" \
              "   and respchartvalue > '21'" \
              "  except" \
              "  select patientunitstayid" \
              "  from respiratorycharting" \
              " where respchartvaluelabel like 'LPM O2'" \
              "   and respchartvalue <= '3'"
        cursor.execute(sql)
        list = cursor.fetchall()
        data = DataFrame(list)
        cursor.close()
        self.connection.close()
        return data

    # todo::find out details for ALP,GCS(intub),GCS(unable) and Spo2
    # find all dynamic data for each unit stay record
    def filter_dynamic(self, id, identification, enrollment):
        '''
        :param id: 患者住院记录
        :param identification: 患者ARDS确诊时间
        :param enrollment: 患者入科时间
        :return: 患者ARDS确诊时间到入科时间内的动态特征数据
        '''
        # find dynamic item in lab table
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        sql = " select labname as name,  labresult as result,labresultoffset as time" \
              " from lab as lb" \
              " where lb.patientunitstayid=" + str(id) + \
              " and labresultoffset >= " + str(identification) + \
              " and labresultoffset < " + str(enrollment) + \
              " and (lb.labresult > 0 and lb.labname in (" \
              "     'albumin' " \
              "    ,'ALT (SGPT)'        " \
              "    ,'PT - INR'        " \
              "    ,'AST (SGOT)'        " \
              "    ,'-bands'         " \
              "    ,'Base Excess'        " \
              "    ,'-basos'        " \
              "    ,'bicarbonate'        " \
              "    ,'HCO3'        " \
              "    ,'total bilirubin'      " \
              "    ,'BUN'     " \
              "    ,'calcium'      " \
              "    ,'Total CO2'      " \
              "    ,'creatinine'     " \
              "    ,'-eos'      " \
              "    ,'FiO2'       " \
              "    ,'glucose'     " \
              "    ,'bedside glucose'      " \
              "    ,'Carboxyhemoglobin'      " \
              "    ,'Methemoglobin'      " \
              "    ,'ionized calcium'      " \
              "    ,'lactate'      " \
              "    ,'magnesium'     " \
              "    ,'paCO2'        " \
              "    ,'paO2'        " \
              "    ,'PEEP'        " \
              "    ,'pH'        " \
              "    ,'platelets x 1000'       " \
              "    ,'potassium'          " \
              "    ,'PTT'         " \
              "    ,'sodium'        " \
              "    ,'Temperature'         " \
              "    ,'WBC x 1000'      ));"
        cursor.execute(sql)
        lab_list = cursor.fetchall()

        # find dynamic item in customlab table
        sql = " select cl.labothername as name,  cast(cl.labotherresult as numeric) as result,cl.labotheroffset as time" \
              " from customlab as cl" \
              " where cl.patientunitstayid = " + str(id) + \
              " and cl.labotheroffset  >= " + str(identification) + \
              " and cl.labotheroffset  < " + str(enrollment) + \
              " and cl.labothername like '%PIP%'"
        cursor.execute(sql)
        customlab_list = cursor.fetchall()
        # find dynamic item in nursecharting table
        sql = " select nc.nursingchartcelltypevalname        as name," \
              "        cast(nc.nursingchartvalue as numeric)  as result," \
              "        nc.nursingchartoffset as time " \
              " from nursecharting as nc" \
              " where nc.patientunitstayid =" + str(id) + \
              "  and nursingchartoffset >= " + str(identification) + \
              "  and nursingchartoffset < " + str(enrollment) + \
              "  and nc.nursingchartcelltypevalname in (" \
              "    'Eyes'     " \
              "   ,'GCS Total'     " \
              "   ,'Motor'      " \
              "   ,'Verbal'      " \
              "   ,'Heart Rate'     " \
              "   ,'Non-Invasive BP Diastolic'     " \
              "   ,'Non-Invasive BP Mean'     " \
              "   ,'Non-Invasive BP Systolic'     " \
              "   ,'Invasive BP Diastolic'     " \
              "   ,'Invasive BP Mean'     " \
              "   ,'Invasive BP Systolic'     " \
              "   ,'Respiratory Rate'    " \
              "    )" \
              "  and nc.nursingchartvalue ~* '[0-9]';"
        cursor.execute(sql)
        nurse_list = cursor.fetchall()

        # find dynamic item in respiratorycharting table
        sql = " select rc.respchartvaluelabel as name, cast(rc.respchartvalue as numeric) as result,rc.respchartoffset as time " \
              " from respiratorycharting as rc" \
              " where rc.patientunitstayid = " + str(id) + \
              "  and respchartoffset >= " + str(identification) + \
              "  and respchartoffset < " + str(enrollment) + \
              "  and rc.respchartvalue ~* '[0-9]'" \
              "  and rc.respchartvaluelabel in (" \
              "   'Mean Airway Pressure'         " \
              "   ,'SaO2'          " \
              "   ,'EtCO2'          " \
              "   ,'ETCO2'          " \
              "   ,'TV/kg IBW'          " \
              "   ,'Plateau Pressure' )"
        cursor.execute(sql)
        respchart_list = cursor.fetchall()

        # find dynamic item in vitalperiodic table where columns like 'pa%'
        sql = " select case when padiastolic > 0 then padiastolic else 0 end ," \
              " case when pasystolic > 0 then pasystolic else 0 end ," \
              " case when pamean > 0 then pamean else 0 end ," \
              " case when cvp > 0 then cvp else 0 end ," \
              " vp.observationoffset                                        as time" \
              " from vitalperiodic as vp" \
              " where vp.patientunitstayid =  " + str(id) + \
              " and vp.observationoffset>= " + str(identification) + \
              " and vp.observationoffset< " + str(enrollment) + ";"
        cursor.execute(sql)
        vital = cursor.fetchall()
        vital_list = []
        # format pa% item
        for item in vital:
            if item[0] > 0:
                vital_list.append(('padiastolic', item[0], item[-1]))
            if item[1] > 0:
                vital_list.append(('pasystolic', item[1], item[-1]))
            if item[2] > 0:
                vital_list.append(('pamean', item[2], item[-1]))
            if item[3] > 0:
                vital_list.append(('cvp', item[3], item[-1]))

        sql = " select case when hematocrit>0 then hematocrit end " \
              " from apacheapsvar" \
              " where patientunitstayid = " + str(id) + ";"
        cursor.execute(sql)
        apacheapsvar = cursor.fetchall()
        apacheapsvar_list = []
        for item in apacheapsvar:
            if not item[0] is None:
                if item[0] > 0:
                    apacheapsvar_list.append(('hematocrit', Decimal(item[0]), 0))
        cursor.close()
        self.connection.close()

        # combine all dynamic item
        result_list = lab_list + customlab_list + nurse_list + respchart_list + vital_list + apacheapsvar_list
        # data = DataFrame(result_list)
        return result_list

    # 找到住院记录对应的静态数据项， 药物使用（10）+疾病诊断（25）+入院来源+年龄+性别+最终结果（存活，死亡）+BMI+入院评分=41个静态特征
    def filter_static(self, item, header):
        '''
        :param item: 包含患者住院记录id，患者ARDS确诊时间以及患者入科时间分别为id,identification,enrollment
        :return:
        '''
        unitid = item[0]
        identification = item[1]
        enrollment = item[2]
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        header['id'] = unitid

        sql = " select max(case when me.drugname like 'WARFARIN%' then 1 else 0 end) as warfarin" \
              " from medication as me" \
              " where drugstartoffset >= " + str(identification) + \
              "   and drugstartoffset < " + str(enrollment) + \
              "   and patientunitstayid = " + str(unitid)
        cursor.execute(sql)
        warfarin = cursor.fetchone()
        if not warfarin is None:
            if not warfarin[0] is None:
                header['warfarin'] = warfarin[0]

        sql = " select " \
              " max(case when infu.drugname like 'Dobutamine%' then 1 else 0 end)     as dobutamine," \
              " max(case when infu.drugname like 'Dopamine%' then 1 else 0 end)       as Dopamine," \
              " max(case when infu.drugname like 'epinephrine%' then 1 else 0 end)    as epinephrine," \
              " max(case when infu.drugname like 'Heparin%' then 1 else 0 end)        as Heparin," \
              " max(case when infu.drugname like 'Milrinone%' then 1 else 0 end)      as Milrinone," \
              " max(case when infu.drugname like 'Norepinephrine%' then 1 else 0 end) as Norepinephrine," \
              " max(case when infu.drugname like 'Phenylephrine%' then 1 else 0 end)  as phenylephrine," \
              " max(case when infu.drugname like 'Vasopressin%' then 1 else 0 end)    as vasopressin" \
              " from infusiondrug as infu" \
              " where infusionoffset >= " + str(identification) + \
              "   and infusionoffset < " + str(enrollment) + \
              "  and patientunitstayid = " + str(unitid)
        cursor.execute(sql)
        medicine_list = cursor.fetchone()
        if not medicine_list is None:
            list = ['dobutamine', 'Dopamine', 'epinephrine', 'Heparin', 'Milrinone', 'Norepinephrine', 'phenylephrine',
                    'vasopressin']
            for i in range(len(medicine_list)):
                if not medicine_list[i] is None:
                    header[list[i]] = medicine_list[i]

        sql = " select max(case when tr.treatmentstring like '%vasopressor%' then 1 else 0 end) as Vasopressor" \
              " from treatment as tr" \
              " where treatmentoffset >= " + str(identification) + \
              "   and treatmentoffset < " + str(enrollment) + \
              "   and patientunitstayid = " + str(unitid)
        cursor.execute(sql)
        Vasopressor = cursor.fetchone()
        if not Vasopressor is None:
            if not Vasopressor[0] is None:
                header['Vasopressor'] = Vasopressor[0]

        sql = " select max(case when di.diagnosisstring like '%acute coronary syndrome%' then 1 else 0 end)  as Acute_Coronary_Syndrome_diagnosis," \
              " max(case when di.diagnosisstring like '%acute myocardial infarction%' then 1 else 0 end)     as Acute_Myocardial_Infarction," \
              " max(case when di.diagnosisstring like '%acute renal failure%' then 1 else 0 end)             as Acute_Renal_Failure," \
              " max(case when di.diagnosisstring like '%arrhythmia%' then 1 else 0 end)                      as Arrhythmia," \
              " max(case when di.diagnosisstring like '%asthma%' then 1 else 0 end)                          as Asthma_Emphysema," \
              " max(case when di.diagnosisstring like '%cancer%' then 1 else 0 end)                          as Cancer," \
              " max(case when di.diagnosisstring like '%cardiac arrest%' then 1 else 0 end)                  as Cardiac_Arrest," \
              " max(case when di.diagnosisstring like '%cardiogenic shock%' then 1 else 0 end)               as Cardiogenic_Shock," \
              " max(case when di.diagnosisstring like '%cardiovascular%' and  di.diagnosispriority like 'Other' then 1 else 0 end)                  as Cardiovascular_Medical," \
              " max(case when di.diagnosisstring like '%pericardi%' then 1 else 0 end)                     as Cardiovascular_Other," \
              " max(case when di.diagnosisstring like '%stroke%' then 1 else 0 end)                          as Cerebrovascular_Accident_Stroke," \
              " max(case when di.diagnosisstring like '%chest pain%' and di.diagnosisstring not like '%cardiovascular%' then 1 else 0 end)                      as Chest_Pain_Unknown_Origin," \
              " max(case when di.diagnosisstring like '%coma%' then 1 else 0 end)                            as Coma," \
              " max(case when di.diagnosisstring like '%CABG%' then 1 else 0 end)                            as Coronary_Artery_Bypass_Graft," \
              " max(case when di.diagnosisstring like '%ketoacidosis%' then 1 else 0 end)                    as Diabetic_Ketoacidosis," \
              " max(case when di.diagnosisstring like '%bleeding%' then 1 else 0 end)                        as Gastrointestinal_Bleed," \
              " max(case when di.diagnosisstring like '%GI obstruction%' then 1 else 0 end)                  as Gastrointestinal_Obstruction," \
              " max(case when di.diagnosisstring like '%neurologic%' and di.diagnosisstring not like '%stroke%' then 1 else 0 end)                      as Neurologic," \
              " max(case when di.diagnosisstring like '%overdose%' then 1 else 0 end)                        as Overdose," \
              " max(case when di.diagnosisstring like '%pneumonia%' then 1 else 0 end)                       as Pneumonia," \
              " max(case when di.diagnosisstring like '%ARDS%' or di.diagnosisstring like '%respiratory distress%' and di.diagnosisstring not like '%edema%' then 1 else 0 end) as Respiratory_Medical_Other," \
              " max(case when di.diagnosisstring like '%sepsis%' then 1 else 0 end)                          as Sepsis," \
              " max(case when di.diagnosisstring like '%thoracotomy%' then 1 else 0 end)                     as Thoracotomy," \
              " max(case when di.diagnosisstring like '%trauma%' then 1 else 0 end)                          as Trauma," \
              " max(case when di.diagnosisstring like '%valve%' then 1 else 0 end)                           as Valve_Disease," \
              " max(case when di.diagnosisstring like '%other infections%' then 1 else 0 end)                as others_diease   " \
              " from diagnosis as di" \
              " where  diagnosisoffset >= " + str(identification) + \
              "    and diagnosisoffset < " + str(enrollment) + \
              "    and patientunitstayid = " + str(unitid)
        cursor.execute(sql)
        diagnosis_list = cursor.fetchone()
        if not diagnosis_list is None:
            list = ['Acute_Coronary_Syndrome_diagnosis', 'Acute_Myocardial_Infarction', 'Acute_Renal_Failure',
                    'Arrhythmia',
                    'Asthma_Emphysema', 'Cancer', 'Cardiac_Arrest', 'Cardiogenic_Shock', 'Cardiovascular_Medical',
                    'Cardiovascular_Other', 'Cerebrovascular_Accident_Stroke', 'Chest_Pain_Unknown_Origin', 'Coma',
                    'Coronary_Artery_Bypass_Graft', 'Diabetic_Ketoacidosis', 'Gastrointestinal_Bleed',
                    'Gastrointestinal_Obstruction', 'Neurologic', 'Overdose', 'Pneumonia', 'Respiratory_Medical_Other',
                    'Sepsis', 'Thoracotomy', 'Trauma', 'Valve_Disease', 'others_diease']
            for i in range(len(diagnosis_list)):
                if not diagnosis_list[i] is None:
                    header[list[i]] = diagnosis_list[i]

        sql = " select " \
              " case when hospitaladmitsource like 'Emergency Department'  then 0" \
              "      when hospitaladmitsource like 'Operating Room'    then 1" \
              "      when hospitaladmitsource like 'Floor'     then 2" \
              "      when hospitaladmitsource like 'Direct Admit'    then 3" \
              "      when hospitaladmitsource ~* '[a-z]' then 4  end                     as admitsource," \
              " case when age like '> 89' then 90 " \
              "      when age ~* '[0-9]'  then cast(age as Numeric) " \
              "      else 91 end as age," \
              " case when gender like 'Female' then 0 else 1 end as gender ," \
              " case when admissionheight > 0 then round(admissionweight / pow(admissionheight / 100, 2),2)  else 0 end  as BMI" \
              " from patient as pa" \
              " where patientunitstayid = " + str(unitid)
        cursor.execute(sql)
        base_list = cursor.fetchone()
        # admitsource,age,gender,BMI,chargestatus,stay_day,leave_time,discharge_location
        if not base_list is None:
            list = ['admitsource', 'age', 'gender', 'BMI']
            for i in range(0, len(list)):
                if not base_list[i] is None:
                    header[list[i]] = base_list[i]

        sql = " select apachescore as admission_score" \
              " from apachepatientresult as aps" \
              " where patientunitstayid = " + str(unitid)
        cursor.execute(sql)
        admission_score = cursor.fetchone()
        if not admission_score is None:
            if not admission_score[0] is None:
                header['admission_score'] = admission_score[0]

        # 关闭连接池
        cursor.close()
        self.connection.close()

    def access_outcome(self, id, enrollment):
        '''
        :param id: 患者住院记录id
        :param enrollment: 患者入科时间
        :return: 患者预后结果评估，快速恢复0，长期住院1，快速死亡3
        '''
        enrollment = int(enrollment)
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        sql = " select labresultoffset as time,labname as name,labresult as value" \
              " from lab" \
              " where patientunitstayid = " + str(id) + \
              " and labname in( 'paO2', 'FiO2')" \
              " and labresultoffset>" + str(enrollment) + \
              " order by time asc"
        cursor.execute(sql)
        pa_fi = cursor.fetchall()

        # 是否为ARDS患者、确诊时间、患者严重程度评级
        flag, identification, severity = access_ards(pa_fi)
        sql = " select" \
              " unitdischargestatus ," \
              " hospitaldischargestatus," \
              " unitdischargeoffset/1440 as unitstay," \
              " (hospitaldischargeoffset - hospitaladmitoffset)/1440 as hospitalstay," \
              " unitdischargelocation," \
              " hospitaldischargelocation," \
              " cast(unitdischargeoffset as numeric), " \
              " cast(hospitaldischargeoffset as numeric) " \
              " from patient" \
              " where patientunitstayid = " + str(id)
        cursor.execute(sql)
        patient = cursor.fetchall()
        cursor.close()
        self.connection.close()
        outcome = -1
        if not patient is None:
            unitdischargestatus = patient[0][0]
            hospitaldischargestatus = patient[0][1]
            unitstay = patient[0][2]
            hospitalstay = patient[0][3]
            unitdischargelocation = patient[0][4]
            hospitaldischargelocation = patient[0][5]
            unitdischargeoffset = int(patient[0][6])
            hospitaldischargeoffset = int(patient[0][7])
            # 先判断患者快速恢复，长期住院，快速死亡和其他状态
            # print('Expired' in unitdischargestatus)
            # 快速恢复
            if unitdischargestatus == 'Expired' and unitdischargeoffset <= enrollment + 1440:
                outcome = 2
            elif hospitaldischargestatus == 'Expired' and hospitaldischargeoffset <= enrollment + 1440:
                outcome = 2
            elif unitdischargestatus != 'Expired' and hospitaldischargestatus != 'Expired' and hospitaldischargeoffset < enrollment + 1440:
                outcome = 0
            else:
                outcome = 1
            # 再判断患者详细状态
            detail_outcome = ''
            detail = -1
            if unitdischargestatus == 'Expired':
                detail_outcome += 'Expired in '
                if hospitaldischargelocation != 'Home':
                    detail_outcome += 'ICU and stay'
                else:
                    detail_outcome += 'Home and stay'
                if unitstay < 28:
                    detail_outcome += ' less than 28 days'
                else:
                    detail_outcome += ' not less than 28 days'
            else:
                if hospitaldischargestatus == 'Expired':
                    detail_outcome += 'Expired in '
                    if unitdischargelocation != 'Home':
                        detail_outcome += 'Hospital and stay'
                    else:
                        detail_outcome += 'Home and stay'
                    if hospitalstay < 28:
                        detail_outcome += ' less than 28 days'
                    else:
                        detail_outcome += ' not less than 28 days'
            if 'Expired in ICU' in detail_outcome:
                detail = 0
            elif 'less than 28 days' in detail_outcome and 'not ' not in detail_outcome:
                detail = 1
            elif 'Home' not in detail_outcome and 'not less than 28 days' in detail_outcome:
                detail = 2
                print('id : ' + str(id) + ' status : ' + str(outcome) + ' detail : ' + str(detail_outcome))
        return outcome, detail


def access_ards(pa_fi):
    '''
    :param pa_fi: 输入患者指定时间内的pao2和fio2数值列表，time:检查时间,name：检查项名称，value：检查项数值
    :return: 是否为ARDS患者、确诊时间、患者严重程度评级
    '''
    # 将数据按照时间进行排序
    pa_fi = sorted(pa_fi, key=lambda x: x[0])
    pa_mean = 0
    fi_mean = 0
    pa = 0
    fi = 0
    # print('pa_fi : ' + str(pa_fi))
    # 若采集项为空或者数据采集时间小于八小时, 则不计算p / f
    if not pa_fi:
        # print('采集项为空')
        return False, -1, -1
    for item in pa_fi:
        # time,name,value
        if item[1] == 'paO2':
            if not item[2] is None and item[2] > 1e-6:
                pa_mean += item[2]
                pa += 1
        else:
            if not item[2] is None and item[2] > 1e-6:
                fi_mean += item[2]
                fi += 1
    # 若只有一种采集项,则不计算p/f
    if pa == 0 or fi == 0:
        return False, -1, -1

    pa_mean = pa_mean / pa
    fi_mean = fi_mean / fi
    # pao2和fio2数值列表规格化
    for i in range(0, len(pa_fi)):
        # time,name,value
        # 1.使用对应均值填充为None
        if pa_fi[i][2] is None:
            extra_item = list(pa_fi[i])
            if pa_fi[i][1] == 'paO2':
                extra_item[2] = pa_mean
            else:
                extra_item[2] = fi_mean
            extra_item = tuple(extra_item)
            del pa_fi[i]
            pa_fi.insert(i, extra_item)
    # print('before : ' + str(pa_fi))
    # 2.若当前时间点的值不存在,需要找到对应的前面一个时间点的数据来填充当前值
    i = 0
    while i < len(pa_fi) - 1:
        # time,name,value
        if pa_fi[i][0] == pa_fi[i + 1][0] and pa_fi[i][1] != pa_fi[i + 1][1]:
            i += 1
        else:
            j = i - 1
            while j >= 0:
                # 当前检查项与指定检查项不同时
                if pa_fi[j][1] != pa_fi[i][1]:
                    # 两个检查项时间不一样说明需要填充前面的数值
                    extra_item = list(pa_fi[j])
                    extra_item[0] = pa_fi[i][0]
                    extra_item = tuple(extra_item)
                    pa_fi.insert(i + 1, extra_item)
                    i += 1
                    break
                else:
                    j -= 1
        i += 1
    # print('after : ' + str(pa_fi))
    # 3.去除无法填充的数据项
    i = 0
    while i < len(pa_fi) - 1:
        if pa_fi[i][0] != pa_fi[i + 1][0]:
            del pa_fi[i]
        else:
            break
    # print('处理好的pao2和fio2数列 : ' + str(pa_fi))
    # 4.计算患者identification时间以及pao2, fio2, p / f的中位数, 方差以及变化率
    pao2_list = []
    fio2_list = []
    p_f = []
    # 分别计算统计pao2,fio2,p/f
    i = 0
    min_pf = 301
    while i < len(pa_fi) - 1:
        if pa_fi[i][1] == 'paO2':
            # 吸入氧气浓度小于1
            fio2 = pa_fi[i + 1][2]
            pao2 = pa_fi[i][2]
        else:
            fio2 = pa_fi[i][2]
            pao2 = pa_fi[i + 1][2]
        if fio2 > 1:
            fio2 = fio2 / 100
        if fio2 < 1e-6:
            if fi_mean > 1:
                fi_mean = fi_mean / 100
            fio2 = fi_mean
        current_p_f = pao2 / Decimal(fio2)
        if current_p_f < min_pf:
            min_pf = current_p_f
        p_f.append((pa_fi[i][0], current_p_f))
        fio2_list.append(fio2)
        pao2_list.append(pao2)
        i += 2

    # print('p_f : ' + str(p_f) + ' pao2 : ' + str(pao2_list) + ' fio2 : ' + str(fio2_list))
    i = 0
    p_f_identification = 0
    if len(p_f) == 1 and p_f[0][1] <= 300:
        p_f_identification = p_f[0][0]
    else:
        while i < len(p_f):
            # 当前时间点的p/f满足柏林定义
            if p_f[i][1] <= 300:
                start = p_f[i][0]
                j = i + 1
                while j < len(p_f):
                    end = p_f[j][0]
                    # 下一个时间点满足柏林定义
                    if p_f[j][1] <= 300:
                        # 当前时间间隔小于八小时
                        if j + 1 < len(p_f) and end - start < 480:
                            j += 1
                        else:
                            p_f_identification = end
                            break
                    else:
                        break
                if p_f_identification > 0:
                    break
                else:
                    i = j
            else:
                i += 1

    if p_f_identification != 0:
        flag = True
    else:
        flag = False
    if min_pf <= 100:
        severity = 1
    elif min_pf <= 200:
        severity = 2
    elif min_pf <= 300:
        severity = 3
    else:
        severity = 4
    return flag, p_f_identification, severity


# find pao2 and fio2 of all unit stay records
def filter_with_pao2_or_fio2(self):
    # 筛选出所有患者记录中的pao2,fio2,并依次按照住院记录id和时间排序
    cursor = self.connection.cursor()
    sql = "set search_path to " + self.search_path + ";"
    cursor.execute(sql)
    sql = " select lb.patientunitstayid, lb.labresultoffset as time, lb.labname as name, lb.labresult as result" \
          " from lab as lb" \
          " where (lab.labname = 'paO2')" \
          "     or (lab.labname = 'FiO2')" \
          " and lb.labresultoffset > 0" \
          " union" \
          " select rc.patientunitstayid," \
          " rc.respchartoffset                 as time," \
          " rc.respchartvaluelabel             as name," \
          " cast(rc.respchartvalue as numeric) as result" \
          " from respiratorycharting as rc" \
          " where rc.respchartvaluelabel = 'FiO2'" \
          " and rc.respchartoffset > 0" \
          " order by patientunitstayid asc, time asc, name asc;"
    cursor.execute(sql)
    list = cursor.fetchall()
    cursor.close()
    self.connection.close()
    data = DataFrame(list)
    return data


def find_pao2_and_fio2_by_id(self, unitid):
    cursor = self.connection.cursor()
    sql = "set search_path to " + self.search_path + ";"
    cursor.execute(sql)
    sql = " select labresultoffset as time, labname as name, labresult as value" \
          " from lab" \
          " where lab.patientunitstayid = " + str(unitid) + \
          "   and ((lab.labname = 'paO2')" \
          "     or (lab.labname = 'FiO2'))" \
          "   and lab.labresultoffset > 0" \
          "   and lab.labresultoffset < 1440" \
          "   order by time asc;"
    cursor.execute(sql)
    data = cursor.fetchall()
    cursor.close()
    self.connection.close()
    return data


# compute median，variances and change rate for all dynamic items
def compute_dynamic(data, header):
    '''
    :param data: 一个元素为元组的数组，元组内部分别为检查项名称name，检查项值value以及检查项时间time
    :param header: 为最终表格表头
    :return:最终动态特征的中位数、方差以及变化率
    '''
    # 对于多名称特征数据进行更名
    pao2_fio2 = []
    new_data = []
    for i in range(0, len(data)):
        name = data[i][0]
        value = data[i][1]
        time = data[i][2]
        new_name = ''
        if name == 'Carboxyhemoglobin':
            new_name = 'Hemoglobin'
        elif name == 'Methemoglobin':
            new_name = 'Hemoglobin'
        elif name == 'HCO3':
            new_name = 'bicarbonate'
        elif name == 'bedside glucose':
            new_name = 'glucose'
        elif name == 'EtCO2':
            new_name = 'etCO2'
        elif name == 'ETCO2':
            new_name = 'etCO2'
        if new_name != '':
            new_data.append((new_name, value))
        else:
            new_data.append((name, value))
        if name == 'paO2' or name == 'FiO2':
            if name == 'FiO2':
                fio2 = value
                pao2_fio2.append((time, 'FiO2', fio2))
            else:
                pao2 = value
                pao2_fio2.append((time, 'paO2', pao2))
        # dataframe修改数据需要添加下面这句，否则修改不成功
    # 将处理后的数据进行排序，有利于后续中位数和方差的计算
    # 计算p/f的值
    p_f = pf_filter.compute_pf(pao2_fio2)
    if not p_f is None:
        # 列表拼接
        new_data += p_f
    new_data.sort()
    # print(str(new_data))
    # print('data : ' + str(new_data))
    result_list = {}
    i = 0
    # 动态数据中位数，方差，变化率的计算
    while i < len(new_data):
        item_name = new_data[i][0]
        item_value = new_data[i][1]
        item_list = []
        if item_value > 1e-6:
            item_list.append(item_value)
        # 汇总每种变量的所有信息
        j = i + 1
        # 取出同一特征item_name的所有数据存储在列表item_list中
        while j < len(new_data):
            current_name = new_data[j][0]
            current_value = new_data[j][1]
            if current_name == item_name:
                if current_value > 1e-6:
                    item_list.append(current_value)
                j += 1
            else:
                break
        # 当数据只有一条时，其中位数、方差、变化率直接可以得出
        # 计算中位数，方差，变化率
        size = len(item_list)
        # 排除已经计算的特征
        if item_name in dynamic_list:
            dynamic_list.remove(item_name)
        if size == 0:
            print('检查项为' + str(item_name) + '所有数值小于1e-6，无有效数值-----------------')
            result_list[item_name + '_median'] = '-'
            result_list[item_name + '_variances'] = '-'
            result_list[item_name + '_changerate'] = '-'
            i = j
            continue
        if size == 1:
            median = item_list[0]
            variance = 0
            change_rate = 0
        if size >= 2:
            if size % 2 == 0:
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
    # 将未计算或者不存在的特征中位数、方差、变化率置0
    for item in dynamic_list:
        result_list[item + '_median'] = '-'
        result_list[item + '_variances'] = '-'
        result_list[item + '_changerate'] = '-'
    # 更新动态数据数值
    for key, value in result_list.items():
        header[key] = value
    return header


def fill_invalid_data_with_average(data):
    data = np.array(data)[:, 0:-1]
    print('shape is : ' + str(data.shape))
    sum = 0
    non_zero_sum = 0
    # apache非法值使用均值填充
    for i in range(0, 8388):
        # 计算当前列非零数值的均值
        item = float(data[i][42])
        # 计算非0和非-项的总和与数量
        if item >= 1:
            sum += item
            non_zero_sum += 1
    if non_zero_sum > 0:
        average = round(sum / non_zero_sum, 3)
        # 使用均值填充不存在的数值项
        for i in range(0, 8388):
            item = float(data[i][42])
            if item < 1:
                data[i][42] = average
    for j in range(43, 203):
        sum = 0
        non_zero_sum = 0
        for i in range(0, 8381):
            # 计算当前列非零数值的均值
            item = data[i][j]
            # 计算非0和非-项的总和与数量
            if item != '-' and item != '0':
                sum += float(item)
                non_zero_sum += 1
        if non_zero_sum > 0:
            average = round(sum / non_zero_sum, 3)
            # 使用均值填充不存在的数值项
            for i in range(0, 8382):
                item = data[i][j]
                if item == '-':
                    data[i][j] = average
        if item == '-':
            data[i][j] = 0
    data = DataFrame(data)
    data.to_csv('result/fill_with_average.csv', mode='w', encoding='utf-8', header=result_header[:-1], index=False)
    print('均值补充完全')
