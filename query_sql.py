import csv
import os
import pandas as pd
from numpy.core.defchararray import lower
from pandas.core.frame import DataFrame
import numpy as np
from decimal import Decimal
import psycopg2

import pf_filter
from filter.param import DATABASE, USER, PASSWORD, HOST, PORT, SEARCH_PATH, sum_list, dynamic_list, static_list


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

    def filter_with_respiratory_failure(self):
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
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

    # filter data with peep
    def filter_with_peep(self):
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        sql = " select distinct l1.patientunitstayid" \
              " from lab as l1" \
              " where l1.labname = 'PEEP'" \
              " and l1.labresult >= 5" \
              " union" \
              " select distinct rc.patientunitstayid" \
              " from respiratorycharting as rc" \
              " where rc.respchartvaluelabel = 'PEEP'" \
              " and rc.respchartvalue >= '5'" \
              " union" \
              " select distinct pe.patientunitstayid" \
              " from physicalexam as pe" \
              " where pe.physicalexamvalue='PEEP'" \
              " and pe.physicalexamtext > '5'"
        cursor.execute(sql)
        list = cursor.fetchall()
        data = DataFrame(list)
        cursor.close()
        self.connection.close()
        return data

    # filter data with age
    def filter_with_age(self):
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        sql = " select distinct patientunitstayid" \
              " from patient as pa" \
              " where pa.age >= '18'"
        cursor.execute(sql)
        list = cursor.fetchall()
        data = DataFrame(list)
        cursor.close()
        self.connection.close()
        return data

    def filter_pf(self, id):
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        sql = " select patientunitstayid as id,labresultoffset as time,labname as name,labresult as value" \
              " from lab" \
              " where lab.patientunitstayid = " + str(id) + \
              "  and ((lab.labname = 'paO2')" \
              "    or (lab.labname = 'FiO2'))" \
              "  and lab.labresultoffset > 0" \
              "  and lab.labresultoffset < 1440" \
              " order by time asc"
        cursor.execute(sql)
        list = cursor.fetchall()
        sql = " select case when pao2>0 then pao2 end, " \
              " case when fio2 > 0 then pao2 end" \
              " from apacheapsvar" \
              " where patientunitstayid = " + str(id) + ";"
        cursor.execute(sql)
        apacheapsvar = cursor.fetchall()

        for item in apacheapsvar:
            if item[0] > 0:
                list.append(['paO2', item[0], 1])
            if item[1] > 0:
                list.append(['FiO2', item[1], 1])

        sql = " select rc.patientunitstayid," \
              " rc.respchartoffset                 as time," \
              " rc.respchartvaluelabel             as name," \
              " cast(rc.respchartvalue as numeric) as result" \
              " from respiratorycharting as rc" \
              " where rc.respchartvaluelabel = 'FiO2'" \
              " and rc.respchartvalue not like '%\%%' " \
              " and rc.respchartoffset > 0" \
              " order by patientunitstayid asc, time asc, name asc;"
        cursor.execute(sql)
        fio2_list = cursor.fetchall()
        for item in fio2_list:
            list.append(item)
        cursor.close()
        self.connection.close()
        return list

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

    # todo::find out details for ALP,GCS(intub),GCS(unable) and Spo2
    # find all dynamic data for each unit stay record
    def filter_dynamic(self, unitid):
        # find dynamic item in lab table
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        sql = " select labname as name,  labresult as result,labresultoffset as time" \
              " from lab as lb" \
              " where lb.patientunitstayid=" + str(unitid) + \
              " and labresultoffset > 0" \
              " and (lb.labresult > 0 and lb.labname in (" \
              "     'Albumin' " \
              "    ,'ALT (SGPT)'        " \
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
              "    ,'WBC x 1000'      ))" \
              " and lb.labresult <= 1440;"
        cursor.execute(sql)
        lab_list = cursor.fetchall()

        # find dynamic item in customlab table
        sql = " select cl.labothername as name,  cast(cl.labotherresult as numeric) as result,cl.labotheroffset as time" \
              " from customlab as cl" \
              " where cl.patientunitstayid = " + str(unitid) + \
              " and cl.labotheroffset > 0 " \
              " and cl.labotheroffset < 1440 " \
              " and cl.labotherresult > '0'" \
              " and cl.labothername = 'PIP'"
        cursor.execute(sql)
        customlab_list = cursor.fetchall()
        # find dynamic item in nursecharting table
        sql = " select nc.nursingchartcelltypevalname        as name," \
              "        cast(nc.nursingchartvalue as numeric)  as result," \
              "        nc.nursingchartoffset as time " \
              " from nursecharting as nc" \
              " where nc.patientunitstayid =" + str(unitid) + \
              "  and nursingchartoffset > 0" \
              "  and nursingchartoffset < 1440" \
              "  and (nc.nursingchartvalue > '0' and nc.nursingchartcelltypevalname in (" \
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
              "   ,'respiratory rate'    " \
              "    ))" \
              "  and nc.nursingchartvalue like '%[0-9]%';"
        cursor.execute(sql)
        nurse_list = cursor.fetchall()

        # find dynamic item in respiratorycharting table
        sql = " select rc.respchartvaluelabel as name, cast(rc.respchartvalue as numeric) as result,rc.respchartoffset as time " \
              " from respiratorycharting as rc" \
              " where rc.patientunitstayid = " + str(unitid) + \
              "  and respchartoffset > 0" \
              "  and respchartoffset < 1440 " \
              "  and (rc.respchartvalue > '0')" \
              "  and rc.respchartvaluelabel in (" \
              "   'Mean airway pressure'         " \
              "   ,'FiO2'" \
              "   ,'SaO2'          " \
              "   ,'Tidal Volume (set)'          " \
              "   ,'Plateau Pressure' )" \
              " and rc.respchartvalue not like '%\%%' " \
              "order by name asc, result asc;"
        cursor.execute(sql)
        respchart_list = cursor.fetchall()

        # find dynamic item in vitalperiodic table where columns like 'pa%'
        sql = " select case when vp.padiastolic > 0 then vp.padiastolic else 0 end ," \
              " case when vp.pasystolic > 0 then vp.pasystolic else 0 end ," \
              " case when vp.pamean > 0 then vp.pamean else 0 end ," \
              " vp.observationoffset                                        as time" \
              " from vitalperiodic as vp" \
              " where vp.patientunitstayid =  " + str(unitid) + \
              " and vp.observationoffset > 0" \
              " and vp.observationoffset<1440 ;"
        cursor.execute(sql)
        vital = cursor.fetchall()

        sql = " select case when hematocrit>0 then hematocrit end ," \
              "        case when pao2>0 then pao2 end, " \
              "        case when fio2>0 then pao2 end " \
              " from apacheapsvar" \
              " where patientunitstayid = " + str(unitid) + ";"
        cursor.execute(sql)
        apacheapsvar = cursor.fetchall()
        apacheapsvar_list = []
        for item in apacheapsvar:
            if not item[0] is None:
                if item[0] > 0:
                    apacheapsvar_list.append(['hematocrit', Decimal(item[0]), 1])
            if not item[1] is None:
                if item[1] > 0:
                    apacheapsvar_list.append(['paO2', Decimal(item[1]), 1])
            if not item[2] is None:
                if item[2] > 0:
                    apacheapsvar_list.append(['FiO2', Decimal(item[2]), 1])

        cursor.close()
        self.connection.close()
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
        result_list = lab_list + customlab_list + nurse_list + respchart_list + vital_list + apacheapsvar_list
        data = DataFrame(result_list)
        return data

    #
    #
    # 找到住院记录对应的静态数据项， 药物使用（10）+疾病诊断（25）+入院来源+年龄+性别+最终结果（存活，死亡）+BMI+入院评分=41个静态特征
    def filter_static(self, unitid, header):
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        header['id'] = unitid

        sql = " select max(case when me.drugname like 'WARFARIN%' then 1 else 0 end) as warfarin" \
              " from medication as me" \
              " where drugstartoffset >= 0" \
              "   and drugstartoffset <= 1440" \
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
              " where infusionoffset <= 1440" \
              "  and infusionoffset >= 0 " \
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
              " where treatmentoffset >= 0" \
              "   and treatmentoffset <= 1440" \
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
              " where  di.diagnosisoffset >= 0" \
              "   and diagnosisoffset <= 1440 " \
              "   and patientunitstayid = " + str(unitid)
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
              " case when hospitaladmitsource like 'Other Hospital'  then 0" \
              "      when hospitaladmitsource like 'Direct Admit'    then 1" \
              "      when hospitaladmitsource like 'Observation'     then 2" \
              "      when hospitaladmitsource like 'ICU' or hospitaladmitsource like 'Other ICU'    then 3" \
              "      when hospitaladmitsource ~* '[a-z]' then 4 else 5 end                     as admitsource," \
              " case when age like '> 89' then 90 " \
              "      when age ~* '[0-9]'  then cast(age as Numeric) " \
              "      else 91 end as age," \
              " case when gender like 'Female' then 0 else 1 end as gender ," \
              " case when admissionheight > 0 then round(admissionweight / pow(admissionheight / 100, 2),2)  else 0 end  as BMI," \
              " case" \
              "     when pa.hospitaldischargestatus like '%Alive%' and" \
              "          (pa.hospitaldischargeoffset - pa.hospitaladmitoffset) / 1440 <= 1  then 0" \
              "     when pa.hospitaldischargestatus like '%Alive%' and" \
              "          (pa.hospitaldischargeoffset - pa.hospitaladmitoffset) / 1440 > 1  then 1" \
              "     when pa.hospitaldischargestatus like '%Expired%' and" \
              "          (pa.hospitaldischargeoffset - pa.hospitaladmitoffset) / 1440 <= 1 " \
              "          and pa.unittype like '%ICU%' then 2" \
              "     when pa.hospitaldischargestatus like '%Expired%' and" \
              "          (pa.hospitaldischargeoffset - pa.hospitaladmitoffset) / 1440 > 1 " \
              "     and  (pa.hospitaldischargeoffset - pa.hospitaladmitoffset) / 1440 <= 28 then 3" \
              "     when pa.hospitaldischargestatus like '%Expired%'  " \
              "     and  (pa.hospitaldischargeoffset - pa.hospitaladmitoffset) / 1440 > 28 " \
              "     and  pa.unitdischargelocation ~* '[a-z]' then 4 " \
              "     else 5 end as status" \
              " from patient as pa" \
              " where patientunitstayid = " + str(unitid)
        cursor.execute(sql)
        base_list = cursor.fetchone()
        if not base_list is None:
            list = ['admitsource', 'age', 'gender', 'BMI', 'status']
            for i in range(len(base_list)):
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
    # add head to data
    data.columns = ['name', 'result', 'time']
    # 对于多名称特征数据进行更名
    pao2_fio2 = []
    for index, row in data.iterrows():
        if "Carboxyhemoglobin" in row['name']:
            row['name'] = 'Hemoglobin'
        elif "Methemoglobin" in row['name']:
            row['name'] = 'Hemoglobin'
        elif "HCO3" in row['name']:
            row['name'] = 'bicarbonate'
        elif 'bedside glucose' in row['name']:
            row['name'] = 'glucose'
        elif 'pao2' or 'FiO2' in row['name']:
            pao2_fio2.append((data.iloc[index]['time'], data.iloc[index]['name'], data.iloc[index]['result']))
        # dataframe修改数据需要添加下面这句，否则修改不成功
        data.iloc[index] = row
    # 将处理后的数据进行排序，有利于后续中位数和方差的计算
    # 计算p/f的值
    p_f = pf_filter.compute_pf(pao2_fio2)
    if not p_f is None:
        p_f = DataFrame(p_f)
        p_f.columns = ['name', 'result', 'time']
        data = pd.concat([data, p_f], axis=0)
    data.sort_values(by='name', inplace=True, ascending=True)
    result_list = {}
    i = 0
    # 动态数据中位数，方差，变化率的计算
    while i < len(data):
        item_name = data.iloc[i, 0]
        item_list = []
        item_list.append(Decimal(data.iloc[i, 1]))
        # 汇总每种变量的所有信息
        j = i + 1
        # 取出同一特征item_name的所有数据存储在列表item_list中
        while j < len(data):
            if data.iloc[j, 0] == item_name:
                item_list.append(Decimal(data.iloc[j, 1]))
                j += 1
            else:
                break
        # 当数据只有一条时，其中位数、方差、变化率直接可以得出
        temp = []
        for item in item_list:
            if item != 0:
                temp.append(item)
        item_list = temp
        # 计算中位数，方差，变化率
        size = len(item_list)
        if size == 0:
            median = 0
            variance = 0
            change_rate = 0
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
        # 排除已经计算的特征
        if item_name in dynamic_list:
            dynamic_list.remove(item_name)
    # 将未计算或者不存在的特征中位数、方差、变化率置0
    for item in dynamic_list:
        result_list[item + '_median'] = 0
        result_list[item + '_variances'] = 0
        result_list[item + '_changerate'] = 0
    # 更新动态数据数值
    for key, value in result_list.items():
        header[key] = value
    return header
