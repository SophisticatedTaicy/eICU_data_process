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

    # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    def filter_with_p_f(self, unitid):
        '''
        :param unitid: ??????????????????id
        :return: ARDS?????????????????????ARDS????????????????????????ARDS???????????????1????????????2????????????3?????????
        '''
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        # ???????????????????????????????????????????????????
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
                # ?????????????????????ARDS????????????????????????????????????????????????????????????????????????
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
                    # print('?????????????????????ARDS??????')
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
            print('??????????????????????????????p/f,??????identification ?????? ' + str(identification) + ' severity ?????? ' + str(severity))
            return
        else:
            # ???????????????????????????????????????????????????
            result['id'] = unitid
            result['identification'] = identification
            result['severity'] = severity
        return result

    # filter data with peep ??????????????????????????????peep?????????>=5???
    def filter_with_peep(self, id, identification):
        '''
        :param id: ??????????????????????????????????????????????????????18?????????????????????????????????????????????id
        :param identification: ???????????????????????????
        :return:True/False ???????????????peep????????????????????????
        '''
        # ??????peep,fio2???pao2???????????????????????????,???????????????peep,fio2,pao2???p/f???????????????
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        # ??????????????????????????????peep????????????,??????peep???????????????????????????????????????peep?????????????????????5,????????????p/f???p/f??????????????????????????????????????????ARDS????????????
        sql = "  select labresultoffset as time,labresult as value" \
              "  from lab" \
              "  where patientunitstayid = " + str(id) + \
              "  and labresultoffset<=" + str(identification) + \
              "  and labresultoffset>" + str(identification - 480) + \
              "  and labname like 'PEEP'"
        cursor.execute(sql)
        peep_list = cursor.fetchall()
        # ????????????????????????????????????peep????????????????????????
        # ???????????????????????????
        peep_list = sorted(peep_list, key=lambda x: x[0])
        if not peep_list:
            return True
        i = 0
        while i < len(peep_list):
            if not peep_list[i][1] is None:
                if peep_list[i][1] >= 5:
                    i += 1
                else:
                    print('identification is : ' + str(identification) + ' ?????????peep?????? ' + str(peep_list))
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
        # ?????????age>='18'????????????2????????????9??????None??????????????????,??????????????????
        sql = "  select patientunitstayid, case when age like '> 89' then 90 when age ~* '[0-9]' and cast(age as Numeric)>=18 then cast(age as Numeric) end as age" \
              "  from patient as pa;"
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
        # ????????????????????????????????????????????????????????????????????????>21(????????????????????????????????????21?????????21????????????????????????????????????????????????????????????????????????3????????????????????????)
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
        :param id: ??????????????????
        :param identification: ??????ARDS????????????
        :param enrollment: ??????????????????
        :return: ??????ARDS???????????????????????????????????????????????????
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

    # ????????????????????????????????????????????? ???????????????10???+???????????????25???+????????????+??????+??????+?????????????????????????????????+BMI+????????????=41???????????????
    def filter_static(self, item, header):
        '''
        :param item: ????????????????????????id?????????ARDS?????????????????????????????????????????????id,identification,enrollment
        :return:
        '''
        unitid = item[0]
        identification = item[1]
        enrollment = item[2]
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)

        sql = " select " \
              " max(case when addrug.drugname like '%WARFARIN SODIUM%' then 1 else 0 end)     as warfarin," \
              " max(case when addrug.drugname like 'DOBUTAMINE HCL%' then 1 else 0 end)     as dobutamine," \
              " max(case when addrug.drugname like 'DOPAMINE HCL%'  then 1 else 0 end)       as Dopamine," \
              " max(case when addrug.drugname like 'EPINEPHRINE%' then 1 else 0 end)    as epinephrine," \
              " max(case when addrug.drugname like 'HEPARIN%'  then 1 else 0 end)        as Heparin," \
              " max(case when addrug.drugname like 'MILRINONE%' then 1 else 0 end)      as Milrinone," \
              " max(case when addrug.drugname like 'NOREPINEPHRINE%' then 1 else 0 end) as Norepinephrine," \
              " max(case when addrug.drugname like '%PHENYLEPHRINE%' then 1 else 0 end)  as phenylephrine" \
              " from admissiondrug as addrug" \
              " where patientunitstayid = " + str(unitid) + \
              " and drugenteredoffset >= " + str(identification) + \
              " and drugenteredoffset < " + str(enrollment)
        cursor.execute(sql)
        admissiondrug_list = cursor.fetchone()
        sql = " select " \
              " max(case when drugname like 'warfarin' then 1 else 0 end)     as warfarin," \
              " max(case when drugname like 'DOPamine' then 1 else 0 end)       as Dopamine," \
              " max(case when drugname like 'EPINEPHrine' then 1 else 0 end)    as epinephrine," \
              " max(case when lower(drugname) like 'heparin%' then 1 else 0 end)        as Heparin," \
              " max(case when lower(drugname) like '%norepinephrine%' then 1 else 0 end) as Norepinephrine," \
              " max(case when lower(drugname) like '%phenylephrine%' then 1 else 0 end)  as phenylephrine," \
              " max(case when me.drugname like 'VASOPRESSIN%' then 1 else 0 end) as vasopressin" \
              " from medication as me" \
              " where patientunitstayid = " + str(unitid) + \
              " and drugstartoffset>=" + str(identification) + \
              " and drugstopoffset< " + str(enrollment)
        cursor.execute(sql)
        medication_list = cursor.fetchone()
        admissiondrug = ['warfarin', 'dobutamine', 'Dopamine', 'epinephrine', 'Heparin', 'Milrinone', 'Norepinephrine',
                         'phenylephrine']
        medication = ['warfarin', 'Dopamine', 'epinephrine', 'Heparin', 'Norepinephrine', 'phenylephrine',
                      'vasopressin']
        if not admissiondrug_list is None:
            for i in range(len(admissiondrug)):
                if not admissiondrug_list[i] is None:
                    header[admissiondrug[i]] = admissiondrug_list[i]
        if not medication_list is None:
            for i in range(len(medication)):
                if not medication_list[i] is None:
                    if medication_list[i] > header[medication[i]]:
                        print('before is : ' + str(header[medication[i]]))
                        header[medication[i]] = medication_list[i]
                        print('after is : ' + str(header[medication[i]]))

        sql = " select max(case when treatmentstring like '%vasopressor%' then 1 else 0 end) as Vasopressor " \
              " from treatment" \
              " where patientunitstayid = " + str(unitid) + \
              " and treatmentoffset >=" + str(identification) + \
              " and treatmentoffset<" + str(enrollment)
        cursor.execute(sql)
        Vasopressor_list = cursor.fetchone()
        if not Vasopressor_list is None and not Vasopressor_list[0] is None:
            header['Vasopressor'] = Vasopressor_list[0]
        else:
            header['Vasopressor'] = 0

        sql = " select " \
              " max(case when apacheadmissiondx like '%Angina, unstable%' then 1 else 0 end)  as Acute_Coronary_Syndrome_diagnosis," \
              " max(case when apacheadmissiondx like '%MI%' then 1 else 0 end)     as Acute_Myocardial_Infarction," \
              " max(case when apacheadmissiondx like '%Renal %'  then 1 else 0 end)             as Acute_Renal_Failure," \
              " max(case when apacheadmissiondx like '%Rhythm%' then 1 else 0 end)                      as Arrhythmia," \
              " max(case when apacheadmissiondx like '%Asthma%' or apacheadmissiondx like '%Emphysema%' then 1 else 0 end)                          as Asthma_Emphysema," \
              " max(case when apacheadmissiondx like '%Cancer%' or apacheadmissiondx like '%Leukemia,%' then 1 else 0 end)                          as Cancer," \
              " max(case when apacheadmissiondx like '%Cardiac arrest%' then 1 else 0 end)                  as Cardiac_Arrest," \
              " max(case when apacheadmissiondx like '%Shock, cardiogenic%' then 1 else 0 end)               as Cardiogenic_Shock," \
              " max(case when apacheadmissiondx like '%Cardiovascular medical%'  then 1 else 0 end)                  as Cardiovascular_Medical," \
              " max(case when apacheadmissiondx like '%Angina, stable%' or apacheadmissiondx like '%Pericardi%'  then 1 else 0 end)                     as Cardiovascular_Other," \
              " max(case when apacheadmissiondx like '%cerebrovascular%' or apacheadmissiondx like '%Hemorrhage/%' then 1 else 0 end)       as Cerebrovascular_Accident_Stroke," \
              " max(case when apacheadmissiondx like '%Chest pain,%'   then 1 else 0 end)                      as Chest_Pain_Unknown_Origin," \
              " max(case when apacheadmissiondx like '%Coma/change%' or apacheadmissiondx like '%Nontraumatic coma%' then 1 else 0 end)      as Coma," \
              " max(case when apacheadmissiondx like '%CABG%' then 1 else 0 end)                            as Coronary_Artery_Bypass_Graft," \
              " max(case when apacheadmissiondx like '%Diabetic%' then 1 else 0 end)                    as Diabetic_Ketoacidosis," \
              " max(case when apacheadmissiondx like '%Bleeding%' or apacheadmissiondx like '%GI perforation/%' then 1 else 0 end)           as Gastrointestinal_Bleed," \
              " max(case when apacheadmissiondx like '%GI obstruction%' then 1 else 0 end)                  as Gastrointestinal_Obstruction," \
              " max(case when apacheadmissiondx like '%,neurologic%' or apacheadmissiondx like '%Neoplasm%' or apacheadmissiondx like '%Seizures%' or apacheadmissiondx like 'Neuro%' then 1 else 0 end)         as Neurologic," \
              " max(case when apacheadmissiondx like '%Overdose,%' or apacheadmissiondx like '%Toxicity, drug%' then 1 else 0 end)                        as Overdose," \
              " max(case when apacheadmissiondx like '%Pneumonia,%' then 1 else 0 end)                       as Pneumonia," \
              " max(case when apacheadmissiondx like '%Apnea%' or apacheadmissiondx like '%respiratory distress%' or apacheadmissiondx like '%edema%' or apacheadmissiondx like '%pulmonary???%'  then 1 else 0 end) as Respiratory_Medical_Other," \
              " max(case when apacheadmissiondx like '%Sepsis,%' then 1 else 0 end)                          as Sepsis," \
              " max(case when apacheadmissiondx like '%Thoracotomy%' then 1 else 0 end)                     as Thoracotomy," \
              " max(case when apacheadmissiondx like '%trauma%' then 1 else 0 end)                          as Trauma," \
              " max(case when apacheadmissiondx like '%valve%'  and apacheadmissiondx not like '%CABG%' then 1 else 0 end)     as Valve_Disease," \
              " max(case when apacheadmissiondx like '%, other' then 1 else 0 end)                as others_diease   " \
              " from patient as di" \
              " where  patientunitstayid = " + str(unitid)

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
              "      when hospitaladmitsource ~* '[a-z]' then 4  end      as admitsource," \
              " case when age like '> 89' then 90 " \
              "      when age ~* '[0-9]'  then cast(age as Numeric) " \
              "      else 91 end as age," \
              " case when gender like 'Female' then 0 else 1 end as gender," \
              " admissionweight,  " \
              " admissionheight" \
              " from patient as pa" \
              " where patientunitstayid = " + str(unitid) + \
              " and age >='18';"
        cursor.execute(sql)
        base_list = cursor.fetchone()
        # admitsource,age,gender,BMI,chargestatus,stay_day,leave_time,discharge_location
        if not base_list is None:
            list = ['admitsource', 'age', 'gender']
            for i in range(0, len(list)):
                if not base_list[i] is None:
                    header[list[i]] = base_list[i]
        for i in range(0, len(base_list)):
            if base_list[3] is None or base_list[3] < 10 or (base_list[4] is None or base_list[4] < 1):
                header['BMI'] = 0
            # ????????????
            else:
                if base_list[4] > 1 and base_list[4] < 2.5:
                    header['BMI'] = round(base_list[3] / pow(base_list[4], 2), 2)
                if base_list[4] >= 100:
                    header['BMI'] = round(base_list[3] / pow(base_list[4] / 100, 2), 2)
                else:
                    header['BMI'] = 0
        sql = " select case when apachescore>0 then apachescore else 0 end as admission_score" \
              " from apachepatientresult as aps" \
              " where patientunitstayid = " + str(unitid)
        cursor.execute(sql)
        admission_score = cursor.fetchone()
        if not admission_score is None:
            if not admission_score[0] is None:
                header['admission_score'] = admission_score[0]

        # ???????????????
        cursor.close()
        self.connection.close()

    def access_outcome(self, id, enrollment):
        '''
        :param id: ??????????????????id
        :param enrollment: ??????????????????
        :return: ???????????????????????????????????????0???????????????1???????????????3
        '''
        enrollment = int(enrollment)
        cursor = self.connection.cursor()
        sql = "set search_path to " + self.search_path + ";"
        cursor.execute(sql)
        sql = " select labresultoffset as time,labname as name,labresult as value" \
              " from lab" \
              " where patientunitstayid = " + str(id) + \
              " and labname in( 'paO2', 'FiO2')" \
              " and labresultoffset>=" + str(enrollment + 1440) + \
              " order by time asc"
        cursor.execute(sql)
        pa_fi = cursor.fetchall()

        # ?????????ARDS????????????????????????????????????????????????
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
            # ????????????????????????????????????????????????????????????????????????
            # ????????????
            if flag == False:
                if unitdischargestatus == 'Expired' and unitdischargeoffset <= enrollment + 1440:
                    outcome = 2
                # ????????????
                elif hospitaldischargestatus == 'Expired' and hospitaldischargeoffset <= enrollment + 1440:
                    outcome = 2
                # ????????????
                elif unitdischargestatus != 'Expired' and unitdischargeoffset <= enrollment + 1440:
                    outcome = 0
                # ????????????
                elif hospitaldischargestatus != 'Expired' and hospitaldischargeoffset <= enrollment + 1440:
                    outcome = 0
                else:
                    outcome = 1
            else:
                outcome = 1
            # ???????????????????????????
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
            elif 'Home' in detail_outcome and 'not less than 28 days' in detail_outcome:
                # ??????28???????????????
                detail = 3
        return outcome, detail, unitstay, hospitalstay


def access_ards(pa_fi):
    '''
    :param pa_fi: ??????????????????????????????pao2???fio2???????????????time:????????????,name?????????????????????value??????????????????
    :return: ?????????ARDS????????????????????????????????????????????????
    '''
    # ?????????????????????????????????
    pa_fi = sorted(pa_fi, key=lambda x: x[0])
    pa_mean = 0
    fi_mean = 0
    pa = 0
    fi = 0
    # ?????????????????????????????????????????????????????????, ????????????p / f
    if not pa_fi:
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
    # ????????????????????????,????????????p/f
    if pa == 0 or fi == 0:
        return False, -1, -1
    pa_mean = pa_mean / pa
    fi_mean = fi_mean / fi
    # pao2???fio2?????????????????????
    for i in range(0, len(pa_fi)):
        # time,name,value
        # 1.???????????????????????????None
        if pa_fi[i][2] is None:
            extra_item = list(pa_fi[i])
            if pa_fi[i][1] == 'paO2':
                extra_item[2] = pa_mean
            else:
                extra_item[2] = fi_mean
            extra_item = tuple(extra_item)
            del pa_fi[i]
            pa_fi.insert(i, extra_item)
    # 2.?????????????????????????????????,?????????????????????????????????????????????????????????????????????
    i = 0
    while i < len(pa_fi) - 1:
        # time,name,value
        if pa_fi[i][0] == pa_fi[i + 1][0] and pa_fi[i][1] != pa_fi[i + 1][1]:
            i += 1
        else:
            j = i - 1
            while j >= 0:
                # ??????????????????????????????????????????
                if pa_fi[j][1] != pa_fi[i][1]:
                    # ???????????????????????????????????????????????????????????????
                    extra_item = list(pa_fi[j])
                    extra_item[0] = pa_fi[i][0]
                    extra_item = tuple(extra_item)
                    pa_fi.insert(i + 1, extra_item)
                    i += 1
                    break
                else:
                    j -= 1
        i += 1
    # 3.??????????????????????????????
    i = 0
    while i < len(pa_fi) - 1:
        if pa_fi[i][0] != pa_fi[i + 1][0]:
            del pa_fi[i]
        else:
            break
    # 4.????????????identification????????????pao2, fio2, p / f????????????, ?????????????????????
    pao2_list = []
    fio2_list = []
    p_f = []
    # ??????????????????pao2,fio2,p/f
    i = 0
    min_pf = 301
    while i < len(pa_fi) - 1:
        if pa_fi[i][1] == 'paO2':
            # ????????????????????????1
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

    i = 0
    p_f_identification = 0
    if len(p_f) == 1 and p_f[0][1] <= 300:
        p_f_identification = p_f[0][0]
    else:
        while i < len(p_f):
            # ??????????????????p/f??????????????????
            if p_f[i][1] <= 300:
                start = p_f[i][0]
                j = i + 1
                while j < len(p_f):
                    end = p_f[j][0]
                    # ????????????????????????????????????
                    if p_f[j][1] <= 300:
                        # ?????????????????????????????????
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
    # ?????????????????????????????????pao2,fio2,???????????????????????????id???????????????
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


# compute median???variances and change rate for all dynamic items
def compute_dynamic(data, header):
    '''
    :param data: ?????????????????????????????????????????????????????????????????????name???????????????value?????????????????????time
    :param header: ?????????????????????
    :return:??????????????????????????????????????????????????????
    '''
    # ???????????????????????????????????????
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
        # dataframe????????????????????????????????????????????????????????????
    # ??????????????????????????????????????????????????????????????????????????????
    # ??????p/f??????
    p_f = pf_filter.compute_pf(pao2_fio2)
    if not p_f is None:
        # ????????????
        new_data += p_f
    new_data.sort()
    # print(str(new_data))
    # print('data : ' + str(new_data))
    result_list = {}
    i = 0
    # ???????????????????????????????????????????????????
    while i < len(new_data):
        item_name = new_data[i][0]
        item_value = new_data[i][1]
        item_list = []
        if item_value > 1e-6:
            item_list.append(item_value)
        # ?????????????????????????????????
        j = i + 1
        # ??????????????????item_name??????????????????????????????item_list???
        while j < len(new_data):
            current_name = new_data[j][0]
            current_value = new_data[j][1]
            if current_name == item_name:
                if current_value > 1e-6:
                    item_list.append(current_value)
                j += 1
            else:
                break
        # ??????????????????????????????????????????????????????????????????????????????
        # ????????????????????????????????????
        size = len(item_list)
        # ???????????????????????????
        if item_name in dynamic_list:
            dynamic_list.remove(item_name)
        if size == 0:
            print('????????????' + str(item_name) + '??????????????????1e-6??????????????????-----------------')
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
        # ???????????????????????????
        result_list[item_name + '_median'] = round(median, 2)
        result_list[item_name + '_variances'] = round(variance, 2)
        result_list[item_name + '_changerate'] = round(change_rate, 2)
        i = j
    # ?????????????????????????????????????????????????????????????????????0
    for item in dynamic_list:
        result_list[item + '_median'] = '-'
        result_list[item + '_variances'] = '-'
        result_list[item + '_changerate'] = '-'
    # ????????????????????????
    for key, value in result_list.items():
        header[key] = value
    return header


def fill_invalid_data_with_average(data):
    data = np.array(data)
    print('shape is : ' + str(data.shape))
    sum = 0
    non_zero_sum = 0
    # BMI??????apache???????????????????????????
    # print(str(data[0:8389, 41]))
    for j in range(40, 42):
        # BMI??????10????????????
        for i in range(0, 8389):
            # print('i is : ' + str(i) + ' j is : ' + str(j) + ' item is : ' + str(data[i][j]))
            # ????????????????????????????????????
            item = float(data[i][j])

            # ?????????0??????-?????????????????????
            if item > 10:
                sum += item
                non_zero_sum += 1
        if non_zero_sum > 0:
            average = round(sum / non_zero_sum, 3)
            # ???????????????????????????????????????
            for i in range(0, 8389):
                item = float(data[i][j])
                if item <= 10:
                    data[i][j] = average
    for j in range(42, 201):
        sum = 0
        non_zero_sum = 0
        for i in range(0, 8389):
            # ????????????????????????????????????
            item = data[i][j]
            # ?????????0??????-?????????????????????
            if item != '-' and item != '0':
                sum += float(item)
                non_zero_sum += 1
        if non_zero_sum > 0:
            average = round(sum / non_zero_sum, 3)
            # ???????????????????????????????????????
            for i in range(0, 8389):
                item = data[i][j]
                if item == '-':
                    data[i][j] = average
    data = DataFrame(data)
    print(str(data))
    data.to_csv('result/fill_with_average.csv', mode='w', encoding='utf-8', header=result_header, index=False)
    print('??????????????????')


def fill__with_0(data):
    data = np.array(data)
    for j in range(41, 200):
        for i in range(0, 8389):
            item = data[i][j]
            if item == '-':
                data[i][j] = 0
    data = DataFrame(data)
    data.to_csv('result/fill_with_0.csv', mode='w', encoding='utf-8', header=result_header, index=False)
    print('done!')
