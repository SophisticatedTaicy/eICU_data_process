from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

USER = "postgres"
PASSWORD = "123456"
HOST = "172.16.60.173"
PORT = "3307"

DATABASE = "eicu"
SEARCH_PATH = 'eicu_crd'

# 初始化特征数据为0，一共200维数据，其中
# 静态数据维度41维，分别为10种药物使用情况、26种疾病患否情况、患者入院来源、患者年龄、患者性别、患者BMI身体健康素质指数、患者最终结果（存活，死亡）以及患者入院Apache评分，
# todo::动态数据199维，包含数据信息统计表中53项动态生物特征的中位数、方差和变化率共159项数据、ALP、GCS(intub)、GCS(unable)、PIP以及SpO2五项尚未找到
# 先对200维数据初始化为0
'''
# 表头解释，表头字段分别为患者住院记录id，患者药物使用情况，患者入院疾病诊断情况，患者相关静态统计学信息
动态特征顺序为：
动脉血气分析、血常规（血液分析）、肝功能、肾功能、凝血功能、心肌酶及心肌治疗标志物、呼吸机参数
'''
result_header = ['id', 'severity',
                 'warfarin', 'dobutamine', 'Dopamine', 'epinephrine', 'Heparin', 'Milrinone', 'Norepinephrine',
                 'phenylephrine', 'vasopressin', 'Vasopressor', 'Acute_Coronary_Syndrome_diagnosis',
                 'Acute_Myocardial_Infarction', 'Acute_Renal_Failure', 'Arrhythmia',
                 'Asthma_Emphysema', 'Cancer', 'Cardiac_Arrest', 'Cardiogenic_Shock', 'Cardiovascular_Medical',
                 'Cardiovascular_Other', 'Cerebrovascular_Accident_Stroke', 'Chest_Pain_Unknown_Origin', 'Coma',
                 'Coronary_Artery_Bypass_Graft', 'Diabetic_Ketoacidosis', 'Gastrointestinal_Bleed',
                 'Gastrointestinal_Obstruction', 'Neurologic', 'Overdose', 'Pneumonia', 'Respiratory_Medical_Other',
                 'Sepsis', 'Thoracotomy', 'Trauma', 'Valve_Disease', 'others_diease', 'admitsource', 'age', 'gender',
                 'BMI', 'admission_score',
                 'pH_median', 'pH_variances', 'pH_changerate', 'paO2_median', 'paO2_variances', 'paO2_changerate',
                 'paCO2_median', 'paCO2_variances', 'paCO2_changerate', 'Base Excess_median', 'Base Excess_variances',
                 'Base Excess_changerate',
                 'bicarbonate_median', 'bicarbonate_variances', 'bicarbonate_changerate',
                 'lactate_median', 'lactate_variances', 'lactate_changerate', 'glucose_median', 'glucose_variances',
                 'glucose_changerate', 'WBC x 1000_median', 'WBC x 1000_variances', 'WBC x 1000_changerate',
                 '-basos_median', '-basos_variances', '-basos_changerate', '-eos_median', '-eos_variances',
                 '-eos_changerate', '-bands_median', '-bands_variances', '-bands_changerate', 'Hemoglobin_median',
                 'Hemoglobin_variances', 'Hemoglobin_changerate', 'hematocrit_median', 'hematocrit_variances',
                 'hematocrit_changerate', 'platelets x 1000_median', 'platelets x 1000_variances',
                 'platelets x 1000_changerate', 'AST (SGOT)_median', 'AST (SGOT)_variances', 'AST (SGOT)_changerate',
                 'ALT (SGPT)_median', 'ALT (SGPT)_variances', 'ALT (SGPT)_changerate', 'total bilirubin_median',
                 'total bilirubin_variances', 'total bilirubin_changerate',
                 'albumin_median', 'albumin_variances', 'albumin_changerate', 'cvp_median', 'cvp_variances',
                 'cvp_changerate', 'BUN_median', 'BUN_variances', 'BUN_changerate', 'creatinine_median',
                 'creatinine_variances', 'creatinine_changerate', 'PT - INR_median', 'PT - INR_variances',
                 'PT - INR_changerate', 'PTT_median', 'PTT_variances', 'PTT_changerate', 'FiO2_median',
                 'FiO2_variances', 'FiO2_changerate', 'PEEP_median', 'PEEP_variances', 'PEEP_changerate',
                 'Plateau Pressure_median', 'Plateau Pressure_variances', 'Plateau Pressure_changerate',
                 'Mean Airway Pressure_median', 'Mean Airway Pressure_variances', 'Mean Airway Pressure_changerate',
                 'TV/kg IBW_median', 'TV/kg IBW_variances', 'TV/kg IBW_changerate', 'Respiratory Rate_median',
                 'Respiratory Rate_variances', 'Respiratory Rate_changerate', 'P/F ratio_median', 'P/F ratio_variances',
                 'P/F ratio_changerate', 'GCS Total_changerate', 'GCS Total_median', 'GCS Total_variances',
                 'Eyes_median', 'Eyes_variances', 'Eyes_changerate', 'Motor_median', 'Motor_variances',
                 'Motor_changerate', 'Verbal_median', 'Verbal_variances', 'Verbal_changerate', 'calcium_median',
                 'calcium_variances', 'calcium_changerate', 'ionized calcium_median', 'ionized calcium_variances',
                 'ionized calcium_changerate', 'magnesium_median', 'magnesium_variances', 'magnesium_changerate',
                 'potassium_median', 'potassium_variances', 'potassium_changerate', 'sodium_median', 'sodium_variances',
                 'sodium_changerate', 'Total CO2_median', 'Total CO2_variances', 'Total CO2_changerate', 'etCO2_median',
                 'etCO2_variances', 'etCO2_changerate', 'SaO2_median', 'SaO2_variances', 'SaO2_changerate',
                 'Temperature_median', 'Temperature_variances', 'Temperature_changerate', 'Heart Rate_median',
                 'Heart Rate_variances', 'Heart Rate_changerate', 'padiastolic_median', 'padiastolic_variances',
                 'padiastolic_changerate', 'pamean_median', 'pamean_variances', 'pamean_changerate',
                 'pasystolic_median', 'pasystolic_variances', 'pasystolic_changerate', 'Invasive BP Diastolic_median',
                 'Invasive BP Diastolic_variances', 'Invasive BP Diastolic_changerate', 'Invasive BP Mean_median',
                 'Invasive BP Mean_variances', 'Invasive BP Mean_changerate', 'Invasive BP Systolic_median',
                 'Invasive BP Systolic_variances', 'Invasive BP Systolic_changerate',
                 'Non-Invasive BP Diastolic_median', 'Non-Invasive BP Diastolic_variances',
                 'Non-Invasive BP Diastolic_changerate', 'Non-Invasive BP Mean_median',
                 'Non-Invasive BP Mean_variances', 'Non-Invasive BP Mean_changerate', 'Non-Invasive BP Systolic_median',
                 'Non-Invasive BP Systolic_variances', 'Non-Invasive BP Systolic_changerate', 'status', 'detail']

dynamic_list = ['albumin', 'ALT (SGPT)', 'AST (SGOT)', '-bands', 'Base Excess', '-basos', 'bicarbonate',
                'total bilirubin', 'BUN', 'calcium', 'Total CO2', 'creatinine', '-eos', 'FiO2', 'glucose',
                'Hemoglobin', 'PT - INR', 'ionized calcium', 'lactate', 'magnesium', 'paCO2', 'paO2', 'P/F ratio',
                'PEEP', 'pH', 'platelets x 1000', 'potassium', 'PTT', 'sodium', 'Temperature', 'WBC x 1000',
                'Mean Airway Pressure', 'Plateau Pressure', 'SaO2', 'TV/kg IBW', 'cvp', 'etCO2',
                'padiastolic', 'pamean', 'pasystolic', 'Eyes', 'GCS Total', 'Motor',
                'Verbal', 'Heart Rate', 'Invasive BP Diastolic', 'Invasive BP Mean',
                'Invasive BP Systolic', 'Non-Invasive BP Diastolic', 'Non-Invasive BP Mean',
                'Non-Invasive BP Systolic', 'Respiratory Rate', 'hematocrit']

# 所有静态数据的特征名
static_list = ['warfarin', 'dobutamine', 'Dopamine', 'epinephrine', 'Heparin', 'Milrinone', 'Norepinephrine',
               'phenylephrine',
               'vasopressin', 'Vasopressor', 'Acute_Coronary_Syndrome_diagnosis', 'Acute_Myocardial_Infarction',
               'Acute_Renal_Failure', 'Arrhythmia', 'Asthma_Emphysema', 'Cancer', 'Cardiac_Arrest',
               'Cardiogenic_Shock',
               'Cardiovascular_Medical', 'Cardiovascular_Other', 'Cerebrovascular_Accident_Stroke',
               'Chest_Pain_Unknown_Origin',
               'Coma', 'Coronary_Artery_Bypass_Graft', 'Diabetic_Ketoacidosis', 'Gastrointestinal_Bleed',
               'Gastrointestinal_Obstruction',
               'Neurologic', 'Overdose', 'Pneumonia', 'Respiratory_Medical_Other', 'Sepsis', 'Thoracotomy', 'Trauma',
               'Valve_Disease', 'others_diease', 'admitsource', 'age', 'gender', 'BMI', 'status', 'admission_score']

params = {'n_estimators': 10,  # 弱分类器的个数
          'max_depth': 10,  # 弱分类器（CART回归树）的最大深度
          'learning_rate': 0.0000000001}

reg_model = GradientBoostingRegressor(
    loss="ls",
    learning_rate=0.18,
    n_estimators=200,
    subsample=0.8,
    max_features=0.8,
    max_depth=3,
    verbose=2
)

classify_model = GradientBoostingClassifier(
    loss='log_loss',
    learning_rate=0.001,
    n_estimators=100,
    subsample=0.8,
    max_features=1,
    max_depth=3,
    verbose=2
)
