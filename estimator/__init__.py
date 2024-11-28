from .foundation.fault_diagnosis_estimator import Fault_Diagnosis_Estimator
from .foundation.predictive_maintenance_estimator import \
    Predictive_Maintenance_Estimator
from .foundation.process_monitoring_estimator import \
    Process_Monitoring_Estimator
from .foundation.rul_estimation_estimator import RUL_Estimation_Estimator
from .foundation.soft_sensor_estimator import Soft_Sensor_Estimator
from .ml.ml_soft_sensor_estimator import ML_Soft_Sensor_Estimator

ML_ESTIMATOR_DICT = {
    'ml_soft_sensor': ML_Soft_Sensor_Estimator,
}

ESTIMATOR_DICT = {
    'soft_sensor': Soft_Sensor_Estimator,
    'process_monitoring': Process_Monitoring_Estimator,
    'fault_diagnosis': Fault_Diagnosis_Estimator,
    'rul_estimation': RUL_Estimation_Estimator,
    'predictive_maintenance': Predictive_Maintenance_Estimator,
}