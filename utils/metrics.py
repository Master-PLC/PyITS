import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_score,
                             r2_score, recall_score, roc_auc_score)

from utils.affiliation.generics import convert_vector_to_events
from utils.affiliation.metrics import pr_from_events


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def R2(pred, true):
    return r2_score(true, pred)


def SMAPE(pred, true):
    """Symmetric Mean Absolute Percentage Error
    """
    return np.mean(2.0 * np.abs(true - pred) / ((np.abs(true) + np.abs(pred)))) * 100


def WAPE(pred, true):
    """Masked weighted absolute percentage error (WAPE)
    """
    return np.sum(np.abs(true - pred)) / np.sum(np.abs(true)) * 100


def MSMAPE(pred, true, epsilon=0.1):
    """Function to calculate series wise smape values
    """
    comparator = np.full_like(true, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(pred) + np.abs(true) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(pred - true) / denom) * 100
    return msmape_per_series


def AffiliationMetrics(pred, true):
    events_pred = convert_vector_to_events(pred)
    events_label = convert_vector_to_events(true)
    Trange = (0, len(pred))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)
    return P, R, F


def metric_collector(pred, true, task_name='soft_sensor', **kwargs):
    if task_name in [
        'soft_sensor', 'process_monitoring', 'rul_estimation',
        'soft_sensor_ml', 'process_monitoring_ml', 'rul_estimation_ml'
    ]:
        assert pred.ndim == 2
        mae = MAE(pred, true)
        mse = MSE(pred, true)
        rmse = RMSE(pred, true)
        mape = MAPE(pred, true)
        mspe = MSPE(pred, true)
        r2 = R2(pred, true)
        return {
            'mae': mae.item(), 'mse': mse.item(), 'rmse': rmse.item(), 
            'mape': mape.item(), 'mspe': mspe.item(), 'r2': r2
        }

    elif task_name in ['fault_diagnosis', 'fault_diagnosis_ml']:
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred, average='weighted', zero_division=1)
        recall = recall_score(true, pred, average='weighted', zero_division=1)
        f1 = f1_score(true, pred, average='weighted', zero_division=1)
        kappa = cohen_kappa_score(true, pred)

        metrics = {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1_score': f1.item(),
            'cohen_kappa': kappa.item()
        }

        if 'probs' in kwargs:
            probs = kwargs['probs']
            roc_auc = roc_auc_score(true, probs, average='weighted', multi_class='ovr')
            avg_precision = average_precision_score(true, probs, average='weighted')
            metrics.update({'roc_auc': roc_auc.item(), 'avg_precision': avg_precision.item()})

        return metrics

    elif task_name in ['predictive_maintenance', 'predictive_maintenance_ml']:
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred, average='weighted', zero_division=1)
        recall = recall_score(true, pred, average='weighted', zero_division=1)
        f1 = f1_score(true, pred, average='weighted', zero_division=1)
        kappa = cohen_kappa_score(true, pred)

        metrics = {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1_score': f1.item(),
            'cohen_kappa': kappa.item(),
        }

        if not true.any():
            metrics.update({'aff_precision': 0, 'aff_recall': 0, 'aff_f1': 0})
        else:
            aff_precision, aff_recall, aff_f1 = AffiliationMetrics(pred, true)
            metrics.update({'aff_precision': aff_precision, 'aff_recall': aff_recall, 'aff_f1': aff_f1})

        if 'probs' in kwargs and len(np.unique(true)) > 1:
            probs = kwargs['probs']
            if len(probs.shape) == 2:
                probs = probs[:, 1]
            roc_auc = roc_auc_score(true, probs, average='weighted', multi_class='ovr')
            avg_precision = average_precision_score(true, probs, average='weighted')
            metrics.update({'roc_auc': roc_auc.item(), 'avg_precision': avg_precision.item()})

        return metrics

    else:
        raise NotImplementedError
