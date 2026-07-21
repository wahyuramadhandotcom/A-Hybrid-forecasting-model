import numpy as np

def rmspe(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-6, y_true)
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

def rmspe_xgb_eval(preds, dtrain):
    y = dtrain.get_label()
    y_safe = np.where(y == 0, 1e-6, y)
    rmspe_value = np.sqrt(np.mean(((y_safe - preds) / y_safe) ** 2))
    return "rmspe", rmspe_value
