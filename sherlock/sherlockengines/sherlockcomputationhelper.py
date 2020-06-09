import pandas as pd
import numpy as np
from scipy import stats
from sherlock.sherlockengines import sherlockstats as ss


def is_categorical(data):
    if str(data.dtype) is 'category':
        return True
    if is_numeric(data):
        v = data
        s = np.unique(v).size
        # Percentage of unique values to the total number of values to consider category
        return (s / v.shape[0]) < 0.2
    return False


def is_numeric(data):
    try:
        numeric = np.issubdtype(data.dtype, np.number)
    except:
        numeric = False
    return numeric


def categorise_variable(data, number_of_categories):
    bins = np.linspace(0, 1, number_of_categories)
    digitized = np.digitize(data, bins)
    return digitized


def categorize_response(y):
    if is_categorical(y):  # Already categorical, no need to change
        return y
    yp = y.copy()
    cut_point = np.median(y)
    yp[np.where(y < cut_point)] = -1
    yp[np.where(y >= cut_point)] = 1
    return yp


def get_variable_properties(data: pd.DataFrame, column_name: str, bootstrap_mean=False):
    d = data[column_name]
    alpha = 1e-2

    bootstrap_ci = .95
    normality, p_normality = ss.is_normal(d, alpha=alpha)
    if bootstrap_mean:
        bootstrap_mean, mean_ci, _ = ss.bootstrap(d.values, bootstrap_ci)
    else:
        bootstrap_mean, mean_ci = 0, [0, 0]

    try:
        res = {
            'column_name': column_name,
            'column_type': d.dtype.name,
            # 'n_unique': np.unique(d.values).size,
            # 'is_numeric': is_numeric(data[column_name]),
            # 'is_category': is_categorical(data[column_name]),
            'n_null': data[column_name].isnull().sum(),
            'min': min(d.values),
            'max': max(d.values),
            # 'sum': np.sum(d.values),
            # 'mean': np.mean(d.values),
            # 'median': np.median(d.values),
            'bootstrap_mean': bootstrap_mean,
            'bootstrap_mean_low': mean_ci[0],
            'bootstrap_mean_high': mean_ci[0],
            'std': d.values.std(),
            'skew': np.float64(stats.skew(d.values)),
            'IQR': np.percentile(d.values, 75) - np.percentile(d.values, 25),
            # 'percentile_25': np.percentile(d.values, 25),
            # 'percentile_50': np.percentile(d.values, 50),
            # 'percentile_75': np.percentile(d.values, 75),
            'is_normal': normality,
            'p_normal': p_normality,
            'raw_data': d
        }
        is_reliable = True
    except:
        is_reliable = False
        res = {}

    return pd.DataFrame.from_dict(res, columns=[column_name], orient='index'), is_reliable


def get_connectivity_full(variable_names, importance_level, importance_mask):
    cnn = np.zeros((variable_names.shape[0], variable_names.shape[0]))
    op = np.zeros((variable_names.shape[0], variable_names.shape[0]))
    from_list = []
    to_list = []

    vs = importance_level
    for i in range(vs.shape[0]):  # For all trials
        for j in range(variable_names.shape[0]):
            cnn[j, j] = 1
            for k in range(variable_names.shape[0]):
                if abs(vs[i, j]) > 0 and abs(vs[i, k]) > 0:
                    cnn[j, k] += 1
                if importance_mask[i, j] == 1 and importance_mask[i, k] == 1:
                    op[j, k] += 1

    cnn = np.divide(cnn, op)
    from_list = np.triu_indices(cnn.shape[0], 1)[0]
    to_list = np.triu_indices(cnn.shape[0], 1)[1]
    strength = cnn[np.triu_indices(cnn.shape[0], 1)]

    df = pd.DataFrame()
    df['from'] = variable_names[from_list]
    df['to'] = variable_names[to_list]
    df['weights'] = strength
    return cnn, df
