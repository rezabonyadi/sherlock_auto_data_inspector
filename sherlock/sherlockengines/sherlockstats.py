from scipy.stats import norm, ks_2samp, ttest_ind, mannwhitneyu
from scipy import stats
import numpy as np
from sklearn import metrics
import pandas as pd
from sherlock.sherlockengines import sherlockcomputationhelper as sch
from sklearn.tree import DecisionTreeClassifier


def one_variable_class_relationship(x, y, variable_name, threshold_p=0.05, threshold_overlap=.1):
    uy = np.unique(y)
    x1 = x[y==uy[0]]
    x2 = x[y==uy[1]]

    test_res = {
        'cohensd': cohensd(x1, x2),
        # 'gini': gini_index_performance(x.reshape(-1, 1), y),
        # 'distribution_overlap_empirical': overlap_distribution_empirical(x1, x2, int(x1.shape[0] / 10)),
        # 'distribution_overlap_theoretical': overlap_distribution_theoretical(x1, x2),
        't_test': ttest_ind(x1, x2)[1],
        'mannwhitney': mannwhitneyu(x1, x2)[1]
    }

    test_res_descision = {
        'cohensd': stats_decision_summarizer('cohensd', 0, test_res['cohensd']),
        # 'gini': stats_decision_summarizer('gini', .99, test_res['gini']),
        # 'distribution_overlap_empirical': stats_decision_summarizer('distribution_overlap_empirical', threshold_overlap,
        #                                                             test_res['distribution_overlap_empirical']) ,
        # 'distribution_overlap_theoretical': stats_decision_summarizer('distribution_overlap_theoretical', threshold_overlap,
        #                                                               test_res['distribution_overlap_theoretical']),
        't_test': stats_decision_summarizer('t_test', threshold_p, test_res['t_test']),
        'mannwhitney': stats_decision_summarizer('mannwhitney', threshold_p, test_res['mannwhitney'])
    }

    return pd.DataFrame.from_dict(test_res, columns=[variable_name], orient='index'), \
           pd.DataFrame.from_dict(test_res_descision, columns=[variable_name], orient='index')


def one_variable_regression_relationship(x, y, variable_name, threshold_p=0.05, threshold_overlap=.1):
    corr = get_correlation(x, y)

    test_res = {
        'correlation': corr[0] if corr[1] < threshold_p else 0
    }

    test_res_descision = {
        'correlation': abs(corr[0]) > .6
    }

    return pd.DataFrame.from_dict(test_res, columns=[variable_name], orient='index'), \
           pd.DataFrame.from_dict(test_res_descision, columns=[variable_name], orient='index')


def stats_decision_summarizer(test_type, thresholds, value):
    descision = (value < thresholds)

    if test_type is 'cohensd':
        descision = True if abs(value) > 0.8 else False # ('med' if abs(value) > 0.5 else 'small')
    if test_type is 'gini':
        descision = True if abs(value) > thresholds else False

    return descision


def get_correlation(v1, v2):
    c, p = stats.pearsonr(v1, v2)
    return c, p


def is_normal(v, alpha=1e-2):
    v = v - v.mean()
    k2, p = stats.normaltest(v)  # If p<alpha then the distribution is significantly different from normal
    return p>alpha, p


def bootstrap(data, p, func=np.mean, n_sample=1000):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    simulations = list()
    sample_size = len(data)
    for c in range(n_sample):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()

    """
    Return 2-sided symmetric confidence interval specified
    by p.
    """
    u_pval = (1 + p) / 2.
    l_pval = (1 - u_pval)
    l_indx = int(np.floor(n_sample * l_pval))
    u_indx = int(np.floor(n_sample * u_pval))
    ci = (simulations[l_indx], simulations[u_indx])
    value = [func(np.random.choice(data, size=len(data), replace=True)) for i in range(n_sample)]

    return np.median(value), ci, value


def overlap_distribution_theoretical(d1, d2):
    # TODO: This function throws an error when the distribution is just one number (std=0).
    '''
    Assumes normality
    :param dist1:
    :param dist2:
    :return:
    '''
    dist1 = d1.copy()
    dist2 = d2.copy()

    dist1 /= np.std(dist1)
    dist2 /= np.std(dist2)

    if np.mean(dist1) > np.mean(dist2):
        tmp = dist1
        dist1 = dist2
        dist2 = tmp

    # if (np.std(dist1)==0) or ((np.std(dist2)==0)):
    #     if (np.std(dist2)==0) and (np.std(dist2)==0):
    #         if np.mean(dist2) == np.mean(dist1):
    #             return 1.0
    #         else:
    #             return 0.0
    #

    m1 = np.mean(dist1)
    m2 = np.mean(dist2)
    std1 = 1.0
    std2 = 1.0

    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    r = np.roots([a, b, c])
    area = norm.cdf(r[0], m2, std2) + (1. - norm.cdf(r[0], m1, std1))

    return area


def overlap_distribution_empirical(d1, d2, number_bins):
    #     COINCIDENT DETECTION SIGNIFICANCE IN MULTIMESSENGER ASTRONOMY
    # Determine the range over which the integration will occur
    arr1 = np.copy(d1)
    arr2 = np.copy(d2)

    arr1 /= np.std(arr1)
    arr2 /= np.std(arr2)

    if number_bins < 10:
        return overlap_distribution_theoretical(arr1, arr2)

    min_value = np.min((arr1.min(), arr2.min()))
    max_value = np.min((arr1.max(), arr2.max()))
    # Determine the bin width
    bin_width = (max_value - min_value) / number_bins
    # For each bin, find min frequency
    lower_bound = min_value  # Lower bound of the first bin is the min_value of both arrays
    min_arr = np.empty(number_bins)  # Array that will collect the min frequency in each bin
    for b in range(number_bins):
        higher_bound = lower_bound + bin_width  # Set the higher bound for the bin
        # Determine the share of samples in the interval
        freq_arr1 = np.ma.masked_where((arr1 < lower_bound) | (arr1 >= higher_bound), arr1).count() / len(arr1)
        freq_arr2 = np.ma.masked_where((arr2 < lower_bound) | (arr2 >= higher_bound), arr2).count() / len(arr2)
        # Conserve the lower frequency
        min_arr[b] = np.min((freq_arr1, freq_arr2))
        lower_bound = higher_bound  # To move to the next range

    return min_arr.sum()


# function to calculate Cohen's d for independent samples
def cohensd(d1: np.array, d2: np.array):
    # see paper: Using Effect Size—or Why the P Value Is Not Enough
    d1 = d1.copy()
    d2 = d2.copy()

    d1 /= np.std(d2)
    d2 /= np.std(d2)
    # calculate the size of samples
    n1, n2 = d1.shape[0], d2.shape[0]
    # calculate the variance of the samples
    s1, s2 = d1.var(ddof=1), d2.var(ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = d1.mean(), d2.mean()
    # calculate the effect size
    return (u1 - u2) / s


def gini_index_performance(v, r):
    model = DecisionTreeClassifier()
    model.fit(v, r)
    r_hat = model.predict(v)
    sc = metrics.accuracy_score(r, r_hat)

    return sc


def apply_transformation(x: np.array, name: str, transformation: str):
    print('Performing transformation for ', name)
    d = x
    valid_transformations = ['log', 'boxcox']

    if not sch.is_numeric(x):
        return None

    if transformation not in valid_transformations:
        print(transformation, ' has not been implemented. Defaulting to log.')
        transformation = 'log'

    if (x < 0).any():
        print('The input includes negatives, defaulting to boxcox.')
        transformation = 'boxcox'

    # TODO: Fix the check for the datatype

    if transformation is 'log':
        if d.min() <= 0:
            v = d + d.min() + 1
            print('For log transformation, min shifted to 1.0.')
        else:
            v = d
        res = np.log10(v)

    if transformation is 'boxcox':
        res, _ = stats.boxcox(d)

    return res

