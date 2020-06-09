import pandas as pd
import numpy as np
from sherlock.sherlockengines import sherlockstats as ss
from sherlock.sherlockengines import sherlockvisualizationhelper as svh
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Lasso


class SherlockModelHandler:
    def __init__(self, is_classification, train_perc=0.9, bootstrap_aggregate_percentage=0.8, runs=2000, model=None,
                 scaler=None, selection_type='stratified'):
        self.is_classification = is_classification
        self.train_perc = train_perc
        self.runs = runs
        self.selection_type=selection_type
        self.bootstrap_aggregate_percentage = bootstrap_aggregate_percentage
        if scaler is None:
            scaler = RobustScaler()
        self.scaler = scaler

        if model is not None:
            self.model = model
        else:
            if self.is_classification:
                self.model = LinearSVC(penalty='l1', dual=False, max_iter=40000)
            else:
                self.model = Lasso(alpha=.0001, fit_intercept=True, max_iter=1000)
                # self.model = LinearRegression(fit_intercept=True)

    def get_variable_importance_measures(self, x, y, var_names):
        scales = np.zeros((self.runs, var_names.shape[0]))
        count = np.zeros((self.runs, var_names.shape[0]))
        rank = np.zeros((self.runs, var_names.shape[0]))
        val = np.zeros((self.runs, var_names.shape[0]))
        selected = np.zeros((self.runs, var_names.shape[0]))

        for i in range(self.runs):
            f_perc = int(np.floor(self.bootstrap_aggregate_percentage * var_names.shape[0]))
            all_f = np.arange(var_names.shape[0])
            fs = np.random.choice(all_f, f_perc, replace=False)
            fs.sort()
            #         print(fs.shape)

            s, r = self.random_selection(y, self.train_perc)
            xx = x[:, fs]
            x_train = xx[s, :]
            x_test = xx[r, :]

            if self.scaler is not None:
                x_train = self.scaler.fit_transform(x_train)
                x_test = self.scaler.transform(x_test)

            self.model.fit(x_train, y[s])
            imp = self.get_feature_importances()
            #         print(imp)
            #         imp /= max(imp) # Normalize to 1, so that the most important one has a "1" coefficient
            val[i, fs] = imp
            imp /= sum(imp)  # Normalize to sum=1, so that the most important one has a "1" coefficient
            scales[i, fs] = imp
            count[i, fs[imp > 0]] += 1
            rank[i, fs] = imp.argsort().argsort()
            selected[i, fs] += 1

        return scales, count, rank, val, selected

    def compare_model_against_shuffle(self, x: np.array, y: np.array):
        threshold_overlap = 0.1
        threshold_p = 0.05

        test_res = dict()
        means = dict()

        all_acc_dt = self.model_runner(x, y, shuffle=False)
        all_acc_sf = self.model_runner(x, y, shuffle=True)
        means['mean_actual'] = np.mean(all_acc_dt)
        means['mean_shuffle'] = np.mean(all_acc_sf)
        means['median_actual'] = np.median(all_acc_dt)
        means['median_shuffle'] = np.median(all_acc_sf)

        test_res['cohensd'] = ss.cohensd(all_acc_sf, all_acc_dt)
        # test_res['distribution_overlap_empirical'] = ss.overlap_distribution_empirical(all_acc_sf, all_acc_dt,
        #                                                                                int(self.runs / 10))
        # test_res['distribution_overlap_theoretical'] = ss.overlap_distribution_theoretical(all_acc_sf, all_acc_dt)
        test_res['t_test'] = ss.ttest_ind(all_acc_sf, all_acc_dt)[1]
        test_res['mannwhitney'] = ss.mannwhitneyu(all_acc_sf, all_acc_dt)[1]

        test_res_descision = {
            'cohensd': (abs(test_res['cohensd']) > 0.8),
            # 'distribution_overlap_empirical': (test_res['distribution_overlap_empirical'] < threshold_overlap),
            # 'distribution_overlap_theoretical': (test_res['distribution_overlap_theoretical'] < threshold_overlap),
            't_test': (test_res['t_test'] < threshold_p),
            'mannwhitney': (test_res['mannwhitney'] < threshold_p)
        }

        return all_acc_dt, all_acc_sf, test_res, test_res_descision, means

    def get_feature_importances(self):
        if hasattr(self.model, 'feature_importances_'):
            imp = abs(self.model.feature_importances_)
        else:
            if hasattr(self.model, 'coef_'):
                if self.model.coef_.shape[0] == 1:
                    imp = abs(self.model.coef_[0])
                else:
                    imp = abs(self.model.coef_)
            else:
                imp = None

        return imp

    def model_runner(self, x, y, shuffle=False):
        # Runs multiple models (models) on x and y, with 'perc' for training, over 'runs' number of runs.
        # Reduces the number of dimensions to num_f if it is larger than zero. Shuffles the x if shuffle is true,
        # useful for testing how reliable the results are.
        res = []

        for i in range(self.runs):
            if shuffle:  # Test against shuffle
                np.random.shuffle(x)

            selected, rest = self.random_selection(y, self.train_perc)
            test_acc = self.single_model_run(x, y, selected, rest)

            res.append(test_acc)

        return np.asarray(res)

    def single_model_run(self, x, y, selected, rest):
        x_train = x[selected, :]
        x_test = x[rest, :]
        if self.scaler is not None:
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.transform(x_test)

        self.model.fit(x_train, y[selected])

        y_hat = self.model.predict(x_test)

        return metrics.accuracy_score(y[rest], y_hat) if self.is_classification else \
            metrics.regression.mean_squared_error(y[rest], y_hat)

    def random_selection(self, y, perc):
        # Random selection method for permutation tests
        selected_indices = []
        if self.selection_type == 'keep_order':
            first_one = np.argmax(y == 1)
            s = np.int32(np.floor(first_one * perc))
            perm = np.random.permutation(first_one)
            indices = perm[0:s]
            selected_indices = np.concatenate([indices, indices + first_one])

        if self.selection_type == 'stratified':
            unique_values = np.unique(y)
            for v in unique_values:
                v_indx = np.where(y == v)[0]
                s = int(v_indx.shape[0] * perc)
                selected_indices.extend(np.random.choice(v_indx, size=s, replace=False))
            selected_indices = np.asarray(selected_indices)

        if self.selection_type == 'random':
            s = int(y.shape[0] * perc)
            selected_indices = (np.random.choice(list(np.arange(y.shape[0])), size=s, replace=False))
            selected_indices = np.asarray(selected_indices)

        rest_indices = np.delete(np.asarray(range(0, y.shape[0])), selected_indices)

        return selected_indices, rest_indices
