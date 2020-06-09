import pandas as pd
from sherlock.sherlocktabular import SherlockTabularDataModel, SherlockTabularBase
from sherlock.sherlockengines import sherlockcomputationhelper as sch
from sherlock.sherlockengines import sherlockvisualizationhelper as svh
from sherlock import _sherlockdefaultsettings
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Lasso
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sherlock.sherlockengines.sherlockmodels.lda import LDA
from logger import Logger


class SherlockTabularAllDataModel(SherlockTabularBase):
    '''
    This class provides methods Sherlock uses to visualize if the data, as a whole, can be modelled linearly.
    It is used for visualization purposes and has no Analytics and interpretation value.
    '''

    def __init__(self, data: SherlockTabularDataModel, settings: dict, logger):
        # Constants is a dictionary of: {max_variables_to_bootstrap}
        # TODO: Comment these variables
        self.settings = _sherlockdefaultsettings.all_data_model

        super().__init__(data, settings, logger)

        self.variables_descriptions_dataframe = None

    def explore(self):
        pass

    def visualize(self, show_report=True):

        print('Visualizing linear model of the samples')

        x = self.data.get_variables()
        y = self.data.get_response()
        scaler = RobustScaler()
        x = scaler.fit_transform(x)
        if self.data.is_classification:
            model = LinearSVC()
            model.fit(x, y)

            # fig = plt.figure(figsize=fig_size)
            plt.rcParams.update({'font.size': 14})

            sns.set(font_scale=2)
            fig, axes = plt.subplots(1, 1, figsize=self.settings['fig_size'])

            y_hat = x.dot(model.coef_[0])
            y_vals = np.unique(y)
            sns.distplot((y_hat[y == y_vals[0]]), color='r', kde=True, hist_kws=dict(edgecolor="b", linewidth=.675),
                         ax=axes)
            sns.distplot((y_hat[y == y_vals[1]]), color='b', kde=True, hist_kws=dict(edgecolor="r", linewidth=.675),
                         ax=axes)
            axes.legend({'class 1', 'class 2'})
            axes.set_title('Linear support vector transformation')

            score = roc_auc_score(y, y_hat.reshape(-1, 1))

            content = ''.join(['The classes are discriminatable with AUC ROC score of: ', str(score)])
            self.messages.append(content)

            # model = LDA()
            # transformed_x = model.transform(x, y)
            #
            # axes[1].scatter(transformed_x[:, 0], transformed_x[:, 1], c=self.data.get_response(), edgecolors='k', s=600)
            # axes[1].set_title('Linear discriminant space')
            # axes[1].set_xlabel('Discriminant component 1')
            # axes[1].set_ylabel('Discriminant component 2')
        else:
            model = LinearRegression()
            model.fit(x, y)

            fig, axes = plt.subplots(1, 1, figsize=self.settings['fig_size'])
            plt.rcParams.update({'font.size': 26})

            y_hat = model.predict(x)
            score = model.score(x, y)
            sns.regplot(x=y, y=y_hat, ax=axes)
            plt.title(''.join(['Linear regressions, R^2: ', str(score)]))
            plt.xlabel('Modelled response')
            plt.ylabel('Actual response')
            plt.tight_layout()

            content = ''.join(['The response can be described by the variables with R^2 of: ', str(score)])
            self.messages.append(content)

        if show_report:
            plt.tight_layout()
            plt.show()

        return fig
