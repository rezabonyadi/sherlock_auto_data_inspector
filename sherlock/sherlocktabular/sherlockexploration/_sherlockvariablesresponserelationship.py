import pandas as pd
from sherlock.sherlocktabular import SherlockTabularDataModel, SherlockTabularBase
from sherlock.sherlockengines import sherlockstats as ss
from sherlock.sherlockengines import sherlockvisualizationhelper as svh
from sherlock import _sherlockdefaultsettings

import networkx as nx
import numpy as np

class SherlockTabularVariablesResponseRelationship(SherlockTabularBase):
    def __init__(self, data, settings, logger):
        # TODO: Comment these variables
        self.settings = _sherlockdefaultsettings.variables_response_relationship_setting

        super().__init__(data, settings, logger)
        self.correlations = None
        self.correlation_reliability_matrix = None

    def explore(self):
        # TODO: This function only works with classification with two classes, implement for more number of classes.
        print('Exploring variables-response interplay...')
        # if self.response_descriptive_level is not None:
        #     return

        threshold_p = self.settings['p_variables_response_relation']
        use_correction = self.settings['use_correction']

        if (self.data.number_variables > 1) and use_correction:
            threshold_p = self.settings['p_variables_response_relation'] / self.data.number_variables

            content = ''.join(['More than one variable: Considering Bonferroni correction to reduce multiple '
                               'comparison effect. The new threshold is: ', str(threshold_p)])
            self.messages.append(content)

        distribution_differences, differences_inference = self.__get_variables_response_relations(threshold_p)

        self.is_describe_response = differences_inference
        self.response_descriptive_level = distribution_differences

        content = 'Explored variables-response interplay...'
        self.messages.append(content)

    def visualize(self, show_report=True):
        fig_size = self.settings['fig_size']
        df = self.is_describe_response.T
        df_details = self.response_descriptive_level.T

        content = 'Some variables in isolation can describe the response patterns.' \
            if self.is_describe_response.values.any() else \
            'None of the variables in isolation can describe the response patterns in isolation.'
        self.messages.append(content)

        uy = np.unique(self.data.get_response())
        temp_x = self.data.get_variables()
        temp_y = self.data.get_response()

        n_variables_described = 0
        figs = []
        for variable in df:
            v_index = np.where(self.data.get_names() == variable)[0][0]
            # true_index = np.where(df[variable].values==True)[0]
            if (df[variable].values == True).any():
                n_variables_described += 1

                content = ''.join(['Variable ', variable, ' can describe the patterns in the response'])
                self.messages.append(content)

                title = ''.join(['Relationship between variable ', variable, ' and response \n'])

                if self.data.is_classification:  # Classification
                    for indx in df.index:
                        title = ''.join([title, indx, ': ', str('{0:1.3f}'.format(df_details[variable][indx])),
                                         ', ', ('supports being discriminatory' if df[variable][indx]
                                                else 'doesn''t support being discriminatory'), '\n'])

                    x1 = temp_x[temp_y == uy[0], v_index]
                    x2 = temp_x[temp_y == uy[1], v_index]
                    fig = svh.show_distributions(x1, x2, title, show_plot=show_report, fig_size=fig_size)
                else:  # Regression
                    for indx in df.index:
                        title = ''.join([title, indx, ': ', str('{0:1.3f}'.format(df_details[variable][indx])),
                                         ', ', ('supports being related' if df[variable][indx]
                                                else 'doesn''t support being related'), '\n'])
                    fig = svh.show_relationship(temp_x[:, v_index], temp_y, title=title,
                                                axes_labels=[variable, 'response'], fig_size=fig_size,
                                                show_plot=show_report)

                figs.append(fig)

        return figs

    def __get_variables_response_relations(self, threshold_p):
        distribution_differences = pd.DataFrame()
        differences_inference = pd.DataFrame()
        x = self.data.get_variables()
        y = self.data.get_response()
        x_type = self.data.get_x_types()
        names = self.data.get_names()
        is_classification = self.data.is_classification
        number_variables = x.shape[1]

        for i in range(number_variables):
            if x_type[i] == 'n': # TODO: It only supports numerical values, not categories as variables
                if is_classification:
                    difference, inference = ss.one_variable_class_relationship(x[:, i], y, names[i],
                                                                               threshold_p=threshold_p)
                else:
                    difference, inference = ss.one_variable_regression_relationship(x[:, i], y, names[i],
                                                                                    threshold_p=threshold_p)
                if distribution_differences.empty:
                    distribution_differences = difference
                    differences_inference = inference
                else:
                    distribution_differences[names[i]] = difference
                    differences_inference[names[i]] = inference

        return distribution_differences.T, differences_inference.T
