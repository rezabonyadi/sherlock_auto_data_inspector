import pandas as pd
from sherlock.sherlocktabular import SherlockTabularDataModel, SherlockTabularBase
from sherlock.sherlockengines import sherlockcomputationhelper as sch
from sherlock.sherlockengines import sherlockvisualizationhelper as svh
from sherlock import _sherlockdefaultsettings
from sherlock.sherlockengines.sherlockmodelhandler import SherlockModelHandler
import numpy as np
from sherlock.sherlockengines import sherlockstats as ss

from matplotlib import pyplot as plt
import seaborn as sns


class SherlockExtractInsights(SherlockTabularBase):
    def __init__(self, data: SherlockTabularDataModel, settings, logger):
        # Constants is a dictionary of: {max_variables_to_bootstrap}
        # TODO: Comment these variables
        self.settings = _sherlockdefaultsettings.against_shuffle

        super().__init__(data, settings, logger)

        self.model_handler = SherlockModelHandler(self.data.is_classification, train_perc=self.settings['train_perc'],
                                                  runs=self.settings['runs'], model=self.settings['model'],
                                                  scaler=self.settings['scaler'],
                                                  bootstrap_aggregate_percentage=self.settings[
                                                      'bootstrap_aggregate_percentage'],
                                                  selection_type=self.settings['selection_type'])

    def explore(self):
        content = 'Inspecting variables for insight...'
        print(content)
        s, c, r, v, sel = self.model_handler.get_variable_importance_measures(self.data.get_variables(),
                                                                              self.data.get_response(), self.data.names)

        c_sum = c.sum(axis=0)
        c_sum /= sel.sum(axis=0)
        self.variables_importance_stability_factor_mean = c_sum
        min_non_zero = int(np.sum(sel, axis=0).min())
        self.variables_importance_levels_original = v
        self.variables_importance_mask = sel

        self.variables_importance_levels = np.zeros((min_non_zero, self.data.number_variables))
        for i in range(self.data.number_variables):
            c_v = v[sel[:, i] == 1, i]
            self.variables_importance_levels[:, i] = c_v[:min_non_zero]

        self.variables_importance_levels_mean = self.variables_importance_levels.mean(axis=0)

    def visualize(self, show_report=True):
        print('Visualizing insights in model parameters...')

        selected_variables_names, variable_importance_value_actual, variables_bil_bootstrap_mean, \
        variables_importance_value_bootstrapped, variables_isf_mean = self._get_visualization_variable_importance_data()

        fig1, ax = plt.subplots(1, 1, figsize=self.settings['fig_size'])
        sns.set(font_scale=2)

        print(variable_importance_value_actual.shape)

        if variable_importance_value_actual.shape[1] > 0:
            sns.boxplot(x='Variable name', y='Importance level',
                        data=variable_importance_value_actual.melt(value_name='Importance level',
                                                                   var_name='Variable name'), ax=ax)
            ax.tick_params(labelrotation=90)

        fig2, ax = plt.subplots(1, 1, figsize=self.settings['fig_size'])
        sns.set(font_scale=2)

        if variables_importance_value_bootstrapped.shape[1] > 0:
            sns.boxplot(x='Variable name', y='Bagged importance level',
                        data=variables_importance_value_bootstrapped.melt(value_name='Bagged importance level',
                                                                          var_name='Variable name'), ax=ax)
            ax.tick_params(labelrotation=90)

        y = variables_bil_bootstrap_mean
        x = variables_isf_mean

        fig3, ax = plt.subplots(1, 1, figsize=self.settings['fig_size'])
        sns.set(font_scale=2)

        for i in range(x.shape[0]):
            r = np.double(2 * np.random.randint(2) - 1)
            ax.text(x[i], y[i] + r * 0.025, selected_variables_names[i], fontsize=14, weight='bold', ha='center')

        ax.scatter(x, y, edgecolors='k', cmap=plt.cm.autumn_r, s=600)
        ax.set_xlabel('Importance stability factor')
        ax.set_ylabel('Bagged importance level')
        ax.set_title('Visualization of persumably important variables')

        fig4, ax = plt.subplots(1, 1, figsize=self.settings['fig_size'])
        sns.set(font_scale=2)
        if selected_variables_names.shape[0] > 0:
            svh.show_importance_nodes(variables_isf_mean, variables_bil_bootstrap_mean, '',
                                      selected_variables_names, ax, emphasise=1.0)
            if show_report:
                plt.tight_layout()  # Or equivalently,  "plt.tight_layout()"
        if show_report:
            plt.show()

        return [fig1, fig2, fig3, fig4]

    def _get_visualization_variable_importance_data(self):
        selected_variables = np.where(self.variables_importance_stability_factor_mean >
                                      self.settings['importance_stability_threshold'])[0]
        selected_variables_names = self.data.names[selected_variables]
        variables_isf_mean = self.variables_importance_stability_factor_mean[selected_variables]
        variables_bil_bootstrap_mean = []

        variables_importance_value_bootstrapped = pd.DataFrame()
        variable_importance_value_actual = pd.DataFrame(self.variables_importance_levels[:, selected_variables],
                                                        columns=selected_variables_names)

        if selected_variables.shape[0] > 0:
            content = ''.join(['It seems like there are some variables which have a significant contribution to '
                               'the observed reponse patterns (importance stability factor larger than ',
                               str(self.settings['importance_stability_threshold']), ', with confidence interval ',
                               str(self.settings['bootstrap_ci']), ')'])
            self.messages.append(content)
        else:
            content = 'It seems like none of the variables had a stable contribution to the observed response pattern ' \
                      '. This can be because of simplicity of the model.'
            self.messages.append(content)

        for i in range(selected_variables.shape[0]):
            variable = selected_variables_names[i]
            mean_v, ci, bootstrapped_samples = ss.bootstrap(abs(variable_importance_value_actual[variable]),
                                                            self.settings['bootstrap_ci'])
            variables_importance_value_bootstrapped[variable] = bootstrapped_samples

            content = ''.join([variable, ': importance stability factor: {0:1.4f}, bagged mean abs level: '
                                         '{1:3.3f}'.format(variables_isf_mean[i], mean_v), '(', str(ci[0]), ', '
                                  , str(ci[1]), ')'])
            self.messages.append(content)

            variables_bil_bootstrap_mean.append(mean_v)

        variables_bil_bootstrap_mean = np.asarray(variables_bil_bootstrap_mean)

        return selected_variables_names, variable_importance_value_actual, variables_bil_bootstrap_mean, \
               variables_importance_value_bootstrapped, variables_isf_mean
