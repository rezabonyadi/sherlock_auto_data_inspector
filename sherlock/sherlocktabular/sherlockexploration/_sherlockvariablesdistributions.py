import pandas as pd
from sherlock.sherlocktabular import SherlockTabularDataModel, SherlockTabularBase
from sherlock.sherlockengines import sherlockcomputationhelper as sch
from sherlock.sherlockengines import sherlockvisualizationhelper as svh
from sherlock import _sherlockdefaultsettings


class SherlockTabularVariablesDistributions(SherlockTabularBase):
    def __init__(self, data: SherlockTabularDataModel, settings, logger):
        # Constants is a dictionary of: {max_variables_to_bootstrap}
        # TODO: Comment these variables
        self.settings = _sherlockdefaultsettings.variables_distribution_setting

        super().__init__(data, settings, logger)

        self.variables_descriptions_dataframe = None

    def explore(self):
        print('Exploring variables distribution properties...')

    def visualize(self, show_report=True):
        figs = []
        for i in range(0, self.data.number_variables, self.settings['num_var_per_graph']):
            s = i
            e = min(self.data.number_variables, i + self.settings['num_var_per_graph'])
            values = self.data.get_variables()[:, s:e]
            variables = self.data.get_names()[s:e]

            _, generated_graph = svh.box_plot_show(variables, values, title='Raw values',
                                                   show_plot=show_report, figsize=self.settings['fig_size'])
            figs.append(generated_graph)

        return figs
