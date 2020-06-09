import pandas as pd
from sherlock.sherlocktabular import SherlockTabularDataModel, SherlockTabularBase
from sherlock.sherlockengines import sherlockcomputationhelper as sch
from sherlock.sherlockengines import sherlockvisualizationhelper as svh
from sherlock import _sherlockdefaultsettings


class SherlockTabularVariablesDescription(SherlockTabularBase):
    def __init__(self, data: SherlockTabularDataModel, settings, logger):
        # Constants is a dictionary of: {max_variables_to_bootstrap}
        # TODO: Comment these variables
        self.settings = _sherlockdefaultsettings.variables_description_setting

        super().__init__(data, settings, logger)

        self.variables_descriptions_dataframe = None

    def explore(self):
        print('Exploring variables distribution properties...')
        columns = self.data.names
        description_numerics = pd.DataFrame()
        df = self.data.data_frame

        do_bootstrap = (self.data.number_variables < self.settings['max_variables_to_bootstrap'])
        if not do_bootstrap:
            self.messages.append('Too many variables to bootstrap. Make max_variables_to_bootstrap larger.')
        i = 0
        total_variables = columns.shape[0]
        for c in columns:
            # svh.progress_show(0, columns.shape[0], np.where(columns==c)[0][0])
            print('\r', ''.join(['This is variable ', str(i), '/', str(total_variables), ', named: ', c]), end='')
            i += 1
            if sch.is_numeric(df[c]):
                d, _ = sch.get_variable_properties(df, c, do_bootstrap)
                if description_numerics.empty:
                    description_numerics = d
                else:
                    description_numerics[c] = d
            else:
                self.messages.append(''.join([c, ': Sherlock does not support non-numeric columns']))

        self.variables_descriptions_dataframe = description_numerics.T

    def visualize(self, show_report=True):
        report = {'data': [], 'variables': self.variables_descriptions_dataframe, 'response': None}

        report['data'].append(''.join(['There are ', str(self.data.number_variables), ' variables']))
        r = self.data.get_response()
        df = pd.DataFrame(r, columns=['Response'])
        d, _ = sch.get_variable_properties(df, 'Response', True)
        report['response'] = d

        return report
