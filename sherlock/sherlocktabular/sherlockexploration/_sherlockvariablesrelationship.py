import pandas as pd
from sherlock.sherlocktabular import SherlockTabularDataModel, SherlockTabularBase
from sherlock.sherlockengines import sherlockstats as ss
from sherlock.sherlockengines import sherlockvisualizationhelper as svh
from sherlock import _sherlockdefaultsettings

import networkx as nx
import numpy as np


class SherlockTabularVariablesRelationships(SherlockTabularBase):
    def __init__(self, data: SherlockTabularDataModel, settings, logger):
        # TODO: Comment these variables
        self.settings = _sherlockdefaultsettings.variables_relationship_settings

        super().__init__(data, settings, logger)
        self.correlations = None
        self.correlation_reliability_matrix = None

    def explore(self):
        # TODO: implement more variable dependence methods
        # TODO: If too many variables, it needs to stop doing things
        print('Exploring variables inter-connection structures...')

        self.correlations = np.zeros((self.data.number_variables, self.data.number_variables))
        self.correlation_reliability_matrix = np.zeros((self.data.number_variables, self.data.number_variables))
        temp_x = self.data.get_variables()
        for i in range(self.data.number_variables):
            for j in range(i, self.data.number_variables):
                c, p = ss.get_correlation(temp_x[:, i], temp_x[:, j])
                self.correlations[i, j] = c
                self.correlations[j, i] = c
                self.correlation_reliability_matrix[i, j] = p
                self.correlation_reliability_matrix[j, i] = p

    def visualize(self, show_report=True):
        correlated_variables = pd.DataFrame(columns=['v1', 'v2', 'Corr', 'p_val'])

        for i in range(self.data.number_variables):
            for j in range(i, self.data.number_variables):
                c, p = self.correlations[i, j], self.correlation_reliability_matrix[i, j]
                if p < self.settings['p_value_correlations']:  # We are sure the correlation is correct
                    if abs(c) > self.settings['correlated_threshold']:
                        if i != j:
                            correlated_variables = correlated_variables.append({'v1': self.data.names[i],
                                                                                'v2': self.data.names[j],
                                                                                'Corr': c, 'p_val': p},
                                                                               ignore_index=True)
        self.correlated_variables_dataframe = correlated_variables

        content = 'It appears that the variables are all independent at the threshold' \
            if self.correlated_variables_dataframe.empty \
            else 'It appears that some variables are correlated with some others.'

        self.messages.append(content)

        if self.correlated_variables_dataframe.empty:
            return None

        if self.data.number_variables > self.settings['max_nodes_to_draw_graph']:
            self.messages.append('Large number of variables: The network graph is not going to look well.')
            self.messages.append('Showing network graph skipped.')
            return None

        cnn = self.correlations
        df = pd.DataFrame()

        df['from'] = self.data.names[np.triu_indices(cnn.shape[0], 1)[0]]
        df['to'] = self.data.names[np.triu_indices(cnn.shape[0], 1)[1]]
        df['weights'] = np.abs(cnn[np.triu_indices(cnn.shape[0], 1)])
        G, df_new = svh.get_network(df, thr=self.settings['network_connections_threshold_variables_relations'])
        title = ''.join(['Correlation between variables, threshold at: ',
                         str(self.settings['network_connections_threshold_variables_relations'])])
        color, bar_title = (np.array(list(nx.algorithms.degree_centrality(G).values()))), 'Degree centrality'

        _, fig = svh.network_show(G, df_new, node_size=self.settings['node_size'], title=title, node_color=color,
                                  color_bar_title=bar_title, show_plot=show_report, fig_size=self.settings['fig_size'])

        return fig
