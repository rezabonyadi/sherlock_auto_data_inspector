import pandas as pd
from sherlock.sherlocktabular import SherlockTabularDataModel, SherlockTabularBase
from sherlock.sherlockengines import sherlockcomputationhelper as sch
from sherlock.sherlockengines import sherlockvisualizationhelper as svh
from sherlock import _sherlockdefaultsettings
from sherlock.sherlockengines.sherlockmodelhandler import SherlockModelHandler
import numpy as np


class SherlockModelAgainstShuffle(SherlockTabularBase):
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
        content = 'Inspecting model and accuracy...'
        print(content)

        temp_x, temp_y = self.data.get_variables(), self.data.get_response()
        all_acc_dt, all_acc_sf, test_res, test_inference, means = \
            self.model_handler.compare_model_against_shuffle(temp_x, temp_y)

        self.test_accuracies = all_acc_dt
        self.shuffle_accuracies = all_acc_sf
        self.difference_from_shuffle_measures = test_res
        self.difference_from_shuffle_inference = test_inference
        self.shuffle_test_means = means
        print(means)

    def visualize(self, show_report=True):
        print('Visualizing model accuracy...')

        x1, x2 = self.test_accuracies.copy(), self.shuffle_accuracies.copy()
        better_than_shuffle = False

        if self.data.is_classification:  # Classification is around accuracy, and we want accuracy to be higher
            better_than_shuffle = (self.shuffle_test_means['mean_actual'] > self.shuffle_test_means['mean_shuffle']) or \
                                  (self.shuffle_test_means['median_actual'] > self.shuffle_test_means['median_shuffle'])
        else:  # Regression is around error, which we want it to be smaller than shuffle
            better_than_shuffle = (self.shuffle_test_means['mean_actual'] < self.shuffle_test_means['mean_shuffle']) or \
                                  (self.shuffle_test_means['median_actual'] < self.shuffle_test_means['median_shuffle'])

        content = ''.join(['There is ', '' if (np.asarray(list(self.difference_from_shuffle_inference.values())).any())
        else 'no ', 'support that the observed pattern in the response is not random (',
                           'better' if better_than_shuffle else 'worse', ' than shuffle)'])
        self.messages.append(content)

        for test, value in self.difference_from_shuffle_inference.items():
            content = ''.join(['The test ', test, ' does not' if (not value) else '',
                               ' support difference from shuffle test.'])
            self.messages.append(content)

        for test, value in self.shuffle_test_means.items():
            content = ''.join(['The ', test, ' is: ', str('{0:10.3}'.format(value))])
            self.messages.append(content)

        title = ''.join(['For the model ', type(self.model_handler.model).__name__, ': ', '\n'])
        for t in self.difference_from_shuffle_measures.keys():
            title = ''.join([title, t, ': ', str('{0:1.3f}'.format(self.difference_from_shuffle_measures[t])), '\n'])

        fig = svh.show_distributions(x1, x2, title, x_titles=['Non_shuffle', 'Shuffle'], show_plot=False,
                               fig_size=self.settings['fig_size'])

        return fig

