from sherlock.sherlocktabular import SherlockTabularDataModel
from sherlock.sherlocktabular.sherlockexploration import SherlockTabularVariablesDistributions, \
    SherlockTabularVariablesDescription, SherlockTabularVariablesRelationships, \
    SherlockTabularVariablesResponseRelationship, SherlockTabularAllDataModel
from sherlock.sherlocktabular.sherlockinsight import SherlockModelAgainstShuffle, SherlockExtractInsights


class Sherlock:
    def __init__(self):
        pass

    def load_data(self, data):
        y = data['response'].values
        names = data.columns[1:-1].values
        x = data.values[:, 1:-1]

        self.data_model = SherlockTabularDataModel(x, y, names=names)

    def visualize_variables_descriptions(self, settings):
        self.variables_descriptions = SherlockTabularVariablesDescription(self.data_model, settings, None)
        self.variables_descriptions.explore()
        return self.variables_descriptions.visualize(show_report=False)

    def visualize_variables_distributions(self, settings):
        self.variables_distributions = SherlockTabularVariablesDistributions(self.data_model, settings, None)
        self.variables_distributions.explore()
        return self.variables_distributions.visualize(show_report=False)

    def visualize_variables_network(self, settings):
        self.variables_network = SherlockTabularVariablesRelationships(self.data_model, settings, None)
        self.variables_network.explore()
        return self.variables_network.visualize(show_report=False)

    def visualize_variables_response_relationship(self, settings):
        self.variables_response_relationship = SherlockTabularVariablesResponseRelationship(self.data_model, settings,
                                                                                            None)
        self.variables_response_relationship.explore()
        return self.variables_response_relationship.visualize(show_report=False)

    def visualize_all_data_model(self, settings):
        self.all_data_model = SherlockTabularAllDataModel(self.data_model, settings, None)
        self.all_data_model.explore()
        return self.all_data_model.visualize(show_report=False)

    def visualize_compare_against_shuffle(self, settings):
        self.against_random_shuffle = SherlockModelAgainstShuffle(self.data_model, settings, None)
        self.against_random_shuffle.explore()
        return self.against_random_shuffle.visualize(show_report=False)

    def visualize_insights(self, settings):
        self.insights = SherlockExtractInsights(self.data_model, settings, None)
        self.insights.explore()
        return self.insights.visualize(show_report=False)


