from enum import Enum
from sklearn.preprocessing import RobustScaler


class KnowledgeTypes(Enum):
    INFORMATION=0
    CONCLUSION=1
    WARNING=2
    ANY=3


class Knowledge:
    def __init__(self, knowledge_type, content, variables, section, threshold):
        self.type = knowledge_type
        self.content = content
        self.variables = variables
        self.section = section
        self.threshold = threshold


class SherlockKnowledge:
    def __init__(self, insight_constants=None, explorer_constants=None):

        if insight_constants is None:
            self.insight_constants = {'scaler': RobustScaler(), 'train_perc': 0.9, 'runs': 200,
                                      'selection_type': 'keep_order', 'importance_stability_threshold': 0.9,
                                      'bootstrap_ci': 0.95, 'model': None, 'layout': 'spring', 'nodes_size': 4000}
        else:
            self.insight_constants = insight_constants

        if explorer_constants is None:
            self.explorer_constants = {'correlated_threshold': 0.9, 'p_value_correlations': .05,
                                       'max_variables_to_bootstrap': 100, 'p_variables_response_relation': 0.05,
                                       'stats_max_population_size': 1000,
                                       'network_connections_threshold_variables_relations': 0.85, 'num_var_per_graph': 10,
                                       'max_nodes_to_draw_graph': 100}
        else:
            self.explorer_constants = explorer_constants

        self.knowledge = []

    def add_knowledge(self, knowledge_type, content, variables, section, threshold=None, show_content=True, show_var=False):
        # TODO: Add it to the right section
        # TODO: show different knowledge by given type
        new_knowledge = Knowledge(knowledge_type, content, variables, section, threshold)

        self.knowledge.append(new_knowledge)
        if show_content:
            print(new_knowledge.content)
        if show_var:
            print('Variable:')
            print(new_knowledge.variables)

    def summarize_knowledge(self):
        print('The summary of all knowledge')
        for item in self.knowledge:
            print(item.type,':', item.content)










