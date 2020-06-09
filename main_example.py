import numpy as np
import pandas as pd
from sherlock.sherlocktabular import SherlockTabularDataModel
from sherlock.sherlocktabular.sherlockexploration import SherlockTabularVariablesDistributions
from sklearn import datasets
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LassoCV, Lasso, LassoLars
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC


# model = LassoCV(fit_intercept=True, alphas=[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5], cv=10)
# model = DecisionTreeRegressor()
model = LinearSVC(penalty='l1', dual=False, max_iter=40000)
# model = Lasso()
scaler = StandardScaler()

insight_constants = {'scaler': scaler, 'train_perc': 0.9, 'runs': 200, 'selection_type': 'stratified',
                     'bootstrap_ci': 0.95, 'model': model, 'bootstrap_aggregate_percentage': 0.5,
                        'importance_stability_threshold': 0.9, 'layout': 'spring', 'nodes_size': 4000}

explorer_constants = {'max_variables_to_bootstrap': 100, 'p_variables_response_relation': 0.05,
                       'stats_max_population_size': 1000, 'network_connections_threshold_variables_relations': 0.85,
                      'num_var_per_graph': 10, 'max_nodes_to_draw_graph': 100, 'correlated_threshold': 0.9,
                      'p_value_correlations': .05}

# df = pd.read_csv('sample_data/data_logical_reasoning_tr3.csv')
# y = df['class'].values
# names = df.columns[1:-1].values
# x = df.values[:,1:-1]

# data=datasets.load_boston()
# x = data['data']
# y = data['target']
# names = data['feature_names']

[x, y] = datasets.load_breast_cancer(return_X_y=True)
names=None

data_model = SherlockTabularDataModel(x, y, names=names)
vd = SherlockTabularVariablesDistributions(data_model, None, None)



# df = pd.read_csv('sample_data/atari_punish.csv')
# y = df['Punishment'].values
# names = df.columns[1:-1].values
# x = df.values[:,1:-1]

# y = np.concatenate([np.zeros((3000)), np.ones((3000))])
# x = np.random.randn(6000, 50)

# data_x = pd.read_excel('sample_data/vars.xlsx')
# data_x = data_x.drop(columns='ID')
# names = data_x.columns
# x = data_x.values[:26]
# ys = {}
# ys['empathy'] = pd.read_excel('sample_data/resp.xlsx', 'Empathy')
# ys['age'] = pd.read_excel('sample_data/resp.xlsx', 'Age')
# ys['emotion'] = pd.read_excel('sample_data/resp.xlsx', 'Emotion')
# y = ys['empathy']['ACG.L'].values

# sh_ai = SherlockAutoInspector(x, y, names, insight_constants=insight_constants, explorer_constants=explorer_constants,
#                               turn_to_categorize=True)
# sh_ai.inspect()
# sh_ai.report()
# sh_ai.central_knowledge.summarize_knowledge()
# ss = 0
