variables_distribution_setting = {'max_variables_to_bootstrap': 90, 'fig_size':(12, 8),  'num_var_per_graph': 10}
variables_description_setting = {'max_variables_to_bootstrap': 90}

variables_relationship_settings = {'node_size': 4000, 'fig_size': (12, 8), 'max_nodes_to_draw_graph': 100,
                         'p_value_correlations': 0.05, 'correlated_threshold': 0.8,
                         'network_connections_threshold_variables_relations': 0.85}

variables_response_relationship_setting = {'p_variables_response_relation': 0.05, 'use_correction': True,
                                           'fig_size':(12, 8)}
all_data_model = {'fig_size': (18, 12)}
against_shuffle = {'scaler': None, 'train_perc': 0.9, 'runs': 200,
                                      'selection_type': 'keep_order', 'importance_stability_threshold': 0.9,
                                      'bootstrap_ci': 0.95, 'model': None, 'layout': 'spring', 'nodes_size': 4000}