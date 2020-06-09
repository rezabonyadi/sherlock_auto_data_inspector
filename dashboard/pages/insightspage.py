from dashboard.utils import dashstyledcomponents as dsc, componentsstyles
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import available_settings


def overall_page():
    variables_tabs = dbc.Tabs([
        dbc.Tab(html.Div(children=[get_compare_against_shuffle()]), label="Compare against shuffle",
                tab_id='compare-shuffle'),
        dbc.Tab(html.Div(children=[get_variables_importance()]), label="Variables importance",
                tab_id='variables-importance'),
    ])
    content = html.Div(children=variables_tabs,
                       style={'outline': 'True', 'margin-top': '1%', 'margin-left': '1%',
                              'margin-right': 'auto', 'width': '95%'})

    return content


def get_compare_against_shuffle():
    # against_shuffle = {'scaler': None, 'train_perc': 0.9, 'runs': 200, 'selection_type': 'keep_order',
    #                    'importance_stability_threshold': 0.9,
    #                    'bootstrap_ci': 0.95, 'model': None, 'layout': 'spring', 'nodes_size': 4000}

    c1 = dsc.NamedInput('Number of runs', id='number-runs', min=1, max=1000000, step=1, value=200,
                        style={'margin': '5%'})
    c3 = dsc.NamedInput('Bagging %', id='bagging-perc', min=.01, max=1.0, step=.01, value=.5, style={'margin': '5%'})

    models = available_settings.available_models.keys()
    options_list = [{'value': r, 'label': r} for r in models]
    c4 = dsc.NamedDropdown("Model", id='select-model', options=options_list, style={'width': '100%', 'margin': '5%'})

    scalers = available_settings.available_models_preprocessing.keys()
    options_list = [{'value': r, 'label': r} for r in scalers]
    c5 = dsc.NamedDropdown("Scaler", id='select-scaler', style={'width': '100%', 'margin': '5%'}, options=options_list)

    selections = available_settings.available_random_selections
    options_list = [{'value': r, 'label': r} for r in selections]
    c6 = dsc.NamedDropdown("Random slection", id='select-random', style={'width': '100%', 'margin': '5%'},
                           options=options_list)
    c7 = dsc.NamedInput('Train %', id='train-perc', min=0.1, max=1.0, step=.01, value=0.9, style={'margin': '5%'})

    div = html.Div(children=[c1, c3, c4, c5, c6, c7], style={'margin': '1%'})

    cl = dbc.Col([div,
                  dsc.spinner_button('Run comparison', 'compare-against-shuffle-spinner',
                                     id='compare-against-shuffle-button', n_clicks=0, color="primary",
                                     style={'margin': '5% 5% 10%'}),
                  # dbc.Button('Run comparison', id='compare-against-shuffle-button',
                  #          n_clicks=0, color="primary", style={'margin': '5% 5% 10%'})
                  ], width=3)

    cr = dbc.Col([html.Div(children=[], id='comparison-against-shuffle-container')], width=9)

    return dbc.Row(children=[cl, cr])


def get_variables_importance():
    # against_shuffle = {'scaler': None, 'train_perc': 0.9, 'runs': 200,
    #                    'selection_type': 'keep_order', 'importance_stability_threshold': 0.9,
    #                    'bootstrap_ci': 0.95, 'model': None, 'layout': 'spring', 'nodes_size': 4000}

    c1 = dsc.NamedInput('Number of runs', id='number-runs-insights', min=1, max=1000000, step=1, value=200,
                        style={'margin': '5%'})
    c3 = dsc.NamedInput('Bagging %', id='bagging-perc-insights', min=.01, max=1.0, step=.01, value=.5, style={'margin': '5%'})

    models = available_settings.available_models.keys()
    options_list = [{'value': r, 'label': r} for r in models]
    c4 = dsc.NamedDropdown("Model", id='select-model-insights', options=options_list, style={'width': '100%', 'margin': '5%'})

    scalers = available_settings.available_models_preprocessing.keys()
    options_list = [{'value': r, 'label': r} for r in scalers]
    c5 = dsc.NamedDropdown("Scaler", id='select-scaler-insights', style={'width': '100%', 'margin': '5%'}, options=options_list)

    selections = available_settings.available_random_selections
    options_list = [{'value': r, 'label': r} for r in selections]
    c6 = dsc.NamedDropdown("Random slection", id='select-random-insights', style={'width': '100%', 'margin': '5%'},
                           options=options_list)
    c7 = dsc.NamedInput('Train %', id='train-perc-insights', min=0.1, max=1.0, step=.01, value=0.9, style={'margin': '5%'})

    div = html.Div(children=[c1, c3, c4, c5, c6, c7], style={'margin': '1%'})

    cl = dbc.Col([div,
                  dsc.spinner_button('Run insight', 'insights-spinner',
                                     id='insights-button', n_clicks=0, color="primary",
                                     style={'margin': '5% 5% 10%'}),
                  # dbc.Button('Run comparison', id='compare-against-shuffle-button',
                  #          n_clicks=0, color="primary", style={'margin': '5% 5% 10%'})
                  ], width=3)

    cr = dbc.Col([html.Div(children=[], id='insights-container')], width=9)

    return dbc.Row(children=[cl, cr])

