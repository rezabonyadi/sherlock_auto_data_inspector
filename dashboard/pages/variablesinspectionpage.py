from dashboard.utils import dashstyledcomponents as dsc, componentsstyles
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import available_settings


def overall_page():
    variables_tabs = dbc.Tabs([
        dbc.Tab(html.Div(children=[get_variables_distribution_layout()]),
                label="Variables distributions", tab_id='var-distributions'),
        dbc.Tab(html.Div(children=[get_variables_relationship_layout()]), label="Variables relationships",
                tab_id='var-relations'),
        dbc.Tab(html.Div(children=[get_variables_response_layout()]), label="Variables/response relationships",
                tab_id='var-response-relations'),
        dbc.Tab(html.Div(children=[get_all_data_model()]), label="All data model", tab_id='modeling-potential')
    ])
    content = html.Div(children=variables_tabs,
                       style={'outline': 'True', 'margin-top': '1%', 'margin-left': '1%',
                              'margin-right': 'auto', 'width': '95%'})

    return content


def get_variables_distribution_layout():
    number_of_graphs = \
        dbc.FormGroup(
        [
            dbc.Label("Max number of variables per page", html_for="var-per-page"),
            dbc.Input(id="n-var-per-page", min=3, max=15, step=1, value=5),
        ], style={'margin-left': '5%'}
    )

    cl = dbc.Col([number_of_graphs,
                  dsc.spinner_button('Show variables distribution', 'variables-distribution-spinner',
                                     id='variables-distribution-button', n_clicks=0, color="primary",
                                     style={'margin': '5% 5% 10%'})
                  ], width=3)

    cr = dbc.Col([html.Div(children=[], id='variables-distributions-results-container')], width=9)

    return dbc.Row(children=[cl, cr])


def get_variables_relationship_layout():
    setting_threshold = dbc.FormGroup(
        [
            dbc.Label("Variables network threshold", html_for="network-threshold"),
            dbc.Input(id="network-threshold-input", min=0.01, max=1.0, step=.01, value=0.83),
        ], style={'margin-left': '5%'}
    )

    setting_correlated = dbc.FormGroup(
        [
            dbc.Label("Correlated threshold", html_for="correlated-threshold"),
            dbc.Input(id="correlated-threshold-text", min=0.01, max=1.0, step=.01, value=0.9),
        ], style={'margin-left': '5%'}
    )

    nodes_size = dbc.FormGroup(
        [
            dbc.Label("Nodes size", html_for="nodes-size"),
            dbc.Input(id="nodes-sizes-text", min=100, max=5000, step=10, value=3000),
        ], style={'margin-left': '5%'}
    )

    setting_p_val = dbc.FormGroup(
        [
            dbc.Label("P-value for correlations", html_for="p-val-threshold"),
            dbc.Input(id="network-pval-input", min=0.001, max=0.1, step=.001, value=0.05),
        ], style={'margin-left': '5%'}
    )

    cl = dbc.Col([setting_p_val,
                  setting_threshold,
                  setting_correlated,
                  nodes_size,
                  dsc.spinner_button('Show variables network', 'variables-network-spinner',
                                     id='variables-network-button', n_clicks=0, color="primary",
                                     style={'margin': '5% 5% 10%'})
                  ], width=3)

    cr = dbc.Col([html.Div(children=[], id='variables-network-results-container')], width=9)

    return dbc.Row(children=[cl, cr])


def get_variables_response_layout():
    variable_response_p = \
        dbc.FormGroup(
        [
            dbc.Label("P value for variable response relation", html_for="var-per-page"),
            dbc.Input(id="p-var-res-relation", min=0.0, max=0.1, step=.00001, value=0.05),
        ], style={'margin-left': '5%'}
    )
    use_correction = \
        dbc.FormGroup(
        [
            dbc.Label("Use p correction", html_for="var-per-page"),
            dcc.Dropdown(id="use-correction", options=[
                        {'label': 'Yes', 'value': 'True'},
                        {'label': 'No', 'value': 'False'},
                    ],
                    value='True'),
        ], style={'margin-left': '5%'}
    )

    cl = dbc.Col([use_correction,
                  variable_response_p,
                  dsc.spinner_button('Show variables/response relationship', 'variables-response-spinner',
                  id='variables-response-button', n_clicks=0, color="primary", style={'margin': '5% 5% 10%'})
                  ], width=3)

    cr = dbc.Col([html.Div(children=[], id='variables-response-results-container')], width=9)

    return dbc.Row(children=[cl, cr])


def get_all_data_model():
    cl = dbc.Col([dsc.spinner_button('Show model', 'all-data-model-spinner', id='all-data-model-button',
                             n_clicks=0, color="primary", style={'margin': '5% 5% 10%'})
                  ], width=1)

    cr = dbc.Col([html.Div(children=[], id='all-data-model-container')], width=11)

    return dbc.Row(children=[cl, cr])