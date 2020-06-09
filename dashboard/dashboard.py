import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_bootstrap_components as dbc
# import dash_styled_components as dsc
# import dashboard_pages
import dash_core_components as dcc
from dashboard.pages import inputdatapage, variablesinspectionpage, insightspage
from dashboard import dashboardcallbacks


def build_app():
    app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
    # Add the layout
    app.layout = get_overall_layout()

    # Add callbacks
    dashboardcallbacks.input_content_callbacks(app)
    dashboardcallbacks.variables_inspection_callbacks(app)
    dashboardcallbacks.insights_callbacks(app)

    return app


def get_overall_layout():
    tabs = dbc.Tabs(
        [
            dbc.Tab(html.Div(children=inputdatapage.overall_page()), label="Input data", tab_id='input-content'),
            dbc.Tab(html.Div(children=variablesinspectionpage.overall_page()), label="Variables inspection",
                    tab_id='variables-content'),
            dbc.Tab(html.Div(children=insightspage.overall_page()), label="Insights",
                    tab_id='insights-content'),
        ], id='main-tab', active_tab='input-content', card=True,
        style={'outline': 'True', 'margin-top': '1%', 'margin-left': '1%', 'margin-right': 'auto', 'width': '95%'})

    layout = html.Div(children=[
        html.Div(id='hidden-div', style={'display': 'none'}),
        html.Div(id='hidden-div1', style={'display': 'none'}),
        dbc.Row(id='r1', children=[get_r1()], justify='center'),
        tabs
    ])
    return layout


def get_r1():
    header = html.H1("Sherlock Auto Data Inspector", style={"text-decoration": "none", "color": "grey", 'textAlign': 'left',
                                                          'margin': '2rem 1rem 2rem 5%', 'font-family': 'sans-serif'})

    r1 = dbc.Card(children=[header], style={'margin-top': '1%', 'margin-left': '2%', 'margin-right': 'auto', 'width': '95%'})

    return r1

