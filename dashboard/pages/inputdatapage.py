from dashboard.utils import dashstyledcomponents as dsc, componentsstyles
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import available_settings


def overall_page():
    r0 = html.H2('Load data and build Sherlock', style={'margin-top': '3%', 'margin-left': '2%'})
    r1 = row_1()
    r2 = row_2()
    r3 = row_3()

    return [r0, r1, r2[0], r2[1], r2[2], r3[0], r3[1], r3[2]]


def row_1():
    c1 = dbc.Col([
        dcc.Upload(id="upload-data",
                   children=html.Div(
                       ["Drag and drop or click to select a file."]),
                   multiple=False,
                   style={"width": "90%", "height": "60px", "lineHeight": "60px", "borderWidth": "1px",
                          "borderStyle": "dashed", "borderRadius": "2px", "textAlign": "center", "margin": "5%"}
                   ),
        html.Div(id='output-data-upload', style={"margin": "5%"})
    ], width=4)

    c2 = dbc.Col([
        dsc.spinner_button('Load data', 'load-data-spinner', id='load-data-button', n_clicks=0, color="primary",
                           block=True, size='lg'),
    ], width={"size": 2, "offset": 5})

    r = dbc.Row(children=[c1, c2], align='center', style={'margin-top': '2%'})
    return r


def row_2():
    r1 = html.Hr()
    r2 = html.H2('Data summary', style={'margin-top': '3%', 'margin-left': '2%'})
    r3 = html.Div(id='loaded-data-description', children=[])
    return [r1, r2, r3]


def row_3():
    r1 = html.Hr()
    r2 = html.H2('Variables summary', style={'margin-top': '3%', 'margin-left': '2%'})
    r3 = html.Div(id='loaded-variables-description', children=[])
    return [r1, r2, r3]
