import base64
import io
import threading
import available_settings
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import pandas as pd
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from sherlock.sherlockengines import sherlockcomputationhelper as sch


def parse_spreadsheet_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        print('Loaded data')
    except Exception as e:
        print(e)
        return None

    return df


def data_description_visual(dataframe):
    return None


def variables_description_visual(dataframe):
    reports = []
    show_graph = True
    if dataframe.shape[0]>200:
        show_graph = False
        return reports
    total_variables = dataframe.shape[0]
    for r in range(dataframe.shape[0]):
        print('\r', ''.join(['Generating component for variable ', str(r), '/', str(total_variables)]), end='')

        d = dataframe.iloc[r, :]
        h = generate_component(d, show_hist=show_graph)
        reports.append(h)
    return reports


def generate_component(d, show_hist=True):

    c1 = html.Div(children=[html.H4(d['column_name'],
                                    className="alert-heading",
                                    style={'color': 'blue'}),
                            html.H5(d['column_type'], style={'color': 'gray'})],
                  style={'width': '20%', 'display': 'inline-block'})

    c2 = html.Div(children=[' '], style={'width': '5%', 'display': 'inline-block'})

    c3 = html.Div(children=[get_var_report(d)],
                  style={'width': '50%', 'display': 'inline-block'})

    c4 = html.Div(children=[' '], style={'width': '5%', 'display': 'inline-block'})

    img = None
    if show_hist:
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(3, 2))
        ax.hist(d['raw_data'].values)
        out_url = fig_to_uri(fig)
        img = html.Div([html.Img(src=out_url)], style={'width': '15%', 'display': 'inline-block'})

    row = html.Div(children=[c1, c2, c3, c4, img],
                   className="row",
                   style={'width': '90%', 'margin-left': '5%', 'margin-top': '1%'})

    alert = html.Div(children=[row, html.Hr()])

    return alert


def get_var_report(d: pd.DataFrame):
    d_dict = d.to_dict()
    report_table = []

    columns_to_report = {
    'n_null': 'Number of null', 'min': 'min', 'max': 'max', 'bootstrap_mean': 'Bootstrap mean',
        'bootstrap_mean_low': 'Bootstrap mean low', 'bootstrap_mean_high': 'Bootstrap mean high',
        'std': 'Standard deviation', 'skew': 'Skew', 'IQR': 'IRQ', 'is_normal': 'Is normal',
        'p_normal': 'P-value normality'}

    for k in columns_to_report.keys():
        value = d_dict[k]
        if sch.is_numeric(value):
            value = "{:.2f}".format(value)
        else:
            value = str(value)
        name = html.Div([dcc.Markdown([''.join(['**', columns_to_report[k], '**'])]), html.P(value)],
                        style={'width': '30%', 'display': 'inline-block'})
        report_table.append(name)

    final_report = html.Div(report_table)
    return final_report


def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)
