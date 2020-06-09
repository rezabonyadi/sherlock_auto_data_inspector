import pandas as pd
# from dashboard.backend import dataset_interactions

import dash_html_components as html
from dash.dependencies import Input, Output, State
from sherlock.sherlockinterface import Sherlock
# from dashboard.utils import components_layout_helper
# from dashboard.pages import report_layout
import io
import available_settings
import base64
from dashboard.utils import callbackshelpers

variables_storage = {}


def input_content_callbacks(app):
    @app.callback(
        Output("output-data-upload", "children"),
        [Input("upload-data", "contents")],
        [State("upload-data", "filename")]
    )
    def upload_data_file(uploaded_file_contents, uploaded_filename):
        """Uploaded data file"""
        if uploaded_file_contents is not None:
            df = callbackshelpers.parse_spreadsheet_contents(uploaded_file_contents, uploaded_filename)
            if df is not None:
                variables_storage['data'] = df
                variables_storage['sherlock'] = Sherlock()
                variables_storage['sherlock'].load_data(df)

                return [''.join(['File ', uploaded_filename, ' was uploaded'])]
        return ['File was not uploaded']

    @app.callback(
        [Output("load-data-spinner", "children"),
         Output('loaded-data-description', 'children'),
         Output('loaded-variables-description', 'children')],
        [Input("load-data-button", "n_clicks")]
    )
    def build_sherlock(n):
        """Uploaded data file"""
        if n:
            df = variables_storage['data']
            sh_ai = Sherlock()
            variables_storage['sherlock'] = sh_ai
            sh_ai.load_data(df)
            report = sh_ai.visualize_variables_descriptions(None)
            data_report = callbackshelpers.data_description_visual(report['data'])
            variables_report = callbackshelpers.variables_description_visual(report['variables'])

            return ['Built Sherlock...'], data_report, variables_report

        return ['File was not uploaded'], None, None


def variables_inspection_callbacks(app):
    @app.callback(
        [Output("variables-distributions-results-container", "children"),
         Output("variables-distribution-spinner", "children")],
        [Input('variables-distribution-button', 'n_clicks')],
        [State('n-var-per-page', 'value')]
    )
    def run_variables_visualization(n, variables_per_page):
        # available_companies_list = glob.glob("/data/*.csv")
        if n:
            print('graphs to show')
            settings = {'num_var_per_graph': int(variables_per_page)}

            sh_ai = variables_storage['sherlock']
            figs = sh_ai.visualize_variables_distributions(settings)

            imgs = []
            for fig in figs:
                out_url = callbackshelpers.fig_to_uri(fig, bbox_inches='tight')
                imgs.append(html.Img(src=out_url))

            return imgs, 'Done'

        return [], None

    @app.callback(
        [Output("variables-network-results-container", "children"),
         Output("variables-network-spinner", "children")],
        [Input('variables-network-button', 'n_clicks')],
        [State('network-threshold-input', 'value'), State('network-pval-input', 'value')
            , State('correlated-threshold-text', 'value'), State('nodes-sizes-text', 'value')]
    )
    def run_variables_network_visualization(n, network_threshold, p_val, correlated_threshold, nodes_sizes):
        # available_companies_list = glob.glob("/data/*.csv")
        if n:
            sh_ai = variables_storage['sherlock']
            settings = {'network_connections_threshold_variables_relations': float(network_threshold),
                        'p_value_correlations': float(p_val), 'correlated_threshold': float(correlated_threshold),
                        'node_size': int(nodes_sizes), 'show_report': False, 'fig_size': (12, 10),
                        'max_nodes_to_draw_graph': 90}

            fig = sh_ai.visualize_variables_network(settings)
            img = None

            if fig is not None:
                out_url = callbackshelpers.fig_to_uri(fig)
                img = html.Img(src=out_url)

            return img, 'Done'
            # new_tags = input_tags.split(';')

        return [], None

    @app.callback(
        [Output("variables-response-results-container", "children"),
         Output("variables-response-spinner", "children")],
        [Input('variables-response-button', 'n_clicks')],
        [State('use-correction', 'value'), State('p-var-res-relation', 'value')]
    )
    def run_variables_response_visualization(n, use_correction, p_value):
        # available_companies_list = glob.glob("/data/*.csv")
        if n:
            sh_ai = variables_storage['sherlock']
            settings = {'p_variables_response_relation': float(p_value), 'use_correction': use_correction == 'True'}

            figs = sh_ai.visualize_variables_response_relationship(settings)

            imgs = []
            for fig in figs:
                out_url = callbackshelpers.fig_to_uri(fig, bbox_inches='tight')
                imgs.append(html.Img(src=out_url))

            return imgs, 'Done'
            # new_tags = input_tags.split(';')

        return [], None

    @app.callback(
        [Output("all-data-model-container", "children"),
         Output("all-data-model-spinner", "children")],
        [Input('all-data-model-button', 'n_clicks')],
    )
    def run_model_potential(n):
        # available_companies_list = glob.glob("/data/*.csv")
        if n:
            sh_ai = variables_storage['sherlock']
            settings = None
            fig = sh_ai.visualize_all_data_model(settings)

            out_url = callbackshelpers.fig_to_uri(fig, bbox_inches='tight')
            img = html.Img(src=out_url)

            return img, 'Done'
            # new_tags = input_tags.split(';')

        return [], None


def insights_callbacks(app):
    @app.callback(
        [Output("comparison-against-shuffle-container", "children"),
         Output("compare-against-shuffle-spinner", "children")],
        [Input('compare-against-shuffle-button', 'n_clicks')],
        [State('number-runs', 'value'), State('bagging-perc', 'value'),
         State('select-model', 'value'), State('select-scaler', 'value'), State('select-random', 'value'),
         State('train-perc', 'value')]
    )
    def compare_against_shuffle(n, n_runs, bagging_perc, model, scaler, random_selector, train_perc):
        if n:
            scaler_object = available_settings.available_models_preprocessing[scaler]
            model_object = available_settings.available_models[model]
            settings = {'scaler': scaler_object, 'train_perc': train_perc, 'runs': n_runs,
                        'selection_type': random_selector, 'model': model_object,
                        'bootstrap_aggregate_percentage': bagging_perc, 'fig_size': (12, 10)}

            sh_ai = variables_storage['sherlock']
            fig = sh_ai.visualize_compare_against_shuffle(settings)
            out_url = callbackshelpers.fig_to_uri(fig, bbox_inches='tight')
            img = html.Img(src=out_url)

            return img, [f'Done']

            # new_tags = input_tags.split(';')

        return None, [f'Not compared']

    @app.callback(
        [Output("insights-container", "children"),
         Output("insights-spinner", "children")],
        [Input('insights-button', 'n_clicks')],
        [State('number-runs-insights', 'value'), State('bagging-perc-insights', 'value'),
         State('select-model-insights', 'value'), State('select-scaler-insights', 'value'),
         State('select-random-insights', 'value'), State('train-perc-insights', 'value')]
    )
    def insight(n, n_runs, bagging_perc, model, scaler, random_selector, train_perc):
        if n:
            scaler_object = available_settings.available_models_preprocessing[scaler]
            model_object = available_settings.available_models[model]
            settings = {'scaler': scaler_object, 'train_perc': train_perc, 'runs': n_runs,
                        'selection_type': random_selector, 'model': model_object,
                        'bootstrap_aggregate_percentage': bagging_perc, 'fig_size': (12, 10)}

            sh_ai = variables_storage['sherlock']
            figs = sh_ai.visualize_insights(settings)
            imgs = []
            for fig in figs:
                out_url = callbackshelpers.fig_to_uri(fig, bbox_inches='tight')
                imgs.append(html.Img(src=out_url))

            # out_url = fig_to_uri(fig, bbox_inches='tight')
            # img = html.Img(src=out_url)

            return imgs, [f'Done']

            # new_tags = input_tags.split(';')

        return None, [f'Not compared']
