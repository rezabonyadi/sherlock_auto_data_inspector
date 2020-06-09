import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


def NamedDropdown(name, **kwargs):
    return html.Div(
        children=[
            html.P(children=f"{name}:"),
            dcc.Dropdown(**kwargs),
        ],
    )


def NamedInput(name, type="number", **kwargs):
    return html.Div(
        children=[
            html.P(children=f"{name}:"),
            dbc.Input(type=type, **kwargs)
            # dcc.Input(type="number", debounce=True, **kwargs),
        ],
    )


def spinner_button(name, spinner_id, **kwargs):
    return html.Div(
        children=[
            dbc.Button(name, **kwargs),
            dbc.Spinner(html.Div(id=spinner_id)),
        ],
    )


def NamedText(name, type="number", **kwargs):
    return html.Div(
        children=[
            html.P(children=f"{name}:"),
            dbc.Textarea(**kwargs)
            # dcc.Input(type="number", debounce=True, **kwargs),
        ],
    )


def gridify_components(component_list, n_c=2):
    rows = []
    col = 0
    c_r = []
    for component in component_list:
        c_r.append(component)
        col += 1
        if col == n_c:
            # New row is needed
            a_row = dbc.Row(children=c_r, justify='center')
            rows.append(a_row)
            c_r = []
            col = 0
    if col > 0:
        # New row is needed
        a_row = dbc.Row(children=c_r)
        rows.append(a_row)

    return rows


def graph_grid(figs, n_c=2, width=None, height=None):
    style_figures = []
    for fig in figs:
        if width is not None:
            fig.update_layout(width=width, height=height)

        graph = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={'margin': '2% 0% 1% 1%'})

        style_figures.append(graph)

    return gridify_components(style_figures, n_c)
