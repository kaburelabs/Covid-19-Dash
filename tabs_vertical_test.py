import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.H1("Dash Tabs component demo"),
        html.Div(
            [
                dcc.Tabs(
                    id="tabs-example",
                    value="tab-1-example",
                    children=[
                        dcc.Tab(label="Tab One", value="tab-1-example"),
                        dcc.Tab(label="Tab Two", value="tab-2-example"),
                    ],
                    vertical=True,
                    parent_style={"float": "left"},
                ),
                html.Div(
                    id="tabs-content-example", style={"float": "left", "width": "400"}
                ),
            ]
        ),
    ]
)


@app.callback(
    Output("tabs-content-example", "children"), [Input("tabs-example", "value")]
)
def render_content(tab):
    if tab == "tab-1-example":
        return html.Div(
            [
                html.H3("Tab content 1"),
                dcc.Graph(
                    id="graph-1-tabs",
                    figure={"data": [{"x": [1, 2, 3], "y": [3, 1, 2], "type": "bar"}]},
                ),
            ]
        )
    elif tab == "tab-2-example":
        return html.Div(
            [
                html.H3("Tab content 2"),
                dcc.Graph(
                    id="graph-2-tabs",
                    figure={"data": [{"x": [1, 2, 3], "y": [5, 10, 6], "type": "bar"}]},
                ),
            ]
        )


if __name__ == "__main__":
    app.run_server(debug=True)
