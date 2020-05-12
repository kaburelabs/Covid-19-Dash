# Dash APP
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.GRID]
                )

app.layout = html.Div(
    [
        dbc.Row(dbc.Col(html.Div("A single column",
                                 style={"height": "150px", "background": "yellow", "color": "black", "textAlign": "center", "fontSize": "25px"}),
                        sm={"offset": 3, "size": 6},
                        md={"offset": 2, "size": 8},
                        lg={"offset": 1, "size": 10})
                ),
        dbc.Row(
            [
                dbc.Col(html.Div("One of three columns",
                                 style={"background": "white", "margin": "16px 0", "height": "100px", "fontSize": "32px", "color": "black"}),
                        xs=12, width=10, md=6, lg=4),
                dbc.Col(html.Div("One of three columns",
                                 style={"background": "white", "margin": "16px 0", "height": "100px", "fontSize": "32px", "color": "black"}),
                        xs=12, width=10, md=6, lg=4),
                dbc.Col(html.Div("One of three columns",
                                 style={"background": "white", "margin": "16px 0", "height": "100px", "fontSize": "32px", "color": "black"}),
                        xs=12, width=10, md=6, lg=4),
            ]
        ),
    ]
)


if __name__ == '__main__':
    app.run_server(debug=True)
