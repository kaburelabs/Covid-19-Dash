import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import datetime
import dash_admin_components as dac
from dash.exceptions import PreventUpdate


app.layout = html.Div(
    [
        # dcc.Location(id="url"),
        dbc.Container(
            [
                navbar(
                    appname="Corona Virus Monitor",
                    logo="assets/fundo_transp-b.png",
                    height="45px",
                ),
                sidebar,
                # collase_buttons,
                display_main_stats(),
                html.Div(id="main_div"),
                charts_buttons,
                charts,
            ]
        )
    ]
)
