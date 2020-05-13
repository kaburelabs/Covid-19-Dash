# Data Libraries
import pandas as pd
import numpy as np
import datetime

# Dash APP
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from python.data import Data
from python.model import Model
from python.result import Result


# Graphs
import plotly.graph_objects as go
import plotly.express as px

# import dash_admin_components as dac

# from application.dash import app

#####################################################################################################################################
# Boostrap CSS and font awesome . Option 1) Run from codepen directly Option 2) Copy css file to assets folder and run locally
#####################################################################################################################################
external_stylesheets = [dbc.themes.SOLAR,
                        'https://codepen.io/unicorndy/pen/GRJXrvP.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
                        "https://fonts.googleapis.com/css2?family=Squada+One&display=swap"]

# #Insert your javascript here. In this example, addthis.com has been added to the web app for people to share their webpage
# external_scripts = [{
#         'type': 'text/javascript', #depends on your application
#         'src': 'insert your addthis.com js here',
#     }]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                meta_tags=[

                    {

                        "name": "viewport",

                        "content": "width=device-width, initial-scale=1, maximum-scale=1",

                    },

                ],
                # external_scripts = external_scripts
                )

app.config.suppress_callback_exceptions = True

app.css.config.serve_locally = True

app.title = 'trich.ai Covid Monitor'

# for heroku to run correctly
server = app.server

# Read data
data = Data()
data.get_data()

# Overwrite your CSS setting by including style locally
colors = {
    'background': '#2D2D2D',
    'text': '#E1E2E5',
    'figure_text': '#ffffff',
    'confirmed_text': 'rgb(117, 190, 255)',
    'deaths_text': 'rgb(225, 116, 108)',
    'recovered_text': 'rgb(123, 194, 145)',
    'highest_case_bg': '#393939',

}

# Creating custom style for  local use
divBorderStyle = {
    'backgroundColor': '#393939',
    'borderRadius': '12px',
    'lineHeight': 0.9,
    "padding": "18px 0px"
}

# Creating custom style for local use
boxBorderStyle = {
    'borderColor': '#393939',
    'borderStyle': 'solid',
    'borderRadius': '10px',
    'borderWidth': 2,
}

######################################
# Retrieve data
######################################

# get data directly from github. The data source provided by Johns Hopkins University.
url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

# Data can also be saved locally and read from local drive
# url_confirmed = 'time_series_covid19_confirmed_global.csv'
# url_deaths = 'time_series_covid19_deaths_global.csv'
# url_recovered = 'time_series_covid19_recovered_global.csv'

df_confirmed = pd.read_csv(url_confirmed)
df_deaths = pd.read_csv(url_deaths)
df_recovered = pd.read_csv(url_recovered)

##############################################################################################################################
# Moving Singapore to the first row in the datatable (You can choose any country of interest to be display on the first row)
##############################################################################################################################


def df_move1st_sg(df_t):

    # Moving Singapore to the first row in the datatable
    df_t["new"] = range(1, len(df_t)+1)
    # df_t.loc[df_t[df_t['Country/Region'] == 'US'].index.values,'new'] = 0
    # print(df_t)
    df_t = df_t.sort_values("new").drop('new', axis=1)
    return df_t

#########################################################################################
# Data preprocessing for getting useful data and shaping data compatible to plotly plot
#########################################################################################


# Total cases
df_confirmed_total = df_confirmed.iloc[:, 4:].sum(axis=0)
df_deaths_total = df_deaths.iloc[:, 4:].sum(axis=0)
df_recovered_total = df_recovered.iloc[:, 4:].sum(axis=0)

# modified deaths dataset for mortality rate calculation
df_deaths_confirmed = df_deaths.copy()
df_deaths_confirmed['confirmed'] = df_confirmed.iloc[:, -1]

# Sorted - df_deaths_confirmed_sorted is different from others, as it is only modified later. Careful of it dataframe structure
df_deaths_confirmed_sorted = df_deaths_confirmed.sort_values(by=df_deaths_confirmed.columns[-2], ascending=False)[
    ['Country/Region', df_deaths_confirmed.columns[-2], df_deaths_confirmed.columns[-1]]]
df_recovered_sorted = df_recovered.sort_values(
    by=df_recovered.columns[-1], ascending=False)[['Country/Region', df_recovered.columns[-1]]]
df_confirmed_sorted = df_confirmed.sort_values(
    by=df_confirmed.columns[-1], ascending=False)[['Country/Region', df_confirmed.columns[-1]]]

# Single day increase
df_deaths_confirmed_sorted['24hr'] = df_deaths_confirmed_sorted.iloc[:, -2] - \
    df_deaths.sort_values(
        by=df_deaths.columns[-1], ascending=False)[df_deaths.columns[-2]]
df_recovered_sorted['24hr'] = df_recovered_sorted.iloc[:, -1] - df_recovered.sort_values(
    by=df_recovered.columns[-1], ascending=False)[df_recovered.columns[-2]]
df_confirmed_sorted['24hr'] = df_confirmed_sorted.iloc[:, -1] - df_confirmed.sort_values(
    by=df_confirmed.columns[-1], ascending=False)[df_confirmed.columns[-2]]

# Aggregate the countries with different province/state together
df_deaths_confirmed_sorted_total = df_deaths_confirmed_sorted.groupby(
    'Country/Region').sum()
df_deaths_confirmed_sorted_total = df_deaths_confirmed_sorted_total.sort_values(
    by=df_deaths_confirmed_sorted_total.columns[0], ascending=False).reset_index()
df_recovered_sorted_total = df_recovered_sorted.groupby('Country/Region').sum()
df_recovered_sorted_total = df_recovered_sorted_total.sort_values(
    by=df_recovered_sorted_total.columns[0], ascending=False).reset_index()
df_confirmed_sorted_total = df_confirmed_sorted.groupby('Country/Region').sum()
df_confirmed_sorted_total = df_confirmed_sorted_total.sort_values(
    by=df_confirmed_sorted_total.columns[0], ascending=False).reset_index()

# Modified recovery csv due to difference in number of rows. Recovered will match ['Province/State','Country/Region']column with Confirmed ['Province/State','Country/Region']
df_recovered['Province+Country'] = df_recovered[['Province/State',
                                                 'Country/Region']].fillna('nann').agg('|'.join, axis=1)
df_confirmed['Province+Country'] = df_confirmed[['Province/State',
                                                 'Country/Region']].fillna('nann').agg('|'.join, axis=1)
df_recovered_fill = df_recovered
df_recovered_fill.set_index("Province+Country")
df_recovered_fill.set_index(
    "Province+Country").reindex(df_confirmed['Province+Country'])
df_recovered_fill = df_recovered_fill.set_index(
    "Province+Country").reindex(df_confirmed['Province+Country']).reset_index()
# split Province+Country back into its respective columns
new = df_recovered_fill["Province+Country"].str.split("|", n=1, expand=True)
df_recovered_fill['Province/State'] = new[0]
df_recovered_fill['Country/Region'] = new[1]
df_recovered_fill['Province/State'].replace('nann', 'NaN')
# drop 'Province+Country' for all dataset
df_confirmed.drop('Province+Country', axis=1, inplace=True)
df_recovered.drop('Province+Country', axis=1, inplace=True)
df_recovered_fill.drop('Province+Country', axis=1, inplace=True)

# Data preprocessing for times series countries graph display
# create temp to store sorting arrangement for all confirm, deaths and recovered.
df_confirmed_sort_temp = df_confirmed.sort_values(
    by=df_confirmed.columns[-1], ascending=False)

df_confirmed_t = df_move1st_sg(df_confirmed_sort_temp)
df_confirmed_t['Province+Country'] = df_confirmed_t[['Province/State',
                                                     'Country/Region']].fillna('nann').agg('|'.join, axis=1)
df_confirmed_t = df_confirmed_t.drop(
    ['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).T

df_deaths_t = df_deaths.reindex(df_confirmed_sort_temp.index)
df_deaths_t = df_move1st_sg(df_deaths_t)
df_deaths_t['Province+Country'] = df_deaths_t[['Province/State',
                                               'Country/Region']].fillna('nann').agg('|'.join, axis=1)
df_deaths_t = df_deaths_t.drop(
    ['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).T
# take note use reovered_fill df
df_recovered_t = df_recovered_fill.reindex(df_confirmed_sort_temp.index)
df_recovered_t = df_move1st_sg(df_recovered_t)
df_recovered_t['Province+Country'] = df_recovered_t[['Province/State',
                                                     'Country/Region']].fillna('nann').agg('|'.join, axis=1)
df_recovered_t = df_recovered_t.drop(
    ['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).T

df_confirmed_t.columns = df_confirmed_t.iloc[-1]
df_confirmed_t = df_confirmed_t.drop('Province+Country')

df_deaths_t.columns = df_deaths_t.iloc[-1]
df_deaths_t = df_deaths_t.drop('Province+Country')

df_recovered_t.columns = df_recovered_t.iloc[-1]
df_recovered_t = df_recovered_t.drop('Province+Country')

df_confirmed_t.index = pd.to_datetime(df_confirmed_t.index)
df_deaths_t.index = pd.to_datetime(df_confirmed_t.index)
df_recovered_t.index = pd.to_datetime(df_confirmed_t.index)

# Highest 10 plot data preprocessing
# getting highest 10 countries with confirmed case
name = df_confirmed_t.columns.str.split("|", 1)
df_confirmed_t_namechange = df_confirmed_t.copy()
# name0 = [x[0] for x in name]
name1 = [x[1] for x in name]
df_confirmed_t_namechange.columns = name1
df_confirmed_t_namechange = df_confirmed_t_namechange.groupby(
    df_confirmed_t_namechange.columns, axis=1).sum()
df_confirmed_t_namechange10 = df_confirmed_t_namechange.sort_values(
    by=df_confirmed_t_namechange.index[-1], axis=1, ascending=False).iloc[:, :10]
df_confirmed_t_stack = df_confirmed_t_namechange10.stack()
df_confirmed_t_stack = df_confirmed_t_stack.reset_index(level=[0, 1])
df_confirmed_t_stack.rename(
    columns={"level_0": "Date", 'level_1': 'Countries', 0: "Confirmed"}, inplace=True)
# getting highest 10 countries with deceased case
name = df_deaths_t.columns.str.split("|", 1)
df_deaths_t_namechange = df_deaths_t.copy()
# name0 = [x[0] for x in name]
name1 = [x[1] for x in name]
df_deaths_t_namechange.columns = name1
df_deaths_t_namechange = df_deaths_t_namechange.groupby(
    df_deaths_t_namechange.columns, axis=1).sum()
df_deaths_t_namechange10 = df_deaths_t_namechange.sort_values(
    by=df_deaths_t_namechange.index[-1], axis=1, ascending=False).iloc[:, :10]
df_deaths_t_stack = df_deaths_t_namechange10.stack()
df_deaths_t_stack = df_deaths_t_stack.reset_index(level=[0, 1])
df_deaths_t_stack.rename(
    columns={"level_0": "Date", 'level_1': 'Countries', 0: "Deceased"}, inplace=True)

# Recreate required columns for map data
map_data = df_confirmed[["Province/State",
                         "Country/Region", "Lat", "Long"]].copy()
map_data['Confirmed'] = df_confirmed.loc[:, df_confirmed.columns[-1]]
map_data['Deaths'] = df_deaths.loc[:, df_deaths.columns[-1]]
map_data['Recovered'] = df_recovered_fill.loc[:, df_recovered_fill.columns[-1]]
map_data['Recovered'] = map_data['Recovered'].fillna(0).astype(
    int)  # too covert value back to int and fillna with zero
# last 24 hours increase
map_data['Deaths_24hr'] = df_deaths.iloc[:, -1] - df_deaths.iloc[:, -2]

map_data['Recovered_24hr'] = df_recovered_fill.iloc[:, -1] - \
    df_recovered_fill.iloc[:, -2]
map_data['Confirmed_24hr'] = df_confirmed.iloc[:, -1] - \
    df_confirmed.iloc[:, -2]
map_data.sort_values(by='Confirmed', ascending=False, inplace=True)
# Moving Singapore to the first row in the datatable
# map_data["new"] = range(1,len(map_data)+1)
# map_data.loc[map_data[map_data['Country/Region'] == 'Brazil'].index.values,'new'] = 0
# map_data = map_data.sort_values("new").drop('new', axis=1)

map_data["text"] = [["Country/Region: {} <br>Province/State: {} <br>Confirmed: {} (+ {} past 24hrs)<br>Deaths: {} (+ {} past 24hrs)<br>Recovered: {} (+ {} past 24hrs)".format(i, j, k, k24, l, l24, m, m24)]
                    for i, j, k, l, m, k24, l24, m24 in zip(map_data['Country/Region'], map_data['Province/State'],
                                                            map_data['Confirmed'], map_data['Deaths'], map_data['Recovered'],
                                                            map_data['Confirmed_24hr'], map_data['Deaths_24hr'], map_data['Recovered_24hr'])]

#############################################################################
# mapbox_access_token keys, not all mapbox function require token to function.
#############################################################################
mapbox_access_token = 'pk.eyJ1Ijoia2FidXJlbGFicyIsImEiOiJjazlraG0xdngwMG14M2xxcHIyOGlzaGhpIn0.pFB9nK-vHIM73dQVAx-LAg'


###########################
# functions to create map
###########################

def gen_map(map_data, zoom, lat, lon):

    return {
        "data": [{
            # specify the type of data to generate, in this case, scatter map box is used
            "type": "scattermapbox",
            "lat": list(map_data['Lat']),  # for markers location
            "lon": list(map_data['Long']),
            # "hoverinfo": "text",
            "hovertext": [["Country/Region: {} <br>Province/State: {} <br>Confirmed: {} (+ {} past 24hrs)<br>Deaths: {} (+ {} past 24hrs)<br>Recovered: {} (+ {} past 24hrs)".format(i, j, k, k24, l, l24, m, m24)]
                          for i, j, k, l, m, k24, l24, m24 in zip(map_data['Country/Region'], map_data['Province/State'],
                                                                  map_data['Confirmed'], map_data['Deaths'], map_data['Recovered'],
                                                                  map_data['Confirmed_24hr'], map_data['Deaths_24hr'], map_data['Recovered_24hr'],)],

            "mode": "markers",
            "name": list(map_data['Country/Region']),
            "marker": {
                    "opacity": 0.7,
                    "size": np.log(map_data['Confirmed'])*4,
            }
        },

        ],
        "layout": dict(
            autosize=True,
            height=350,
            font=dict(color=colors['figure_text']),
            titlefont=dict(color=colors['text'], size='14'),
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
            hovermode="closest",
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            legend=dict(font=dict(size=10), orientation='h'),
            mapbox=dict(
                accesstoken=mapbox_access_token,
                style='mapbox://styles/mapbox/dark-v10',
                center=dict(
                    lon=lon,
                    lat=lat,
                ),
                zoom=zoom,
            )
        ),
    }


##############################################
# Functions to create display for highest cases
##############################################

def high_cases(countryname, total, single, color_word='#63b6ff', confirmed_total=1, deaths=False,):

    if deaths:
        percent = (total/confirmed_total)*100
        return html.Div([html.Span(countryname + ' | ' + f"{int(total):,d}",
                                   style={'backgroundColor': colors['highest_case_bg'], 'borderRadius': '6px', }),
                         html.Span(' +' + f"{int(single):,d}",
                                   style={'color': color_word, 'margin': 2, 'fontWeight': 'bold', 'fontSize': 14, }),
                         html.Span(f' ({percent:.2f}%)',
                                   style={'color': color_word, 'margin': 2, 'fontWeight': 'bold', 'fontSize': 14, }),
                         ],
                        style={
            'textAlign': 'center',
            'color': 'rgb(200,200,200)',
            'fontsize': 12,
        }
        )

    return html.Div([html.Span(countryname + ' | ' + f"{int(total):,d}",
                               style={'backgroundColor': colors['highest_case_bg'], 'borderRadius': '6px', }),
                     html.Span(' +' + f"{int(single):,d}",
                               style={'color': color_word, 'margin': 2, 'fontWeight': 'bold', 'fontSize': 14, }),
                     ],
                    style={
        'textAlign': 'center',
        'color': 'rgb(200,200,200)',
        'fontsize': 12,
    }
    )

#########################################################################
# Convert datetime to Display datetime with following format - 06-Apr-2020
#########################################################################


def datatime_convert(date_str, days_to_add=0):

    format_str = '%m/%d/%y'  # The format
    datetime_obj = datetime.datetime.strptime(date_str, format_str)
    datetime_obj += datetime.timedelta(days=days_to_add)
    return datetime_obj.strftime('%d-%b-%Y')


def return_outbreakdays(date_str):
    format_str = '%d-%b-%Y'  # The format
    datetime_obj = datetime.datetime.strptime(date_str, format_str).date()

    d0 = datetime.date(2020, 1, 22)
    delta = datetime_obj - d0
    return delta.days


noToDisplay = 8

confirm_cases = []
for i in range(noToDisplay):
    confirm_cases.append(high_cases(
        df_confirmed_sorted_total.iloc[i, 0], df_confirmed_sorted_total.iloc[i, 1], df_confirmed_sorted_total.iloc[i, 2]))

deaths_cases = []
for i in range(noToDisplay):
    deaths_cases.append(high_cases(df_deaths_confirmed_sorted_total.iloc[i, 0], df_deaths_confirmed_sorted_total.iloc[
                        i, 1], df_deaths_confirmed_sorted_total.iloc[i, 3], '#ff3b4a', df_deaths_confirmed_sorted_total.iloc[i, 2], True))

confirm_cases_24hrs = []
for i in range(noToDisplay):
    confirm_cases_24hrs.append(high_cases(df_confirmed_sorted_total.sort_values(by=df_confirmed_sorted_total.columns[-1], ascending=False).iloc[i, 0],
                                          df_confirmed_sorted_total.sort_values(
                                              by=df_confirmed_sorted_total.columns[-1], ascending=False).iloc[i, 1],
                                          df_confirmed_sorted_total.sort_values(
                                              by=df_confirmed_sorted_total.columns[-1], ascending=False).iloc[i, 2],
                                          ))

deaths_cases_24hrs = []
for i in range(noToDisplay):
    deaths_cases_24hrs.append(high_cases(df_deaths_confirmed_sorted_total.sort_values(by=df_deaths_confirmed_sorted_total.columns[-1], ascending=False).iloc[i, 0],
                                         df_deaths_confirmed_sorted_total.sort_values(
                                             by=df_deaths_confirmed_sorted_total.columns[-1], ascending=False).iloc[i, 1],
                                         df_deaths_confirmed_sorted_total.sort_values(
                                             by=df_deaths_confirmed_sorted_total.columns[-1], ascending=False).iloc[i, 3],
                                         '#ff3b4a',
                                         df_deaths_confirmed_sorted_total.sort_values(
                                             by=df_deaths_confirmed_sorted_total.columns[-1], ascending=False).iloc[i, 2],
                                         True))

####################################################
# Prepare plotly figure to attached to dcc component
# Global outbreak Plot
####################################################
# Change date index to datetimeindex and share x-axis with all the plot


def draw_global_graph(df_confirmed_total, df_deaths_total, df_recovered_total, graph_type='Total Cases'):
    df_confirmed_total.index = pd.to_datetime(df_confirmed_total.index)

    if graph_type == 'Daily Cases':
        df_confirmed_total = (
            df_confirmed_total - df_confirmed_total.shift(1)).drop(df_confirmed_total.index[0])
        df_deaths_total = (
            df_deaths_total - df_deaths_total.shift(1)).drop(df_deaths_total.index[0])
        df_recovered_total = (
            df_recovered_total - df_recovered_total.shift(1)).drop(df_recovered_total.index[0])

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_confirmed_total.index, y=df_confirmed_total,
                             mode='lines+markers',
                             name='Confirmed',
                             line=dict(color='#3372FF', width=2),
                             fill='tozeroy',))
    fig.add_trace(go.Scatter(x=df_confirmed_total.index, y=df_recovered_total,
                             mode='lines+markers',
                             name='Recovered',
                             line=dict(color='#33FF51', width=2),
                             fill='tozeroy',))
    fig.add_trace(go.Scatter(x=df_confirmed_total.index, y=df_deaths_total,
                             mode='lines+markers',
                             name='Deaths',
                             line=dict(color='#FF3333', width=2),
                             fill='tozeroy',))

    fig.update_layout(
        title="General Cases",
        title_x=.5,
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color=colors['figure_text'],
        ),
        legend=dict(
            x=0.02,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=5
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0,
                    r=0,
                    t=65,
                    b=0
                    ),
        height=350,

    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')

    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')
    return fig

####################################################
# Function to plot Highest 10 countries cases
####################################################


def draw_highest_10(df_confirmed_t_stack, df_deaths_t_stack, graphHigh10_type='Confirmed Cases'):

    if graphHigh10_type == 'Confirmed Cases':
        fig = px.line(df_confirmed_t_stack, x="Date", y="Confirmed", color='Countries',
                      color_discrete_sequence=px.colors.qualitative.Light24)
    else:
        fig = px.line(df_deaths_t_stack, x="Date", y="Deceased", color='Countries',
                      title='Deceased cases', color_discrete_sequence=px.colors.qualitative.Light24)

    fig.update_layout(
        title="World Total Corona Cases ",
        title_x=.5,
        xaxis_title=None,
        yaxis_title=None,
        font=dict(
            family="Courier New, monospace",
            size=14,
            color=colors['figure_text'],
        ),
        legend=dict(
            x=0.02,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=9,
                color=colors['figure_text']
            ),
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0,
                    r=0,
                    t=65,
                    b=0
                    ),
        height=350,

    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')

    # fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')

    return fig


####################################################
# Function to plot Single Country Scatter Plot
####################################################

def draw_singleCountry_Scatter(df_confirmed_t, df_deaths_t, df_recovered_t, selected_row=0, daily_change=False):

    if daily_change:
        df_confirmed_t = (df_confirmed_t - df_confirmed_t.shift(1)
                          ).drop(df_confirmed_t.index[0])
        df_deaths_t = (df_deaths_t - df_deaths_t.shift(1)
                       ).drop(df_deaths_t.index[0])
        df_recovered_t = (df_recovered_t - df_recovered_t.shift(1)
                          ).drop(df_recovered_t.index[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_confirmed_t.index, y=df_confirmed_t.iloc[:, selected_row],
                             mode='lines+markers',
                             name='Confirmed',
                             line=dict(color='#3372FF', width=2),
                             fill='tozeroy',))
    fig.add_trace(go.Scatter(x=df_recovered_t.index, y=df_recovered_t.iloc[:, selected_row],
                             mode='lines+markers',
                             name='Recovered',
                             line=dict(color='#33FF51', width=2),
                             fill='tozeroy',))
    fig.add_trace(go.Scatter(x=df_deaths_t.index, y=df_deaths_t.iloc[:, selected_row],
                             mode='lines+markers',
                             name='Deceased',
                             line=dict(color='#FF3333', width=2),
                             fill='tozeroy',))

    new = df_recovered_t.columns[selected_row].split("|", 1)
    if new[0] == 'nann':
        title = new[1]
    else:
        title = new[1] + " - " + new[0]

    fig.update_layout(
        title=title + ' (Total Cases)',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            x=0.02,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=5
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0, t=65, b=0),
        height=350,

    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')

    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')

    return fig


####################################################
# Function to plot Single Country Bar with scatter Plot
####################################################

def draw_singleCountry_Bar(df_confirmed_t, df_deaths_t, df_recovered_t, selected_row=0, graph_line='Line Chart'):

    df_confirmed_t = (df_confirmed_t - df_confirmed_t.shift(1)
                      ).drop(df_confirmed_t.index[0])
    df_deaths_t = (df_deaths_t - df_deaths_t.shift(1)
                   ).drop(df_deaths_t.index[0])
    df_recovered_t = (df_recovered_t - df_recovered_t.shift(1)
                      ).drop(df_recovered_t.index[0])

    fig = go.Figure()
    if graph_line == 'Line Chart':
        fig.add_trace(go.Bar(x=df_confirmed_t.index, y=df_confirmed_t.iloc[:, selected_row],
                             name='Confirmed',
                             marker_color='#3372FF'
                             ))
        fig.add_trace(go.Bar(x=df_recovered_t.index, y=df_recovered_t.iloc[:, selected_row],
                             name='Recovered',
                             marker_color='#33FF51'
                             ))
        fig.add_trace(go.Bar(x=df_deaths_t.index, y=df_deaths_t.iloc[:, selected_row],
                             name='Deceased',
                             marker_color='#FF3333'
                             ))

    else:
        fig.add_trace(go.Scatter(x=df_confirmed_t.index, y=df_confirmed_t.iloc[:, selected_row],
                                 mode='lines+markers',
                                 name='Confirmed',
                                 line=dict(color='#3372FF', width=2),
                                 fill='tozeroy',))
        fig.add_trace(go.Scatter(x=df_recovered_t.index, y=df_recovered_t.iloc[:, selected_row],
                                 mode='lines+markers',
                                 name='Recovered',
                                 line=dict(color='#33FF51', width=2),
                                 fill='tozeroy',))
        fig.add_trace(go.Scatter(x=df_deaths_t.index, y=df_deaths_t.iloc[:, selected_row],
                                 mode='lines+markers',
                                 name='Deceased',
                                 line=dict(color='#FF3333', width=2),
                                 fill='tozeroy',))

    new = df_recovered_t.columns[selected_row].split("|", 1)
    if new[0] == 'nann':
        title = new[1]
    else:
        title = new[1] + " - " + new[0]

    fig.update_layout(
        title=title + ' (Daily Cases)',
        barmode='stack',
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff",
        ),
        legend=dict(
            x=0.02,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color=colors['figure_text']
            ),
            bgcolor=colors['background'],
            borderwidth=5
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        margin=dict(l=0, r=0, t=65, b=0),
        height=350,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')

    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#3A3A3A')

    return fig


# Creating custom style for  local use
divBorderStyle_nav = {
    'backgroundColor': '#393939',
    'borderBottomLeftRadius': '12px',
    'borderBottomRightRadius': '12px',
    'lineHeight': 0.9,
    "padding": "18px 36px"
}


def navbar(logo="/assets/logo-placeholder.png", height="35px",  appname="PlaceHolder Name"):

    navbar = dbc.Navbar(
        [sidebar,
            dbc.Col(html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div("trich.ai", className="trich-navbar white", style={"fontSize": "2.5em"}))
                        # html.Img(src=logo, height=height)),
                    ],
                    align="center",
                    # no_gutters=True,
                ),
                href="https://trich.ai",
            ), width={"offset": 1, "size": 5}),
            dbc.Col(dbc.NavbarBrand(
                appname, className="ml-4 font-sm white", style={"color": "white"}), width={"size": 6, "order": "last"}),
         ],
        color="#393939", className="bottom32",  # className="bottom16",
        # style={'height': '100px', "borderBottom":".5px solid lightgrey", "padding":"18px 0px"}
        style=divBorderStyle_nav
        # dark=True,
    )

    return navbar


divBorderStyle_disp_first = {
    'backgroundColor': '#393939',
    'borderRadius': '12px',
    'lineHeight': 0.9,
    "padding": "18px 24px", "marginBottom": "30px"}


def display_main_stats():

    total_cases = dbc.Col(html.Div([
        html.Div(children='Total Cases: ',
                 className="text-center text-conf bold font-md"
                 ),
        html.Div(f"{df_confirmed_total[-1]:,d}",
                 className="text-center text-conf font-xl"
                 ),
        html.Div('Increase: +' + f"{df_confirmed_total[-1] - df_confirmed_total[-2]:,d}"
                 + ' (' + str(round(((df_confirmed_total[-1] - df_confirmed_total[-2])/df_confirmed_total[-1])*100, 2)) + '%)',
                 id="total-cases-display",
                 className="text-center text-conf font-sm"
                 ),
        dbc.Tooltip(
            "Increase on the Past 24hrs", style={"fontSize": "15px"},
            target="total-cases-display"),

    ], className="border-div"
    ),
        lg={"size": 4, "offset": 0},
        md={"size": 6, "offset": 3},
        sm={"size": 8, "offset": 2},
        xs={"size": 10, "offset": 1},
        style={"marginBottom": "32px"}
    )

    total_deaths = dbc.Col(html.Div([
        html.Div(children='Total Deaths: ', className="text-center text-death bold font-md"
                 ),
        html.Div(f"{df_deaths_total[-1]:,d}",
                 className="text-center text-death font-xl"
                 ),
        html.Div('Mortality Rate: ' + str(round(df_deaths_total[-1] / df_confirmed_total[-1] * 100, 3)) + '%',
                 className="text-center text-death font-sm"
                 )

    ], className="border-div"
    ), lg={"size": 4, "offset": 0},
        md={"size": 6, "offset": 0},
        sm={"size": 8, "offset": 2},
        xs={"size": 10, "offset": 1},
        style={"marginBottom": "32px"}
    )

    total_recovered = dbc.Col(html.Div([
        html.Div(children='Total Recovered: ', className="text-center text-recov bold font-md"
                 ),
        html.Div(f"{df_recovered_total[-1]:,d}",
                 className="text-center text-recov font-xl"
                 ),
        html.Div('Recovery Rate: ' + str(round(df_recovered_total[-1]/df_confirmed_total[-1] * 100, 3)) + '%',
                 className="text-center text-recov font-sm"
                 ),
    ], className="border-div"
    ), lg={"size": 4, "offset": 0},
        md={"size": 6, "offset": 0},
        sm={"size": 8, "offset": 2},
        xs={"size": 10, "offset": 1},
        style={"marginBottom": "32px"}
    )

    display = dbc.Row([total_cases, total_deaths,
                       total_recovered], className="bottom32")

    return display


title_application_display = html.Div(children='COVID-19 Global Cases',
                                     className="text-white text-center dark font-lg")
charts_buttons = html.Div(
    [
        dbc.Row([
            dbc.Col(
                dcc.RadioItems(
                    id='graph-type',
                    options=[{'label': i, 'value': i}
                             for i in ['Total Cases', 'Daily Cases']],
                    value='Total Cases',
                    labelStyle={'display': 'inline-block',
                                'padding': "0 12px"},
                    style={
                        'fontSize': 18,
                    },

                ), width=6),
            dbc.Col(
                dcc.RadioItems(
                    id='graph-high10-type',
                    options=[{'label': i, 'value': i}
                             for i in ['Confirmed Cases', 'Death Cases']],
                    value='Confirmed Cases',
                    labelStyle={'display': 'inline-block',
                                'padding': "0 12px"},
                    style={
                        'fontSize': 18}
                ), width=6)
        ])

    ], style={"display": None}
)


def map_scatter_corona(data, zoom, lat_focus, long_focus, title):
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=data.Lat,
        lon=data.Long,

        mode='markers',
        marker=go.scattermapbox.Marker(
            size=np.log1p(data["Confirmed"]),
            color='rgb(255, 0, 0)',
            opacity=0.7
        ),
        text=data['Country/Region'],
        hoverinfo='text', hovertext=data["text"]
    ))

    fig.add_trace(go.Scattermapbox(
        lat=data.Lat,
        lon=data.Long,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=8,
            color='rgb(242, 177, 172)',
            opacity=0.7
        ),
        hoverinfo='none'
    ))

    fig.update_layout(
        title=title,
        autosize=True,
        font=dict(color=colors['figure_text']),
        # titlefont=dict(color=colors['text'], size='14'),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        hovermode="closest",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        title_x=.5,
        height=350,
        # width=1140,
        showlegend=False,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=lat_focus,
                lon=long_focus
            ),
            pitch=1,
            zoom=zoom,
            style='dark'
        ),
    )

    return fig


display_highest = dbc.Row([

    dbc.Col([
        html.Div([html.Div('Countries with highest cases: ', className="text-white font-sm"
                           ),
                  html.Div(' + past 24hrs',
                           className="text-conf bold font-sm")
                  ], className="text-display text-center bg-display-cases radius12 font-sm padding8"
                 ),
        html.Div(confirm_cases),
    ],
        # className="three columns",
        width={"size": 12, "offset": 0}, md={"size": 6, "offset": 0}, lg={"size": 3, "offset": 0}
    ),

    dbc.Col([
            html.Div([html.Div('Countries with highest mortality: ',
                               ),
                      html.Div(' + past 24hrs (Mortality Rate)',
                               style={'color': '#f2786f',
                                      'fontWeight': 'bold', 'fontSize': 14, })
                      ], className="text-display text-center bg-display-deaths radius12 font-sm padding8"
                     ),

            html.Div(deaths_cases),
            ],
            # className="three columns",
            width={"size": 12, "offset": 0}, md={"size": 6, "offset": 0}, lg={"size": 3, "offset": 0}
            ),

    dbc.Col([
        html.Div([html.Div('Single day highest cases: ',
                           ),
                  html.Div(' + past 24hrs',
                           style={'color': colors['confirmed_text'],
                                  'fontWeight': 'bold', 'fontSize': 14, })
                  ], className="text-display text-center bg-display-cases radius12 font-sm padding8"
                 ),

        html.Div(confirm_cases_24hrs),
    ],
        # className="three columns",
        width={"size": 12, "offset": 0}, md={"size": 6, "offset": 0}, lg={"size": 3, "offset": 0}
    ),

    dbc.Col([

            html.Div([html.Div('Single day highest mortality: ',
                               ),
                      html.Div(' + past 24hrs (Mortality Rate)',
                               style={'color': '#f2786f',
                                      'fontWeight': 'bold', 'fontSize': 14, })
                      ], className="text-display text-center bg-display-deaths radius12 font-sm padding8"
                     ),

            html.Div(deaths_cases_24hrs),
            ],
            # className="three columns",
            width={"size": 12, "offset": 0}, md={"size": 6, "offset": 0}, lg={"size": 3, "offset": 0}
            ),
], className="left text-white bgGrey padding16",
    # style={
    #     'textAlign': 'left',
    #     'color': colors['text'],
    #     'backgroundColor': colors['background'],
    #     'padding': 20}
)


def main_text_structure():

    header = dbc.Col(html.Div(children='Covid-19 (Coronavirus) Interactive Outbreak Tracker', className="text-white dark left font-lg bottom16"
                              ),
                     width=12
                     )
    days_calc = dbc.Col([html.Span('Dashboard: Covid-19 outbreak. (Updated once a day, based on consolidated last day total) Last Updated: ',
                                   className="text-white font-sm"
                                   ),
                         html.Span(datatime_convert(df_confirmed.columns[-1], 1) + '  00:01 (UTC).',
                                   className="text-white font-sm"),
                         ], width=12)

    outbreak_days = dbc.Col([
        html.Span('Outbreak since 22-Jan-2020: ',
                  className="text-white"
                  ),
        html.Span(str(return_outbreakdays(datatime_convert(df_confirmed.columns[-1], 1))) + '  days.',
                  className="text-conf"
                  )
    ], width=12, className="bottom32"
    )

    row_app_starting = dbc.Row([header, days_calc, outbreak_days],
                               style={"width": "90%", "margin": "0 auto"})

    return row_app_starting


# Input
inputs = dbc.FormGroup([
    html.Div("Select a Country", style={"textAlign": "center"}),
    dcc.Dropdown(id="country", options=[
                 {"label": x, "value": x} for x in data.countrylist], value="World", style={"color": "black"})
], style={"width": "90%", "margin": "0 auto", "marginBottom": "32px"})


charts = dbc.Row([
    dbc.Col([
        dcc.RadioItems(
            id='graph-type',
            options=[{'label': i, 'value': i}
                     for i in ['Total Cases', 'Daily Cases']],
            value='Total Cases',
            labelStyle={'display': 'inline-block',
                        'padding': "0 12px"},
            style={
                'fontSize': 18,
            },

        ),
        dcc.Loading(
            dcc.Graph(
                id='global-graph'
            ), type="graph")], width=12, sm=12, lg=6, style={"marginBottom": "32px"}
    ),
    dbc.Col([
        dcc.RadioItems(
            id='graph-high10-type',
            options=[{'label': i, 'value': i}
                     for i in ['Confirmed Cases', 'Death Cases']],
            value='Confirmed Cases',
            labelStyle={'display': 'inline-block',
                        'padding': "0 12px"},
            style={
                'fontSize': 18}
        ),
        dcc.Loading(
            dcc.Graph(
                id='high10-graph'
            ), type="graph")], width=12, sm=12, lg=6, style={"marginBottom": "32px"}
    )
])

# html.Div(
#     [
#         dbc.Row([
#             dbc.Col(
#                 dcc.RadioItems(
#                     id='graph-type',
#                     options=[{'label': i, 'value': i}
#                              for i in ['Total Cases', 'Daily Cases']],
#                     value='Total Cases',
#                     labelStyle={'display': 'inline-block',
#                                 'padding': "0 12px"},
#                     style={
#                         'fontSize': 18,
#                     },

#                 ), width=6),
#             dbc.Col(
#                 dcc.RadioItems(
#                     id='graph-high10-type',
#                     options=[{'label': i, 'value': i}
#                              for i in ['Confirmed Cases', 'Death Cases']],
#                     value='Confirmed Cases',
#                     labelStyle={'display': 'inline-block',
#                                 'padding': "0 12px"},
#                     style={
#                         'fontSize': 18}
#                 ), width=6)
#         ])

#     ], style={"display": None}
# )
divBorderStyle_sidebar = {
    'backgroundColor': '#393939',
    'lineHeight': 0.9,
    "width": "248px",
    "zIndex": "999"


}

sidebar_header = dbc.Row(
    [
        # dbc.Col(html.H5("Features Panel", style={"paddingLeft": "16px"})
        #         # html.Img(src="assets/fundo_transp-b.png", height="35px"), style={"padding":"0 0 0 18px"}
        #         ),  # html.Div("Select: ", className="display-4")),
        dbc.Col(
            [
                html.Button(
                    # use the Bootstrap navbar-toggler classes to style
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    # the navbar-toggler classes don't set color
                    style={
                        "color": "rgb (0, 0, 0.1)",
                        "border-color": "rgb (0, 0, 0.1)",
                    },
                    id="navbar-toggle",
                ),
                html.Button(
                    # use the Bootstrap navbar-toggler classes to style
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    # the navbar-toggler classes don't set color
                    style={
                        "color": "#c9b49e",
                        "border-color": "#c9b49e",
                    },
                    id="sidebar-toggle",
                ),
            ],
            # the column containing the toggle will be only as wide as the
            # toggle, resulting in the toggle being right aligned
            width={"size": 1, "offset": 9},
            # vertically align the toggle in the center
            align="center",
        ),
    ]
)

mensagem = dcc.Markdown('''
                            ## Technological Stack:
                            * Plotly - https://github.com/plotly/dash
                            * CSS/Boostrap - https://dash-bootstrap-components.opensource.faculty.ai/docs/
                            * pandas - https://pandas.pydata.org/
                            * Numpy - https://numpy.org/
                            * Scipy - https://www.scipy.org/
                            * Scikit-learn - https://scikit-learn.org/stable/

                            ## Dataset:
                            * provided by Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) - https://systems.jhu.edu/

                            ## References and inspirations:
                            * aatisibh: https://aatishb.com/covidtrends/
                            * App reference to build this one: https://covid19-dashboard-online.herokuapp.com/
                            * ARticle Reference of the Forecasting: https://towardsdatascience.com/how-to-embed-bootstrap-css-js-in-your-python-dash-app-8d95fc9e599e


                            ''')


sidebar = html.Div(
    [
        sidebar_header,
        # we wrap the horizontal rule and short blurb in a div that can be
        # hidden on a small screen
        html.Div(
            [
                html.Hr(),
                html.Div(
                    "Select the category of information" " you are interested in.",
                    className="lead", style={"fontSize": "18px", "width": "80%", "margin": "0 auto 16px"}
                ),
            ],
            id="blurb",
        ),
        # use the Collapse component to animate hiding / revealing links
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.Button("Display and Stats", id="btn_display", color="info", style={
                               "width": "80%", 'margin': '0 auto 16px'}),
                    dbc.Button("Chart Distribution", id="btn_distribution", color="info",  style={
                               "width": "80%", 'margin': '0 auto 16px'}),
                    dbc.Button("Tables & Map", id="btn_tabs", color="info", style={
                               "width": "80%", 'margin': '0 auto 16px'}),
                    dbc.Button("Forecasting", id="btn_maps",  color="info", style={
                               "width": "80%", 'margin': '0 auto 16px'}),
                    dbc.Button("Sources", id="open", color="info", style={
                               "width": "80%", 'margin': '0 auto'}),
                ],
                vertical=True,
                pills=False,
            ),
            id="collapse",
        ),
    ],
    id="sidebar", style=divBorderStyle_sidebar
)

map_charts_group = dbc.Row([
    dbc.Col(
        [
            dcc.Graph(id='map-graph-group'
                      )
        ], width=10
    )
])

map_charts = dbc.Row([
    dbc.Col(
        [
            dcc.Graph(id='map-graph',
                      )
        ], width=10
    )
])

table_values = dbc.Row([
    dbc.Col(
        [
            dt.DataTable(
                data=map_data.to_dict('records'),
                columns=[
                    {"name": i, "id": i, "deletable": False, "selectable": True} for i in ['Province/State', 'Country/Region',
                                                                                           'Confirmed',
                                                                                           'Deaths', 'Recovered']
                ],
                fixed_rows={'headers': True, 'data': 0},
                style_header={
                    'backgroundColor': 'rgb(30, 30, 30)',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'backgroundColor': 'rgb(100, 100, 100)',
                    'color': colors['text'],
                    'maxWidth': 0,
                    'fontSize':14,
                },
                style_table={
                    'maxHeight': '350px',
                    'overflowY': 'auto'
                },
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',

                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'even'},
                        'backgroundColor': 'rgb(60, 60, 60)',
                    },
                    {
                        'if': {'column_id': 'Confirmed'},
                        'color': colors['confirmed_text'],
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': 'Deaths'},
                        'color': colors['deaths_text'],
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': 'Recovered'},
                        'color': colors['recovered_text'],
                        'fontWeight': 'bold'
                    },
                ],
                style_cell_conditional=[
                    {'if': {'column_id': 'Province/State'},
                        'width': '26%'},
                    {'if': {'column_id': 'Country/Region'},
                        'width': '26%'},
                    {'if': {'column_id': 'Confirmed'},
                        'width': '16%'},
                    {'if': {'column_id': 'Deaths'},
                        'width': '11%'},
                    {'if': {'column_id': 'Recovered'},
                        'width': '16%'},
                ],
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="single",
                row_selectable="single",
                row_deletable=False,
                selected_columns=[],
                selected_rows=[],
                page_current=0,
                page_size=1000,
                id='datatable'
            ),
        ],
        width=12, md=12, lg=6, xl=6, className="bottom32"
    ),
    dbc.Col(
        [
            dcc.Graph(id='map-graph-group'
                      )
        ], width=12, md=12, lg=6, xl=6, className="bottom32"
    )
])

dropdowns = dbc.Row([
    dbc.Col(
        dcc.Dropdown(
            options=[{'label': i, 'value': i}
                     for i in map_data["Country/Region"].unique()],
            placeholder="Select a Country", id="dropdown-country", value="Brazil"),
        width=4),
    dbc.Col(
        dcc.Dropdown(
            placeholder="Select the Province",
            id="dropdown-province"),
        width=4)
])

custom_graphs = dbc.Row(
    [
        dbc.Col(
            [
                dcc.Graph(id='line-graph',
                          )
            ], md=6, width=12, style={"marginBottom": "64px"}
        ),
        dbc.Col(
            [
                dcc.Graph(id='bar-graph',
                          )
            ], md=6, width=12, style={"marginBottom": "64px"}
        ),

    ], style={
        'textAlign': 'left',
        'color': colors['text'],
        'backgroundColor': colors['background']}
)


app.layout = html.Div([
    # sidebar,
    navbar(appname="Corona Virus Monitor",
           logo="assets/fundo_transp-b.png", height="40px"),
    dbc.Container([

                  # collase_buttons,
                  main_text_structure(),
                  display_main_stats(),
                  title_application_display,
                  dcc.Loading(
                      html.Div(id="main_div"), type="cube")
                  # modal)

                  ], style={"maxWidth": "1140px"})

])

# charts = dbc.Row([
#     dbc.Col(
#         dcc.Graph(
#             id='global-graph'
#         ), sm=12, width=12, md=12, lg=6
#     ),
#     dbc.Col(
#         dcc.Graph(
#             id='high10-graph'
#         ), sm=12, width=12, md=12, lg=6)
# ])

forecast = html.Div([
    dbc.Row(dbc.Col(inputs, width={"size": 10, "offset": 1}, sm={"size": 8, "offset": 2}, md={"size": 6, "offset": 3}, lg={"size": 4, "offset": 4},
                    style={"margin": "0 auto", "width": "100%"}), ),
    # Body
    dbc.Row([
        # input + panel
        dbc.Col(md=5, lg=4, children=[
            # html.Br(), html.Br(), html.Br(),
            html.Div(id="output-panel")
        ], style={"margin": "0 auto 32px", "width": "100%"}),
        # plots
        dbc.Col(md=12, lg=8, children=[
            dbc.Col(html.Div("Forecast 30 days from today"),
                    style={"width": "100%", "margin": "0 auto"}),
            dbc.Tabs(className="nav nav-pills", children=[
                dbc.Tab(dcc.Graph(id="plot-total"),
                        label="Total cases"),
                dbc.Tab(dcc.Graph(id="plot-active"),
                        label="Active cases")
            ])
        ])
    ], style={"marginBottom": "64px"})
])


@app.callback(
    Output('main_div', 'children'),
    [Input('btn_display', 'n_clicks_timestamp'),
     Input('btn_distribution', 'n_clicks_timestamp'),
     Input('btn_maps', 'n_clicks_timestamp'),
     Input('btn_tabs', 'n_clicks_timestamp'),
     Input('open', 'n_clicks_timestamp')])
def update_graph(btn_display, btn_distribution, btn_maps, btn_tabs, btn_src):
    print(btn_display, btn_distribution, btn_maps, btn_tabs, btn_src)

    btn_df = pd.DataFrame({"display": [btn_display], "distribution": [btn_distribution],
                           "maps": [btn_maps], "tab": [btn_tabs], "source": [btn_src]})

    btn_df = btn_df.fillna(0)

    print(btn_df.idxmax(axis=1).values)

    if btn_df.idxmax(axis=1).values == "display":
        return display_highest

    if btn_df.idxmax(axis=1).values == "distribution":
        return [charts]

    if btn_df.idxmax(axis=1).values == "maps":
        return forecast

    if btn_df.idxmax(axis=1).values == "tab":
        return html.Div([table_values, custom_graphs])

    if btn_df.idxmax(axis=1).values == "source":
        return html.Div(mensagem, style={"width": "80%", "margin": "0 auto", "height": "450px"})


@app.callback(
    Output('global-graph', 'figure'),
    [Input('graph-type', 'value')])
def update_graph(graph_type):

    general_values = draw_global_graph(
        df_confirmed_total, df_deaths_total, df_recovered_total, graph_type)
    # print(general_values)
    return general_values


@app.callback(
    Output('high10-graph', 'figure'),
    [Input('graph-high10-type', 'value')])
def update_graph_high10(graph_high10_type):

    fig_high10 = draw_highest_10(

        df_confirmed_t_stack, df_deaths_t_stack, graph_high10_type)

    return fig_high10


@app.callback(
    [Output('map-graph-group', 'figure'),
     Output('line-graph', 'figure'),
     Output('bar-graph', 'figure')],
    [Input('datatable', 'data'),
     Input('datatable', 'selected_rows'),
     # Input('graph-line','value')
     ])
def map_selection(data, selected_rows):
    # print("MAPA MULTIPLO", data, selected_rows)
    aux = pd.DataFrame(data)
    temp_df = aux.iloc[selected_rows, :]
    zoom = 1

    graph_line = "Line Chart"

    if len(selected_rows) == 0:
        fig1 = draw_global_graph(
            df_confirmed_total, df_deaths_total, df_recovered_total)
        fig2 = draw_singleCountry_Bar(
            df_confirmed_t, df_deaths_t, df_recovered_t, 0, graph_line)
        return map_scatter_corona(aux, zoom, 10, -5, title='bla'), fig1, fig2
    else:
        fig1 = draw_singleCountry_Scatter(
            df_confirmed_t, df_deaths_t, df_recovered_t, selected_rows[0])
        fig2 = draw_singleCountry_Bar(
            df_confirmed_t, df_deaths_t, df_recovered_t, selected_rows[0], graph_line)
        zoom = 2
        return map_scatter_corona(aux, zoom, temp_df['Lat'].iloc[0], temp_df['Long'].iloc[0], title="bla"), fig1, fig2


@app.callback(
    Output('dropdown-province', 'options'),
    [Input('dropdown-country', 'value')])
def map_selection(dropdown_country):
    # print("MAPA UNICO", dropdown_country)
    var_province = [{'label': i, 'value': i}
                    for i in map_data[map_data["Country/Region"] == dropdown_country]["Province/State"].unique()]
    # print(var_province)
    return var_province


@app.callback(
    Output('map-graph', 'figure'),
    [Input('dropdown-country', 'value')])
def map_selection(dropdown_country):
    # print("MAPA UNICO", dropdown_country)

    # aux = pd.DataFrame(data)
    # temp_df = aux.iloc[selected_rows, :]
    zoom = 1
    if len(dropdown_country) is None:
        return map_scatter_corona(map_data, 1, 15, 40, title="Corona Virus Around the world")
    else:
        return map_scatter_corona(map_data, 2, map_data[map_data['Country/Region'] == dropdown_country]["Long"].iloc[0], map_data[map_data['Country/Region'] == dropdown_country]["Long"].iloc[0], title="Corona Virus Around the world")


@app.callback(
    Output("sidebar", "className"),
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className")])
def toggle_classname(n, classname):
    if n and classname == "":
        return "collapsed"
    return ""


@app.callback(
    Output("collapse", "is_open"),
    [Input("navbar-toggle", "n_clicks")],
    [State("collapse", "is_open")])
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# @app.callback(
#     Output("modal", "is_open"),
#     [Input("open", "n_clicks"), Input("close", "n_clicks")],
#     [State("modal", "is_open")],
# )
# def toggle_modal(n1, n2, is_open):
#     if n1 or n2:
#         return not is_open
#     return is_open


# Python function to plot total cases
@app.callback(output=Output("plot-total", "figure"), inputs=[Input("country", "value")])
def plot_total_cases(country):
    data.process_data(country)
    model = Model(data.dtf)
    model.forecast()
    model.add_deaths(data.mortality)
    result = Result(model.dtf)
    return result.plot_total(model.today)


# Python function to plot active cases
@app.callback(output=Output("plot-active", "figure"), inputs=[Input("country", "value")])
def plot_active_cases(country):
    data.process_data(country)
    model = Model(data.dtf)
    model.forecast()
    model.add_deaths(data.mortality)
    result = Result(model.dtf)
    return result.plot_active(model.today)


# Creating custom style for  local use
divBorderStyle = {
    'backgroundColor': '#393939',
    'borderRadius': '12px',
    'lineHeight': 0.9,
    "padding": "18px 24px",

}

font_xl = {
    "fontSize": "24px", "marginBottom": "16px", "color": "white"
}


font_lg = {
    "fontSize": "21px", "marginBottom": "24px", "color": "white"
}

font_md = {
    "fontSize": "18px", "marginBottom": "8px", "color": "white"
}


# Python function to render output panel
@app.callback(output=Output("output-panel", "children"),
              inputs=[Input("country", "value")])
def render_output_panel(country):
    data.process_data(country)
    model = Model(data.dtf)
    model.forecast()
    model.add_deaths(data.mortality)
    result = Result(model.dtf)
    peak_day, num_max, total_cases_until_today, total_cases_in_30days, active_cases_today, active_cases_in_30days = result.get_panel()
    peak_color = "white" if model.today > peak_day else "red"

    panel = html.Div([

        html.Div([

            html.Div(f"Data for {country}",
                     className="text-white font-lg width-full",
                     style={"margin": "16px auto"}),

            html.Br(), html.Br(),
            html.Div("Total cases until today:",
                     className="text-white font-md width-full bottom8",
                     ),
            html.Div("{:,.0f}".format(total_cases_until_today),
                     className="text-white font-lg width-full bottom16"),

            html.Div("Total cases in 30 days:",
                     className="text-danger font-md width-full bottom8", ),
            html.Div("{:,.0f}".format(total_cases_in_30days),
                     className="text-danger font-lg width-full bottom16"),

            html.Div("Active cases today:",
                     className="text-white font-md width-full bottom8",),
            html.Div("{:,.0f}".format(active_cases_today),
                     className="text-white font-lg width-full bottom16"),

            html.Div("Active cases in 30 days:",
                     className="text-danger font-md width-full bottom8"),
            html.Div("{:,.0f}".format(active_cases_in_30days),
                     className="text-danger font-lg width-full bottom16"),

            html.Div("Peak day:", style={"color": peak_color,
                                         "fontSize": "18px", "marginBottom": "8px",
                                         }),
            html.Div(peak_day.strftime("%Y-%m-%d"),
                     style=font_lg),
            html.Div("with {:,.0f} cases".format(
                num_max), style={"color": peak_color,
                                 "fontSize": "21px", "marginBottom": "8px",
                                 })
        ], style=divBorderStyle)
    ])

    return panel


if __name__ == '__main__':
    app.run_server(debug=True)
