
import pandas as pd
import plotly.graph_objects as go

colors = {
    'background': '#2D2D2D',
    'text': '#E1E2E5',
    'figure_text': '#ffffff',
    'confirmed_text': 'rgb(117, 190, 255)',
    'deaths_text': 'rgb(225, 116, 108)',
    'recovered_text': 'rgb(123, 194, 145)',
    'highest_case_bg': '#393939',

}


class Result():

    def __init__(self, dtf):
        self.dtf = dtf

    @staticmethod
    def calculate_peak(dtf):
        data_max = dtf["delta_data"].max()
        forecast_max = dtf["delta_forecast"].max()
        if data_max >= forecast_max:
            peak_day = dtf[dtf["delta_data"] == data_max].index[0]
            return peak_day, data_max
        else:
            peak_day = dtf[dtf["delta_forecast"] == forecast_max].index[0]
            return peak_day, forecast_max

    @staticmethod
    def calculate_max(dtf):
        total_cases_until_today = dtf["data"].max()
        total_cases_in_30days = dtf["forecast"].max()
        active_cases_today = dtf["delta_data"].max()
        active_cases_in_30days = dtf["delta_forecast"].max()
        return total_cases_until_today, total_cases_in_30days, active_cases_today, active_cases_in_30days

    def plot_total(self, today):
        # main plots
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.dtf.index, y=self.dtf["data"], mode='markers', name='data', line={"color": "white"}))
        fig.add_trace(go.Scatter(
            x=self.dtf.index, y=self.dtf["forecast"], mode='none', name='forecast', fill='tozeroy', fillcolor='rgba(224, 131, 145, .3)'))
        fig.add_trace(go.Bar(x=self.dtf.index,
                             y=self.dtf["deaths"], name='deaths', marker_color='red'))

        # add slider
        # fig.update_xaxes(rangeslider_visible=True)
        # set background color
        fig.update_layout(
            plot_bgcolor=colors["background"], paper_bgcolor=colors['background'], autosize=False, height=350)
        # add vline
        fig.add_shape({"x0": today, "x1": today, "y0": 0, "y1": self.dtf["forecast"].max(),
                       "type": "line", "line": {"width": 2, "dash": "dot", "color": "#919191"}})
        fig.add_trace(go.Scatter(x=[today], y=[self.dtf["forecast"].max()], text=[
                      "today"], mode="text", line={"color": "white"}, showlegend=False))

        fig.update_layout(
            margin=dict(l=0,
                        r=0,
                        t=65,
                        b=0
                        ),
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
            ))

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')

        return fig

    def plot_active(self, today):
        # main plots
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.dtf.index, y=self.dtf["delta_forecast"], mode='none', name='forecast',
            fill='tozeroy', fillcolor='rgba(224, 131, 145, .3)'))
        fig.add_trace(go.Bar(
            x=self.dtf.index, y=self.dtf["delta_data"], name='data', marker_color='white'))
        # add slider
        # fig.update_xaxes(rangeslider_visible=True)
        # set background color
        fig.update_layout(
            plot_bgcolor=colors["background"], paper_bgcolor=colors['background'], autosize=False)
        # add vline
        fig.add_shape({"x0": today, "x1": today, "y0": 0, "y1": self.dtf["delta_forecast"].max(),
                       "type": "line", "line": {"width": 2, "dash": "dot"}})
        fig.add_trace(go.Scatter(x=[today], y=[self.dtf["delta_forecast"].max()], text=[
                      "today"], mode="text", line={"color": "white"}, showlegend=False))

        fig.update_layout(
            margin=dict(l=0,
                        r=0,
                        t=65,
                        b=0
                        ),
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
            ), height=350)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#3A3A3A')

        return fig

    def get_panel(self):
        peak_day, num_max = self.calculate_peak(self.dtf)
        total_cases_until_today, total_cases_in_30days, active_cases_today, active_cases_in_30days = self.calculate_max(
            self.dtf)
        return peak_day, num_max, total_cases_until_today, total_cases_in_30days, active_cases_today, active_cases_in_30days
