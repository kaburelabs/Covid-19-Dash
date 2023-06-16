import pandas as pd


class Data:
    def get_data(self):
        self.dtf_cases = pd.read_csv(
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
            sep=",",
        )
        self.dtf_deaths = pd.read_csv(
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
            sep=",",
        )
        # self.geo = self.dtf_cases[['Country/Region','Lat','Long']].drop_duplicates("Country/Region", keep='first')
        self.countrylist = ["World"] + self.dtf_cases[
            "Country/Region"
        ].unique().tolist()

    @staticmethod
    def group_by_country(dtf, country):
        dtf = (
            dtf.drop(["Province/State", "Lat", "Long"], axis=1)
            .groupby("Country/Region")
            .sum()
            .T
        )
        dtf["World"] = dtf.sum(axis=1)
        dtf = dtf[country]
        # print(dtf.index)
        dtf.index = pd.to_datetime(dtf.index, format="%m/%d/%y")
        ts = pd.DataFrame(index=dtf.index, data=dtf.values, columns=["data"])
        return ts

    @staticmethod
    def calculate_mortality(ts_deaths, ts_cases):
        last_deaths = ts_deaths["data"].iloc[-1]
        last_cases = ts_cases["data"].iloc[-1]
        mortality = last_deaths / last_cases
        return mortality

    def process_data(self, country):
        self.dtf = self.group_by_country(self.dtf_cases, country)
        deaths = self.group_by_country(self.dtf_deaths, country)
        self.dtf["deaths"] = deaths
        self.mortality = self.calculate_mortality(deaths, self.dtf)
