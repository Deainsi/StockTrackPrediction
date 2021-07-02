import datetime as dt
import json
from urllib import request

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model

from app import app


@app.callback(Output('session', 'data'), Output('prediction','children'), Output('prediction','style'),Input('Stock', 'value'), State('session', 'data'))
def update_session(ticker, dat):
    api_key = "8NZ8WBHSJNM1SA6I"

    url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
    color = {}
    if dat:
        if ticker in dat:

            if pd.read_json(dat[f'{ticker}'], orient='split').Close.iloc[-30] > pd.read_json(dat[f'{ticker}'], orient='split').Close.iloc[-31]:
                color['color']="#046613"
            else:
                color['color'] = "#bd1604"

            return dat,  f"{pd.read_json(dat[f'{ticker}'], orient='split').Close.iloc[-30]:.2f}", color

    with request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        data = data['Time Series (Daily)']

        df = pd.DataFrame(columns=['Date', 'Close'])
        for k, v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(), float(v['4. close'])]
            df.loc[-1, :] = data_row
            df.index = df.index + 1

        scaler = MinMaxScaler()
        lstm_model = load_model('mm.h5')
        df = df.sort_index()

        rd = df.Close.iloc[-40:].values
        rd = rd.reshape(-1, 1)
        rd = scaler.fit_transform(rd)
        rd = np.reshape(rd, (1, rd.shape[0], 1))
        for i in range(30):
            prediction = lstm_model.predict(rd)
            rd[0] = np.append(np.delete(rd[0], 0, 0), prediction).reshape(rd[0].shape)

        rd = rd.reshape((40, 1))
        rd = scaler.inverse_transform(rd)
        rd = rd.reshape(-1)
        dates = pd.date_range(df.Date.iloc[-1] + dt.timedelta(days=1), periods=30)
        cdf = pd.DataFrame({'Date': dates,
                            'Close': rd[-30:]})

        df = pd.concat([df, cdf], ignore_index=True, join='outer')

        if df.Close.iloc[-30] > df.Close.iloc[-31]:
            color['color'] = "#046613"
        else:
            color['color'] = "#bd1604"

        dat = dat or {}
        dat[f'{ticker}'] = df.to_json(date_format='iso', orient='split')
        return dat, f'{cdf.Close.iloc[-30]:.2f}', color


@app.callback(Output('Graph', 'figure'),
              Input('1M', 'n_clicks'),
              Input('6M', 'n_clicks'),
              Input('YMX', 'n_clicks'),
              Input('1Y', 'n_clicks'),
              Input('5Y', 'n_clicks'),
              Input('MAX', 'n_clicks'),
              Input('session', 'data'),
              State('Stock', 'value'))
def update_time_figure(b1, b2, b3, b4, b5, b6, data, ticker):
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    dr = {
        '1M': relativedelta(months=1),
        '6M': relativedelta(months=6),
        '1Y': relativedelta(years=1),
        '5Y': relativedelta(years=5),
    }
    cdf = pd.read_json(data[f'{ticker}'], orient='split')
    cdf['Date'] = cdf['Date'].dt.date
    if input_id == 'MAX' or input_id == 'session':
        rdf = cdf
    elif input_id == '1M' or input_id == '6M' or input_id == '1Y' or input_id == '5Y':
        time_range = dr[input_id]
        rdf = cdf.loc[cdf['Date'] >= dt.date.today() - time_range]
    elif input_id == 'YMX':
        rdf = cdf.loc[cdf['Date'] >= dt.date(dt.date.today().year, 1, 1)]
    else:
        raise PreventUpdate

    if rdf.Close.iloc[-1] > rdf.Close.iloc[0]:
        color = "#046613"
    else:
        color = "#bd1604"

    fig = px.line(rdf, x='Date', y='Close')

    fig.update_layout(plot_bgcolor="#211d1d",
                      yaxis={'zeroline': False},
                      xaxis={'zeroline': False},
                      paper_bgcolor="#2a2828",
                      font_color="#fff",
                      hovermode='x'
                      )
    fig.update_traces(line=dict(color=color))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig
