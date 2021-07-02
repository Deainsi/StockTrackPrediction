import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colors = {
    'background': '#111111',
    'text': '#FFFFFF'
}
with open('companies.csv', 'r') as companies:
    cdf = pd.DataFrame(columns=['company', 'symbol'])
    next(companies)
    for line in companies:
        data = line.split(',')
        data_row = [data[0], data[1]]
        cdf.loc[-1, :] = data_row
        cdf.index += 1
    cdf = cdf.drop(0)

layout = html.Div([
    dcc.Store(id='session', storage_type='session'),
    html.Div([
        html.Div([
            dcc.Dropdown(id="Stock", options=[
                {'label': f'{c} / {s}', 'value': c} for c, s in
                cdf[['company', 'symbol']].values
            ],
                         value="GME")
        ],
            style={
                'width': '50%',
                'color': '#000'
            }),
        html.Div([
            html.Button(children=['1M'], id="1M"),
            html.Button(children=['6M'], id="6M"),
            html.Button(children=['YMX'], id="YMX"),
            html.Button(children=['1Y'], id="1Y"),
            html.Button(children=['5Y'], id="5Y"),
            html.Button(children=['MAX'], id="MAX"),
        ],
            style={'border': 'none', 'color': '#fff'}),
        html.Div([
            html.Span("Prediction for tomorrow: "),
            html.Span(id="prediction")
        ])
    ],
        style={
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'space-evenly'
        }
    ),
    html.Div(dcc.Graph(id="Graph", figure={'layout': {
        "plot_bgcolor": "#211d1d",
        "yaxis": {'zeroline': False,
                  'showgrid': False},
        "xaxis": {'zeroline': False,
                  'showgrid': False},
        "paper_bgcolor": "#2a2828",
        "font_color": colors['text']
    }}),
             )

],
    style={'backgroundColor': colors['background']},
)
