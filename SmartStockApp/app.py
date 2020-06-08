import datetime  # Date and time

import dash
import dash_bootstrap_components as dbc  # Themes
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go  # Graphs structure
import ta  # Technical indicators
# Fix yahoo finance API
import talib as tal
from dash.dependencies import Input, Output, State  # Decorator components
from pandas_datareader import data as web  # Get data from web

# from tensorflow_core.python.data.util import structure

print(dcc.__version__)  # 0.6.0 or above is required

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Ronan Hayes

app = dash.Dash(__name__)

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

##################################################
#################################################
#################################################
home_page = html.Div(
    className='homepage',
    children=[
        html.H1('SmartStock'),
        html.Br(),
        dcc.Link('Technical Analysis', href='/page-1',
                 className='linkbtnhome'
                 ),
        dcc.Link('Machine Learning', href='/page-2',
                 className='linkbtnhome'
                 ),

        html.Br(),
        html.P('Please do not use this application as investment advice.'),
        html.Br(),
        html.P('Created by Ronan Hayes', style={'color': 'rgba(223, 176, 255, 0.75)'}),
        # dbdbdb is the light grey color

    ])


#################################################################################################
#################################################################################################
#################################################################################################

# Page 1
##################################################
# Technical indicators
# Create list of what TA's to add to charts
# MACD


def macd(df, nfast=12, nslow=26):
    mCd = {
        'macd': ta.trend.macd(df, nfast, nslow),
        'signal': ta.trend.macd_signal(df, nfast, nslow),
        'diff': ta.trend.macd_diff(df, nfast, nslow)
    }
    return mCd


# Awesome Oscillator
def ao(high, low, s=5, len=34):
    return ta.momentum.ao(high, low, s, len, fillna=True)


# Aroon Up & Down
def aroon(close, n=25):
    aRoon = {

        'up': ta.trend.aroon_up(close, n, fillna=True),
        'down': ta.trend.aroon_down(close, n, fillna=True)
    }
    return aRoon


# Williams %R
def wr(high, low, close, lbp=14):
    return ta.momentum.wr(high, low, close, lbp, fillna=False)


page_1_layout = html.Div(className='row',
                         children=[
                             html.Div(
                                 className='four columns div-user-controls bg-black',
                                 children=[

                                     html.H1('SmartStock'),

                                     html.Div(id='page-1-content'),
                                     dcc.Link('Home', href='/',
                                              className='linkbtns'),
                                     dcc.Link('Machine Learning', href='/page-2',
                                              className='linkbtns'),

                                     html.H2('Technical Analysis'),
                                     html.P('This page allows you to apply and analyse various different technical '
                                            'indicators by plotting graphs and overlays. You can search for any '
                                            'stock, currency pair or cryptocurrency between your required dates. Below are some example symbols: ',
                                            className='inputnames'),
                                     html.P('Equity: NFLX | TSLA | FB', className='inputnames'),
                                     html.P('Currency: GBPEUR=X | EURUSD=X | AUDUSD=X', className='inputnames'),
                                     html.P('CryptoCurrency: BTC-USD | ETH-USD | XRP-USD', className='inputnames'),
                                     html.P('Indexs: ^DJI | ^IXIC | ^GSPC', className='inputnames'),

                                     html.P('Please enter your chosen desired stock below:'),
                                     # Searches for the requested ticker
                                     html.Div(children=[
                                         dcc.Input(id='ticker-typer',
                                                   type='text',
                                                   className='tickertyper',
                                                   value='AAPL'),

                                         dbc.Button('Search',
                                                    id='button',
                                                    className='searchbar')]),

                                     html.P('Input your start and finish dates:'),

                                     html.Div(
                                         dcc.DatePickerRange(id='date',
                                                             start_date=datetime.date(2019, 5, 23),
                                                             end_date=datetime.date.today(),
                                                             number_of_months_shown=2)),

                                     html.P('Select one or more technical indicator:'),
                                     # Gets the indicators wanted
                                     html.Div(
                                         dcc.Dropdown(id='indicators',
                                                      options=[
                                                          {'label': 'Awesome Oscillator', 'value': '0'},
                                                          {'label': 'Bollinger Bands', 'value': '1'},
                                                          {'label': 'Exponential Moving Average', 'value': '2'},
                                                          {'label': 'MACD', 'value': '3'},
                                                          {'label': 'Volume', 'value': '4'},
                                                          {'label': 'Relative Strenth Index', 'value': '5'},
                                                          {'label': 'Stochastic Oscillator', 'value': '6'},
                                                          {'label': 'Aroon Oscillator', 'value': '7'},
                                                          {'label': 'Williams %R', 'value': '8'}],
                                                      value=[],
                                                      multi=True,
                                                      className='dropdowntech',

                                                      )
                                     )]

                             ),

                             # Creates the title
                             html.Div(className='eight columns div-for-charts bg-grey',
                                      children=[
                                          html.H2(id='title'),
                                          html.Div('A candlestick chart is a style of financial chart used to '
                                                   'describe price movements of your chosen stock. '
                                                   'Each candle displays the Open, High, Low Close prices for each '
                                                   'day. A candle is made up of a body (rectangle) and shadows ('
                                                   'straight line). The upper shadow (top of the line) represents the '
                                                   'Highest price, while the lower shadow represents the Lowest '
                                                   'price. The top of the rectangle repressnts the Close price of the '
                                                   'day, while the bottom represents the Open. '
                                                   ''
                                                   'A green candlestick '
                                                   'means that the opening price on that day was lower than the '
                                                   'closing price that day (i.e. the price moved up during the day); '
                                                   'a red candlestick means that the opening price was higher than '
                                                   'the closing price that day (i.e. the price moved down during the '
                                                   'day). '),

                                          # Creates the stock graph
                                          dcc.Graph(id='stock-graph', relayoutData={'autosize': False}),

                                          # Creates the Awesome Oscillator graph
                                          html.Div(id='ao-toggle',
                                                   className='graphbg',
                                                   children=[
                                                       html.Div('The Awesome Oscillator is an indicator used to '
                                                                'measure market momentum. AO calculates the '
                                                                'difference of a 34 Period and 5 Period Simple Moving '
                                                                'Averages. The Simple Moving Averages that are used '
                                                                'are not calculated using closing price but rather '
                                                                'each bars midpoints. AO is generally used to affirm '
                                                                'trends or to anticipate possible reversals.'),
                                                       dcc.Graph(id='ao-graph')]),

                                          # Creates the MACD graph
                                          html.Div(id='macd-toggle',
                                                   className='graphbg',
                                                   children=[html.Div('Moving Average Convergence Divergence (MACD) '
                                                                      'is a trend-following momentum indicator that '
                                                                      'shows the relationship between two moving '
                                                                      'averages of a securitys price. The MACD is '
                                                                      'calculated by subtracting the 26-period '
                                                                      'Exponential Moving Average (EMA) from the '
                                                                      '12-period EMA.'),

                                                             dcc.Graph(id='macd-graph')]),

                                          # Creates the volume index
                                          html.Div(id='volume-toggle',
                                                   className='graphbg',
                                                   children=[
                                                       html.Div('Volume of trade measures the total number of shares '
                                                                'or contracts transacted for a specified security '
                                                                'during a specified time period. It includes the '
                                                                'total number of shares transacted between a buyer '
                                                                'and seller during a transaction. When securities are '
                                                                'more actively traded, their trade volume is high, '
                                                                'and when securities are less actively traded, '
                                                                'their trade volume is low.'),
                                                       dcc.Graph(id='volume-graph')]),

                                          # Creates the RSI graph
                                          html.Div(id='rsi-toggle',
                                                   className='graphbg',
                                                   children=[
                                                       html.Div('The Relative Strength Index (RSI) is a well versed '
                                                                'momentum based oscillator which is used to measure '
                                                                'the speed (velocity) as well as the change ('
                                                                'magnitude) of directional price movements. '
                                                                'Essentially RSI, when graphed, provides a visual '
                                                                'mean to monitor both the current, as well as '
                                                                'historical, strength and weakness of a particular '
                                                                'market. The strength or weakness is based on closing '
                                                                'prices over the duration of a specified trading '
                                                                'period creating a reliable metric of price and '
                                                                'momentum changes. '),
                                                       dcc.Graph(id='rsi-graph')]),

                                          # Creates the Stochastic Oscillator graph
                                          html.Div(id='stoch-toggle',
                                                   className='graphbg',
                                                   children=[html.Div('The Stochastic Oscillator (STOCH) is a range '
                                                                      'bound momentum oscillator. The Stochastic '
                                                                      'indicator is designed to display the location '
                                                                      'of the close compared to the high/low range '
                                                                      'over a user defined number of periods. '
                                                                      'Typically, the Stochastic Oscillator is used '
                                                                      'for three things; Identifying overbought and '
                                                                      'oversold levels, spotting divergences and also '
                                                                      'identifying bull and bear set ups or signals.'),
                                                             dcc.Graph(id='stoch-graph')]),

                                          # Creates the Aroon graph
                                          html.Div(id='aroon-toggle',
                                                   className='graphbg',
                                                   children=[html.Div('The Aroon Indicator (often referred to as '
                                                                      'Aroon Up Down) is a range bound, '
                                                                      'technical indicator that is actually a set of '
                                                                      'two separate measurements designed to measure '
                                                                      'how many periods have passed since price has '
                                                                      'recorded an n-period high or low low with “n” '
                                                                      'being a number of periods set at the trader’s '
                                                                      'discretion. For example a 14 Day Aroon-Up will '
                                                                      'take the number of days since price last '
                                                                      'recorded a 14 day high and then calculate a '
                                                                      'number between 0 and 100. A 14 Day Aroon-Down '
                                                                      'will do the same thing except is will '
                                                                      'calculate a number based of the number of days '
                                                                      'since a 14 day low. This number is intended to '
                                                                      'quantify the strength of a trend (if there is '
                                                                      'one). The closer the number is to 100, '
                                                                      'the stronger the trend.'),
                                                             dcc.Graph(id='aroon-graph')]),

                                          # Creates the WR graph
                                          html.Div(id='wr-toggle',
                                                   className='graphbg',
                                                   children=[html.Div('Williams %R (%R) is a momentum-based '
                                                                      'oscillator used in technical analysis, '
                                                                      'primarily to identify overbought and oversold '
                                                                      'conditions. The %R is based on a comparison '
                                                                      'between the current close and the highest high '
                                                                      'for a user defined look back period. %R '
                                                                      'Oscillates between 0 and -100 (note the '
                                                                      'negative values) with readings closer to zero '
                                                                      'indicating more overbought conditions and '
                                                                      'readings closer to -100 indicating oversold. '
                                                                      'Typically %R can generate set ups based on '
                                                                      'overbought and oversold conditions as well '
                                                                      'overall changes in momentum'),
                                                             dcc.Graph(id='wr-graph')])
                                      ]),

                         ])


# The function is responsable for changing the elements in the graph depending on the dropdown input
@app.callback(
    [Output('title', 'children'),

     Output('stock-graph', 'figure'),
     Output('ao-graph', 'figure'),
     Output('macd-graph', 'figure'),
     Output('volume-graph', 'figure'),
     Output('rsi-graph', 'figure'),
     Output('stoch-graph', 'figure'),
     Output('aroon-graph', 'figure'),
     Output('wr-graph', 'figure'),
     Output('ao-toggle', 'style'),
     Output('macd-toggle', 'style'),
     Output('volume-toggle', 'style'),
     Output('rsi-toggle', 'style'),
     Output('stoch-toggle', 'style'),
     Output('aroon-toggle', 'style'),
     Output('wr-toggle', 'style')],
    [Input('button', 'n_clicks'),
     Input('indicators', 'value'),
     Input('date', 'start_date'),
     Input('date', 'end_date'),
     Input('stock-graph', 'relayoutData')],
    [State('ticker-typer', 'value')]
)
def update_figure(button, indicators, start_date, end_date, relayoutData, company):
    # Variables initializer

    awesomegraph = {}
    macdgraph = {}
    volumegraph = {}
    rsigraph = {}
    stochasticgraph = {}
    aroongraph = {}
    wrgraph = {}

    # Get the requested ticker
    stock_df = web.DataReader(company.upper(), 'yahoo', start_date, end_date)

    # Define figure limits

    # Stock graph
    fig = {
        'data': [go.Candlestick(x=stock_df.index,
                                open=stock_df.Open,
                                close=stock_df.Close,
                                high=stock_df.High,
                                low=stock_df.Low,
                                name='Open, High, Low, Close',
                                showlegend=False)],
        'layout': go.Layout(
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            yaxis_gridcolor='rgba(223, 176, 255, 0.25)',
            xaxis_gridcolor='rgba(223, 176, 255, 0.25)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin={'b': 15},
            hovermode='x',
            autosize=False,
            title={'text': '{} Candlestick Chart'.format(company.upper()), 'font': {'color': 'white'}, 'x': 0.5},
            xaxis={'rangeslider': {'visible': False},
                   'range': [stock_df.index.min(), stock_df.index.max()]},
        )
    }
    # Why is automatically resizes???

    # Add volume permenantly
    # Problem = need it to be subplot

    # fig['data'].append(dict(x=stock_df.index, y=stock_df.Volume,
    #                        marker=dict(color="species"),
    #                        type='bar', yaxis='y', name='Volume'))

    # Indicator graphs

    # Awesome Oscillator
    if '0' in indicators:
        stock_df['Awesome'] = ta.momentum.ao(stock_df.High, stock_df.Low, fillna=True)
        awesome_toggle = {'display': 'block'}
        awesomegraph = {'data': [go.Bar(x=stock_df.index,
                                        y=stock_df['Awesome'],
                                        name='Awesome Oscillator',
                                        marker_color='#61efff',
                                        showlegend=False)],
                        'layout': go.Layout(
                            template='plotly_dark',
                            xaxis_title='Date',
                            yaxis_title='Momentum',
                            yaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                            xaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            plot_bgcolor='rgba(0, 0, 0, 0)',
                            margin={'b': 15},
                            hovermode='x',
                            autosize=False,
                            title={'text': 'Awesome Oscillator', 'font': {'color': 'white'},
                                   'x': 0.5},
                            xaxis={'rangeslider': {'visible': False},
                                   'range': [stock_df.index.min(), stock_df.index.max()]},
                        )

                        }
    else:
        awesome_toggle = {'display': 'none'}

    # Bollinger Bands
    if '1' in indicators:
        fig['data'].append(go.Scatter(x=stock_df.index,
                                      y=ta.volatility.bollinger_hband(stock_df.Close),
                                      name='Bollinger High Band'))

        fig['data'].append(go.Scatter(x=stock_df.index,
                                      y=ta.volatility.bollinger_mavg(stock_df.Close),
                                      name='Bollinger Middle Band'))

        fig['data'].append(go.Scatter(x=stock_df.index,
                                      y=ta.volatility.bollinger_lband(stock_df.Close),
                                      name='Bollinger Low Band'))

    # Exponential moving average
    if '2' in indicators:
        fig['data'].append(go.Scatter(x=stock_df.index,
                                      y=tal.EMA(stock_df.Close, timeperiod=30),
                                      name='Exponential Moving Average of {} days'.format('Nine')))

    # MACD analysis
    if '3' in indicators:
        macd_toggle = {'display': 'block'}
        macdgraph = {'data': [go.Scatter(x=stock_df.index,
                                         y=macd(stock_df.Close)['macd'],
                                         name='MACD',
                                         showlegend=False),
                              go.Scatter(x=stock_df.index,
                                         y=macd(stock_df.Close)['signal'],
                                         name='Signal',
                                         showlegend=False),
                              go.Bar(x=stock_df.index,
                                     y=macd(stock_df.Close)['diff'],
                                     marker_color='#61ff79',
                                     name='Difference',
                                     showlegend=False)],
                     'layout': go.Layout(
                         template='plotly_dark',
                         xaxis_title='Date',
                         yaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                         xaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                         paper_bgcolor='rgba(0, 0, 0, 0)',
                         plot_bgcolor='rgba(0, 0, 0, 0)',
                         margin={'b': 15},
                         hovermode='x',
                         autosize=False,
                         title={'text': 'Moving Average Convergence Divergence', 'font': {'color': 'white'},
                                'x': 0.5},
                         xaxis={'range': [stock_df.index[26], stock_df.index.max()]},  # due to MA
                     )
                     }
    else:
        macd_toggle = {'display': 'none'}

    # Volume
    if '4' in indicators:
        volume_toggle = {'display': 'block'}
        volumegraph = {'data': [go.Bar(x=stock_df.index,
                                       y=stock_df.Volume,
                                       marker_color='#f9ff52',
                                       name='Volume',
                                       showlegend=False)],
                       'layout': go.Layout(
                           template='plotly_dark',
                           xaxis_title='Date',
                           yaxis_title='Volume',
                           yaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                           xaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                           paper_bgcolor='rgba(0, 0, 0, 0)',
                           plot_bgcolor='rgba(0, 0, 0, 0)',
                           margin={'b': 15},
                           hovermode='x',
                           autosize=False,
                           title={'text': 'Volume', 'font': {'color': 'white'},
                                  'x': 0.5},
                           xaxis={'range': [stock_df.index.min(), stock_df.index.max()]},
                       )
                       }
    else:
        volume_toggle = {'display': 'none'}

    # RSI analysis
    if '5' in indicators:
        rsi_toggle = {'display': 'block'}
        rsigraph = {'data': [go.Scatter(x=stock_df.index,
                                        y=ta.momentum.rsi(stock_df.Close),
                                        line=dict(color='#d278ff'),
                                        name='Relative Strength Index',
                                        showlegend=False),

                             # Over bought is > 70
                             go.Scatter(x=stock_df.index,
                                        y=[70] * stock_df.index.size,
                                        line=dict(dash='dash', color='#b8ccfc'),
                                        name='Overbought',
                                        showlegend=False),
                             # Under bought is < 30
                             go.Scatter(x=stock_df.index,
                                        y=[30] * stock_df.index.size,
                                        fill='tonexty',
                                        line=dict(dash='dash', color='#b8ccfc'),
                                        name='Oversold',
                                        showlegend=False)],
                    'layout': go.Layout(
                        template='plotly_dark',
                        xaxis_title='Date',
                        yaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                        xaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        margin={'b': 15},
                        hovermode='x',
                        autosize=False,
                        title={'text': 'Relative Strength Index', 'font': {'color': 'white'},
                               'x': 0.5},
                        xaxis={'range': [stock_df.index.min(), stock_df.index.max()]},
                    )
                    }
    else:
        rsi_toggle = {'display': 'none'}

    # Stochastic Oscillator
    if '6' in indicators:
        stochastic_toggle = {'display': 'block'}
        stochasticgraph = {'data': [go.Scatter(x=stock_df.index,
                                               y=ta.momentum.stoch(stock_df.High, stock_df.Low, stock_df.Close,
                                                                   fillna=True),
                                               name='Stochastic',
                                               showlegend=False),

                                    go.Scatter(x=stock_df.index,
                                               y=ta.momentum.stoch_signal(stock_df.High, stock_df.Low, stock_df.Close),
                                               name='Signal',
                                               showlegend=False),

                                    go.Scatter(x=stock_df.index,
                                               y=[20] * stock_df.index.size,
                                               line=dict(dash='dash', color='#f8d4ff'),
                                               name='Oversold',
                                               showlegend=False),

                                    go.Scatter(x=stock_df.index,
                                               y=[80] * stock_df.index.size,
                                               line=dict(dash='dash', color='#f8d4ff'),
                                               fill='tonexty',
                                               name='Overbought',
                                               showlegend=False)],

                           'layout': go.Layout(
                               template='plotly_dark',
                               xaxis_title='Date',
                               yaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                               xaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                               paper_bgcolor='rgba(0, 0, 0, 0)',
                               plot_bgcolor='rgba(0, 0, 0, 0)',
                               margin={'b': 15},
                               hovermode='x',
                               autosize=False,
                               title={'text': 'Stochastic Oscillator', 'font': {'color': 'white'},
                                      'x': 0.5},
                               xaxis={'range': [stock_df.index.min(), stock_df.index.max()]},
                           )
                           }
    else:
        stochastic_toggle = {'display': 'none'}

    # Aroon
    if '7' in indicators:
        aroon_toggle = {'display': 'block'}
        aroongraph = {'data': [go.Scatter(x=stock_df.index,
                                          y=aroon(stock_df.Close)['up'],
                                          name='Aroon UP',
                                          showlegend=False),
                               go.Scatter(x=stock_df.index,
                                          y=aroon(stock_df.Close)['down'],
                                          name='Aroon DOWN',
                                          showlegend=False)],

                      'layout': go.Layout(
                          template='plotly_dark',
                          xaxis_title='Date',
                          yaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                          xaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                          paper_bgcolor='rgba(0, 0, 0, 0)',
                          plot_bgcolor='rgba(0, 0, 0, 0)',
                          margin={'b': 15},
                          hovermode='x',
                          autosize=False,
                          title={'text': 'Aroon Oscillator', 'font': {'color': 'white'},
                                 'x': 0.5},
                          xaxis={'range': [stock_df.index.min(), stock_df.index.max()]},
                      )
                      }
    else:
        aroon_toggle = {'display': 'none'}

    if '8' in indicators:
        # stock_df['WR'] = ta.momentum.wr(stock_df.High, stock_df.Low, fillna=False)
        wr_toggle = {'display': 'block'}
        wrgraph = {'data': [go.Scatter(x=stock_df.index,
                                       y=ta.momentum.wr(stock_df.High, stock_df.Low, stock_df.Close),
                                       name='Williams %R',
                                       line=dict(color='pink'),
                                       showlegend=False),

                            # >80
                            go.Scatter(x=stock_df.index,
                                       y=[-80] * stock_df.index.size,
                                       line=dict(dash='dash', color='#ff707e'),
                                       name='Oversold',
                                       showlegend=False),
                            # < 20
                            go.Scatter(x=stock_df.index,
                                       y=[-20] * stock_df.index.size,
                                       fill='tonexty',
                                       line=dict(dash='dash', color='#ff707e'),
                                       name='Overbought',
                                       showlegend=False)],
                   'layout': go.Layout(
                       template='plotly_dark',
                       xaxis_title='Date',
                       yaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                       xaxis_gridcolor='rgba(223, 176, 255, 0.25)',
                       paper_bgcolor='rgba(0, 0, 0, 0)',
                       plot_bgcolor='rgba(0, 0, 0, 0)',
                       margin={'b': 15},
                       hovermode='x',
                       autosize=False,
                       title={'text': 'Williams %R', 'font': {'color': 'white'},
                              'x': 0.5},
                       xaxis={'range': [stock_df.index.min(), stock_df.index.max()]},
                   )
                   }
    else:
        wr_toggle = {'display': 'none'}

    return ('{} stock prices over the last year'.format(company.upper()), fig,
            awesomegraph, macdgraph, volumegraph, rsigraph, stochasticgraph, aroongraph, wrgraph,
            awesome_toggle, macd_toggle, volume_toggle, rsi_toggle, stochastic_toggle, aroon_toggle, wr_toggle)


#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

import numpy as np
import pandas_datareader as web
from datetime import date
import pandas as pd
import talib

import xgboost as xgb

start = date(1970, 1, 1)
# app = dash.Dash()
end = date.today()
prediction = 0
data = []

page_2_layout = html.Div(className='row',
                         children=[
                             html.Div(
                                 className='four columns div-user-controls',
                                 children=[

                                     html.H1('SmartStock'),

                                     html.Div(id='page-2-content'),
                                     dcc.Link('Home', href='/',
                                              className='linkbtns'),
                                     dcc.Link('Technical Analysis', href='/page-1',
                                              className='linkbtns'),

                                     html.H2('Machine Learning'),
                                     html.P(
                                         'Using various machine learning models you can use historical price data to '
                                         'predict tomorrows closing price. Example symbols to predict:',
                                         className='inputnames'),
                                     html.P('Equity: NFLX | TSLA | FB', className='inputnames'),
                                     html.P('Currency: GBPEUR=X | EURUSD=X | AUDUSD=X', className='inputnames'),
                                     html.P('CryptoCurrency: BTC-USD | ETH-USD | XRP-USD', className='inputnames'),
                                     html.P('Indexs: ^DJI | ^IXIC | ^GSPC', className='inputnames'),

                                     html.P(
                                         'Input Stock:'),
                                     # Searches for the requested ticker
                                     html.Div(

                                         dcc.Input(id='ml_search',
                                                   type='text',
                                                   # style={'backgroundColor': 'white', 'color': 'white'},
                                                   value='AAPL',
                                                   className='move')),

                                     html.Br(),
                                     html.P(
                                         'Select Model:'),

                                     # Gets the indicators wanted
                                     html.Div(
                                         dcc.Dropdown(
                                             id='dropdown2',
                                             options=[
                                                 {'label': 'XGBOOST', 'value': 'XGBOOST'},
                                                 {'label': 'SVR', 'value': 'SVR'},
                                                 {'label': 'LIGHTGBM', 'value': 'LIGHTGBM'},
                                                 {'label': 'DECISION TREE', 'value': 'DECISIONTREE'},
                                                 {'label': 'RANDOM FOREST', 'value': 'RANDOMFOREST'},
                                                 {'label': 'LINEAR REGRESSION', 'value': 'LINEARREGRESSION'},

                                             ],
                                             value='XGBOOST'
                                         ),

                                     ),
                                     html.Br(),
                                     html.P('Please allow for training time...')]

                             ),

                             html.Div(className='eight columns div-for-charts bg-grey',
                                      children=[

                                          dcc.Graph(id="stock-Chart",
                                                    figure=go.Figure(
                                                        layout=dict(paper_bgcolor='#31302F',
                                                                    plot_bgcolor='#31302F'))),

                                          html.P('Do not use tomorrows prediction as investment advice.',
                                                 style={'margin-left': '350px', 'margin-top': '50px'})

                                      ]),
                         ])


#########################################################
@app.callback(
    dash.dependencies.Output("stock-Chart", "figure"),
    [dash.dependencies.Input("ml_search", "value"),
     dash.dependencies.Input("dropdown2", "value")])
def update(input_value, input_value2):
    data = []
    if input_value2 == "XGBOOST":
        print('xgboost')
        company = input_value  # Anything from Yahoo - check ticker on website - NFLX (Stock) GBP=X (GBP/USD) GBPEUR=X(GBP/EUR)
        print(company)
        dataset = web.DataReader(company, "yahoo", start, end)

        df = pd.DataFrame(dataset)

        # Add Technical Indicators to Dataframe
        high = np.array(df.High)
        low = np.array(df.Low)
        close = np.array(df.Close)

        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)

        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)

        df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)

        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)

        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

        df['CMO'] = talib.CMO(close, timeperiod=14)

        df['DX'] = talib.DX(high, low, close, timeperiod=14)

        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)

        df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)

        df['MOM'] = talib.MOM(close, timeperiod=10)

        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)

        df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)

        df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)

        df['ROC'] = talib.ROC(close, timeperiod=10)

        df['ROCP'] = talib.ROCP(close, timeperiod=10)

        df['ROCR100'] = talib.ROCR100(close, timeperiod=10)

        df['RSI'] = talib.RSI(close, timeperiod=14)

        df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        df['DEMA'] = talib.DEMA(close, timeperiod=30)

        df['EMA'] = talib.EMA(close, timeperiod=30)

        df['MA'] = talib.MA(close, timeperiod=30, matype=0)

        df['SAR'] = talib.SAR(high, low, acceleration=0, maximum=0)

        df['WMA'] = talib.WMA(close, timeperiod=30)

        # Drop any rows with empty features
        df.dropna(inplace=True)
        df.isnull().sum()

        # Make a copy of the last row (today) so we can predict tomorrows price
        # Drop the columns we won't be using
        tomorrow = df.copy()
        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        tomorrow = tomorrow.drop(drop_cols, 1)

        # Shift Close back 1 so we can predict next days price
        df['Close'] = df['Close'].shift(-1)

        # Drop first 58 columns (Performs better for SVR)
        df = df.iloc[58:]  # Because of moving averages and MACD line
        df = df[:-1]  # Because of shifting close price - This removes last column this is why its not todays prediction
        #  Moreover, after shifting Close price column, last row price is equal to 0 which is not true.

        # COnvert index into number range
        df.index = range(len(df))

        test_size = 0.25

        test_split_idx = int(df.shape[0] * (1 - test_size))

        train_df = df.loc[:test_split_idx].copy()
        test_df = df.loc[test_split_idx + 1:].copy()

        from sklearn.preprocessing import MinMaxScaler

        def normalize_sets(scaler, train_df, test_df, features):
            for feature in features:
                train_df[feature] = scaler.fit_transform(train_df[feature].values.reshape(-1, 1))
                test_df[feature] = scaler.fit_transform(test_df[feature].values.reshape(-1, 1))

        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        train_df = train_df.drop(drop_cols, 1)
        test_df = test_df.drop(drop_cols, 1)

        scaler = MinMaxScaler()
        normalize_sets(scaler, train_df, test_df,
                       features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                           , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        label_column = 'Close'
        normalize_sets(scaler, train_df, test_df, features=[label_column])

        y_train = train_df[label_column].copy()
        X_train = train_df.drop([label_column], 1)
        y_test = test_df[label_column].copy()
        X_test = test_df.drop([label_column], 1)

        best_params = {'gamma': 0.001, 'learning_rate': 0.05, 'max_depth': 12, 'n_estimators': 300, 'random_state': 42}

        model = xgb.XGBRegressor(**best_params, objective='reg:squarederror')
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)

        y_true_unnorm = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
        y_pred_unnorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        predicted_prices = df.loc[test_split_idx + 1:].copy()
        predicted_prices['Close'] = y_pred_unnorm
        tc = go.Scatter(x=df.index, y=df.Close, marker_color='#94dfff', name='Ground Truth')
        tc2 = go.Scatter(x=predicted_prices.index, y=predicted_prices.Close, marker_color='#ff3d3d', name='Prediction')

        data = []
        data.append(tc)
        data.append(tc2)

        # Predicting Tomorrows Price
        tomorrow.dropna(inplace=True)
        tomorrow = tomorrow.iloc[58:]
        tomorrow.index = range(len(tomorrow))

        # In[22]:

        def normalize_new(scaler2, tomorrow, features):
            for feature in features:
                tomorrow[feature] = scaler2.fit_transform(tomorrow[feature].values.reshape(-1, 1))

        # Normalize
        scaler2 = MinMaxScaler()
        normalize_new(scaler2, tomorrow,
                      features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                          , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        # Split x y

        label_column = 'Close'
        normalize_new(scaler2, tomorrow, features=[label_column])

        predict = tomorrow.drop([label_column], 1)
        predict = predict[-1:]

        # Data used to predict tomorrow
        pred_tomorrow = predict[-1:]

        # Call the model to predict tomorrows price
        prediction = model.predict(pred_tomorrow)

        # Show tomorrows predicted price
        prediction = scaler2.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()

    elif input_value2 == "SVR":
        print("svr")
        company = input_value
        print(company)
        dataset = web.DataReader(company, "yahoo", start, end)

        df = pd.DataFrame(dataset)

        # Add Technical Indicators to Dataframe
        high = np.array(df.High)
        low = np.array(df.Low)
        close = np.array(df.Close)

        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)

        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)

        df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)

        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)

        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

        df['CMO'] = talib.CMO(close, timeperiod=14)

        df['DX'] = talib.DX(high, low, close, timeperiod=14)

        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)

        df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)

        df['MOM'] = talib.MOM(close, timeperiod=10)

        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)

        df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)

        df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)

        df['ROC'] = talib.ROC(close, timeperiod=10)

        df['ROCP'] = talib.ROCP(close, timeperiod=10)

        df['ROCR100'] = talib.ROCR100(close, timeperiod=10)

        df['RSI'] = talib.RSI(close, timeperiod=14)

        df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        df['DEMA'] = talib.DEMA(close, timeperiod=30)

        df['EMA'] = talib.EMA(close, timeperiod=30)

        df['MA'] = talib.MA(close, timeperiod=30, matype=0)

        df['SAR'] = talib.SAR(high, low, acceleration=0, maximum=0)

        df['WMA'] = talib.WMA(close, timeperiod=30)

        # Drop any rows with empty features
        df.dropna(inplace=True)
        df.isnull().sum()

        # Make a copy of the last row (today) so we can predict tomorrows price
        # Drop the columns we won't be using
        tomorrow = df.copy()
        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        tomorrow = tomorrow.drop(drop_cols, 1)

        df['Close'] = df['Close'].shift(-1)

        # Drop first 58 columns (Performs better for SVR)
        df = df.iloc[58:]  # Because of moving averages and MACD line
        df = df[:-1]  # Because of shifting close price - This removes last column this is why its not todays prediction
        #  Moreover, after shifting Close price column, last row price is equal to 0 which is not true.

        # COnvert index into number range
        df.index = range(len(df))

        test_size = 0.25

        test_split_idx = int(df.shape[0] * (1 - test_size))

        train_df = df.loc[:test_split_idx].copy()
        test_df = df.loc[test_split_idx + 1:].copy()

        from sklearn.preprocessing import MinMaxScaler

        def normalize_sets(scaler, train_df, test_df, features):
            for feature in features:
                train_df[feature] = scaler.fit_transform(train_df[feature].values.reshape(-1, 1))
                test_df[feature] = scaler.fit_transform(test_df[feature].values.reshape(-1, 1))

        # Drop
        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        train_df = train_df.drop(drop_cols, 1)
        test_df = test_df.drop(drop_cols, 1)

        # Normalize
        scaler = MinMaxScaler()
        normalize_sets(scaler, train_df, test_df,
                       features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                           , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        # Split x y

        label_column = 'Close'
        normalize_sets(scaler, train_df, test_df, features=[label_column])

        y_train = train_df[label_column].copy()
        X_train = train_df.drop([label_column], 1)
        y_test = test_df[label_column].copy()
        X_test = test_df.drop([label_column], 1)

        from sklearn import svm
        from sklearn.svm import SVR

        best_params = {'C': 1.5, 'epsilon': 0.1, 'gamma': 1e-07, 'kernel': 'linear'}

        model = svm.SVR(**best_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_true_unnorm = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
        y_pred_unnorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        predicted_prices = df.loc[test_split_idx + 1:].copy()
        predicted_prices['Close'] = y_pred_unnorm
        tc = go.Scatter(x=df.index, y=df.Close, marker_color='#94dfff', name='Ground Truth')
        tc2 = go.Scatter(x=predicted_prices.index, y=predicted_prices.Close, marker_color='#ff3d3d', name='Prediction')

        data.append(tc)
        data.append(tc2)

        # Predicting Tomorrows Price
        tomorrow.dropna(inplace=True)
        tomorrow.index = range(len(tomorrow))

        def normalize_new(scaler2, tomorrow, features):
            for feature in features:
                tomorrow[feature] = scaler2.fit_transform(tomorrow[feature].values.reshape(-1, 1))

        # Normalize
        scaler2 = MinMaxScaler()
        normalize_new(scaler2, tomorrow,
                      features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                          , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        # Split x y

        label_column = 'Close'
        normalize_new(scaler2, tomorrow, features=[label_column])

        predict = tomorrow.drop([label_column], 1)
        predict = predict[-1:]

        # Data used to predict tomorrow
        pred_tomorrow = predict[-1:]

        # Call the model to predict tomorrows price
        prediction = model.predict(pred_tomorrow)

        # Show tomorrows predicted price
        prediction = scaler2.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()

    elif input_value2 == "LIGHTGBM":
        print("lightgbm")
        company = input_value
        print(company)
        dataset = web.DataReader(company, "yahoo", start, end)

        df = pd.DataFrame(dataset)

        # Add Technical Indicators to Dataframe
        high = np.array(df.High)
        low = np.array(df.Low)
        close = np.array(df.Close)

        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)

        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)

        df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)

        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)

        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

        df['CMO'] = talib.CMO(close, timeperiod=14)

        df['DX'] = talib.DX(high, low, close, timeperiod=14)

        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)

        df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)

        df['MOM'] = talib.MOM(close, timeperiod=10)

        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)

        df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)

        df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)

        df['ROC'] = talib.ROC(close, timeperiod=10)

        df['ROCP'] = talib.ROCP(close, timeperiod=10)

        df['ROCR100'] = talib.ROCR100(close, timeperiod=10)

        df['RSI'] = talib.RSI(close, timeperiod=14)

        df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        df['DEMA'] = talib.DEMA(close, timeperiod=30)

        df['EMA'] = talib.EMA(close, timeperiod=30)

        df['MA'] = talib.MA(close, timeperiod=30, matype=0)

        df['SAR'] = talib.SAR(high, low, acceleration=0, maximum=0)

        df['WMA'] = talib.WMA(close, timeperiod=30)

        # Drop any rows with empty features
        df.dropna(inplace=True)
        df.isnull().sum()

        # Make a copy of the last row (today) so we can predict tomorrows price
        # Drop the columns we won't be using
        tomorrow = df.copy()
        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        tomorrow = tomorrow.drop(drop_cols, 1)

        df['Close'] = df['Close'].shift(-1)

        # Drop first 58 columns (Performs better for SVR)
        df = df.iloc[58:]  # Because of moving averages and MACD line
        df = df[:-1]  # Because of shifting close price - This removes last column this is why its not todays prediction
        #  Moreover, after shifting Close price column, last row price is equal to 0 which is not true.

        # COnvert index into number range
        df.index = range(len(df))

        test_size = 0.25

        test_split_idx = int(df.shape[0] * (1 - test_size))

        train_df = df.loc[:test_split_idx].copy()
        test_df = df.loc[test_split_idx + 1:].copy()

        from sklearn.preprocessing import MinMaxScaler

        def normalize_sets(scaler, train_df, test_df, features):
            for feature in features:
                train_df[feature] = scaler.fit_transform(train_df[feature].values.reshape(-1, 1))
                test_df[feature] = scaler.fit_transform(test_df[feature].values.reshape(-1, 1))

        # Drop
        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        train_df = train_df.drop(drop_cols, 1)
        test_df = test_df.drop(drop_cols, 1)

        # Normalize
        scaler = MinMaxScaler()
        normalize_sets(scaler, train_df, test_df,
                       features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                           , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        # Split x y

        label_column = 'Close'
        normalize_sets(scaler, train_df, test_df, features=[label_column])

        y_train = train_df[label_column].copy()
        X_train = train_df.drop([label_column], 1)
        y_test = test_df[label_column].copy()
        X_test = test_df.drop([label_column], 1)

        from lightgbm import LGBMRegressor

        # best_params = {'C': 1.5, 'epsilon': 0.1, 'gamma': 1e-07, 'kernel': 'linear'}

        model = LGBMRegressor()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_true_unnorm = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
        y_pred_unnorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        predicted_prices = df.loc[test_split_idx + 1:].copy()
        predicted_prices['Close'] = y_pred_unnorm
        tc = go.Scatter(x=df.index, y=df.Close, marker_color='#94dfff', name='Ground Truth')
        tc2 = go.Scatter(x=predicted_prices.index, y=predicted_prices.Close, marker_color='#ff3d3d', name='Prediction')

        data.append(tc)
        data.append(tc2)

        # Predicting Tomorrows Price
        tomorrow.dropna(inplace=True)
        tomorrow.index = range(len(tomorrow))

        def normalize_new(scaler2, tomorrow, features):
            for feature in features:
                tomorrow[feature] = scaler2.fit_transform(tomorrow[feature].values.reshape(-1, 1))

        # Normalize
        scaler2 = MinMaxScaler()
        normalize_new(scaler2, tomorrow,
                      features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                          , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        # Split x y

        label_column = 'Close'
        normalize_new(scaler2, tomorrow, features=[label_column])

        predict = tomorrow.drop([label_column], 1)
        predict = predict[-1:]

        # Data used to predict tomorrow
        pred_tomorrow = predict[-1:]

        # Call the model to predict tomorrows price
        prediction = model.predict(pred_tomorrow)

        # Show tomorrows predicted price
        prediction = scaler2.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()

    elif input_value2 == "DECISIONTREE":
        print("decisontree")
        company = input_value
        print(company)
        dataset = web.DataReader(company, "yahoo", start, end)

        df = pd.DataFrame(dataset)

        # Add Technical Indicators to Dataframe
        high = np.array(df.High)
        low = np.array(df.Low)
        close = np.array(df.Close)

        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)

        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)

        df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)

        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)

        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

        df['CMO'] = talib.CMO(close, timeperiod=14)

        df['DX'] = talib.DX(high, low, close, timeperiod=14)

        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)

        df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)

        df['MOM'] = talib.MOM(close, timeperiod=10)

        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)

        df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)

        df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)

        df['ROC'] = talib.ROC(close, timeperiod=10)

        df['ROCP'] = talib.ROCP(close, timeperiod=10)

        df['ROCR100'] = talib.ROCR100(close, timeperiod=10)

        df['RSI'] = talib.RSI(close, timeperiod=14)

        df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        df['DEMA'] = talib.DEMA(close, timeperiod=30)

        df['EMA'] = talib.EMA(close, timeperiod=30)

        df['MA'] = talib.MA(close, timeperiod=30, matype=0)

        df['SAR'] = talib.SAR(high, low, acceleration=0, maximum=0)

        df['WMA'] = talib.WMA(close, timeperiod=30)

        # Drop any rows with empty features
        df.dropna(inplace=True)
        df.isnull().sum()

        # Make a copy of the last row (today) so we can predict tomorrows price
        # Drop the columns we won't be using
        tomorrow = df.copy()
        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        tomorrow = tomorrow.drop(drop_cols, 1)

        df['Close'] = df['Close'].shift(-1)

        # Drop first 58 columns (Performs better for SVR)
        df = df.iloc[58:]  # Because of moving averages and MACD line
        df = df[:-1]  # Because of shifting close price - This removes last column this is why its not todays prediction
        #  Moreover, after shifting Close price column, last row price is equal to 0 which is not true.

        # COnvert index into number range
        df.index = range(len(df))

        test_size = 0.25

        test_split_idx = int(df.shape[0] * (1 - test_size))

        train_df = df.loc[:test_split_idx].copy()
        test_df = df.loc[test_split_idx + 1:].copy()

        from sklearn.preprocessing import MinMaxScaler

        def normalize_sets(scaler, train_df, test_df, features):
            for feature in features:
                train_df[feature] = scaler.fit_transform(train_df[feature].values.reshape(-1, 1))
                test_df[feature] = scaler.fit_transform(test_df[feature].values.reshape(-1, 1))

        # Drop
        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        train_df = train_df.drop(drop_cols, 1)
        test_df = test_df.drop(drop_cols, 1)

        # Normalize
        scaler = MinMaxScaler()
        normalize_sets(scaler, train_df, test_df,
                       features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                           , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        # Split x y

        label_column = 'Close'
        normalize_sets(scaler, train_df, test_df, features=[label_column])

        y_train = train_df[label_column].copy()
        X_train = train_df.drop([label_column], 1)
        y_test = test_df[label_column].copy()
        X_test = test_df.drop([label_column], 1)

        from sklearn.tree import DecisionTreeRegressor

        # best_params = {'C': 1.5, 'epsilon': 0.1, 'gamma': 1e-07, 'kernel': 'linear'}

        model = DecisionTreeRegressor(max_depth=10)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_true_unnorm = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
        y_pred_unnorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        predicted_prices = df.loc[test_split_idx + 1:].copy()
        predicted_prices['Close'] = y_pred_unnorm
        tc = go.Scatter(x=df.index, y=df.Close, marker_color='#94dfff', name='Ground Truth')
        tc2 = go.Scatter(x=predicted_prices.index, y=predicted_prices.Close, marker_color='#ff3d3d', name='Prediction')

        data.append(tc)
        data.append(tc2)

        # Predicting Tomorrows Price
        tomorrow.dropna(inplace=True)
        tomorrow.index = range(len(tomorrow))

        def normalize_new(scaler2, tomorrow, features):
            for feature in features:
                tomorrow[feature] = scaler2.fit_transform(tomorrow[feature].values.reshape(-1, 1))

        # Normalize
        scaler2 = MinMaxScaler()
        normalize_new(scaler2, tomorrow,
                      features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                          , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        # Split x y

        label_column = 'Close'
        normalize_new(scaler2, tomorrow, features=[label_column])

        predict = tomorrow.drop([label_column], 1)
        predict = predict[-1:]

        # Data used to predict tomorrow
        pred_tomorrow = predict[-1:]

        # Call the model to predict tomorrows price
        prediction = model.predict(pred_tomorrow)

        # Show tomorrows predicted price
        prediction = scaler2.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()

    elif input_value2 == "RANDOMFOREST":
        print("randomforest")
        company = input_value
        print(company)
        dataset = web.DataReader(company, "yahoo", start, end)

        df = pd.DataFrame(dataset)

        # Add Technical Indicators to Dataframe
        high = np.array(df.High)
        low = np.array(df.Low)
        close = np.array(df.Close)

        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)

        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)

        df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)

        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)

        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

        df['CMO'] = talib.CMO(close, timeperiod=14)

        df['DX'] = talib.DX(high, low, close, timeperiod=14)

        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)

        df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)

        df['MOM'] = talib.MOM(close, timeperiod=10)

        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)

        df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)

        df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)

        df['ROC'] = talib.ROC(close, timeperiod=10)

        df['ROCP'] = talib.ROCP(close, timeperiod=10)

        df['ROCR100'] = talib.ROCR100(close, timeperiod=10)

        df['RSI'] = talib.RSI(close, timeperiod=14)

        df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        df['DEMA'] = talib.DEMA(close, timeperiod=30)

        df['EMA'] = talib.EMA(close, timeperiod=30)

        df['MA'] = talib.MA(close, timeperiod=30, matype=0)

        df['SAR'] = talib.SAR(high, low, acceleration=0, maximum=0)

        df['WMA'] = talib.WMA(close, timeperiod=30)

        # Drop any rows with empty features
        df.dropna(inplace=True)
        df.isnull().sum()

        # Make a copy of the last row (today) so we can predict tomorrows price
        # Drop the columns we won't be using
        tomorrow = df.copy()
        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        tomorrow = tomorrow.drop(drop_cols, 1)

        df['Close'] = df['Close'].shift(-1)

        # Drop first 58 columns (Performs better for SVR)
        df = df.iloc[58:]  # Because of moving averages and MACD line
        df = df[:-1]  # Because of shifting close price - This removes last column this is why its not todays prediction
        #  Moreover, after shifting Close price column, last row price is equal to 0 which is not true.

        # COnvert index into number range
        df.index = range(len(df))

        test_size = 0.25

        test_split_idx = int(df.shape[0] * (1 - test_size))

        train_df = df.loc[:test_split_idx].copy()
        test_df = df.loc[test_split_idx + 1:].copy()

        from sklearn.preprocessing import MinMaxScaler

        def normalize_sets(scaler, train_df, test_df, features):
            for feature in features:
                train_df[feature] = scaler.fit_transform(train_df[feature].values.reshape(-1, 1))
                test_df[feature] = scaler.fit_transform(test_df[feature].values.reshape(-1, 1))

        # Drop
        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        train_df = train_df.drop(drop_cols, 1)
        test_df = test_df.drop(drop_cols, 1)

        # Normalize
        scaler = MinMaxScaler()
        normalize_sets(scaler, train_df, test_df,
                       features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                           , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        # Split x y

        label_column = 'Close'
        normalize_sets(scaler, train_df, test_df, features=[label_column])

        y_train = train_df[label_column].copy()
        X_train = train_df.drop([label_column], 1)
        y_test = test_df[label_column].copy()
        X_test = test_df.drop([label_column], 1)

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import mean_squared_error

        # best_params = {'C': 1.5, 'epsilon': 0.1, 'gamma': 1e-07, 'kernel': 'linear'}

        model = RandomForestRegressor(max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_true_unnorm = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
        y_pred_unnorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        predicted_prices = df.loc[test_split_idx + 1:].copy()
        predicted_prices['Close'] = y_pred_unnorm
        tc = go.Scatter(x=df.index, y=df.Close, marker_color='#94dfff', name='Ground Truth')
        tc2 = go.Scatter(x=predicted_prices.index, y=predicted_prices.Close, marker_color='#ff3d3d', name='Prediction')

        data.append(tc)
        data.append(tc2)

        # Predicting Tomorrows Price
        tomorrow.dropna(inplace=True)
        tomorrow.index = range(len(tomorrow))

        def normalize_new(scaler2, tomorrow, features):
            for feature in features:
                tomorrow[feature] = scaler2.fit_transform(tomorrow[feature].values.reshape(-1, 1))

        # Normalize
        scaler2 = MinMaxScaler()
        normalize_new(scaler2, tomorrow,
                      features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                          , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        # Split x y

        label_column = 'Close'
        normalize_new(scaler2, tomorrow, features=[label_column])

        predict = tomorrow.drop([label_column], 1)
        predict = predict[-1:]

        # Data used to predict tomorrow
        pred_tomorrow = predict[-1:]

        # Call the model to predict tomorrows price
        prediction = model.predict(pred_tomorrow)

        # Show tomorrows predicted price
        prediction = scaler2.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()

    elif input_value2 == "LINEARREGRESSION":
        print("linearregression")
        company = input_value
        print(company)
        dataset = web.DataReader(company, "yahoo", start, end)

        df = pd.DataFrame(dataset)

        # Add Technical Indicators to Dataframe
        high = np.array(df.High)
        low = np.array(df.Low)
        close = np.array(df.Close)

        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)

        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)

        df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)

        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)

        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

        df['CMO'] = talib.CMO(close, timeperiod=14)

        df['DX'] = talib.DX(high, low, close, timeperiod=14)

        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)

        df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)

        df['MOM'] = talib.MOM(close, timeperiod=10)

        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)

        df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)

        df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)

        df['ROC'] = talib.ROC(close, timeperiod=10)

        df['ROCP'] = talib.ROCP(close, timeperiod=10)

        df['ROCR100'] = talib.ROCR100(close, timeperiod=10)

        df['RSI'] = talib.RSI(close, timeperiod=14)

        df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        df['DEMA'] = talib.DEMA(close, timeperiod=30)

        df['EMA'] = talib.EMA(close, timeperiod=30)

        df['MA'] = talib.MA(close, timeperiod=30, matype=0)

        df['SAR'] = talib.SAR(high, low, acceleration=0, maximum=0)

        df['WMA'] = talib.WMA(close, timeperiod=30)

        # Drop any rows with empty features
        df.dropna(inplace=True)
        df.isnull().sum()

        # Make a copy of the last row (today) so we can predict tomorrows price
        # Drop the columns we won't be using
        tomorrow = df.copy()
        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        tomorrow = tomorrow.drop(drop_cols, 1)

        df['Close'] = df['Close'].shift(-1)

        # Drop first 58 columns (Performs better for SVR)
        df = df.iloc[58:]  # Because of moving averages and MACD line
        df = df[:-1]  # Because of shifting close price - This removes last column this is why its not todays prediction
        #  Moreover, after shifting Close price column, last row price is equal to 0 which is not true.

        # COnvert index into number range
        df.index = range(len(df))

        test_size = 0.25

        test_split_idx = int(df.shape[0] * (1 - test_size))

        train_df = df.loc[:test_split_idx].copy()
        test_df = df.loc[test_split_idx + 1:].copy()

        from sklearn.preprocessing import MinMaxScaler

        def normalize_sets(scaler, train_df, test_df, features):
            for feature in features:
                train_df[feature] = scaler.fit_transform(train_df[feature].values.reshape(-1, 1))
                test_df[feature] = scaler.fit_transform(test_df[feature].values.reshape(-1, 1))

        # Drop
        drop_cols = ['Volume', 'Open', 'Low', 'High', 'Adj Close']
        train_df = train_df.drop(drop_cols, 1)
        test_df = test_df.drop(drop_cols, 1)

        # Normalize
        scaler = MinMaxScaler()
        normalize_sets(scaler, train_df, test_df,
                       features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                           , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        # Split x y

        label_column = 'Close'
        normalize_sets(scaler, train_df, test_df, features=[label_column])

        y_train = train_df[label_column].copy()
        X_train = train_df.drop([label_column], 1)
        y_test = test_df[label_column].copy()
        X_test = test_df.drop([label_column], 1)

        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import mean_squared_error

        # best_params = {'C': 1.5, 'epsilon': 0.1, 'gamma': 1e-07, 'kernel': 'linear'}

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_true_unnorm = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
        y_pred_unnorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        predicted_prices = df.loc[test_split_idx + 1:].copy()
        predicted_prices['Close'] = y_pred_unnorm
        tc = go.Scatter(x=df.index, y=df.Close, marker_color='#94dfff', name='Ground Truth')
        tc2 = go.Scatter(x=predicted_prices.index, y=predicted_prices.Close, marker_color='#ff3d3d', name='Prediction')

        data.append(tc)
        data.append(tc2)

        # Predicting Tomorrows Price
        tomorrow.dropna(inplace=True)
        tomorrow.index = range(len(tomorrow))

        def normalize_new(scaler2, tomorrow, features):
            for feature in features:
                tomorrow[feature] = scaler2.fit_transform(tomorrow[feature].values.reshape(-1, 1))

        # Normalize
        scaler2 = MinMaxScaler()
        normalize_new(scaler2, tomorrow,
                      features=['ADX', 'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM',
                                'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR100', 'RSI', 'ULTOSC'
                          , 'WILLR', 'DEMA', 'EMA', 'MA', 'SAR', 'WMA'])

        # Split x y

        label_column = 'Close'
        normalize_new(scaler2, tomorrow, features=[label_column])

        predict = tomorrow.drop([label_column], 1)
        predict = predict[-1:]

        # Data used to predict tomorrow
        pred_tomorrow = predict[-1:]

        # Call the model to predict tomorrows price
        prediction = model.predict(pred_tomorrow)

        # Show tomorrows predicted price
        prediction = scaler2.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()

        rmse = 20

    # layout = {"title": "Tommorow's prediction : " + str(prediction)}

    layout = go.Layout(
        template='plotly_dark',
        xaxis_title='Index Range',
        yaxis_title='Price ($)',
        yaxis_gridcolor='rgba(223, 176, 255, 0.25)',
        xaxis_gridcolor='rgba(223, 176, 255, 0.25)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        height=600,
        margin={'b': 15},
        hovermode='x',
        autosize=False,
        title={'text': str(company) + ' - Tomorrows Predicted Price: ' + str(prediction) + ' using ' + str(
            input_value2), 'font': {'color': 'white'},
               'x': 0.5},
        xaxis={'rangeslider': {'visible': False},
               'range': [df.index.min(), df.index.max()]},
    )
    return {"data": data, "layout": layout}


#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################


#################################################################################################
#################################################################################################
################################################################################################
###############################################################################################
#########################################################

# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return home_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server(debug=False)
