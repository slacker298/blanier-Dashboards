# Last update 6-9 to update our refresh key

import os
import time
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from textwrap import dedent as d

import pandas as pd #Pandas-- data framework
import numpy as np #Numpy-- logic framework
# import scipy #SciPy-- scientific function/ math framework
import datetime #for writing out unique files
import sys #Supports pick up of files from directories
import math
import time
import json
import ciso8601
import requests

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)

import urllib
class TDAmeritrade:

    def __init__(self, refresh_token=None, consumer_key=None, access_token=None):
        if refresh_token != None:
            self.refresh_token = refresh_token
        else:
            self.refresh_token = pd.read_csv('refresh_key.csv').at[0,'refresh_token'] # Grab the latest - will be switched to memsql query later

        if consumer_key != None:
            self.consumer_key = consumer_key
        else:
            self.consumer_key = '#########' # Hardcoded for October2020

        self.access_token_timer = round(time.time()) + 1800
        if access_token != None:
            self.access_token = access_token
        else:
            self.access_token = self.get_refresh_key()['access_token']


    def get_refresh_key(self, new_refresh=False):
#         https://developer.tdameritrade.com/authentication/apis/post/token-0
        self.check_token_time()

        refresh_token=self.refresh_token
        consumer_key=self.consumer_key

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        if new_refresh == True:
            body = {
                "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "access_type": 'offline',
            "client_id": '{}@AMER.OAUTHAP'.format(consumer_key), #(yes, add "@AMER.OAUTHAP" without quotes)
                }
        else:
            body = {
                "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": '{}@AMER.OAUTHAP'.format(consumer_key), #(yes, add "@AMER.OAUTHAP" without quotes)
                }

        r = requests.post('https://api.tdameritrade.com/v1/oauth2/token', data=body, headers=headers)

        r2 = r.json()

        r2['create_ts'] = round(time.time())

        return r2

    def check_token_time(self):
        if self.access_token_timer <= time.time():
            self.update_access_token()
        return

    def update_access_token(self):
        self.access_token_timer = round(time.time()) + 1800
        self.access_token = self.get_refresh_key()['access_token']


    def save_refresh_key(self):
        d_ = self.get_refresh_key(new_refresh=True)
        pd.DataFrame.from_dict(d_, orient='index').to_csv('refresh_key.csv')


    def get_option_chain(self, body):
#         https://developer.tdameritrade.com/option-chains/apis/get/marketdata/chains#
        self.check_token_time()

        headers = {'Content-Type': 'application/json', 'Authorization':'Bearer {}'.format(self.access_token)}

        body['apikey'] = self.consumer_key

        request_str = 'https://api.tdameritrade.com/v1/marketdata/chains?apikey={apikey}'.format(apikey=self.consumer_key)

        for k in body.keys():
            v = body[k]
            if v != '':
                add_s = '&{k}={v}'.format(k=k,v=v)
                request_str += add_s

        r = requests.get(request_str, headers=headers)

        r2 = r.json()

        return r2

    def get_price_history(self, symbol, body):
#         https://developer.tdameritrade.com/price-history/apis/get/marketdata/%7Bsymbol%7D/pricehistory
        self.check_token_time()

        # Need to add alot of settings to this one - check the url

        headers = {'Content-Type': 'application/json', 'Authorization':'Bearer {}'.format(self.access_token)}

        request_str = 'https://api.tdameritrade.com/v1/marketdata/{symbol}/pricehistory?apikey={apikey}'.format(apikey=self.consumer_key, symbol=symbol)

        for k in body.keys():
            v = body[k]
            if v != '':
                add_s = '&{k}={v}'.format(k=k,v=v)
                request_str += add_s

        r = requests.get(request_str, headers=headers)

        r2 = r.json()

        return r2


    def get_quotes(self, symbol):

        self.check_token_time()

        headers = {'Content-Type': 'application/json', 'Authorization':'Bearer {}'.format(self.access_token)}

        if len(symbol) == 1:
            symbol = symbol[0]
            r = requests.get('https://api.tdameritrade.com/v1/marketdata/{symbol}/quotes?apikey={apikey}'.format(apikey=self.consumer_key, symbol=symbol))
        else:
            s = ''
            for sym in symbol:
                s = s + sym + ','

            s = s[:-1]
            symbols = urllib.parse.quote(s)

            r = requests.get('https://api.tdameritrade.com/v1/marketdata/quotes?apikey={apikey}&symbol={symbol}'.format(apikey=self.consumer_key, symbol=symbols))
        r2 = r.json()

        return r2


    ### Lets define some custom sell functions ###
    def limit_trailing_sell(self, symbol, min_price, trail_amt, perc=False):

        ## Get quote data from symbol

#         max_price = max(prices)

#         if max_price >= min_price: # Enter the sell logic

#             # Compare if we have trailed enough to sell

#             if perc!=False: # It's a percent trail
#                 trail_amt = max_price*(trail_amt/100)

#             sell_price = max_price - trail_amt

#             if current_price <= sell_price:

#                 self.execute_trade(symbol, price, etc...)




        return

    def days_to_exp(self, exp_date, curr_date):

      if (type(exp_date) == int) or (type(exp_date) == float):

          diff = (exp_date/1000) - (curr_date/1000)

      else:

          exp_date = str(exp_date)[0:10] + ' 15:00:00'
          diff = time.mktime(datetime.datetime.strptime(exp_date, "%Y-%m-%d %H:%M:%S").timetuple()) - (curr_date/1000)

      r = (diff/60/60/24)

      return r


    def get_option_chain_df(self, body):

        oc = self.get_option_chain(body)

        call_data = oc['callExpDateMap']
        put_data = oc['putExpDateMap']
        und_price = oc['underlyingPrice']

        call_df = pd.io.json.json_normalize(call_data).melt()
        put_df = pd.io.json.json_normalize(put_data).melt()

        call_df['value'] = call_df['value'].apply(lambda x: x[0])
        put_df['value'] = put_df['value'].apply(lambda x: x[0])


        call_df2 = pd.io.json.json_normalize(call_df['value'])
        put_df2 = pd.io.json.json_normalize(put_df['value'])

        rdf = pd.concat([call_df2, put_df2], ignore_index=True)
        rdf['underlyingPrice'] = und_price
        rdf['und_symbol'] = body['symbol']


        rdf['expirationDateStr'] = pd.to_datetime(rdf['expirationDate'],unit='ms')
        rdf['expirationDateStr'] = rdf['expirationDateStr'].apply(lambda x: str(x)[0:10])

        rdf['dte_calc'] = rdf.apply(lambda row: self.days_to_exp(row['expirationDate'], row['quoteTimeInLong']),axis=1)


        return rdf


    def calc_profit(self, sell_price, buy_price, strike, volume, type_):

        if type_ == 'PUT':
            profit = max(((strike - (sell_price-buy_price))*volume), -1*buy_price*volume)
        else:
            profit = max((((sell_price-buy_price)-strike)*volume), -1*buy_price*volume)

        return profit


    def bulk_calc_profit_df(self, df_, sell_price, volume):

        df_['profit'] = df_.apply(lambda row: self.calc_profit(sell_price, row['ask'], row['strikePrice'], volume, row['putCall']),axis=1)
        df_['sell_price'] = sell_price

        return df_

    def generate_option_symbols(self, underlying, strikes, dates, types='both'):

        if types=='both':
            typelist = ['C', 'P']
        elif types=='PUT':
            typelist = ['P']
        elif types=='CALL':
            typelist = ['C']

        symbol_list = []
        for k in strikes:
            for d in dates:
                for t in typelist:
                    s = '{und}_{date}{t}{strike}'.format(und=underlying, date=d,t=t,strike=k)
                    symbol_list.append(s)

        return symbol_list

    def execute_trade_equity(self, order_type, price, qty, buysell, symbol, duration='DAY'):

        self.check_token_time()

        headers = {'Content-Type': 'application/json', 'Authorization':'Bearer {}'.format(self.access_token)}

        trade_request = {
                'orderType':order_type,
                "session": 'NORMAL',
                "duration": duration,
                'orderStrategyType': 'SINGLE',
                'orderLegCollection': [{
                    "instruction": buysell,
                    'quantity': qty,
                    'instrument': {
                        'symbol': symbol,
                        'assetType': 'EQUITY'}
                }]
            }

        if order_type == 'LIMIT':
            trade_request['price'] = price

        r = requests.post('https://api.tdameritrade.com/v1/accounts/#########/orders', data=json.dumps(trade_request), headers=headers)

        return r

    def execute_trade_option(self, order_type, price, qty, buysell, symbol, exp_date, putCall, strike, duration='DAY'):

        self.check_token_time()

        headers = {'Content-Type': 'application/json', 'Authorization':'Bearer {}'.format(self.access_token)}

        sym_date = exp_date[5:7]+exp_date[8:]+exp_date[2:4]
        option_symbol = self.generate_option_symbols(symbol, [strike], [sym_date], types=putCall)[0]


        trade_request = {
                'orderType':order_type,
                "session": 'NORMAL',
                "duration": duration,
                'orderStrategyType': 'SINGLE',
                'orderLegCollection': [{
                    "instruction": buysell,
                    'quantity': qty,
                    'instrument': {
                        'symbol': option_symbol,
                        'assetType': 'OPTION',
                         "putCall": putCall
                    }
                }]
            }

        if order_type == 'LIMIT':
            trade_request['price'] = price

        r = requests.post('https://api.tdameritrade.com/v1/accounts/#########/orders', data=json.dumps(trade_request), headers=headers)

        return r

    def get_option_chain_flowalgo_db(self, symbol, putCall, strike, exp_date):

        self.check_token_time()

        headers = {'Content-Type': 'application/json', 'Authorization':'Bearer {}'.format(self.access_token)}

        body = {}
        body['apikey'] = self.consumer_key
        body['symbol'] = symbol
        body['contractType'] = putCall
        body['strike'] = strike
        body['fromDate'] = exp_date
        body['toDate'] = exp_date

        request_str = 'https://api.tdameritrade.com/v1/marketdata/chains?apikey={apikey}'.format(apikey=self.consumer_key)

        for k in body.keys():
            v = body[k]
            if v != '':
                add_s = '&{k}={v}'.format(k=k,v=v)
                request_str += add_s

        r = requests.get(request_str, headers=headers)

        r2 = r.json()

        return r2

    def get_option_chain_flowalgo_db_df(self, symbol, putCall, strike, exp_date):

        oc = self.get_option_chain_flowalgo_db(symbol, putCall, strike, exp_date)

        call_data = oc['callExpDateMap']
        put_data = oc['putExpDateMap']
        und_price = oc['underlyingPrice']

        call_df = pd.io.json.json_normalize(call_data).melt()
        put_df = pd.io.json.json_normalize(put_data).melt()

        call_df['value'] = call_df['value'].apply(lambda x: x[0])
        put_df['value'] = put_df['value'].apply(lambda x: x[0])


        call_df2 = pd.io.json.json_normalize(call_df['value'])
        put_df2 = pd.io.json.json_normalize(put_df['value'])

        rdf = pd.concat([call_df2, put_df2], ignore_index=True)
        rdf['underlyingPrice'] = und_price
        rdf = rdf.drop('value',axis=1)

        return rdf

    def get_account_positions(self):
#     https://developer.tdameritrade.com/account-access/apis/get/accounts/%7BaccountId%7D-0
        self.check_token_time()

        headers = {'Content-Type': 'application/json', 'Authorization':'Bearer {}'.format(self.access_token)}

        acct_url = 'https://api.tdameritrade.com/v1/accounts/#########?fields=positions%2Corders'

        r = requests.get(acct_url, headers=headers)

        r2 = r.json()

        return r2

    def get_symbols(self, sym):
#     https://developer.tdameritrade.com/instruments/apis/get/instruments
        self.check_token_time()

        headers = {'Content-Type': 'application/json', 'Authorization':'Bearer {}'.format(self.access_token)}

        acct_url = 'https://api.tdameritrade.com/v1/instruments?symbol={}&projection=symbol-regex'.format(sym)

        r = requests.get(acct_url, headers=headers)

        r2 = r.json()

        return r2

def stringify_eid(eids):
    s = ''
    for id_ in eids:
        s += "'{}',".format(id_)
    return s[:-1]

def random_color():
    rgbl=[random.randint(0,255),random.randint(0,255),random.randint(0,255)]
    random.shuffle(rgbl)
    return tuple(rgbl)


### GRAPHING AND CALCULATING FUNCTIONS ###

def rolling_consecutives(s):

    cons = 0
    outlist = []
    for i in range(len(s)):

        if s[i] == 1:
            cons += 1
        else:
            cons = 0

        outlist.append(cons)

    return outlist

def rsi_calc(rs):

    try:
        r = 100 - (100 / (1+rs))
    except:
        r = np.nan

    return r

def calc_avgUD(col, n=14):

    rlist = [np.nan]*n
    for i in range(n, len(col)):

        avg = col[i-n:i].mean()
        rlist.append(avg)

    return rlist

def wilders_smoothing(alpha, val, old_ws):

    avgt = (alpha * val) + ((1-alpha) * old_ws)

    return avgt

def calc_avgUD_ws(col, n=14):

    rlist = [0]*n
    for i in range(n, len(col)):

        avg = wilders_smoothing(1/n, col[i], rlist[i-1])
        rlist.append(avg)

    return rlist

def get_historical_stock_data(tda_function, symbol, startDate, endDate, frequencyType='minute', frequency=1, ext_hours='true', rsi_method='wilders'):

    if type(startDate)==str:
        startDate = int(time.mktime(ciso8601.parse_datetime(startDate).timetuple())*1000)
    if type(endDate)==str:
        endDate = int(time.mktime(ciso8601.parse_datetime(endDate).timetuple())*1000)
    if ext_hours==True:
        ext_hours='true'

    payload = {'startDate':startDate, 'endDate':endDate, 'frequency':frequency, 'frequencyType':frequencyType, 'needExtendedHoursData':ext_hours}
    if frequencyType not in ['minute', 'hour']: #fkn td documenation blows  https://developer.tdameritrade.com/content/price-history-samples
        payload['periodType']='month'

    ## We like 45min and 4hr aggs....but we gotta make them oursevles
    if (frequencyType=='minute') and (frequency==45):

        payload['frequency'] = 15

        rdf_pull = tda_function.get_price_history(symbol, payload)
        rdf = pd.DataFrame(rdf_pull['candles'])

        rdf['time_group'] = [round((x+2)/3)*3 for x in rdf.index.values]
        rdf = rdf.groupby('time_group',as_index=False).agg({'open': 'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum', 'datetime':'first'}).drop('time_group',axis=1)

    elif (frequencyType=='hour'):
        payload['frequency'] = 30
        payload['frequencyType'] = 'minute'

        rdf_pull = tda_function.get_price_history(symbol, payload)
        rdf = pd.DataFrame(rdf_pull['candles'])

        h=frequency
        h=h*2

        rdf['time_group'] = [round((x)//(h))*(h) for x in rdf.index.values]
        rdf = rdf.groupby('time_group',as_index=False).agg({'open': 'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum', 'datetime':'first'}).drop('time_group',axis=1)

    else:

        rdf_pull = tda_function.get_price_history(symbol, payload)
        rdf = pd.DataFrame(rdf_pull['candles'])

    rdf['symbol'] = rdf_pull['symbol']
    rdf['date'] = rdf['datetime'].apply(lambda x: datetime.datetime.utcfromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    rdf['change'] = rdf['close'] - rdf['open']
    rdf['spread'] = rdf['high'] - rdf['low']

    rdf['lag_change'] = rdf['change'].shift()
    rdf['change_diff'] = rdf['change'] - rdf['lag_change']
    rdf['spread_diff'] = rdf['spread'].diff()

    rdf['neg_change'] = rdf['change'].apply(lambda x : x<0).astype('int')
    rdf['pos_change'] = rdf['change'].apply(lambda x : x>=0).astype('int')

    rdf['consec_neg'] = rolling_consecutives(rdf['neg_change'].values)
    rdf['consec_pos'] = rolling_consecutives(rdf['pos_change'].values)

    rdf['close_diff'] = rdf['close'].diff()
    rdf['pos_close_diff'] = rdf['close_diff'].apply(lambda x : max(x,0))
    rdf['neg_close_diff'] = rdf['close_diff'].apply(lambda x : abs(min(x,0)))


    if rsi_method == 'wilders':
        rdf['AvgU'] = calc_avgUD_ws(rdf['pos_close_diff'])
        rdf['AvgD'] = calc_avgUD_ws(rdf['neg_close_diff'])
    else:
        rdf['AvgU'] = calc_avgUD(rdf['pos_close_diff'])
        rdf['AvgD'] = calc_avgUD(rdf['neg_close_diff'])

    rdf['RS'] = rdf['AvgU']/rdf['AvgD']
    rdf['RSI'] = rdf['RS'].apply(lambda x: rsi_calc(x))
    rdf = rdf.drop(['close_diff', 'pos_close_diff', 'neg_close_diff', 'AvgU', 'AvgD', 'RS'],axis=1)

    return rdf

def graph_rsi(gdf, rsi_name='RSI', overbought=70, underbought=30):

    fig = go.Figure()

    fig.add_shape(
        # Line Horizontal
            type="line",
            x0=0,
            y0=overbought,
            x1=400,
            y1=overbought,
            line=dict(
                color="red",
                width=1.5,
            ))

    fig.add_shape(
        # Line Horizontal
            type="line",
            x0=0,
            y0=underbought,
            x1=400,
            y1=underbought,
            line=dict(
                color="red",
                width=1.5,
            ))

    actual_line = go.Scatter(
        x=gdf.index.values,
        y=gdf[rsi_name],
        mode='lines+markers',
        opacity=0.7,
        fill='tonexty',
        marker={
            'size': 5,
            'line': {'width': 0.5, 'color': 'white'}
        }
    )

    fig.add_trace(actual_line)

    fig.update_layout({'xaxis':{'title': 'minute'},
            'yaxis':{'title': 'RSI', 'range':[0,100]},
            'height':300,
            #margin={'l': 40, 'b': 40, 't': 10, 'r': 10, 'pad':4},
            # 'legend':{'x': 0, 'y': 1},
            'hovermode':'closest'})

    fig.update_layout(legend_orientation="h")
    fig.update_layout(legend=dict(x=-.1, y=1.1))

    return {'data': fig.data,
        'layout': fig.layout}

def plot_candles(df):
#     https://plotly.com/python/candlestick-charts/

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    candles = go.Candlestick(x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=df.at[0,'symbol'])

    fig.add_trace(candles,secondary_y = False)

    bar_graph = go.Bar(
            x = df['date'],
            y = df['volume'],
            marker_color='blue',
            opacity=0.5,
            name='Volume'
        )

    fig.add_trace(bar_graph,secondary_y = True)
    fig.update_layout(legend_orientation="h")
    fig.update_layout(legend=dict(x=-.1, y=1.1))

    return {'data': fig.data,
        'layout': fig.layout}

def ema(input_df_, n, xname='close', extra_name = '', return_df=True):

    df_ = input_df_.copy()
    sma_col = '{x}{n}day_SMA'.format(n=n, x=extra_name)
    ema_col = '{x}{n}day_EMA'.format(n=n, x=extra_name)

    df_[sma_col] = df_[xname].rolling(window=n).mean()

    m = (2/ (n+1))

    prev_ema = df_[sma_col].dropna().values[0]
    nnans = df_.loc[df_[sma_col].isna()].shape[0]

    ema_list = [np.nan]*nnans + [prev_ema]

    cdf = df_[[xname, sma_col]].dropna().reset_index(drop=True)

    for i in range(1, cdf.shape[0]):

        close = cdf.at[i,xname]
        curr_ema = ((close-prev_ema) * m) + prev_ema
        ema_list.append(curr_ema)
        prev_ema = curr_ema

    if return_df == True:
        df_[ema_col] = ema_list
        return df_

    return ema_list


def graph_options_volume(tda_api, symbol, date, stack_bars=False):

    df_ = tda_api.get_option_chain_df({'symbol':symbol, 'fromDate':date, 'toDate':date})
    df_['bid_ask_spread'] = abs(df_['bid'] - df_['ask'])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colormap = {'PUT':'red', 'CALL':'green'}

    for tn in df_['putCall'].unique():
        gdf = df_.loc[df_['putCall']==tn]

        actual_line = go.Scatter(
            x=gdf['strikePrice'],
            y=gdf['bid_ask_spread'],
            text=gdf['strikePrice'],
            mode='lines+markers',
            line=dict(color=colormap[tn]),
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=(tn),
        )

        fig.add_trace(actual_line,secondary_y = False)

        bar_graph = go.Bar(
                x = gdf['strikePrice'],
                y = gdf['openInterest'],
                marker_color=colormap[tn],
                opacity=0.5,
                name='Open Interest {}'.format(tn)
            )

        fig.add_trace(bar_graph,secondary_y = True)

        bar_graph = go.Bar(
                x = gdf['strikePrice'],
                y = gdf['totalVolume'],
                marker_color=colormap[tn],
                opacity=0.3,
                name='Volume {}'.format(tn)
            )

        fig.add_trace(bar_graph,secondary_y = True)

    if stack_bars == 'True':
        fig.update_layout(barmode='stack')

    fig.update_layout({'xaxis':{'title': 'Strike Price'},# 'range':[200,330]},
            'yaxis':{'title': 'Bid-Ask Spread'},
            'yaxis2' : {'title':'Volume'},
            'height':700,
            #margin={'l': 40, 'b': 40, 't': 10, 'r': 10, 'pad':4},
            # 'legend':{'x': 0, 'y': 1},
            'hovermode':'closest'})

    fig.update_layout(title='Options for {sym} with ExpDate {d}'.format(sym=symbol, d=date), title_y = 0.95, title_x=0.5, title_font_size=30)
    fig.update_layout(legend_orientation="h")
    fig.update_layout(legend=dict(x=-.1, y=1.1))

    return {'data': fig.data,
        'layout': fig.layout}

def itm_calc(putcall, strike, und_price):

    if putcall == 'PUT':
        if und_price < strike:
            return True
        return False
    else:
        if und_price < strike:
            return False
        return True

def max_pain_calc(putcall, strike, und_price, open_int):

    if putcall == 'PUT':
        mp = ((strike - und_price) * open_int) * 100
    if putcall == 'CALL':
        mp = ((und_price - strike) * open_int) * 100
    return mp


def calc_max_pain_graph(df_, min_range=0, max_range=420):

    df_2 = pd.concat([df_]*(max_range-min_range))
    plist = []
    plen = df_.shape[0]
    for i in range(min_range, max_range):
        p = list(np.ones(plen)*i)
        plist += p

    df_2['und_price'] = plist

    df_2['ITM'] = df_2.apply(lambda row: itm_calc(row['putCall'], row['strikePrice'], row['und_price']),axis=1)
    df_3 = df_2.loc[df_2['ITM']==True]
    df_3['max_pain_value'] = df_3.apply(lambda row: max_pain_calc(row['putCall'], row['strikePrice'], row['und_price'], row['openInterest']),axis=1)
    df_4 = df_3.groupby(['und_price', 'putCall']).sum().sort_values(by='max_pain_value').reset_index()[['putCall','und_price', 'max_pain_value']]
    rdf = df_4
    return rdf

def graph_max_pain(tda_api, symbol, date):#, minrange, maxrange):

    curr_unix = time.time()
    curr_dt = ciso8601.parse_datetime(datetime.datetime.utcfromtimestamp(curr_unix).strftime('%Y-%m-%d'))
    dayofweek = curr_dt.weekday()

    while dayofweek not in [0,1,2,3,4]:
        curr_unix -= 86400
        curr_dt = ciso8601.parse_datetime(datetime.datetime.utcfromtimestamp(curr_unix).strftime('%Y-%m-%d'))
        dayofweek = curr_dt.weekday()

    last_trading_day = datetime.datetime.utcfromtimestamp(curr_unix).strftime('%Y-%m-%d')
#     print(last_trading_day)
    payload = {'frequency':1, 'frequencyType':'daily', 'needExtendedHoursData':'false'}
    payload['periodType']='month'

    sdfpull = tda_api.get_price_history(symbol, payload)
    sdf = pd.DataFrame(sdfpull['candles'])

    # hl2 = round(((max(0,int(sdf['low'].min()))) + (int(sdf['high'].max()*1.01)))/2)
    minrange=max(0,int(sdf['low'].min()*0.99))
    maxrange=int(sdf['high'].max()*1.01)
    # minrange = hl2 - 250
    # maxrange = hl2 + 250


    temp_df = tda_api.get_option_chain_df({'symbol':symbol, 'fromDate':date, 'toDate':date})

    plotdf = calc_max_pain_graph(temp_df, min_range=minrange, max_range=maxrange)

    fig = go.Figure()

    colormap = {'PUT':'red', 'CALL':'green'}

    for tn in plotdf['putCall'].unique():
        gdf = plotdf.loc[plotdf['putCall']==tn]

        bar_graph = go.Bar(
                x = gdf['und_price'],
                y = gdf['max_pain_value'],
                marker_color=colormap[tn],
                opacity=0.5,
                name=str(tn)
            )

        fig.add_trace(bar_graph)


    fig.update_layout({'xaxis':{'title': 'Underlying Price'},# 'range':[200,330]},
            'yaxis':{'title': 'Writer Loss'},
            'height':700,
            #margin={'l': 40, 'b': 40, 't': 10, 'r': 10, 'pad':4},
            # 'legend':{'x': 0, 'y': 1},
            'hovermode':'closest'})


    mp = plotdf.groupby('und_price').sum().sort_values(by='max_pain_value').reset_index().at[0,'und_price']

    fig.update_layout(legend_orientation="h")
    fig.update_layout(legend=dict(x=-.1, y=1.1))
    fig.update_layout(title='Max Pain for {sym} {exp} --> {mp}'.format(mp=mp, sym=symbol, exp=date), title_y = 0.95, title_x=0.5, title_font_size=30)

    return {'data': fig.data,
        'layout': fig.layout}


def graph_gamma(tda_api, symbol, date, volume_cuttoff=100, pclimit=1000):

    df_ = tda_api.get_option_chain_df({'symbol':symbol, 'fromDate':date, 'toDate':date})

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    plot_df = df_#.loc[(tdf['strikePrice']>=250) & (tdf['strikePrice']<=310)]# & (tdf['totalVolume']>500)]
    plot_df = plot_df.loc[plot_df['totalVolume']>=volume_cuttoff]
    plot_df2 = plot_df.loc[(plot_df['percentChange']<=pclimit) & (plot_df['percentChange']>=(-1*pclimit))]

    colormap = {'PUT':'red', 'CALL':'green'}
    colormap2 = {'PUT':'palevioletred', 'CALL':'lightgreen'}

#     plot_df['hlchange'] = ((plot_df['highPrice'] - plot_df['lowPrice']) / plot_df['lowPrice'])*100
#     plot_df['hcchange'] = ((plot_df['highPrice'] - plot_df['closePrice']) / plot_df['closePrice'])*100
#     plot_df['lcchange'] = ((plot_df['lowPrice'] - plot_df['closePrice']) / plot_df['closePrice'])*100

    for tn in plot_df['putCall'].unique():
        gdf = plot_df.loc[plot_df['putCall']==tn]

    #     Put metrics are fucked up, lets just look at total number of contracrts on the bar
        actual_line2 = go.Scatter(
            x=gdf['strikePrice'],
            y=gdf['gamma'],
            text=gdf['strikePrice'],
            mode='lines+markers',
            line=dict(color=colormap2[tn]),
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'},
                'symbol':3
            },
            name=str(tn)+' gamma',
        )
        fig.add_trace(actual_line2,secondary_y = True)

    for tn in plot_df['putCall'].unique():
        gdf = plot_df2.loc[plot_df2['putCall']==tn]

        actual_line = go.Scatter(
            x=gdf['strikePrice'],
            y=gdf['percentChange'],
            text=gdf['strikePrice'],
            mode='lines+markers',
            line=dict(color=colormap[tn]),
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='% Change {}'.format(tn),
        )

        fig.add_trace(actual_line,secondary_y = False)

#         actual_line = go.Scatter(
#             x=gdf['strikePrice'],
#             y=gdf['lcchange'],
#             text=gdf['strikePrice'],
#             mode='lines+markers',
#             line=dict(color=colormap[tn]),
#             opacity=0.3,
#             marker={
#                 'size': 5,
#                 'line': {'width': 0.5, 'color': 'white'}
#             },
#             name='% Change Low-Close {}'.format(tn),
#         )

#         fig.add_trace(actual_line,secondary_y = False)

#         actual_line = go.Scatter(
#             x=gdf['strikePrice'],
#             y=gdf['hlchange'],
#             text=gdf['strikePrice'],
#             mode='lines+markers',
#             line=dict(color=colormap[tn]),
#             visible='legendonly',
#             opacity=0.3,
#             marker={
#                 'size': 5,
#                 'line': {'width': 0.5, 'color': 'white'}
#             },
#             name='% Change High-Low {}'.format(tn),
#         )

#         fig.add_trace(actual_line,secondary_y = False)

#         actual_line = go.Scatter(
#             x=gdf['strikePrice'],
#             y=gdf['hcchange'],
#             text=gdf['strikePrice'],
#             mode='lines+markers',
#             line=dict(color=colormap[tn]),
#             visible='legendonly',
#             opacity=0.3,
#             marker={
#                 'size': 5,
#                 'line': {'width': 0.5, 'color': 'white'}
#             },
#             name='% Change High-Close {}'.format(tn),
#         )

#         fig.add_trace(actual_line,secondary_y = False)

    #     bar_graph = go.Bar(
    #                 x = gdf['strikePrice'],
    #                 y = gdf['numberOfContracts'],
    #                 marker_color=colormap[tn],
    #                 opacity=0.3,
    #                 name='Volume {}'.format(tn)
    #             )

    #     fig.add_trace(bar_graph,secondary_y = True)

    fig.update_layout({'xaxis':{'title': 'Strike Price'},
            'yaxis':{'title': '% Change (%)'},
            'yaxis2' : {'title':'Gamma'},
            'height':700,
            #margin={'l': 40, 'b': 40, 't': 10, 'r': 10, 'pad':4},
            # 'legend':{'x': 0, 'y': 1},
            'hovermode':'closest'})

    fig.update_layout(title='% Change and Gamma for {sym} {date}'.format(sym=symbol, date=date), title_y = 1, title_x=0.5, title_font_size=30)
    fig.update_layout(legend_orientation="h")
    fig.update_layout(legend=dict(x=-.1, y=1.1))

    return {'data': fig.data,
        'layout': fig.layout}

def plot_spx_resistance(df, resistance_lvls, plot_res=True):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    candles = go.Candlestick(x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=df.at[0,'symbol'])

    fig.add_trace(candles,secondary_y = False)

    colors = ['red', 'orange', 'gold', 'lawngreen', 'lightseagreen', 'royalblue',
          'blueviolet']
    clr_idx = 0

    xmin=df['date'].min()
    xmax=df['date'].max()

    if (plot_res==True) or (plot_res=='True'):
        for res in resistance_lvls:
            if (res < df['low'].min()-50) or (res > df['high'].max()+50):
                continue
            clr = colors[clr_idx%7]
            fig.add_shape(
                # Line Horizontal
                    type="line",
                    x0=xmin,
                    y0=res,
                    x1=xmax,
                    y1=res,
                    opacity=0.5,
                    name=str(res),
                    line=dict(
                        color=clr,
                        width=2.5,
                    ),
            )
            clr_idx += 1

    yaxis_min = df.loc[df['low']>0]['low'].min()*0.997
    yaxis_max = df['high'].max()*1.003

    fig.update_layout(legend_orientation="h", height=1000, xaxis={'title':'date', 'type':'category'}, yaxis={'title':'Price ($)', 'range':[yaxis_min,yaxis_max]})
    fig.update_layout(legend=dict(x=-.1, y=1.2))

    return {'data': fig.data,
        'layout': fig.layout}

def get_resistance_lvls(tda_api, d1, d2, sym='$SPX.X', frequencyType='minute', frequency=1, nn=5, plot_res=True):

    df = get_historical_stock_data(tda_api, sym, d1, d2, frequencyType=frequencyType, frequency=frequency, ext_hours='false')


    dflows = df.loc[df['change']<=0]
    dfhighs = df.loc[df['change']>0]

    nn = 5

    dlist = []
    for lows in [3, 5, 15]:
        dflows['lows'] = dflows['low'].rolling(lows).min()
        vals = dflows.groupby('lows').count().sort_values(by='open',ascending=False).reset_index()[0:nn]['lows'].values
        cnt = dflows.groupby('lows').count().sort_values(by='open',ascending=False).reset_index()[0:nn]['open'].values
        for i in range(nn):
            d = {'vals':vals[i], 'cnt':cnt[i]}
            dlist.append(d)
    low_df = pd.DataFrame(dlist)

    dlist = []
    for highs in [3, 5, 15]:
        dfhighs['highs'] = dfhighs['high'].rolling(highs).max()
        vals = dfhighs.groupby('highs').count().sort_values(by='open',ascending=False).reset_index()[0:nn]['highs'].values
        cnt = dfhighs.groupby('highs').count().sort_values(by='open',ascending=False).reset_index()[0:nn]['open'].values
        for i in range(nn):
            d = {'vals':vals[i], 'cnt':cnt[i]}
            dlist.append(d)
    high_df = pd.DataFrame(dlist)


    round_num = df['change'].apply(lambda x: abs(x)).mean() + (2*df['change'].std())
    res_df = pd.concat([high_df,low_df],ignore_index=True)
    res_df['round_val'] = res_df['vals'].apply(lambda x: math.floor(x/round_num)*round_num)
    res_vals = res_df.sort_values(by='vals').drop_duplicates(subset=['round_val'])['vals'].values

    spxdf = get_historical_stock_data(tda_api, sym, d1, d2,frequencyType='minute', frequency=1, ext_hours='false')

    g = plot_spx_resistance(spxdf, res_vals, plot_res=plot_res)

    return g

def pull_all_symbols(tda_api):

    ## Special cases
        # $SPX.X
        # $NDX.X
        # $RUT.X
        # $VIX.X
    data_pull = tda_api.get_symbols('[A-Z]*')
    df = pd.DataFrame(data_pull)
    df2 = df.T
    df3 = df2.loc[df2['exchange']!='Pink Sheet']

    syms = list(df3['symbol'].unique())
    for s in ['$NDX.X', '$RUT.X', '$VIX.X', '$SPX.X']:
        syms.append(s)

    return syms

def calc_strike_diff(k, S, putCall, round_num=1):

    r = None

    if putCall == 'PUT':
        r = S - k
        r = round(r/round_num)*round_num
    if putCall == 'CALL':
        r = k-S
        r = round(r/round_num)*round_num
    return r

def get_even_option_strike(input_df):

    df2 = input_df.copy()
    df2 = df2.pivot(index='strikePrice', columns='putCall', values='ask').dropna().reset_index()
    df2['pc_diff'] = abs(df2['CALL'] - df2['PUT'])
    df2['underlyingPrice'] = round(input_df.at[0,'underlyingPrice'], 2)
    df2 = df2.loc[(df2['CALL']>0) & (df2['PUT']>0)]
    df2 = df2.sort_values(by='pc_diff').reset_index(drop=True)

    return df2.at[0,'strikePrice']

def assemble_pc_df(input_df, lim=200, round_num=5):

    df = input_df.copy()
#     print(df.shape)
    even_ask_strike = get_even_option_strike(df)
#     return even_ask_strike

    df['strike_diff'] = df.apply(lambda row: calc_strike_diff(row['strikePrice'], row['underlyingPrice'], row['putCall'], round_num=round_num),axis=1)
    df = df.loc[(df['strike_diff']>=(lim*-1)) & (df['strike_diff']<=lim)].reset_index(drop=True)

#     print(df.shape)
    df2 = df.sort_values(by='totalVolume',ascending=False)
    df2 = df2[['putCall', 'strikePrice', 'mark', 'strike_diff']].copy()
    df2 = df2.drop_duplicates(subset=['putCall','strike_diff'])
#     print(df2.shape)


    df2 = df2.pivot(index='strike_diff', columns='putCall', values='mark').dropna().reset_index()
    df2['pc_diff'] = df2['CALL'] - df2['PUT']
    df2['underlyingPrice'] = round(df.at[0,'underlyingPrice']/round_num)*round_num
    df2['even_ask_strike'] = even_ask_strike

    return df2

def find_closest_priced_option(input_df):

    df = input_df.copy()

    und_price = df.at[0,'underlyingPrice']

    closest_call_list = []
    closest_put_list = []

    for i in range(df.shape[0]):

        current_call_price = df.at[i, 'CALL']
        current_put_price = df.at[i, 'PUT']

        temp_df = df.copy()
        temp_df['closest_call'] = abs(temp_df['CALL'] - current_put_price)
        temp_df['closest_put'] = abs(temp_df['PUT'] - current_call_price)

        closest_put = temp_df.sort_values(by='closest_put').reset_index(drop=True).at[0,'strike_diff']
        closest_call = temp_df.sort_values(by='closest_call').reset_index(drop=True).at[0,'strike_diff']

        closest_call_list.append(closest_call)
        closest_put_list.append(closest_put)

    df['closest_call_to_put_price'] = closest_call_list
    df['closest_put_to_call_price'] = closest_put_list

    df['strike_where_put_equals_this_strike_call'] = und_price - df['closest_put_to_call_price']
    df['strike_where_call_equals_this_strike_put'] = df['closest_call_to_put_price'] + und_price

    df['closest_diff'] = df['strike_where_put_equals_this_strike_call'] - df['strike_where_call_equals_this_strike_put']

    return df


def pull_spx_options(tda_api, d1, d2, sym='$SPX.X', lim=200, round_num=5):

    spx_df = tda_api.get_option_chain_df({'symbol':sym, 'fromDate':d1,'toDate':d2, 'range':'ALL'})

    final_spx_df = pd.DataFrame()

    for exp in spx_df['expirationDate'].unique():
        temp_df = spx_df.loc[spx_df['expirationDate']==exp].reset_index(drop=True)
        temp_df2 = assemble_pc_df(temp_df, lim=lim, round_num=round_num)
        temp_df3 = find_closest_priced_option(temp_df2)
        temp_df3['expirationDate'] = exp
        final_spx_df = pd.concat([final_spx_df, temp_df3],ignore_index=True)

    final_spx_df['expirationDate'] = pd.to_datetime(final_spx_df['expirationDate'], unit='ms')
    final_spx_df['expirationDate'] = final_spx_df['expirationDate'].apply(lambda x: str(x)[0:10])

    final_spx_df['put_strike'] = final_spx_df['underlyingPrice'] - final_spx_df['strike_diff']
    final_spx_df['call_strike'] =  final_spx_df['underlyingPrice'] + final_spx_df['strike_diff']
    final_spx_df['strike_strings'] = final_spx_df.apply(lambda row: combine_strikes(row['put_strike'], row['call_strike']),axis=1)

    return final_spx_df

def combine_strikes(put_strike, call_strike):

    put_strike = str(put_strike)
    call_strike = str(call_strike)
    new_string = 'PUT STRIKE: {p} -- CALL STRIKE: {c}'.format(p=put_strike, c=call_strike)
    return new_string

def plot_SPX_options(input_df, sym, req_strikes=None):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    df = input_df.copy()
    df = df.sort_values(by=['expirationDate', 'strike_diff'])

    if req_strikes!=None:
        df = df.loc[df['expirationDate'].isin(req_strikes)]

    for d in df['expirationDate'].unique():

        try:
            gdf = df.loc[df['expirationDate']==d].reset_index(drop=True)
        except:
            continue

        actual_line = go.Scatter(
        x=gdf['strike_diff'],
        y=gdf['pc_diff'],
        text=gdf['strike_strings'],
        mode='lines+markers',
    #         line=dict(color='lime'),
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=d,
            legendgroup=d,
        )

        fig.add_trace(actual_line,secondary_y = False)


    fig.update_layout(legend_orientation="h", height=1000, xaxis={'title':'$ Out of the Money'}, yaxis={'title':'Call Price - Put Price'}, title={'text':'{} Put Call price differences over equivelant strikes'.format(sym)})
    fig.update_layout(legend=dict(x=-.1, y=1.2))

    return {'data': fig.data,
        'layout': fig.layout}

def plot_SPX_options_3d(input_df):

    df = input_df.copy()

    fig = go.Figure()

    x = df['strike_diff']
    y = df['expirationDate']
    z = df['pc_diff']

    data=go.Scatter3d(
        x=x,
        y=y,
        z=z,
        text=df['closest_diff'],
        mode='markers',
        marker=dict(
            size=10,
            color=z,
            opacity=0.8,
            colorscale='Viridis',
            symbol='circle'
        )
    )

    fig.add_trace(data)

    # tight layout
    fig.update_layout(scene = dict(
                        xaxis_title='$ Out of the Money',
                        yaxis_title='Expiration Date',
                        zaxis_title='call Price - Put Price'),
                        height=1000,margin=dict(l=0, r=0, b=0, t=0), showlegend=True,
                     )

    return {'data': fig.data,
        'layout': fig.layout}

def graph_options_oi(tda_api, symbol, start_date, end_date, stack_bars=False):

    df_ = tda_api.get_option_chain_df({'symbol':symbol, 'fromDate':start_date, 'toDate':end_date})
    df_ = df_.groupby(['strikePrice','putCall'],as_index=False).sum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colormap = {'PUT':'red', 'CALL':'green'}

    for tn in df_['putCall'].unique():
        gdf = df_.loc[df_['putCall']==tn]


        bar_graph = go.Bar(
                x = gdf['strikePrice'],
                y = gdf['openInterest'],
                marker_color=colormap[tn],
                opacity=0.5,
                name='Open Interest {}'.format(tn)
            )

        fig.add_trace(bar_graph,secondary_y = False)

        bar_graph = go.Bar(
                x = gdf['strikePrice'],
                y = gdf['totalVolume'],
                marker_color=colormap[tn],
                opacity=0.3,
                name='Volume {}'.format(tn)
            )

        fig.add_trace(bar_graph,secondary_y = True)

    fig.update_layout({'xaxis':{'title': 'Strike Price'},# 'range':[200,330]},
            'yaxis':{'title': 'Open Interest'},
            'yaxis2' : {'title':'Volume'},
            'height':700,
            #margin={'l': 40, 'b': 40, 't': 10, 'r': 10, 'pad':4},
            # 'legend':{'x': 0, 'y': 1},
            'hovermode':'closest'})

    if stack_bars == 'True':
        fig.update_layout(barmode='stack')

    fig.update_layout(title='Options for {sym} from ExpDate {d1} to {d2}'.format(sym=symbol, d1=start_date, d2=end_date), title_y = 0.95, title_x=0.5, title_font_size=30)
    fig.update_layout(legend_orientation="h")
    fig.update_layout(legend=dict(x=-.1, y=1.1))
    # fig.update_layout(barmode='stack')

    return {'data': fig.data,
        'layout': fig.layout}

def pull_dix():

    r = requests.get('https://squeezemetrics.com/monitor/static/DIX.csv?_t={}'.format(round(time.time()*1000)))

    cols = ['date','price','dix','gex']
    r2 = r.text[20:].replace('\r\n', ',').split(',')
    r2 = r2[:-1]
    temp_idx = cols*int(len(r2)/4)
    ddf = pd.DataFrame(r2, index=temp_idx)


    idx_str = ''
    for i in range(int(len(r2)/4)):
        idx_str += '{i},{i},{i},{i},'.format(i=i)
    idx_str = idx_str[:-1]
    r2_idx = idx_str.split(',')
    r2_idx = [int(i) for i in r2_idx]


    ddf2 = ddf.reset_index().pivot(columns='index', index=r2_idx)
    ddf2.columns = ddf2.columns.droplevel()
    for col in ['dix', 'gex', 'price']:
        ddf2[col] = ddf2[col].astype('float')

    return ddf2

from sklearn.preprocessing import MinMaxScaler

def plot_dix(min_date, dix_rolling=None, gex_rolling=None):

    df = pull_dix()

    df = df.loc[df['date']>=min_date].reset_index(drop=True)

    mms = MinMaxScaler((df['gex'].min(), df['gex'].max()))

    if gex_rolling != None:
        df['gex'] = df['gex'].rolling(gex_rolling).mean()
    if dix_rolling != None:
        df['dix'] = df['dix'].rolling(dix_rolling).mean()

    df['dix_scaled'] = mms.fit_transform(df['dix'].values.reshape(-1,1))

    fig = make_subplots(specs=[[{"secondary_y": True}]])



    fig.add_trace(go.Scatter(x=df['date'], y=df['price'],
                             line=dict(color="#63e686", width=5),
                             name='SPX'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df['date'], y=df['dix_scaled'],
                             line=dict(color="#2eb9ff", width=5),
                             hovertext=df['dix'],
                             name='DIX'),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df['gex'],
                             line=dict(color="#ffb62e", width=5),
                             name='GEX'),
                  secondary_y=True)

    fig.update_layout(legend_orientation="h", height=1000, title={'text':'Daily DIX & GEX'},xaxis={'title':'date'}, yaxis={'title':'SPX ($)'}, yaxis2={'title':'GEX'})
    fig.update_layout(legend=dict(x=-.1, y=1.2))

    return {'data': fig.data,
        'layout': fig.layout}


################## VIX CALC CODE ##################
def find_closest_midpoints(input_df):

    df = input_df.copy()

    df1 = df[['putCall', 'expirationDateStr', 'strikePrice', 'mark', 'bid', 'ask']]

    put_df = df1.loc[df1['putCall']=='PUT'][['expirationDateStr', 'strikePrice', 'mark', 'bid', 'ask']]
    call_df = df1.loc[df1['putCall']=='CALL'][['expirationDateStr', 'strikePrice', 'mark', 'bid', 'ask']]

    put_df.columns = ['expirationDateStr', 'strikePrice', 'put_mark', 'put_bid', 'put_ask']
    call_df.columns = ['expirationDateStr', 'strikePrice', 'call_mark', 'call_bid', 'call_ask']

    df2 = put_df.merge(call_df, on=['expirationDateStr', 'strikePrice'])

    closest_df = df2.copy()
    bidask_df = df2[['expirationDateStr', 'strikePrice', 'put_bid', 'put_ask', 'call_bid', 'call_ask']].copy()

    closest_df['diff'] = closest_df['put_mark'] - df2['call_mark']
    closest_df['diff'] = closest_df['diff'].apply(lambda x: abs(x))
    closest_df = closest_df.loc[(closest_df['put_mark']>0) & (closest_df['call_mark']>0)]
    closest_df = closest_df.sort_values(by='diff')
    closest_df = closest_df.drop_duplicates(subset='expirationDateStr')

    df3 = closest_df[['expirationDateStr', 'strikePrice', 'diff']]
    df3.columns = ['expirationDateStr', 'closest_strikes', 'closest_diff']

    df3 = df3.sort_values(by='expirationDateStr')

    rdf = input_df.merge(df3, on='expirationDateStr',how='left')

    rdf = rdf.merge(bidask_df, on=['expirationDateStr', 'strikePrice'], how='left')

    return rdf

def filter_bids(input_df):

    df = input_df.copy()

    # First, filter out based on the midpoints
    df = df.sort_values(by=['expirationDateStr','putCall', 'strikePrice'])

    put_df = df.loc[(df['putCall']=='PUT')]
    call_df = df.loc[(df['putCall']=='CALL')]
    put_df = put_df.sort_values(by='strikePrice',ascending=False)
    call_df = call_df.sort_values(by='strikePrice')
    df = pd.concat([put_df,call_df])

    df['lag_bid'] = df.groupby(['expirationDateStr','putCall'])['bid'].shift()
    df['bidsum1'] = df['bid'] + df['lag_bid']

    put_df = df.loc[(df['putCall']=='PUT') & (df['strikePrice']<=df['closest_strikes'])]
    call_df = df.loc[(df['putCall']=='CALL') & (df['strikePrice']>=df['closest_strikes'])]

    put_df = put_df.sort_values(by='strikePrice',ascending=False)
    call_df = call_df.sort_values(by='strikePrice')

    df = pd.concat([put_df,call_df])

    df['bidprod'] = df.groupby(['expirationDateStr','putCall'])['bidsum1'].cumprod()

    # Exclude options with bids = 0
    df2 = df.loc[df['bid']>0]
    # Exclude all options after 2 0 bids in a row
    df2 = df2.loc[df2['bidprod']!=0]

    ## Now apply prices to each options - special case for those that are == spot price

    rdf1 = df2.loc[df2['strikePrice']!=df2['closest_strikes']]
    rdf11 = rdf1.loc[rdf1['putCall']=='PUT']
    rdf12 = rdf1.loc[rdf1['putCall']=='CALL']

    rdf2 = df2.loc[df2['strikePrice']==df2['closest_strikes']]


    rdf11['Q'] = (rdf11['put_bid'] + rdf11['put_ask'] ) / 2
    rdf12['Q'] = (rdf12['call_bid'] + rdf12['call_ask'] ) / 2

    rdf2['Q'] = (rdf2['call_bid'] + rdf2['call_ask'] + rdf2['put_bid'] + rdf2['put_ask'] ) / 4

    rdf = pd.concat([rdf11, rdf12, rdf2],ignore_index=True)

    # Define Minutes to exp
    rdf['mte_calc'] = rdf['dte_calc']*24*60
    rdf['T'] = rdf['mte_calc']/ 525600 # minutes in a year

    return rdf

def calc_VIX(input_df, r=0.00067):

    df = input_df.copy()

    df = df.loc[df['strikePrice']==df['closest_strikes']].sort_values(by=['expirationDateStr', 'putCall']).reset_index(drop=True)

    start_exp = df.at[0,'expirationDateStr']
    end_exp = df.at[df.shape[0]-1, 'expirationDateStr']

    df = df.loc[df['expirationDateStr'].isin([start_exp, end_exp])].reset_index(drop=True)
    df['baa'] = (df['bid'] + df['ask'] ) / 2

    # Now calc it!
    p1 = df.at[1,'baa']
    c1 = df.at[0,'baa']

    p2 = df.at[3,'baa']
    c2 = df.at[2,'baa']

    T1 = df.at[1,'T']
    T2 = df.at[3,'T']

    k1 = df.at[1,'closest_strikes']
    k2 = df.at[3,'closest_strikes']

    F1 = k1 + (math.exp(r*T1) * (c1-p1))
    F2 = k2 + (math.exp(r*T2) * (c2-p2))

    ## Now we need to calc the variance
    df2 = input_df.copy()
    df2 = df2.loc[df2['expirationDateStr'].isin([start_exp, end_exp])].sort_values(by='strikePrice')
    # To get the delta X, forward and lag variables
    df2['lag_strike'] = df2.groupby(['expirationDateStr','putCall'])['strikePrice'].shift()
    df2['forward_strike'] = df2.groupby(['expirationDateStr','putCall'])['strikePrice'].shift(-1)

    df2['deltaX'] = abs(df2['forward_strike'] - df2['lag_strike']) / 2

    df2 = df2.sort_values(by=['expirationDateStr','putCall', 'strikePrice']).reset_index(drop=True)
    df2['deltaX'] = df2['deltaX'].interpolate(method='pad') # Fill in nans with "closest adjectent strike" which...will probably be whatever it was near it
    df2['deltaX'] = df2['deltaX'].bfill()

    # Make columns to help with sum term
    df2['strikes_sq'] = df2['strikePrice'].apply(lambda x: x**2)
    df2['sth1'] = (df2['deltaX'] / df2['strikes_sq']) * df2['Q']
    df2['sth1'] = df2['sth1'] * math.exp(r * T1)
    df2['sth2'] = df2['sth1'] * math.exp(r * T2)

    ndf = df2.loc[df2['expirationDateStr']==start_exp]
    fdf = df2.loc[df2['expirationDateStr']==end_exp]

    st1 = ndf['sth1'].sum()
    st2 = fdf['sth2'].sum()

    nt_k0 = ndf['closest_strikes'].values[0]
    ft_k0 = fdf['closest_strikes'].values[0]

    fterm1 = (((F1 / nt_k0) - 1 ) ** 2) * (1/T1)
    fterm2 = (((F2 / ft_k0) - 1 ) ** 2) * (1/T2)

    sterm1 = (2/T1) * st1
    sterm2 = (2/T2) * st2

    var1 = sterm1 - fterm1
    var2 = sterm2 - fterm2


    # df2['sum_term1'] = st1# Sum term - deltaX, X, Qx
    # df2['sum_term2'] = st2

    # df2['fterm1'] = (F1 / df2['closest_strikes']) - 1
    # df2['fterm1'] = df2['fterm1'].apply(lambda x: x**2)
    # df2['fterm1'] = df2['fterm1'] * (1/T1)

    # df2['fterm2'] = (F2 / df2['closest_strikes']) - 1
    # df2['fterm2'] = df2['fterm2'].apply(lambda x: x**2)
    # df2['fterm2'] = df2['fterm2'] * (1/T2)

    # df2['sterm1'] = (2/T1) * df2['sum_term1'] * math.exp(r * T1)
    # df2['sterm2'] = (2/T2) * df2['sum_term2'] * math.exp(r * T2)

    # df2['var1'] = df2['sterm1'] - df2['fterm1']
    # df2['var2'] = df2['sterm2'] - df2['fterm2']

    # var1 = df2.loc[df2['expirationDateStr']==start_exp]['var1'].values[0]
    # var2 = df2.loc[df2['expirationDateStr']==end_exp]['var2'].values[0]

    # Just put the last term here...too tired to do it right now
    N1 = df.at[0,'mte_calc']
    N2 = df.at[2, 'mte_calc']

    N30 = 43200 # Number of minutes in 30 days
    N365 = 525600 # Number of minutes in a year
    # print(var1, var2)


    vix1 = (T1 * var1 * ( (N2-N30) / (N2-N1)))
    vix2 = (T2 * var2 * ( (N30-N1) / (N2-N1)))
    vix3 = (vix1 + vix2) * (N365/N30)

    VIX = 100 * (vix3**0.5)

    # print(vix1, vix2)
    # print(VIX)

    return VIX

def get_VIX(tda_api, neardate=None, fardate=None):

    if neardate==None:
        neardate = datetime.datetime.now() + datetime.timedelta(days=24)
        dow = neardate.weekday()
        while dow != 4:
            neardate = neardate + datetime.timedelta(days=1)
            dow = neardate.weekday()

    if fardate==None:
        if type(neardate)==str:
            neardate = datetime.datetime.strptime(neardate, '%Y-%m-%d')
        fardate = neardate + datetime.timedelta(days=7)

    neardate_str = str(neardate)[0:10]
    fardate_str = str(fardate)[0:10]

    df = tda_api.get_option_chain_df({'symbol':'$SPX.X', 'fromDate':neardate_str,'toDate':fardate_str})

    # Make sure we only include weeklys and not the AM expirations
    df['spx_symbol'] = df['symbol'].apply(lambda x: x[0:4])
    df = df.loc[df['expirationDateStr'].isin([neardate_str, fardate_str])]
    df = df.loc[df['spx_symbol']=='SPXW']
    df = df.reset_index(drop=True)

    df2 = find_closest_midpoints(df)
    df3 = filter_bids(df2)
    df4 = calc_VIX(df3)

    return df4

def dash_vix_values(tda_api):


    neardate = datetime.datetime.now() + datetime.timedelta(days=24)
    dow = neardate.weekday()
    while dow != 4:
        neardate = neardate + datetime.timedelta(days=1)
        dow = neardate.weekday()

    if type(neardate)==str:
        neardate = datetime.datetime.strptime(neardate, '%Y-%m-%d')
    fardate = neardate + datetime.timedelta(days=7)

    neardate0 = neardate + datetime.timedelta(days=-7)
    fardate0 = fardate + datetime.timedelta(days=-7)

    neardate2 = neardate + datetime.timedelta(days=7)
    fardate2 = fardate + datetime.timedelta(days=7)

    neardate_str = str(neardate)[0:10]
    fardate_str = str(fardate)[0:10]

    neardate_str0 = str(neardate0)[0:10]
    fardate_str0 = str(fardate0)[0:10]

    neardate_str2 = str(neardate2)[0:10]
    fardate_str2 = str(fardate2)[0:10]

    vix0 = round(get_VIX(tda_api, neardate=neardate0, fardate=fardate0), 3)
    vix1 = round(get_VIX(tda_api, neardate=neardate, fardate=fardate), 3)
    vix2 = round(get_VIX(tda_api, neardate=neardate2, fardate=fardate2), 3)

    r0 = [neardate_str0, fardate_str0, vix0]
    r1 = [neardate_str, fardate_str, vix1]
    r2 = [neardate_str2, fardate_str2, vix2]

    rdict = [r0, r1, r2]

    return rdict


################## END VIX CALC CODE ##################


################## START VOLATILITY TERM-STRUCTURE CODE ##################
def pullCalendarIVdf(tda_api, sym, start_dt, end_dt, filter_fd=False):

    df = tda_api.get_option_chain_df({'symbol':sym, 'fromDate':start_dt,'toDate':end_dt, 'range':'OTM'})
    df['volatility'] = df['volatility'].apply(lambda x: pd.to_numeric(x, errors='coerce'))


    merge_cols = ['putCall', 'strikePrice', 'volatility','bid', 'ask','last',
                'mark', 'delta', 'gamma', 'theta', 'vega', 'rho','expirationDateStr']

    df = df.dropna(subset=['volatility'])
    # df = df.loc[df['bid']>0]
    df = df.loc[df['bid']>0]
    # df = df.loc[df['openInterest']>0]
    # df = df.loc[df['totalVolume']>0]

    # Filter if required
    if (filter_fd == 'True') or (filter_fd==True):
        df = df.loc[df['expirationDateStr']>df['expirationDateStr'].min()].reset_index(drop=True)

    ## Initialize dataframe to merge on with first date
    first_dt = df['expirationDateStr'].unique()[0]
    mdf = df.loc[df['expirationDateStr']==first_dt][merge_cols]

    col_list1 = []
    for col in merge_cols:
        if col in ['putCall', 'strikePrice']:
            col_list1.append(col)
        else:
            colname1 = col+'_{}'.format(first_dt)
            col_list1.append(colname1)

    mdf.columns = col_list1


    for i in range(1, df['expirationDateStr'].nunique()):

        curr_date = df['expirationDateStr'].unique()[i]

        tempdf1 = df.loc[df['expirationDateStr']==curr_date][merge_cols]

        col_list1 = []
        for col in merge_cols:
            if col in ['putCall', 'strikePrice']:
                col_list1.append(col)
            else:
                colname1 = col+'_{}'.format(curr_date)
                col_list1.append(colname1)

        tempdf1.columns = col_list1

        mdf = mdf.merge(tempdf1, on=['putCall','strikePrice'], how='left')

    return mdf

def graphCalendarIV(sym, input_df, pc):

    mdf = input_df.copy()

    col_list = []
    cdict = {'CALL':'Greens', 'PUT':'Reds'}

    for col in list(mdf):
        if 'volatility' in col:
            col_list.append(col)

    fig = go.Figure()

    tpdf = mdf.copy()
    tpdf = tpdf.loc[tpdf['putCall']==pc]

    for col in col_list:
        tpdf[col] = tpdf[col].astype('float',errors='ignore')
        tpdf.dropna(subset=[col],inplace=True)

    dim_list = []
    grange = [tpdf[col_list].dropna().values.min(), tpdf[col_list].dropna().values.max()]

    for col in col_list:
        tdict = {'range':grange, 'label':col, 'values':tpdf[col]}
        dim_list.append(tdict)

    data=go.Parcoords(
        line = dict(color = tpdf['strikePrice'],
                colorscale = cdict[pc],
                    cmax=tpdf['strikePrice'].max(),
                    cmin=tpdf['strikePrice'].min(),
                showscale = True),
            dimensions = dim_list
        )
    fig.add_trace(data)

    fig.update_layout(title='Volatility Term Structure for {sym} {pc}'.format(sym=sym,pc=pc), title_y = 0.95, title_x=0.5, title_font_size=30)

    return {'data': fig.data,
        'layout': fig.layout}
################## END VOLATILITY TERM-STRUCTURE CODE ##################

#### START short interest CODE ####
def parse_finra_df(input_r, num_cols=['shortParQuantity', 'shortExemptParQuantity', 'totalParQuantity']):

    dlist = input_r.text.split('\n')

    dlist = [d.replace('"','') for d in dlist]

    dlist2 = dlist[1:]
    dlist2 = [d.split(',') for d in dlist2]


    df = pd.DataFrame(data=dlist2,columns=dlist[0].split(','))

    for col in num_cols:
        df[col] = df[col].astype('int',errors='ignore').fillna(np.nan)
        df = df.dropna(subset=[col])
        df[col] = df[col].astype('int',errors='ignore').fillna(np.nan)

    return df

def pull_finra_dataset(group_name, dataset_name, body=None):

    headers = {'Authorization': 'Basic c2xhY2tlcjI5ODpXdGZjb3B0ZXIxOTgya2s='}
    base_finra_url = 'https://api.finra.org'
    data_req_str = base_finra_url+'/data/group/{group_name}/name/{dataset_name}'.format(group_name=group_name,dataset_name=dataset_name)

    r = requests.post(data_req_str,data=json.dumps(body), headers=headers)

    df = parse_finra_df(r)

    return df

def merge_price_and_finra(tda_api, finra_df, sym):

    ddf = get_historical_stock_data(tda_api, sym, finra_df['tradeReportDate'].min(), finra_df['tradeReportDate'].max(), frequencyType='daily')
    ddf['tradeReportDate'] = ddf['date'].apply(lambda x: str(x)[0:10])

    fdf2 = finra_df.groupby('tradeReportDate',as_index=False).sum()
    fdf3 = ddf.merge(fdf2, on='tradeReportDate')

    return fdf3

def fix_ndt_date(dt):

    dt2 = dt[-4:] + '-'+ dt[0:2] + '-' + dt[3:5]
    return dt2

def pull_nqt_si(curr_sym):

    body = {'method': "BL_ShortInterest.SearchShortInterests", 'params': [curr_sym], 'version': "1.1"}

    headers = {'Host': 'www.nasdaqtrader.com',
    'Connection': 'keep-alive',
    'Content-Length': '95',
    'Content-type': 'application/json',
    'Accept': '*/*',
    'Origin': 'http://www.nasdaqtrader.com',
    'Referer': 'http://www.nasdaqtrader.com/Trader.aspx?id=ShortInterest'}

    r = requests.post('http://www.nasdaqtrader.com/RPCHandler.axd', data=json.dumps(body),headers=headers)

    ## code to parse this god awful wall of text
    ndt_colnames = ['Settlement Date',	'Short Interest', 'Percent Change',	'Average Daily Share Volume','Days to Cover']
    r2 = r.text
    r2 = r2.replace('</td>\\r\\n\\t</tr>\\r\\n\\t<tr class=\\"genTablealt\\">\\r\\n\\t\\t<td>',';').replace('</td>\\r\\n\\t</tr>\\r\\n\\t<tr>\\r\\n\\t\\t<td>',';').replace('</td>\\r\\n\\t\\t<td>',';')
    r3 = r2.split(';')

    r4 = [r3[0][-10:]] + r3[1:]

    # Fix the last entry which has artifcats from http
    final_idx = r4[len(r4)-1].find('<')
    r4[len(r4)-1] = r4[len(r4)-1][0:final_idx]

    # reshape so we can put it in a dataframe
    rdata = np.array(r4).reshape(-1,5)
    ndt_df = pd.DataFrame(data=rdata, columns=ndt_colnames)
    ndt_df['tradeReportDate'] = ndt_df['Settlement Date'].apply(lambda x: fix_ndt_date(x))

    return ndt_df

def make_si_gdf(tda_api, curr_sym):

    df = pull_finra_dataset('OTCMarket', 'REGSHODAILY', body={"compareFilters": [ {  "compareType": "equal", "fieldName": "securitiesInformationProcessorSymbolIdentifier", "fieldValue" : curr_sym}]})
    df2 = merge_price_and_finra(tda_api, df, curr_sym)

    try:
        ndt_df = pull_nqt_si(curr_sym)
        df3 = df2.merge(ndt_df, on='tradeReportDate',how='left')
    except:
        df3=df2

    return df3

def plot_short_candles(df):
#     https://plotly.com/python/candlestick-charts/

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    candles = go.Candlestick(x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=df.at[0,'symbol'])

    fig.add_trace(candles,secondary_y = False)

    bar_graph = go.Bar(
            x = df['date'],
            y = df['volume'],
            marker_color='blue',
            opacity=0.5,
            name='tda Volume',
            visible='legendonly'
        )
    fig.add_trace(bar_graph,secondary_y = True)

    bar_graph = go.Bar(
            x = df['date'],
            y = df['totalParQuantity'],
            marker_color='darkblue',
            opacity=0.5,
            name='finra Volume',
            visible='legendonly'
        )
    fig.add_trace(bar_graph,secondary_y = True)

    bar_graph = go.Bar(
            x = df['date'],
            y = df['shortParQuantity'],
            marker_color='orange',
            opacity=0.5,
            name='short Volume'
        )
    fig.add_trace(bar_graph,secondary_y = True)

    try: # Not all symbols have confirmed short data
        bar_graph = go.Bar(
                x = df['date'],
                y = df['Short Interest'],
                marker_color='red',
                opacity=0.8,
                name='confirmed short interest'
            )
        fig.add_trace(bar_graph,secondary_y = True)
    except:
        pass

    fig.update_layout(legend_orientation="h", barmode='overlay', height=800)
    fig.update_layout(legend=dict(x=-.1, y=1.1))

    return {'data': fig.data,
        'layout': fig.layout}

#### END short interest CODE ####




# Get default day for the date picker

curr_unix = time.time()
curr_dt = ciso8601.parse_datetime(datetime.datetime.utcfromtimestamp(curr_unix).strftime('%Y-%m-%d'))
dayofweek = curr_dt.weekday()
while dayofweek != 4:
    curr_unix += 86400
    curr_dt = ciso8601.parse_datetime(datetime.datetime.utcfromtimestamp(curr_unix).strftime('%Y-%m-%d'))
    dayofweek = curr_dt.weekday()
#     print(dayofweek)

default_date = datetime.datetime.utcfromtimestamp(curr_unix).strftime('%Y-%m-%d')
min_date = datetime.datetime.utcfromtimestamp(time.time()-86400)

t1 = TDAmeritrade()

all_symbols = pull_all_symbols(t1)

################## DASH CODE ##################

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([

    html.H1(
        children="Lanier's Short Interest Dashboard",
        style={
            'textAlign': 'center'
        }
    ),

    html.Div([
        dcc.Markdown(d("""
                Choose Symbol
            """)),
        dcc.Dropdown(
            id='symbol-dropdown',
            options=[{'label': s, 'value': s} for s in all_symbols],
            value='TSLA'
        ),
    ]),

    # html.Div([
    #     dcc.Markdown(d("""
    #             Choose date range:
    #         """)),
    # dcc.DatePickerRange(
    #     id='date-picker',
    #     min_date_allowed=date(1900, 1, 1),
    #     max_date_allowed=date(2017, 9, 19),
    #     initial_visible_month=date(2020, 1, 1),
    #     end_date=date=datetime.datetime.now()
    #     )
    # ],style={'width': '24%', 'display': 'inline-block'}),

    html.Div([
    dcc.Graph(id='short-graph') ]),


])


# Need to just filter the DF to within the selected ranges, then run the get_graph_df
# function to get the code necessary to create the graph.  Pretty straightforward
### CALLBACK CODE ###

@app.callback(
    dash.dependencies.Output('short-graph', 'figure'),
    [dash.dependencies.Input('symbol-dropdown', "value")])
    # dash.dependencies.Input('date-picker', 'start_date'),
    # dash.dependencies.Input('date-picker', 'end_date')]])
def update_figure(sym): #, start_dt, end_dt):

    sdf = make_si_gdf(t1, sym)
    sgraph = plot_short_candles(sdf)
    return sgraph

# @app.callback(
#     dash.dependencies.Output('dgx-graph', 'figure'),
#     [dash.dependencies.Input('dix-date', "date"),
#     dash.dependencies.Input('gex-rolling', "value")])
# def update_dix(ddate, grolling):
#
#     date1 = str(ddate)[0:10]
#
#     dgraph = plot_dix(ddate, dix_rolling=grolling, gex_rolling=grolling)
#
#     return dgraph
#
# @app.callback(
#     [dash.dependencies.Output('vix0', 'children'),
#     dash.dependencies.Output('vix1', 'children'),
#     dash.dependencies.Output('vix2', 'children')],
#     [dash.dependencies.Input('update-vix-button', "n_clicks")])
# def update_vix(nclicks):
#
#     vix_vals = dash_vix_values(t1)
#
#     vix0_vals = vix_vals[0]
#     vix1_vals = vix_vals[1]
#     vix2_vals = vix_vals[2]
#
#     v0 = 'Neardate: {nd} // Fardate: {fd} // VIX: {v}'.format(nd=vix0_vals[0], fd=vix0_vals[1], v=vix0_vals[2])
#     v1 = 'Neardate: {nd} // Fardate: {fd} // VIX: {v}'.format(nd=vix1_vals[0], fd=vix1_vals[1], v=vix1_vals[2])
#     v2 = 'Neardate: {nd} // Fardate: {fd} // VIX: {v}'.format(nd=vix2_vals[0], fd=vix2_vals[1], v=vix2_vals[2])
#
#     return v0,v1,v2




if __name__ == '__main__':
    app.run_server(debug=True)#,host='0.0.0.0', port=9000)
