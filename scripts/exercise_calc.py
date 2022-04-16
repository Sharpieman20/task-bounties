

import asyncio
import websockets
import requests
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests
import json
import ccxt

import datetime

from pathlib import Path
import sys
import argparse
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI
import pandas as pd
import time


import pandas as pd
from pathlib import Path
from scipy.stats import norm
from math import log, sqrt, pi, exp
import numpy as np
from datetime import datetime, timedelta


def compute_realized_vols(coins):

    symbols_to_coins = {f'{coin}/USD':coin for coin in coins}
    symbols = symbols_to_coins.keys()

    cftx = ccxt.ftx(
        {
            "apiKey":"mvnd3WRG56rAP-6MpxFyZ5sN2WnjEO8UsQB6u-6s",
            "secret":"m_m926E36IMlrhCv6bGkLdSdaVegbm5PggpCrFCM"
        }
    )

    markets = cftx.load_markets()
    binance = ccxt.binance()

    # 10 days
    time_since = datetime.timestamp(datetime.now())*1000-10*24*60*60*1000

    data_dict = {}
    result_dict = {}
    for symbol in symbols:
        data = cftx.fetchOHLCV(symbol, '1d', since=time_since, limit=10)
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
        # df['prev_open'] = df['open'].shift(1)
        # df['yield'] = (df['open'] - df['prev_open']) / df['prev_open']
        # sig = sqrt(df['yield'].size)*df['yield'].std()
        # result_dict[symbol] = sig
        data_dict[symbol] = df
        df["vol"] = np.log(df.high/df.low)**2
        sample = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
        result_dict[symbol] = np.sqrt(df.vol.sum()*1/(4*10*np.log(2)))*np.sqrt(365)*100
    
    coin_result_dict = {coin:result_dict[sym] for sym, coin in symbols_to_coins.items()}

    return coin_result_dict



def get_deribit_name_for_option(option):
    # coin = 
    day = '15'
    month = 'APR'
    year = '22'
    date_str = f'{day}{month}{year}'
    fmt = f'{coin}-{date_str}-STRIKE-K'

class DeribitApi:
    subscription_calls_per_second = {
        'tier4': 5,
        'tier3': 10,
        'tier2': 20,
        'tier1': 30
    }

    def __init__(self, subscription_tier='tier4'):
        super().__init__()

        self.timeout = 1.1*(1/DeribitApi.subscription_calls_per_second[subscription_tier])
        self.requestcount = 0
    
    def call_api(self, msg):
        time.sleep(self.timeout)
        async def call_api_inner(msg):
            async with websockets.connect('wss://www.deribit.com/ws/api/v2') as websocket:
                await websocket.send(msg)
                return await websocket.recv()

        return asyncio.get_event_loop().run_until_complete(call_api_inner(json.dumps(msg)))
    
    # get all options listed on deribit for asset during time period
    def get_listed_options(self, asset):
        # print(asset)
        msg = \
        {
        "jsonrpc" : "2.0",
        "id" : 7617,
        "method" : "public/get_instruments",
        "params" : {
            "currency" : asset,
            "kind" : "option",
            "expired" : False
        }
        }
        return self.call_api(msg)
    
    def get_historical_volatility(self, coin):
        msg = \
        {
            "jsonrpc" : "2.0",
            "id" : 8387,
            "method" : "public/get_historical_volatility",
            "params" : {
                "currency" : coin
            }
        }
        return self.call_api(msg)
    
    def get_price(self, option):
        # request_msg = {
        #     'jsonrpc': '2.0',
        #     'method': 'public/get_volatility_index_data',
        #     'params': None

        # }
        # msg = \
        # {
        #     "jsonrpc" : "2.0",
        #     'method' : 'public/get_volatility_index_data',
        #     "params" : {
        #         "currency" : "BTC",
        #         "start_timestamp" : 1609545010000,
        #         "end_timestamp" : 1609545010000,
        #         "resolution" : "60"
        #     }
        # }
        msg = \
        {
            "jsonrpc" : "2.0",
            "method" : "public/get_last_trades_by_instrument",
            "params" : {
                "instrument_name" : option,
                "count" : 1
            }
        }
        res = self.call_api(msg)
        # print(res)
        return res




def get_relevant_option(options, strike, expiry_date, thresh=24*60*60*1):
    min_expiry_time_diff = 999999999999
    best_option = None
    seconds_threshold = thresh
    min_strike_diff = 99999999999
    for option in options:
        # print(f"{option['instrument_name']} {datetime.fromtimestamp(option['expiration_timestamp']/1000).strftime('%Y-%m-%d')}")
        expiry_time_diff = expiry_date.timestamp() - (option['expiration_timestamp']/1000)
        if not option['option_type'] == 'call':
            continue
        if abs(expiry_time_diff) < seconds_threshold:
            # print(option['expiration_timestamp']/1000)
            strike_diff = strike - option['strike']
            # print(strike_diff)
            if abs(strike_diff) < min_strike_diff:
                min_strike_diff = abs(strike_diff)
                best_option = option
    if best_option is None:
        return get_relevant_option(options, strike, expiry_date, thresh*2)
    return best_option

class WrappedCoinGecko(CoinGeckoAPI):
    subscription_calls_per_minute = {
        'free': 50,
        'analyst': 500,
        'pro': 1000,
        'enterprise': 1000
    }

    def __init__(self, subscription_tier):
        super().__init__()

        self.timeout = 60 / self.subscription_calls_per_minute[subscription_tier]
        self.requestcount = 0

        self.request_super = super()._CoinGeckoAPI__request
    
    def _CoinGeckoAPI__request(self, url):
        if self.requestcount > 0:
            time.sleep(self.timeout)
        self.requestcount += 1
        return self.request_super(url)

def d1_numer(S,K,T,r,sigma):
    return log(S/K)+((r+((sigma**2)/2.))*T)

def d1_denom(S,K,T,r,sigma):
    return sigma*sqrt(T)

def d1(S,K,T,r,sigma):
    return(log(S/K)+(r+(sigma**2)/2.)*T)/(sigma*sqrt(T))

def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma)-sigma*sqrt(T)

def bs_call(S,K,T,r,sigma):
    return S*norm.cdf(d1(S,K,T,r,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))

def get_option_implied_vol(cur_price, option, rv):
    # d2(cur_price, option['strike'], )
    # time_until_expiry_days = ((option['expiration_timestamp']/1000)-time.time())/(60*60*24)
    # print(cur_price)
    # print(option['strike'])
    # print(time_until_expiry_days)
    # print(rv)
    # print(option['price'])
    # # option['price']
    # expected_price = bs_call(cur_price, option['strike'], time_until_expiry_days, 0.02, rv)
    # print(expected_price)
    # raise
    # pass
    return option['iv']

def get_option_expiry_chance(cur_price, strike_price, expiry_time, iv):
    # d2(cur_price, option['strike'], )
    # print(expiry_time.timestamp())
    time_until_expiry_days = (expiry_time.timestamp()-time.time())/(60*60*24)
    time_until_expiry_days /= 365
    # print(cur_price)
    # print(strike_price)
    # print(time_until_expiry_days)
    # print(iv/100.0)
    adj_iv = (iv/100.0)
    frac = 0.04
    # frac = 4
    # print(option['price'])
    # # option['price']
    # expiry_chance = bs_call(cur_price, strike_price, time_until_expiry_days, 0.04, adj_iv)
    # expiry_chance_2 = bs_call(cur_price+1, strike_price, time_until_expiry_days, 0.04, adj_iv)
    # print(log(cur_price/strike_price))
    # print((frac+((adj_iv**2)/2.))*time_until_expiry_days)
    # print(d1_numer(cur_price, strike_price, time_until_expiry_days, frac, adj_iv))
    # print(d1_denom(cur_price, strike_price, time_until_expiry_days, frac, adj_iv))
    # print(norm.cdf(expiry_chance)
    # print(expiry_chance_2)
    # print(norm.cdf(expiry_chance))
    # print(d1(cur_price, strike_price, time_until_expiry_days, frac, adj_iv))
    expiry_chance = norm.cdf(d1(cur_price, strike_price, time_until_expiry_days, frac, adj_iv))

    return expiry_chance

def scrape_live_prices(cg, coins, currency):
    # live assumes you don't need the output saved, so just does a simple print
    coin_prices_dict = {}
    for coin in coins:
        data = cg.get_coin_by_id(coin['id'])
        md = data['market_data']
        current_price_info = md['current_price']
        price_in_currency = current_price_info[currency]
        coin_prices_dict[coin['symbol'].upper()] = price_in_currency
    return coin_prices_dict


def get_friktion_strikes():
    friktion_strike_dict = {
        
        'BTC': 44500,
        'ETH': 3400,
        'SOL': 123
    }
    return friktion_strike_dict

def get_friktion_expiries():
    friktion_expiries = {
        'BTC': '2022-04-22',
        'ETH': '2022-04-22',
        'SOL': '2022-04-22'
    }
    parsed_expiries = {}
    for key in friktion_expiries.keys():
        parsed_expiries[key] = datetime.strptime(friktion_expiries[key], '%Y-%m-%d')
    return parsed_expiries

def main_loop():
    cg = WrappedCoinGecko('free')

    my_strikes = get_friktion_strikes()
    my_expiries = get_friktion_expiries()

    my_coins = list(my_strikes.keys())

    all_coins = cg.get_coins_list()

    coin_info = []

    for coin_dict in all_coins:
        if coin_dict['symbol'].upper() in my_coins:
            coin_info.append(coin_dict)
            my_coins.remove(coin_dict['symbol'].upper())
    
    main_coins = ['BTC', 'ETH']

    deribit = DeribitApi()
    while True:
        coins = my_strikes.keys()
        realized_vols = compute_realized_vols(my_strikes.keys())
        live_prices = scrape_live_prices(cg, coin_info, 'usd')
        # print(live_prices)
        main_ivs = {}
        for coin in main_coins:
            all_options = json.loads(deribit.get_listed_options(coin))
            # print(type(all_options))
            relevant_option = get_relevant_option(all_options['result'], my_strikes[coin], my_expiries[coin])
            # print(relevant_option)
            last_trade = deribit.get_price(relevant_option['instrument_name'])
            relevant_option['price'] = json.loads(last_trade)['result']['trades'][0]['price']
            relevant_option['iv'] = json.loads(last_trade)['result']['trades'][0]['iv']
            # print(realized_vols[coin])
            # print(live_prices[coin])
            # iv = get_option_implied_vol(live_prices[coin], relevant_option, realized_vols[coin])
            main_ivs[coin] = relevant_option['iv']
            # raise
            # print(iv)
        for coin in coins:
            my_rv = realized_vols[coin]
            maincoin_rv = realized_vols[main_coins[0]]
            my_ratio = my_rv / maincoin_rv
            my_iv = my_ratio * main_ivs[main_coins[0]]
            my_expiry_chance = get_option_expiry_chance(live_prices[coin], my_strikes[coin], my_expiries[coin], my_iv)
            print(f"{coin} {my_expiry_chance}")
        print()

        time.sleep(10)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--delay', type=float, help=f'time in minutes between updates')

    args = parser.parse_args()
    
    main_loop()
