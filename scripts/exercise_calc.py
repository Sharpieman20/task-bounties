import asyncio
import websockets
import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ccxt
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI
from scipy.stats import norm
from math import log, sqrt, pi, exp
import re

def compute_realized_vols(coins):
    symbols_to_coins = {f'{coin}/USD':coin for coin in coins}
    symbols = symbols_to_coins.keys()

    cftx = ccxt.ftx(
        {
            'apiKey':'mvnd3WRG56rAP-6MpxFyZ5sN2WnjEO8UsQB6u-6s',
            'secret':'m_m926E36IMlrhCv6bGkLdSdaVegbm5PggpCrFCM'
        }
    )

    markets = cftx.load_markets()
    binance = ccxt.binance()

    time_since = datetime.timestamp(datetime.now())*1000-10*24*60*60*1000

    data_dict = {}
    result_dict = {}
    for symbol in symbols:
        data = cftx.fetchOHLCV(symbol, '1d', since=time_since, limit=10)
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        data_dict[symbol] = df
        df['vol'] = np.log(df.high/df.low)**2
        sample = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        result_dict[symbol] = np.sqrt(df.vol.sum()*1/(4*10*np.log(2)))*np.sqrt(365)*100
    
    coin_result_dict = {coin:result_dict[sym] for sym, coin in symbols_to_coins.items()}

    return coin_result_dict

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
        msg = \
        {
            'jsonrpc' : '2.0',
            'id' : 7617,
            'method' : 'public/get_instruments',
            'params' : {
                'currency' : asset,
                'kind' : 'option',
                'expired' : False
            }
        }
        return self.call_api(msg)
    
    def get_historical_volatility(self, coin):
        msg = \
        {
            'jsonrpc' : '2.0',
            'id' : 8387,
            'method' : 'public/get_historical_volatility',
            'params' : {
                'currency' : coin
            }
        }
        return self.call_api(msg)
    
    def get_price(self, option):
        msg = \
        {
            'jsonrpc' : '2.0',
            'method' : 'public/get_last_trades_by_instrument',
            'params' : {
                'instrument_name' : option,
                'count' : 1
            }
        }
        res = self.call_api(msg)
        return res

def get_relevant_option(options, strike, expiry_date, thresh=24*60*60*1):
    min_expiry_time_diff = 999999999999
    best_option = None
    seconds_threshold = thresh
    min_strike_diff = 99999999999
    for option in options:
        expiry_time_diff = expiry_date.timestamp() - (option['expiration_timestamp']/1000)
        if not option['option_type'] == 'call':
            continue
        if abs(expiry_time_diff) < seconds_threshold:
            strike_diff = strike - option['strike']
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

    def __init__(self, subscription_tier='free'):
        super().__init__()

        self.timeout = 60 / self.subscription_calls_per_minute[subscription_tier]
        self.requestcount = 0

        self.request_super = super()._CoinGeckoAPI__request
    
    def _CoinGeckoAPI__request(self, url):
        if self.requestcount > 0:
            time.sleep(self.timeout)
        self.requestcount += 1
        return self.request_super(url)

def d1(S,K,T,r,sigma):
    return(log(S/K)+(r+(sigma**2)/2.)*T)/(sigma*sqrt(T))

def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma)-sigma*sqrt(T)

def bs_call(S,K,T,r,sigma):
    return S*norm.cdf(d1(S,K,T,r,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))

def get_option_expiry_chance(cur_price, strike_price, expiry_time, iv):
    time_until_expiry_days = (expiry_time.timestamp()-time.time())/(60*60*24)
    time_until_expiry_days /= 365
    adj_iv = (iv/100.0)
    frac = 0.04
    expiry_chance = norm.cdf(d1(cur_price, strike_price, time_until_expiry_days, frac, adj_iv))

    return expiry_chance

def scrape_live_prices(cg, coins, currency):
    coin_prices_dict = {}
    for coin in coins:
        data = cg.get_coin_by_id(coin['id'])
        md = data['market_data']
        current_price_info = md['current_price']
        price_in_currency = current_price_info[currency]
        coin_prices_dict[coin['symbol'].upper()] = price_in_currency
    return coin_prices_dict

def get_info_for_entry(entry):
    match_info = re.fullmatch('^([^-]+)-(\D+?)?-?(\d+)-(\D+)-(\d+)$', entry['product'])
    entry_info = {
        'coin': match_info[1],
        'strike': float(match_info[3]),
        'side': match_info[4].lower(),
        'expiry': datetime.fromtimestamp(int(match_info[5]))
    }
    return entry_info

def get_entry_dict(input_set):
    return [get_info_for_entry(x) for x in input_set]

def main(args):
    input_set = json.load((Path.cwd() / 'input.json').open())

    cg = WrappedCoinGecko()
    deribit = DeribitApi()

    entry_list = get_entry_dict(input_set)

    my_coins = list(entry['coin'] for entry in entry_list)

    all_coins = cg.get_coins_list()

    coin_info = []

    for coin_dict in all_coins:
        if coin_dict['symbol'].upper() in my_coins:
            coin_info.append(coin_dict)
            my_coins.remove(coin_dict['symbol'].upper())
    
    main_coins = ['BTC', 'ETH']

    main_loop(cg, deribit, coin_info, main_coins, entry_list, args.delay*60)

def main_loop(cg, deribit, coin_info, main_coins, entry_list, delay):
    while True:
        realized_vols = compute_realized_vols(coin['symbol'] for coin in coin_info)
        live_prices = scrape_live_prices(cg, coin_info, 'usd')
        main_ivs = {}
        for coin in main_coins:
            all_options = json.loads(deribit.get_listed_options(coin))
            relevant_option = get_relevant_option(all_options['result'], my_strikes[coin], my_expiries[coin])
            last_trade = deribit.get_price(relevant_option['instrument_name'])
            relevant_option['price'] = json.loads(last_trade)['result']['trades'][0]['price']
            relevant_option['iv'] = json.loads(last_trade)['result']['trades'][0]['iv']
            main_ivs[coin] = relevant_option['iv']
        for coin in entry_list:
            my_rv = realized_vols[coin]
            maincoin_rv = realized_vols[main_coins[0]]
            my_ratio = my_rv / maincoin_rv
            my_iv = my_ratio * main_ivs[main_coins[0]]
            my_expiry_chance = get_option_expiry_chance(live_prices[coin], my_strikes[coin], my_expiries[coin], my_iv)
            print(f'{coin} {my_expiry_chance}')
        time.sleep(delay)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--delay', type=float, help=f'time in minutes between updates', default=0.01)

    args = parser.parse_args()

    main(args)
