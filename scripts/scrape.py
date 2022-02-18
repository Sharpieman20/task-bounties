'''
Simple script for downloading coingecko data.

Gives either live or historical depending on arguments you pass (run with --help to see args).
Historical data is saved to the passed output_dir in csv format.
It can be loaded with pd.read_csv(output_dir / 'mycoin.csv', index_col='date').

Coingecko only supports 1 day resolution for >90 days, so I just went with daily resolution.
If it's not something we have already, I already have a similar script written for binance (they have trade by trade historical data).
Let me know and I can share that one as well, and adapt it or this one to work with whatever our other programs expect.

EXAMPLE:
python3 scrape.py doge btc sol eth
DOGE currently has price      0.15592 USD
BTC  currently has price  43518.00000 USD
ETH  currently has price   3084.06000 USD
SOL  currently has price    111.86000 USD

python3 scrape.py sol sbr --start 2022-01-01 --end 2022-02-01
Downloading data for sbr
Downloading data for sol


Author: Sharpieman20
For: Friktion
'''

from pathlib import Path
import sys
import argparse
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI
import pandas as pd
import time

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

def scrape_date(cg, coin, date_to_scrape):
    return cg.get_coin_history_by_id(coin['id'], date_to_scrape.strftime('%d-%m-%Y'))

def do_scrape_inner(cg, coin, start_date, end_date):
    all_days = []
    for day in range(end_date.toordinal() - start_date.toordinal()):
        all_days.append(scrape_date(cg, coin, start_date + timedelta(day)))
    return all_days

def save_historical_prices(coin, data, output_dir, start_date, currency):
    prices_by_date = {}
    day_ind = 0
    for day_dict in data:
        md = day_dict['market_data']
        price = md['current_price']
        price_in_currency = price[currency]
        date = start_date + timedelta(day_ind)
        prices_by_date[date] = price_in_currency
        day_ind += 1
    df = pd.DataFrame.from_dict(prices_by_date, orient='index', columns=['price'])
    df.index.name = 'date'
    df.to_csv(output_dir / f'{coin["symbol"]}.csv')

def scrape_historical_prices(cg, output_dir, coins, start_date, end_date, currency):
    all_data = []
    for coin in coins:
        print(f'Downloading data for {coin["symbol"]}')
        historical_data = do_scrape_inner(cg, coin, start_date, end_date)
        save_historical_prices(coin, historical_data, output_dir, start_date, currency)
    print(f'Downloads complete to {output_dir.relative_to(Path.cwd())}')

def scrape_live_prices(cg, coins, currency, prettyprint):
    # live assumes you don't need the output saved, so just does a simple print
    for coin in coins:
        data = cg.get_coin_by_id(coin['id'])
        md = data['market_data']
        current_price_info = md['current_price']
        price_in_currency = current_price_info[currency]
        if prettyprint:
            print(f'{coin["symbol"].upper():4} currently has price {price_in_currency:12.5f} {currency.upper()}')
        else:
            print(f'{coin["symbol"]} {price_in_currency}')

def parse_date(date_to_parse):
    date_fmt_one = 'YYYY-mm-dd'
    if len(date_to_parse) > len(date_fmt_one):
        return datetime.strptime(date_to_parse, '%Y-%m-%d %H:%M:%S')
    return datetime.strptime(date_to_parse, '%Y-%m-%d')

def main(args):
    cg = WrappedCoinGecko(args.subscription)

    symbol = args.symbols
    currency = args.currency

    all_coins = cg.get_coins_list()

    my_coins = [sym.lower() for sym in args.symbols]

    if args.input:
        for symbol in (Path.cwd() / args.input).open('r').readlines():
            my_coins.append(symbol.rstrip().lower())

    coin_info = []

    for coin_dict in all_coins:
        if coin_dict['symbol'] in my_coins:
            coin_info.append(coin_dict)
            my_coins.remove(coin_dict['symbol'])

    output_dir = Path.cwd() / args.output

    if not output_dir.exists():
        output_dir.mkdir()

    get_live = True

    if args.start is not None:
        get_live = False

    if get_live:
        scrape_live_prices(cg, coin_info, args.currency, args.pretty)
    else:
        parsed_start_date = parse_date(args.start)
        parsed_end_date = parse_date(args.end)
        scrape_historical_prices(cg, output_dir, coin_info, parsed_start_date, parsed_end_date, args.currency)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    date_format_to_show = 'format YYYY-MM-DD or YYYY-mm-dd hh:mm:ss'

    parser.add_argument('symbols', type=str, nargs='+', help='the symbol of the coin for which we want data')
    parser.add_argument('--start', type=str, help=f'start time, in {date_format_to_show}')
    parser.add_argument('--end', type=str, help=f'end time, in {date_format_to_show}')
    parser.add_argument('--currency', type=str, default='usd', help='the currency the results should be denominated in')
    parser.add_argument('--input', type=str,  help='location for input to the script, relative to the directory from which it is ran. used for live if you would like to get prices for a list of coins.')
    parser.add_argument('--output', type=str, default='scrape_output', help='location for output of the script, relative to the directory from which it is ran')
    parser.add_argument('--subscription', type=str, default='free', help='the subscription tier you are using. changes the rate limiting')
    parser.add_argument('--pretty', dest='pretty', action='store_true', help='included by default - means live output will be formatted in a readable way')
    parser.add_argument('--plain', dest='pretty', action='store_false', help='pass if live output should not be pretty printed')

    parser.set_defaults(pretty=True)

    args = parser.parse_args()
    
    main(args)


