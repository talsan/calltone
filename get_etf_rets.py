import config
import requests

sp500_ishares_ticker = 'IVV'

request = dict(url='https://www.alphavantage.co/query',
               params={'function': 'TIME_SERIES_DAILY_ADJUSTED',
                       'symbol': sp500_ishares_ticker,
                       'outputsize': 'full',
                       'apikey': config.AlphaVantage.AV_KEY})

response = requests.get(**request)
ticker_time_series = response.json()

