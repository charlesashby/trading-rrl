import wget
import os
import ssl
import requests



ssl._create_default_https_context = ssl._create_unverified_context
# Download data from truefx.com
# ex: 1: https://truefx.com/dev/data/2019/2019-01/AUDJPY-2019-01.zip
#     2: https://truefx.com/dev/data/2009/DECEMBER-2009/AUDJPY-2009-12.zip

truefx_uri_1 = 'https://truefx.com/dev/data/{}/{}-{}/{}-{}-{}.zip'
truefx_uri_2 = 'https://truefx.com/dev/data/{}/{}-{}/{}-{}-{}.zip'
data_path = '/media/ashbylepoc/data/trading'
currency_pairs = ['AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD',
                  'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER',
          'OCTOBER', 'NOVEMBER', 'DECEMBER']

for year in range(2010, 2019):
    year_path = '{}/{}'.format(data_path, year)
    os.mkdir(year_path)

    for month in range(12):
        month_pad = '{:02d}'.format(month + 1)
        month_path = '{}/{}'.format(year_path, month_pad)
        os.mkdir(month_path)
        uri = 1
        # try:
        #     r = requests.get(truefx_uri_1.format(year, year, month_pad, 'AUDJPY', year, month_pad))
        #     uri = 1
        # except Exception:
        #     # r = requests.get(truefx_uri_2.format(year, months))
        #     uri = 2
        for currency in currency_pairs:
            print('downloading: {}/{}-{}-{}.zip'.format(month_path, currency, year, month_pad))

            if uri == 1:
                wget.download(truefx_uri_1.format(year, year, month_pad, currency, year, month_pad),
                              out='{}/{}-{}-{}.zip'.format(month_path, currency, year, month_pad))
            # print(truefx_uri.format(year, year, month_pad, currency, year, month_pad))
            else:
                wget.download(truefx_uri_2.format(year, months[month], year, currency, year, month_pad),
                              out='{}/{}-{}-{}.zip'.format(month_path, currency, year, month_pad))