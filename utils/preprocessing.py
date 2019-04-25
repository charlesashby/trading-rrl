import csv, pickle, glob
import datetime
import h5py
import numpy as np
import copy
import matplotlib.pyplot as plt


def to_dt64(date, pattern='%Y%m%d %H:%M:%S'):
    return np.datetime64(datetime.datetime.strptime(date[:17], pattern)).view('i8')


def dt64_to_dt(dt64):
    return datetime.datetime.utcfromtimestamp((dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))


def build_date_bins(start_date, end_date, interval):
    """ 'interval' is in minutes,
        Create bins for storing dates; We split dates in interval
        of 'interval' minutes starting from the first date to
        the last one in the file
    """
    if interval > 1:
        bins = np.array([np.datetime64(datetime.datetime(start_date.year, start_date.month,
                                                         start_date.day, start_date.hour,
                                                         (start_date.minute // interval) * interval)
                                       + datetime.timedelta(minutes=interval) * i) for i
                         in range(int((end_date - start_date).total_seconds() / 60 / interval))])
    else:
        bins = np.array([np.datetime64(datetime.datetime(start_date.year, start_date.month,
                                                         start_date.day, start_date.hour,
                                                         start_date.minute)
                                       + datetime.timedelta(minutes=interval) * i) for i
                         in range(int((end_date - start_date).total_seconds() / 60))])
    return bins


def unzip_files(currency='EURUSD'):
    import glob
    import zipfile
    for year in glob.glob('/media/ashbylepoc/data/trading/*'):
        for month in glob.glob('{}/*'.format(year)):
            for file in glob.glob('{}/*'.format(month)):
                if currency in file:
                    zip_ref = zipfile.ZipFile(file, 'r')
                    zip_ref.extractall(month)
                    zip_ref.close()
                    print(file)


# Pre-process tick data; convert to one-minute data
files = [file[:-1] for file in open('utils/eurusd-files.txt', 'r').readlines()]
for k, file in enumerate(files):
    with open(file) as f:
        reader = csv.reader(f)
        data = []
        for i, row in enumerate(reader):
            data.append(row)

    start_date = datetime.datetime.strptime(data[0][1][:17], '%Y%m%d %H:%M:%S')
    end_date = datetime.datetime.strptime(data[-1][1][:17], '%Y%m%d %H:%M:%S')
    bins = build_date_bins(start_date, end_date, interval=30)
    date_bins_i8 = bins.view('i8')

    # Store all dates as an int
    dates_i8 = [to_dt64(tick[1]) for tick in data]

    # Put all dates in their respective bin (sorted_dates as length len(dates_i8))
    sorted_dates = np.digitize(dates_i8, date_bins_i8)

    #
    prices = [[] for _ in range(len(bins))]
    for i, tick in enumerate(data):
        bin_idx = sorted_dates[i] - 1
        tick = copy.copy(tick)
        time = bins[bin_idx]
        tick[1] = time
        prices[bin_idx].append(tick)

    #
    clean_data = np.zeros(shape=[len(bins), 8], dtype=object)
    for i, price in enumerate(prices):
        if price:
            volume = len(price)
            open_bid = float(price[0][2])
            open_ask = float(price[0][3])
            close_bid = float(price[-1][2])
            close_ask = float(price[-1][3])
            hi = float(max(price, key=lambda k: k[2])[2])
            lo = float(min(price, key=lambda k: k[2])[2])
            time = price[0][1]
            # 7, 8, 9, 10, 11, 12, 13
            clean_data[i] = np.array([time, open_bid, open_ask, close_bid,
                                      close_ask, hi, lo, volume], dtype=object)
        else:
            prv_price = np.copy(clean_data[i - 1])
            prv_price[0] = bins[i]
            prv_price[-1] = 0
            clean_data[i] = prv_price

    # Save everything to pickle object
    save_file = 'data/30m_intervals/{}.pickle'.format(file[:-4].split('/')[-1])
    with open(save_file, 'wb') as f:
        pickle.dump(clean_data, f)
        print('Processing file: {}/{} - Saving to: {}'.format(k, len(files), save_file))


""" Once the data is preprocessed, we still need to change some stuff:
    - We only keep the busiest hours of trading to ensure liquidity (9:00-17:00) 
    - We need to remove week-ends from the data 
    - Create time series:
        - 2000/500 ticks for train/test. Total accuracy is averaged over the overlapping 
          sets in the dataset
        - Recurrent neural network take as input the whole sequence and weights are updated
          during the forward pass (not sure how yet)
        - (close_bid(i) - open_ask(i-1)) / close_bid(i)
"""


def filter_dates(date):
    """ We only want dates in the busiest hours
        to ensure liquidity
    """
    if not (date.weekday() in [5, 6] or (date.hour > 22 and date.minute > 30)):
        return True
    else:
        return False


files = sorted([file for file in glob.glob('data/30m_intervals/*')])
clean_data = []
for i, file in enumerate(files):
    print('Processing file: {}/{}'.format(i, len(files)))
    with open(file, 'rb') as f:
        data = pickle.load(f)
        for tick in data:
            if tick[-1] != 0:
                date = dt64_to_dt(tick[0])
                tick_data = np.array([date.year, date.month, date.day, date.weekday(),
                                      date.hour, date.minute, date.second] + list(tick[1:]))
                clean_data.append(tick_data)


# Creating csv file for tradingrrl.py code
files = sorted([file for file in glob.glob('data/30m_intervals/*')])
clean_data = []
for i, file in enumerate(files):
    print('Processing file: {}/{}'.format(i, len(files)))
    with open(file, 'rb') as f:
        data = pickle.load(f)
        for tick in data:
            if tick[-1] != 0:
                date = dt64_to_dt(tick[0])
                tick_data = ['{}.{}.{}'.format(date.year, date.month, date.day),
                             '{}:{}'.format(date.hour, date.minute)] \
                            + [tick[1], tick[2], tick[3], tick[4], tick[-1]]
                clean_data.append(tick_data)

with open('eurusd.csv', 'a') as f:
    import csv
    writer = csv.writer(f)
    for line in clean_data:
        writer.writerow(line)

""" We use the last 100 log-returns and log-change in volumes
    as features for the neural network. It is computed as:
        
        log((open_bid_(t) - close_ask_(t-1)) / open_bid_(t))
"""

clean_data = np.array(clean_data)
# features = np.zeros(shape=[len(clean_data) - 100, 114])
features = np.zeros(shape=[len(clean_data) - 1, 15])
for i, tick in enumerate(clean_data[1:]):
    returns = (clean_data[1+i, 7] - clean_data[i, 10]) / clean_data[1+i, 7]
    # log_returns = np.log(clean_data[100 + i, 7] / clean_data[i: 100 + i, 10])
    features[i] = np.array([returns] + list(tick))


""" Create pickle file with np.array containing last 100 price changes
"""
with open('returns_usdjpy30.pickle', 'rb') as f:
    data = pickle.load(f)


features = np.zeros(shape=[data.shape[0] - 100, 100])
for i, tick in enumerate(data[100:]):
    features[i] = np.array(data[i: i + 100].reshape(100))

with open('data_last_100_usdjpy.pickle', 'wb') as f:
    pickle.dump(features, f)


""" Create pickle file with tensor of rank 3 with
    [datetime, return, price]
"""
data = []
with open('USDJPY30.csv', 'r') as f:
    reader = csv.reader(f)
    for i, tick in enumerate(reader):
        data.append(tick)

features = []
for i, tick in enumerate(data[1:]):
    date = to_dt64('{} {}'.format(tick[0], tick[1]), pattern='%Y.%m.%d %H:%M')
    r = float(data[i][5]) - float(data[i - 1][5])
    price = float(data[i][5])
    features.append(np.array([date, r, price]))

features = np.array(features)
with open('data_30_usdjpy.pickle', 'wb') as f:
    pickle.dump(features, f)