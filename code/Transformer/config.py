START_DATE = "2010-01-01"
END_DATE = "2022-05-07"

INF = 1100

## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    # "return_ratio",
]


ADDITIONAL_FEATURE = [
    'label_short_term',
    'label_long_term'
]

TEMPORAL_FEATURE = [
    'open', 
    'close', 
    'high', 
    'low', 
    'volume', 
    'dopen', 
    'dclose', 
    'dhigh', 
    'dlow', 
    'dvolume'
]


# use CSI -300 ticker
USE_CSI_300_TICKET = ['600519.SS',
 '601318.SS',
 '600036.SS',
 '000858.SZ',
 '600276.SS',
 '601166.SS',
 '601888.SS',
 '300059.SZ',
 '000651.SZ',
 '600900.SS',
 '600887.SS',
 '000001.SZ',
 '000725.SZ',
 '600030.SS',
 '300015.SZ',
 '601398.SS',
 '000568.SZ',
 '600031.SS',
 '600309.SS',
 '000002.SZ',
 '600809.SS',
 '601919.SS',
 '002142.SZ',
 '600436.SS',
 '601328.SS',
 '601899.SS',
 '002304.SZ',
 '002352.SZ',
 '002230.SZ',
 '300014.SZ',
 '600000.SS',
 '600438.SS',
 '600837.SS',
 '000661.SZ',
 '000100.SZ',
 '000063.SZ',
 '002241.SZ',
 '002271.SZ',
 '600585.SS',
 '600690.SS',
 '601601.SS',
 '601668.SS',
 '002027.SZ',
 '600016.SS',
 '600763.SS',
 '600196.SS',
 '000338.SZ',
 '600048.SS',
 '600703.SS',
 '002129.SZ',
 '600050.SS',
 '601688.SS',
 '600660.SS',
 '600104.SS',
 '600570.SS',
 '601766.SS',
 '601169.SS',
 '600999.SS',
 '002311.SZ',
 '002371.SZ',
 '600019.SS',
 '002049.SZ',
 '600406.SS',
 '601088.SS',
 '601988.SS',
 '000538.SZ',
 '000625.SZ',
 '600745.SS',
 '600028.SS',
 '600893.SS',
 '600346.SS',
 '601628.SS',
 '600588.SS',
 '601009.SS',
 '601390.SS',
 '601857.SS',
 '600009.SS',
 '600132.SS',
 '600584.SS',
 '000776.SZ',
 '000895.SZ',
 '002001.SZ',
 '600111.SS',
 '600426.SS',
 '601939.SS',
 '000166.SZ',
 '002050.SZ',
 '002179.SZ']


use_ticker_dict = {'CSI':USE_CSI_300_TICKET, 'TEST': USE_CSI_300_TICKET[:5]}

CSI_date = ['20110419', '20181228', '20180102', '20201231',  '20190402', '20211231']

date_dict = {'CSI': CSI_date, 'TEST': CSI_date}
