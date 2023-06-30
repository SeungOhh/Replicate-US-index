from datetime import date, datetime, timedelta
from dateutil import relativedelta
import os

from pandas.tseries.offsets import BMonthEnd
import numpy as np
import pandas as pd

import sqlite3


#%%
def getToday():
    today = date.today().strftime('%Y%m%d')
    return today

def getYesterday():
    yesterday = date.today() + relativedelta.relativedelta(days=-1)
    yesterday = yesterday.strftime('%Y%m%d')
    return yesterday

def getTmrw():
    tmrw = date.today() + relativedelta.relativedelta(days=1)
    tmrw = tmrw.strftime('%Y%m%d')
    return tmrw
    
def getLastMonth():
    lastMonth = date.today() + relativedelta.relativedelta(months=-1)
    lastMonth = lastMonth.strftime('%Y%m%d')
    return lastMonth

def getNextMonth():
    nextMonth = date.today() + relativedelta.relativedelta(months=1)
    nextMonth = nextMonth.strftime('%Y%m%d')
    return nextMonth


def lastBusinessDay(startDate = '2018-01-01', periods = 45):
    s = pd.date_range(startDate, periods=periods, freq='BM')
    return s

def lastDayofMonth(date_arr, include_first_day = False, include_last_day = False, shift = 0, bungi = False):
    # include_first_day: 첫날 추가
    # include_last_day: 마지막날 추가
    # shift
    # bungi: 분기 마지막날로 변경
    
    df_date = pd.DataFrame()
    df_date['date'] = date_arr
    df_date['year'] = pd.to_datetime(date_arr, format = '%Y-%m-%d').strftime('%Y')
    df_date['month'] = pd.to_datetime(date_arr, format = '%Y-%m-%d').strftime('%m')
    df_date['day'] = pd.to_datetime(date_arr, format = '%Y-%m-%d').strftime('%d')
    arr_last_day = []


    if bungi == False:
        for i in range(len(df_date)-1):
            if df_date.iloc[i].day > df_date.iloc[i+1].day:
                arr_last_day.append(1)
            else:
                arr_last_day.append(0)
    
    else: # 분기만
        for i in range(len(df_date)-1):
            if (df_date.iloc[i].day > df_date.iloc[i+1].day) & (df_date.iloc[i].month in ['03','06','09','12']):
                arr_last_day.append(1)
            else:
                arr_last_day.append(0)


    arr_last_day.append(0)
    df_date['lastDay'] = arr_last_day
    df_date['lastDay'] = df_date['lastDay'].shift(shift)
    
    if include_first_day == True:
        df_date.loc[df_date.index[0], 'lastDay'] = 1
    
    if include_last_day == True:
        df_date.loc[df_date.index[-1], 'lastDay'] = 1

    
    df_date = df_date[df_date['lastDay']==1]
    date_out = df_date['date']
    return date_out
    
    


def addDaysToStringDate( date, add_days ):
    date_added = datetime.strptime(date, '%Y-%m-%d') + timedelta(days = add_days)
    date_fin = datetime.strftime(date_added , '%Y-%m-%d')
    return date_fin


def addMonthsToStringDate(date, delta):
    date =  datetime.strptime(date, '%Y-%m-%d')
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    if not m: m = 12
    d = min(date.day, [31,
        29 if y%4==0 and (not y%100==0 or y%400 == 0) else 28,
        31,30,31,30,31,31,30,31,30,31][m-1])
    
    date_out = date.replace(day=d,month=m, year=y)
    date_out = datetime.strftime(date_out, '%Y-%m-%d')
    
    return date_out


#%% arr을 len1과 len2로 변경
def multiple_split(arr, len1, len2):
    print('--starting to split data')
    len_sum = len1 + len2
    iter_num = np.ceil(len(arr)/len_sum).astype(int)

    # init
    out1 = []
    out2 = []
    out1.append(arr[0:len1])
    out2.append(arr[len1 : len1 + len2])

    
    # iter
    for i in range(1,iter_num):
        # if i%100 == 0:
            # print('iter num2 = ', i)        
        out1.append(arr[i*len_sum : i*len_sum + len1])
        out2.append(arr[i*len_sum + len1 : i*len_sum + len_sum])

    out1 = np.vstack(out1)
    out2 = np.vstack(out2)
    print('--starting to split data: Done')
    return out1, out2


#%%
def expand_dim(arr, multiple=20):
    arr_out = []
    for k1 in range(len(arr)):
        for k2 in range(multiple):
            arr_out.append(arr[k1])
    return arr_out



#%%
def sqldb2df(dbname, conditions):
    
    con = sqlite3.connect(dbname)
    cursor = con.cursor()
    
    cursor.execute(conditions)
    
    
    data = cursor.fetchall() 
    columns = [column[0] for column in cursor.description]
    
    df_data = pd.DataFrame(data, columns= columns)
        
    return df_data



# df1 컬럼명에 맞춰서 df2 작성해서 내놓음.
# 없는 컬럼은 fillnumber로 체움
def df_match_cols(df1, df2, fill_number = 0):

    diff_cols = set(df1.columns) - set(df2.columns)
    
    for i in list(diff_cols):
        df2[i] = fill_number
    
    df2 = df2[df1.columns]
    
    return df2




#%%
if __name__ == "__main__":
    # today = getToday()
    # print('Today: ', today)
    # yesterday = getYesterday()
    # print('Yesterday: ', yesterday)
    # tmrw = getTmrw()
    # print('Tomorrow: ', tmrw)
    # lastMonth = getLastMonth()
    # print('lastMonth: ', lastMonth)
    # nextMonth = getNextMonth()
    # print('nextMonth: ', nextMonth)

    # lastBusinessdates = lastBusinessDay()
    
    
    # date = pd.date_range('2020-01-01', '2021-08-31')
    # lastDate = lastDayofMonth(date, include_first_day = True, include_last_day = True)
    # lastDate_bungi = lastDayofMonth(date, include_first_day = False, include_last_day = False, bungi = True)
    
    # date_added = addDaysToStringDate('2020-01-01', 90)
    # print('date_added: ', date_added)
    
    # df = sqldb2df("GLOBAL_MARKET_PRICE_US_20211119.db", "SELECT * FROM SPX_MEMB_GICS")
    
    
    # 
    # df1 = pd.DataFrame(columns = ['aaa','bbb','ccc'], data = [[111,222,333],[444,555,666]])
    # df2 = pd.DataFrame(columns = ['aaa','bbb','ddd'], data = [[1,2,3],[4,5,6]])
    # df3 = df_match_cols(df1, df2, fill_number = 0)
    
    
    # file_exist
    pwd = os.getcwd()
    a = file_exist(pwd, 'test.txt')
    # a = file_exist(file_path, file_name)
    
    
    
    
    
    