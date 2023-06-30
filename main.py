#%%
import pandas as pd
import numpy as np
import os
from syUtil import * 
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.stats import zscore

#%%
# choose parameters
# Parameter 1: index
file_index = 'phil_semicond.xlsx'
# file_index = 'Global X Autonomous & Electric Vehicles ETF.xlsx'
# file_index = 'iShares Robotics and Artificial Intelligence.xlsx'
# file_index = 'iShares Biotechnology.xlsx'
# file_index = 'VanEck Steel.xlsx'
# file_index = 'csi300.xlsx'


# Parameter 2: Universe 선정 방식
# 'corr', 'corr + beta', 'corr + beta rank'
# option_univ = 'corr + beta'
option_univ = 'corr + beta rank'

# Parameter 3: Universe 종목 수 선정
num_jong = 50

# Parameter 4: Weight 선정 방식 결정
# 'equal weight', 'linear reg', 'nn linear reg'
# option_weight = 'linear reg'
# option_weight = 'equal weight'
option_weight = 'nn linear reg'





#%%
# load data and match index
df_price = pd.read_pickle('.//input//data_pkl//dataguide_adj_price.pkl')
df_turnover = pd.read_pickle('.//input//data_pkl//dataguide_turnover.pkl')
df_mktcap = pd.read_pickle('.//input//data_pkl//dataguide_market_cap.pkl')

# reset column names 
df_info = pd.read_excel('.//input//info.xlsx', header= 5)
df_info.columns = ['CODE', 'SECURITY_NAME', 'sector', 'group', 'industry']


#%%
# load index file 
file_path = './/input//index//'


date_start = df_price.index[0]
df_index = pd.read_excel( file_path + file_index )
df_index = df_index[df_index['Date']>=date_start]
df_index = df_index.set_index(df_index['Date']).drop('Date', axis=1)
df_index = df_index['Adj Close']


# load currecy file
df_krwusd = pd.read_excel( file_path + 'krwusd.xlsx' )
df_krwusd = df_krwusd[df_krwusd['Date']>=date_start]
df_krwusd = df_krwusd.set_index(df_krwusd['Date']).drop('Date', axis=1)
df_krwusd = df_krwusd['Adj Close']
df_krwusd = df_krwusd.reindex(df_index.index)

df_index = df_index * df_krwusd


# match index & columns
columns_to_remove = df_price.columns[df_price.iloc[0].isnull()] # 첫날 가격이 없는 종목은 제외

df_price = df_price.reindex(df_index.index)
df_price = df_price.drop(columns_to_remove, axis=1)

df_turnover = df_turnover.reindex(df_index.index)
df_turnover = df_turnover.drop(columns_to_remove, axis=1)

df_mktcap = df_mktcap.reindex(df_index.index)
df_mktcap = df_mktcap.drop(columns_to_remove, axis=1)






#%%
# calc 1d return 
shift_n = -1 # 국내 주가가 해당 인덱스 대비 며칠이나 늦게 반영되는지, -1: 하루 늦게, 0 동시에, 1: 우리나라가 주가가 먼저

df_1d = df_price.diff()/df_price.shift(1)
df_1d = df_1d.shift(shift_n).fillna(0)  # 미국보다 국내가 하루 늦게 반영되기 때문
df_1d_index = df_index.diff()/df_index.shift(1)
df_1d_index = df_1d_index.fillna(0).to_frame()
df_1d_index = df_1d_index.rename(columns = {'Adj Close': 'index'})


df_5d = df_price.diff(periods=5)/df_price.shift(1)
df_5d = df_5d.shift(shift_n).fillna(0)  # 미국보다 국내가 하루 늦게 반영되기 때문
df_5d_index = df_index.diff(periods=5)/df_index.shift(1)
df_5d_index = df_5d_index.fillna(0).to_frame()
df_5d_index = df_5d_index.rename(columns = {'Adj Close': 'index'})


df_20d = df_price.diff(periods=20)/df_price.shift(1)
df_20d = df_20d.shift(shift_n).fillna(0)  # 미국보다 국내가 하루 늦게 반영되기 때문
df_20d_index = df_index.diff(periods=20)/df_index.shift(1)
df_20d_index = df_20d_index.fillna(0).to_frame()
df_20d_index = df_20d_index.rename(columns = {'Adj Close': 'index'})


df_1d_comb = pd.concat([df_1d_index, df_1d], axis=1)
df_5d_comb = pd.concat([df_5d_index, df_5d], axis=1)
df_20d_comb = pd.concat([df_20d_index, df_20d], axis=1)







#%%
# correlation
def getCorr(date_ini, date_mid, num_jong, option_univ):
    # correlation coefficient 정의
    
    ########## 기본적인 correlation coefficient 계산
    p1 = df_1d_comb[np.logical_and(df_1d_comb.index >= date_ini, df_1d_comb.index <= date_mid)]
    p2 = df_5d_comb[np.logical_and(df_5d_comb.index >= date_ini, df_5d_comb.index <= date_mid)]
    p3 = df_20d_comb[np.logical_and(df_20d_comb.index >= date_ini, df_20d_comb.index <= date_mid)]
    
    ########## return 0 이상인 것만 모아서 계산
    # p1 = p1[p1['index'] > 0]
    # p2 = p2[p2['index'] > 0]
    # p3 = p3[p3['index'] > 0]
    
    ########## correlation 계산 
    corr1 = p1.corr().fillna(0)['index']
    corr2 = p2.corr().fillna(0)['index']
    corr3 = p3.corr().fillna(0)['index']
    
    ########## covariance 계산
    cov1 = p1.cov().fillna(0)['index']
    cov2 = p2.cov().fillna(0)['index']
    cov3 = p3.cov().fillna(0)['index']    

    ########## beta 계산
    beta1 = cov1/cov1['index']
    beta2 = cov2/cov2['index']
    beta3 = cov3/cov3['index']
    
    
    
    ########## 최종 Universe 계산
    if option_univ == 'corr':
        df_corr = corr1 + corr2 + corr3
    elif option_univ == 'corr + beta':
        df_corr = corr1 * beta1 + corr2 * beta2/np.sqrt(5) + corr3 * beta3/np.sqrt(20)
    elif option_univ == 'corr + beta rank':
        df_corr = corr1.rank() + corr2.rank() + corr3.rank() + beta1.rank() + beta2.rank() + beta3.rank()
    else:
        raise 'option Univ 값에 문제가 있습니다.'

               
    ########## Return 값 정리
    df_corr = df_corr.reset_index()
    df_corr.columns = ['CODE', 'corr']
    df_corr.loc[df_corr['CODE'] == 'index', 'corr'] = 99999999 # sorting 할때 index값 처음으로 오게끔
    
    
    df_corr = df_corr.sort_values( by='corr', ascending = False)
    df_corr = df_corr.merge(df_info, on='CODE', how = 'left')
    
    df_corr_all = df_corr.copy()
    
    df_corr_filt = df_corr.iloc[:num_jong] # n 종목 선정
    jong_list = df_corr_filt['CODE']


    ########## 특정 섹터만 선정
    # df_corr = df_corr[np.logical_or(df_corr['CODE'] == 'index', df_corr['group'] == '반도체')]
    # df_corr_filt = df_corr.iloc[:30]
    # jong_list = df_corr_filt['CODE']
    
    return jong_list, df_corr_all






# linear regression
def getCoef( df, option_weight ):
    # print('starting regression')
    # assuming df1 and df2 are your dataframes
    X = df[df.columns[1:]] # rest
    y = df[df.columns[:1]] # index
    
    ########## Linear Regression
    if option_weight == 'linear reg':
        coefficients, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
    
    ########## Non-negative Linear Regression model
    elif option_weight == 'nn linear reg':
        coefficients, residuals = nnls(X, y['index'])
    
    ########## Equal weight
    elif option_weight == 'equal weight':
        coefficients = np.zeros([len(X.columns),1]) + 1/len(X.columns)
        residuals = np.nan
    
    return coefficients, residuals




# average 1 day return
def getResult( df, coefficients ):
    X = df[df.columns[1:]]
    result = np.matmul(X, coefficients)
    return result




#%%
# 종목별 비중 계산

# 날짜
dates = df_price.index
dates_end_of_months = lastDayofMonth(dates) # 월말


df_in_sample_y  = pd.DataFrame()
df_out_sample_y = pd.DataFrame()

# 월간 변경 종목수 계산에 필요한 변수
num_jong_diff_list = []
num_jong_diff = []
jong_list_prev = []
coefficients_sum = []



# 매월 말 
for i in range(len(dates_end_of_months)-12):
    print('Current date:', dates_end_of_months.iloc[i])
    # set dates
    date_ini = dates_end_of_months.iloc[i]
    date_mid = dates_end_of_months.iloc[i+11]
    date_fin = dates_end_of_months.iloc[i+12]
    
    
    ###################### 유니버스 선정
    jong_list, df_corr_all = getCorr(date_ini, date_mid, num_jong, option_univ) 

    
    ###################### 종목별 비중 결정
    df_X = df_1d_comb[np.logical_and(df_1d_comb.index >= date_ini, df_1d_comb.index <= date_mid)] # date_ini 부터 date_mid 까지의 가격정보로 coef 계산
    df_X = df_X[jong_list] # universe 종목만 선정
    
    # coefficient 계산: coefficient = weight 비중
    coefficients, residuals =  getCoef( df_X, option_weight )
    coefficients_sum.append(coefficients.sum())
    
    # 유니버스 종목 변경 확인
    num_jong_diff_list.append(len(set(jong_list_prev) - set(jong_list)))
    
    
    ###################### 수익률 단순 계산
    if i ==0:
        df_in_sample_x = df_1d_comb[np.logical_and(df_1d_comb.index >= date_ini, df_1d_comb.index <= date_mid)]
        df_in_sample_x = df_in_sample_x[jong_list]
        df_in_sample_y_  = getResult(df_in_sample_x, coefficients)
        df_in_sample_y  = pd.concat([df_in_sample_y, df_in_sample_y_], axis=0)
        
    
    df_out_sample_x = df_1d_comb[np.logical_and(df_1d.index >= date_mid, df_1d.index < date_fin)] 
    df_out_sample_x = df_out_sample_x[jong_list]
    df_out_sample_y_ = getResult(df_out_sample_x, coefficients)                 # 해당기간 포트폴리오 수익률
    df_out_sample_y = pd.concat([df_out_sample_y, df_out_sample_y_], axis=0)    # 누적 수익률 series
    jong_list_prev = list(jong_list)


# 첫날 수익률은 0으로
df_out_sample_y.iloc[0] = 0
print('Turnover: ', round(np.mean(num_jong_diff_list)/len(jong_list),2))






#%%
# 최종 결과 확인


# 수익률 역산
init_in_sample = df_index[df_index.index==df_in_sample_y.index[0]][0] # 첫날 가격을 index 가격과 동일하게
df_in_sample = (df_in_sample_y + 1).cumprod() * init_in_sample        # 첫날 이후 가격 계산

init_out_sample = df_index[df_index.index==df_out_sample_y.index[0]][0] # 첫날 가격을 index 가격과 동일하게
df_out_sample = (df_out_sample_y + 1).cumprod() * init_out_sample       # 첫날 이후 가격 계산

# 최종 결과 주가로
df_fin = pd.concat([df_index, df_in_sample, df_out_sample],axis=1)
df_fin.columns = ['index', 'in sample', 'out of sample']

# 최종 결과 1d 수익률로
df_fin_ret = pd.concat([df_1d_index, df_in_sample_y, df_out_sample_y],axis=1)
df_fin_ret.columns = ['index', 'in sample', 'out of sample']




# 평가1-1: out of sample data
err = (df_fin_ret['out of sample'] - df_fin_ret['index'])
err_sum = abs(err).sum()
err_sum = round(err_sum,2)          # absolute err
err_mean = np.mean(err)
err_rms = (err**2).mean()           # rms error
err_rms = round(err_rms * 1000, 2)  # 보기 편하도록 1000 곱함

err_std = round(np.std(err), 4)

# 평가1-2: in sample data
err_ = (df_fin_ret['in sample'] - df_fin_ret['index'])
err_sum_ = abs(err_).sum()
err_sum_ = round(err_sum_,2)  # absolute err
err_rms_ = (err_**2).mean()          # rms error
err_rms_ = round(err_rms_ * 1000, 2) # 보기 편하도록 1000 곱함



# 평가2-1: " 1d return에 대한 " correlation coefficient
corr_result = np.corrcoef(df_fin_ret.dropna(subset='out of sample')['out of sample'], df_fin_ret.dropna(subset='out of sample')['index'])[1,0]
corr_result = round(corr_result,2)

cov_result = np.cov(df_fin_ret.dropna(subset='out of sample')['out of sample'], df_fin_ret.dropna(subset='out of sample')['index'])
beta_result = cov_result[1,0] / cov_result[0,0]
beta_result = round(beta_result,2)

# 평가2-2: " 주가에 대한 " correlation coefficient
# corr_result = np.corrcoef(df_fin.dropna(subset='out of sample')['out of sample'], df_fin.dropna(subset='out of sample')['index'])[1,0]
# cov_result = np.cov(df_fin.dropna(subset='out of sample')['out of sample'], df_fin.dropna(subset='out of sample')['index'])
# beta_result = cov_result[1,0] / cov_result[0,0]










#%%
# plotting
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1, 1]})


# Plot in-sample data
axes[0].plot(df_fin['index'], label=file_index[:-5])
axes[0].plot(df_fin['in sample'], label='in sample')
axes[0].plot(df_fin['out of sample'], label='out of sample')
axes[0].set_title(file_index[:-5] + ',   ' +  option_univ + ',   ' + option_weight + ',   n:' + str(num_jong))
axes[0].set_xlim([df_index.index[0], df_index.index[-1]])
y_diff = max(df_index) - min(df_index)
y_min = min(df_index) - y_diff * 0.1
y_max = max(df_index) + y_diff * 0.1
axes[0].set_ylim(y_min, y_max)
axes[0].legend(loc='upper left')
axes[0].text(df_index.index[-140], y_min + y_diff * 0.15, "Corr coef: " + str(corr_result), bbox=dict(facecolor='green', alpha=0.3))
axes[0].text(df_index.index[-140], y_min + y_diff * 0.05 , "beta: " + str(beta_result), bbox=dict(facecolor='yellow', alpha=0.3))



axes[1].plot(df_fin_ret['index'], label=file_index[:-5])
axes[1].plot(df_fin_ret['in sample'], label='in sample')
axes[1].plot(df_fin_ret['out of sample'], label='out of sample')
axes[1].set_title('1d return')
axes[1].set_xlim([df_index.index[0], df_index.index[-1]])
axes[1].legend(loc='upper left')
axes[1].set_ylim([-0.1, 0.1])
axes[1].grid(True)


axes[2].plot(err_, label='1d ret err: in sample', color='gray')
axes[2].plot(err, label='1d ret err: out of sample', color='black')
axes[2].set_title('1d return Err (%)')
axes[2].set_xlim([df_index.index[0], df_index.index[-1]])
axes[2].set_ylim([-0.1, 0.1])
axes[2].legend()
axes[2].legend(loc='upper left')
axes[2].grid(True)
axes[2].text(df_index.index[-140],   -0.08, "RMS ERR sum: " + str(err_rms), bbox=dict(facecolor='black', alpha=0.3))
axes[2].text(df_index.index[20],   -0.08, "RMS ERR sum: " + str(err_rms_), bbox=dict(facecolor='gray', alpha=0.3))
                          
plt.tight_layout()  # Optional, improves the spacing between subplots
plt.show()
