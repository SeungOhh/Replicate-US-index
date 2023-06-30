import pandas as pd
import os 
import datetime
import pickle




#%%

def get_dataguide(path):
    # path = './/input//data_xlsx//001_Dataguide_종목_수정종가.xlsx'
    df_data = pd.read_excel( path )
    
    # #%%
    # column명 다시 정해서
    code_columns = df_data.loc[7] # 코드
    code_columns[0] = 'DATE'
    jong_columns = df_data.loc[8] # 종목명
    jong_columns[0] = 'DATE'
    
    
    # 데이터 정리 
    df_data = df_data.loc[13:]
    df_data.columns = code_columns
    df_data.set_index('DATE', inplace=True)
    
    
    # 종목 수 확인
    num_cols = df_data.shape[1]
    num_jongs = int((num_cols)) # 종가, 시총, 거래량
    print('종목수는: ', num_jongs, ' 입니다.')
    
    return df_data



#%%
# 파일리스트
file_list = os.listdir('.//input//data_xlsx') # 모든 파일 보기
file_list_xlsx = [ file for file in file_list if file.endswith('xlsx') ] # 엑셀파일만 보기

# file_list_jong = [ file for file in file_list_xlsx if 'Dataguide_종목' in file ] # 종목파일만 보기
# file_list_event = [ file for file in file_list_xlsx if 'Dataguide_Event' in file ] # 그 외 파일만 보기

file_list_jong = file_list_xlsx 


#%%
# 파일저장
for idx, i in enumerate(file_list_jong):
    
    # 특정 파일만 할 경우 
    # if i != '901_Dataguide_종목_분기별_당기순이익.xlsx':
    #     pass
    # else:
    #     file_name = i[:-5]
    #     df_data = get_dataguide('.//input//data_xlsx//' + i)
    #     df_data.to_pickle('.//input//data_pkl//' + file_name + '.pkl')
    #     print(idx, ' / ', len(file_list_jong), ':  ', file_name)
    
    
    # 모든 파일에 대해 할 경우
    file_name = i[:-5]
    df_data = get_dataguide('.//input//data_xlsx//' + i)
    df_data.to_pickle('.//input//data_pkl//' + file_name + '.pkl')
    print(idx, ' / ', len(file_list_jong), ':  ', file_name)
    





