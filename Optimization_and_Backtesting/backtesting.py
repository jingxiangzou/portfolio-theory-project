# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import math 
import os
import datetime as dt
import matplotlib.pyplot as plt



def find_date_index(input_date, dates_lst):
    
    for ind in range(len(dates_lst)):
        
        if input_date <= dates_lst[ind]:
            
            return ind - 1 
        else:
            pass
            
            
def lst_process(lst):
    
    lst1 = np.array(lst) + np.array([1] * len(lst))
    kks = np.array([np.prod(lst1[:i]) for i in np.arange(1, (len(lst)+1), 1)])
    result = kks - np.array([1] * len(lst))
    
    return result
            



class ShowResults:
    
    def __init__(self, gmv_fn1='gmv.csv', hrp_fn2='hrp.csv', rp_fn3='rp.csv', tickers_fn='tickers15.csv'):
        
        self.gmv_df = pd.read_csv(gmv_fn1) # reads the GMV weights 
        self.hrp_df = pd.read_csv(hrp_fn2) # reads the HRP weights
        self.rp_df  = pd.read_csv(rp_fn3)  # reads the RP weights
        self.tickers15_df = pd.read_csv(tickers_fn) # reads the tickers file
        
        self.time()
        self.main()
        self.illustration()
        
    def time(self):
        
        start = dt.datetime(2013, 2, 28)
        end = dt.datetime(2023, 2, 28)
        
        adj_days = pd.read_csv('last_day_of_month_trading.csv')
                
        self.tickers15_df.index = pd.to_datetime(adj_days['Date'])
        self.gmv_df.index = pd.to_datetime(adj_days['Date'])
        self.hrp_df.index = pd.to_datetime(adj_days['Date'])
        self.rp_df.index = pd.to_datetime(adj_days['Date'])
        
        return 0
        
        
    
    def main(self):
        
        start = dt.datetime(2013, 2, 28)
        end = dt.datetime(2023, 2, 28)
        
        file_path = "C:\\Users\\jxzou\\OneDrive\\桌面\\730 lyasoff project\\stock_data_alpha"
        file_p = file_path + "\\"
        k_list = []
        for i, j, k in os.walk(file_path):
            k_list.append(k)
            
        rets_files = k_list[0] 
        N = len(rets_files) 
        lst_merged = []
        
        for i in range(N):
            
            df = pd.read_csv(file_p + rets_files[i])
            df['Date'] = pd.to_datetime(df['date'])
            df.index = df['Date']
            df = df[df.Date < end]
            df = df[df.Date > start]
            lst_merged.append(df["return"])

        self.df_merged = pd.DataFrame(lst_merged[0])
        
        for i in range(N-1):
            self.df_merged = pd.concat([self.df_merged, lst_merged[i+1]], axis=1)
                
        kkss = [ele.split('.')[1] for ele in rets_files]
        self.df_merged.columns = kkss
        
        new_df = pd.DataFrame(index = self.df_merged.index)
        change_days = self.tickers15_df.index.to_list()
        self.all_days = new_df.index.to_list()
        
        self.rets_oneN, self.rets_gmv, self.rets_hrp, self.rets_rp = [], [], [], []
        
        for day in self.all_days:
            ind_num = find_date_index(day, change_days[1:])
            ind_date = change_days[ind_num + 1]
            
            print(self.gmv_df)
            
            # weights that day
            ws_gmv = np.array(self.gmv_df.loc[ind_date].to_list()[1:])
            ws_hrp = np.array(self.hrp_df.loc[ind_date].to_list()[1:])
            ws_rp = np.array(self.rp_df.loc[ind_date].to_list()[1:])
            
            lstt = self.tickers15_df.loc[ind_date].to_list()
            subset_tickers = [ele.split('.')[0] for ele in lstt[1:]]
            print(subset_tickers)
            rets = np.array(self.df_merged[subset_tickers].loc[day].to_list())
            print(rets)
            
            self.rets_oneN.append(np.mean(rets))
            self.rets_gmv.append(sum(rets * ws_gmv))
            self.rets_hrp.append(sum(rets * ws_hrp))
            self.rets_rp.append(sum(rets * ws_rp))
            
        self.df_illi = pd.DataFrame({'Naive': self.rets_oneN, 'GMV': self.rets_gmv,
                                     'HRP': self.rets_hrp, 'RP':self.rets_rp})
        
        print(self.df_illi)
        
        self.df_illi.index = np.array(self.all_days)
        
        print(self.df_illi)
        
        
    def illustration(self):
        
        res_nai = lst_process(self.df_illi['Naive'].to_list())
        res_gmv = lst_process(self.df_illi['GMV'].to_list())
        res_hrp = lst_process(self.df_illi['HRP'].to_list())
        res_rp = lst_process(self.df_illi['RP'].to_list())
        
        self.df_demo = pd.DataFrame({'Naive': res_nai, 'GMV': res_gmv,
                                     'HRP': res_hrp, 'RP':res_rp})
        
        self.df_demo.index = np.array(self.all_days)
        
        self.df_demo.plot(title='P/L of each portfolio', xlabel = 'time', ylabel = 'returns', figsize=(8, 6))
        
        return 0
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
        
        
        
        
        
        
if __name__ == '__main__':
    
    ShowResults()
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        