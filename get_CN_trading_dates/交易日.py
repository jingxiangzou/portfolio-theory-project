# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:18:46 2023

@author: jxzou
"""

from WindPy import * 
import chinese_calendar
import pandas as pd
import numpy as np





import datetime 
 
def get_tradeday(start_str,end_str):
    start = datetime.datetime.strptime(start_str, '%Y-%m-%d') # 将字符串转换为datetime格式
    end = datetime.datetime.strptime(end_str, '%Y-%m-%d')
    # 获取指定范围内工作日列表
    lst = chinese_calendar.get_workdays(start,end)
    expt = []
    # 找出列表中的周六，周日，并添加到空列表
    for time in lst:
        if time.isoweekday() == 6 or time.isoweekday() == 7:
            expt.append(time)
    # 将周六周日排除出交易日列表
    for time in expt:
        lst.remove(time)
    date_list = [item.strftime('%Y-%m-%d') for item in lst] #列表生成式，strftime为转换日期格式
    return date_list
 
if __name__ == '__main__':
    lst = get_tradeday('2013-02-20','2023-03-01')
    
    lst1 = [datetime.datetime.strptime(ele, '%Y-%m-%d') for ele in lst]
    
    df = pd.DataFrame(np.array(lst1).T)
    
    df.index = df[0]
    
    
    
    
    
    print(df)
    
    print(type(df.iloc[0, 0]))
    
    
    
    
    
    df1 = df.resample('M').last()
    
    print(df1)
    
    df1.to_csv('last_day_of_month_trading.csv')
    
 
    
    
    
    
