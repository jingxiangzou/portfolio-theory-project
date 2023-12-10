import numpy as np
import pandas as pd
import os
from pypfopt import HRPOpt
import cvxpy as cp
import scipy.stats
from scipy.stats import norm
from pypfopt import EfficientFrontier, objective_functions
import warnings
import datetime
from scipy.optimize import minimize
import math
import os 
import datetime as dt
TOLERANCE = 1e-10



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

def _allocation_risk(weights, covariances):
    """
    :param weights: (n*n) numpy.matrix eg: cov1 = np.matrix('1 2 3 ; 4 5 6 ; 1 6 3')
    :param covariances: (n*1) numpy.matrix
    :return: a double value
    """
    # We calculate the risk of the weights distribution
    portfolio_risk = np.sqrt(np.dot(np.dot(weights, covariances), weights.T))

    # It returns the risk of the weights distribution
    return portfolio_risk


def _assets_risk_contribution_to_allocation_risk(weights, covariances):
    """
    :param weights: (n*n) numpy.matrix eg: cov1 = np.matrix('1 2 3 ; 4 5 6 ; 1 6 3')
    :param covariances: (n*1) numpy.matrix
    :return: a n * 1 matrix
    """
    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    rc_array = np.squeeze(np.asarray(np.dot(covariances, weights.T)))
    wl = []
    for i in range(len(rc_array)):
        wl.append(weights[i] * (portfolio_risk ** (-1)) * rc_array[i])

    # We calculate the contribution of each asset to the risk of the weights
    # distribution

    # np.multiply is the element-wise multiplication of two arrays
    # It returns the contribution of each asset to the risk of the weights
    # distribution
    return np.array(wl)


def _risk_budget_objective_error(weights, args):
    """
    :param weights: (n*1) np.matrix
    :param args[0] : covariances: (n*n) mp.matrix
    :param args[1] : assets_risk_budget: np.array ,
        The desired contribution of each asset to the portfolio risk
        whose elements sums up to 1, is equal to 0.1 in a 10 asset risk parity
    :return : a double value, is the summation of squared error of RC_i for asset_i (i from 1 to 10)
    """
    covariances = args[0]
    assets_risk_budget = args[1]

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = \
        _assets_risk_contribution_to_allocation_risk(weights, covariances)

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = 0.1 * portfolio_risk

    # np.asmatrix yields a 1 * n matrix
    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = \
        sum([np.square(x - assets_risk_target) for x in assets_risk_contribution])

    # It returns the calculated error
    return error


def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):

    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})

    # Optimisation process in scipy
    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})

    # Recover the weights from the optimised object
    weights = optimize_result.x

    # It returns the optimised weights
    return weights


class Optimization:
    
    def __init__(self, covariance_matrix, expected_returns=None):
        
        self.covmat = covariance_matrix 
        # covmat should be a pandas DataFrame with column_names and index being the same tickers
        self.mu = expected_returns
        self.main()
        
    def rp_op(self): # risk parity optimization 
        
        initial_ws = [1/15] * 15
        self.rp_weights = _get_risk_parity_weights(self.covmat, initial_ws, initial_ws)
        
        return 0
        
    def hrp_op(self): # HRP optimization 
    
        hrp = HRPOpt(cov_matrix=self.covmat)
        hrp.optimize()
        self.hrp_weights = pd.Series(hrp.clean_weights()).to_numpy()
        
        return 0
    
    def gmv_op(self): # GMV optimization

        w = cp.Variable(self.covmat.shape[0])
        # Defining risk objective
        risk = cp.quad_form(w, self.covmat)
        objective = cp.Minimize(risk)
        # Budget and weights constraints
        constraints = [cp.sum(w) == 1, 
                       w >= 0]
        # Solver 
        prob = cp.Problem(objective, constraints)
        prob.solve()
    
        self.gmv_weights = np.round(w.value, 5)
    
        return 0
       
    def main(self):
        
        self.hrp_op()
        self.gmv_op()
        self.rp_op()

        return 0


class AdjCovBackTest:
    
    def __init__(self, choice_file='new_choices.csv'):
        
        self.tickers_name = choice_file
        self.tickers15_df = pd.DataFrame(columns=np.arange(0, 15, 1))
        self.gmv_df = pd.DataFrame(columns=np.arange(0, 15, 1))
        self.hrp_df = pd.DataFrame(columns=np.arange(0, 15, 1))
        self.rp_df = pd.DataFrame(columns=np.arange(0, 15, 1))
        self.szcz = pd.read_csv('szcz.csv')
        
        self.data()
        self.time()
        self.main()
        self.illustration()
        
    def data(self):
        
        # get the filenames of the big_covmats
        
        file_path = "C:\\Users\\jxzou\\OneDrive\\桌面\\730 lyasoff project\\StockCov"
        file_p = file_path + "\\"
        k_list = []
        for i, j, k in os.walk(file_path):
            
            k_list.append(k)
             
        bigcvmt_files = k_list[0] # the list of big_cov file names 

        # we proceed to get the tickers of each month
        tickers_df = pd.read_csv(self.tickers_name)
        n_months = len(tickers_df)
        
        
        for month_ele in range(n_months):
            
            current_names = tickers_df.iloc[month_ele, 1:].to_list()
            current_tickers = [ele.split('.')[1] for ele in current_names]
            current_bigcvmt = pd.read_csv(file_p + bigcvmt_files[month_ele])
            new_colnames = [ele.split('.')[0] for ele in current_bigcvmt.columns]
            
            indexes = []
            for ele in current_tickers:
                
                if ele in new_colnames:
                    indexes.append(new_colnames.index(ele))
                else:
                    pass
                
            portfolio_covmat = current_bigcvmt.iloc[indexes, indexes]
            
            # now scan for NAN along both columns and rows
            row_nan_ind = []
            col_nan_ind = []
            
            for i in range(len(portfolio_covmat)):
                for j in range(len(portfolio_covmat.columns)):
                    
                    if portfolio_covmat.iloc[i, j] * 0 != 0: # NAN detector, as we know NAN * 0 = NAN 
                        row_nan_ind.append(i)
                        col_nan_ind.append(j)

            s_row = set(row_nan_ind)
            s_col = set(col_nan_ind)
            ind_lst = list(set(np.arange(0, len(portfolio_covmat), 1)).difference(s_row.union(s_col)))
          
            portfolio_covmat = portfolio_covmat.iloc[ind_lst, ind_lst]
            portfolio_covmat = portfolio_covmat.iloc[:15, :15]
            portfolio_covmat.index = portfolio_covmat.columns
            
            print(portfolio_covmat)
            
            self.tickers15_df.loc[month_ele] = portfolio_covmat.columns
                        
            op = Optimization(portfolio_covmat)
            self.gmv_df.loc[month_ele] = op.gmv_weights
            self.hrp_df.loc[month_ele] = op.hrp_weights
            self.rp_df.loc[month_ele] = op.rp_weights
            
            print(month_ele)
            print(bigcvmt_files[month_ele])
      
        return 0
    
    def time(self):
                
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
        
        self.ret_slice_use = [] # slice of returns for estimation of covmat
        
        
        hadex = 0
        for day in self.all_days:
            
            print(day)
            ind_num = find_date_index(day, change_days[1:])
            ind_date = change_days[ind_num]
            
            # weights that day
            ws_gmv = np.array(self.gmv_df.loc[ind_date].to_list()[1:])
            ws_hrp = np.array(self.hrp_df.loc[ind_date].to_list()[1:])
            ws_rp = np.array(self.rp_df.loc[ind_date].to_list()[1:])
            
            lstt = self.tickers15_df.loc[ind_date].to_list()
            subset_tickers = [ele.split('.')[0] for ele in lstt[1:]]
            
            rets = np.array(self.df_merged[subset_tickers].loc[day].to_list())
            
            for i in range(len(rets)):
                if rets[i] * 0 != 0:
                    rets[i] = 0
                if rets[i] > 10:
                    rets[i] = 0
                    
            self.rets_oneN.append(np.mean(rets))
            self.rets_gmv.append(sum(rets * ws_gmv))
            self.rets_hrp.append(sum(rets * ws_hrp))
            self.rets_rp.append(sum(rets * ws_rp))
            
            hadex += 1
            
        self.df_illi = pd.DataFrame({'Naive': self.rets_oneN, 'GMV': self.rets_gmv,
                                     'HRP': self.rets_hrp, 'RP':self.rets_rp})
            
        self.df_illi.index = np.array(self.all_days)
        
        
    def illustration(self):
        
        df_sz = self.szcz
        df_sz.index = df_sz['eob']
        df_sz = df_sz.iloc[:, :3]
        df1 = df_sz.pct_change()
        start = dt.datetime(2013, 3, 1)
        end = dt.datetime(2023, 2, 27)
        df1['Dates'] = [dt.datetime.strptime(ele, "%Y-%m-%d %H:%M:%S+08:00") for ele in np.array(df1.index)]
        df1.index = df1['Dates']
        df1 = df1[df1.index >= start]
        df1 = df1[df1.index <= end]
        
        print(df1.head())
        print(df1.tail())
        
        r_nai = self.df_illi['Naive'].to_list()
        r_gmv = self.df_illi['GMV'].to_list()
        r_hrp = self.df_illi['HRP'].to_list()
        r_rp  = self.df_illi['RP'].to_list()
        r_sz  = df1['close'].to_list()
        
        self.df_every_rets = pd.DataFrame({'Naive': r_nai, 'GMV': r_gmv,
                                     'HRP': r_hrp, 'RP':r_rp, 'SZCZ':r_sz})
        self.df_every_rets.index = np.array(self.all_days)
        self.df_every_rets.plot(title='Daily Returns of each portfolio', xlabel = 'time', ylabel = 'returns', figsize=(8, 6))
        self.df_every_rets.to_csv('daily returns.csv')
        
        
        res_nai = lst_process(self.df_illi['Naive'].to_list())
        res_gmv = lst_process(self.df_illi['GMV'].to_list())
        res_hrp = lst_process(self.df_illi['HRP'].to_list())
        res_rp = lst_process(self.df_illi['RP'].to_list())
        res_sz = lst_process(df1['close'].to_list())
        
        self.df_demo = pd.DataFrame({'Naive': res_nai, 'GMV': res_gmv,
                                     'HRP': res_hrp, 'RP':res_rp, 'SZCZ':res_sz})
        
        self.df_demo.index = np.array(self.all_days)
        
        self.df_demo.plot(title='P/L of each portfolio', xlabel = 'time', ylabel = 'returns', figsize=(8, 6))
        
        SR_nai = np.mean(r_nai) / np.std(r_nai) * np.sqrt(252)
        SR_gmv = np.mean(r_gmv) / np.std(r_gmv) * np.sqrt(252)
        SR_hrp = np.mean(r_hrp) / np.std(r_hrp) * np.sqrt(252)
        SR_rp  = np.mean(r_rp) / np.std(r_rp) * np.sqrt(252)
        SR_sz = np.mean(r_sz) / np.std(r_sz) * np.sqrt(252)
        
        self.df_demo.to_csv('absolute return since inception.csv')
        
        print('NAI', SR_nai, np.mean(r_nai) * 252, np.std(r_nai) * np.sqrt(252))
        print('GMV', SR_gmv, np.mean(r_gmv) * 252, np.std(r_gmv) * np.sqrt(252))
        print('HRP', SR_hrp, np.mean(r_hrp) * 252, np.std(r_hrp) * np.sqrt(252))
        print('RP', SR_rp, np.mean(r_rp) * 252, np.std(r_rp) * np.sqrt(252))
        print('SZ', SR_sz, np.mean(r_sz) * 252, np.std(r_sz) * np.sqrt(252))

        return 0
        
    

if __name__ == '__main__':
        
    bt1 = AdjCovBackTest()
    
    
    

        