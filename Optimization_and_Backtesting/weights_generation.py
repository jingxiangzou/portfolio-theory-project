import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import yfinance as yf
import pypfopt
from pypfopt import HRPOpt
from pypfopt import expected_returns
import cvxpy as cp
import scipy.stats
from scipy.stats import norm
from pypfopt import EfficientFrontier, objective_functions
import warnings
import datetime
from scipy.optimize import minimize
import math
import os 
TOLERANCE = 1e-10





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



class BackTest:
    
    def __init__(self, choice_file='choices.csv'):
        
        self.tickers_name = choice_file
        self.tickers15_df = pd.DataFrame(columns=np.arange(0, 15, 1))
        self.gmv_df = pd.DataFrame(columns=np.arange(0, 15, 1))
        self.hrp_df = pd.DataFrame(columns=np.arange(0, 15, 1))
        self.rp_df = pd.DataFrame(columns=np.arange(0, 15, 1))
        
        self.data()
        
        self.gmv_df.to_csv('gmv.csv')
        self.hrp_df.to_csv('hrp.csv')
        self.rp_df.to_csv('rp.csv')
        self.tickers15_df.to_csv('tickers15.csv')
        
        
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
    
    
    def main(self):
    
        return 0
        
        
        
    

if __name__ == '__main__':
        
    bt1 = BackTest()
    
    
    

        