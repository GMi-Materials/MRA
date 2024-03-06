# -*- coding: utf-8 -*-

author = "Gerfried Millner (GMi)"
version = "1.0.0"   
date = "06.03.2024"
email = "gerfried.millner@mcl.at "
status = "Deliver"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
from scipy import stats
import numbers
import time


def X_scaling(X_in):
    """scale input data using z-score. Binary features are not scaled to conserve their information!
    Parameters
    ----------
    X_in : df
        Dataframe of input values.
        
    Returns
    -------
    X_scal : df
        Dataframe of scaled input values.

    """
    
    X_scal = pd.DataFrame()

    for i, col in enumerate(X_in.columns):
        
        if len(X_in[col].unique()) <= 2:
            X_scal = pd.concat([X_scal,X_in[col]],axis=1)
        else:
            X_scal = pd.concat([X_scal,stats.zscore(X_in[col])],axis=1)
    
    return X_scal


def corr_col(X, res, method='pearson', list_used=[]):
    """determine feature with highest correlation with residuals.

    Parameters
    ----------
    X : df
        Dataframe of input values.
    res : list or array
        Array of residuals.
    method : str, optional
        Used correlation method. The default is 'pearson'. Other options include 'spearman' and 'kendall'.
    list_used : list, optional
        List of already identified features. The default is [].

    Returns
    -------
    col_corr : str
        Feature name that has the highest correlation with the residual and is not in list_used.

    """

    X_new = X.copy()
    X_new['Residuen'] = res
    for feat in list_used:
        X_new = X_new.drop(feat, axis=1)

    corr_ma = np.abs(X_new.corr(method=method))
    if len(corr_ma)>1:
        id_max = np.argsort(corr_ma['Residuen'])[-2]
    else:
        id_max = 0
    col_corr = corr_ma.columns[id_max]

    return col_corr

def MRA(X, y, model_fct, corr_method='pearson', tol=0.01, min_feat=2):
    """Implement feature selection method based on correlation with reiduals.

    Parameters
    ----------
    X : df
        Dataframe of input values.
    y : df
        Dataframe of target values.
    model_fct : callable
        Model function of machine learning model. Must only take X and y as an input and has the output:
            trained model, R^2 value, list or array of residuals
    corr_method : str, optional
        Used correlation method. The default is 'pearson'. Other options include 'spearman' and 'kendall'.
    tol : float, optional
        Tolearnce for determining if a model is not improving. The default is 0.01.
    min_feat : int, optional
        Minimal number of selected features. The default is 2.

    Returns
    -------
    X_n : df
        Dataframe of selected features.
    list_feat_used : list
        List of selected features.
    list_r2 : list
        List of R^2 values of each iteration.

    """
      
    # determine first feature via correlation
    Xy = X.copy()
    Xy['y'] = np.ravel(y)
    corr_XY = Xy.corr(method=corr_method)
    col_corr0 = np.abs(corr_XY['y']).sort_values(ascending=False).index[1]

    X_init = pd.DataFrame()
    X_init[col_corr0] = X[col_corr0]

    m_init, r2_init, res_init  = model_fct(X_scaling(X_init), stats.zscore(y))
        
    ### loop ###

    r2_hist = r2_init.copy()
    X_n = X_init.copy()
    res_n = res_init.copy()

    list_feat_used = [col_corr0]
    
    list_r2 = [r2_init]

    diff_hist = True

    for i in range(len(X.columns)):

        print('#'*15+' i='+str(i))
        print(list_feat_used)
        start = time.time()
        
        col_co = corr_col(X, res_n, method=corr_method, list_used=list_feat_used)
        print('col_co = '+col_co)

        list_feat_used.append(col_co)    

        X_n[col_co] = X[col_co]
        
        m_n, r2_new, res_n = model_fct(X_scaling(X_n), stats.zscore(y))

        print('r2_new = '+format(r2_new, '.5'))
        print('delta_R2 = '+str(r2_new - r2_hist))
        print('Time(s) = '+format(time.time()-start,'.2')+' s')
        
        list_r2.append(r2_new)
        
        if r2_new - r2_hist < tol and i>=min_feat and diff_hist==False:
            print('BREAK @i='+str(i)+', delta_R2 = '+str(r2_new - r2_hist))
            break
        diff_hist = True
        if r2_new - r2_hist < tol:
            diff_hist = False
            print('diff_hist = False, diff_R2 = '+str(r2_new - r2_hist))
        
        r2_hist = r2_new.copy()

    return X_n, list_feat_used, list_r2


if __name__ == '__main__':
    #### Change X and Y according to your use-case
    from sklearn.datasets import make_regression
    
    X_array, Y_array = make_regression(n_samples=1000, n_features=100, n_informative=15, noise=1, random_state=42)
    X, Y = pd.DataFrame(X_array, columns=['feat_'+str(x) for x in range(100)]), pd.DataFrame(Y_array, columns=['Target'])
    
    ###############################
    ########## Examples ###########
    ###############################
    
    #######################
    ## Linear Regression ##
    #######################
    from sklearn.linear_model import LinearRegression
    
    def LR_model(X,y):

        model_untrained = LinearRegression() 
        model = model_untrained.fit(X, np.ravel(y))
        
        predict = [float((x*np.std(y) + np.mean(y)).iloc[0]) for x in model.predict(X)]
            
        r2 = model.score(X, y)
        res = np.ravel(y) - predict

        return model, r2, res
        
    X_MRA_lr, list_feat_mra_r, list_r2_mra_lr = MRA(X, Y, LR_model, corr_method='pearson', tol=0.01, min_feat=2)

    
    ######################
    ## Ridge Regression ##
    ######################
    from sklearn.linear_model import Ridge
    
    def Ridge_model(X,y, alpha):
    
        model_untrained = Ridge(alpha=alpha)
        model = model_untrained.fit(X, np.ravel(y))
    
        predict = [float((x*np.std(y) + np.mean(y)).iloc[0]) for x in model.predict(X)]
     
        r2 = model.score(X, y)
        res = np.ravel(y) - predict
    
        return model, r2, res
    
    Ridge_model_fct = lambda X,y: Ridge_model(X,y, alpha=12.941319924606)
    
    X_MRA_ridge, list_feat_mra_ridge, list_r2_mra_ridge = MRA(X, Y, Ridge_model_fct, corr_method='pearson', tol=0.01, min_feat=2)
    
    ######################
    ######### XGB ########
    ######################
    import xgboost as xgb

    def xgb_model(X, y, par):

        dtrain = xgb.DMatrix(X, y)

        evallist = [(dtrain, 'eval'), (dtrain, 'train')]
        num_round = 100
        
        model = xgb.train(par, dtrain, num_round, evals=evallist, early_stopping_rounds=10, verbose_eval=False)
        
        predict = [float((x*np.std(y) + np.mean(y)).iloc[0]) for x in model.predict(dtrain)]

        r2 = r2_score(np.ravel(y), predict)
        res = np.ravel(y) - predict

        return model, r2, res
    
    param_MRA = {'booster': 'gbtree',
    'eta': 0.1036801101178091,
    'max_depth': 10,
    'lambda': 0.7995264114138942	,
    'alpha': 0,
    'verbosity': 0}
    
    XGB_model_fct = lambda X,y: xgb_model(X,y, par=param_MRA)
    
    X_MRA_xgb, list_feat_mra_xgb, list_r2_mra_xgb = MRA(X, Y, XGB_model_fct, corr_method='spearman', tol=0.01, min_feat=2)