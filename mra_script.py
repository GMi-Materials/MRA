# -*- coding: utf-8 -*-

author = "Gerfried Millner (GMi)"
version = "1.2.0"   
date = "21.01.2026"
email = "gerfried.millner@gmail.com"
status = "Deliver"

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy import stats
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
    
    X_scal = X_in.copy()
    
    for col in X_in.columns:
        if len(X_in[col].unique()) > 2:
            X_scal[col] = stats.zscore(X_in[col])
    
    return X_scal

def corr_col(X, res, method='pearson', list_used=None):
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
        List of already identified features. The default is None.

    Returns
    -------
    col_corr : str or None
        Feature name that has the highest correlation with the residual and is not in list_used.
        Returns None if no valid features remain.

    """
    if list_used is None:
        list_used = []

    X_new = X.copy()
    X_new['Residuen'] = res
    
    # Drop already used features
    for feat in list_used:
        if feat in X_new.columns:
            X_new = X_new.drop(feat, axis=1)

    corr_ma = np.abs(X_new.corr(method=method))
    
    # Check if there are any features left besides 'Residuen'
    if len(corr_ma) <= 1:
        return None
    
    # Get correlation with residuals, excluding 'Residuen' itself
    residual_corr = corr_ma['Residuen'].drop('Residuen')
    
    if len(residual_corr) == 0:
        return None
    
    # Remove NaN values
    residual_corr = residual_corr.dropna()
    
    if len(residual_corr) == 0:
        return None
    
    # Find feature with highest correlation
    col_corr = residual_corr.idxmax()
    
    # Check if result is NaN (shouldn't happen after dropna, but be safe)
    if pd.isna(col_corr):
        return None
    
    return col_corr

def MRA(X, y, model_fct, corr_method='pearson', tol=0.01, min_feat=2, start_features=None):
    """Implement feature selection method based on correlation with residuals.

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
        Tolerance for determining if a model is not improving. The default is 0.01.
    min_feat : int, optional
        Minimal number of selected features. The default is 2.
        Making the minimal number of features: min_feat + len(start_features)
    start_features : list, optional
        List of feature names to start with. If None, determines first feature via correlation. The default is None.

    Returns
    -------
    X_n : df
        Dataframe of selected features.
    list_feat_used : list
        List of selected features.
    list_r2 : list
        List of R^2 values of each iteration.

    """
      
    # determine initial features
    if start_features is None:
        # determine first feature via correlation
        Xy = X.copy()
        Xy['y'] = np.ravel(y)
        corr_XY = Xy.corr(method=corr_method)
        col_corr0 = np.abs(corr_XY['y']).sort_values(ascending=False).index[1]
        list_feat_used = [col_corr0]
    else:
        # use provided features
        list_feat_used = list(start_features)
    
    X_init = pd.DataFrame()
    for feat in list_feat_used:
        X_init[feat] = X[feat]

    m_init, r2_init, res_init  = model_fct(X_init, y)
        
    ### loop ###

    r2_hist = r2_init
    X_n = X_init.copy()
    res_n = res_init.copy()
    
    list_r2 = [r2_init]

    diff_hist = True

    for i in range(len(X.columns)):
        print('#'*15+' i='+str(i))
        print(list_feat_used)
        start = time.time()
        
        col_co = corr_col(X, res_n, method=corr_method, list_used=list_feat_used)
        
        # Check if no more features available
        if col_co is None or len(list_feat_used) >= X.shape[1]:
            print("No more unused features left; stopping.")
            break
            
        print('col_co = '+col_co)

        list_feat_used.append(col_co)    
        X_n[col_co] = X[col_co]
        
        m_n, r2_new, res_n = model_fct(X_n, y)

        print('r2_new = '+format(r2_new, '.5'))
        print('delta_R2 = '+format(r2_new - r2_hist, '.5'))
        print('Time(s) = '+format(time.time()-start,'.2')+' s')
        
        list_r2.append(r2_new)
        
        if r2_new - r2_hist < tol and i>=min_feat and diff_hist==False:
            print('BREAK @i='+str(i)+', delta_R2 = '+format(r2_new - r2_hist, '.5'))
            break
        diff_hist = True
        if r2_new - r2_hist < tol:
            diff_hist = False
            print('diff_hist = False, diff_R2 = '+format(r2_new - r2_hist, '.5'))
        
        r2_hist = r2_new

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
        
        X_scaled = X_scaling(X)
        y_scaled = stats.zscore(y)
        
        y_mean = np.mean(np.ravel(y))
        y_std = np.std(np.ravel(y))

        model_untrained = LinearRegression() 
        model = model_untrained.fit(X_scaled, np.ravel(y_scaled))
        
        predict = model.predict(X_scaled) * y_std + y_mean
            
        r2 = model.score(X_scaled, y_scaled)
        res = np.ravel(y) - predict

        return model, r2, res
        
    X_MRA_lr, list_feat_mra_r, list_r2_mra_lr = MRA(X, Y, LR_model, corr_method='pearson', tol=0.01, min_feat=2)

    
    ######################
    ## Ridge Regression ##
    ######################
    from sklearn.linear_model import Ridge
    
    def Ridge_model(X,y, alpha):
        
        X_scaled = X_scaling(X)
        y_scaled = stats.zscore(y)
        
        y_mean = np.mean(np.ravel(y))
        y_std = np.std(np.ravel(y))
    
        model_untrained = Ridge(alpha=alpha)
        model = model_untrained.fit(X_scaled, np.ravel(y_scaled))
    
        predict = model.predict(X_scaled) * y_std + y_mean
     
        r2 = model.score(X_scaled, y_scaled)
        res = np.ravel(y) - predict
    
        return model, r2, res
    
    Ridge_model_fct = lambda X,y: Ridge_model(X,y, alpha=12.941319924606)
    
    X_MRA_ridge, list_feat_mra_ridge, list_r2_mra_ridge = MRA(X, Y, Ridge_model_fct, corr_method='pearson', tol=0.01, min_feat=2)
    
    ######################
    ######### XGB ########
    ######################
    import xgboost as xgb

    def xgb_model(X, y, par):
        
        X_scaled = X_scaling(X)
        y_scaled = stats.zscore(y)
        
        y_mean = np.mean(np.ravel(y))
        y_std = np.std(np.ravel(y))

        dtrain = xgb.DMatrix(X_scaled, y_scaled)

        evallist = [(dtrain, 'eval'), (dtrain, 'train')]
        num_round = 100
        
        model = xgb.train(par, dtrain, num_round, evals=evallist, early_stopping_rounds=10, verbose_eval=False)
        
        predict = model.predict(dtrain) * y_std + y_mean

        r2 = r2_score(np.ravel(y_scaled), model.predict(dtrain))
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