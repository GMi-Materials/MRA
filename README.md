# MRA
Multiple Regression Analysis (MRA)

In this feature selection method, features are gradually added until there is minimal change in prediction accuracy. Initially, the feature with the highest linear correlation, determined by the Pearson or other correlation coefficient, is selected. A Machine Learning model is then trained with this feature as the sole input. Following this, the $R^2$ value of the model and the residuals (the difference between true and predicted values) are computed. The feature exhibiting the highest correlation coefficient with the residuals is subsequently integrated into the model's input data. The process iterates with additional feature incorporation and model training until there is no notable improvement in model performance for two consecutive iterations, marking the end of the loop.

It is recommended to choose a linear correlation coefficient when using a linear model (e.g. linear regression) and a non-linear one, when using more sophisticated models that can capture non-linear relationships.

The promising implementation of the method is shown in the paper from Millner et al.: 

G. Millner, M. Mücke, L. Romaner, D. Scheiber, Tensile strength prediction of steel sheets: An insight into data-driven models, dimensionality reduction, and feature importance, Modelling and Simulation in Materials Science and Engineering (2024). \
URL: http://iopscience.iop.org/article/10.1088/1361-651X/ad6fc0

## Dependencies
-  python 3
-  numpy
-  pandas
-  scikit-learn
-  scipy

## Example Usage
Note that X and y are supposed to be normalized using z-score normalization in this examples! If you choose another standardization technique (or non at all), please adapt the code accordingly.
### Linear Regression 
```python
from sklearn.linear_model import LinearRegression

def LR_model(X,y):

    model_untrained = LinearRegression() 
    model = model_untrained.fit(X, np.ravel(y))
    
    predict = [float((x*np.std(y) + np.mean(y)).iloc[0]) for x in model.predict(X)]
        
    r2 = model.score(X, y)
    res = np.ravel(y) - predict

    return model, r2, res
    
X_MRA_lr, list_feat_mra_r, list_r2_mra_lr = MRA(X, Y, LR_model, corr_method='pearson', tol=0.01, min_feat=2)
```

### Ridge Regression 
```python
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
```

### XGB 
```python
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
```
