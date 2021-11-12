from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, plot_confusion_matrix, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import missingno as msno 
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns

from copy import copy


def plot_missing_values(df, kind='matrix'):
    if kind == 'matrix': return msno.matrix(df)
    if kind == 'bar': return msno.bar(df) 
    if kind == 'heatmap': msno.heatmap(df) 

        

def knn_score(df, var_names, tgt_name, **params):
    '''
    Calculate the knn accuracy of a clustered dataset with known labels.
    '''

    df = df.copy().reset_index(drop=True)
    
    X = df[var_names]
    y = df[tgt_name]

    kfold = KFold(n_splits=len(X))

    prediction = y.copy().to_frame()
    for ndx_train, ndx_valid in kfold.split(X):
        X_train, X_valid = X.loc[ndx_train], X.loc[ndx_valid]
        y_train, y_valid = y.loc[ndx_train], y.loc[ndx_valid]

        clf = KNeighborsClassifier(**params)        
        clf.fit(X_train, y_train)
        prd = clf.predict(X_valid)
        
        prediction.loc[ndx_valid, 'Prediction'] = prd

    classes = y.value_counts().index
    conf_ma = confusion_matrix( prediction[tgt_name], prediction['Prediction'], labels=classes)
    df_coma = pd.DataFrame(conf_ma, index=classes, columns=classes)
    accuracy = accuracy_score( prediction[tgt_name], prediction['Prediction'])
    
    return accuracy, df_coma, prediction      
    

    
def sklearn_cv_classification(X, y, base_model, params={}, X_test=None, n_folds=5, seeds=None, score_func=balanced_accuracy_score):
    
    if seeds is None:
        seeds = [1]
        
    losses = []
    predic = None
    
    cv_predictions = X[[]].copy()
    cv_predictions['CV-pred'] = None
    cv_predictions['y'] = y
    
    n_models = len(seeds)*n_folds
    
    if X_test is not None: 
        predictions = X_test[[]].copy()
    
    for seed in seeds:
        
        assert isinstance(seed, int)
        
        params['seed'] = seed
        
        kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
        
        for _n, (ndx_train, ndx_valid) in enumerate( kfold.split(X, y) ):
            
            print('='*22+f' Fold {_n:3d} '+'='*22)
            
            _X_train, _X_valid = X.iloc[ndx_train], X.iloc[ndx_valid]
            _y_train, _y_valid = y[ndx_train], y[ndx_valid]
                        
            _model = base_model()
            _model.fit(_X_train, _y_train)
            
            _pred = _model.predict(_X_valid)
            _loss = score_func( _y_valid, _pred )
            
            cv_predictions.iloc[ndx_valid, 0] = _pred
                  
            print(f'Fold {_n} score: {_loss}')
            
            losses.append( _loss )
            
            if X_test is not None:
                _pred_test = _model.predict(X_test)
                predictions[f'seed-{seed}-fold-{_n}'] = _pred_test.astype(int)
                    
    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    cv_predictions = cv_predictions.astype(int)
    print(f'CV-loss {loss_mean}+/-{loss_std}')
          
    return loss_mean, loss_std, cv_predictions, predictions


def softmax(a, axis=None):
    """
    Computes exp(a)/sumexp(a); relies on scipy logsumexp implementation.
    :param a: ndarray/tensor
    :param axis: axis to sum over; default (None) sums over everything
    """
    from scipy.special import logsumexp
    lse = logsumexp(a, axis=axis)  # this reduces along axis
    if axis is not None:
        lse = np.expand_dims(lse, axis)  # restore that axis for subtraction
    return np.exp(a - lse)


def decode_prediction(df, encoder):
    return df.apply(encoder.inverse_transform)


def remove_features_with_anti_target(df, features, anti_target='is_test', target_auc=0.6):
    
    features = copy(features)
    
    auc = 1
    
    history = dict(n_features=[], auc=[], features=[])
    
    while len(features) > 0:
        anti_target = 'is_test'

        dtrain = xgb.DMatrix(df[features], df[anti_target])

        params = {'objective': 'binary:logistic',
                  'max_depth': 2,
                  }
        
        cv = xgb.cv(dtrain=dtrain, params=params, metrics=['auc'])
        
        auc = cv['train-auc-mean'].mean()
        
        history['auc'].append(auc)
        history['n_features'].append(len(features))
        history['features'].append(copy(features))
        
        if auc < target_auc: break
        
        params['metric'] = 'logloss'
        
        model = xgb.XGBClassifier(params=params, verbosity = 0)
        model.fit(df[features], df[anti_target])

        fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        fi = fi.sort_values('Importance', ascending=False).reset_index(drop=True)

        best_feature = fi.loc[0, 'Feature']
        features.remove(best_feature)
    
    if len(features) == 1: logging.warning('Target AUC could not be reached, returning last remaining feature.')
        
    return pd.DataFrame(history)


def quick_pca(df, n_components=None, labels=None, plot=True, scale=True, **plot_kws):
    df = df.copy()
    if scale: 
        scaler = StandardScaler()
        df.loc[:,:] = scaler.fit_transform(df)
    pca = PCA(n_components)
    res = pd.DataFrame(pca.fit_transform(df))
    res.columns = res.columns.values + 1
    res = res.add_prefix('PCA-')
    if labels is not None:
        res['label'] = list(labels)
    if plot: sns.pairplot(res, hue='label' if labels is not None else None,
                          height=4, **plot_kws)
    return res
    

# STOP

import re

with open(__file__, 'r') as this_file:
    for line in this_file.readlines():
        if re.search('STOP', line):
            break
        print(line, end="")    

try:
    from .torch import *
except:
    pass

