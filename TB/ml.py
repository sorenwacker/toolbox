from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, plot_confusion_matrix, mean_squared_error
from sklearn.decomposition import PCA
import missingno as msno 

def show_na(df, kind='matrix'):
    if kind == 'matrix': return msno.matrix(df)
    if kind == 'bar': return msno.bar(df) 
    if kind == 'heatmap': msno.heatmap(df) 

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
