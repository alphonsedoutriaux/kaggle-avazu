import warnings
import numpy as np
import pandas as pd
import random
import gzip
import io
import datetime

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import date, datetime
from xgboost import XGBClassifier

# 1. Preprocessing

# 1.1 Randomized line extraction

line_count = 40428968
n = 100000
skip = sorted(random.sample(range(1,line_count + 1), line_count - n))
df = pd.read_csv("train.gz", skiprows=skip, index_col=0)

# 1.3 Preprocessing

df["weekday"] = df["hour"].apply(lambda x: datetime.strptime(str(x), '%y%m%d%H').weekday())
df["hour"] = df["hour"].apply(lambda x: int(str(x)[-2:]))
df["area"] = df["C15"] * df["C16"]

# Separation X / y

y = df[['click']]
X = df[['C1', 'hour', 'banner_pos', 'site_category', 'app_category', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 
           'C19', 'C20','C21', 'area', 'weekday', 'User_freq']]

# C20 column preparation

X['C20'] = X['C20'].replace(-1, np.nan)
X['C20'] = X['C20'].replace(np.nan, X['C20'].median())

# One hot encoding with get_dummies

columns_to_encode = ['device_type', 'device_conn_type', 'site_category', 'app_category', 'banner_pos', 'C18']
df_full_columns = pd.read_csv("train.gz", usecols=columns_to_encode)
for col in columns_to_encode:
    X[col] = X[col].astype('category', categories = df_full_columns[col].unique().tolist())
X = pd.get_dummies(X, columns=columns_to_encode, prefix = columns_to_encode)

# 1.4. Folding

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, shuffle=False)

# 2. LogisticRegression (a gridsearch was implemented to determine the hyperparameters)

lr = LogisticRegression(penalty='l1',
                        solver='liblinear',
                        C=0.1,
                        verbose=2)
lr.fit(X,y)

# 3. XGBoost (a gridsearch was implemented to determine the hyperparameters)

xgb = XGBClassifier(verbose=2,
                    n_jobs=-1,
                    max_depth=4, 
                    learning_rate=0.2,
                    colsample_bytree=0.9,
                    n_estimators=600,
                    reg_alpha=1,
                    reg_lambda=1, 
                    objective='binary:logistic',
                    booster='gbtree')
xgb.fit(X,y)

# ## 4. RandomForests

rf = RandomForestClassifier(n_jobs=-1)
rf.fit(X,y)

# ## 5. Blending

lr_preds = lr.predict_proba(X_test)
xgb_preds = xgb.predict_proba(X_test)
rf_preds = rf.predict_proba(X_test)

preds_table = pd.DataFrame({"LR":lr_preds[:,1], "XGBoost":xgb_preds[:,1], "RandomForests":rf_preds[:,1]}, index=X_test.index)

# We use a LogisiticRegression to determine how to blend the models predictions

lr_blending = LogisticRegression(penalty='l1',
                                 solver='liblinear',
                                 C=0.5,
                                 verbose=2)
lr_blending.fit(preds_table, y_test)

# 5. Predictions on test.gz

test = pd.read_csv("test.gz")

test["weekday"] = test["hour"].apply(lambda x: datetime.strptime(str(x), '%y%m%d%H').weekday())
test["hour"] = test["hour"].apply(lambda x: int(str(x)[-2:]))
test["area"] = test["C15"] * test["C16"]

test['User_freq'] = test['device_id'] + test['device_ip'] + test['device_model'] #on crée une feature user 
values = test['User_freq'].value_counts() # on remplace la valeur de user par sa fréquence d'apparition
test['User_freq'] = test['User_freq'].apply(lambda row: values[row])

test = test[['C1', 'hour', 'banner_pos', 'site_category', 'app_category', 'device_type', 'device_conn_type', 'C14','C15', 'C16', 'C17', 'C18', 
           'C19', 'C20','C21', 'area', 'weekday', 'User_freq']]

test['C20'] = test['C20'].replace(-1, np.nan)
test['C20'] = test['C20'].replace(np.nan, X['C20'].median())

for col in columns_to_encode:
    test[col] = test[col].astype('category',categories = df_full_columns[col].unique().tolist())

test = pd.get_dummies(test, columns=columns_to_encode, prefix = columns_to_encode)

lr_preds_test = lr.predict_proba(test)
xgb_preds_test = xgb.predict_proba(test)
rf_preds_test = rf.predict_proba(test)

preds_table_final = pd.DataFrame({"LR":lr_preds_test[:,1], "XGBoost":xgb_preds_test[:,1], "RandomForests":rf_preds_test[:,1]}, index=test.index)

# 5.2. Export for the evaluation on kaggle.com

preds = lr_blending.predict_proba(preds_table_final)[:,1]
submission_file = pd.read_csv("sampleSubmission.gz")
submission_file['click'] = preds
submission_file.to_csv(path_or_buf= './submission_files/preds_' + datetime.now().strftime("%d%m%Y-%H%M%S") + '.gz', index = False, sep = ',', compression='gzip')

