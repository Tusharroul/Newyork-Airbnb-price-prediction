import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import statsmodels.graphics.gofplots as sm
import scipy.stats as stats
import datetime
import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFE
import random




# os.chdir("F:\Board infinity\Capstone project\AB_NYC")
os.chdir("C:\\Users\\hp\\Documents\\R_and_PY_programming\\tushar\\binf_capstone_projects\\project1_airbnb\\Airbnb_deployment")

df = pd.read_csv("AB_NYC_2019.csv")
df_temp = df.copy()

df_temp.room_type.unique()


df.fillna({'reviews_per_month':0}, inplace=True)

df['last_review'] =pd.to_datetime(df.last_review) 
df.sort_values(by='last_review', ascending=False)



d=[]
for i in range(len(df)):
    date1=df['last_review'][48852]
    date2=df['last_review'][i]
    if date2 is not np.nan:
        delta =  (date1 - date2).days
        d.append(delta)
    else:
        d.append(np.nan)

df['no_of_days_from_latest_reviews_date']=d
df.head()


df.no_of_days_from_latest_reviews_date.fillna(df.no_of_days_from_latest_reviews_date.max(),inplace=True)

df=df.drop(['id','name','host_name','neighbourhood','last_review','host_id'],axis=1)


# using log transformation to treat skewness and outliers
df['minimum_nights'] = np.log1p(df['minimum_nights'])
df['minimum_nights'] = np.log1p(df['minimum_nights'])
df.number_of_reviews = np.log1p(df.number_of_reviews)
df.reviews_per_month = np.log1p( df.reviews_per_month)
df.reviews_per_month = np.log1p( df.reviews_per_month)
df['price']= np.log1p(df['price'])

df_Airbnb=df.copy()

df_Airbnb=pd.get_dummies(df_Airbnb,drop_first=True)

X=df_Airbnb.copy().drop('price',axis=1)
y=df_Airbnb['price'].copy()

rf_reg = RandomForestRegressor(n_estimators=100,max_depth=6)
rfe = RFE(rf_reg,8)

scaler = StandardScaler()
X = scaler.fit_transform(X)
fit = rfe.fit(X,y)


# feature sorted by their rank
X_pre=df_Airbnb.copy().drop('price',axis=1)
names=X_pre.columns
sort =sorted( zip(map(lambda x : round(x,4),rfe.ranking_),names) )


important_features = []
for i in range(10):
    important_features.append(sort[i][1])
    
    
X=df_Airbnb.copy().drop('price',axis=1)
y=df_Airbnb['price'].copy()
X = X[important_features]


scaler = StandardScaler()
X = scaler.fit_transform(X)

scaler

pickle.dump(scaler,open('scaler.pkl','wb'))
model = LGBMRegressor(max_depth=4,
                                   n_estimators=1000,
                                   learning_rate=0.02,
                                   num_leaves=20,
                                   metric=['l2', 'auc'],
                                   colsample_bytree=0.8,
                                   objective='regression', 
                                   n_jobs=-1,
                                  num_iterations= 1000,)
model.fit(X,y)

pickle.dump(model,open('model.pkl','wb'))


