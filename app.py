import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

scaler =pickle.load(open('scaler.pkl','rb'))

column_names = ['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
       'neighbourhood', 'latitude', 'longitude', 'room_type',
       'minimum_nights', 'number_of_reviews', 'last_review',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365']


important_features = ['availability_365',
 'latitude',
 'longitude',
 'minimum_nights',
 'neighbourhood_group_Manhattan',
 'no_of_days_from_latest_reviews_date',
 'room_type_Private room',
 'room_type_Shared room',
 'calculated_host_listings_count',
 'reviews_per_month']



@app.route('/')
def home():
    return render_template('index.html')
    
 
@app.route('/predict',methods=['POST'])
def predict():
    
    features = []
    for inp in request.form.values():
        features.append(inp)
    features.pop()
    
    test = pd.DataFrame(columns=column_names,data=np.array(features).reshape(1,-1))
    

    
    test['id'] = test['id'].astype(np.int64)
    test['name']=test['name'].astype(object)
    test['host_id']=test['host_id'].astype(np.int64)
    test['host_name']=test['host_name'].astype(object)
    test['neighbourhood_group']=test['neighbourhood_group'].astype(object)
    test['neighbourhood']=test['neighbourhood'].astype(object)
    test['latitude']=test['latitude'].astype(np.float64)
    test['longitude']=test['longitude'].astype(np.float64)
    test['room_type']=test['room_type'].astype(object)
    test['minimum_nights']=test['minimum_nights'].astype(np.int64)
    test['number_of_reviews']=test['number_of_reviews'].astype(np.int64)
    test['last_review']=test['last_review'].astype(object)
    test['reviews_per_month']=test['reviews_per_month'].astype(np.float64)
    test['calculated_host_listings_count']=test['calculated_host_listings_count'].astype(np.int64)
    test['availability_365']=test['availability_365'].astype(np.int64)
    
    
    test['last_review'] = pd.to_datetime(test.last_review) 
    
    test['no_of_days_from_latest_reviews_date']= (pd.to_datetime('2019-07-08').date()-test['last_review'][0].date()).days
    
    test.drop(['id','name','host_name','neighbourhood','last_review','host_id'],axis=1,inplace=True)
    
    test['minimum_nights'] = np.log1p(test['minimum_nights'])
    test['minimum_nights'] = np.log1p(test['minimum_nights'])
    test.number_of_reviews = np.log1p(test.number_of_reviews)
    test.reviews_per_month = np.log1p( test.reviews_per_month)
    test.reviews_per_month = np.log1p( test.reviews_per_month)
    
    if test['neighbourhood_group'][0] == 'Brooklyn':
       test['neighbourhood_group_Brooklyn']= 1 
       test['neighbourhood_group_Manhattan']=0
       test['neighbourhood_group_Queens']=0
       test['neighbourhood_group_Staten Island']=0
    
    elif test['neighbourhood_group'][0] == 'Manhattan':
         test['neighbourhood_group_Brooklyn']= 0 
         test['neighbourhood_group_Manhattan']=1
         test['neighbourhood_group_Queens']=0
         test['neighbourhood_group_Staten Island']=0
         
    elif test['neighbourhood_group'][0] == 'Queens':
         test['neighbourhood_group_Brooklyn']= 0 
         test['neighbourhood_group_Manhattan']=0
         test['neighbourhood_group_Queens']=1
         test['neighbourhood_group_Staten Island']=0
         
    elif test['neighbourhood_group'][0] == 'Staten Island':
         test['neighbourhood_group_Brooklyn']= 0 
         test['neighbourhood_group_Manhattan']=0
         test['neighbourhood_group_Queens']=0
         test['neighbourhood_group_Staten Island']=1
    
    else :
         test['neighbourhood_group_Brooklyn']= 0 
         test['neighbourhood_group_Manhattan']=0
         test['neighbourhood_group_Queens']=0
         test['neighbourhood_group_Staten Island']=0
         
         
         
    if test['room_type'][0]=='Private room':
        test['room_type_Private room']=1
        test['room_type_Shared room']=0
    
    elif test['room_type'][0]=='Shared room':
          test['room_type_Private room']=0
          test['room_type_Shared room']=1
           
    else :
        test['room_type_Private room']=0
        test['room_type_Shared room']=0
    
    
    test = test[important_features]    
    test = scaler.fit_transform(test)
    
    ypred = model.predict(test)
    
    
    
    ## result should be in inverse log1p
    output = np.exp(ypred[0])-1
    
    return render_template('index.html',prediction_text = "Price should be $ {}".format(output))




if __name__ == '__main__':
    app.run(debug=True)