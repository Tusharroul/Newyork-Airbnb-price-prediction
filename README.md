# Newyork Airbnb price prediction and Building Model
Airbnb is an online-based marketing company that connects people
looking for accommodation (Airbnb guests) to people looking to rent
their properties (Airbnb hosts) on a short-term or long-term basis. So,
this company has all the data related to the Neighbourhood, room type,
host name, price etc. This project focuses on the gleaning patterns and
other relevant information about Airbnb listings in NYC. To be more
specific, here we have to investigate that how do prices vary with respect
neighbourhoods, room type and Number of reviews etc. The task is to
train a model that can predict the price of the accommodation as per
different attributes. Basically, it’s Regression problem.

Tools / Skills used
1.	Python Programming
2.	Jupyter notebook
3.	Pandas
4.	Numpy
5.	Matplotlib
6.	Seaborn
7.	Exploratory Data Analysis
8.	Data Engineering
9.	Data Visualization
10.	Machine Learning

Data description

Hotel id                                                        
Name of the hotel
Host id
Host name
Neighborhood group
Neighborhood
Latitude & Longitude
Room Type Price
Minimum nights
Number of reviews
Last review
Reviews per month
Calculated host listing count
Availability

User Guide
After installing the libraries in the dependencies section, you can simply run all of the cells in the notebook. This should generate all of the plots here.

Building model
I have used Linear regression, Random forest, Lasso, Ridge, LGBMRegressor on this model. At last I got better result in LGBMRegressor.
So I made my model by the use of LGBMRegressor in Flask.


