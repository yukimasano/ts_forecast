# ts_forecast
Various time-series forecasting algorithms for 1-D data in python.
(For code description see below)

In particular, the algorithms implemented/compared are: AR(I), Ridge Regression, Lasso Regression, RandomForestRegressor (those three use sklearn) and LSTM (using Keras). Hyperparameters are only optimized via gridsearch at the moment.


# Data
Here I explore some time-series prediction algorithms on four different, real data sets: 
* monthly numbers of observed sunspots
* time-series of a chaotic dynamical system (Mackey-Glass)
* atmospheric CO2 concentration.

# Results

## Sunspots 91 x 12-step forecasts

<img src="https://user-images.githubusercontent.com/29401818/32951524-230d4cc2-cba2-11e7-91fc-8c61f4660e98.png" height="300">

## Mackey-Glass chaotic dynamics

### 200 x 1-step forecasts

<img src="https://user-images.githubusercontent.com/29401818/32951521-22e5ccf6-cba2-11e7-8e40-6d15476d4bd1.png" height="300">

### 1 x 200-step forecast

<img src="https://user-images.githubusercontent.com/29401818/32951522-22fa043c-cba2-11e7-8bcc-4fa80bead88b.png" height="300">

## CO2 1 x 100-step forecast

<img src="https://user-images.githubusercontent.com/29401818/32951520-22d1ccba-cba2-11e7-8a3e-01044e2da038.png" height="300">

# (Mini-)Conclusion

I find that the methods' performance is highly dependent on the data and the assumptions and problem specific knowledge are thus vital for achieving high performance. 

# Code
`time_series_1.py`: contains all the forecasting functions
`ts_forecast_plots.ipynb`: explores the above functions on the different datasets for 1-step and multi-step forecasts.
