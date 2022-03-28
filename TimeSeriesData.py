
# load dataset using read_csv()
from pandas import read_csv
ORGseries = read_csv('C:\\Users\\erayt\\Desktop\\Time Series Analysis With Python\\AMD_Data.csv', header=0, index_col=0, parse_dates=True, squeeze=True) # change the company code as needed
print(type(ORGseries))
print(ORGseries.head())

# a description of dataset
print()
print('DESCRIPTION OF THE TIME SERIES DATA:')
print(ORGseries.describe())

# drop unnecessary columns, change the dropped columns as needed
del ORGseries['High']
del ORGseries['Low']
del ORGseries['Open']
del ORGseries['Close']
#del ORGseries['Adj Close']
del ORGseries['Volume']
del ORGseries['Name']

series = ORGseries

# create a line plot
print()
print('#LINE PLOT')
from matplotlib import pyplot
series.plot()
pyplot.title('Line Plot')
pyplot.show()

# create a dot plot
print()
print('#DOT PLOT')
series.plot(style='k.')
pyplot.title('Dot Plot')
pyplot.show()

# create a histogram plot
print()
print('#HISTOGRAM')
series.hist()
pyplot.title('Histogram')
pyplot.show()

# create a density plot
print()
print('#DENSITY PLOT')
series.plot(kind='kde')
pyplot.title('Density Plot')
pyplot.show()

# create a scatter plot
# Time series modeling assumes a relationship between an observation and the previous observation.
# A useful type of plot to explore the relationship between each observation and a lag of that observation is called the scatter plot.
    # If the points cluster along a diagonal line from the bottom-left to the top-right of the plot, it suggests a positive correlation relationship.
    # If the points cluster along a diagonal line from the top-left to the bottom-right, it suggests a negative correlation relationship.
    # Either relationship is good as they can be modeled.
from pandas.plotting import lag_plot
print()
print('#SCATTER PLOT')
lag_plot(series)
pyplot.title('Lag Scatter Plot')
pyplot.show()

# create an autocorrelation plot
# We can quantify the strength and type of relationship between observations and their lags.
    # results in a number between -1 and 1. The sign of this number indicates a negative or positive correlation respectively.
print()
print('#AUTOCORRELATION PLOT')
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)
pyplot.title('Autocorrelation Plot 1')
pyplot.show()

# The ACF of trended time series tend to have positive values that slowly decrease as the lags increase.
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=100)
pyplot.title('Autocorrelation Plot 2')
pyplot.show()

# square root transform a time series
from pandas import read_csv
from pandas import DataFrame
from numpy import sqrt
from matplotlib import pyplot
dataframe = DataFrame(series.values)
dataframe.columns = ['Adj Close']
dataframe['Adj Close'] = sqrt(dataframe['Adj Close'])
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.title('sqrt() Transform Applied')
pyplot.plot(dataframe['Adj Close'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['Adj Close'])
pyplot.show()

# log transform a time series
from numpy import log
dataframe = DataFrame(series.values)
dataframe.columns = ['Adj Close']
dataframe['Adj Close'] = log(dataframe['Adj Close'])
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.title('log Transform Applied')
pyplot.plot(dataframe['Adj Close'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['Adj Close'])
pyplot.show()

# EXPLORATORY ANALYSIS
print()
print('#EXPLORATORY ANALYSIS')
# Decomposing Time Series Data Components
    # Systematic: level: The average value in the series, trend: The increasing or decreasing value in the series, seasonality: The repeating short-term cycle in the series
    # Non-Systematic: noise: The random variation in the series

# Multiplicative Decomposition
    # An additive model is linear where changes over time are consistently made by the same amount. A linear trend is a straight line. 
    # A linear seasonality has the same frequency (width of cycles) and amplitude (height of cycles).
    # Hence, A multiplicative model is nonlinear, such as quadratic or exponential is suitable for our timeseries. Changes increase or decrease over time. 
    # A nonlinear trend is a curved line. A nonlinear seasonality has an increasing or decreasing frequency and/or amplitude over time.
print()
print('#MULTIPLICATIVE DECOMPOSITION')
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(series, model='multiplicative', period = 30)
print(result.trend)
print(result.seasonal)
print(result.resid) # the time series after the trend, and seasonal components are removed
print(result.observed)
result.plot()
pyplot.title('Multiplicative Decomposition')
pyplot.show()

# Hodrick–Prescott decomposition
# is a mathematical tool used in macroeconomics, especially in real business cycle theory, to remove the cyclical component of a time series from raw data. 
# It is used to obtain a smoothed-curve representation of a time series, one that is more sensitive to long-term than to short-term fluctuations.
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR

import matplotlib.pyplot as plt
import seaborn as sns
cycle, trend = sm.tsa.filters.hpfilter(series, 50)
fig, ax = plt.subplots(3,1)
ax[0].plot(series)
ax[0].set_title('Price')
ax[1].plot(trend)
ax[1].set_title('Trend')
ax[2].plot(cycle)
ax[2].set_title('Cycle')
plt.show()

# Trends
    # Systematic change in a time series that does not appear to be periodic is known as a trend.
    # Deterministic Trends: These are trends that consistently increase or decrease.
    # Stochastic Trends: These are trends that increase and decrease inconsistently.
    # Global Trends: These are trends that apply to the whole time series.
    # Local Trends: These are trends that apply to parts or subsequences of a time series.
print()
print('#TRENDS')
# Detrend by Model Fitting
    # Detrend by Differencing is not used because it works well for data with a linear trend
# use a linear model to detrend a time series
from pandas import datetime
from sklearn.linear_model import LinearRegression
import numpy
series = ORGseries
# fit linear model
X = [i for i in range(0, len(series))]
X = numpy.reshape(X, (len(X), 1))
y = series.values
model = LinearRegression()
model.fit(X, y)
# calculate trend
trend = model.predict(X)
# plot trend
pyplot.plot(y)
pyplot.plot(trend)
pyplot.title('Trend')
pyplot.show()
# detrend
detrended = [y[i]-trend[i] for i in range(0, len(series))]
# plot detrended
pyplot.plot(detrended)
pyplot.title('Detrended')
pyplot.show()

# Seasonality
    # A repeating pattern within each year is known as seasonal variation, although the term is applied more generally to repeating patterns within any fixed period.
print()
print('#SEASONALITY')

# Seasonal adjustment with Modeling
# model seasonality with a polynomial model
from numpy import polyfit
series = ORGseries
# fit polynomial: x^2*b1 + x*b2 + ... + bn
X = [i%365 for i in range(0, len(series))]
y = series.values
degree = 4
coef = polyfit(X, y, degree)
print('Coefficients: %s' % coef)
# create curve
curve = list()
for i in range(len(X)):
    value = coef[-1]
    for d in range(degree):
        value += X[i]**(degree-d) * coef[d]
    curve.append(value)
# plot curve over original data
pyplot.plot(series.values)
pyplot.plot(curve, color='red', linewidth=3)
pyplot.show()

# Stationarity
    # Non-stationary time series show seasonal effects, trends. Summary statistics like the mean and variance do change over time, providing a drift in the concepts a model may try to capture. 
    # Classical time series analysis and forecasting methods are concerned with making non-stationary time series data stationary by identifying and removing trends and removing seasonal effects.
print()
print('#STATIONARITY')

# A quick check for stationarity, Summary Statistics
# calculate statistics of partitioned time series data
series = ORGseries
X = series.values
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print()
print('Mean, Variance check in partitioned time series data')
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

# Augmented Dickey-Fuller Test
# calculate stationarity test of time series data
    # The more negative this statistic, the more likely we are to reject the null hypothesis (we have a stationary dataset). 
    # This means that the process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure
from statsmodels.tsa.stattools import adfuller
series = ORGseries
X = series.values
result = adfuller(X)
print()
print('Augmented Dickey-Fuller Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
