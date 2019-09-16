import tensorflow as tf
import numpy as np
from tensorflow import keras

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1, np.cos(season_time * 7 * np.pi), 1/np.exp(5 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4*365+1, dtype='float32')
baseline = 10
series = trend(time, 0.1)
amplitude = 40
slope = 0.01
noise_level = 2

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)


# We have the time series, split to start forecasting
split_time = 1100
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


# Naive Forecast
naive_forecast = series[split_time-1:-1]
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())


# Try a moving average
def moving_average_forecast(series, window_size):
    """
    Forecasts the mean of the last few values
    If window_size=1, then this is equicalent to naive forecast
    """
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time+window_size].mean())

    return np.array(forecast)


moving_avg = moving_average_forecast(series, 30)[split_time-30:]
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())

# It is worse than naive forecast
# Since it doesn't anticipate trend or seasonality
# -> Remove them by using differencing
diff_series = (series[365:] - series[:-365]) 
diff_time = time[365:]

diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time-365-50:]

# Bring back the trend and sesonality
diff_moving_avg_plus_past = series[split_time-365:-365] + diff_moving_avg
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())

# It is better than naive forecast
# But it is too random since we are just adding past values, which are noisy
# -> Use a moving average on past values to remove some of the noise
diff_moving_avg_plus_smooth_past  = moving_average_forecast(series[split_time - 370: -360], 10) + diff_moving_avg
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())
