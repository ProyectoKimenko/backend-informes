import pandas as pd 

def analyze_data():
    file_path = './data/test.csv' 
    data = pd.read_csv(file_path)

    data['Timestamp'] = pd.to_datetime(data['Category'], unit='ms')
    data.drop(columns=['Category'], inplace=True)
    data.set_index('Timestamp', inplace=True)

    data_resampled = data.resample('1T').mean()
    data_resampled['RollingMin'] = data_resampled['Flow rate'].rolling(window=50, min_periods=1).min()
    total_water_wasted = data_resampled['RollingMin'].sum()

    return total_water_wasted, data_resampled

