import pandas as pd 

def analyze_data():
    file_path = './data/test4.csv' 
    data = pd.read_csv(file_path)

    data['Timestamp'] = pd.to_datetime(data['Category'], unit='ms')
    data.drop(columns=['Category'], inplace=True)
    data.set_index('Timestamp', inplace=True)
    data_resampled = data.resample('1T').mean().ffill()

    data_resampled['RollingMin_b'] = data_resampled['Flow rate'].shift(-299).rolling(window=300, min_periods=1).min()
    data_resampled['RollingMin_f'] = data_resampled['Flow rate'].rolling(window=300, min_periods=1).min()
    data_resampled['RollingMin_c'] = data_resampled['Flow rate'].rolling(window=300, min_periods=1, center=True).min()
    data_resampled['RollingMin'] = data_resampled[['RollingMin_f', 'RollingMin_b',"RollingMin_c"]].max(axis=1)
    data_resampled['RollingMin'] = data_resampled.apply(lambda row: row['Flow rate'] if row['RollingMin'] > row['Flow rate'] else row['RollingMin'], axis=1)
    
    data_resampled['RollingMin'] = data_resampled['RollingMin'].ffill()
    print(data_resampled)
    total_water_wasted = data_resampled['RollingMin'].sum()
    total_water_consumed = data_resampled['Flow rate'].sum()
    efficiency_percentage = (total_water_consumed - total_water_wasted) / total_water_consumed * 100

    return int(total_water_wasted), data_resampled, int(total_water_consumed), int(efficiency_percentage)

