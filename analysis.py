import pandas as pd 

def analyze_data(window_size = 60):
    file_path = './data/test4.csv' 
    data = pd.read_csv(file_path)

    data['Timestamp'] = pd.to_datetime(data['Category'], unit='ms')
    data.drop(columns=['Category'], inplace=True)
    data.set_index('Timestamp', inplace=True)
    data_resampled = data.resample('1T').mean().ffill()

    data_resampled['RollingMin_b'] = data_resampled['Flow rate'].shift(-window_size).rolling(window=window_size, min_periods=1).min()
    data_resampled['RollingMin_f'] = data_resampled['Flow rate'].rolling(window=window_size, min_periods=1).min()
    data_resampled['RollingMin_c'] = data_resampled['Flow rate'].rolling(window=window_size, min_periods=1, center=True).min()
    data_resampled['RollingMin'] = data_resampled[['RollingMin_f', 'RollingMin_b',"RollingMin_c"]].max(axis=1)
    data_resampled['RollingMin'] = data_resampled.apply(lambda row: row['Flow rate'] if row['RollingMin'] > row['Flow rate'] else row['RollingMin'], axis=1)
    
    data_resampled['RollingMin'] = data_resampled['RollingMin'].ffill()
    print(data_resampled)
    total_water_wasted = data_resampled['RollingMin'].sum()
    total_water_consumed = data_resampled['Flow rate'].sum()
    efficiency_percentage = 100 -(total_water_consumed - total_water_wasted) / total_water_consumed * 100

    return int(total_water_wasted), data_resampled, int(total_water_consumed), int(efficiency_percentage)

# import pandas as pd

# def analyze_data(window_ranges):
#     """
#     Analyze the water flow data with dynamic window sizes based on time ranges.
    
#     Parameters:
#     - window_ranges: A list of dictionaries with 'window_size', 'timestamp_start', and 'timestamp_end'
    
#     Example:
#     window_ranges = [
#         {'window_size': 30, 'timestamp_start': '2024-08-28 00:00:00', 'timestamp_end': '2024-08-28 06:00:00'},
#         {'window_size': 60, 'timestamp_start': '2024-08-28 06:00:00', 'timestamp_end': '2024-08-28 18:00:00'},
#         {'window_size': 45, 'timestamp_start': '2024-08-28 18:00:00', 'timestamp_end': '2024-08-29 00:00:00'}
#     ]
#     """
#     file_path = './data/test4.csv'  # Replace with your file path
#     data = pd.read_csv(file_path)

#     # Convert the 'Category' column to datetime format
#     data['Timestamp'] = pd.to_datetime(data['Category'], unit='ms')
#     data.drop(columns=['Category'], inplace=True)
#     data.set_index('Timestamp', inplace=True)

#     # Resample the data to 1-minute intervals to ensure uniformity
#     data_resampled = data.resample('1T').mean().ffill()

#     # Initialize a 'WindowSize' column to store the rolling window sizes
#     data_resampled['WindowSize'] = 60  # Default window size (can be overwritten by the ranges)

#     # Assign the appropriate window size for each range
#     for window_range in window_ranges:
#         window_size = window_range['window_size']
#         timestamp_start = pd.to_datetime(window_range['timestamp_start'])
#         timestamp_end = pd.to_datetime(window_range['timestamp_end'])
        
#         # Set window size for the specified time range
#         mask = (data_resampled.index >= timestamp_start) & (data_resampled.index <= timestamp_end)
#         data_resampled.loc[mask, 'WindowSize'] = window_size

#     # Apply rolling calculations with dynamic window size
#     data_resampled['RollingMin'] = data_resampled.apply(
#         lambda row: data_resampled['Flow rate'].rolling(window=int(row['WindowSize']), min_periods=1, center=True).min().loc[row.name],
#         axis=1
#     )

#     # Ensure RollingMin is not greater than Flow rate
#     data_resampled['RollingMin'] = data_resampled.apply(lambda row: min(row['Flow rate'], row['RollingMin']), axis=1)

#     # Forward fill to ensure no NaN values
#     data_resampled['RollingMin'] = data_resampled['RollingMin'].ffill()

#     # Calculate totals
#     total_water_wasted = data_resampled['RollingMin'].sum()
#     total_water_consumed = data_resampled['Flow rate'].sum()

#     # Efficiency percentage
#     efficiency_percentage = 100 - ((total_water_consumed - total_water_wasted) / total_water_consumed * 100)

#     return int(total_water_wasted), data_resampled, int(total_water_consumed), int(efficiency_percentage)

# # Example usage
# window_ranges = [
#     {'window_size': 30, 'timestamp_start': '2024-08-28 00:00:00', 'timestamp_end': '2024-08-28 06:00:00'},
#     {'window_size': 60, 'timestamp_start': '2024-08-28 06:00:00', 'timestamp_end': '2024-08-28 18:00:00'},
#     {'window_size': 45, 'timestamp_start': '2024-08-28 18:00:00', 'timestamp_end': '2024-08-29 00:00:00'}
# ]

# total_water_wasted, data_resampled, total_water_consumed, efficiency_percentage = analyze_data(window_ranges)
# print(f"Total Water Wasted: {total_water_wasted} liters")
# print(f"Total Water Consumed: {total_water_consumed} liters")
# print(f"Efficiency Percentage: {efficiency_percentage}%")
