import pandas as pd
from supabase import create_client
import os
import logging

logger = logging.getLogger(__name__)

def analyze_data(window_size: int = 60, start_epoch: int = None, end_epoch: int = None, place_id: int = None):
    logger.info(f"Analyzing data with window_size={window_size}, start={start_epoch}, end={end_epoch}, place_id={place_id}")
    
    # Initialize Supabase client
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
    
    # Build the query on the new "measurements" table
    query = supabase.table("measurements").select("*")
    if place_id is not None:
        query = query.eq("place_id", place_id)
    if start_epoch is not None:
        query = query.gte("timestamp", start_epoch)
    if end_epoch is not None:
        query = query.lte("timestamp", end_epoch)
    
    # Execute query and log the response
    response = query.execute()
    logger.info(f"Query response data length: {len(response.data) if response.data else 0}")
    
    # Convert to DataFrame
    data = pd.DataFrame(response.data)
    
    # Add debugging information
    logger.info(f"DataFrame shape: {data.shape}")
    logger.info(f"Available columns in DataFrame: {data.columns.tolist()}")
    
    if data.empty:
        # Return empty DataFrames with expected columns
        empty_df = pd.DataFrame(columns=["flow_rate", "RollingMin"])
        return {
            'weekday_data': empty_df,
            'weekend_data': empty_df,
            'weekday_peak': {'day': None, 'consumption': 0},
            'weekend_peak': {'day': None, 'consumption': 0},
            'weekday_total': 0,
            'weekend_total': 0,
            'weekday_wasted': 0,
            'weekend_wasted': 0,
            'weekday_efficiency': 0,
            'weekend_efficiency': 0
        }
    
    # Convert timestamp to datetime and set as index
    # Assuming timestamp is in milliseconds
    data['Timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data.set_index('Timestamp')  # Set Timestamp as the index

    # Resample data to 1-minute intervals, forward fill missing values
    data_resampled = data.resample('1T').mean().ffill()
    logger.info(f"Columns after resampling: {data_resampled.columns.tolist()}")
    
    # Separate weekday and weekend data
    weekday_data = data_resampled[data_resampled.index.weekday < 5].copy()
    weekend_data = data_resampled[data_resampled.index.weekday >= 5].copy()
    
    # Add RollingMin calculations for both weekday and weekend data
    for df in [weekday_data, weekend_data]:
        # Compute rolling minimums across different windows (forward, backward, centered)
        df['RollingMin_b'] = df['flow_rate'].shift(-window_size).rolling(window=window_size, min_periods=1).min()
        df['RollingMin_f'] = df['flow_rate'].rolling(window=window_size, min_periods=1).min()
        df['RollingMin_c'] = df['flow_rate'].rolling(window=window_size, min_periods=1, center=True).min()
        df['RollingMin'] = df[['RollingMin_f', 'RollingMin_b', 'RollingMin_c']].max(axis=1)
        # Ensure RollingMin never exceeds the actual flow_rate at that minute
        df['RollingMin'] = df.apply(lambda row: min(row['flow_rate'], row['RollingMin']) if pd.notnull(row['RollingMin']) else row['flow_rate'], axis=1)
        df['RollingMin'] = df['RollingMin'].ffill()
    
    def safe_int(value):
        try:
            if isinstance(value, (float, int)):
                if pd.isnull(value):
                    return 0
                return int(value)
            return 0
        except:
            return 0

    # Calculate metrics for weekdays
    total_water_wasted_weekdays = safe_int(weekday_data['RollingMin'].sum())
    total_water_consumed_weekdays = safe_int(weekday_data['flow_rate'].sum())
    
    if total_water_consumed_weekdays > 0:
        efficiency_percentage_weekdays = 100 - safe_int((total_water_wasted_weekdays / total_water_consumed_weekdays) * 100)
    else:
        efficiency_percentage_weekdays = 0

    # Calculate metrics for weekends
    total_water_wasted_weekends = safe_int(weekend_data['RollingMin'].sum())
    total_water_consumed_weekends = safe_int(weekend_data['flow_rate'].sum())
    
    if total_water_consumed_weekends > 0:
        efficiency_percentage_weekends = 100 - safe_int((total_water_wasted_weekends / total_water_consumed_weekends) * 100)
    else:
        efficiency_percentage_weekends = 0
    
    # Calculate peak consumption for weekdays
    if not weekday_data.empty:
        weekday_peak_idx = weekday_data['flow_rate'].idxmax()
        weekday_peak_consumption = weekday_data['flow_rate'].max()
        weekday_peak_day = weekday_peak_idx.strftime('%Y-%m-%d %H:%M')
    else:
        weekday_peak_idx = None
        weekday_peak_consumption = 0
        weekday_peak_day = 'N/A'
    
    # Calculate peak consumption for weekends
    if not weekend_data.empty:
        weekend_peak_idx = weekend_data['flow_rate'].idxmax()
        weekend_peak_consumption = weekend_data['flow_rate'].max()
        weekend_peak_day = weekend_peak_idx.strftime('%Y-%m-%d %H:%M')
    else:
        weekend_peak_idx = None
        weekend_peak_consumption = 0
        weekend_peak_day = 'N/A'
    
    return {
        'weekday_data': weekday_data,
        'weekend_data': weekend_data,
        'weekday_peak': {
            'day': weekday_peak_day,
            'consumption': safe_int(weekday_peak_consumption)
        },
        'weekend_peak': {
            'day': weekend_peak_day,
            'consumption': safe_int(weekend_peak_consumption)
        },
        'weekday_total': total_water_consumed_weekdays,
        'weekend_total': total_water_consumed_weekends,
        'weekday_wasted': total_water_wasted_weekdays,
        'weekend_wasted': total_water_wasted_weekends,
        'weekday_efficiency': efficiency_percentage_weekdays,
        'weekend_efficiency': efficiency_percentage_weekends
    }
