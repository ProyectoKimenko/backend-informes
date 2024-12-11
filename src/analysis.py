import pandas as pd
from supabase import create_client
import os
import logging

logger = logging.getLogger(__name__)

def analyze_data(window_size: int = 60, start_epoch: int = None, end_epoch: int = None):
    logger.info(f"Analyzing data with window_size={window_size}, start={start_epoch}, end={end_epoch}")
    
    # Initialize Supabase client
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
    
    # Build the query
    query = supabase.table("refugioAleman").select("*")
    if start_epoch is not None:
        query = query.gte("Category", start_epoch)
    if end_epoch is not None:
        query = query.lte("Category", end_epoch)
    
    # Execute query and log the response
    response = query.execute()
    logger.info(f"Query response data length: {len(response.data) if response.data else 0}")
    
    # Convert to DataFrame
    data = pd.DataFrame(response.data)
    
    # Add debugging information
    logger.info(f"DataFrame shape: {data.shape}")
    logger.info(f"Available columns in DataFrame: {data.columns.tolist()}")
    
    if data.empty:
        logger.warning(f"No data found for period: start={start_epoch}, end={end_epoch}")
        return {
            'weekday_data': pd.DataFrame(),
            'weekend_data': pd.DataFrame(),
            'weekday_wasted': 0,
            'weekend_wasted': 0,
            'weekday_total': 0,
            'weekend_total': 0,
            'weekday_efficiency': 0,
            'weekend_efficiency': 0,
            'weekday_peak': {'day': None, 'consumption': 0},
            'weekend_peak': {'day': None, 'consumption': 0}
        }
    
    # Convert Category to Timestamp and set as index
    data['Timestamp'] = pd.to_datetime(data['Category'], unit='ms')
    data = data.set_index('Timestamp')  # Set Timestamp as the index
    
    # Now we can resample the data
    data_resampled = data.resample('1T').mean().ffill()
    
    # Separate weekday and weekend data
    weekday_data = data_resampled[data_resampled.index.weekday < 5].copy()
    weekend_data = data_resampled[data_resampled.index.weekday >= 5].copy()
    
    # Calculate rolling minimums for weekdays and weekends
    for df in [weekday_data, weekend_data]:
        df['RollingMin_b'] = df['Flow rate'].shift(-window_size).rolling(window=window_size, min_periods=1).min()
        df['RollingMin_f'] = df['Flow rate'].rolling(window=window_size, min_periods=1).min()
        df['RollingMin_c'] = df['Flow rate'].rolling(window=window_size, min_periods=1, center=True).min()
        df['RollingMin'] = df[['RollingMin_f', 'RollingMin_b', 'RollingMin_c']].max(axis=1)
        df['RollingMin'] = df.apply(lambda row: row['Flow rate'] if row['RollingMin'] > row['Flow rate'] else row['RollingMin'], axis=1)
        df['RollingMin'] = df['RollingMin'].ffill()
    
    # Safe calculation function
    def safe_int(value):
        try:
            if pd.isna(value):
                return 0
            return int(value)
        except:
            return 0

    # Calculate metrics for weekdays with safety checks
    total_water_wasted_weekdays = safe_int(weekday_data['RollingMin'].sum())
    total_water_consumed_weekdays = safe_int(weekday_data['Flow rate'].sum())
    
    if total_water_consumed_weekdays > 0:
        # Calculate efficiency as percentage NOT wasted (100% - wasted%)
        efficiency_percentage_weekdays = 100 - safe_int((total_water_wasted_weekdays / total_water_consumed_weekdays) * 100)
    else:
        efficiency_percentage_weekdays = 0

    # Calculate metrics for weekends with safety checks
    total_water_wasted_weekends = safe_int(weekend_data['RollingMin'].sum())
    total_water_consumed_weekends = safe_int(weekend_data['Flow rate'].sum())
    
    if total_water_consumed_weekends > 0:
        # Calculate efficiency as percentage NOT wasted (100% - wasted%)
        efficiency_percentage_weekends = 100 - safe_int((total_water_wasted_weekends / total_water_consumed_weekends) * 100)
    else:
        efficiency_percentage_weekends = 0
    
    # Calculate peak consumption for weekdays
    weekday_peak_idx = weekday_data['Flow rate'].idxmax() if not weekday_data.empty else None
    weekday_peak_consumption = weekday_data['Flow rate'].max() if not weekday_data.empty else 0
    weekday_peak_day = weekday_peak_idx.strftime('%Y-%m-%d %H:%M') if weekday_peak_idx is not None else 'N/A'
    
    # Calculate peak consumption for weekends
    weekend_peak_idx = weekend_data['Flow rate'].idxmax() if not weekend_data.empty else None
    weekend_peak_consumption = weekend_data['Flow rate'].max() if not weekend_data.empty else 0
    weekend_peak_day = weekend_peak_idx.strftime('%Y-%m-%d %H:%M') if weekend_peak_idx is not None else 'N/A'
    
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
