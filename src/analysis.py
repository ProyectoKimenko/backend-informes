import pandas as pd
from supabase import create_client
import os
import logging

logger = logging.getLogger(__name__)

def analyze_data(window_size: int = 60, start_epoch: int = None, end_epoch: int = None, place_id: int = None):
    logger.info(f"Analyzing data with window_size={window_size}, start={start_epoch}, end={end_epoch}, place_id={place_id}")
    
    # Initialize Supabase client
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
    
    # Build the query on the "measurements" table
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

    # Resample data to 1-minute intervals, fill missing values with 0
    data_resampled = data.resample('min').mean().fillna(0)  # Using 'min' to avoid FutureWarning
    logger.info(f"Columns after resampling: {data_resampled.columns.tolist()}")
    
    # Separate weekday and weekend data
    weekday_data = data_resampled[data_resampled.index.weekday < 5].copy()
    weekend_data = data_resampled[data_resampled.index.weekday >= 5].copy()
    
    # Function to calculate RollingMin (leaks) based on runs of flow >0 longer than window_size, with gap tolerance
    def calculate_leaks(df, window_size):
        if df.empty:
            df['RollingMin'] = pd.Series(dtype=float)
            return df
        # Identify continuous runs of positive flow
        df['is_positive'] = df['flow_rate'] > 0
        df['group_id'] = (df['is_positive'] != df['is_positive'].shift()).cumsum()
        groups = df.groupby('group_id')
        
        # Collect positive groups
        positive_groups = []
        for g_id, group in groups:
            if group['is_positive'].iloc[0]:
                positive_groups.append(group)
        
        if not positive_groups:
            df['RollingMin'] = 0.0
            df = df.drop(columns=['is_positive', 'group_id'], errors='ignore')
            return df
        
        # Merge positive groups if gap <= tolerance
        tolerance = 0  # minutes for small gaps
        merged_groups = []
        current_merged = positive_groups[0]
        for next_group in positive_groups[1:]:
            # Calculate gap in minutes
            gap_delta = next_group.index.min() - current_merged.index.max()
            gap_min = int((gap_delta.total_seconds() / 60) - 1)  # -1 since consecutive have delta=1min, gap=0
            if gap_min <= tolerance:
                current_merged = pd.concat([current_merged, next_group])
            else:
                merged_groups.append(current_merged)
                current_merged = next_group
        merged_groups.append(current_merged)
        
        df['RollingMin'] = 0.0  # Initialize as 0 (no leak unless in a long run)
        
        # First pass: identify all runs that exceed window_size
        for merged_group in merged_groups:
            run_length = len(merged_group)
            if run_length > window_size:
                # For each long run, check for sub-periods with higher constant consumption
                # Use a sliding window approach
                for start_idx in range(len(merged_group) - window_size):
                    end_idx = start_idx + window_size + 1
                    window_data = merged_group.iloc[start_idx:end_idx]
                    
                    # Calculate the minimum flow in this window
                    window_min = window_data['flow_rate'].min()
                    
                    # Apply this minimum to the entire window
                    for idx in window_data.index:
                        # Keep the maximum value if already set (from a previous window)
                        current_val = df.loc[idx, 'RollingMin']
                        df.loc[idx, 'RollingMin'] = max(current_val, window_min)
                
                # Also check longer sub-periods within the run
                # Check windows of size 2*window_size, 3*window_size, etc.
                for multiplier in range(2, 4):  # Check up to 3x window_size
                    extended_window = window_size * multiplier
                    if extended_window < len(merged_group):
                        for start_idx in range(len(merged_group) - extended_window):
                            end_idx = start_idx + extended_window + 1
                            window_data = merged_group.iloc[start_idx:end_idx]
                            
                            # For longer windows, use a more robust metric (percentile 10)
                            window_leak = window_data['flow_rate'].quantile(0.001)
                            
                            # Apply to the entire window
                            for idx in window_data.index:
                                current_val = df.loc[idx, 'RollingMin']
                                df.loc[idx, 'RollingMin'] = max(current_val, window_leak)

        # Drop auxiliary columns
        df = df.drop(columns=['is_positive', 'group_id'], errors='ignore')
        return df
    
    # Apply to weekday and weekend data
    weekday_data = calculate_leaks(weekday_data, window_size)
    weekend_data = calculate_leaks(weekend_data, window_size)
    
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
