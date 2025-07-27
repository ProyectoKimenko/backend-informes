import pandas as pd
from supabase import create_client
import os
import logging
import numpy as np
from numba import njit
import sys
from datetime import timezone, datetime

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_local_timezone_offset():
    """Get the local timezone offset from UTC as a string (e.g., 'UTC-4', 'UTC+2')"""
    local_now = datetime.now()
    utc_now = datetime.utcnow()
    
    # Calculate offset in hours
    offset_seconds = (local_now - utc_now).total_seconds()
    offset_hours = int(offset_seconds / 3600)
    
    if offset_hours >= 0:
        return f"UTC+{offset_hours}"
    else:
        return f"UTC{offset_hours}"  # offset_hours already has the minus sign

# Numba-optimized functions for rolling calculations
@njit
def update_rolling_min(flow: np.ndarray, offset_size: int) -> np.ndarray:
    """
    Compute the rolling minimum with maximum propagation using Numba for speed.
    
    Args:
        flow (np.ndarray): Array of flow rate values.
        offset_size (int): Size of the rolling window (excluding the current point).
    
    Returns:
        np.ndarray: Array with rolling minimum values propagated forward.
    """
    n = len(flow)
    result = np.zeros(n)
    for start in range(n - offset_size):
        end = start + offset_size + 1
        win_min = np.min(flow[start:end])
        for j in range(start, end):
            if result[j] < win_min:
                result[j] = win_min
    return result

@njit
def update_rolling_quant(flow: np.ndarray, offset_size: int, q: float) -> np.ndarray:
    """
    Compute the rolling quantile with maximum propagation using Numba for speed.
    
    Args:
        flow (np.ndarray): Array of flow rate values.
        offset_size (int): Size of the rolling window (excluding the current point).
        q (float): Quantile value (e.g., 0.001 for the 0.1% quantile).
    
    Returns:
        np.ndarray: Array with rolling quantile values propagated forward.
    """
    n = len(flow)
    result = np.zeros(n)
    for start in range(n - offset_size):
        end = start + offset_size + 1
        win = np.sort(flow[start:end])
        pos = q * (len(win) - 1)
        floor = int(pos)
        frac = pos - floor
        if floor + 1 < len(win):
            win_q = win[floor] + frac * (win[floor + 1] - win[floor])
        else:
            win_q = win[floor]
        for j in range(start, end):
            if result[j] < win_q:
                result[j] = win_q
    return result

def analyze_data_from_df(data_df: pd.DataFrame, window_size: int = 60, start_epoch: int = None, end_epoch: int = None) -> dict:
    """
    Analyze water consumption data from a pre-loaded DataFrame without database queries.
    All timestamps are handled in UTC timezone.
    
    Args:
        data_df (pd.DataFrame): Input DataFrame with 'timestamp' and 'flow_rate' columns.
        window_size (int, optional): Size of the rolling window in minutes. Defaults to 60.
        start_epoch (int, optional): Start timestamp in milliseconds (UTC). Defaults to None.
        end_epoch (int, optional): End timestamp in milliseconds (UTC). Defaults to None.
    
    Returns:
        dict: dictionary containing weekday and weekend data, peaks, totals, wasted water, and efficiencies.
    """
    # Handle empty DataFrame
    if data_df.empty:
        logger.info("Input DataFrame is empty")
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

    # Debug logging for input data
    logger.info(f"Input data shape: {data_df.shape}")
    if not data_df.empty:
        logger.info(f"Timestamp range in raw data: {data_df['timestamp'].min()} to {data_df['timestamp'].max()}")
        logger.info(f"First few raw timestamps: {data_df['timestamp'].head().tolist()}")

    # Filter data by time range if specified
    filtered_data = data_df.copy()
    if start_epoch is not None:
        before_filter = len(filtered_data)
        filtered_data = filtered_data[filtered_data['timestamp'] >= start_epoch]
        logger.info(f"After start_epoch filter ({start_epoch}): {before_filter} -> {len(filtered_data)} records")
    
    if end_epoch is not None:
        before_filter = len(filtered_data)
        filtered_data = filtered_data[filtered_data['timestamp'] <= end_epoch]
        logger.info(f"After end_epoch filter ({end_epoch}): {before_filter} -> {len(filtered_data)} records")

    # Handle empty filtered DataFrame
    if filtered_data.empty:
        logger.info("No data after filtering by time range")
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

    # Convert timestamp to datetime with UTC timezone explicitly
    filtered_data['Timestamp'] = pd.to_datetime(filtered_data['timestamp'], unit='ms', utc=True)
    filtered_data = filtered_data.set_index('Timestamp')
    
    # Debug logging for timestamp conversion
    logger.info(f"After timestamp conversion, datetime range: {filtered_data.index.min()} to {filtered_data.index.max()}")
    logger.info(f"First few converted timestamps: {filtered_data.index[:5].tolist()}")
    
    # Resample to minutes (maintaining UTC timezone)
    data_resampled = filtered_data.resample('min').mean().fillna(0)
    
    logger.info(f"After resampling to minutes: {len(data_resampled)} records")
    logger.info(f"Resampled time range: {data_resampled.index.min()} to {data_resampled.index.max()}")

    # Split into weekday and weekend data
    weekday_data = data_resampled[data_resampled.index.weekday < 5].copy()
    weekend_data = data_resampled[data_resampled.index.weekday >= 5].copy()
    
    logger.info(f"Weekday data: {len(weekday_data)} records")
    logger.info(f"Weekend data: {len(weekend_data)} records")
    
    if not weekday_data.empty:
        logger.info(f"Weekday range: {weekday_data.index.min()} to {weekday_data.index.max()}")
    if not weekend_data.empty:
        logger.info(f"Weekend range: {weekend_data.index.min()} to {weekend_data.index.max()}")

    # Calculate leaks for each dataset
    def calculate_leaks(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        if df.empty:
            df['RollingMin'] = pd.Series(dtype=float)
            return df

        df = df.copy()
        df['is_positive'] = df['flow_rate'] > 0
        df['group_id'] = (df['is_positive'] != df['is_positive'].shift()).cumsum()

        # Identify positive flow groups
        groups = df.groupby('group_id')
        positive_groups = [group for g_id, group in groups if group['is_positive'].iloc[0]]

        if not positive_groups:
            df['RollingMin'] = 0.0
            df = df.drop(columns=['is_positive', 'group_id'], errors='ignore')
            return df

        # Merge groups with zero tolerance
        tolerance = 0
        merged_groups = []
        current_merged = positive_groups[0]
        for next_group in positive_groups[1:]:
            gap_delta = next_group.index.min() - current_merged.index.max()
            gap_min = int((gap_delta.total_seconds() / 60) - 1)
            if gap_min <= tolerance:
                current_merged = pd.concat([current_merged, next_group])
            else:
                merged_groups.append(current_merged)
                current_merged = next_group
        merged_groups.append(current_merged)

        # Calculate RollingMin for each merged group
        df['RollingMin'] = 0.0
        for merged_group in merged_groups:
            run_length = len(merged_group)
            if run_length > window_size:
                flow = merged_group['flow_rate'].values
                idx = merged_group.index
                rolling_mins = update_rolling_min(flow, window_size)
                df.loc[idx, 'RollingMin'] = np.maximum(df.loc[idx, 'RollingMin'], rolling_mins)

                for multiplier in range(2, 4):
                    extended_window = window_size * multiplier
                    if extended_window < run_length:
                        rolling_leaks = update_rolling_quant(flow, extended_window, 0.001)
                        df.loc[idx, 'RollingMin'] = np.maximum(df.loc[idx, 'RollingMin'], rolling_leaks)

        df = df.drop(columns=['is_positive', 'group_id'], errors='ignore')
        return df

    weekday_data = calculate_leaks(weekday_data, window_size)
    weekend_data = calculate_leaks(weekend_data, window_size)

    # Safely convert values to integers
    def safe_int(value):
        try:
            if isinstance(value, (float, int)):
                if pd.isnull(value):
                    return 0
                return int(value)
            return 0
        except Exception:
            return 0

    total_water_wasted_weekdays = safe_int(weekday_data['RollingMin'].sum())
    total_water_consumed_weekdays = safe_int(weekday_data['flow_rate'].sum())

    if total_water_consumed_weekdays > 0:
        efficiency_percentage_weekdays = 100 - safe_int((total_water_wasted_weekdays / total_water_consumed_weekdays) * 100)
    else:
        efficiency_percentage_weekdays = 0

    total_water_wasted_weekends = safe_int(weekend_data['RollingMin'].sum())
    total_water_consumed_weekends = safe_int(weekend_data['flow_rate'].sum())

    if total_water_consumed_weekends > 0:
        efficiency_percentage_weekends = 100 - safe_int((total_water_wasted_weekends / total_water_consumed_weekends) * 100)
    else:
        efficiency_percentage_weekends = 0

    # Get local timezone offset for display
    tz_offset = get_local_timezone_offset()

    # Determine peak consumption with improved formatting including local timezone
    if not weekday_data.empty:
        weekday_peak_idx = weekday_data['flow_rate'].idxmax()
        weekday_peak_consumption = weekday_data['flow_rate'].max()
        # Format with local timezone information
        weekday_peak_day = weekday_peak_idx.strftime(f'%Y-%m-%d %H:%M {tz_offset}')
    else:
        weekday_peak_consumption = 0
        weekday_peak_day = 'N/A'

    if not weekend_data.empty:
        weekend_peak_idx = weekend_data['flow_rate'].idxmax()
        weekend_peak_consumption = weekend_data['flow_rate'].max()
        # Format with local timezone information
        weekend_peak_day = weekend_peak_idx.strftime(f'%Y-%m-%d %H:%M {tz_offset}')
    else:
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

def analyze_data(window_size: int = 60, start_epoch: int = None, end_epoch: int = None, place_id: int = None) -> dict:
    """
    Analyze water consumption data by querying Supabase and processing with analyze_data_from_df.
    
    Args:
        window_size (int, optional): Size of the rolling window in minutes. Defaults to 60.
        start_epoch (int, optional): Start timestamp in milliseconds. Defaults to None.
        end_epoch (int, optional): End timestamp in milliseconds. Defaults to None.
        place_id (int, optional): ID of the place to filter data. Defaults to None.
    
    Returns:
        dict: dictionary containing analysis results.
    """
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
    
    query = supabase.table("measurements").select("*")
    if place_id is not None:
        query = query.eq("place_id", place_id)
    if start_epoch is not None:
        query = query.gte("timestamp", start_epoch)
        logger.info(f"Database query with start_epoch >= {start_epoch}")
    if end_epoch is not None:
        query = query.lte("timestamp", end_epoch)
        logger.info(f"Database query with end_epoch <= {end_epoch}")
    
    response = query.execute()
    data = pd.DataFrame(response.data)
    
    # Debug logging for database results
    logger.info(f"Database returned {len(data)} records")
    if not data.empty:
        logger.info(f"Database timestamp range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    return analyze_data_from_df(data, window_size, start_epoch, end_epoch)