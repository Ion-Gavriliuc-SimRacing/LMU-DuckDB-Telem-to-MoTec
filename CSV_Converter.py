import duckdb
import pandas as pd
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog

def get_frequency(df_channels_list, channel_name_in_list):
    freq = df_channels_list[df_channels_list['channelName'] == channel_name_in_list]['frequency']
    if not freq.empty:
        return freq.iloc[0]
    # Fallback for 'Throttle Pos' if not explicitly listed but 'Brake Pos' is a good proxy
    if channel_name_in_list == 'Throttle Pos':
        return get_frequency(df_channels_list, 'Brake Pos')
    raise ValueError(f"Frequency not found for channel: {channel_name_in_list}")

def resample_and_interpolate(
    df_raw, df_channels_list, channel_name, value_col_name, master_time, freq=None, is_wheel_speed=False, is_gear=False
):
    df = df_raw.copy()
    current_freq = freq
    if not freq and not is_gear:
        current_freq = get_frequency(df_channels_list, channel_name)

    if is_gear:
        # Questa logica non viene più usata, ma è mantenuta nella funzione.
        df = df.rename(columns={'ts': 'time', 'value': value_col_name})
        # Handle duplicate timestamps by keeping the last value for each time
        df = df.drop_duplicates(subset=['time'], keep='last')
        df = df[['time', value_col_name]].set_index('time')
    elif is_wheel_speed:
        df['time'] = np.arange(len(df)) / current_freq
        
        # MODIFICA PER LA CONVERSIONE M/S -> KM/H (M/S * 3.6)
        df[value_col_name] = df[['value1', 'value2', 'value3', 'value4']].mean(axis=1) * 3.6
        
        df = df[['time', value_col_name]].set_index('time')
    else:
        df['time'] = np.arange(len(df)) / current_freq
        df = df.rename(columns={'value': value_col_name})
        df = df[['time', value_col_name]].set_index('time')

    df_resampled = df.reindex(master_time.values)

    if is_gear:
        df_resampled[value_col_name] = df_resampled[value_col_name].ffill().bfill()
    else:
        df_resampled[value_col_name] = df_resampled[value_col_name].interpolate(method='linear', limit_direction='both')

    return df_resampled.reset_index()

def process_telemetry_and_save_csv(duckdb_file_path):
    # 1. Ottiene la directory dello script in esecuzione
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    # 2. Definisce la cartella di output 'CSV'
    output_dir = os.path.join(script_dir, "CSV")
    
    # 3. Assicura che la cartella 'CSV' esista
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(duckdb_file_path))[0]
    
    # 4. Costruisce il percorso finale nella cartella 'CSV'
    output_csv_path = os.path.join(output_dir, f"{base_name}_telemetry_enhanced.csv")

    try:
        con = duckdb.connect(database=duckdb_file_path, read_only=True)
        df_channels_list = con.execute("SELECT * FROM channelsList;").fetchdf()

        # Load raw data
        df_throttle_raw = con.execute("SELECT * FROM \"Throttle Pos\"").fetchdf()
        df_brake_raw = con.execute("SELECT * FROM \"Brake Pos\"").fetchdf()
        df_speed_raw = con.execute("SELECT * FROM \"Wheel Speed\"").fetchdf()
        df_boost_raw = con.execute("SELECT * FROM \"Turbo Boost Pressure\"").fetchdf()
        df_coolant_temp_raw = con.execute("SELECT * FROM \"Engine Water Temp\"").fetchdf()
        # Rimosso: df_gear_raw = con.execute("SELECT * FROM \"Gear\"").fetchdf()
        df_rpm_raw = con.execute("SELECT * FROM \"Engine RPM\"").fetchdf()

        # Get frequencies for time generation
        freq_throttle = get_frequency(df_channels_list, 'Throttle Pos')
        freq_brake = get_frequency(df_channels_list, 'Brake Pos')
        freq_speed = get_frequency(df_channels_list, 'Wheel Speed')
        freq_boost = get_frequency(df_channels_list, 'Turbo Boost Pressure')
        freq_coolant_temp = get_frequency(df_channels_list, 'Engine Water Temp')
        freq_rpm = get_frequency(df_channels_list, 'Engine RPM')

        # Calculate end time for each raw dataframe
        max_time_throttle = (len(df_throttle_raw) - 1) / freq_throttle if len(df_throttle_raw) > 0 else 0
        max_time_brake = (len(df_brake_raw) - 1) / freq_brake if len(df_brake_raw) > 0 else 0
        max_time_speed = (len(df_speed_raw) - 1) / freq_speed if len(df_speed_raw) > 0 else 0
        max_time_boost = (len(df_boost_raw) - 1) / freq_boost if len(df_boost_raw) > 0 else 0
        max_time_coolant_temp = (len(df_coolant_temp_raw) - 1) / freq_coolant_temp if len(df_coolant_temp_raw) > 0 else 0
        max_time_rpm = (len(df_rpm_raw) - 1) / freq_rpm if len(df_rpm_raw) > 0 else 0
        # Rimosso: max_time_gear
        
        session_end_time = np.max([
            max_time_throttle, max_time_brake, max_time_speed, max_time_boost,
            max_time_coolant_temp, max_time_rpm
        ])

        # Create master time axis
        master_time_step = 0.01
        master_time = pd.Series(np.arange(0, session_end_time + master_time_step, master_time_step), name='time')

        # Resample and interpolate each channel
        df_throttle = resample_and_interpolate(df_throttle_raw, df_channels_list, 'Throttle Pos', 'Throttle_Pos', master_time, freq_throttle)
        df_brake = resample_and_interpolate(df_brake_raw, df_channels_list, 'Brake Pos', 'Brake_Pos', master_time, freq_brake)
        df_speed = resample_and_interpolate(df_speed_raw, df_channels_list, 'Wheel Speed', 'Speed_KPH', master_time, freq_speed, is_wheel_speed=True)
        df_boost = resample_and_interpolate(df_boost_raw, df_channels_list, 'Turbo Boost Pressure', 'Boost_Pressure', master_time, freq_boost)
        df_coolant_temp = resample_and_interpolate(df_coolant_temp_raw, df_channels_list, 'Engine Water Temp', 'Coolant_Temp', master_time, freq_coolant_temp)
        # Rimosso: df_gear = resample_and_interpolate(...)
        df_rpm = resample_and_interpolate(df_rpm_raw, df_channels_list, 'Engine RPM', 'Engine_RPM', master_time, freq_rpm)

        # Merge all interpolated dataframes
        df_telemetry_enhanced = pd.DataFrame({'time': master_time})
        df_telemetry_enhanced = pd.merge(df_telemetry_enhanced, df_throttle, on='time', how='left')
        df_telemetry_enhanced = pd.merge(df_telemetry_enhanced, df_brake, on='time', how='left')
        df_telemetry_enhanced = pd.merge(df_telemetry_enhanced, df_speed, on='time', how='left')
        df_telemetry_enhanced = pd.merge(df_telemetry_enhanced, df_boost, on='time', how='left')
        df_telemetry_enhanced = pd.merge(df_telemetry_enhanced, df_coolant_temp, on='time', how='left')
        # Rimosso: merge df_gear
        df_telemetry_enhanced = pd.merge(df_telemetry_enhanced, df_rpm, on='time', how='left')

        con.close()
        df_telemetry_enhanced.to_csv(output_csv_path, index=False)
        print(f"Successfully saved enhanced telemetry data to '{output_csv_path}'")

    except FileNotFoundError:
        print(f"Error: DuckDB file not found at '{duckdb_file_path}'")
        sys.exit(1)
    except duckdb.CatalogException as e:
        print(f"Error querying DuckDB tables: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error in data processing: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def select_and_process_file():
    duckdb_file_path = None
    try:
        root = tk.Tk()
        root.withdraw()
        print("Please select a DuckDB file using the GUI dialog...")
        duckdb_file_path = filedialog.askopenfilename(
            title="Select DuckDB File",
            filetypes=[("DuckDB files", "*.duckdb"), ("All files", "*.* ")]
        )
    except tk.TclError:
        print("\nTkinter GUI is not available (no display environment detected).")
        print("Please manually enter the path to your DuckDB file:")
        duckdb_file_path = input("DuckDB file path: ").strip()
        if not duckdb_file_path:
            print("No file path entered. Operation cancelled.")
            sys.exit(0)

    if duckdb_file_path:
        print(f"Selected/Entered file: {duckdb_file_path}")
        print("Processing telemetry data...")
        process_telemetry_and_save_csv(duckdb_file_path)
    else:
        print("No file selected. Operation cancelled.")

if __name__ == "__main__":
    select_and_process_file()