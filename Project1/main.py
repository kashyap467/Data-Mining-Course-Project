import pandas as pd


"""
GET DATE OF FIRST MODE SWTICH FROM MANUEL TO AUTO FROM INSULIN DATA CSV FILE
"""
# Read insulin data
insulin_df = pd.read_csv('InsulinData.csv')
insulin_df = insulin_df[::-1]       # Sort the order of time into increasing order

# Combine date and time columns into new column 'DateTime' in pandas DateTime format
insulin_df['DateTime'] = pd.to_datetime(insulin_df['Date'] + " " + insulin_df['Time'])
# print("Insulin Data: ", insulin_data['DateTime'][:5])

# Get timestamp where mode switches from Manual to Auto
times_modeSwitch = insulin_df[insulin_df['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']['DateTime']
time_modeSwitch_1st = times_modeSwitch.iloc[0]


"""
GET GLUCOSE DATA FROM CGM DATA CSV FILE
"""
# Read glucose data
cgm_df = pd.read_csv('CGMData.csv')
cgm_df = cgm_df[::-1]       # Sort the order of time into increasing order

# Combine date and time columns into new column 'DateTime' in pandas DateTime format
cgm_df['DateTime'] = pd.to_datetime(cgm_df['Date'] + " " + cgm_df['Time'])
# print("Glucose Data: ", cgm_data['DateTime'][:5])

# Get only required columns
cgm_df = cgm_df[['Date', 'Time', 'DateTime', 'Sensor Glucose (mg/dL)']]


"""
PREPROCESSING CGM DATA
"""
# Group samples by date and get their count
cgm_sampleCount_dates = cgm_df.groupby(['Date']).count()
# print("CGM samples count per day: ", cgm_sampleCount_dates)


def cleanAnomalies(df, dates, time_modeSwitch) :
    '''Method to clean anomalies in the given data'''
    date_modeSwitch = time_modeSwitch.strftime('%m-%d-%Y')

    # Get indices of days with readings > 288 or < 263 as anomalies
    anomaly_idxs = list(dates[(dates['Time'] > 288) | (dates['Time'] < 263)].index)
    anomaly_idxs += [date_modeSwitch]           # Also include the sample where mode switches

    df = df.drop(df[df['Date'].isin(anomaly_idxs)].index, axis=0)       # Drop anomalies from the glucose data
    return df


# Clean anomalies in the glucose data
cgm_df = cleanAnomalies(cgm_df, cgm_sampleCount_dates, time_modeSwitch_1st)

# Fill in the missing values
cgm_df['Sensor Glucose (mg/dL)'] = cgm_df['Sensor Glucose (mg/dL)'].interpolate(method='linear', limit_direction = 'both')

# Get index where mode switches in CGM data using the timestamp found from Insulin data
cgm_idx_modeSwitch = len(cgm_df[ cgm_df['DateTime'] <= time_modeSwitch_1st ])


"""
SPLIT CGM DATA INTO MULTIPLE DATA FRAMES W.R.T TWO MODES AND THREE INTERVAL TIMES
"""
def extract_data_by_mode(df, idx_modeSwitch, mode):
    '''Method to split given data according to three interval times and given mode'''
    df_mode = df[:idx_modeSwitch]  if  mode == 0  else  df[idx_modeSwitch:]     # Manuel - 0, Auto - 1
    df_mode_whole = df_mode.set_index('DateTime')       # Whole day
    df_mode_day = df_mode_whole.between_time('06:00:00','23:59:59')     # Day time
    df_mode_night = df_mode_whole.between_time('00:00:00','05:59:59')   # Over night
    
    return df_mode_whole, df_mode_day, df_mode_night

# Get Manual mode (0) data belonging to each of the three interval times
df_manual_whole, df_manual_day, df_manual_night = extract_data_by_mode(cgm_df, cgm_idx_modeSwitch, 0)
# Get Auto mode (1) data belonging to each of the three interval times
df_auto_whole, df_auto_day, df_auto_night = extract_data_by_mode(cgm_df, cgm_idx_modeSwitch, 1)


"""
EXTRACT METRICS FROM GIVEN DATA ACCORDING TO GIVEN GLUCOSE MEASUREMENTS
"""
cgm_measures = (((180,0), (250,0)), ((70, 180), (70, 150)), ((0, 70), (0, 54)))

def extract_metrics(df, days):
    '''Extract metrics from given data and return respective means using given days'''
    metrics = []

    for type in cgm_measures : 
        for level in type : 
            if level[0] == 0 : 
                percent = df[ df['Sensor Glucose (mg/dL)'] < level[1] ].groupby(['Date']).count()
            elif level[1] == 0 : 
                percent = df[ df['Sensor Glucose (mg/dL)'] > level[0] ].groupby(['Date']).count()
            else : 
                percent = df[ df['Sensor Glucose (mg/dL)'].between(level[0], level[1]) ].groupby(['Date']).count()
            percent = (percent['Time'].sum() * 100) / (288 * days)
            metrics.append(percent)

    return metrics


"""
GET THE 18 METRIC MEASUREMENTS FROM MANUAL MODE DATA
"""
# Get number of days in Manual mode
count_days_manual = (df_manual_whole.groupby(['Date']).count()).count()[0]
# print("Num Days in Manual: ", count_days_manual)

# Get the 6 metrics for whole day
metrics_manual_whole = extract_metrics(df_manual_whole, count_days_manual)
# Get the 6 metrics for daytime
metrics_manual_day = extract_metrics(df_manual_day, count_days_manual)
# Get the 6 metrics for overnight
metrics_manual_night = extract_metrics(df_manual_night, count_days_manual)


"""
GET THE 18 METRIC MEASUREMENTS FROM AUTO MODE DATA
"""
# Get number of days in Auto mode
count_days_auto = (df_auto_whole.groupby(['Date']).count()).count()[0]
# print("Num Days in Auto: ", count_days_auto)

# Get the 6 metrics for whole day
metrics_auto_whole = extract_metrics(df_auto_whole, count_days_auto)
# Get the 6 metrics for daytime
metrics_auto_day = extract_metrics(df_auto_day, count_days_auto)
# Get the 6 metrics for overnight
metrics_auto_night = extract_metrics(df_auto_night, count_days_auto)


"""
WRITING RESULTS TO FILE IN THE FORMAT:  [OVERNIGHT, DAYTIME, WHOLEDAY]  FOR  EACH MODE
"""
metrics_manual = metrics_manual_night + metrics_manual_day + metrics_manual_whole + [1.1]
print(metrics_manual)
metrics_auto = metrics_auto_night + metrics_auto_day + metrics_auto_whole + [1.1]
# col_headers = [ 'Percentage time in hyperglycemia (CGM > 180 mg/dL)', 
#                 'Percentage of time in hyperglycemia critical (CGM > 250 mg/dL)', 
#                 'Percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
#                 'Percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)',
#                 'Percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
#                 'Percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)' ]
# df_result = pd.DataFrame([metrics_manual, metrics_auto], columns = col_headers * 3, index = ['Manual Mode', 'Auto Mode'])
df_result = pd.DataFrame([metrics_manual, metrics_auto])
df_result = df_result.fillna(0)
# df_result.to_csv('Results.csv', header=False, index=False)