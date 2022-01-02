#region IMPORTS

from math import ceil
import pandas as pd
import numpy as np

#endregion

""" HELPER METHODS """

#region DATA PREPROCESSING METHODS

def clean_data(ins_df, cgm_df) : 
    # Sort into increasing order of time
    ins_df = ins_df.sort_values(by='date_time')
    # Drop all NaN values, then reset index and drop index column
    ins_df = ins_df[ins_df['BWZ Carb Input (grams)'] != 0.0].dropna()
    ins_df = ins_df.reset_index().drop(columns='index')
    # Interpolate the missing values
    cgm_df['Sensor Glucose (mg/dL)'] = cgm_df['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
    return ins_df, cgm_df

def get_cgms_boluses(cgm_df, timestamps, ins_bolus) : 
    meal_df, ins_bolus_meal = [], []

    for i in range(len(timestamps)) : 
        timestamp = timestamps[i]
        # Find the meal stretch for Meal case
        meal_start = pd.to_datetime(timestamp - pd.Timedelta(minutes = 30))
        meal_end = pd.to_datetime(timestamp + pd.Timedelta(minutes = 120))
        # Condition to check if the timestamp is within the strech of meal start and end times
        is_timestamp_outOf_mealStrech = (cgm_df['date_time'] >= meal_start) & (cgm_df['date_time'] <= meal_end)

        # Filter and collect CGM values that satisfy this condition along with ground truths
        cgm_filter = cgm_df.loc[is_timestamp_outOf_mealStrech]['Sensor Glucose (mg/dL)'].values.tolist()
        if len(cgm_filter) >= 30 : 
            meal_df.append(cgm_filter[:30])
            ins_bolus_meal.append(ins_bolus[i])

    return meal_df, ins_bolus_meal

def extract_mealData_insulinBolus(insulin_df, glucose_df) : 
    # Clean the data from 0's, NaN's and missing values
    ins_df, cgm_df = clean_data(insulin_df, glucose_df)

    # Get timestamps of Meal times & corresponding insulin bolus values
    meal_times, ins_bolus = [], []
    for i in range(len(ins_df['date_time']) - 1) : 
        val = ins_df['date_time'][i]
        val2 = ins_df['date_time'][i+1]
        if (val2 - val).seconds / 60.0 >= 120 : 
            meal_times.append(val)
            ins_bolus.append(round(ins_df['BWZ Estimate (U)'][i]))
    
    # Find the meal data and their insulin boluses, and return
    meal_df, ins_bolus_df = get_cgms_boluses(cgm_df, meal_times, ins_bolus)
    return meal_df, ins_bolus_df

def assign_bin(cur_cgm, min_cgm) : 
    # Find which bin the current CGM reading falls into. Bin size is 20 mg/dL
    bin_num = ceil((cur_cgm - min_cgm) / 20)  if cur_cgm != min_cgm  else  1
    return bin_num

def form_itemsets(meal_df, bolus_df) : 
    meal_df = np.asarray(meal_df)
    bin_max, bin_meal = [], []
    # Get the minimum CGM reading in the meal data matrix
    min_cgm = meal_df.min()

    for sample in meal_df : 
        # Get CGM reading at current time stamp and max CGM reading near that meal time strech
        max_cgm, meal_cgm = sample.max(), sample[-6]
        bin_max.append(assign_bin(max_cgm, min_cgm))
        bin_meal.append(assign_bin(meal_cgm, min_cgm))
    
    # Form itemsets with bin numbers of max CGM and CGM at meal time with respective insulin bolus values
    itemsets = pd.DataFrame({'b_max': bin_max, 'b_meal': bin_meal, 'i_bolus': bolus_df})
    return itemsets

#endregion


""" MAIN CODE """

if __name__ == '__main__' : 

    #region READ & EXTRACT MEAL DATA FROM FILES

    # Read insulin data
    insulin_data_cols = ['Date','Time','BWZ Carb Input (grams)', 'BWZ Estimate (U)']
    insulin_df = pd.read_csv('InsulinData.csv', usecols = insulin_data_cols)
    # insulin_df1 = pd.read_csv('InsulinData.csv', usecols = insulin_data_cols)
    # insulin_df2 = pd.read_csv('Insulin_patient2.csv', usecols = insulin_data_cols)
    # insulin_df = pd.concat([insulin_df1, insulin_df2])

    # Read glucose data
    cgm_data_cols = ['Date','Time','Sensor Glucose (mg/dL)']
    cgm_df = pd.read_csv('CGMData.csv', usecols = cgm_data_cols)
    # cgm_df1 = pd.read_csv('CGMData.csv', usecols = cgm_data_cols)
    # cgm_df2 = pd.read_csv('CGM_patient2.csv', usecols = cgm_data_cols)
    # cgm_df = pd.concat([cgm_df1, cgm_df2])

    # Set new column 'date_time' with date+time
    insulin_df['date_time'] = pd.to_datetime(insulin_df['Date'] + " " + insulin_df['Time'])
    cgm_df['date_time'] = pd.to_datetime(cgm_df['Date'] + " " + cgm_df['Time'])

    # Extract Meal data and form Itemsets
    meal_df, ins_bolus_df = extract_mealData_insulinBolus(insulin_df, cgm_df)
    itemsets_df = form_itemsets(meal_df, ins_bolus_df)
    for col in itemsets_df.columns : 
        itemsets_df[col] = pd.to_numeric(itemsets_df[col], downcast='integer')

    #endregion

    #region FIND most frequent itemsets, largest confidence rules, anomalous rules

    # Find the most frequent itemsets (x,y,z)
    # Grouping all combinations and getting their count (frequency of (x,y,z))
    itemset_cols = ['b_max','b_meal','i_bolus']
    sets = itemsets_df.groupby(itemset_cols).size().reset_index(name='freqXYZ')
    max_freq_val = sets['freqXYZ'].max()     # Get the maximum frequency value

    # Filter the itemsets that have the maximum frequency
    max_freq_sets = sets.loc[sets['freqXYZ'] == max_freq_val]
    # Converting the column values into tuples of form (x,y,z) and writing to csv file
    max_freq_sets = max_freq_sets[itemset_cols].apply(lambda item : (item[0], item[1], item[2]), axis=1)


    # Calculating confidence for rules
    # Confidence for rule (x,y) -> z  = (frequency of (x,y,z) / frequency of (x,y))
    #   frequency of (x,y,z) already calculated above

    #   Computing frequnecies of all combinations of (x,y)
    rule_cols = ['b_max','b_meal']
    rules = itemsets_df.groupby(rule_cols).size().reset_index(name='freqXY')
    # Joining itemsets and rules into one df so that both frequencies are available at one place
    rules = pd.merge(sets, rules, on=rule_cols)
    rules['confidence'] = rules['freqXYZ'] / rules['freqXY']
    max_conf_val = rules['confidence'].max()    # Get the maximum confidence value

    # Find the largest confidence rules ((x,y) -> z)
    max_conf_rules = rules.loc[rules['confidence'] == max_conf_val]     # Filter rules with max confidence
    # Converting the column values into rules of form ({x,y} -> z) and writing to csv file
    max_conf_rules = max_conf_rules[itemset_cols].apply(lambda item : '{{{0},{1}}} -> {2}'.format(item[0], item[1], item[2]), axis=1)

    # Find anomalous rules (rules with confidence < 15%)
    anom_rules = rules.loc[rules['confidence'] < 0.15]      # Filtering rules with confidence < 15%
    # Converting the column values into rules of form ({x,y} -> z) and writing to csv file
    anom_rules = anom_rules[itemset_cols].apply(lambda item : '{{{0},{1}}} -> {2}'.format(item[0], item[1], item[2]), axis=1)

    #endregion

    #region WRITE RESULTS TO CSV FILES
    max_freq_sets.to_csv('most_frequent_itemsets.csv', header=False, index=False)
    max_conf_rules.to_csv('largest_confidence_rules.csv', header=False, index=False)
    anom_rules.to_csv('anomalous_rules.csv', header=False, index=False)
    #endregion