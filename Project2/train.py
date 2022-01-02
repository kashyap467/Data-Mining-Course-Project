#region IMPORTS
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
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

def get_cgms_of_valid_timestamps(cgm_df, timestamps, isMeal) : 
    data_extracted = []
    num_cols = 0    # Number of columns to consider
    length = len(timestamps) if isMeal else len(timestamps)-1
    for i in range(length) : 
        timestamp = timestamps[i]
        if isMeal == True : 
            # Find the meal stretch for Meal case
            meal_start = pd.to_datetime(timestamp - pd.Timedelta(minutes = 30))
            meal_end = pd.to_datetime(timestamp + pd.Timedelta(minutes = 120))
            num_cols = 30
        else : 
            # Find the meal stretch for No Meal case
            meal_start = pd.to_datetime(timestamp + pd.Timedelta(minutes = 120))
            meal_end = timestamps[i+1]
            num_cols = 24
        # Condition to check if the timestamp is within the strech of meal start and end times
        is_timestamp_outOf_mealStrech = (cgm_df['date_time'] >= meal_start) & (cgm_df['date_time'] <= meal_end)
        # Filter and collect CGM values that satisfy this condition
        cgm_filter = cgm_df.loc[is_timestamp_outOf_mealStrech]['Sensor Glucose (mg/dL)'].values.tolist()
        data_extracted.append(cgm_filter[:num_cols])

    return pd.DataFrame(data_extracted)

def extract_meal_data(insulin_df, glucose_df) : 
    # Clean the data from 0's, NaN's and missing values
    ins_df, cgm_df = clean_data(insulin_df, glucose_df)

    # Get timestamps that can satisfy Meal time case
    meal_timestamps = []
    for i in range(len(ins_df['date_time']) - 1) : 
        val = ins_df['date_time'][i]
        val2 = ins_df['date_time'][i+1]
        if (val2 - val).seconds / 60.0 >= 120 : 
            meal_timestamps.append(val)

    # Filter non conflicting timestamps
    meal_data_extracted = get_cgms_of_valid_timestamps(cgm_df, meal_timestamps, isMeal=True)
    return meal_data_extracted.dropna()

def extract_nomeal_data(insulin_df, glucose_df) : 
    # Clean the data from 0's, NaN's and missing values
    ins_df, cgm_df = clean_data(insulin_df, glucose_df)

    # Get timestamps that can satisfy No Meal time case
    no_meal_timestamps = []
    for i in range(len(ins_df['date_time'])-1) : 
        val = ins_df['date_time'][i]
        val2 = ins_df['date_time'][i+1]
        if (val2 - val).seconds / 60.0 >= 240 : 
            no_meal_timestamps.append(val)

    # Filter non conflicting timestamps
    no_meal_data_extracted = get_cgms_of_valid_timestamps(cgm_df, no_meal_timestamps, isMeal=False)
    return no_meal_data_extracted.dropna()

#endregion

#region FEATURE EXTRACTION METHODS

def get_max_zero_crosses(row_values, k_max) : 
    zero_crosses = []

    # Get slopes for all consecutive values on feature row
    slopes = np.diff(row_values)
    # Keep sign of 1st slope
    init_slope_sign = 1  if  slopes[0] > 0   else  0

    # Check where ever slope sign changes (means it crosses zero line)
    for x in range(1, len(slopes)) : 
        cur_slope_sign = 1 if slopes[x] > 0 else 0
        if init_slope_sign != cur_slope_sign:
            # Append that slope to zero crosses list
            zero_crosses.append([slopes[x] - slopes[x-1], x])
        init_slope_sign = cur_slope_sign

    # Sort the zero crosses list in descending order, then return asked kth max value
    return sorted(zero_crosses, reverse=True)[k_max]  if  k_max < len(zero_crosses)  else [0, 0]

def get_fast_fourier_transform(row_values, k_max) : 
    # Get fast fourier transform values of each value in given row
    fast_fourier = fft(row_values)
    # Sort the amplitudes in decreasing order
    amplitude = sorted([np.abs(amp) for amp in fast_fourier])
    # return the asked kth max element
    return amplitude[-k_max]

def extract_features(cgm_data) : 

    features = pd.DataFrame()
    num_rows = cgm_data.shape[0]

    for i in range(0, num_rows) : 
        row_values = cgm_data.iloc[i, :].tolist()

        #For each row, get 1st, 2nd, 3rd highest zero crosses
        zero_cross_1st_max = get_max_zero_crosses(row_values, 0)
        zero_cross_2nd_max = get_max_zero_crosses(row_values, 1)
        zero_cross_3rd_max = get_max_zero_crosses(row_values, 2)

        # For each row, get 2nd, 3rd, 4th highest FFT amplitudes
        fft_2nd_max = get_fast_fourier_transform(row_values, 2)
        fft_3rd_max = get_fast_fourier_transform(row_values, 3)
        fft_4th_max = get_fast_fourier_transform(row_values, 4)

        # Append these values as features of this row into features list
        features_per_sample_dict = {
         'ZeroCross Max 1': zero_cross_1st_max[0], 'ZeroCross Max 1 index': zero_cross_1st_max[1],
         'ZeroCross Max 2': zero_cross_2nd_max[0], 'ZeroCross Max 2 index': zero_cross_2nd_max[1],
         'ZeroCross Max 3': zero_cross_3rd_max[0], 'ZeroCross Max 3 index': zero_cross_3rd_max[1],
         'FFTAmpl Max 2': fft_2nd_max, 'FFTAmpl Max 3': fft_3rd_max, 'FFTAmpl Max 4': fft_4th_max
        }
        features = features.append(features_per_sample_dict, ignore_index=True)

    return features

#endregion

#region METHOD TO TRAIN MODEL AND GET METRICS
def train(samples, labels) : 

    model = 'Meal_NoMeal_Classifier'
    acc_scores = []

    # For K-Fold validation
    kfold_k = 5
    kfold = KFold(kfold_k, True, 1)

    for tr, ts in kfold.split(samples, labels) : 
        # Split samples and labels according to K-Fold validation
        tr_X, ts_X = samples.iloc[tr], samples.iloc[ts]
        tr_Y, ts_Y = labels.iloc[tr], labels.iloc[ts]

        # Train model on current train data set
        model = SVC(kernel='rbf', gamma='scale', degree=3)
        model.fit(tr_X, tr_Y)

        # Generate predictions on current test data set and find accuracy
        pred_Y = model.predict(ts_X)
        acc_scores.append(accuracy_score(ts_Y, pred_Y))
    
    # print('Accuracy: ', (np.sum(acc_scores) / kfold_k) * 100)
    
    return model
#endregion


""" MAIN CODE """

if __name__ == '__main__' : 

    #region READ & EXTRACT Meal & NoMeal DATA FROM FILES

    # Read insulin data
    insulin_data_cols = ['Date','Time','BWZ Carb Input (grams)']
    insulin_df1 = pd.read_csv('InsulinData.csv', usecols = insulin_data_cols)
    insulin_df2 = pd.read_csv('Insulin_patient2.csv', usecols = insulin_data_cols)
    insulin_df = pd.concat([insulin_df1, insulin_df2])

    # Read glucose data
    cgm_data_cols = ['Date','Time','Sensor Glucose (mg/dL)']
    cgm_df1 = pd.read_csv('CGMData.csv', usecols = cgm_data_cols)
    cgm_df2 = pd.read_csv('CGM_patient2.csv', usecols = cgm_data_cols)
    cgm_df = pd.concat([cgm_df1, cgm_df2])

    # Set new column 'date_time' with date+time
    insulin_df['date_time'] = pd.to_datetime(insulin_df['Date'] + " " + insulin_df['Time'])
    cgm_df['date_time'] = pd.to_datetime(cgm_df['Date'] + " " + cgm_df['Time'])

    # Extract Meal & NoMeal data
    meal_df = extract_meal_data(insulin_df, cgm_df)
    noMeal_df = extract_nomeal_data(insulin_df, cgm_df)

    #endregion

    #region EXTRACT FEATURES FROM DATA

    #Initialize PCA
    pca = PCA(n_components = 7)

    # Extract features for Meal Time case, normalize and pass through PCA
    meal_features = extract_features(meal_df)
    meal_features = (meal_features - meal_features.mean()) / (meal_features.max() - meal_features.min())
    meal_stdScalar = StandardScaler().fit_transform(meal_features)
    meal_pca = pd.DataFrame(pca.fit_transform(meal_stdScalar))
    meal_pca['class'] = 1

    # Extract features for No Meal Time case, normalize and pass through PCA
    noMeal_features = extract_features(noMeal_df)
    noMeal_features = (noMeal_features - noMeal_features.mean()) / (noMeal_features.max() - noMeal_features.min())
    noMeal_stdScalar = StandardScaler().fit_transform(noMeal_features)
    noMeal_pca = pd.DataFrame(pca.fit_transform(noMeal_stdScalar))
    noMeal_pca['class'] = 0

    #endregion

    #region TRAIN MODEL & DUMP AS PICKLE FILE

    # Join all PCA feature samples and split them into samples, labels
    pca_data = pd.concat([meal_pca, noMeal_pca])
    samples = pca_data.iloc[:, :-1]
    labels = pca_data.iloc[:, -1]

    # Train the model on the extracted features
    model = train(samples, labels)

    # Dump the trained model into a pickle file
    with open('model.pkl', 'wb') as file :
        pickle.dump(model, file)

    #endregion
