# from numpy.core.fromnumeric import argmax
import pandas as pd
import numpy as np
from math import ceil, log
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from matplotlib import pyplot as plt



#region PREPROCESSING AND HELPER METHODS

def get_cluster_centers(clusters) : 
    return [cluster.mean(axis=0) for cluster in clusters]


def clean_data(ins_df, cgm_df) : 
    # Sort into increasing order of time
    ins_df = ins_df.sort_values(by='date_time')
    # Drop all NaN values, then reset index and drop index column
    ins_df = ins_df[ins_df['BWZ Carb Input (grams)'] != 0.0].dropna()
    ins_df = ins_df.reset_index().drop(columns='index')
    # Interpolate the missing values
    cgm_df['Sensor Glucose (mg/dL)'] = cgm_df['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
    return ins_df, cgm_df


def get_cgms_ground_truths(cgm_df, timestamps, carbs) : 
    meal_df, ground_truths = [], []
    min_carb = min(carbs)

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
            truth = int(ceil((carbs[i] - min_carb) / 20)) if carbs[i] != min_carb  else  1
            ground_truths.append(truth)

    return pd.DataFrame(meal_df), ground_truths


def extract_data_labels_numBins(insulin_df, glucose_df) : 
    # Clean the data from 0's, NaN's and missing values
    ins_df, cgm_df = clean_data(insulin_df, glucose_df)

    # Get timestamps & carbs amount of Meal times
    meal_times, meal_carbs = [], []
    for i in range(len(ins_df['date_time']) - 1) : 
        val = ins_df['date_time'][i]
        val2 = ins_df['date_time'][i+1]
        if (val2 - val).seconds / 60.0 >= 120 : 
            meal_times.append(val)
            meal_carbs.append(ins_df['BWZ Carb Input (grams)'][i])
    
    # Find the number of bins, meal data and their ground truths and return
    num_bins = ceil((max(meal_carbs) - min(meal_carbs)) / 20)
    meal_df, ground_truths = get_cgms_ground_truths(cgm_df, meal_times, meal_carbs)

    return meal_df, ground_truths, int(num_bins)

#endregion


#region FEATURE EXTRACTION

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

    features = pd.DataFrame(dtype=float)
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
         'Max-Min' : np.amax(row_values) - np.amin(row_values),
         'ArgMax-ArgMin' : np.argmax(row_values) - np.argmin(row_values),
         'ZeroCross Max 1': zero_cross_1st_max[0], 'ZeroCross Max 1 index': zero_cross_1st_max[1],
         'ZeroCross Max 2': zero_cross_2nd_max[0], 'ZeroCross Max 2 index': zero_cross_2nd_max[1],
         'ZeroCross Max 3': zero_cross_3rd_max[0], 'ZeroCross Max 3 index': zero_cross_3rd_max[1],
         'FFTAmpl Max 2': fft_2nd_max, 'FFTAmpl Max 3': fft_3rd_max, 'FFTAmpl Max 4': fft_4th_max
        }
        features = features.append(features_per_sample_dict, ignore_index=True)

    return features

#endregion


#region METHODS TO CALCULATE METRICS

def compute_sse(cluster_data, center) : 
    # get errors for each data point
    errors = [np.linalg.norm(point - center) for point in cluster_data]
    # calculate sum of squared errors and return
    return np.sum(np.square(errors))

def compute_entropy(cluster_data) :
    points_sum = np.sum(cluster_data)
    # get frequencies as probabilities
    probs = [point/points_sum for point in cluster_data]
    # calculate entropy and return
    return sum([-1*(prob) * log(prob,2) if prob != 0 else 0 for prob in probs])

def compute_purity(cluster_data) :
    points_sum = np.sum(cluster_data)
    # get frequencies as probabilities
    probs = [point/points_sum for point in cluster_data]
    # calculate purity and return
    return max(probs)


def get_metrics_from_clusters(clusters, num_bins) : 

    # Form the Cluster-Bin Matrix
    cluster_bin_matrix = np.empty([num_bins, num_bins])
    cluster_bin_matrix.fill(0)
    for c, cluster in enumerate(clusters):
        for d in range(cluster.shape[0]):
            cluster_bin_matrix[c][int(cluster.iloc[d,-1]) - 1] += 1
    matrix_sum = np.sum(cluster_bin_matrix)

    # Calculate total SSE
    cluster_centers = get_cluster_centers(clusters)
    cluster_SSEs = [compute_sse(clusters[i].iloc[:, :-1].to_numpy(), cluster_centers[i][:-1].to_numpy()) for i in range(len(clusters))]
    total_SSE = np.sum(cluster_SSEs)

    # Calculate total Entropy
    cluster_entropies = [compute_entropy(cluster) for cluster in cluster_bin_matrix]
    entropies = [sum(cluster_bin_matrix[i]) * entropy for i,entropy in enumerate(cluster_entropies)]
    total_entropy = sum(entropies) / matrix_sum

    # Calculate total Purity
    cluster_purities = [compute_purity(cluster) for cluster in cluster_bin_matrix]
    purities = [sum(cluster_bin_matrix[i]) * purity for i,purity in enumerate(cluster_purities)]
    total_purity = sum(purities) / matrix_sum

    return total_SSE, total_entropy, total_purity

#endregion


#region GENERATE CLUSTERS

def create_clusters(data_points, labels) : 
    num_clusters = len(set(labels))
    num_clusters += -1  if -1 in labels else 0
    clusters = [pd.DataFrame() for _ in range(num_clusters)]
    for i in range(len(labels)) : 
        if labels[i] != -1 : 
            clusters[labels[i]] = clusters[labels[i]].append(data_points.iloc[i])
            clusters[labels[i]].reset_index().drop(columns='index')
    return clusters


def create_n_clusters(dbscan, meal_df, num_clusters_desired) : 
    clusters = create_clusters(meal_df, dbscan.labels_)

    # Split the obtained DBScan clusters into desired number of clusters
    while(len(clusters) < num_clusters_desired) : 
        cluster_centers = get_cluster_centers(clusters)
        cluster_SSEs = [compute_sse(clusters[i].iloc[:, :-1].to_numpy(), cluster_centers[i][:-1].to_numpy()) for i in range(len(clusters))]
        cluster_split = clusters[np.argmax(cluster_SSEs)]
        kMeans = KMeans(n_clusters=2, random_state=0)
        kMeans = kMeans.fit(cluster_split.iloc[:, :-1])
        clusters_updated = create_clusters(cluster_split, kMeans.labels_)
        del cluster_split
        clusters.extend(clusters_updated)

    return clusters


# def knn(dataset, n):
#     neighbors = NearestNeighbors(n_neighbors=n)
#     neighbors_fit = neighbors.fit(dataset)
#     distances, indices = neighbors_fit.kneighbors(dataset)
#     distances = np.sort(distances, axis=0)
#     distances = distances[:,1]
#     plt.plot(distances)
#     plt.show()

#endregion


if __name__ == '__main__':

    num_samples = 10
    num_comps = 5

    #region READ & EXTRACT DATA FROM FILES

    # Read insulin data
    insulin_data_cols = ['Date','Time','BWZ Carb Input (grams)']
    insulin_df = pd.read_csv('InsulinData.csv', usecols = insulin_data_cols)
    
    # Read glucose data
    cgm_data_cols = ['Date','Time','Sensor Glucose (mg/dL)']
    cgm_df = pd.read_csv('CGMData.csv', usecols = cgm_data_cols)

    # Set new column 'date_time' with date+time
    insulin_df['date_time'] = pd.to_datetime(insulin_df['Date'] + " " + insulin_df['Time'])
    cgm_df['date_time'] = pd.to_datetime(cgm_df['Date'] + " " + cgm_df['Time'])

    meal_df, ground_truths, num_bins = extract_data_labels_numBins(insulin_df , cgm_df)

    #endregion

    #region EXTRACT FEATURES AND CALCULATED METRICS

    # Extract features for Meal Time case, normalize and pass through PCA
    meal_features = extract_features(meal_df)
    meal_stdScalar = StandardScaler().fit_transform(meal_features)
    meal_pca = pd.DataFrame(PCA(n_components = num_comps).fit_transform(meal_stdScalar))
    meal_pca['ground_truth'] = ground_truths

    # Finding metrics for KMeans
    kMeans = KMeans(n_clusters=num_bins, random_state=0).fit(meal_pca.iloc[:, :-1])
    clusters_kmeans = create_clusters(meal_pca, kMeans.labels_)
    kmeans_sse, kmeans_entropy, kmeans_purity = get_metrics_from_clusters(clusters_kmeans, num_bins)

    # knn(meal_pca.iloc[:,:-1], num_samples)

    # Finding metrics for DbScan
    dbScan = DBSCAN(eps=1.2, min_samples=num_samples).fit(meal_pca.iloc[:, :-1])
    clusters_dbScan = create_n_clusters(dbScan, meal_pca, num_bins)
    dbScan_sse, dbScan_entropy, dbScan_purity = get_metrics_from_clusters(clusters_dbScan, num_bins)

    #endregion

    # region WRITE CALCULATED METRICS TO csv FILE
    res_df = pd.DataFrame()
    res_df['kmeans_sse'], res_df['dbscan_sse'] = [kmeans_sse], [dbScan_sse]
    res_df['kmeans_entropy'], res_df['dbscan_entropy'] = [kmeans_entropy], [dbScan_entropy]
    res_df['kmeans_purity'], res_df['dbscan_purity'] = [kmeans_purity], [dbScan_purity]
    res_df.to_csv('Results.csv', header = False, index = False)
    #endregion