#region IMPORTS
import pandas as pd
import pickle
import pickle_compat
from train import extract_features
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#endregion

pickle_compat.patch()

# Load the model from pickle file, and test data from test.csv file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    test_df = pd.read_csv('test.csv', header=None)

# Initialize PCA
pca = PCA(n_components = 7)

# Extract features for test data, normalize and pass through PCA
test_features = extract_features(test_df)
test_features = (test_features - test_features.mean()) / (test_features.max() - test_features.min())
test_stdScalar = StandardScaler().fit_transform(test_features)
test_pca = pca.fit_transform(test_stdScalar)

# Run model on features to get predictions and write them to Results.csv file
test_predictions = pd.DataFrame(model.predict(test_pca))
test_predictions.to_csv("Results.csv", header=None, index=False)